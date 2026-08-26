# src/hf_sync.py

"""
Hugging Face Hub Synchronization Module (Token-Based).

This file provides functions to manage the persistence of experiment results
and logs by synchronizing the local workspace with a Hugging Face Hub dataset
repository.

This version is modified to pass the Hugging Face token directly from a
configuration dictionary for all API interactions, removing the need for a
separate login step or reliance on cached credentials.

Functions:
- initialize_workspace: Downloads the remote repo to the local machine on startup.
- sync_workspace_to_hub: Uploads local output directories to the remote repo.
- periodic_sync_check: A helper to trigger synchronization during long loops.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple

from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    hf_hub_download,
    snapshot_download,
)
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

from src.distributed_execution import (
    DistributedManifestMismatch,
    validate_manifest_compatibility,
)


class DistributedSyncError(RuntimeError):
    """Raised when a required distributed checkpoint cannot be made durable."""


class DistributedManifestMismatchError(DistributedSyncError):
    """Raised when a run ID already has a different immutable manifest."""


_DISTRIBUTED_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_DISTRIBUTED_ALLOWED_DIRS = frozenset({"results"})
_DISTRIBUTED_ALLOWED_ROOT_FILES = frozenset({"status.json"})
_DISTRIBUTED_ALLOWED_SUFFIXES = frozenset({".json", ".jsonl", ".log", ".csv", ".pkl"})
_DISTRIBUTED_SECRET_NAME_MARKERS = (
    "api_key",
    "apikey",
    "credential",
    "secret",
    "token",
)
_DISTRIBUTED_TEMP_SUFFIXES = (".bak", ".lock", ".part", ".swp", ".tmp", "~")


def _startup_progress_enabled(config: dict) -> bool:
    """Return whether startup diagnostics should be printed to the notebook."""
    return bool(config.get("HF_STARTUP_DIAGNOSTICS_ENABLED", True))


def _print_startup_progress(config: dict, message: str) -> None:
    """Emit an immediate, flush-safe marker for long Hugging Face operations."""
    if _startup_progress_enabled(config):
        print(f"[HF startup {time.strftime('%H:%M:%S')}] {message}", flush=True)


class _StartupOperationReporter:
    """Print start/completion and optional heartbeats around one blocking call."""

    def __init__(self, config: dict, operation: str):
        self.config = config
        self.operation = operation
        self.started_at = 0.0
        self.stop_event = threading.Event()
        self.heartbeat_thread: Optional[threading.Thread] = None

    def __enter__(self):
        self.started_at = time.monotonic()
        _print_startup_progress(self.config, f"START: {self.operation}")
        if not _startup_progress_enabled(self.config):
            return self

        interval = self.config.get("HF_STARTUP_DIAGNOSTIC_HEARTBEAT_SECONDS", 30.0)
        try:
            interval = float(interval)
        except (TypeError, ValueError):
            interval = 30.0

        if interval > 0:
            self.heartbeat_thread = threading.Thread(
                target=self._print_heartbeats,
                args=(interval,),
                name="hf-startup-progress",
                daemon=True,
            )
            self.heartbeat_thread.start()
        return self

    def _print_heartbeats(self, interval: float) -> None:
        while not self.stop_event.wait(interval):
            if self.stop_event.is_set():
                return
            elapsed = time.monotonic() - self.started_at
            _print_startup_progress(
                self.config,
                f"WAITING ({elapsed:.0f}s): {self.operation} is still running.",
            )

    def __exit__(self, exc_type, exc_value, _traceback) -> bool:
        self.stop_event.set()
        if self.heartbeat_thread is not None:
            self.heartbeat_thread.join(timeout=1.0)
        elapsed = time.monotonic() - self.started_at
        if exc_type is None:
            _print_startup_progress(
                self.config,
                f"DONE ({elapsed:.1f}s): {self.operation}",
            )
        else:
            _print_startup_progress(
                self.config,
                f"FAILED after {elapsed:.1f}s: {self.operation} ({exc_type.__name__})",
            )
        return False


def _distributed_execution_enabled(config: dict) -> bool:
    return bool(config.get("DISTRIBUTED_EXECUTION_ENABLED", False))


def _distributed_finalizer_mode(config: dict) -> bool:
    return bool(config.get("DISTRIBUTED_FINALIZER_MODE", False))


def _validate_distributed_identity(config: dict) -> Tuple[str, int, int]:
    run_id = str(config.get("DISTRIBUTED_RUN_ID") or "").strip()
    if (
        not run_id
        or not _DISTRIBUTED_RUN_ID_PATTERN.fullmatch(run_id)
        or run_id in {".", ".."}
    ):
        raise ValueError(
            "DISTRIBUTED_RUN_ID must contain only letters, numbers, dots, "
            "underscores, or hyphens, and cannot be '.' or '..'."
        )

    raw_worker_id = config.get("DISTRIBUTED_WORKER_ID")
    raw_worker_count = config.get("DISTRIBUTED_WORKER_COUNT")
    if isinstance(raw_worker_id, bool) or isinstance(raw_worker_count, bool):
        raise ValueError("DISTRIBUTED_WORKER_ID and DISTRIBUTED_WORKER_COUNT must be integers.")
    try:
        worker_id = int(raw_worker_id)
        worker_count = int(raw_worker_count)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "DISTRIBUTED_WORKER_ID and DISTRIBUTED_WORKER_COUNT must be integers."
        ) from exc
    if worker_count <= 0:
        raise ValueError("DISTRIBUTED_WORKER_COUNT must be a positive integer.")
    if worker_id < 0 or worker_id >= worker_count:
        raise ValueError(
            "DISTRIBUTED_WORKER_ID must satisfy "
            "0 <= DISTRIBUTED_WORKER_ID < DISTRIBUTED_WORKER_COUNT."
        )
    return run_id, worker_id, worker_count


def get_distributed_worker_remote_prefix(config: dict) -> str:
    """Return the worker-owned, POSIX-normalized path inside the Hub repo."""
    run_id, worker_id, _ = _validate_distributed_identity(config)
    return str(PurePosixPath("distributed_runs", run_id, "workers", f"worker-{worker_id:03d}"))


def _get_distributed_manifest_remote_path(config: dict) -> str:
    run_id, _, _ = _validate_distributed_identity(config)
    return str(PurePosixPath("distributed_runs", run_id, "manifest.json"))


def _get_distributed_revision(config: dict) -> str:
    revision = config.get("DISTRIBUTED_HF_REVISION", "main")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("DISTRIBUTED_HF_REVISION must be a non-empty branch name.")
    return revision.strip()


def _get_distributed_worker_root(config: dict, *, create: bool) -> Path:
    configured_root = config.get("_DISTRIBUTED_WORKER_ROOT")
    if not isinstance(configured_root, (str, os.PathLike)) or not str(configured_root).strip():
        raise ValueError("_DISTRIBUTED_WORKER_ROOT must be configured for distributed execution.")

    worker_root = Path(configured_root).expanduser().resolve()
    if worker_root == Path(worker_root.anchor):
        raise ValueError("_DISTRIBUTED_WORKER_ROOT cannot be a filesystem root.")
    if worker_root.exists() and not worker_root.is_dir():
        raise ValueError("_DISTRIBUTED_WORKER_ROOT must refer to a directory.")
    if create:
        worker_root.mkdir(parents=True, exist_ok=True)
    return worker_root


def _require_distributed_hf_config(config: dict) -> Tuple[str, str, str]:
    if not config.get("PERSIST_RESULTS_ONLINE"):
        raise DistributedSyncError(
            "Distributed execution requires PERSIST_RESULTS_ONLINE=True so committed "
            "batches survive Kaggle session termination."
        )

    hf_token, hf_username, repo_name = _get_hf_config(config)
    placeholder_values = ("YOUR_HUGGING_FACE", "YOUR-HF-", "YOUR_HF_")
    values = (hf_token, hf_username, repo_name)
    if not all(values) or any(
        marker in str(value).upper()
        for value in values
        for marker in placeholder_values
    ):
        raise DistributedSyncError(
            "Distributed Hugging Face credentials/repository settings are missing "
            "or still contain placeholders."
        )
    return str(hf_token), str(hf_username), str(repo_name)


def _distributed_repo_id(config: dict) -> Tuple[str, str]:
    hf_token, hf_username, repo_name = _require_distributed_hf_config(config)
    return hf_token, f"{hf_username}/{repo_name}"


def _get_retry_settings(config: dict) -> Tuple[int, float]:
    max_retries = config.get("DISTRIBUTED_HF_SYNC_MAX_RETRIES", 5)
    retry_base = config.get("DISTRIBUTED_HF_SYNC_RETRY_BASE_SECONDS", 2.0)
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("DISTRIBUTED_HF_SYNC_MAX_RETRIES must be a non-negative integer.")
    if isinstance(retry_base, bool) or not isinstance(retry_base, (int, float)) or retry_base < 0:
        raise ValueError("DISTRIBUTED_HF_SYNC_RETRY_BASE_SECONDS must be non-negative.")
    return max_retries, float(retry_base)


def _http_status_code(error: BaseException) -> Optional[int]:
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None)


def _is_retryable_hf_error(error: BaseException) -> bool:
    status_code = _http_status_code(error)
    if status_code is not None:
        return status_code in {408, 409, 412, 425, 429} or status_code >= 500
    if isinstance(error, (ConnectionError, TimeoutError, HfHubHTTPError)):
        return True
    module_name = type(error).__module__
    return module_name.startswith(("httpcore", "httpx", "requests", "urllib3"))


def _wait_before_retry(logger: logging.Logger, operation: str, attempt: int, retry_base: float) -> None:
    delay = min(30.0, retry_base * (2 ** max(0, attempt - 1)))
    logger.warning("%s failed transiently; retrying in %.1f second(s).", operation, delay)
    if delay:
        time.sleep(delay)


def _canonical_json_hash(raw_bytes: bytes, *, source: str) -> str:
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DistributedSyncError(f"Invalid JSON manifest at {source}: {exc}") from exc
    return hashlib.sha256(canonical).hexdigest()


def _remote_manifest_at_revision(
    *,
    repo_id: str,
    remote_path: str,
    revision: str,
    token: str,
) -> Optional[Tuple[dict, str]]:
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="dataset",
            revision=revision,
            token=token,
        )
    except EntryNotFoundError:
        return None
    remote_bytes = Path(downloaded_path).read_bytes()
    source = f"{repo_id}/{remote_path}@{revision}"
    remote_hash = _canonical_json_hash(remote_bytes, source=source)
    try:
        payload = json.loads(remote_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DistributedSyncError(f"Invalid JSON manifest at {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DistributedSyncError(f"Distributed manifest at {source} must be a JSON object.")
    return payload, remote_hash


def ensure_distributed_manifest(config: dict, manifest_path) -> str:
    """Create or verify the immutable distributed-run manifest.

    The manifest is created only when it is absent at the current branch head.
    An existing semantically different JSON manifest is never overwritten.
    Concurrent creators are resolved optimistically with ``parent_commit``.

    Returns the canonical SHA-256 hash of the authoritative remote manifest.
    """
    if not _distributed_execution_enabled(config):
        raise ValueError("ensure_distributed_manifest requires DISTRIBUTED_EXECUTION_ENABLED=True.")

    run_id, _, _ = _validate_distributed_identity(config)
    hf_token, repo_id = _distributed_repo_id(config)
    revision = _get_distributed_revision(config)
    max_retries, retry_base = _get_retry_settings(config)
    remote_path = _get_distributed_manifest_remote_path(config)
    logger = logging.getLogger(__name__)

    configured_manifest_path = Path(manifest_path).expanduser()
    if configured_manifest_path.is_symlink():
        raise DistributedSyncError("The expected local distributed manifest cannot be a symlink.")
    local_manifest_path = configured_manifest_path.resolve()
    if not local_manifest_path.is_file():
        raise DistributedSyncError("The expected local distributed manifest must be a regular file.")
    local_bytes = local_manifest_path.read_bytes()
    expected_hash = _canonical_json_hash(local_bytes, source=str(local_manifest_path))
    try:
        expected_manifest = json.loads(local_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DistributedSyncError(
            f"Invalid JSON manifest at {local_manifest_path}: {exc}"
        ) from exc
    if not isinstance(expected_manifest, dict):
        raise DistributedSyncError("The expected local distributed manifest must be a JSON object.")

    api = HfApi(token=hf_token)
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision)
            parent_commit = getattr(repo_info, "sha", None)
            if not parent_commit:
                raise DistributedSyncError(f"Could not resolve the head commit for {repo_id}@{revision}.")

            remote_manifest_record = _remote_manifest_at_revision(
                repo_id=repo_id,
                remote_path=remote_path,
                revision=parent_commit,
                token=hf_token,
            )
            if remote_manifest_record is not None:
                remote_manifest, remote_hash = remote_manifest_record
                try:
                    model_changes = validate_manifest_compatibility(
                        remote_manifest,
                        expected_manifest,
                        allow_model_rotation=bool(
                            config.get("DISTRIBUTED_ALLOW_MODEL_ROTATION", False)
                        ),
                    )
                except DistributedManifestMismatch as exc:
                    raise DistributedManifestMismatchError(
                        "The remote distributed manifest already exists with different content "
                        f"(expected {expected_hash}, found {remote_hash}): {exc}"
                    ) from exc
                if model_changes:
                    logger.warning(
                        "Authorized distributed model rotation against immutable manifest %s: %s",
                        remote_path,
                        model_changes,
                    )
                logger.info("Validated immutable distributed manifest %s.", remote_path)
                return remote_hash

            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                parent_commit=parent_commit,
                operations=[CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=local_bytes)],
                commit_message=f"Create distributed run manifest: {run_id}",
            )
            logger.info("Created immutable distributed manifest %s.", remote_path)
            return expected_hash
        except DistributedManifestMismatchError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries or not _is_retryable_hf_error(exc):
                break
            _wait_before_retry(logger, "Distributed manifest commit", attempt + 1, retry_base)

    raise DistributedSyncError(
        f"Could not create or validate distributed manifest {remote_path}."
    ) from last_error


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _initialize_distributed_workspace(config: dict):
    logger = logging.getLogger(__name__)
    _validate_distributed_identity(config)
    hf_token, repo_id = _distributed_repo_id(config)
    revision = _get_distributed_revision(config)
    worker_root = _get_distributed_worker_root(config, create=True)
    worker_prefix = get_distributed_worker_remote_prefix(config)
    manifest_remote_path = _get_distributed_manifest_remote_path(config)

    allow_patterns = [manifest_remote_path, f"{worker_prefix}/**"]
    logger.info(
        "Initializing distributed worker from %s@%s using only its isolated prefix %s.",
        repo_id,
        revision,
        worker_prefix,
    )

    try:
        with tempfile.TemporaryDirectory(prefix="distributed-hf-download-") as staging_dir:
            staging_root = Path(staging_dir)
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(staging_root),
                token=hf_token,
                revision=revision,
                allow_patterns=allow_patterns,
            )

            staged_manifest = staging_root.joinpath(*PurePosixPath(manifest_remote_path).parts)
            if not staged_manifest.is_file():
                raise DistributedSyncError(
                    f"Distributed manifest {manifest_remote_path} does not exist at {repo_id}@{revision}."
                )
            _canonical_json_hash(staged_manifest.read_bytes(), source=str(staged_manifest))
            configured_manifest = config.get("_DISTRIBUTED_MANIFEST_PATH")
            run_root = Path(
                config.get("_DISTRIBUTED_RUN_ROOT") or worker_root.parent.parent
            ).expanduser().resolve()
            local_manifest = (
                Path(configured_manifest).expanduser().resolve()
                if configured_manifest
                else run_root / "manifest.json"
            )
            try:
                local_manifest.relative_to(run_root)
            except ValueError as exc:
                raise DistributedSyncError(
                    "_DISTRIBUTED_MANIFEST_PATH must remain inside _DISTRIBUTED_RUN_ROOT."
                ) from exc
            _atomic_copy(staged_manifest, local_manifest)

            staged_worker_root = staging_root.joinpath(*PurePosixPath(worker_prefix).parts)
            if staged_worker_root.is_dir():
                for source in sorted(staged_worker_root.rglob("*")):
                    if not source.is_file() or source.is_symlink():
                        continue
                    relative_path = source.relative_to(staged_worker_root)
                    destination = (worker_root / relative_path).resolve()
                    try:
                        destination.relative_to(worker_root)
                    except ValueError as exc:
                        raise DistributedSyncError(
                            f"Downloaded worker artifact escaped the worker root: {relative_path}"
                        ) from exc
                    _atomic_copy(source, destination)

        logger.info("Distributed worker workspace initialized at %s.", worker_root)
        print(f"Distributed worker checkpoint restored from Hugging Face ({repo_id})")
        return str(worker_root)
    except Exception as exc:
        if isinstance(exc, DistributedSyncError):
            raise
        raise DistributedSyncError(
            f"Failed to initialize distributed worker workspace from {repo_id}@{revision}."
        ) from exc


def download_distributed_run_for_merge(config: dict) -> str:
    """Download the immutable manifest and every worker shard for finalization.

    This is a one-time coordinator action after workers finish; normal workers
    never call it and therefore never read another worker's checkpoint.
    """
    if not _distributed_execution_enabled(config):
        raise ValueError("Distributed merge download requires distributed execution mode.")
    run_id, _, _ = _validate_distributed_identity(config)
    hf_token, repo_id = _distributed_repo_id(config)
    revision = _get_distributed_revision(config)
    base_output = Path(config["BASE_OUTPUT_DIR"]).expanduser().resolve()
    run_root = base_output / "distributed_runs" / run_id
    remote_root = PurePosixPath("distributed_runs", run_id)
    allow_patterns = [
        str(remote_root / "manifest.json"),
        str(remote_root / "workers" / "**"),
        str(remote_root / "merged" / "**"),
    ]
    try:
        with tempfile.TemporaryDirectory(prefix="distributed-merge-download-") as staging_dir:
            staging_root = Path(staging_dir)
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(staging_root),
                token=hf_token,
                revision=revision,
                allow_patterns=allow_patterns,
            )
            staged_run = staging_root.joinpath(*remote_root.parts)
            if not (staged_run / "manifest.json").is_file():
                raise DistributedSyncError(
                    f"Distributed manifest is missing for run {run_id} at {repo_id}@{revision}."
                )
            for source in sorted(staged_run.rglob("*")):
                if not source.is_file() or source.is_symlink():
                    continue
                relative = source.relative_to(staged_run)
                destination = (run_root / relative).resolve()
                try:
                    destination.relative_to(run_root)
                except ValueError as exc:
                    raise DistributedSyncError(
                        f"Merge download escaped the local run root: {relative}"
                    ) from exc
                _atomic_copy(source, destination)
        return str(run_root)
    except Exception as exc:
        if isinstance(exc, DistributedSyncError):
            raise
        raise DistributedSyncError(
            f"Failed to download distributed run {run_id} for merge."
        ) from exc


def sync_distributed_merged_results(config: dict, merged_root) -> object:
    """Atomically publish a validated merged directory under its run prefix."""
    if not _distributed_execution_enabled(config):
        raise ValueError("Merged-result sync requires distributed execution mode.")
    run_id, _, _ = _validate_distributed_identity(config)
    hf_token, repo_id = _distributed_repo_id(config)
    revision = _get_distributed_revision(config)
    max_retries, retry_base = _get_retry_settings(config)
    local_root = Path(merged_root).expanduser().resolve()
    configured_run_root = Path(
        config.get("_DISTRIBUTED_RUN_ROOT")
        or Path(config["BASE_OUTPUT_DIR"]) / "distributed_runs" / run_id
    ).expanduser().resolve()
    try:
        local_root.relative_to(configured_run_root)
    except ValueError as exc:
        raise DistributedSyncError("Merged results must remain inside the distributed run root.") from exc
    marker = local_root / "MERGE_COMPLETE.json"
    if not marker.is_file():
        raise DistributedSyncError("Refusing to upload merged results without MERGE_COMPLETE.json.")

    artifacts: List[Tuple[Path, Path]] = []
    layer2_only = bool(config.get("_DISTRIBUTED_LAYER2_SYNC_ONLY", False))
    for candidate in sorted(local_root.rglob("*")):
        if not candidate.is_file() or candidate.is_symlink():
            continue
        relative = candidate.resolve().relative_to(local_root)
        if layer2_only and not (
            relative.as_posix() == "MERGE_COMPLETE.json"
            or (len(relative.parts) >= 3 and relative.parts[0].casefold() == "results")
        ):
            continue
        if (
            any(part.startswith(".") for part in relative.parts)
            or any(marker_text in part.casefold() for part in relative.parts for marker_text in _DISTRIBUTED_SECRET_NAME_MARKERS)
            or candidate.name.casefold().endswith(_DISTRIBUTED_TEMP_SUFFIXES)
            or candidate.suffix.casefold() not in _DISTRIBUTED_ALLOWED_SUFFIXES
        ):
            continue
        artifacts.append((candidate.resolve(), relative))
    if not artifacts:
        raise DistributedSyncError("No allowlisted merged artifacts were found.")

    remote_prefix = PurePosixPath("distributed_runs", run_id, "merged")
    api = HfApi(token=hf_token)
    last_error: Optional[BaseException] = None
    logger = logging.getLogger(__name__)
    for attempt in range(max_retries + 1):
        try:
            repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision)
            parent_commit = getattr(repo_info, "sha", None)
            if not parent_commit:
                raise DistributedSyncError(f"Could not resolve the head commit for {repo_id}@{revision}.")
            operations = [
                CommitOperationAdd(
                    path_in_repo=str(remote_prefix / PurePosixPath(relative.as_posix())),
                    path_or_fileobj=str(local_path),
                )
                for local_path, relative in artifacts
            ]
            return api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                parent_commit=parent_commit,
                operations=operations,
                commit_message=f"Publish validated distributed merge: {run_id}",
            )
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries or not _is_retryable_hf_error(exc):
                break
            _wait_before_retry(logger, "Distributed merged-result commit", attempt + 1, retry_base)
    raise DistributedSyncError(
        f"Could not publish validated merged results for run {run_id}."
    ) from last_error


def _distributed_artifact_allowed(relative_path: Path) -> bool:
    if relative_path.is_absolute() or not relative_path.parts:
        return False
    lowered_parts = tuple(part.casefold() for part in relative_path.parts)
    if any(part.startswith(".") for part in lowered_parts):
        return False
    if any(marker in part for part in lowered_parts for marker in _DISTRIBUTED_SECRET_NAME_MARKERS):
        return False
    lowered_name = lowered_parts[-1]
    if lowered_name == "manifest.json" or lowered_name.endswith(_DISTRIBUTED_TEMP_SUFFIXES):
        return False
    if len(relative_path.parts) == 1:
        return lowered_name in _DISTRIBUTED_ALLOWED_ROOT_FILES
    # Worker resume needs only direct result artifacts (run log + Layer-1
    # cache) and the root status. Historical journals and ever-growing console
    # logs would otherwise be uploaded again after every batch.
    if len(relative_path.parts) != 2 or lowered_parts[0] not in _DISTRIBUTED_ALLOWED_DIRS:
        return False
    return relative_path.suffix.casefold() in _DISTRIBUTED_ALLOWED_SUFFIXES


def _collect_distributed_checkpoint_files(worker_root: Path) -> List[Tuple[Path, Path]]:
    artifacts: List[Tuple[Path, Path]] = []
    for candidate in sorted(worker_root.rglob("*")):
        if not candidate.is_file() or candidate.is_symlink():
            continue
        resolved_candidate = candidate.resolve()
        try:
            relative_path = resolved_candidate.relative_to(worker_root)
        except ValueError as exc:
            raise DistributedSyncError(f"Artifact escaped distributed worker root: {candidate}") from exc
        if _distributed_artifact_allowed(relative_path):
            artifacts.append((resolved_candidate, relative_path))
    return artifacts


def _sync_distributed_worker(config: dict):
    logger = logging.getLogger(__name__)
    run_id, worker_id, _ = _validate_distributed_identity(config)
    hf_token, repo_id = _distributed_repo_id(config)
    revision = _get_distributed_revision(config)
    max_retries, retry_base = _get_retry_settings(config)
    worker_root = _get_distributed_worker_root(config, create=False)
    worker_prefix = get_distributed_worker_remote_prefix(config)
    artifacts = _collect_distributed_checkpoint_files(worker_root)
    if not artifacts:
        raise DistributedSyncError(
            f"No allowlisted checkpoint artifacts were found below {worker_root}."
        )

    api = HfApi(token=hf_token)
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision)
            parent_commit = getattr(repo_info, "sha", None)
            if not parent_commit:
                raise DistributedSyncError(f"Could not resolve the head commit for {repo_id}@{revision}.")

            # CommitOperationAdd objects are intentionally recreated on every
            # attempt because huggingface_hub mutates them during create_commit.
            operations = [
                CommitOperationAdd(
                    path_in_repo=str(PurePosixPath(worker_prefix) / PurePosixPath(relative.as_posix())),
                    path_or_fileobj=str(local_path),
                )
                for local_path, relative in artifacts
            ]
            commit_info = api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                parent_commit=parent_commit,
                operations=operations,
                commit_message=(
                    f"Checkpoint {run_id} worker-{worker_id:03d}"
                ),
            )
            logger.info(
                "Atomically synchronized %d distributed checkpoint artifact(s) to %s/%s.",
                len(artifacts),
                repo_id,
                worker_prefix,
            )
            print(
                f"Distributed checkpoint durable on Hugging Face: "
                f"worker-{worker_id:03d}"
            )
            return commit_info
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries or not _is_retryable_hf_error(exc):
                break
            _wait_before_retry(logger, "Distributed worker checkpoint", attempt + 1, retry_base)

    raise DistributedSyncError(
        f"Could not make worker-{worker_id:03d} checkpoint durable "
        f"at {repo_id}@{revision}; no new batch should be started."
    ) from last_error


def _get_hf_config(config: dict) -> tuple:
    """Helper function to safely extract Hugging Face credentials from the config."""
    # This now correctly points to the token for synchronization.
    hf_token = config.get("HF_SYNC_TOKEN")
    hf_username = config.get("HF_HUB_USERNAME")
    repo_name = config.get("HF_HUB_REPO_NAME")
    return hf_token, hf_username, repo_name


def _get_workspace_download_settings(config: dict) -> Tuple[int, int, float]:
    """Validate rate-limit controls for a full non-distributed restore."""
    max_workers = config.get("HF_DOWNLOAD_MAX_WORKERS", 1)
    max_retries = config.get("HF_DOWNLOAD_MAX_RETRIES", 8)
    retry_base = config.get("HF_DOWNLOAD_RETRY_BASE_SECONDS", 15.0)
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers <= 0:
        raise ValueError("HF_DOWNLOAD_MAX_WORKERS must be a positive integer.")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("HF_DOWNLOAD_MAX_RETRIES must be a non-negative integer.")
    if isinstance(retry_base, bool) or not isinstance(retry_base, (int, float)) or retry_base < 0:
        raise ValueError("HF_DOWNLOAD_RETRY_BASE_SECONDS must be non-negative.")
    return max_workers, max_retries, float(retry_base)

def initialize_workspace(config: dict):
    """
    Downloads all files from the HF Hub repo to the local output directory.

    This populates the workspace with results from previous runs. It uses the
    HF token directly from the provided configuration dictionary.
    """
    logger = logging.getLogger(__name__)
    _print_startup_progress(config, "initialize_workspace() entered.")

    # Distributed workers restore only the immutable manifest and their own
    # checkpoint prefix. This branch is deliberately fail-closed because a
    # missing remote checkpoint can otherwise cause duplicate paid API calls.
    if _distributed_execution_enabled(config):
        _print_startup_progress(config, "Distributed restore selected; handing off to distributed initialization.")
        return _initialize_distributed_workspace(config)
    
    # 1. Check if persistence is enabled
    if not config.get("PERSIST_RESULTS_ONLINE"):
        logger.info("Online persistence is disabled. Skipping workspace initialization.")
        _print_startup_progress(config, "Online persistence is disabled; no remote restore will run.")
        return

    # 2. Extract credentials safely
    hf_token, hf_username, repo_name = _get_hf_config(config)

    # Check for empty credentials or default placeholders
    if not all([hf_token, hf_username, repo_name]) or "YOUR_HUGGING_FACE" in str(hf_token):
        logger.warning("HF token, username, or repo name not found in config. Cannot initialize workspace.")
        print("\n⚠️ WARNING: Missing or default Hugging Face configuration. Skipping workspace initialization.")
        return

    repo_id = f"{hf_username}/{repo_name}"
    # Changed: Download to BASE_OUTPUT_DIR instead of OUTPUTS_DIR for consistency
    # This allows all subdirectories (outputs, results, logs) to be downloaded and available
    local_outputs_dir = config["BASE_OUTPUT_DIR"]

    _print_startup_progress(
        config,
        f"Workspace restore configured for repo '{repo_id}' into '{local_outputs_dir}'.",
    )

    logger.info(f"Initializing workspace from Hugging Face Hub repo: {repo_id}")

    # 3. Handle specific revision settings
    revision = None
    if config.get("HF_SYNC_REVISION_ENABLED", False):
        revision_id = config.get("HF_SYNC_REVISION_ID")
        if revision_id:
            revision = revision_id
            logger.info(f"Downloading specific revision: {revision}")
        else:
            logger.warning("HF_SYNC_REVISION_ENABLED is True, but HF_SYNC_REVISION_ID is not set. Downloading latest.")
    
    # 4. Attempt API interactions
    try:
        # Instantiate the API client
        api = HfApi(token=hf_token)
        _print_startup_progress(config, "Hugging Face API client created.")

        # Ensure the repository exists, creating it if necessary
        with _StartupOperationReporter(config, "checking/creating the Hugging Face dataset repository"):
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        logger.info(f"Repository {repo_id} exists or was created successfully.")

        # Pin the branch head once so every retry restores one coherent latest
        # snapshot even if another notebook uploads while this one downloads.
        with _StartupOperationReporter(config, "resolving the Hugging Face revision"):
            repo_info = api.repo_info(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
            )
        resolved_revision = getattr(repo_info, "sha", None)
        if not resolved_revision:
            raise RuntimeError(f"Could not resolve a commit for {repo_id}.")

        max_workers, max_retries, retry_base = _get_workspace_download_settings(config)
        logger.info(
            "Restoring Hugging Face snapshot %s with %d worker(s) and up to %d retry round(s).",
            resolved_revision,
            max_workers,
            max_retries,
        )
        _print_startup_progress(
            config,
            "Snapshot download will start next "
            f"(workers={max_workers}, retry_rounds={max_retries}).",
        )

        # local_dir retains completed files plus Hub metadata. A retry therefore
        # continues the incomplete snapshot instead of downloading 5+ GB again.
        for attempt in range(max_retries + 1):
            try:
                with _StartupOperationReporter(
                    config,
                    f"downloading workspace snapshot (attempt {attempt + 1}/{max_retries + 1})",
                ):
                    downloaded_path = snapshot_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        local_dir=local_outputs_dir,
                        token=hf_token,
                        revision=resolved_revision,
                        max_workers=max_workers,
                    )
                break
            except Exception as exc:
                if attempt >= max_retries or not _is_retryable_hf_error(exc):
                    raise
                _wait_before_retry(
                    logger,
                    "Hugging Face workspace restore",
                    attempt + 1,
                    retry_base,
                )

        logger.info(f"Workspace synchronized. Files from {repo_id} are downloaded to {local_outputs_dir}.")
        print(f"✅ Hugging Face workspace successfully initialized from {repo_id}")
        return downloaded_path

    except HfHubHTTPError as e:
        # Clear, non-halting warning for HTTP/Authentication issues
        error_msg = f"HTTP Error initializing workspace from {repo_id}. Check your HF token permissions. Details: {str(e)}"
        print(f"\n⚠️ WARNING: HF Sync Failed (Download/Init HTTP Error) - {str(e)}")
        print("Continuing pipeline execution with local files...\n")
        logger.warning(error_msg)
        
    except Exception as e:
        # Clear, non-halting warning for any other unexpected errors
        error_msg = f"An unexpected error occurred during workspace initialization: {str(e)}"
        print(f"\n⚠️ WARNING: HF Sync Failed (Download/Init Unexpected Error) - {str(e)}")
        print("Continuing pipeline execution with local files...\n")
        logger.warning(error_msg)

def sync_workspace_to_hub(config: dict):
    """
    Uploads the entire local outputs directory to the Hugging Face Hub repo.

    Uses the HF token directly from the provided configuration dictionary.
    """
    logger = logging.getLogger(__name__)

    # A one-time finalizer checkpoints only the already validated merged tree.
    # This route is also used by Layer 2's existing periodic/final sync hooks.
    if _distributed_execution_enabled(config) and _distributed_finalizer_mode(config):
        merged_root = config.get("_DISTRIBUTED_MERGED_ROOT")
        if not merged_root:
            raise DistributedSyncError(
                "_DISTRIBUTED_MERGED_ROOT is required in distributed finalizer mode."
            )
        return sync_distributed_merged_results(config, merged_root)

    # Distributed synchronization is blocking, atomic, worker-isolated, and
    # raises on failure. Existing callers therefore stop before another batch
    # whenever the current checkpoint could not be made remotely durable.
    if _distributed_execution_enabled(config):
        return _sync_distributed_worker(config)

    if not config.get("PERSIST_RESULTS_ONLINE"):
        return # Silently exit if persistence is disabled.

    hf_token, hf_username, repo_name = _get_hf_config(config)

    # Check for empty credentials or default placeholders
    if not all([hf_token, hf_username, repo_name]) or "YOUR_HUGGING_FACE" in str(hf_token):
        logger.warning("HF token, username, or repo name not found or contains placeholders. Cannot sync workspace.")
        print("\n⚠️ WARNING: Hugging Face sync configuration is incomplete or contains placeholder values. Sync skipped.")
        return

    repo_id = f"{hf_username}/{repo_name}"
    # Changed: Upload BASE_OUTPUT_DIR instead of OUTPUTS_DIR to include both /outputs/ and /results/ directories
    # This ensures Layer-2 analytics output (saved to RESULTS_DIR) gets synced to HF Hub
    local_outputs_dir = config["BASE_OUTPUT_DIR"]

    logger.info(f"Starting synchronization of '{local_outputs_dir}' to HF Hub repo: {repo_id}")

    try:
        # 1. Instantiate the API client with the token.
        api = HfApi(token=hf_token)

        # 2. Upload only the specific folders using an allowlist to prevent leaks
        api.upload_large_folder(
            folder_path=local_outputs_dir,
            repo_id=repo_id,
            repo_type="dataset",
            # ONLY upload files inside these specific directories
            allow_patterns=[
                "outputs/logs/**",     # Logs folder in Kaggle mode
                "outputs/results/**",  # Results folder in Kaggle mode
                "logs/**",             # Logs folder in Offline mode
                "results/**"           # Results folder in Offline mode
            ]
        )
        logger.info(f"Successfully synced '{local_outputs_dir}' to {repo_id}.")
        print(f"✅ Backup complete: Results synced to Hugging Face ({repo_id})")
        
    except Exception as e:
        # Clear, non-halting warning
        print(f"\n⚠️ WARNING: HF Sync Failed (Upload Error) - {str(e)}")
        print("Continuing pipeline execution. Results are still saved locally...\n")
        logger.warning(f"Failed to sync workspace to Hugging Face Hub: {str(e)}")

def periodic_sync_check(loop_counter: int, config: dict):
    """
    Checks if a sync is needed based on the counter and sync interval.
    """
    if _distributed_execution_enabled(config) and not config.get("PERSIST_RESULTS_ONLINE"):
        raise DistributedSyncError(
            "Distributed execution requires PERSIST_RESULTS_ONLINE=True."
        )
    if not config.get("PERSIST_RESULTS_ONLINE"):
        return

    sync_interval = config.get("HF_SYNC_INTERVAL", 10)


    if (loop_counter + 1) % sync_interval == 0:
        print(f"\n--- Reached sync interval at item #{loop_counter + 1}. Syncing results to Hugging Face Hub. ---")
        sync_workspace_to_hub(config)
        print("--- Sync complete. ---\n")


_sync_lock = threading.Lock()
_active_sync_thread = None  

def _threaded_sync(config: dict):
    """Wrapper that only runs if no other sync is currently running."""
    # blocking=False means if the lock is already taken, it immediately gives up and returns
    if not _sync_lock.acquire(blocking=False):
        print("--- Background sync already in progress. Skipping this duplicate trigger. ---")
        return
    
    try:
        sync_workspace_to_hub(config)
    finally:
        # Always release the lock when finished, even if it crashes
        _sync_lock.release()

def periodic_batch_sync_check(batch_counter: int, config: dict):
    """Synchronize only after a committed batch, never from a worker thread."""
    global _active_sync_thread

    if _distributed_execution_enabled(config) and not config.get("PERSIST_RESULTS_ONLINE"):
        raise DistributedSyncError(
            "Distributed execution requires PERSIST_RESULTS_ONLINE=True."
        )
    
    if not config.get("PERSIST_RESULTS_ONLINE"):
        return
    if _distributed_execution_enabled(config):
        print(
            f"\n--- Committed distributed batch #{batch_counter + 1}. "
            "Synchronizing its checkpoint before continuing. ---"
        )
        return sync_workspace_to_hub(config)

    sync_interval = config.get("HF_SYNC_INTERVAL_BATCHES", config.get("HF_SYNC_INTERVAL", 10))
    if (batch_counter + 1) % max(1, int(sync_interval)) == 0:
        print(f"\n--- Reached committed batch #{batch_counter + 1}. Spawning background thread to sync to HF Hub. ---")
        
        _active_sync_thread = threading.Thread(target=_threaded_sync, args=(config,), daemon=False)
        _active_sync_thread.start()
        
        print("--- Background sync check requested. Pipeline continuing immediately... ---\n")

def wait_for_final_sync():
    """Call this at the end of your script to ensure uploads finish."""
    global _active_sync_thread
    if _active_sync_thread and _active_sync_thread.is_alive():
        print("\n⏳ Waiting for final Hugging Face background sync to finish before exiting...")
        _active_sync_thread.join()
        print("✅ Background sync complete!")
