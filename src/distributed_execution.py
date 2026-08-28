"""Deterministic, restart-safe execution across independent Kaggle workers.

The module deliberately contains no provider or Hugging Face imports.  It owns
the immutable run manifest, contiguous assignment, local worker layout,
checkpoint validation, status files, and the fail-closed local merge.  Hub I/O
is implemented in :mod:`src.hf_sync` so these invariants remain easy to test
without a network or optional ML dependencies.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


MANIFEST_SCHEMA_VERSION = 2
STATUS_SCHEMA_VERSION = 1
ASSIGNMENT_STRATEGY = "contiguous_balanced_chunks"
MODEL_ROTATION_CONFIG_KEYS = frozenset({
    "AVALAI_MODEL_NAME_ADAPTATION",
    "AVALAI_MODEL_NAME_FINAL_SOLVER",
    "AVALAI_MODEL_NAME_EVALUATOR",
})
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class DistributedExecutionError(RuntimeError):
    """Raised when a distributed worker cannot proceed safely."""


class DistributedManifestMismatch(DistributedExecutionError):
    """Raised when a run id is reused for different immutable inputs."""


class DistributedMergeError(DistributedExecutionError):
    """Raised when worker shards are not complete and mutually consistent."""


def distributed_enabled(config: Mapping[str, Any]) -> bool:
    return bool(config.get("DISTRIBUTED_EXECUTION_ENABLED", False))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: os.PathLike[str] | str, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _load_json(path: os.PathLike[str] | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_distributed_config(config: Mapping[str, Any]) -> Tuple[str, int, int]:
    """Validate and return ``(run_id, worker_id, worker_count)``."""
    run_id = str(config.get("DISTRIBUTED_RUN_ID") or "").strip()
    if not run_id or not _RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."}:
        raise DistributedExecutionError(
            "DISTRIBUTED_RUN_ID must contain only letters, numbers, '.', '_' or '-'."
        )
    try:
        worker_count = int(config.get("DISTRIBUTED_WORKER_COUNT", 5))
        worker_id = int(config.get("DISTRIBUTED_WORKER_ID", 0))
    except (TypeError, ValueError) as exc:
        raise DistributedExecutionError(
            "DISTRIBUTED_WORKER_ID and DISTRIBUTED_WORKER_COUNT must be integers."
        ) from exc
    if worker_count < 1:
        raise DistributedExecutionError("DISTRIBUTED_WORKER_COUNT must be at least 1.")
    if worker_id < 0 or worker_id >= worker_count:
        raise DistributedExecutionError(
            f"DISTRIBUTED_WORKER_ID must be in [0, {worker_count - 1}], got {worker_id}."
        )
    return run_id, worker_id, worker_count


def worker_directory_name(worker_id: int) -> str:
    if int(worker_id) < 0:
        raise DistributedExecutionError("Worker id cannot be negative.")
    return f"worker-{int(worker_id):03d}"


def validate_experiment_name(name: Any) -> str:
    value = str(name or "")
    if not _EXPERIMENT_NAME_RE.fullmatch(value):
        raise DistributedExecutionError(
            "Distributed experiment_name must be explicit, start with a letter/number, "
            "use only letters, numbers, '.', '_' or '-', and be at most 128 characters."
        )
    if value.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise DistributedExecutionError(
            f"Distributed experiment_name is reserved on Windows: {value}"
        )
    return value


def _result_json_filename(value: Any, setting_name: str) -> str:
    """Return one direct JSON result filename, rejecting path traversal/collisions."""
    filename = str(value or "").strip()
    path = Path(filename)
    if (
        not filename
        or filename in {".", ".."}
        or path.name != filename
        or "/" in filename
        or "\\" in filename
        or path.suffix.casefold() != ".json"
    ):
        raise DistributedExecutionError(
            f"{setting_name} must be a direct .json filename inside RESULTS_DIR."
        )
    return filename


def indexed_output_artifacts(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Describe sparse, index-addressed outputs produced by special workflows."""
    if config.get("APPLY_TRANSFORMATION_DATASET_CONSTRUCTION", False):
        return [{
            "filename": _result_json_filename(
                config.get(
                    "TRANSFORMATION_DS_OUTPUT_FILENAME",
                    "transformation_dataset.json",
                ),
                "TRANSFORMATION_DS_OUTPUT_FILENAME",
            ),
            "index_path": ["metadata", "original_index"],
        }]
    if config.get("APPLY_CORE_SIMP_PHASE1", False):
        return [{
            "filename": _result_json_filename(
                config.get("CORE_SIMP_DATASET_NAME", "core_simp_dataset.json"),
                "CORE_SIMP_DATASET_NAME",
            ),
            "index_path": ["original_index"],
        }]
    if config.get("APPLY_MERGING_DATASET_CONSTRUCTION", False):
        return [{
            "filename": _result_json_filename(
                config.get(
                    "MERGING_DS_OUTPUT_FILENAME",
                    "merging_fine_tuning_dataset.json",
                ),
                "MERGING_DS_OUTPUT_FILENAME",
            ),
            "index_path": ["metadata", "original_index"],
        }]
    return []


def apply_distributed_run_log_contract(
    result: Dict[str, Any],
    *,
    index: int,
    question: str,
    completed: bool,
) -> Dict[str, Any]:
    """Add the common shard-validation fields to a special-workflow result."""
    result["target_query_original_hard_list_idx"] = int(index)
    result["target_query_text"] = str(question)
    result["pipeline_status"] = "SUCCESS" if completed else "FAILED"
    return result


def assigned_indices(total_questions: int, worker_id: int, worker_count: int) -> List[int]:
    """Return one worker's contiguous, balanced range of global indices.

    Any remainder is assigned one item at a time to the lowest worker ids, so
    chunk sizes differ by at most one and every index is owned exactly once.
    """
    if total_questions < 0:
        raise DistributedExecutionError("Question count cannot be negative.")
    if worker_count < 1 or worker_id < 0 or worker_id >= worker_count:
        raise DistributedExecutionError("Invalid worker id/count for assignment.")
    base_size, remainder = divmod(total_questions, worker_count)
    start = worker_id * base_size + min(worker_id, remainder)
    stop = start + base_size + (1 if worker_id < remainder else 0)
    return list(range(start, stop))


def _assigned_worker_id(index: int, total_questions: int, worker_count: int) -> int:
    """Return the owner of one valid global index under contiguous chunking."""
    if index < 0 or index >= total_questions:
        raise DistributedExecutionError("Question index is outside the dataset.")
    if worker_count < 1:
        raise DistributedExecutionError("Worker count must be at least 1.")

    base_size, remainder = divmod(total_questions, worker_count)
    larger_region_size = (base_size + 1) * remainder
    if index < larger_region_size:
        return index // (base_size + 1)
    # When base_size is zero, every valid index is in the larger region.
    return remainder + (index - larger_region_size) // base_size


_SECRET_PARTS = (
    "API_KEY", "API_KEYS", "TOKEN", "PASSWORD", "SECRET", "CREDENTIAL",
    "AUTHORIZATION", "PRIVATE_KEY",
)
_RUNTIME_PREFIXES = (
    "DISTRIBUTED_", "HF_SYNC_", "BATCH_", "QUESTION_PARALLEL_",
)
_RUNTIME_EXACT = {
    "BASE_OUTPUT_DIR", "DATA_DIR", "LOGS_DIR", "OUTPUTS_DIR", "RESULTS_DIR",
    "EMBEDDINGS_DIR", "LAYER1_CACHE_DIR", "LOCAL_EMBEDDING_MODEL_PATH",
    "LOCAL_EXEMPLAR_CORPUS_PATH", "HARD_QUESTIONS_INDICES_PATH",
    "EMBEDDED_EXEMPLAR_CORPUS_QUESTIONS_PATH", "ADVANCED_RAG_FULL_LOG_PATH",
    "ADVANCED_RAG_EVALUATION_RESULTS_PATH", "PERSIST_RESULTS_ONLINE",
    "PRINT_API_CALL_DETAILS", "ENABLE_API_RETRY", "MAX_API_RETRIES",
    "API_RETRY_DELAY_SECONDS", "RETRY_ALL_API_ERRORS", "API_REQUEST_TIMEOUT_SECONDS",
    "API_KEY_ERROR_COOLDOWN_SECONDS", "ENABLE_MIN_TIME_BETWEEN_API_CALLS",
    "MIN_TIME_BETWEEN_API_CALLS_SECONDS", "HF_HUB_USERNAME", "HF_HUB_REPO_NAME",
    "GLOBAL_API_CALL_DELAY_SECONDS", "GLOBAL_CONSECUTIVE_ERROR_LIMIT",
    "GLOBAL_CONSECUTIVE_ERROR_PAUSE_SECONDS", "GLOBAL_PERIODIC_PAUSE_INTERVAL_MINUTES",
    "GLOBAL_PERIODIC_PAUSE_DURATION_SECONDS", "OFFLINE_MODE", "HARD_QUESTIONS_LENGTH",
}


def _is_runtime_or_secret_key(key: str) -> bool:
    upper = key.upper()
    if upper.startswith("_"):
        return True
    if any(part in upper for part in _SECRET_PARTS):
        return True
    if upper in _RUNTIME_EXACT or upper.startswith(_RUNTIME_PREFIXES):
        return True
    if upper.endswith("_DIR"):
        return True
    if upper.endswith("_PATH") and not upper.endswith("_PATH_HF"):
        return True
    if "QUOTA" in upper or "COOLDOWN" in upper or "TIMEOUT" in upper:
        return True
    return False


def sanitize_scientific_config(value: Any) -> Any:
    """Return JSON-safe scientific settings with credentials/runtime removed."""
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_scientific_config(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not _is_runtime_or_secret_key(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_scientific_config(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def active_avalai_models(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the model settings whose rotation can be explicitly authorized."""
    return {
        key: config.get(key)
        for key in sorted(MODEL_ROTATION_CONFIG_KEYS)
    }


def record_avalai_model_provenance(
    run_log: Dict[str, Any],
    config: Mapping[str, Any],
    run_mode: str,
) -> None:
    """Record active models for new work, including solve-only resumes."""
    models = active_avalai_models(config)
    flags = run_log.setdefault("config_flags_used", {})
    if isinstance(flags, dict):
        flags.update(models)
    record = {"run_mode": str(run_mode), "models": models}
    history = run_log.setdefault("avalai_model_config_history", [])
    if isinstance(history, list) and (not history or history[-1] != record):
        history.append(record)


def _manifest_without_rotatable_models(
    manifest: Mapping[str, Any],
    *,
    remove_code_fingerprint: bool = False,
) -> Dict[str, Any]:
    """Remove only model-rotation fields and hashes derived from those fields."""
    normalized = copy.deepcopy(dict(manifest))
    normalized.pop("manifest_sha256", None)
    normalized.pop("scientific_config_sha256", None)
    if remove_code_fingerprint:
        normalized.pop("code_fingerprint", None)
    for experiment in normalized.get("experiments", []):
        if not isinstance(experiment, dict):
            continue
        experiment_config = experiment.get("config")
        if not isinstance(experiment_config, dict):
            continue
        for key in MODEL_ROTATION_CONFIG_KEYS:
            experiment_config.pop(key, None)
    return normalized


def _manifest_model_changes(
    existing_manifest: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    existing_by_name = {
        experiment.get("experiment_name"): experiment
        for experiment in existing_manifest.get("experiments", [])
        if isinstance(experiment, Mapping)
    }
    changes: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for expected_experiment in expected_manifest.get("experiments", []):
        if not isinstance(expected_experiment, Mapping):
            continue
        name = expected_experiment.get("experiment_name")
        existing_experiment = existing_by_name.get(name, {})
        existing_config = existing_experiment.get("config", {})
        expected_config = expected_experiment.get("config", {})
        experiment_changes: Dict[str, Dict[str, Any]] = {}
        for key in sorted(MODEL_ROTATION_CONFIG_KEYS):
            previous = existing_config.get(key) if isinstance(existing_config, Mapping) else None
            active = expected_config.get(key) if isinstance(expected_config, Mapping) else None
            if previous != active:
                experiment_changes[key] = {"original": previous, "active": active}
        if experiment_changes:
            changes[str(name)] = experiment_changes
    return changes


def validate_manifest_compatibility(
    existing_manifest: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
    *,
    allow_model_rotation: bool = False,
    allow_legacy_code_fingerprint: bool = False,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Validate identity with a narrow opt-in bridge for pre-patch runs."""
    validate_manifest_integrity(existing_manifest)
    validate_manifest_integrity(expected_manifest)
    if _canonical_json(existing_manifest) == _canonical_json(expected_manifest):
        return {}

    if allow_model_rotation and (
        _canonical_json(_manifest_without_rotatable_models(
            existing_manifest,
            remove_code_fingerprint=allow_legacy_code_fingerprint,
        ))
        == _canonical_json(_manifest_without_rotatable_models(
            expected_manifest,
            remove_code_fingerprint=allow_legacy_code_fingerprint,
        ))
    ):
        changes = _manifest_model_changes(existing_manifest, expected_manifest)
        if changes:
            return changes

    changed = [
        key for key in (
            "run_id", "worker_count", "assignment_strategy", "code_fingerprint",
            "scientific_config_sha256", "dataset_sha256", "question_count",
        )
        if existing_manifest.get(key) != expected_manifest.get(key)
    ]
    legacy_rotation_guidance = ""
    if (
        allow_model_rotation
        and not allow_legacy_code_fingerprint
        and "code_fingerprint" in changed
    ):
        legacy_rotation_guidance = (
            " For a pre-patch run, set DISTRIBUTED_CODE_FINGERPRINT to the exact "
            "code_fingerprint stored in the existing manifest."
        )
    raise DistributedManifestMismatch(
        "Existing run manifest does not match this invocation"
        + (f" (changed: {', '.join(changed)})" if changed else "")
        + "."
        + legacy_rotation_guidance
        + " Otherwise use the original inputs/config or choose a new DISTRIBUTED_RUN_ID."
    )


def resolve_code_fingerprint(project_root: os.PathLike[str] | str) -> str:
    """Combine Git HEAD (when available) with the actual Python source bytes."""
    root = Path(project_root).resolve()
    revision: Optional[str] = None
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        revision = result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass

    digest = hashlib.sha256()
    for source in sorted(root.rglob("*.py")):
        if any(part in {".git", "__pycache__", ".ipynb_checkpoints"} for part in source.parts):
            continue
        relative = source.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    source_hash = digest.hexdigest()
    return f"git:{revision}:source:{source_hash}" if revision else f"source:{source_hash}"


def fingerprint_exemplar_data(exemplar_data: Optional[Mapping[str, Any]]) -> str:
    """Stream a stable identity for the retrieval corpus without serializing it."""
    digest = hashlib.sha256()
    if not exemplar_data:
        digest.update(b"null")
        return digest.hexdigest()
    for field in ("questions", "solutions"):
        values = exemplar_data.get(field, [])
        digest.update(field.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(str(len(values)).encode("ascii"))
        except TypeError:
            values = list(values)
            digest.update(str(len(values)).encode("ascii"))
        digest.update(b"\0")
        for value in values:
            encoded = str(value).encode("utf-8")
            digest.update(str(len(encoded)).encode("ascii"))
            digest.update(b":")
            digest.update(encoded)
            digest.update(b"\0")
    embeddings = exemplar_data.get("embeddings")
    shape = getattr(embeddings, "shape", None)
    digest.update(f"embedding_shape:{shape!r}".encode("utf-8"))
    return digest.hexdigest()


def build_run_manifest(
    config: Mapping[str, Any],
    experiment_configs: Sequence[Mapping[str, Any]],
    hard_questions: Sequence[str],
    hard_solutions: Optional[Sequence[Any]] = None,
    *,
    code_fingerprint: Optional[str] = None,
    exemplar_fingerprint: Optional[str] = None,
    project_root: Optional[os.PathLike[str] | str] = None,
) -> Dict[str, Any]:
    """Build the immutable manifest shared by every worker in a run."""
    run_id, _worker_id, worker_count = validate_distributed_config(config)
    if hard_solutions is not None and len(hard_solutions) != len(hard_questions):
        raise DistributedExecutionError(
            "hard_solutions must be None or have exactly the same length as hard_questions."
        )
    if not experiment_configs:
        raise DistributedExecutionError("At least one experiment config is required.")

    effective_experiments: List[Dict[str, Any]] = []
    names: set[str] = set()
    for overrides in experiment_configs:
        effective = dict(config)
        effective.update(dict(overrides))
        name = validate_experiment_name(effective.get("experiment_name"))
        if name in names:
            raise DistributedExecutionError(f"Duplicate experiment_name in distributed run: {name}")
        names.add(name)
        experiment_record = {
            "experiment_name": name,
            "config": sanitize_scientific_config(effective),
        }
        artifacts = {
            "run_log": f"{name}_run_log.json",
            "layer1_cache": (
                layer1_cache_filename(experiment_record)
                if effective.get("APPLY_LAYER1_BASE_EXECUTION", False)
                else None
            ),
        }
        indexed_outputs = indexed_output_artifacts(effective)
        if indexed_outputs:
            artifacts["indexed_outputs"] = indexed_outputs
        experiment_record["artifacts"] = artifacts
        effective_experiments.append(experiment_record)

    question_records: List[Dict[str, Any]] = []
    for index, question in enumerate(hard_questions):
        answer = None if hard_solutions is None else hard_solutions[index]
        question_records.append({
            "index": index,
            "worker_id": _assigned_worker_id(index, len(hard_questions), worker_count),
            "question_sha256": _sha256_text(str(question)),
            "answer_sha256": None if answer is None else _sha256_text(str(answer)),
        })

    if code_fingerprint is None:
        root = project_root or Path(__file__).resolve().parents[1]
        code_fingerprint = resolve_code_fingerprint(root)

    scientific_payload = {"experiments": effective_experiments}
    dataset_payload = {
        "question_count": len(question_records),
        "records": question_records,
    }
    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "worker_count": worker_count,
        "assignment_strategy": ASSIGNMENT_STRATEGY,
        "checkpoint_repository": {
            "repo_id": (
                f"{config.get('HF_HUB_USERNAME')}/{config.get('HF_HUB_REPO_NAME')}"
            ),
            "revision": str(config.get("DISTRIBUTED_HF_REVISION", "main")),
        },
        "code_fingerprint": str(code_fingerprint),
        "exemplar_sha256": str(exemplar_fingerprint or "not-provided"),
        "scientific_config_sha256": _sha256_json(scientific_payload),
        "dataset_sha256": _sha256_json(dataset_payload),
        "experiments": effective_experiments,
        "question_count": len(question_records),
        "questions": question_records,
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    return manifest


def validate_manifest_integrity(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise DistributedManifestMismatch(
            f"Unsupported distributed manifest schema: {manifest.get('schema_version')}"
        )
    stored = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    actual = _sha256_json(unsigned)
    if not stored or stored != actual:
        raise DistributedManifestMismatch("Manifest SHA-256 is missing or invalid.")
    questions = manifest.get("questions")
    if not isinstance(questions, list) or len(questions) != manifest.get("question_count"):
        raise DistributedManifestMismatch("Manifest question records are malformed.")
    worker_count = manifest.get("worker_count")
    if not isinstance(worker_count, int) or worker_count < 1:
        raise DistributedManifestMismatch("Manifest worker_count is invalid.")
    if manifest.get("assignment_strategy") != ASSIGNMENT_STRATEGY:
        raise DistributedManifestMismatch(
            f"Manifest assignment_strategy must be {ASSIGNMENT_STRATEGY!r}."
        )
    total_questions = len(questions)
    for expected_index, record in enumerate(questions):
        if record.get("index") != expected_index:
            raise DistributedManifestMismatch("Manifest question indices are not contiguous and ordered.")
        expected_worker_id = _assigned_worker_id(
            expected_index, total_questions, worker_count
        )
        if record.get("worker_id") != expected_worker_id:
            raise DistributedManifestMismatch(
                "Manifest assignment does not match contiguous balanced chunks."
            )
    experiments = manifest.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise DistributedManifestMismatch("Manifest experiments are missing or malformed.")
    experiment_names: set[str] = set()
    for experiment in experiments:
        if not isinstance(experiment, Mapping):
            raise DistributedManifestMismatch("Manifest contains a malformed experiment record.")
        try:
            name = validate_experiment_name(experiment.get("experiment_name"))
        except DistributedExecutionError as exc:
            raise DistributedManifestMismatch(str(exc)) from exc
        if name in experiment_names:
            raise DistributedManifestMismatch(f"Manifest repeats experiment_name {name}.")
        experiment_names.add(name)
        expected_artifacts = {
            "run_log": f"{name}_run_log.json",
            "layer1_cache": (
                layer1_cache_filename(experiment)
                if experiment.get("config", {}).get("APPLY_LAYER1_BASE_EXECUTION", False)
                else None
            ),
        }
        indexed_outputs = indexed_output_artifacts(experiment.get("config", {}))
        if indexed_outputs:
            expected_artifacts["indexed_outputs"] = indexed_outputs
        if experiment.get("artifacts") != expected_artifacts:
            raise DistributedManifestMismatch(
                f"Manifest artifact names are invalid for experiment {name}."
            )


def write_or_validate_manifest(
    path: os.PathLike[str] | str,
    expected_manifest: Mapping[str, Any],
    *,
    allow_model_rotation: bool = False,
) -> Path:
    """Create a local manifest once, or reject any unauthorized difference."""
    validate_manifest_integrity(expected_manifest)
    destination = Path(path)
    if destination.exists():
        existing = _load_json(destination)
        validate_manifest_compatibility(
            existing,
            expected_manifest,
            allow_model_rotation=allow_model_rotation,
        )
        return destination
    return _atomic_write_json(destination, dict(expected_manifest))


def configure_worker_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """Set isolated worker paths without duplicating shared data/model assets."""
    run_id, worker_id, _worker_count = validate_distributed_config(config)
    base_output = Path(str(config.get("BASE_OUTPUT_DIR") or ".")).resolve()
    run_root = base_output / "distributed_runs" / run_id
    worker_root = run_root / "workers" / worker_directory_name(worker_id)
    results_dir = worker_root / "results"
    logs_dir = worker_root / "logs"
    outputs_dir = worker_root / "outputs"
    for directory in (results_dir, logs_dir, outputs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config.update({
        "RESULTS_DIR": str(results_dir),
        "LAYER1_CACHE_DIR": str(results_dir),
        "LOGS_DIR": str(logs_dir),
        "OUTPUTS_DIR": str(outputs_dir),
        "_DISTRIBUTED_RUN_ROOT": str(run_root),
        "_DISTRIBUTED_WORKER_ROOT": str(worker_root),
        "_DISTRIBUTED_MANIFEST_PATH": str(run_root / "manifest.json"),
        "_DISTRIBUTED_WORKER_STATUS_PATH": str(worker_root / "status.json"),
    })
    return {
        "run_root": str(run_root),
        "worker_root": str(worker_root),
        "manifest_path": str(run_root / "manifest.json"),
        "status_path": str(worker_root / "status.json"),
    }


def start_worker_session(config: Dict[str, Any]) -> None:
    """Install one shared soft cutoff for every batch coordinator this session."""
    now = time.monotonic()
    origin = float(config.get("DISTRIBUTED_SESSION_ORIGIN_MONOTONIC", now))
    if origin <= 0 or origin > now:
        raise DistributedExecutionError(
            "DISTRIBUTED_SESSION_ORIGIN_MONOTONIC must be a prior monotonic timestamp."
        )
    stop_hours = float(config.get("DISTRIBUTED_SESSION_STOP_NEW_BATCH_HOURS", 11.0))
    api_hours = float(config.get("DISTRIBUTED_SESSION_API_DEADLINE_HOURS", 11.75))
    if stop_hours <= 0:
        raise DistributedExecutionError("DISTRIBUTED_SESSION_STOP_NEW_BATCH_HOURS must be > 0.")
    if api_hours < stop_hours:
        raise DistributedExecutionError(
            "DISTRIBUTED_SESSION_API_DEADLINE_HOURS must be >= the stop-new-batch cutoff."
        )
    config["_DISTRIBUTED_SESSION_ID"] = uuid.uuid4().hex
    config["_DISTRIBUTED_SESSION_STARTED_AT"] = _utc_now()
    config["_DISTRIBUTED_SESSION_STARTED_MONOTONIC"] = origin
    config["_DISTRIBUTED_STOP_NEW_BATCH_MONOTONIC"] = origin + stop_hours * 3600.0
    config["_DISTRIBUTED_API_DEADLINE_MONOTONIC"] = origin + api_hours * 3600.0
    config["_DISTRIBUTED_TIME_LIMIT_REACHED"] = False


def stop_new_batches_due(config: Mapping[str, Any], now: Optional[float] = None) -> bool:
    deadline = config.get("_DISTRIBUTED_STOP_NEW_BATCH_MONOTONIC")
    return bool(
        distributed_enabled(config)
        and deadline is not None
        and (time.monotonic() if now is None else now) >= float(deadline)
    )


def api_deadline_due(config: Mapping[str, Any], now: Optional[float] = None) -> bool:
    deadline = config.get("_DISTRIBUTED_API_DEADLINE_MONOTONIC")
    return bool(
        distributed_enabled(config)
        and deadline is not None
        and (time.monotonic() if now is None else now) >= float(deadline)
    )


def remaining_api_seconds(config: Mapping[str, Any], now: Optional[float] = None) -> Optional[float]:
    deadline = config.get("_DISTRIBUTED_API_DEADLINE_MONOTONIC")
    if not distributed_enabled(config) or deadline is None:
        return None
    current = time.monotonic() if now is None else now
    return max(0.0, float(deadline) - current)


def worker_question_records(manifest: Mapping[str, Any], worker_id: int) -> List[Mapping[str, Any]]:
    validate_manifest_integrity(manifest)
    return [record for record in manifest["questions"] if record["worker_id"] == worker_id]


def _run_log_index(log: Mapping[str, Any]) -> Optional[int]:
    try:
        return int(log.get("target_query_original_hard_list_idx"))
    except (TypeError, ValueError):
        return None


def layer1_state_complete(state: Any) -> bool:
    if not isinstance(state, Mapping):
        return False
    statuses = state.get("step_statuses")
    statuses_complete = (
        isinstance(statuses, Mapping)
        and bool(statuses)
        and all(str(value).upper() == "SUCCESS" for value in statuses.values())
    )
    metadata = state.get("metadata")
    last_step_complete = isinstance(metadata, Mapping) and metadata.get("last_completed_step") == 4
    overall = str(state.get("overall_status", "")).upper() == "SUCCESS"
    # Current Layer-1 states write all three signals together.  Requiring them
    # prevents a partially populated state from being accepted because one
    # optimistic/advisory field happened to say success.
    return statuses_complete and last_step_complete and overall


def run_log_successful(log: Mapping[str, Any], require_layer1: bool = False) -> bool:
    if str(log.get("pipeline_status", "")).upper() != "SUCCESS":
        return False
    if require_layer1 and not layer1_state_complete(log.get("layer1_base_execution_state")):
        return False
    return True


def validate_worker_run_logs(
    logs: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    worker_id: int,
    *,
    allow_identical_duplicates: bool = False,
) -> Dict[int, Mapping[str, Any]]:
    """Reject another worker's indices or a changed question in a checkpoint."""
    records = {record["index"]: record for record in worker_question_records(manifest, worker_id)}
    validated: Dict[int, Mapping[str, Any]] = {}
    for log in logs:
        if not isinstance(log, Mapping):
            raise DistributedExecutionError("Worker run log contains a non-object entry.")
        index = _run_log_index(log)
        if index is None:
            raise DistributedExecutionError("Worker run log contains an entry without a valid global index.")
        if index not in records:
            raise DistributedExecutionError(
                f"Worker {worker_id} checkpoint contains unassigned question index {index}."
            )
        question_text = log.get("target_query_text")
        if not isinstance(question_text, str) or _sha256_text(question_text) != records[index]["question_sha256"]:
            raise DistributedExecutionError(
                f"Worker {worker_id} checkpoint question hash mismatch at global index {index}."
            )
        if index in validated:
            if allow_identical_duplicates and _canonical_json(validated[index]) == _canonical_json(log):
                continue
            raise DistributedExecutionError(f"Conflicting duplicate run-log entry for index {index}.")
        validated[index] = log
    return validated


def pending_indices_for_worker(
    logs: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    worker_id: int,
    *,
    require_layer1: bool,
) -> List[int]:
    validated = validate_worker_run_logs(logs, manifest, worker_id)
    return [
        record["index"] for record in worker_question_records(manifest, worker_id)
        if not run_log_successful(validated.get(record["index"], {}), require_layer1=require_layer1)
    ]


def write_worker_status(
    config: Mapping[str, Any],
    state: str,
    *,
    manifest: Mapping[str, Any],
    experiment_progress: Optional[Mapping[str, Any]] = None,
    message: Optional[str] = None,
) -> Path:
    run_id, worker_id, worker_count = validate_distributed_config(config)
    normalized_state = str(state).upper()
    allowed = {"RUNNING", "PAUSED_TIME_LIMIT", "COMPLETE", "INCOMPLETE", "SYNC_ERROR"}
    if normalized_state not in allowed:
        raise DistributedExecutionError(f"Unknown worker state: {state}")
    payload: Dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "run_id": run_id,
        "worker_id": worker_id,
        "worker_count": worker_count,
        "manifest_sha256": manifest["manifest_sha256"],
        "state": normalized_state,
        "session_id": config.get("_DISTRIBUTED_SESSION_ID"),
        "session_started_at": config.get("_DISTRIBUTED_SESSION_STARTED_AT"),
        "updated_at": _utc_now(),
        "assigned_indices": assigned_indices(manifest["question_count"], worker_id, worker_count),
        "experiments": dict(experiment_progress or {}),
    }
    if config.get("DISTRIBUTED_ALLOW_MODEL_ROTATION", False):
        payload["model_rotation"] = {
            "enabled": True,
            "active_avalai_models": active_avalai_models(config),
            "runtime_code_fingerprint": config.get("_DISTRIBUTED_RUNTIME_CODE_FINGERPRINT"),
        }
    if message:
        payload["message"] = str(message)
    status_path = config.get("_DISTRIBUTED_WORKER_STATUS_PATH")
    if not status_path:
        raise DistributedExecutionError("Worker status path is not configured.")
    return _atomic_write_json(str(status_path), payload)


def layer1_cache_filename(experiment: Mapping[str, Any]) -> str:
    config = experiment.get("config", experiment)
    name = str(experiment.get("experiment_name") or config.get("experiment_name") or "default_experiment")
    safe_name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    top_k = int(config.get("TOP_N_CANDIDATES_RETRIEVAL", 5))
    candidates = config.get("LAYER1_ONE_SHOT_CANDIDATES_N")
    if candidates is None:
        candidates = config.get("LAYER1_N_CANDIDATES")
    if candidates is None:
        candidates = top_k
    zero_shot = int(config.get("LAYER1_ZERO_SHOT_CANDIDATES_N", 0))
    return f"layer1_cache_k{top_k}_n{int(candidates)}_z{zero_shot}_{safe_name}.json"


def validate_worker_layer1_cache(
    cache: Mapping[str, Any],
    manifest: Mapping[str, Any],
    worker_id: int,
    *,
    require_complete: bool = False,
) -> Dict[str, Any]:
    queries = cache.get("queries")
    if not isinstance(cache.get("metadata"), Mapping) or not isinstance(queries, Mapping):
        raise DistributedMergeError(f"Worker {worker_id} has a malformed Layer-1 cache shard.")
    expected = {record["index"]: record for record in worker_question_records(manifest, worker_id)}
    normalized: Dict[str, Any] = {}
    for key, state in queries.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise DistributedMergeError(f"Worker {worker_id} cache has invalid query key {key!r}.") from exc
        if index not in expected:
            raise DistributedMergeError(
                f"Worker {worker_id} Layer-1 cache contains unassigned index {index}."
            )
        if require_complete and not layer1_state_complete(state):
            raise DistributedMergeError(
                f"Worker {worker_id} Layer-1 cache is incomplete at index {index}."
            )
        query_data = state.get("target_query_data", {}) if isinstance(state, Mapping) else {}
        query_text = query_data.get("query_text") if isinstance(query_data, Mapping) else None
        metadata = state.get("metadata", {}) if isinstance(state, Mapping) else {}
        if query_data.get("query_index") != index or metadata.get("query_index") != index:
            raise DistributedMergeError(
                f"Worker {worker_id} Layer-1 nested query index mismatch at index {index}."
            )
        if not isinstance(query_text, str) or _sha256_text(query_text) != expected[index]["question_sha256"]:
            raise DistributedMergeError(
                f"Worker {worker_id} Layer-1 question hash mismatch at index {index}."
            )
        expected_answer_hash = expected[index].get("answer_sha256")
        if expected_answer_hash is not None:
            ground_truth = query_data.get("ground_truth_answer")
            if _sha256_text(str(ground_truth)) != expected_answer_hash:
                raise DistributedMergeError(
                    f"Worker {worker_id} Layer-1 ground-truth hash mismatch at index {index}."
                )
        normalized[str(index)] = state
    missing = sorted(set(expected) - {int(key) for key in normalized})
    if require_complete and missing:
        raise DistributedMergeError(
            f"Worker {worker_id} Layer-1 cache is missing assigned indices: {missing[:20]}"
        )
    return normalized


def _indexed_output_entry_index(
    entry: Mapping[str, Any], index_path: Sequence[str], filename: str
) -> int:
    value: Any = entry
    for key in index_path:
        if not isinstance(value, Mapping) or key not in value:
            raise DistributedMergeError(
                f"Indexed output {filename} contains an entry without "
                f"{'.'.join(index_path)}."
            )
        value = value[key]
    if isinstance(value, bool):
        raise DistributedMergeError(
            f"Indexed output {filename} contains a boolean global index."
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DistributedMergeError(
            f"Indexed output {filename} contains an invalid global index: {value!r}."
        ) from exc


def merge_distributed_run(
    run_root: os.PathLike[str] | str,
    *,
    output_dir: Optional[os.PathLike[str] | str] = None,
) -> Dict[str, Any]:
    """Validate every shard, then publish legacy-compatible merged artifacts.

    Validation finishes before the first merged file is written.  A
    ``MERGE_COMPLETE.json`` marker is written last; consumers must require it.
    """
    root = Path(run_root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise DistributedMergeError(f"Missing distributed manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    validate_manifest_integrity(manifest)
    worker_count = int(manifest["worker_count"])
    total_questions = int(manifest["question_count"])
    destination = Path(output_dir).resolve() if output_dir else root / "merged"

    statuses: Dict[int, Mapping[str, Any]] = {}
    incomplete: List[str] = []
    for worker_id in range(worker_count):
        worker_root = root / "workers" / worker_directory_name(worker_id)
        status_path = worker_root / "status.json"
        if not status_path.exists():
            incomplete.append(f"worker {worker_id}: missing status")
            continue
        status = _load_json(status_path)
        statuses[worker_id] = status
        if status.get("run_id") != manifest["run_id"]:
            incomplete.append(f"worker {worker_id}: run_id mismatch")
        elif status.get("worker_id") != worker_id or status.get("worker_count") != worker_count:
            incomplete.append(f"worker {worker_id}: identity mismatch")
        elif status.get("manifest_sha256") != manifest["manifest_sha256"]:
            incomplete.append(f"worker {worker_id}: manifest mismatch")
        elif status.get("assigned_indices") != assigned_indices(
            total_questions, worker_id, worker_count
        ):
            incomplete.append(f"worker {worker_id}: assigned-index list mismatch")
        elif status.get("state") != "COMPLETE":
            incomplete.append(f"worker {worker_id}: state={status.get('state', 'UNKNOWN')}")
    if incomplete:
        raise DistributedMergeError(
            "Distributed run is not mergeable; " + "; ".join(incomplete)
        )

    planned_outputs: Dict[Path, Any] = {}
    duplicate_recoveries: Dict[str, int] = {}
    for experiment in manifest["experiments"]:
        name = experiment["experiment_name"]
        merged_logs: Dict[int, Mapping[str, Any]] = {}
        duplicate_count = 0
        merged_cache_queries: Dict[str, Any] = {}
        cache_metadata: Optional[Dict[str, Any]] = None
        require_layer1 = bool(experiment["config"].get("APPLY_LAYER1_BASE_EXECUTION", False))
        indexed_output_specs = experiment.get("artifacts", {}).get(
            "indexed_outputs", []
        )
        if not isinstance(indexed_output_specs, list):
            raise DistributedMergeError(
                f"Manifest indexed outputs for {name} are malformed."
            )
        merged_indexed_outputs: Dict[str, Dict[int, Mapping[str, Any]]] = {}
        normalized_output_specs: List[Tuple[str, List[str]]] = []
        for spec in indexed_output_specs:
            if not isinstance(spec, Mapping):
                raise DistributedMergeError(
                    f"Manifest indexed output for {name} is malformed."
                )
            filename = _result_json_filename(
                spec.get("filename"), f"manifest artifact for {name}"
            )
            index_path = spec.get("index_path")
            if (
                not isinstance(index_path, list)
                or not index_path
                or any(not isinstance(key, str) or not key for key in index_path)
            ):
                raise DistributedMergeError(
                    f"Manifest index path for {name}/{filename} is malformed."
                )
            normalized_output_specs.append((filename, index_path))
            merged_indexed_outputs[filename] = {}

        for worker_id in range(worker_count):
            worker_results = root / "workers" / worker_directory_name(worker_id) / "results"
            log_path = worker_results / f"{name}_run_log.json"
            if not log_path.exists():
                raise DistributedMergeError(f"Worker {worker_id} is missing run log for {name}.")
            logs = _load_json(log_path)
            if not isinstance(logs, list):
                raise DistributedMergeError(f"Worker {worker_id} run log for {name} is not a list.")
            try:
                validated_logs = validate_worker_run_logs(
                    logs, manifest, worker_id, allow_identical_duplicates=True
                )
                duplicate_count += len(logs) - len(validated_logs)
            except DistributedExecutionError as exc:
                raise DistributedMergeError(str(exc)) from exc
            expected_worker_indices = set(assigned_indices(total_questions, worker_id, worker_count))
            if set(validated_logs) != expected_worker_indices:
                missing = sorted(expected_worker_indices - set(validated_logs))
                raise DistributedMergeError(
                    f"Worker {worker_id} run log for {name} is missing indices: {missing[:20]}"
                )
            for index, log in validated_logs.items():
                if not run_log_successful(log, require_layer1=require_layer1):
                    raise DistributedMergeError(
                        f"Worker {worker_id} has a non-successful result for {name}, index {index}."
                    )
                if index in merged_logs:
                    if _canonical_json(merged_logs[index]) != _canonical_json(log):
                        raise DistributedMergeError(f"Conflicting duplicate global index {index} for {name}.")
                    duplicate_count += 1
                else:
                    merged_logs[index] = log

            for filename, index_path in normalized_output_specs:
                artifact_path = worker_results / filename
                if not artifact_path.exists():
                    if expected_worker_indices:
                        raise DistributedMergeError(
                            f"Worker {worker_id} is missing indexed output for "
                            f"{name}: {filename}"
                        )
                    entries = []
                else:
                    entries = _load_json(artifact_path)
                if not isinstance(entries, list):
                    raise DistributedMergeError(
                        f"Worker {worker_id} indexed output {filename} is not a list."
                    )
                output_by_index = merged_indexed_outputs[filename]
                for entry in entries:
                    if not isinstance(entry, Mapping):
                        raise DistributedMergeError(
                            f"Worker {worker_id} indexed output {filename} contains "
                            "a non-object entry."
                        )
                    index = _indexed_output_entry_index(entry, index_path, filename)
                    if index not in expected_worker_indices:
                        raise DistributedMergeError(
                            f"Worker {worker_id} indexed output {filename} contains "
                            f"unassigned index {index}."
                        )
                    if index in output_by_index:
                        if _canonical_json(output_by_index[index]) != _canonical_json(entry):
                            raise DistributedMergeError(
                                f"Conflicting indexed-output duplicate {index} in {filename}."
                            )
                        duplicate_count += 1
                    else:
                        output_by_index[index] = entry

            if require_layer1:
                cache_path = worker_results / layer1_cache_filename(experiment)
                if not expected_worker_indices and not cache_path.exists():
                    continue
                if not cache_path.exists():
                    raise DistributedMergeError(
                        f"Worker {worker_id} is missing Layer-1 cache for {name}: {cache_path.name}"
                    )
                cache = _load_json(cache_path)
                normalized_queries = validate_worker_layer1_cache(
                    cache, manifest, worker_id, require_complete=True
                )
                if cache_metadata is None:
                    cache_metadata = dict(cache["metadata"])
                for key, state in normalized_queries.items():
                    if key in merged_cache_queries and _canonical_json(merged_cache_queries[key]) != _canonical_json(state):
                        raise DistributedMergeError(f"Conflicting Layer-1 duplicate index {key} for {name}.")
                    merged_cache_queries[key] = state

        if set(merged_logs) != set(range(total_questions)):
            raise DistributedMergeError(f"Merged run log for {name} does not cover every global index.")
        planned_outputs[destination / "results" / f"{name}_run_log.json"] = [
            merged_logs[index] for index in range(total_questions)
        ]
        duplicate_recoveries[name] = duplicate_count

        for filename, _index_path in normalized_output_specs:
            output_path = destination / "results" / filename
            if output_path in planned_outputs:
                raise DistributedMergeError(
                    f"Multiple experiments publish the same merged artifact: {filename}."
                )
            entries_by_index = merged_indexed_outputs[filename]
            planned_outputs[output_path] = [
                entries_by_index[index] for index in sorted(entries_by_index)
            ]

        if require_layer1:
            if set(map(int, merged_cache_queries)) != set(range(total_questions)):
                raise DistributedMergeError(f"Merged Layer-1 cache for {name} has a gap or extra index.")
            metadata = dict(cache_metadata or {})
            metadata.update({
                "experiment_name": name,
                "total_queries": total_questions,
                "completed_queries": list(range(total_questions)),
                "queries_in_progress": [],
                "distributed_run_id": manifest["run_id"],
                "distributed_manifest_sha256": manifest["manifest_sha256"],
                "distributed_worker_count": worker_count,
                "merged_at": _utc_now(),
            })
            planned_outputs[destination / "results" / layer1_cache_filename(experiment)] = {
                "metadata": metadata,
                "queries": {
                    str(index): merged_cache_queries[str(index)] for index in range(total_questions)
                },
            }

    # Nothing above this line mutates the canonical merged directory.
    for path, payload in planned_outputs.items():
        _atomic_write_json(path, payload)
    completion = {
        "schema_version": 1,
        "state": "COMPLETE",
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "worker_count": worker_count,
        "question_count": total_questions,
        "experiments": [experiment["experiment_name"] for experiment in manifest["experiments"]],
        "identical_duplicate_recoveries": duplicate_recoveries,
        "completed_at": _utc_now(),
        "output_files": sorted(path.relative_to(destination).as_posix() for path in planned_outputs),
    }
    marker = _atomic_write_json(destination / "MERGE_COMPLETE.json", completion)
    return {"manifest": manifest, "completion": completion, "marker_path": str(marker)}


__all__ = [
    "ASSIGNMENT_STRATEGY", "MODEL_ROTATION_CONFIG_KEYS", "DistributedExecutionError", "DistributedManifestMismatch",
    "active_avalai_models", "apply_distributed_run_log_contract",
    "indexed_output_artifacts", "record_avalai_model_provenance",
    "DistributedMergeError", "api_deadline_due", "assigned_indices", "build_run_manifest",
    "configure_worker_paths", "distributed_enabled", "fingerprint_exemplar_data", "layer1_cache_filename",
    "layer1_state_complete", "merge_distributed_run", "pending_indices_for_worker",
    "remaining_api_seconds", "resolve_code_fingerprint", "run_log_successful",
    "sanitize_scientific_config", "start_worker_session", "stop_new_batches_due",
    "validate_distributed_config", "validate_experiment_name", "validate_manifest_compatibility", "validate_manifest_integrity",
    "validate_worker_layer1_cache", "validate_worker_run_logs",
    "worker_directory_name", "worker_question_records", "write_or_validate_manifest",
    "write_worker_status",
]
