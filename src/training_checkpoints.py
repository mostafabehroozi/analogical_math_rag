"""Durable QLoRA training: upload every Trainer checkpoint to the Hub and resume from it.

A Kaggle session ends after 12 hours and takes ``/kaggle/working`` with it, so a
training run that needs longer would restart from zero. ``HubTrainingCheckpoints``
mirrors the newest ``checkpoint-N`` folder (adapter, optimizer, scheduler and RNG
state) plus the best-validation one to a private dataset repository. A new session
finds them by *training identity* (model, QLoRA settings and the exact tokenized
training rows) and hands the folder to ``Trainer.train(resume_from_checkpoint=...)``.

Each upload is ONE Hub commit that also deletes superseded checkpoints, so the
repository never holds a half-written checkpoint. Nothing here imports
``transformers`` at module level; only ``trainer_callback()`` needs it.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
import time
from array import array
from dataclasses import asdict, is_dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from huggingface_hub import CommitOperationAdd, CommitOperationDelete

from src.hub_resilience import DEFAULT_RETRY_DELAYS, call_with_retries
from src.merging_evaluation_checkpoints import canonical_fingerprint, differing_identity_keys
from src.utils import save_json_atomic


FORMAT_VERSION = 1
MANIFEST_FILE = "manifest.json"
POINTER_FILE = "latest.json"
LOCAL_MARKER = "training_identity.json"
UNTRUSTED_DIR = "_untrusted_checkpoints"
TRAINING_SOURCE_FILE = "training_source.json"
_CHECKPOINT = re.compile(r"^checkpoint-([0-9]+)$")
# Settings that change how often Trainer saves or where it writes, not what it learns.
_UNTRACKED_QLORA_KEYS = frozenset({"output_dir", "gpu_index", "checkpoint_steps"})
_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin", "model.safetensors", "pytorch_model.bin")


def qlora_settings(config: Any) -> Dict[str, Any]:
    """QLoRA settings that define a training run (paths and save cadence excluded)."""
    settings = asdict(config) if is_dataclass(config) and not isinstance(config, type) else dict(config)
    return {key: value for key, value in settings.items() if key not in _UNTRACKED_QLORA_KEYS}


def fingerprint_tokenized_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the exact token ids and labels Trainer will see, without building a JSON copy."""
    import hashlib

    digest = hashlib.sha256()
    for row in rows:
        for key in ("input_ids", "labels"):
            values = row[key]
            digest.update(len(values).to_bytes(8, "little"))
            digest.update(array("q", [int(value) for value in values]).tobytes())
    return digest.hexdigest()


def training_checkpoint_identity(
    qlora_config: Any, tokenized_splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *, workflow: str, extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Everything that decides what training learns; a checkpoint resumes only on an exact match."""
    identity = {
        "format_version": FORMAT_VERSION,
        "workflow": workflow,
        "qlora": qlora_settings(qlora_config),
        "data": {
            name: {"rows": len(tokenized_splits[name]),
                   "sha256": fingerprint_tokenized_rows(tokenized_splits[name])}
            for name in ("train", "validation")
        },
        "extra": dict(extra or {}),
    }
    return json.loads(json.dumps(identity, allow_nan=False))


def merging_training_source(manifest: Mapping[str, Any], qlora_config: Any) -> Dict[str, Any]:
    """Identity stored beside a trained merging adapter (parallels the simplification one)."""
    return json.loads(json.dumps({
        "format_version": FORMAT_VERSION, "workflow": "merging",
        "construction_sources": sorted(source["sha256"] for source in manifest["sources"]),
        "split_question_ids_sha256": canonical_fingerprint(manifest["split_question_ids"]),
        "split_seed": manifest["seed"], "split_ratios": manifest["split_ratios"],
        "qlora": qlora_settings(qlora_config),
    }))


def verify_adapter_training_source(
    adapter_dir: str | Path, expected: Mapping[str, Any], *,
    allow_unrecorded: bool = False, log: Callable[[str], None] = print,
) -> bool:
    """Refuse an adapter trained on other data or settings.

    Adapters uploaded before identities were recorded have no ``training_source.json``;
    with ``allow_unrecorded`` they are accepted with a warning instead of an error.
    Returns True when the identity was verified.
    """
    path = Path(adapter_dir) / TRAINING_SOURCE_FILE
    remedy = ("Use a new HF_MODEL_REPO_ID (and WORK_DIR) for each training configuration, "
              "or set HF_REUSE_ADAPTER_IF_AVAILABLE=False and TRAIN=True to retrain.")
    if not path.is_file():
        if allow_unrecorded:
            log(f"Adapter {adapter_dir} has no {TRAINING_SOURCE_FILE}, so its training data and "
                "settings cannot be verified (it was trained before identities were recorded).")
            return False
        raise ValueError(f"Adapter {adapter_dir} has no {TRAINING_SOURCE_FILE}, so its training "
                         f"data and settings cannot be verified. {remedy}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(dict(expected)))
    if saved != expected:
        changed = sorted(key for key in set(saved) | set(expected) if saved.get(key) != expected.get(key))
        raise ValueError(f"Adapter {adapter_dir} was trained with different {', '.join(changed)}. {remedy}")
    return True


def _finite(value: Any) -> Any:
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _is_missing_repository(error: BaseException) -> bool:
    return (type(error).__name__ == "RepositoryNotFoundError"
            and getattr(getattr(error, "response", None), "status_code", None) == 404)


class HubTrainingCheckpoints:
    """Upload Trainer checkpoints to a dataset repo and restore the newest matching one.

    ``identity`` comes from ``training_checkpoint_identity``. Credentials belong only in
    the constructor arguments. One notebook owns a training identity at a time.
    """

    # Subclasses for other fine-tuning workflows override these two names only.
    default_repo_name = "merging-qwen3-4b-training-state"
    label = "merging"

    def __init__(
        self, output_dir: str | Path, identity: Mapping[str, Any], *,
        token: Optional[str] = None, repo_id: Optional[str] = None,
        remote_prefix: str = "merging_training", private: bool = True,
        upload_enabled: bool = False, restore_enabled: bool = False, upload_every: int = 1,
        raise_on_upload_error: bool = False,
        api: Optional[Any] = None, download_fn: Optional[Callable[..., str]] = None,
        retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
        sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print,
    ):
        if not isinstance(identity, Mapping):
            raise ValueError("Training checkpoint identity must be a mapping.")
        if isinstance(upload_every, bool) or not isinstance(upload_every, int) or upload_every < 1:
            raise ValueError("upload_every must be a positive integer.")
        prefix = PurePosixPath(remote_prefix)
        if not remote_prefix or prefix.is_absolute() or ".." in prefix.parts or "\\" in remote_prefix:
            raise ValueError("remote_prefix must be a relative Hub directory.")
        self.identity = json.loads(json.dumps(dict(identity), allow_nan=False))
        self.fingerprint = canonical_fingerprint(self.identity)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.remote_root = f"{prefix}/{self.fingerprint}"
        self.manifest = {"format_version": FORMAT_VERSION, "fingerprint": self.fingerprint,
                         "identity": self.identity}
        self.upload_enabled = bool(upload_enabled)
        self.restore_enabled = bool(restore_enabled)
        self.upload_every = upload_every
        self.raise_on_upload_error = bool(raise_on_upload_error)
        self.private = private
        self.token = token
        self.repo_id = repo_id
        self._explicit_repo_id = bool(repo_id)
        self.api = api
        self.download_fn = download_fn
        self.retry_delays = tuple(retry_delays)
        self.sleep = sleep
        self.log = log
        self.upload_failures = 0
        self._saves = 0
        self._pending: Optional[Tuple[Path, Optional[str]]] = None
        self._uploaded_step: Optional[int] = None
        self._remote_ready = False

    # ------------------------------------------------------------------ Hub plumbing

    def _hub_call(self, what: str, operation: Callable[[], Any]) -> Any:
        return call_with_retries(operation, what, delays=self.retry_delays, sleep=self.sleep, log=self.log)

    def _configure_hub(self) -> None:
        if self.api is None:
            if not isinstance(self.token, str) or not self.token.strip():
                raise ValueError("A Hugging Face token is required for training checkpoint synchronization.")
            from huggingface_hub import HfApi
            self.api = HfApi(token=self.token)
        if self.download_fn is None:
            from huggingface_hub import snapshot_download
            self.download_fn = snapshot_download
        if not self.repo_id:
            identity = self.api.whoami()
            username = identity.get("name") if isinstance(identity, Mapping) else None
            if not username:
                raise RuntimeError("Could not determine the Hugging Face checkpoint owner.")
            self.repo_id = f"{username}/{self.default_repo_name}"
        if not isinstance(self.repo_id, str) or not re.fullmatch(r"[^/\s]+/[^/\s]+", self.repo_id):
            raise ValueError("Training checkpoint repo_id must use the 'owner/repository' format.")

    def _list(self) -> Tuple[List[str], str]:
        """File names in the repository and the revision they were read at."""
        info = self._hub_call("Reading the training checkpoint repository", lambda: self.api.repo_info(
            repo_id=self.repo_id, repo_type="dataset"))
        revision = getattr(info, "sha", None)
        if not revision:
            raise RuntimeError("Could not resolve the training checkpoint repository revision.")
        siblings = getattr(info, "siblings", None)
        if siblings is None:
            names = self._hub_call("Listing the training checkpoint repository", lambda: self.api.list_repo_files(
                repo_id=self.repo_id, repo_type="dataset", revision=revision))
        else:
            names = [item.rfilename for item in siblings]
        return list(names), revision

    def _download(self, revision: str, patterns: Sequence[str], local_dir: str | Path) -> None:
        self._hub_call("Downloading the training checkpoint", lambda: self.download_fn(
            repo_id=self.repo_id, repo_type="dataset", revision=revision, token=self.token,
            local_dir=str(local_dir), allow_patterns=list(patterns), max_workers=1))

    def _read_remote_json(self, revision: str, name: str) -> Any:
        with tempfile.TemporaryDirectory(prefix="training-checkpoint-read-") as staging:
            self._download(revision, [name], staging)
            return json.loads(Path(staging, *PurePosixPath(name).parts).read_text(encoding="utf-8"))

    @staticmethod
    def _save_json(path: Path, payload: Any, indent: int = 4) -> None:
        if not save_json_atomic(payload, str(path), indent=indent):
            raise OSError(f"Failed to save {path}.")

    # ------------------------------------------------------------------ local side

    @property
    def marker_path(self) -> Path:
        return self.output_dir / LOCAL_MARKER

    def local_checkpoints(self) -> List[Tuple[int, Path]]:
        found = []
        if self.output_dir.is_dir():
            for path in self.output_dir.iterdir():
                match = _CHECKPOINT.fullmatch(path.name)
                if match and path.is_dir() and (path / "trainer_state.json").is_file():
                    found.append((int(match.group(1)), path))
        return sorted(found)

    def _adopt_local_directory(self) -> None:
        """Trust local checkpoints only when they carry this training identity.

        Checkpoints from another configuration (or from before identities were
        recorded) are moved out of Trainer's way instead of being resumed or rotated.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        saved = None
        if self.marker_path.is_file():
            try:
                saved = json.loads(self.marker_path.read_text(encoding="utf-8")).get("fingerprint")
            except (OSError, ValueError):
                saved = None
        if saved == self.fingerprint:
            return
        stale = self.local_checkpoints()
        if stale:
            aside = self.output_dir / UNTRUSTED_DIR / (saved or "unrecorded")[:12]
            aside.mkdir(parents=True, exist_ok=True)
            for _, path in stale:
                destination = aside / path.name
                if destination.exists():
                    shutil.rmtree(destination)
                path.replace(destination)
            self.log(f"Moved {len(stale)} local checkpoint(s) from another training configuration to "
                     f"{aside}; they are not resumed.")
        self._save_json(self.marker_path, {"format_version": FORMAT_VERSION,
                                           "fingerprint": self.fingerprint, "identity": self.identity})

    @staticmethod
    def _repoint_best(latest_dir: Path, best_dir: Optional[Path]) -> None:
        """Trainer stores the best checkpoint as an absolute path; point it at this machine."""
        path = latest_dir / "trainer_state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state["best_model_checkpoint"] = str(best_dir) if best_dir is not None else None
        if not save_json_atomic(state, str(path), indent=2):
            raise OSError(f"Failed to update {path}.")

    # ------------------------------------------------------------------ restore

    def restore(self) -> Optional[str]:
        """Return the checkpoint folder to resume from, or None to train from the start.

        The newest of this session's own checkpoints and the Hub's wins. Also verifies
        write access (when uploading) before any GPU time is spent.
        """
        self._adopt_local_directory()
        local = self.local_checkpoints()
        chosen: Optional[Path] = self._restore_remote(local[-1][0] if local else -1) if self.restore_enabled else None
        if chosen is None and local:
            chosen = local[-1][1]
            self.log(f"Resuming training from the local {chosen.name}.")
        if self.upload_enabled:
            self._prepare_remote()
        return str(chosen) if chosen is not None else None

    def _restore_remote(self, local_step: int) -> Optional[Path]:
        self._configure_hub()
        try:
            names, revision = self._list()
        except Exception as error:
            if _is_missing_repository(error) and (not self._explicit_repo_id or self.upload_enabled):
                self.log("No Hugging Face training-checkpoint repository exists yet; training starts from the beginning.")
                return None
            raise
        root = self.remote_root + "/"
        mine = [name for name in names if name.startswith(root)]
        if not mine:
            self.log("No Hugging Face training checkpoint matches this configuration; training starts from the beginning.")
            self._print_other_identities(names, revision)
            return None
        if root + POINTER_FILE not in mine:
            if any(_CHECKPOINT.fullmatch(PurePosixPath(name).relative_to(root).parts[0]) for name in mine
                   if len(PurePosixPath(name).relative_to(root).parts) > 1):
                raise ValueError("Remote training checkpoint has checkpoint files but no pointer.")
            self.log("The Hub holds this training identity but no checkpoint yet; training starts from the beginning.")
            return None
        if root + MANIFEST_FILE not in mine:
            raise ValueError("Remote training checkpoint has files but no identity manifest.")
        manifest = self._read_remote_json(revision, root + MANIFEST_FILE)
        if manifest != self.manifest:
            raise ValueError("Remote training checkpoint identity/version does not match this run.")
        pointer = self._validate_pointer(self._read_remote_json(revision, root + POINTER_FILE))
        step = pointer["global_step"]
        if step <= local_step:
            self.log(f"Local checkpoint at step {local_step} is at least as new as the Hub's step {step}; keeping the local one.")
            return None
        wanted = [pointer["latest"]]
        if pointer["best"] and pointer["best"] != pointer["latest"]:
            wanted.append(pointer["best"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".restore-", dir=str(self.output_dir)))
        try:
            self._download(revision, [f"{root}{name}/*" for name in wanted], staging)
            folders = {name: staging.joinpath(*PurePosixPath(root).parts, name) for name in wanted}
            for name, folder in folders.items():
                for relative, size in pointer["files"][name].items():
                    file = folder.joinpath(*PurePosixPath(relative).parts)
                    if not file.is_file() or file.stat().st_size != size:
                        raise ValueError(f"Remote checkpoint file {name}/{relative} was not restored completely.")
            for name in reversed(wanted):
                target = self.output_dir / name
                if target.exists():
                    shutil.rmtree(target)
                folders[name].replace(target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        latest = self.output_dir / pointer["latest"]
        self._repoint_best(latest, self.output_dir / pointer["best"] if pointer["best"] else None)
        total = pointer.get("max_steps")
        self.log(f"Restored {pointer['latest']} (step {step}{f' of {total}' if total else ''}) from Hugging Face "
                 f"dataset {self.repo_id}@{revision[:12]}.")
        return latest

    def _validate_pointer(self, pointer: Any) -> Dict[str, Any]:
        def invalid(reason: str) -> ValueError:
            return ValueError(f"Invalid remote training checkpoint pointer: {reason}.")

        if not isinstance(pointer, dict) or pointer.get("format_version") != FORMAT_VERSION:
            raise invalid("unsupported format")
        if pointer.get("fingerprint") != self.fingerprint:
            raise invalid("it belongs to another training identity")
        step, latest, best, files = (pointer.get(key) for key in ("global_step", "latest", "best", "files"))
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise invalid("global_step")
        if not isinstance(latest, str) or not _CHECKPOINT.fullmatch(latest) or int(latest.split("-")[1]) != step:
            raise invalid("latest checkpoint name")
        if best is not None and (not isinstance(best, str) or not _CHECKPOINT.fullmatch(best)):
            raise invalid("best checkpoint name")
        if not isinstance(files, dict) or latest not in files or (best is not None and best not in files):
            raise invalid("file listing")
        for name, listing in files.items():
            if not isinstance(listing, dict) or not listing or any(
                not isinstance(relative, str) or PurePosixPath(relative).is_absolute()
                or ".." in PurePosixPath(relative).parts or isinstance(size, bool)
                or not isinstance(size, int) or size < 0
                for relative, size in listing.items()
            ):
                raise invalid(f"file listing of {name}")
            self._require_complete(name, listing)
        return pointer

    @staticmethod
    def _require_complete(name: str, listing: Mapping[str, Any]) -> None:
        if "trainer_state.json" not in listing or not any(weights in listing for weights in _WEIGHT_FILES):
            raise ValueError(f"{name} is incomplete: it needs trainer_state.json and model weights.")

    # ------------------------------------------------------------------ upload

    def _prepare_remote(self) -> None:
        if self._remote_ready:
            return
        self._configure_hub()
        self._hub_call("Creating the training checkpoint repository", lambda: self.api.create_repo(
            repo_id=self.repo_id, repo_type="dataset", private=self.private, exist_ok=True))
        # A first small commit proves write access before the expensive part starts.
        self._commit(
            lambda: [CommitOperationAdd(path_in_repo=f"{self.remote_root}/{MANIFEST_FILE}",
                                        path_or_fileobj=self._json_bytes(self.manifest))],
            f"Training checkpoint manifest {self.label}",
        )
        self._remote_ready = True

    @staticmethod
    def _json_bytes(payload: Any) -> bytes:
        return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")

    def _commit(self, build_operations: Callable[[], List[Any]], message: str) -> Any:
        # Operation objects are rebuilt per attempt: the Hub client marks them as it goes.
        return self._hub_call("Uploading the training checkpoint", lambda: self.api.create_commit(
            repo_id=self.repo_id, repo_type="dataset", operations=build_operations(), commit_message=message))

    @staticmethod
    def _file_sizes(folder: Path) -> Dict[str, int]:
        return {path.relative_to(folder).as_posix(): path.stat().st_size
                for path in sorted(folder.rglob("*")) if path.is_file()}

    def upload(self, checkpoint_dir: str | Path, best_name: Optional[str] = None) -> None:
        """Atomically publish ``checkpoint-N`` (and the best one) and drop older checkpoints."""
        self._prepare_remote()
        latest = Path(checkpoint_dir)
        match = _CHECKPOINT.fullmatch(latest.name)
        if not match or not (latest / "trainer_state.json").is_file():
            raise ValueError(f"{latest} is not a complete Trainer checkpoint folder.")
        step = int(match.group(1))
        folders = {latest.name: latest}
        if best_name and best_name != latest.name:
            candidate = latest.parent / best_name
            if (candidate / "trainer_state.json").is_file():
                folders[best_name] = candidate
        pointer_best = best_name if best_name in folders else None
        files = {name: self._file_sizes(folder) for name, folder in folders.items()}
        for name, listing in files.items():
            self._require_complete(name, listing)
        state = json.loads((latest / "trainer_state.json").read_text(encoding="utf-8"))
        pointer = {
            "format_version": FORMAT_VERSION, "fingerprint": self.fingerprint,
            "latest": latest.name, "best": pointer_best, "global_step": step,
            "max_steps": state.get("max_steps"), "epoch": _finite(state.get("epoch")),
            "best_metric": _finite(state.get("best_metric")), "files": files,
            "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        root = self.remote_root + "/"
        names, _ = self._list()
        remote_folders = {
            PurePosixPath(name).relative_to(root).parts[0] for name in names
            if name.startswith(root) and len(PurePosixPath(name).relative_to(root).parts) > 1
            and _CHECKPOINT.fullmatch(PurePosixPath(name).relative_to(root).parts[0])
        }

        def operations() -> List[Any]:
            ops: List[Any] = []
            for name, folder in folders.items():
                if name in remote_folders and name != latest.name:
                    continue  # a checkpoint folder never changes once saved
                ops.extend(CommitOperationAdd(path_in_repo=f"{root}{name}/{relative}",
                                              path_or_fileobj=str(folder / relative))
                           for relative in files[name])
            ops.extend(CommitOperationDelete(path_in_repo=f"{root}{stale}", is_folder=True)
                       for stale in sorted(remote_folders - set(folders)))
            ops.append(CommitOperationAdd(path_in_repo=root + MANIFEST_FILE,
                                          path_or_fileobj=self._json_bytes(self.manifest)))
            ops.append(CommitOperationAdd(path_in_repo=root + POINTER_FILE,
                                          path_or_fileobj=self._json_bytes(pointer)))
            return ops

        self._commit(operations, f"Training checkpoint {self.label} step {step}")
        self._uploaded_step = step
        total = state.get("max_steps")
        self.log(f"Training checkpoint {latest.name} uploaded to Hugging Face dataset {self.repo_id}"
                 f"{f' (step {step} of {total})' if total else ''}.")

    def on_checkpoint_saved(self, checkpoint_dir: str | Path, best_name: Optional[str] = None) -> bool:
        """Call after each Trainer save; returns True when this save reached the Hub.

        An upload failure is reported and retried at the next save instead of ending a
        long training run; ``raise_on_upload_error=True`` makes it fatal.
        """
        self._pending = (Path(checkpoint_dir), best_name)
        self._saves += 1
        if not self.upload_enabled or self._saves % self.upload_every:
            return False
        return self._upload_pending()

    def flush(self) -> bool:
        """Upload the newest save if the upload interval skipped it."""
        if not self.upload_enabled or self._pending is None:
            return False
        match = _CHECKPOINT.fullmatch(self._pending[0].name)
        if match and int(match.group(1)) == self._uploaded_step:
            return False
        return self._upload_pending()

    def _upload_pending(self) -> bool:
        checkpoint_dir, best_name = self._pending
        try:
            self.upload(checkpoint_dir, best_name)
            return True
        except Exception as error:
            self.upload_failures += 1
            if self.raise_on_upload_error:
                raise
            self.log(f"WARNING: could not upload {checkpoint_dir.name} to the Hub "
                     f"({type(error).__name__}: {str(error)[:200]}). Training continues and the next "
                     f"save retries; {self.upload_failures} upload(s) have failed so far.")
            return False

    def trainer_callback(self) -> Any:
        """A ``TrainerCallback`` that uploads on every save (imports ``transformers`` here only)."""
        from transformers import TrainerCallback

        store = self

        class HubCheckpointCallback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                best = Path(state.best_model_checkpoint).name if state.best_model_checkpoint else None
                store.on_checkpoint_saved(Path(args.output_dir) / f"checkpoint-{state.global_step}", best)

            def on_train_end(self, args, state, control, **kwargs):
                store.flush()

        return HubCheckpointCallback()

    # ------------------------------------------------------------------ read-only status

    def _other_identities(self, names: Sequence[str], revision: str) -> List[Dict[str, Any]]:
        prefix = PurePosixPath(self.remote_root).parent
        grouped: Dict[str, List[str]] = {}
        for name in names:
            try:
                parts = PurePosixPath(name).relative_to(prefix).parts
            except ValueError:
                continue
            if len(parts) >= 2 and parts[0] != self.fingerprint:
                grouped.setdefault(parts[0], []).append("/".join(parts[1:]))
        others = []
        for fingerprint, files in sorted(grouped.items()):
            entry: Dict[str, Any] = {"fingerprint": fingerprint, "global_step": None,
                                     "max_steps": None, "differs": None}
            try:
                if POINTER_FILE in files:
                    pointer = self._read_remote_json(revision, f"{prefix}/{fingerprint}/{POINTER_FILE}")
                    entry["global_step"], entry["max_steps"] = pointer.get("global_step"), pointer.get("max_steps")
                if MANIFEST_FILE in files:
                    saved = self._read_remote_json(revision, f"{prefix}/{fingerprint}/{MANIFEST_FILE}")
                    entry["differs"] = differing_identity_keys(self.identity, saved.get("identity", {}))
            except Exception as error:
                entry["error"] = f"{type(error).__name__}: {str(error)[:120]}"
            others.append(entry)
        return others

    def _print_other_identities(self, names: Sequence[str], revision: str) -> None:
        try:
            others = self._other_identities(names, revision)
            if not others:
                return
            self.log(f"The Hub holds training checkpoints from {len(others)} other configuration(s); they are not reused:")
            for entry in others[:5]:
                step = f"step {entry['global_step']}" if entry["global_step"] is not None else "no checkpoint"
                differs = ", ".join(entry["differs"][:6]) if entry["differs"] else "unknown"
                self.log(f"  {entry['fingerprint'][:12]}: {step}; differs in: {differs}")
        except Exception as error:
            self.log(f"Could not inspect other training checkpoints ({type(error).__name__}: {str(error)[:120]}).")

    def status(self) -> Dict[str, Any]:
        """Read-only: this configuration's newest Hub checkpoint and any from other identities."""
        self._configure_hub()
        result: Dict[str, Any] = {"repo_id": self.repo_id, "exists": True, "matching": None, "others": []}
        try:
            names, revision = self._list()
        except Exception as error:
            if _is_missing_repository(error):
                result["exists"] = False
                return result
            raise
        root = self.remote_root + "/"
        if root + POINTER_FILE in names:
            pointer = self._read_remote_json(revision, root + POINTER_FILE)
            result["matching"] = {key: pointer.get(key) for key in (
                "latest", "best", "global_step", "max_steps", "epoch", "best_metric", "uploaded_at")}
        result["others"] = self._other_identities(names, revision)
        return result


def describe_hub_progress(
    token: Optional[str], *, training: Optional[HubTrainingCheckpoints] = None,
    adapter: Optional[Mapping[str, Any]] = None, evaluation: Optional[Mapping[str, Any]] = None,
    api: Optional[Any] = None, log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Print and return what the Hub already holds for each pipeline stage (read-only).

    ``adapter``: ``repo_id`` (None for the default), ``default_repo_name``, ``reuse``.
    ``evaluation``: ``repo_id``, ``default_repo_name``, ``prefix``, ``populations``.
    A stage that cannot be checked is reported and never stops the notebook.
    """
    owner: Dict[str, Optional[str]] = {}
    clients: List[Any] = [api] if api is not None else []

    def hub() -> Any:
        if not clients:
            from huggingface_hub import HfApi
            clients.append(HfApi(token=token))
        return clients[0]

    def repo_for(config: Mapping[str, Any]) -> str:
        if config.get("repo_id"):
            return config["repo_id"]
        if "name" not in owner:
            identity = hub().whoami()
            owner["name"] = identity.get("name") if isinstance(identity, Mapping) else None
        if not owner["name"]:
            raise RuntimeError("could not determine the Hugging Face username")
        return f"{owner['name']}/{config['default_repo_name']}"

    report: Dict[str, Any] = {}
    lines = ["Hugging Face progress check (read-only):"]

    if training is not None:
        try:
            status = training.status()
            report["training"] = status
            where = f"{status['repo_id']}"
            if not training.restore_enabled:
                lines.append(f"  training    {where}: resume disabled (HF_TRAIN_RESTORE_ENABLED=False)")
            elif not status["exists"]:
                lines.append(f"  training    {where}: no repository yet -> training starts from the beginning")
            elif status["matching"]:
                m = status["matching"]
                of = f" of {m['max_steps']}" if m.get("max_steps") else ""
                lines.append(f"  training    {where}: {m['latest']} (step {m['global_step']}{of}) matches this "
                             "configuration -> training resumes from it")
            else:
                lines.append(f"  training    {where}: no checkpoint for this configuration -> training starts from the beginning")
            for other in status["others"][:3]:
                step = f"step {other['global_step']}" if other["global_step"] is not None else "no checkpoint"
                differs = ", ".join((other["differs"] or ["unknown"])[:4])
                lines.append(f"                other configuration {other['fingerprint'][:12]} ({step}) is not reused; differs in: {differs}")
        except Exception as error:
            report["training"] = {"error": str(error)}
            lines.append(f"  training    could not check ({type(error).__name__}: {str(error)[:150]})")

    if adapter is not None:
        try:
            repo_id = repo_for(adapter)
            if not adapter.get("reuse", True):
                lines.append(f"  adapter     {repo_id}: reuse disabled (HF_REUSE_ADAPTER_IF_AVAILABLE=False)")
            else:
                try:
                    info = call_with_retries(lambda: hub().model_info(repo_id=repo_id), "Reading the adapter repository", log=log)
                except Exception as error:
                    if not _is_missing_repository(error):
                        raise
                    info = None
                if info is None:
                    report["adapter"] = {"repo_id": repo_id, "exists": False}
                    lines.append(f"  adapter     {repo_id}: not found -> training runs (or resumes) and uploads it")
                else:
                    files = {item.rfilename for item in info.siblings}
                    complete = ("adapter_config.json" in files and bool(files & {"adapter_model.safetensors", "adapter_model.bin"})
                                and bool(files & {"tokenizer_config.json", "tokenizer.json"}))
                    recorded = TRAINING_SOURCE_FILE in files
                    report["adapter"] = {"repo_id": repo_id, "exists": True, "complete": complete,
                                         "revision": info.sha, "training_source": recorded}
                    if not complete:
                        lines.append(f"  adapter     {repo_id}: present but incomplete -> the notebook will stop; fix or remove it")
                    else:
                        note = "" if recorded else " (no training_source.json: its training data cannot be verified)"
                        lines.append(f"  adapter     {repo_id}@{str(info.sha)[:7]}: complete -> training is skipped{note}")
        except Exception as error:
            report["adapter"] = {"error": str(error)}
            lines.append(f"  adapter     could not check ({type(error).__name__}: {str(error)[:150]})")

    if evaluation is not None:
        try:
            repo_id = repo_for(evaluation)
            try:
                info = call_with_retries(lambda: hub().repo_info(repo_id=repo_id, repo_type="dataset"),
                                         "Reading the evaluation repository", log=log)
                names = ([item.rfilename for item in info.siblings] if getattr(info, "siblings", None) is not None
                         else list(hub().list_repo_files(repo_id=repo_id, repo_type="dataset")))
            except Exception as error:
                if not _is_missing_repository(error):
                    raise
                names = None
            if names is None:
                report["evaluation"] = {"repo_id": repo_id, "exists": False}
                lines.append(f"  evaluation  {repo_id}: no repository yet -> every benchmark starts from the beginning")
            else:
                lines.append(f"  evaluation  {repo_id} (an identity is matched when each benchmark starts):")
                per_population: Dict[str, Dict[str, int]] = {}
                for population in evaluation["populations"]:
                    root = f"{evaluation['prefix']}/{population}/"
                    identities: Dict[str, int] = {}
                    for name in names:
                        if name.startswith(root):
                            parts = PurePosixPath(name).relative_to(root).parts
                            if len(parts) == 2:
                                identities.setdefault(parts[0], 0)
                                if re.fullmatch(r"question-[0-9]{8,}\.json", parts[1]):
                                    identities[parts[0]] += 1
                    per_population[population] = identities
                    if identities:
                        detail = "; ".join(f"{fp[:12]}: {count} question files" for fp, count in sorted(identities.items())[:4])
                        lines.append(f"      {population}: {len(identities)} saved identit{'y' if len(identities) == 1 else 'ies'} ({detail})")
                    else:
                        lines.append(f"      {population}: nothing saved yet")
                report["evaluation"] = {"repo_id": repo_id, "exists": True, "populations": per_population}
        except Exception as error:
            report["evaluation"] = {"error": str(error)}
            lines.append(f"  evaluation  could not check ({type(error).__name__}: {str(error)[:150]})")

    log("\n".join(lines))
    return report
