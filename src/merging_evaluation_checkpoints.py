"""Durable, identity-checked checkpoints for the merging evaluation notebook."""

import copy
import hashlib
import json
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Mapping, Optional

from src.utils import save_json_atomic


FORMAT_VERSION = 1
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_QUESTION_PATTERN = re.compile(r"^question-([0-9]{8,})\.json$")


def canonical_fingerprint(value: Any) -> str:
    """Hash JSON content independently of dictionary insertion order."""
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def fingerprint_merging_adapter(adapter_dir) -> str:
    """Identify adapter/tokenizer contents equally before and after Hub restore."""
    folder = Path(adapter_dir).expanduser().resolve()
    config = folder / "adapter_config.json"
    weights = sorted(
        path for path in folder.glob("adapter_model*")
        if path.is_file() and path.suffix in {".safetensors", ".bin", ".json"}
    )
    if not config.is_file() or not any(path.suffix in {".safetensors", ".bin"} for path in weights):
        raise ValueError(f"No complete merging adapter found in {folder}.")
    tokenizer_files = [
        path for path in folder.iterdir()
        if path.is_file() and (
            path.name.startswith("tokenizer")
            or path.name in {
                "special_tokens_map.json", "added_tokens.json", "vocab.json", "vocab.txt",
                "merges.txt", "spiece.model", "sentencepiece.bpe.model", "chat_template.jinja",
            }
        )
    ]
    tokenizer_files.extend(path for path in (folder / "chat_templates").glob("*.jinja") if path.is_file())
    files = sorted({config, *weights, *tokenizer_files}, key=lambda path: path.relative_to(folder).as_posix())
    digests = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digests[path.relative_to(folder).as_posix()] = digest.hexdigest()
    return canonical_fingerprint(digests)


class MergingEvaluationCheckpoint:
    """Save each successful inference and resume one benchmark from a dataset repo.

    ``identity`` must contain a benchmark plus every input/configuration that
    affects generation or judgment. Credentials belong only in constructor
    arguments. One notebook owns each checkpoint prefix at a time.
    """

    def __init__(
        self, report_dir, identity: Mapping[str, Any], *, token: Optional[str] = None,
        repo_id: Optional[str] = None, remote_prefix: str = "merging_evaluations",
        upload_enabled: bool = False, restore_enabled: bool = False,
        private: bool = True, upload_every: int = 10,
        api: Optional[Any] = None, download_fn: Optional[Callable[..., str]] = None,
    ):
        if not isinstance(identity, Mapping):
            raise ValueError("Evaluation checkpoint identity must be a mapping.")
        self.identity = json.loads(json.dumps(dict(identity), allow_nan=False))
        benchmark = self.identity.get("benchmark")
        if not isinstance(benchmark, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", benchmark) or benchmark in {".", ".."}:
            raise ValueError("Checkpoint identity must contain a valid benchmark name.")
        if isinstance(upload_every, bool) or not isinstance(upload_every, int) or upload_every < 1:
            raise ValueError("upload_every must be a positive integer.")
        prefix = PurePosixPath(remote_prefix)
        if not remote_prefix or prefix.is_absolute() or ".." in prefix.parts or "\\" in remote_prefix:
            raise ValueError("remote_prefix must be a relative Hub directory.")
        self.fingerprint = canonical_fingerprint(self.identity)
        self.checkpoint_dir = Path(report_dir).expanduser().resolve() / "checkpoints" / self.fingerprint
        self.remote_prefix = str(prefix / benchmark / self.fingerprint)
        self.upload_enabled = bool(upload_enabled)
        self.restore_enabled = bool(restore_enabled)
        self.upload_every = upload_every
        self.token = token
        self.repo_id = repo_id
        self._explicit_repo_id = bool(repo_id)
        self.api = api
        self.download_fn = download_fn
        self.cache_hits = 0
        self._pending_operations = 0
        self._dirty_files = {"manifest.json"}
        self._states: Dict[int, Dict[str, Any]] = {}
        self.manifest = {
            "format_version": FORMAT_VERSION, "fingerprint": self.fingerprint,
            "identity": self.identity,
        }
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._states = self._read_folder(self.checkpoint_dir, allow_missing=True)
        self._dirty_files.update(self._question_path(index).name for index in self._states)
        # Write identity before copying restored state, so interruption during
        # restore still leaves a valid, resumable local checkpoint directory.
        self._save_file(self.checkpoint_dir / "manifest.json", self.manifest)
        if self.upload_enabled or self.restore_enabled:
            self._configure_hub()
        if self.restore_enabled:
            self._restore()
        if self.upload_enabled:
            # Authenticate and validate repository permissions before paid work.
            self.api.create_repo(
                repo_id=self.repo_id, repo_type="dataset", private=private, exist_ok=True,
            )
            # A first small checkpoint also verifies actual write access.
            self.sync()
        print(
            f"Evaluation checkpoint: {benchmark}, identity {self.fingerprint[:12]}, "
            f"{sum(state['completed'] for state in self._states.values())} completed questions restored."
        )

    def _configure_hub(self) -> None:
        if self.api is None:
            if not isinstance(self.token, str) or not self.token.strip():
                raise ValueError("A Hugging Face token is required for evaluation checkpoint synchronization.")
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
            self.repo_id = f"{username}/merging-qwen3-4b-evaluation"
        if not isinstance(self.repo_id, str) or not re.fullmatch(r"[^/\s]+/[^/\s]+", self.repo_id):
            raise ValueError("Evaluation dataset repo_id must use the 'owner/repository' format.")

    @staticmethod
    def _save_file(path: Path, payload: Any) -> None:
        if not save_json_atomic(payload, str(path)):
            raise OSError(f"Failed to save merging evaluation checkpoint {path}.")

    @staticmethod
    def _load_file(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid merging evaluation checkpoint {path}.") from exc

    def _validate_state(self, state: Any, expected_index: Optional[int] = None) -> None:
        if not isinstance(state, dict):
            raise ValueError("Question checkpoint must be a JSON object.")
        index = state.get("benchmark_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or (expected_index is not None and index != expected_index):
            raise ValueError("Question checkpoint benchmark index is invalid or differs from its filename.")
        for cache_name in ("generations", "judgments"):
            cache = state.get(cache_name)
            if not isinstance(cache, dict) or any(
                not isinstance(key, str) or not _HASH_PATTERN.fullmatch(key) or not isinstance(value, dict)
                for key, value in cache.items()
            ):
                raise ValueError(f"Invalid question checkpoint {cache_name} cache.")
        if any(value.get("status") != "SUCCESS" for value in state["generations"].values()):
            raise ValueError("Generation checkpoints must contain successful inference results.")
        for name in ("phase_1_runs", "phase_2_runs"):
            if not isinstance(state.get(name), list) or any(not isinstance(run, dict) for run in state[name]):
                raise ValueError(f"Invalid question checkpoint {name}.")
            if any(
                run.get("benchmark_index") != index
                or run.get("benchmark") != self.identity["benchmark"]
                for run in state[name]
            ):
                raise ValueError("Question checkpoint phase results belong to another benchmark row.")
        if not isinstance(state.get("completed"), bool):
            raise ValueError("Question checkpoint completed flag must be a boolean.")
        if state["completed"]:
            phases = [
                f"phase_{number}_runs" for number in (1, 2)
                if self.identity.get(f"run_phase_{number}", False)
            ]
            if any(not state[phase] for phase in phases) or (
                not any(state[phase] for phase in ("phase_1_runs", "phase_2_runs"))
                and not all(self.identity.get(f"run_phase_{number}") is False for number in (1, 2))
            ):
                raise ValueError("Completed question checkpoint is missing enabled phase results.")
        if state.get("retrieved") is not None and not isinstance(state["retrieved"], list):
            raise ValueError("Question checkpoint retrieved exemplars must be a list.")
        # Ensure neither unsupported objects nor non-finite floats reach files.
        canonical_fingerprint(state)

    def _read_folder(self, folder: Path, *, allow_missing: bool = False) -> Dict[int, Dict[str, Any]]:
        manifest_path = folder / "manifest.json"
        files = list(folder.glob("*.json"))
        if not manifest_path.is_file():
            if allow_missing and not files:
                return {}
            raise ValueError("Evaluation checkpoint files exist without their identity manifest.")
        if self._load_file(manifest_path) != self.manifest:
            raise ValueError("Evaluation checkpoint identity/version does not match this benchmark run.")
        states = {}
        for path in files:
            if path.name == "manifest.json":
                continue
            match = _QUESTION_PATTERN.fullmatch(path.name)
            if not match:
                raise ValueError(f"Unexpected evaluation checkpoint filename: {path.name}.")
            index = int(match.group(1))
            if path.name != self._question_path(index).name:
                raise ValueError("Question checkpoint filename is not canonical.")
            state = self._load_file(path)
            self._validate_state(state, index)
            if index in states:
                raise ValueError("Duplicate question index in evaluation checkpoint folder.")
            states[index] = state
        return states

    @staticmethod
    def _merge_states(local: Dict[str, Any], remote: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(local)
        for cache_name in ("generations", "judgments"):
            for key, value in remote[cache_name].items():
                if key in merged[cache_name] and merged[cache_name][key] != value:
                    local_success = merged[cache_name][key].get("status") == "SUCCESS"
                    remote_success = value.get("status") == "SUCCESS"
                    if cache_name == "judgments" and local_success != remote_success:
                        if local_success:
                            continue
                    elif cache_name == "judgments" and not local_success and not remote_success:
                        # Neither failed judgment is reusable; the notebook retries it.
                        continue
                    else:
                        raise ValueError(f"Conflicting {cache_name} results in local and remote evaluation checkpoints.")
                merged[cache_name][key] = copy.deepcopy(value)
        if local.get("retrieved") is not None and remote.get("retrieved") is not None and local["retrieved"] != remote["retrieved"]:
            raise ValueError("Conflicting retrieved exemplars in evaluation checkpoints.")
        if merged.get("retrieved") is None:
            merged["retrieved"] = copy.deepcopy(remote.get("retrieved"))
        for phase in ("phase_1_runs", "phase_2_runs"):
            if local["completed"] and remote["completed"] and local[phase] != remote[phase]:
                raise ValueError(f"Conflicting {phase} results in evaluation checkpoints.")
            if remote["completed"] and not local["completed"] or not merged[phase]:
                merged[phase] = copy.deepcopy(remote[phase])
        merged["completed"] = local["completed"] or remote["completed"]
        return merged

    def _restore(self) -> None:
        try:
            info = self.api.repo_info(repo_id=self.repo_id, repo_type="dataset")
        except Exception as exc:
            # A genuinely absent dataset repository is a new run. Authentication,
            # connection and other Hub errors must not restart costly inference.
            missing = (
                type(exc).__name__ == "RepositoryNotFoundError"
                and getattr(getattr(exc, "response", None), "status_code", None) == 404
            )
            if missing and (not self._explicit_repo_id or self.upload_enabled):
                print("No Hugging Face evaluation dataset checkpoint repository exists yet; starting a fresh run.")
                return
            raise
        revision = getattr(info, "sha", None)
        if not revision:
            raise RuntimeError("Could not resolve the evaluation dataset checkpoint revision.")
        siblings = getattr(info, "siblings", None)
        if siblings is None:
            names = self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset", revision=revision)
        else:
            names = [item.rfilename for item in siblings]
        prefix = self.remote_prefix + "/"
        matching = [name for name in names if name.startswith(prefix)]
        if not matching:
            print("No matching Hugging Face evaluation checkpoint exists; starting this benchmark identity.")
            return
        if prefix + "manifest.json" not in matching:
            raise ValueError("Remote evaluation checkpoint has files but no identity manifest.")
        with tempfile.TemporaryDirectory(prefix="merging-evaluation-restore-") as staging_dir:
            self.download_fn(
                repo_id=self.repo_id, repo_type="dataset", revision=revision,
                token=self.token, local_dir=staging_dir,
                allow_patterns=[prefix + "*.json"], max_workers=1,
            )
            folder = Path(staging_dir).joinpath(*PurePosixPath(self.remote_prefix).parts)
            for name in matching:
                relative = PurePosixPath(name).relative_to(PurePosixPath(self.remote_prefix))
                if relative.suffix == ".json" and not folder.joinpath(*relative.parts).is_file():
                    raise ValueError(f"Remote checkpoint file was not restored: {relative}.")
            remote_states = self._read_folder(folder)
            for index, remote_state in remote_states.items():
                self._states[index] = (
                    self._merge_states(self._states[index], remote_state)
                    if index in self._states else remote_state
                )
            for index, state in self._states.items():
                self._validate_state(state, index)
                self._save_file(self._question_path(index), state)
                self._dirty_files.add(self._question_path(index).name)
        print(f"Restored evaluation checkpoints from Hugging Face dataset {self.repo_id}@{revision[:12]}.")

    def _question_path(self, index: int) -> Path:
        return self.checkpoint_dir / f"question-{index:08d}.json"

    def load_question(self, index: int) -> Dict[str, Any]:
        """Return a mutable saved state, or an empty state for one benchmark row."""
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("benchmark_index must be a non-negative integer.")
        if index not in self._states:
            self._states[index] = {
                "benchmark_index": index, "generations": {}, "judgments": {},
                "retrieved": None, "phase_1_runs": [], "phase_2_runs": [], "completed": False,
            }
        return self._states[index]

    def save_question(self, state: Dict[str, Any]) -> None:
        """Atomically save progress and synchronously upload at the chosen interval."""
        self._validate_state(state)
        index = state["benchmark_index"]
        self._save_file(self._question_path(index), state)
        self._dirty_files.add(self._question_path(index).name)
        self._states[index] = state
        self._pending_operations += 1
        if self.upload_enabled and self._pending_operations >= self.upload_every:
            self.sync()

    def wrap_generator(self, generator: Callable[..., Dict[str, Any]], state: Dict[str, Any]):
        """Reuse successful outputs, including progress within an unfinished tree."""
        self._validate_state(state)

        def generate(prompt: str, **kwargs: Any) -> Dict[str, Any]:
            key = canonical_fingerprint({"prompt": prompt, "kwargs": kwargs})
            if key in state["generations"]:
                self.cache_hits += 1
                return copy.deepcopy(state["generations"][key])
            result = generator(prompt, **kwargs)
            if isinstance(result, Mapping) and result.get("status") == "SUCCESS":
                state["generations"][key] = copy.deepcopy(dict(result))
                self.save_question(state)
            return result

        return generate

    def complete_question(self, state: Dict[str, Any], p1, p2) -> None:
        """Commit a completed row's reports and make the checkpoint durable online."""
        state["phase_1_runs"] = copy.deepcopy(list(p1))
        state["phase_2_runs"] = copy.deepcopy(list(p2))
        state["completed"] = True
        self.save_question(state)
        self.sync()

    def sync(self) -> Any:
        """Upload only checkpoint JSON to a dataset repo; failures propagate."""
        if not self.upload_enabled:
            return None
        if not self._dirty_files:
            return None
        result = self.api.upload_folder(
            repo_id=self.repo_id, repo_type="dataset", folder_path=str(self.checkpoint_dir),
            path_in_repo=self.remote_prefix,
            allow_patterns=sorted(self._dirty_files | {"manifest.json"}),
            commit_message=f"Checkpoint merging evaluation {self.identity['benchmark']}",
        )
        self._pending_operations = 0
        self._dirty_files.clear()
        print(f"Evaluation checkpoint uploaded to Hugging Face dataset {self.repo_id}.")
        return result
