"""Prove that a distributed worker's execution code is unchanged.

The run manifest hashes every Python file and Git HEAD, so even edits to the
finalizer, tests, or the separate fine-tuning tools would block an unfinished
worker.  A saved Git revision lets us reconstruct and authenticate that
original source before comparing only the code that can affect worker
execution: ``config.py``, the ``run_experiments`` entry module, and every
local module they import, directly or transitively.  If the revision or
source hash is missing, the proof fails closed.

When worker code really did change (for example a bug fix landed between two
Kaggle sessions), ``DISTRIBUTED_ALLOW_WORKER_CODE_CHANGE`` lets the operator
resume anyway.  The decision is recorded as provenance in the worker status
and in every run log the session produces; the immutable manifest itself is
never rewritten.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from pathlib import Path
import re
import subprocess
from typing import Any, MutableMapping


_LEGACY_FINGERPRINT = re.compile(r"^git:([0-9a-f]{40}):source:([0-9a-f]{64})$")
_IGNORED_MODULES = {
    "src/distributed_code_compatibility.py",
}
# The distributed worker is started from ``run_experiments`` in this module
# with the settings from ``config.py``.  Everything the worker can execute is
# reachable from these two files through static imports, so they are the roots
# of the compatibility proof.  A snapshot without the entry module cannot be
# narrowed and falls back to treating every production module as a root.
_WORKER_ENTRY_MODULE = "src/orchestration.py"
_WORKER_CONFIG_MODULE = "config.py"
_OPTIONAL_WORKER_MODULES = {
    # This module is used by the separate fine-tuning notebooks.  The
    # distributed experiment worker does not import it.  It is nevertheless
    # checked if a worker module starts importing it.
    "src/merging_finetuning.py",
}
ALLOW_WORKER_CODE_CHANGE_KEY = "DISTRIBUTED_ALLOW_WORKER_CODE_CHANGE"
WORKER_CODE_CHANGE_STATE_KEY = "_DISTRIBUTED_WORKER_CODE_CHANGE"
_DYNAMIC_IMPORT_MODULES = {"importlib", "runpy", "pkgutil", "imp"}
_IMPORT_SEARCH_STATE = {"path", "modules", "meta_path", "path_hooks", "path_importer_cache"}
_INFRASTRUCTURE_FUNCTIONS = {
    "src/orchestration.py": {
        "finalize_distributed_experiments",
        "_auto_pin_local_legacy_code_fingerprint",
        "_prepare_distributed_worker",
        "_worker_code_resumable",
    },
    "src/hf_sync.py": {
        "ensure_distributed_manifest",
        "initialize_workspace",
        "_initialize_distributed_workspace",
        "_worker_code_resumable",
    },
    "src/distributed_execution.py": {
        "validate_manifest_compatibility",
        "record_worker_code_provenance",
        "write_worker_status",
    },
}


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", *args],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def _committed_python(root: Path, revision: str) -> dict[str, bytes]:
    paths = _git(root, "ls-tree", "-r", "--name-only", "-z", revision)
    sources = {}
    for raw_path in paths.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8")
        if relative.endswith(".py") and not any(
            part in {".git", "__pycache__", ".ipynb_checkpoints"}
            for part in Path(relative).parts
        ):
            sources[relative] = _git(root, "show", f"{revision}:{relative}")
    return sources


def _working_python(root: Path) -> dict[str, bytes]:
    sources = {}
    for path in root.rglob("*.py"):
        if any(part in {".git", "__pycache__", ".ipynb_checkpoints"} for part in path.parts):
            continue
        sources[path.relative_to(root).as_posix()] = path.read_bytes()
    return sources


def _legacy_source_hash(sources: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(sources, key=Path):
        source = sources[relative]
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source)
        digest.update(b"\0")
    return digest.hexdigest()


def _worker_source_tree(relative: str, source: bytes) -> ast.Module | None:
    if relative in _IGNORED_MODULES:
        return None
    tree = ast.parse(source, filename=relative)
    excluded = _INFRASTRUCTURE_FUNCTIONS.get(relative, set())
    tree.body = [
        node for node in tree.body
        if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in excluded)
        and not (
            isinstance(node, ast.ImportFrom)
            and node.module == "src.distributed_code_compatibility"
        )
        and not _is_export_list(node)
    ]
    return tree


def _is_export_list(node: ast.AST) -> bool:
    """``__all__`` only names re-exports; it never changes worker behavior."""
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "__all__"
    )


def _worker_source_ast(relative: str, source: bytes) -> str | None:
    tree = _worker_source_tree(relative, source)
    return None if tree is None else ast.dump(tree, include_attributes=False)


def _local_import_paths(module: str, sources: dict[str, bytes]) -> set[str]:
    """Resolve local modules and package initializers conservatively."""
    if not module:
        return set()
    components = module.split(".")
    paths = set()
    for length in range(1, len(components) + 1):
        prefix = "/".join(components[:length])
        module_path = prefix + ".py"
        package_path = prefix + "/__init__.py"
        if module_path in sources and package_path in sources:
            raise ValueError(f"Ambiguous local module/package import: {'.'.join(components[:length])}.")
        if package_path in sources:
            paths.add(package_path)
        if module_path in sources:
            if length != len(components):
                raise ValueError(f"Local import traverses a non-package module: {module}.")
            paths.add(module_path)
    return paths


def _worker_source_roots(sources: dict[str, bytes]) -> list[str]:
    """Return the modules whose static import closure is the worker's code.

    With the worker entry module present, the roots are that module and
    ``config.py``.  Imports inside functions are still followed because the
    whole AST is walked, so optional workflows remain covered.  Fine-tuning,
    reporting, and notebook-only helpers under ``src`` that the worker never
    imports are therefore outside the proof, as are root-level training and
    benchmark tools, unless production code imports them.
    """
    if _WORKER_ENTRY_MODULE in sources:
        return [
            path for path in (_WORKER_CONFIG_MODULE, _WORKER_ENTRY_MODULE)
            if path in sources
        ]
    return [
        path for path in sources
        if (path.startswith("src/") or path == _WORKER_CONFIG_MODULE)
        and path not in _IGNORED_MODULES | _OPTIONAL_WORKER_MODULES
    ]


def _worker_source_paths(sources: dict[str, bytes]) -> set[str]:
    """Include the worker roots and every local import they can use.

    Reading imports from both source snapshots prevents a removed dependency
    from disappearing from the compatibility proof.
    """
    pending = _worker_source_roots(sources)
    if not pending:
        raise ValueError("No production worker modules or config.py exist in this source snapshot.")
    selected = set()
    while pending:
        relative = pending.pop()
        if relative in selected:
            continue
        selected.add(relative)
        tree = _worker_source_tree(relative, sources[relative])
        if tree is None:
            continue
        sys_aliases = {
            alias.asname or alias.name
            for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names if alias.name == "sys"
        }
        imports = set()
        for node in ast.walk(tree):
            # A static import graph cannot authenticate arbitrary runtime code
            # loading.  Refuse the bridge rather than silently omit a helper.
            if isinstance(node, ast.Name) and node.id in {"__import__", "exec", "eval", "__path__"}:
                raise ValueError(f"Dynamic imports cannot be proven in {relative}.")
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in sys_aliases
                and node.attr in _IMPORT_SEARCH_STATE
            ):
                raise ValueError(f"Dynamic import search state cannot be proven in {relative}.")
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                if any(module.split(".")[0] in _DYNAMIC_IMPORT_MODULES for module in modules):
                    raise ValueError(f"Dynamic import infrastructure cannot be proven in {relative}.")
                imports.update(modules)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = relative.split("/")[:-1]
                    if node.level > len(package):
                        raise ValueError(f"Relative import escapes its package in {relative}.")
                    base = package[:len(package) - node.level + 1]
                    if node.module:
                        base.extend(node.module.split("."))
                    module = ".".join(base)
                else:
                    module = node.module or ""
                if module.split(".")[0] in _DYNAMIC_IMPORT_MODULES:
                    raise ValueError(f"Dynamic import infrastructure cannot be proven in {relative}.")
                if module == "sys" and any(alias.name in _IMPORT_SEARCH_STATE for alias in node.names):
                    raise ValueError(f"Dynamic import search state cannot be proven in {relative}.")
                imports.add(module)
                # ``from package import child`` may load a child module.
                # Only resolve it if that child actually exists; a plain
                # module's imported attributes are not submodules.
                for alias in node.names:
                    child = (module + "." + alias.name).strip(".")
                    child_path = child.replace(".", "/")
                    if child_path + ".py" in sources or any(
                        path.startswith(child_path + "/") for path in sources
                    ):
                        imports.add(child)
        for module in imports:
            for path in _local_import_paths(module, sources):
                if path not in selected:
                    pending.append(path)
    return selected


def diagnose_worker_code_compatibility(
    project_root: str | Path, saved_fingerprint: str
) -> tuple[bool, str]:
    """Explain whether the saved Git source matches the active worker code."""
    match = _LEGACY_FINGERPRINT.fullmatch(str(saved_fingerprint))
    if match is None:
        return False, "The saved fingerprint has no Git revision and source hash."
    root = Path(project_root).resolve()
    try:
        original = _committed_python(root, match.group(1))
    except (OSError, UnicodeError, subprocess.SubprocessError):
        return False, "The saved Git revision is unavailable in this checkout."
    if _legacy_source_hash(original) != match.group(2):
        return False, "The saved source hash differs from the Git revision's Python files."
    try:
        current = _working_python(root)
        worker_paths = _worker_source_paths(original) | _worker_source_paths(current)
        original_worker = {
            path: _worker_source_ast(path, source)
            for path, source in original.items()
            if path in worker_paths
        }
        current_worker = {
            path: _worker_source_ast(path, source)
            for path, source in current.items()
            if path in worker_paths
        }
    except (OSError, UnicodeError, SyntaxError):
        return False, "A Python source file could not be read or parsed."
    except ValueError as exc:
        return False, str(exc)
    changed = sorted(
        path for path in original_worker.keys() | current_worker.keys()
        if original_worker.get(path) != current_worker.get(path)
        and (original_worker.get(path) is not None or current_worker.get(path) is not None)
    )
    if changed:
        shown = ", ".join(changed[:8])
        more = f" (and {len(changed) - 8} more)" if len(changed) > 8 else ""
        return False, f"Worker source changed in: {shown}{more}."
    return True, "The Git source and worker execution code match."


def worker_code_unchanged_since_manifest(
    project_root: str | Path, saved_fingerprint: str
) -> bool:
    """Preserve the boolean interface for callers needing only a verdict."""
    return diagnose_worker_code_compatibility(project_root, saved_fingerprint)[0]


def worker_code_change_allowed(config: Any) -> bool:
    """Return whether the operator explicitly accepted changed worker code."""
    try:
        return bool(config.get(ALLOW_WORKER_CODE_CHANGE_KEY, False))
    except AttributeError:
        return False


def worker_code_change_hint() -> str:
    """Describe the sanctioned ways to continue after a worker code change."""
    return (
        "To keep this run and resume with the current code, set "
        f"CONFIG[\"{ALLOW_WORKER_CODE_CHANGE_KEY}\"] = True before run_experiments; "
        "the change is recorded as provenance in worker status and run logs. "
        "To keep the exact original behavior instead, restore the original "
        "checkout in an isolated directory (see DISTRIBUTED_KAGGLE_RUN.md) and "
        "restart the kernel before importing project modules, or start a new "
        "DISTRIBUTED_RUN_ID."
    )


def accept_worker_code_change(
    config: MutableMapping[str, Any],
    *,
    saved_fingerprint: Any,
    runtime_fingerprint: Any,
    reason: str,
    source: str,
) -> bool:
    """Record an operator-authorized worker code change and return True.

    Returns False, without touching ``config``, when
    ``DISTRIBUTED_ALLOW_WORKER_CODE_CHANGE`` is not enabled.  Otherwise the
    decision is stored under ``_DISTRIBUTED_WORKER_CODE_CHANGE`` so the worker
    status and every run log of this session carry it.  The saved manifest
    fingerprint is still the run identity; only the runtime provenance differs.
    """
    if not worker_code_change_allowed(config):
        return False
    record = {
        "manifest_code_fingerprint": (
            None if saved_fingerprint is None else str(saved_fingerprint)
        ),
        "runtime_code_fingerprint": (
            None if runtime_fingerprint is None else str(runtime_fingerprint)
        ),
        "reason": str(reason),
        "authorized_by": ALLOW_WORKER_CODE_CHANGE_KEY,
        "checked_against": str(source),
    }
    previous = config.get(WORKER_CODE_CHANGE_STATE_KEY)
    if previous != record:
        config[WORKER_CODE_CHANGE_STATE_KEY] = record
        logging.getLogger(__name__).warning(
            "Resuming distributed run with CHANGED worker code because %s=True. "
            "%s Results from this session are produced by runtime code %s while "
            "the immutable manifest records %s; both are recorded as provenance.",
            ALLOW_WORKER_CODE_CHANGE_KEY,
            reason,
            runtime_fingerprint,
            saved_fingerprint,
        )
    return True
