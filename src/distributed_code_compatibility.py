"""Prove that a legacy distributed worker's execution code is unchanged.

The old manifest hashed every Python file and Git HEAD, so even edits to the
finalizer or tests block an unfinished worker.  A saved Git revision lets us
reconstruct and authenticate that original source before comparing the code
that can affect worker execution.  If the revision or source hash is missing,
the proof fails closed.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re
import subprocess


_LEGACY_FINGERPRINT = re.compile(r"^git:([0-9a-f]{40}):source:([0-9a-f]{64})$")
_IGNORED_MODULE = "src/distributed_code_compatibility.py"
_INFRASTRUCTURE_FUNCTIONS = {
    "src/orchestration.py": {
        "finalize_distributed_experiments",
        "_auto_pin_local_legacy_code_fingerprint",
    },
    "src/hf_sync.py": {"ensure_distributed_manifest"},
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


def _worker_source_ast(relative: str, source: bytes) -> str | None:
    if relative == _IGNORED_MODULE or Path(relative).name.startswith("test_"):
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
    ]
    return ast.dump(tree, include_attributes=False)


def worker_code_unchanged_since_manifest(
    project_root: str | Path, saved_fingerprint: str
) -> bool:
    """Accept only a Git-authenticated legacy source with unchanged worker AST."""
    match = _LEGACY_FINGERPRINT.fullmatch(str(saved_fingerprint))
    if match is None:
        return False
    root = Path(project_root).resolve()
    try:
        original = _committed_python(root, match.group(1))
        if _legacy_source_hash(original) != match.group(2):
            return False
        current = _working_python(root)
        original_worker = {
            path: _worker_source_ast(path, source)
            for path, source in original.items()
        }
        current_worker = {
            path: _worker_source_ast(path, source)
            for path, source in current.items()
        }
        return {
            path: value for path, value in original_worker.items() if value is not None
        } == {
            path: value for path, value in current_worker.items() if value is not None
        }
    except (OSError, UnicodeError, SyntaxError, subprocess.SubprocessError):
        return False
