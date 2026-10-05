"""Create an isolated, source-authenticated checkout for a saved distributed run.

Run this command in a separate process before importing project modules in the
notebook. It never changes the active checkout, run manifest, or checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from src.distributed_code_compatibility import _committed_python, _legacy_source_hash
from src.distributed_execution import (
    DistributedExecutionError, resolve_code_fingerprint, validate_manifest_integrity,
)


_SAVED_SOURCE = re.compile(r"git:([0-9a-f]{40}):source:([0-9a-f]{64})\Z")


def prepare_resume_checkout(
    project_root: str | Path,
    manifest: Mapping[str, Any],
    destination: str | Path,
    *,
    expected_run_id: str,
) -> Path:
    """Authenticate the saved source before creating a detached Git worktree.

    Missing history or uncommitted source in the original run fails closed. An
    existing destination is accepted only when its Git root and full source
    fingerprint match exactly; existing files are never removed or reset.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("The authoritative manifest must be a JSON object.")
    validate_manifest_integrity(manifest)
    if manifest["run_id"] != expected_run_id:
        raise ValueError("The manifest belongs to a different DISTRIBUTED_RUN_ID.")
    saved = str(manifest["code_fingerprint"])
    match = _SAVED_SOURCE.fullmatch(saved)
    if match is None:
        raise ValueError("Recovery requires a saved Git revision and full source hash.")
    root = Path(project_root).expanduser().resolve(strict=True)
    revision, source_hash = match.groups()
    try:
        original = _committed_python(root, revision)
    except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
        raise ValueError(
            f"Saved revision {revision} is unavailable. Fetch the repository history "
            "and retry; no checkout was created."
        ) from exc
    if _legacy_source_hash(original) != source_hash:
        raise ValueError(
            "The saved source hash does not match its Git revision. The original "
            "run may have used uncommitted Python files; restoring HEAD alone is unsafe."
        )

    requested = Path(destination).expanduser().absolute()
    if requested.is_symlink():
        raise ValueError("The recovery destination cannot be a symlink.")
    target = requested.resolve()
    if target == root or target.is_relative_to(root) or root.is_relative_to(target):
        raise ValueError("Use a separate destination outside the active project checkout.")

    def git(*args: str, cwd: Path = root) -> str:
        return subprocess.run(
            ["git", "-c", f"safe.directory={cwd.as_posix()}",
             "-c", "core.autocrlf=false", "-c", "core.eol=lf", *args],
            cwd=cwd, check=True, capture_output=True, text=True, timeout=60,
        ).stdout.strip()

    if target.exists():
        if not target.is_dir():
            raise ValueError("The recovery destination already exists and is not a directory.")
        try:
            actual_root = Path(git("rev-parse", "--show-toplevel", cwd=target)).resolve()
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("The recovery destination already exists; it will not be overwritten.") from exc
        if actual_root != target or resolve_code_fingerprint(target) != saved:
            raise ValueError("The existing destination does not match the saved source; it will not be overwritten.")
        return target

    git("worktree", "add", "--detach", str(target), revision)
    if resolve_code_fingerprint(target) != saved:
        raise ValueError(
            f"The new checkout at {target} failed full source verification. "
            "Do not run it; it has been left in place for inspection."
        )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Downloaded authoritative manifest.json")
    parser.add_argument("--run-id", required=True, help="Expected DISTRIBUTED_RUN_ID")
    parser.add_argument("--destination", type=Path, required=True, help="New, separate source checkout")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        checkout = prepare_resume_checkout(
            args.project_root, manifest, args.destination, expected_run_id=args.run_id,
        )
    except (OSError, ValueError, KeyError, TypeError, DistributedExecutionError,
            subprocess.SubprocessError) as exc:
        parser.exit(1, f"Recovery stopped: {exc}\n")
    print(f"Verified original source: {checkout}")
    print(f"Saved revision: {manifest['code_fingerprint'].split(':')[1]}")
    print("Use a fresh kernel and select this directory before any config/src imports.")
    print("Use the saved notebook/setup, identical scientific inputs/settings, and your original worker ID.")


if __name__ == "__main__":
    main()
