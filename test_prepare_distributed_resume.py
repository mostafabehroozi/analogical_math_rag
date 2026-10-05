"""Offline source recovery checks; no Hub or provider calls."""

import copy
from pathlib import Path
import subprocess
import tempfile
import unittest

from prepare_distributed_resume import prepare_resume_checkout
from src.distributed_execution import (
    DistributedManifestMismatch, build_run_manifest, resolve_code_fingerprint,
)


class ResumeCheckoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "active"
        self.root.mkdir()
        (self.root / "config.py").write_text("CONFIG = {'value': 1}\n", encoding="utf-8")
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "original")
        self.original = resolve_code_fingerprint(self.root)
        self.manifest = build_run_manifest(
            {"DISTRIBUTED_RUN_ID": "saved-run", "DISTRIBUTED_WORKER_COUNT": 1},
            [{"experiment_name": "grouping"}], ["question"], ["answer"],
            code_fingerprint=self.original,
        )
        self.destination = self.root.parent / "resume"

    def git(self, *args):
        return subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=self.root,
                              check=True, capture_output=True, text=True).stdout.strip()

    def prepare(self, manifest=None, destination=None, run_id="saved-run"):
        return prepare_resume_checkout(self.root, manifest or self.manifest,
                                       destination or self.destination, expected_run_id=run_id)

    def test_restores_exact_source_without_resetting_active_checkout(self):
        active_source = "CONFIG = {'value': 2}\n"
        (self.root / "config.py").write_text(active_source, encoding="utf-8")
        self.assertEqual(self.prepare(), self.destination.resolve())
        self.assertEqual(resolve_code_fingerprint(self.destination), self.original)
        self.assertEqual((self.root / "config.py").read_text(), active_source)
        self.assertEqual(self.prepare(), self.destination.resolve())
        (self.destination / "config.py").write_text("CONFIG = {}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "will not be overwritten"):
            self.prepare()
        self.assertEqual((self.destination / "config.py").read_text(), "CONFIG = {}\n")

    def test_unauthenticated_or_wrong_manifest_never_creates_checkout(self):
        uncommitted = build_run_manifest(
            {"DISTRIBUTED_RUN_ID": "saved-run", "DISTRIBUTED_WORKER_COUNT": 1},
            [{"experiment_name": "grouping"}], ["question"], ["answer"],
            code_fingerprint=self.original[:-1] + ("0" if self.original[-1] != "0" else "1"),
        )
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            self.prepare(uncommitted)
        with self.assertRaisesRegex(ValueError, "different DISTRIBUTED_RUN_ID"):
            self.prepare(run_id="other-run")
        tampered = copy.deepcopy(self.manifest)
        tampered["worker_count"] = 2
        with self.assertRaises(DistributedManifestMismatch):
            self.prepare(tampered)
        self.assertFalse(self.destination.exists())

    def test_existing_and_nested_destinations_are_not_overwritten(self):
        self.destination.mkdir()
        sentinel = self.destination / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "will not be overwritten"):
            self.prepare()
        self.assertEqual(sentinel.read_text(), "keep")
        with self.assertRaisesRegex(ValueError, "separate destination"):
            self.prepare(destination=self.root / "resume")
        self.assertFalse((self.root / "resume").exists())


if __name__ == "__main__":
    unittest.main()
