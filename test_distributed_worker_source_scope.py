"""Offline regression checks for the authenticated worker source boundary."""

from pathlib import Path
import subprocess
import tempfile
import unittest

from src.distributed_code_compatibility import (
    _committed_python,
    _legacy_source_hash,
    diagnose_worker_code_compatibility,
)


class WorkerSourceScopeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.write("src/pipeline.py", "def solve():\n    return 1\n")
        self.write("config.py", "SETTING = 1\n")
        self.write("adaptive_analogical_training.py", "def train():\n    return 1\n")
        self.write("reports/audit.py", "AUDIT = 1\n")

    def write(self, relative, source):
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source, encoding="utf-8")

    def save(self):
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
             "commit", "-qm", "original"],
            cwd=self.root, check=True,
        )
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.root, text=True,
        ).strip()
        sources = _committed_python(self.root, revision)
        self.saved = f"git:{revision}:source:{_legacy_source_hash(sources)}"

    def verdict(self):
        return diagnose_worker_code_compatibility(self.root, self.saved)

    def assert_rejected_path(self, relative):
        compatible, reason = self.verdict()
        self.assertFalse(compatible, reason)
        self.assertIn(relative, reason)

    def test_unimported_training_report_and_new_tool_changes_are_safe(self):
        self.save()
        self.write("adaptive_analogical_training.py", "def train():\n    return 2\n")
        (self.root / "reports/audit.py").unlink()
        self.write("benchmark_new.py", "this is not valid Python\n")
        self.assertTrue(self.verdict()[0], self.verdict()[1])

    def test_unreferenced_production_module_remains_checked(self):
        self.save()
        self.write("src/pipeline.py", "def solve():\n    return 2\n")
        self.assert_rejected_path("src/pipeline.py")

    def test_config_changes_remain_checked(self):
        self.save()
        self.write("config.py", "SETTING = 2\n")
        self.assert_rejected_path("config.py")

    def test_imported_root_helpers_are_checked_for_edits_and_deletion(self):
        self.write("src/pipeline.py", "import root_helper\n")
        self.write("root_helper.py", "RESULT = 1\n")
        self.save()
        self.write("root_helper.py", "RESULT = 2\n")
        self.assert_rejected_path("root_helper.py")
        (self.root / "root_helper.py").unlink()
        self.assert_rejected_path("root_helper.py")

    def test_new_local_module_cannot_shadow_an_external_import(self):
        self.write("src/pipeline.py", "import external_provider\n")
        self.save()
        self.write("external_provider.py", "RESULT = 1\n")
        self.assert_rejected_path("external_provider.py")

    def test_package_initializers_and_relative_transitive_helpers_are_checked(self):
        self.write("src/pipeline.py", "from helpers import child\n")
        self.write("helpers/__init__.py", "from . import child\n")
        self.write("helpers/child.py", "from .nested import RESULT\n")
        self.write("helpers/nested.py", "RESULT = 1\n")
        self.save()
        self.write("helpers/nested.py", "RESULT = 2\n")
        self.assert_rejected_path("helpers/nested.py")
        self.write("helpers/nested.py", "RESULT = 1\n")
        self.write("helpers/__init__.py", "from . import child\nVALUE = 2\n")
        self.assert_rejected_path("helpers/__init__.py")

    def test_relative_import_parent_packages_are_checked(self):
        self.write("src/pipeline.py", "from helpers.nested.child import RESULT\n")
        self.write("helpers/__init__.py", "VALUE = 1\n")
        self.write("helpers/value.py", "RESULT = 1\n")
        self.write("helpers/nested/__init__.py", "VALUE = 1\n")
        self.write("helpers/nested/child.py", "from ..value import RESULT\n")
        self.save()
        self.write("helpers/value.py", "RESULT = 2\n")
        self.assert_rejected_path("helpers/value.py")

    def test_test_named_helper_imported_by_production_is_checked(self):
        self.write("src/pipeline.py", "from test_helper import RESULT\n")
        self.write("test_helper.py", "RESULT = 1\n")
        self.save()
        self.write("test_helper.py", "RESULT = 2\n")
        self.assert_rejected_path("test_helper.py")

    def test_optional_finetuning_is_checked_if_imported(self):
        self.write("src/merging_finetuning.py", "RESULT = 1\n")
        self.save()
        self.write("src/merging_finetuning.py", "RESULT = 2\n")
        self.assertTrue(self.verdict()[0], self.verdict()[1])
        self.write("src/pipeline.py", "from src.merging_finetuning import RESULT\n")
        self.assert_rejected_path("src/merging_finetuning.py")

    def test_malformed_runtime_source_is_rejected(self):
        self.save()
        self.write("src/pipeline.py", "def broken(:\n")
        compatible, reason = self.verdict()
        self.assertFalse(compatible)
        self.assertIn("parsed", reason)

    def test_dynamic_imports_fail_closed_even_when_source_is_unchanged(self):
        for loader in (
            "import importlib\nloader = importlib.import_module\n",
            "from importlib import import_module as load\n",
            "loader = __import__\n",
            "exec('import hidden_helper')\n",
            "loader = exec\n",
            "import sys as runtime\nruntime.path.insert(0, 'tools')\nimport helper\n",
            "from sys import path as search\nsearch.insert(0, 'tools')\n",
            "__path__.append('external_plugins')\n",
        ):
            with self.subTest(loader=loader):
                self.write("src/pipeline.py", loader)
                self.save()
                compatible, reason = self.verdict()
                self.assertFalse(compatible)
                self.assertIn("Dynamic", reason)

    def test_ambiguous_module_and_package_resolution_fails_closed(self):
        self.write("src/pipeline.py", "import helpers\n")
        self.write("helpers.py", "RESULT = 1\n")
        self.write("helpers/__init__.py", "RESULT = 1\n")
        self.save()
        compatible, reason = self.verdict()
        self.assertFalse(compatible)
        self.assertIn("Ambiguous", reason)

    def test_original_full_source_hash_still_requires_authentication(self):
        self.save()
        self.saved = self.saved[:-1] + ("0" if self.saved[-1] != "0" else "1")
        compatible, reason = self.verdict()
        self.assertFalse(compatible)
        self.assertIn("saved source hash", reason)


if __name__ == "__main__":
    unittest.main()
