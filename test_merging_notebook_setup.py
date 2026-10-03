"""Offline regression checks for pip replacing packages in a running kernel."""

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch


class MergingNotebookSetupTests(TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent
        notebook = json.loads((root / "merging_finetuning.ipynb").read_text(encoding="utf-8"))
        setup = "".join(notebook["cells"][1]["source"])
        # Execute the actual post-pip guard without installing anything or using a GPU.
        cls.guard = setup.split("%pip install -q -r requirements-merging-finetuning.txt\n", 1)[1]
        cls.expected = next(
            line.split("==", 1)[1].strip()
            for line in (root / "requirements-merging-finetuning.txt").read_text().splitlines()
            if line.startswith("transformers==")
        )
        cls.requirements = (root / "requirements-merging-finetuning.txt").read_text()

    def run_guard(self, loaded, installed=None):
        versions = {"transformers": self.expected, **(installed or {})}
        packages = ("transformers", "tokenizers", "peft", "accelerate", "datasets", "bitsandbytes", "huggingface_hub")
        # Isolate every checked package from real imports made by other tests.
        modules = {name: SimpleNamespace(__version__=loaded.get(name)) for name in packages}
        with patch.dict("sys.modules", modules), patch(
            "importlib.metadata.version", side_effect=lambda name: versions.get(name, "1.0")
        ), patch.object(Path, "read_text", return_value=self.requirements), redirect_stdout(StringIO()):
            exec(self.guard, {"Path": Path})

    def test_fresh_kernel_can_continue(self):
        with patch.dict("sys.modules"):
            import sys
            for name in ("transformers", "tokenizers", "peft", "accelerate", "datasets", "bitsandbytes", "huggingface_hub"):
                sys.modules.pop(name, None)
            self.run_guard({})

    def test_matching_imports_can_continue_without_restart(self):
        self.run_guard({"transformers": self.expected, "tokenizers": "1.0"})

    def test_stale_transformers_stops_before_tokenizer_import(self):
        with self.assertRaisesRegex(RuntimeError, "Restart.*transformers: loaded 4.57.1"):
            self.run_guard({"transformers": "4.57.1"})

    def test_stale_dependency_also_requires_restart(self):
        with self.assertRaisesRegex(RuntimeError, "Restart.*tokenizers"):
            self.run_guard({"transformers": self.expected, "tokenizers": "0.20"})

    def test_failed_install_reports_pip_failure(self):
        with self.assertRaisesRegex(RuntimeError, "Dependency installation failed"):
            self.run_guard({}, {"transformers": "4.57.1"})
