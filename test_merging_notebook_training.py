"""Run the merging notebook's real training-recovery cells offline."""

import functools
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from src.merging_finetuning import QLoRAConfig
from src.training_checkpoints import (
    HubTrainingCheckpoints, describe_hub_progress, merging_training_source,
    training_checkpoint_identity, verify_adapter_training_source,
)
from src.utils import save_json_atomic
from test_training_checkpoints import FakeTrainingHub, make_checkpoint

NOTEBOOK = Path(__file__).resolve().parent / "merging_finetuning.ipynb"
TOKENS = {"train": [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "attention_mask": [1, 1, 1]}],
          "validation": [{"input_ids": [4, 5], "labels": [-100, 5], "attention_mask": [1, 1]}]}
MANIFEST = {"sources": [{"sha256": "source-digest"}], "split_question_ids": {"train": ["q1"]},
            "seed": 42, "split_ratios": [0.8, 0.1, 0.1]}


def cell(marker):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    matches = ["".join(c["source"]) for c in notebook["cells"]
               if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(matches) == 1, f"{marker!r} matched {len(matches)} cells"
    return matches[0]


class StatusCellTests(TestCase):
    def run_cell(self, work_dir, hub, **overrides):
        lines = []
        namespace = {
            "WORK_DIR": Path(work_dir), "TRAIN": True,
            "qlora_config": QLoRAConfig(output_dir=str(work_dir)),
            "prepared": {"tokenized": TOKENS, "manifest": MANIFEST},
            "HubTrainingCheckpoints": functools.partial(
                HubTrainingCheckpoints, api=hub, download_fn=hub.download, log=lines.append),
            "describe_hub_progress": functools.partial(describe_hub_progress, api=hub, log=lines.append),
            "training_checkpoint_identity": training_checkpoint_identity,
            "merging_training_source": merging_training_source,
            "HF_TOKEN": "test-token", "HF_TRAIN_UPLOAD_ENABLED": True, "HF_TRAIN_RESTORE_ENABLED": True,
            "HF_TRAIN_STATE_REPO_ID": None, "HF_TRAIN_STATE_PRIVATE": True,
            "HF_TRAIN_STATE_REMOTE_PREFIX": "merging_training", "HF_TRAIN_UPLOAD_EVERY_SAVES": 1,
            "HF_REUSE_ADAPTER_IF_AVAILABLE": True, "HF_EVAL_RESTORE_ENABLED": True,
            "HF_MODEL_REPO_ID": None, "HF_EVAL_DATASET_REPO_ID": None,
            "HF_EVAL_REMOTE_PREFIX": "merging_evaluations", "EVAL_BENCHMARKS": ["math500", "gsm8k"],
        }
        namespace.update(overrides)
        exec(cell("training_state = HubTrainingCheckpoints("), namespace)
        return namespace, "\n".join(lines)

    def test_new_session_reports_every_stage_and_names_the_default_repositories(self):
        hub = FakeTrainingHub(exists=False)
        with TemporaryDirectory() as directory:
            namespace, text = self.run_cell(directory, hub)
            state = namespace["training_state"]
            self.assertEqual(state.identity, training_checkpoint_identity(
                QLoRAConfig(output_dir=str(directory)), TOKENS, workflow="merging"))
            self.assertEqual(namespace["training_source"], merging_training_source(MANIFEST, QLoRAConfig()))
            for expected in ("researcher/merging-qwen3-4b-training-state",
                             "no repository yet -> training starts from the beginning",
                             "researcher/merging-qwen3-4b-qlora: not found -> training runs",
                             "researcher/merging-qwen3-4b-evaluation: no repository yet"):
                self.assertIn(expected, text)

    def test_a_checkpoint_from_an_earlier_session_is_announced_before_any_model_loads(self):
        hub = FakeTrainingHub()
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            namespace, _ = self.run_cell(first, hub)
            store = namespace["training_state"]
            store.restore()
            store.on_checkpoint_saved(make_checkpoint(first, 40), "checkpoint-40")
            _, text = self.run_cell(second, hub)
            self.assertIn("checkpoint-40 (step 40 of 100) matches this configuration -> training resumes", text)

    def test_changed_settings_do_not_see_the_old_checkpoint(self):
        hub = FakeTrainingHub()
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            namespace, _ = self.run_cell(first, hub)
            store = namespace["training_state"]
            store.restore()
            store.on_checkpoint_saved(make_checkpoint(first, 40), "checkpoint-40")
            _, text = self.run_cell(second, hub, qlora_config=QLoRAConfig(output_dir=str(second), epochs=1.0))
            self.assertIn("no checkpoint for this configuration", text)
            self.assertIn("is not reused; differs in: qlora.epochs", text)

    def test_evaluation_only_runs_skip_training_recovery(self):
        hub = FakeTrainingHub(exists=False)
        with TemporaryDirectory() as directory:
            namespace, text = self.run_cell(directory, hub, TRAIN=False)
            self.assertNotIn("training-state", text)
            self.assertFalse(namespace["training_state"].upload_enabled)
            self.assertFalse(namespace["training_state"].restore_enabled)

    def test_nothing_is_contacted_when_every_hub_feature_is_off(self):
        hub = FakeTrainingHub()
        with TemporaryDirectory() as directory:
            def forbidden(*args, **kwargs):
                raise AssertionError("the Hub must not be contacted")
            self.run_cell(directory, hub, describe_hub_progress=forbidden, HF_TRAIN_UPLOAD_ENABLED=False,
                          HF_TRAIN_RESTORE_ENABLED=False, HF_REUSE_ADAPTER_IF_AVAILABLE=False,
                          HF_EVAL_RESTORE_ENABLED=False)


class RestoreAndTrainCellTests(TestCase):
    source = merging_training_source(MANIFEST, QLoRAConfig())

    def namespace(self, work_dir, **overrides):
        calls = []
        identity = training_checkpoint_identity(QLoRAConfig(), TOKENS, workflow="merging")

        def load_qlora_model(config):
            calls.append("load_model")
            return SimpleNamespace(get_nb_trainable_parameters=lambda: (1, 2)), SimpleNamespace()

        state = SimpleNamespace(identity=identity, restore=lambda: calls.append("restore") or "hub/checkpoint-40")
        namespace = {
            "HF_REUSE_ADAPTER_IF_AVAILABLE": False, "download_adapter_from_hub": lambda *a, **k: None,
            "verify_adapter_training_source": verify_adapter_training_source,
            "training_source": self.source, "TRAIN": True, "RESUME_CHECKPOINT": None,
            "training_state": state, "WORK_DIR": Path(work_dir), "HF_TOKEN": "t", "HF_MODEL_REPO_ID": None,
            "load_qlora_model": load_qlora_model, "qlora_config": QLoRAConfig(),
            "tokenize_splits": lambda splits, tokenizer, length: (TOKENS, {}),
            "prepared": {"splits": {}}, "TRAIN_MAX_LENGTH": 3072,
            "training_checkpoint_identity": training_checkpoint_identity,
            "CompletionOnlyCollator": lambda tokenizer: "collator",
            "smoke_test_training_step": lambda *a: calls.append("smoke") or 1.5,
        }
        namespace.update(overrides)
        return namespace, calls

    def restore_cell(self, namespace):
        with redirect_stdout(StringIO()) as output:
            exec(cell("hub_adapter = None\nif HF_REUSE_ADAPTER_IF_AVAILABLE:"), namespace)
        return output.getvalue()

    def test_training_resumes_from_the_hub_checkpoint_before_the_model_loads(self):
        with TemporaryDirectory() as directory:
            namespace, calls = self.namespace(directory)
            self.restore_cell(namespace)
            self.assertEqual(namespace["resume_checkpoint"], "hub/checkpoint-40")
            self.assertEqual(calls, ["restore", "load_model", "smoke"])

    def test_a_manual_resume_checkpoint_overrides_the_hub(self):
        with TemporaryDirectory() as directory:
            namespace, calls = self.namespace(directory, RESUME_CHECKPOINT="/local/checkpoint-7")
            self.restore_cell(namespace)
            self.assertEqual(namespace["resume_checkpoint"], "/local/checkpoint-7")
            self.assertNotIn("restore", calls)

    def test_tokens_that_differ_from_the_fingerprinted_ones_stop_before_training(self):
        changed = {**TOKENS, "train": [{"input_ids": [9], "labels": [9], "attention_mask": [1]}]}
        with TemporaryDirectory() as directory:
            namespace, _ = self.namespace(directory, tokenize_splits=lambda *a: (changed, {}))
            with self.assertRaisesRegex(RuntimeError, "differ from those fingerprinted"):
                self.restore_cell(namespace)

    def adapter(self, directory, saved):
        folder = Path(directory) / "hub_adapter" / "rev"
        folder.mkdir(parents=True)
        if saved is not None:
            (folder / "training_source.json").write_text(json.dumps(saved), encoding="utf-8")
        return {"repo_id": "researcher/adapter", "revision": "rev", "adapter_path": str(folder)}

    def test_a_matching_hub_adapter_skips_training_entirely(self):
        with TemporaryDirectory() as directory:
            namespace, calls = self.namespace(
                directory, HF_REUSE_ADAPTER_IF_AVAILABLE=True,
                download_adapter_from_hub=lambda *a, **k: self.adapter(directory, self.source))
            self.restore_cell(namespace)
            self.assertIsNone(namespace["model"])
            self.assertIsNone(namespace["resume_checkpoint"])
            self.assertEqual(calls, [])

    def test_an_adapter_from_other_settings_stops_the_notebook(self):
        other = merging_training_source(MANIFEST, QLoRAConfig(epochs=1.0))
        with TemporaryDirectory() as directory:
            namespace, _ = self.namespace(
                directory, HF_REUSE_ADAPTER_IF_AVAILABLE=True,
                download_adapter_from_hub=lambda *a, **k: self.adapter(directory, other))
            with self.assertRaisesRegex(ValueError, "different qlora"):
                self.restore_cell(namespace)

    def test_an_adapter_uploaded_before_identities_existed_is_accepted_with_a_warning(self):
        with TemporaryDirectory() as directory:
            namespace, _ = self.namespace(
                directory, HF_REUSE_ADAPTER_IF_AVAILABLE=True,
                download_adapter_from_hub=lambda *a, **k: self.adapter(directory, None))
            output = self.restore_cell(namespace)
            self.assertIn("cannot be verified", output)
            self.assertEqual(namespace["hub_adapter"]["revision"], "rev")

    def test_train_cell_passes_resume_checkpoint_and_upload_callback_then_records_identity(self):
        captured, uploads = {}, []
        with TemporaryDirectory() as directory:
            work_dir = Path(directory)

            def train(model, tokenizer, tokenized, config, **kwargs):
                captured.update(kwargs)
                (work_dir / "best_adapter").mkdir()
                return SimpleNamespace(state=SimpleNamespace(best_model_checkpoint="checkpoint-40"))

            namespace = {
                "TRAIN": True, "hub_adapter": None, "train_qlora": train, "model": object(),
                "tokenizer": object(), "tokenized": TOKENS, "qlora_config": QLoRAConfig(),
                "resume_checkpoint": "hub/checkpoint-40",
                "training_state": SimpleNamespace(trainer_callback=lambda: "hub-callback"),
                "training_source": self.source, "save_json_atomic": save_json_atomic, "WORK_DIR": work_dir,
                "HF_UPLOAD_ENABLED": True, "HF_TOKEN": "t", "HF_MODEL_REPO_ID": None,
                "HF_MODEL_REPO_PRIVATE": True, "HF_UPLOAD_COMMIT_MESSAGE": "Upload",
                "upload_adapter_to_hub": lambda *a, **k: uploads.append((a, k)) or {"url": "hub"},
            }
            with redirect_stdout(StringIO()):
                exec(cell("trainer = train_qlora("), namespace)
            self.assertEqual(captured, {"resume_from_checkpoint": "hub/checkpoint-40",
                                        "callbacks": ["hub-callback"]})
            verify_adapter_training_source(work_dir / "best_adapter", self.source)
            self.assertEqual(len(uploads), 1)

    def test_reload_cell_verifies_the_adapter_before_loading_it(self):
        loaded = []
        with TemporaryDirectory() as directory:
            namespace = {
                "model": None, "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
                "hub_adapter": {"adapter_path": str(Path(directory))}, "WORK_DIR": Path(directory),
                "verify_adapter_training_source": verify_adapter_training_source,
                "training_source": self.source, "qlora_config": QLoRAConfig(),
                "load_adapter_for_inference": lambda *a, **k: loaded.append(a) or ("model", "tokenizer"),
                "LocalFusionGenerator": lambda *a, **k: "generator", "MAX_LENGTH": 4096,
            }
            (Path(directory) / "training_source.json").write_text(
                json.dumps(merging_training_source(MANIFEST, QLoRAConfig(epochs=1.0))), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different qlora"):
                exec(cell("adapter_path = hub_adapter['adapter_path']"), namespace)
            self.assertEqual(loaded, [])
            (Path(directory) / "training_source.json").write_text(json.dumps(self.source), encoding="utf-8")
            exec(cell("adapter_path = hub_adapter['adapter_path']"), namespace)
            self.assertEqual(len(loaded), 1)
