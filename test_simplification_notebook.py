"""Run the simplification notebook's real cells offline through restarts and failures."""

import functools
import hashlib
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from src.merging_evaluation_checkpoints import canonical_fingerprint, fingerprint_merging_adapter
from src.merging_finetuning import QLoRAConfig, normalize_question
from src.simplification_finetuning import (
    HELDOUT_POPULATION, SIMPLIFICATION_INSTRUCTION, SimplificationEvaluationCheckpoint,
    SimplificationTrainingCheckpoints, evaluate_question, evaluation_prompts, load_heldout_population,
    retryable_failures, solver_prompt, summarize_evaluation, training_identity,
    verify_adapter_training_source,
)
from src.training_checkpoints import describe_hub_progress, training_checkpoint_identity
from src.utils import save_json_atomic
from test_merging_evaluation_checkpoints import FakeDatasetHub
from test_training_checkpoints import FakeTrainingHub, make_checkpoint

NOTEBOOK = Path(__file__).resolve().parent / "simplification_finetuning.ipynb"
QUESTION = "Find x if 2x = 6."
PROXY = "Find x if x = 3."


def cell(marker):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    matches = ["".join(c["source"]) for c in notebook["cells"]
               if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(matches) == 1, f"{marker!r} matched {len(matches)} cells"
    return matches[0]


def external_record(index, question):
    return {"record_id": f"math500:{index}", "question": question, "ground_truth": "3",
            "benchmark_index": index, "label_kind": "unlabeled", "status": "EXTERNAL_BENCHMARK",
            "source_benchmark": "math500"}


class EvaluationLoopTests(TestCase):
    def setUp(self):
        self.generated, self.judged, self.judge_routes = [], [], []
        self.interrupt_at = None
        self.direct_status = "SUCCESS"
        self.judge_failures = 0

    def generate(self, prompt, *, use_adapter, seed, temperature, top_p, max_new_tokens):
        if self.interrupt_at == len(self.generated):
            self.interrupt_at = None
            raise RuntimeError("session interrupted during inference")
        self.generated.append((prompt, use_adapter))
        if prompt.startswith(SIMPLIFICATION_INSTRUCTION):
            question = prompt.split("Original question:\n", 1)[1]
            text = PROXY if use_adapter and question == QUESTION else question
            return {"status": "SUCCESS", "text": text, "input_tokens": 10, "output_tokens": 5}
        status = self.direct_status if prompt == solver_prompt(QUESTION) else "SUCCESS"
        # Distinct prompts get distinct answers, so identical-answer judgment reuse stays visible.
        answer = f"x = 3 [{hashlib.sha256(prompt.encode()).hexdigest()[:8]}]"
        return {"status": status, "text": answer, "input_tokens": 10, "output_tokens": 5}

    def judge(self, answer, ground_truth, evaluator, config):
        self.judge_routes.append(config.get("_TARGET_BENCHMARK_FOR_QUERY"))
        if self.judge_failures:
            self.judge_failures -= 1
            # Provider error details can hold objects that are not JSON.
            return {"is_correct": None, "status": "API_ERROR", "error_details": {"raw": object()}}
        self.judged.append((answer, ground_truth))
        return {"is_correct": True, "status": "SUCCESS", "error_details": None}

    def evaluate(self, work_dir, populations=("math500",), records=None, prepared=None, hub=None):
        records = records or [external_record(0, QUESTION)]

        def load_external(data, name, config):
            return ([dict(row) for row in records], {"benchmark": name, "accepted_questions_excluded": 0},
                    {**config, "_TARGET_BENCHMARK_FOR_QUERY": name})

        checkpoint_class = SimplificationEvaluationCheckpoint if hub is None else functools.partial(
            SimplificationEvaluationCheckpoint, api=hub, download_fn=hub.download)
        namespace = {
            "RUN_EVALUATION": True, "EVAL_POPULATIONS": list(populations), "MAX_EVAL_QUESTIONS": None,
            "WORK_DIR": Path(work_dir), "CONFIG": {}, "evaluator": object(), "generator": self.generate,
            "prepared": prepared or {"splits": {"train": [], "validation": [], "test": []}},
            "SEED": 42, "SIMPLIFIER_MAX_NEW_TOKENS": 64, "SOLVER_MAX_NEW_TOKENS": 64,
            "EVAL_PROGRESS": False, "EVAL_VERBOSE": True,
            "HELDOUT_POPULATION": HELDOUT_POPULATION, "load_heldout_population": load_heldout_population,
            "load_external_evaluation_benchmark": load_external,
            "SIMPLIFICATION_INSTRUCTION": SIMPLIFICATION_INSTRUCTION,
            "evaluate_question": evaluate_question, "summarize_evaluation": summarize_evaluation,
            "retryable_failures": retryable_failures, "save_json_atomic": save_json_atomic,
            "SimplificationEvaluationCheckpoint": checkpoint_class,
            "canonical_fingerprint": canonical_fingerprint,
            "evaluation_identity": {"protocol_version": 1, "adapter_sha256": "test"},
            "HF_TOKEN": "test-token" if hub else None, "HF_EVAL_DATASET_REPO_ID": None,
            "HF_EVAL_REMOTE_PREFIX": "simplification_evaluations",
            "HF_EVAL_UPLOAD_ENABLED": hub is not None, "HF_EVAL_RESTORE_ENABLED": hub is not None,
            "HF_EVAL_DATASET_PRIVATE": True, "HF_EVAL_UPLOAD_EVERY": 10,
        }
        with patch("src.evaluation.evaluate_single_answer_with_llm", side_effect=self.judge), \
                redirect_stdout(StringIO()):
            exec(cell("retryable_failures(case)"), namespace)
        folder = Path(work_dir) / "evaluations"
        return [json.loads((folder / name / "evaluation.json").read_text(encoding="utf-8"))
                for name in populations]

    def counts(self):
        return len(self.generated), len(self.judged)

    def test_interrupted_question_reuses_saved_generations_and_judgments(self):
        with TemporaryDirectory() as directory:
            self.interrupt_at = 3  # after both rewrites and the direct answer
            with self.assertRaisesRegex(RuntimeError, "during inference"):
                self.evaluate(directory)
            self.assertEqual(self.counts(), (3, 1))
            [[case]] = self.evaluate(directory)
            self.assertEqual(case["solver_status"], "COMPLETE")
            self.assertEqual(self.counts(), (5, 2), "Only the two missing solver calls and one judgment run")
            self.evaluate(directory)
            self.assertEqual(self.counts(), (5, 2), "A completed question is skipped")

    def test_failed_judgment_is_retried_without_regeneration(self):
        with TemporaryDirectory() as directory:
            self.judge_failures = 1
            [[case]] = self.evaluate(directory)
            self.assertEqual(case["direct"]["evaluation"]["status"], "API_ERROR")
            summary = json.loads((Path(directory) / "evaluations/math500/summary.json").read_text())
            self.assertEqual(summary["solver"]["arms"]["direct"]["unknown"], 1)
            self.assertEqual(self.counts(), (5, 1))
            [[case]] = self.evaluate(directory)
            self.assertEqual(case["solver_status"], "COMPLETE")
            self.assertEqual(self.counts(), (5, 2), "Only the failed judgment is repeated")

    def test_length_limited_answer_is_final_and_not_regenerated(self):
        with TemporaryDirectory() as directory:
            self.direct_status = "LENGTH_LIMIT"
            [[case]] = self.evaluate(directory)
            self.assertIsNone(case["direct"]["evaluation"])
            self.assertEqual(retryable_failures(case), [])
            before = self.counts()
            self.evaluate(directory)
            self.assertEqual(self.counts(), before)

    def test_new_session_restores_completed_questions_from_the_hub(self):
        hub = FakeDatasetHub()
        records = [external_record(0, QUESTION), external_record(1, "Copy this question?")]
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            [cases] = self.evaluate(first, records=records, hub=hub)
            self.assertEqual(hub.creates[0]["repo_id"], "researcher/simplification-qwen3-4b-evaluation")
            self.assertTrue(all(name.startswith("simplification_evaluations/math500/") for name in hub.files))
            before = self.counts()
            [restored] = self.evaluate(second, records=records, hub=hub)
            self.assertEqual(self.counts(), before, "A fresh working directory reuses Hub progress")
            self.assertEqual(restored, cases)

    def test_heldout_population_reports_labeled_copy_and_change_behavior(self):
        test_split = [
            {"record_id": "log:7", "question": QUESTION, "question_id": normalize_question(QUESTION),
             "label_kind": "simplify", "status": "SUCCESS", "ground_truth": "So \\boxed{3}"},
            {"record_id": "log:9", "question": "Copy me?", "question_id": "copy me?",
             "label_kind": "copy", "status": "SKIPPED_FAILSAFE", "ground_truth": None},
        ]
        prepared = {"splits": {"train": [], "validation": [], "test": test_split}}
        with TemporaryDirectory() as directory:
            [cases] = self.evaluate(directory, populations=(HELDOUT_POPULATION,), prepared=prepared)
            self.assertEqual([case["benchmark_index"] for case in cases], [7, 9])
            self.assertEqual(set(self.judge_routes), {"numina_hard"})
            summary = json.loads((Path(directory) / "evaluations/heldout/summary.json").read_text())
            self.assertEqual(summary["behavior"]["simplify"]["adapted"]["change_rate"], 1.0)
            self.assertEqual(summary["behavior"]["copy"]["adapted"]["exact_copy_rate"], 1.0)
            recovery = {"WORK_DIR": Path(directory), "EVAL_BENCHMARKS": [],
                        "EVAL_POPULATIONS": [HELDOUT_POPULATION]}
            with redirect_stdout(StringIO()):
                exec(cell("report_index ="), recovery)
            report = (Path(directory) / "evaluations/heldout/evaluation_report.txt").read_text()
            self.assertIn("Benchmark: heldout", report)
            self.assertIn("simplify", report)


class AdapterCellTests(TestCase):
    manifest = {"sources": [{"path": "log.json", "sha256": "digest", "rows": 4}],
                "instruction": SIMPLIFICATION_INSTRUCTION, "seed": 42, "split_ratios": [0.8, 0.1, 0.1]}

    def must_not_train(self, *args, **kwargs):
        raise AssertionError("Training must not run")

    def restore(self, work_dir, saved_source, expected_source):
        adapter = Path(work_dir) / "hub_adapter" / "revision-1"
        adapter.mkdir(parents=True)
        (adapter / "training_source.json").write_text(json.dumps(saved_source), encoding="utf-8")
        requests = []

        def download(path, token, repo_id, **kwargs):
            requests.append(kwargs)
            return {"repo_id": "researcher/simplification-qwen3-4b-qlora", "revision": "revision-1",
                    "adapter_path": str(adapter)}

        namespace = {
            "HF_REUSE_ADAPTER_IF_AVAILABLE": True, "download_adapter_from_hub": download,
            "WORK_DIR": Path(work_dir), "HF_TOKEN": "test-token", "HF_MODEL_REPO_ID": None,
            "verify_adapter_training_source": verify_adapter_training_source,
            "training_source": expected_source, "TRAIN": True,
            "load_qlora_model": self.must_not_train, "train_qlora": self.must_not_train,
        }
        with redirect_stdout(StringIO()):
            exec(cell("hub_adapter = None"), namespace)
            exec(cell("trainer = train_qlora("), namespace)
        return namespace, requests

    def test_matching_hub_adapter_skips_training(self):
        source = training_identity(self.manifest, QLoRAConfig())
        with TemporaryDirectory() as directory:
            namespace, requests = self.restore(directory, source, source)
            self.assertIsNone(namespace["model"])
            self.assertEqual(namespace["hub_adapter"]["revision"], "revision-1")
            self.assertEqual(requests[0]["default_repo_name"], "simplification-qwen3-4b-qlora")

    def test_hub_adapter_from_other_settings_stops_before_evaluation(self):
        with TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "different qlora"):
            self.restore(directory, training_identity(self.manifest, QLoRAConfig(epochs=1.0)),
                         training_identity(self.manifest, QLoRAConfig()))

    def test_new_adapter_carries_its_identity_into_the_simplification_repo(self):
        source = training_identity(self.manifest, QLoRAConfig())
        uploads = []
        with TemporaryDirectory() as directory:
            work_dir = Path(directory)

            def train(*args, **kwargs):
                (work_dir / "best_adapter").mkdir()
                return SimpleNamespace(state=SimpleNamespace(best_model_checkpoint="checkpoint-20"))

            namespace = {
                "TRAIN": True, "hub_adapter": None, "train_qlora": train, "model": None,
                "tokenizer": None, "tokenized": {}, "qlora_config": None, "resume_checkpoint": None,
                "training_state": SimpleNamespace(trainer_callback=lambda: "hub-checkpoint-callback"),
                "training_source": source, "save_json_atomic": save_json_atomic, "WORK_DIR": work_dir,
                "HF_UPLOAD_ENABLED": True, "HF_TOKEN": "test-token", "HF_MODEL_REPO_ID": None,
                "HF_MODEL_REPO_PRIVATE": True,
                "upload_adapter_to_hub": lambda *args, **kwargs: uploads.append(kwargs) or {"url": "hub"},
            }
            with redirect_stdout(StringIO()):
                exec(cell("trainer = train_qlora("), namespace)
            verify_adapter_training_source(work_dir / "best_adapter", source)
            self.assertEqual(uploads[0]["default_repo_name"], "simplification-qwen3-4b-qlora")


TOKENS = {"train": [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "attention_mask": [1, 1, 1]}],
          "validation": [{"input_ids": [4, 5], "labels": [-100, 5], "attention_mask": [1, 1]}]}


class TrainingRecoveryCellTests(TestCase):
    """The notebook's training stage survives a lost Kaggle session."""

    def status(self, work_dir, hub, **overrides):
        lines = []
        namespace = {
            "WORK_DIR": Path(work_dir), "TRAIN": True, "RUN_EVALUATION": True,
            "qlora_config": QLoRAConfig(output_dir=str(work_dir)), "prepared": {"tokenized": TOKENS},
            "SimplificationTrainingCheckpoints": functools.partial(
                SimplificationTrainingCheckpoints, api=hub, download_fn=hub.download, log=lines.append),
            "describe_hub_progress": functools.partial(describe_hub_progress, api=hub, log=lines.append),
            "training_checkpoint_identity": training_checkpoint_identity,
            "HF_TOKEN": "test-token", "hub_features": ["HF_TRAIN_UPLOAD_ENABLED"],
            "HF_TRAIN_UPLOAD_ENABLED": True, "HF_TRAIN_RESTORE_ENABLED": True,
            "HF_TRAIN_STATE_REPO_ID": None, "HF_TRAIN_STATE_PRIVATE": True,
            "HF_TRAIN_STATE_REMOTE_PREFIX": "simplification_training", "HF_TRAIN_UPLOAD_EVERY_SAVES": 1,
            "HF_MODEL_REPO_ID": None, "HF_REUSE_ADAPTER_IF_AVAILABLE": True,
            "HF_EVAL_DATASET_REPO_ID": None, "HF_EVAL_REMOTE_PREFIX": "simplification_evaluations",
            "EVAL_POPULATIONS": [HELDOUT_POPULATION, "math500"],
        }
        namespace.update(overrides)
        exec(cell("training_state = SimplificationTrainingCheckpoints("), namespace)
        return namespace, "\n".join(lines)

    def test_status_cell_uses_the_simplification_repositories_and_finds_earlier_progress(self):
        hub = FakeTrainingHub()
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            namespace, text = self.status(first, hub)
            self.assertIn("researcher/simplification-qwen3-4b-training-state", text)
            self.assertIn("researcher/simplification-qwen3-4b-evaluation", text)
            self.assertEqual(namespace["training_state"].identity["workflow"], "simplification")
            store = namespace["training_state"]
            store.restore()
            store.on_checkpoint_saved(make_checkpoint(first, 60), "checkpoint-60")
            self.assertEqual(hub.creates[0]["repo_id"], "researcher/simplification-qwen3-4b-training-state")
            _, text = self.status(second, hub)
            self.assertIn("checkpoint-60 (step 60 of 100) matches this configuration -> training resumes", text)
            self.assertIn("heldout: nothing saved yet", text)

    def test_changed_settings_never_resume_the_old_checkpoint(self):
        hub = FakeTrainingHub()
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            namespace, _ = self.status(first, hub)
            namespace["training_state"].restore()
            namespace["training_state"].on_checkpoint_saved(make_checkpoint(first, 60), "checkpoint-60")
            namespace, text = self.status(second, hub, qlora_config=QLoRAConfig(output_dir=str(second), lora_rank=8))
            self.assertIn("differs in: qlora.lora_rank", text)
            self.assertIsNone(namespace["training_state"].restore())

    def restore_cell(self, work_dir, **overrides):
        calls = []
        identity = training_checkpoint_identity(QLoRAConfig(), TOKENS, workflow="simplification")

        def load_qlora_model(config):
            calls.append("load_model")
            return SimpleNamespace(get_nb_trainable_parameters=lambda: (1, 2)), SimpleNamespace()

        namespace = {
            "HF_REUSE_ADAPTER_IF_AVAILABLE": False, "TRAIN": True, "RESUME_CHECKPOINT": None,
            "WORK_DIR": Path(work_dir), "HF_TOKEN": "t", "HF_MODEL_REPO_ID": None,
            "training_state": SimpleNamespace(identity=identity,
                                              restore=lambda: calls.append("restore") or "hub/checkpoint-60"),
            "load_qlora_model": load_qlora_model, "qlora_config": QLoRAConfig(),
            "tokenize_splits": lambda splits, tokenizer, length: (TOKENS, {}),
            "prepared": {"splits": {}}, "MAX_LENGTH": 4096,
            "training_checkpoint_identity": training_checkpoint_identity,
            "CompletionOnlyCollator": lambda tokenizer: "collator",
            "smoke_test_training_step": lambda *a: calls.append("smoke") or 1.5,
            "download_adapter_from_hub": lambda *a, **k: None,
            "verify_adapter_training_source": verify_adapter_training_source,
            "training_source": {},
        }
        namespace.update(overrides)
        with redirect_stdout(StringIO()):
            exec(cell("hub_adapter = None"), namespace)
        return namespace, calls

    def test_training_resumes_from_the_hub_before_the_model_loads(self):
        with TemporaryDirectory() as directory:
            namespace, calls = self.restore_cell(directory)
            self.assertEqual(namespace["resume_checkpoint"], "hub/checkpoint-60")
            self.assertEqual(calls, ["restore", "load_model", "smoke"])

    def test_manual_resume_checkpoint_overrides_the_hub(self):
        with TemporaryDirectory() as directory:
            namespace, calls = self.restore_cell(directory, RESUME_CHECKPOINT="/local/checkpoint-3")
            self.assertEqual(namespace["resume_checkpoint"], "/local/checkpoint-3")
            self.assertNotIn("restore", calls)

    def test_tokens_that_differ_from_the_fingerprinted_ones_stop_before_training(self):
        changed = {**TOKENS, "train": [{"input_ids": [9], "labels": [9], "attention_mask": [1]}]}
        with TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, "differ from those fingerprinted"):
            self.restore_cell(directory, tokenize_splits=lambda *a: (changed, {}))

    def test_train_cell_resumes_and_uploads_every_checkpoint(self):
        captured = {}
        with TemporaryDirectory() as directory:
            work_dir = Path(directory)

            def train(model, tokenizer, tokenized, config, **kwargs):
                captured.update(kwargs)
                (work_dir / "best_adapter").mkdir()
                return SimpleNamespace(state=SimpleNamespace(best_model_checkpoint="checkpoint-60"))

            namespace = {
                "TRAIN": True, "hub_adapter": None, "train_qlora": train, "model": None, "tokenizer": None,
                "tokenized": {}, "qlora_config": None, "resume_checkpoint": "hub/checkpoint-60",
                "training_state": SimpleNamespace(trainer_callback=lambda: "hub-callback"),
                "training_source": {}, "save_json_atomic": save_json_atomic, "WORK_DIR": work_dir,
                "HF_UPLOAD_ENABLED": False,
            }
            with redirect_stdout(StringIO()):
                exec(cell("trainer = train_qlora("), namespace)
            self.assertEqual(captured, {"resume_from_checkpoint": "hub/checkpoint-60",
                                        "callbacks": ["hub-callback"]})


class EvaluationIdentityTests(TestCase):
    def test_identity_tracks_adapter_and_judge_settings_without_credentials(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()), \
                patch("src.api_manager.AvalAIAPIManager") as manager:
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "adapter_model.safetensors").write_bytes(b"weights v1")
            namespace = {
                "RUN_EVALUATION": True, "setup_kaggle_mode": lambda *args: None, "CONFIG": {},
                "EVAL_PARSE_BOXED_GROUND_TRUTH": True, "EVALUATOR_TEMPERATURE": 0.0,
                "EVALUATOR_MAX_TOKENS": 100, "AVALAI_BASE_URL": "https://judge.invalid/v1",
                "AVALAI_EVALUATOR_MODEL": "judge-model", "AVALAI_MODEL_QUOTAS": {"default": {"rpm": 30}},
                "AVALAI_REQUEST_TIMEOUT_SECONDS": 60.0, "AVALAI_CALL_DELAY_SECONDS": 5.0,
                "optional_secret": lambda name: "secret-test-value",
                "model": SimpleNamespace(config=SimpleNamespace(_commit_hash="base-revision")),
                "adapter_path": str(adapter), "fingerprint_merging_adapter": fingerprint_merging_adapter,
                "prepared": {"manifest": {"sources": [{"sha256": "run-log-digest"}]}},
                "evaluation_prompts": evaluation_prompts, "canonical_fingerprint": canonical_fingerprint,
                "BASE_MODEL_NAME": "base-model", "SEED": 42, "MAX_LENGTH": 4096,
                "SIMPLIFIER_MAX_NEW_TOKENS": 512, "SOLVER_MAX_NEW_TOKENS": 1024,
            }
            source = cell("evaluation_identity = {")
            exec(source, namespace)
            identity = namespace["evaluation_identity"]
            original = canonical_fingerprint(identity)
            self.assertNotIn("secret-test-value", json.dumps(identity))
            self.assertEqual(manager.call_args.kwargs["api_key_or_list"], ["secret-test-value"])
            self.assertEqual(identity["evaluator"]["AVALAI_MODEL_NAME_EVALUATOR"], "judge-model")
            self.assertEqual(namespace["CONFIG"]["GLOBAL_API_CALL_DELAY_SECONDS"]["avalai"], 5.0)
            self.assertEqual(identity["prompts"], evaluation_prompts())

            (adapter / "adapter_model.safetensors").write_bytes(b"weights v2")
            exec(source, namespace)
            self.assertNotEqual(canonical_fingerprint(namespace["evaluation_identity"]), original)
            (adapter / "adapter_model.safetensors").write_bytes(b"weights v1")
            namespace["CONFIG"]["AVALAI_REASONING_EFFORT_EVALUATOR"] = "high"
            exec(source, namespace)
            self.assertNotEqual(canonical_fingerprint(namespace["evaluation_identity"]), original)
