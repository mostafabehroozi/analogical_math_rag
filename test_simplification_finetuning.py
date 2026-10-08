"""Offline checks for simplification labels, evaluation retries, and adapter identity."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.merging_evaluation_checkpoints import MergingEvaluationCheckpoint
from src.merging_finetuning import QLoRAConfig, download_adapter_from_hub, upload_adapter_to_hub
from src.prompts import create_core_simp_zero_shot_prompt
from src.simplification_finetuning import (
    HELDOUT_POPULATION,
    SimplificationEvaluationCheckpoint,
    copy_metrics,
    evaluate_question,
    load_heldout_population,
    load_simplification_data,
    prepare_simplification_data,
    recover_original_question,
    retryable_failures,
    summarize_evaluation,
    training_identity,
    verify_adapter_training_source,
)
from test_merging_evaluation_checkpoints import FakeDatasetHub
from test_merging_hub_recovery import FakeHub


def save(path, rows):
    path.write_text(json.dumps(rows), encoding="utf-8")


def failsafe(question, index):
    return {
        "status": "SKIPPED_FAILSAFE", "original_index": index,
        "trace": [{
            "sub_step": "generate_proxy",
            "input_context": {"prompt": create_core_simp_zero_shot_prompt(question)},
        }],
    }


class SimplificationDataTests(unittest.TestCase):
    def test_targets_and_exclusions_from_run_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = [
                {"status": "SUCCESS", "original_index": 0, "original_question": "Hard question",
                 "proxy_question": "Easy question", "ground_truth": "42"},
                {"status": "REJECTED_BY_FILTER", "original_index": 1,
                 "original_question": "Reject question", "proxy_question": "Bad proxy",
                 "ground_truth": "5"},
                failsafe("Keep  all  spaces?", 2),
                {"status": "SKIPPED_PERFECT_BASELINE", "original_index": 3,
                 "original_question": "Already easy"},
                {"status": "FAILURE", "original_index": 4},
            ]
            save(root / "logs.json", logs)
            loaded = load_simplification_data(root / "logs.json")
            by_status = {row["status"]: row for row in loaded["records"]}
            self.assertEqual(by_status["SUCCESS"]["label"], "Easy question")
            self.assertEqual(by_status["REJECTED_BY_FILTER"]["label"], "Reject question")
            self.assertEqual(by_status["SKIPPED_FAILSAFE"]["label"], "Keep  all  spaces?")
            self.assertIsNone(by_status["SKIPPED_FAILSAFE"]["ground_truth"])
            self.assertEqual(len(loaded["records"]), 3)
            reasons = [a["reason"] for a in loaded["audit"]]
            self.assertIn("excluded_status:FAILURE", reasons)
            self.assertIn("excluded_status:SKIPPED_PERFECT_BASELINE", reasons)

    def test_missing_or_ambiguous_prompt_is_not_guessed(self):
        self.assertIsNone(recover_original_question({"status": "SKIPPED_FAILSAFE", "trace": []}))
        prompt = create_core_simp_zero_shot_prompt("Q")
        row = {"trace": [{"sub_step": "generate_proxy", "input_context": {
            "prompt": prompt + "\nOriginal Question:\nsecond"
        }}]}
        self.assertIsNone(recover_original_question(row))
        self.assertIsNone(recover_original_question({"original_question": "A", "target_query_text": "B"}))

    def test_success_takes_priority_and_only_one_label_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logs.json"
            save(path, [
                {"status": "SUCCESS", "original_question": "Q", "proxy_question": "P",
                 "base_score": 0.2, "augmented_score": 0.4},
                {"status": "REJECTED_BY_FILTER", "original_question": "Q"},
                {"status": "SUCCESS", "original_question": "Q", "proxy_question": "Better P",
                 "base_score": 0.2, "augmented_score": 0.8},
                {"status": "REJECTED_BY_FILTER", "original_question": "Unique"},
                {"status": "REJECTED_BY_FILTER", "original_question": "Unique"},
                {"status": "SUCCESS", "original_question": "Missing proxy", "proxy_question": ""},
            ])
            loaded = load_simplification_data(path)
            self.assertEqual([r["question"] for r in loaded["records"]], ["Q", "Unique"])
            self.assertEqual(loaded["records"][0]["label"], "Better P")
            reasons = [a["reason"] for a in loaded["audit"]]
            for reason in ("superseded_by_success", "duplicate_question", "missing_proxy"):
                self.assertIn(reason, reasons)

    def test_generation_failure_inside_rejected_log_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logs.json"
            save(path, [{
                "status": "REJECTED_BY_FILTER", "original_question": "Q",
                "trace": [{"sub_step": "baseline_solve_attempt_1", "output_result": {"status": "FAILURE"}}],
            }])
            loaded = load_simplification_data(path)
            self.assertEqual(loaded["records"], [])
            self.assertIn("generation_failure_in_trace", [a["reason"] for a in loaded["audit"]])

    def test_grouped_split_keeps_questions_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "logs.json", [
                {"status": "REJECTED_BY_FILTER", "original_index": i, "original_question": f"Question {i}"}
                for i in range(10)
            ])
            prepared = prepare_simplification_data(root / "logs.json", root / "output", seed=9)
            groups = [{r["question_id"] for r in rows} for rows in prepared["splits"].values()]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            self.assertTrue((root / "output" / "simplification_data_manifest.json").is_file())


class EvaluationTests(unittest.TestCase):
    def test_exact_copy_is_distinct_from_normalized_copy(self):
        metric = copy_metrics({"question": "What is 2 + 2?"}, {"status": "SUCCESS", "text": "what  is 2 + 2?"})
        self.assertFalse(metric["exact_copy"])
        self.assertTrue(metric["normalized_copy"])

    def test_changed_proxy_uses_base_solver_and_injected_judge(self):
        calls, judged = [], []

        def generator(prompt, *, use_adapter, seed, temperature, top_p, max_new_tokens):
            calls.append((prompt, use_adapter))
            if "Original question:\n" in prompt:
                return {"status": "SUCCESS", "text": "Easier Q" if use_adapter else "Original Q"}
            return {"status": "SUCCESS", "text": "Solved."}

        def judge(answer, ground_truth):
            judged.append((answer, ground_truth))
            return {"status": "SUCCESS", "is_correct": True}

        record = {"record_id": "log:0", "question": "Original Q", "label_kind": "simplify",
                  "status": "SUCCESS", "ground_truth": "42"}
        with patch("src.evaluation.evaluate_single_answer_with_llm") as default_judge:
            result = evaluate_question(record, generator, object(), {}, judge=judge)
        default_judge.assert_not_called()
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(not adapted for prompt, adapted in calls if "Original question:\n" not in prompt))
        self.assertEqual(judged, [("Solved.", "42"), ("Solved.", "42")])
        self.assertEqual(result["arms"]["adapted"]["solver_status"], "EVALUATED")
        self.assertEqual(result["arms"]["base"]["solver_status"], "REUSED_DIRECT")
        self.assertEqual(retryable_failures(result), [])

    def test_copy_reuses_one_direct_solution_and_one_default_judgment(self):
        def generator(prompt, **kwargs):
            text = "What is 2 + 2?" if "Original question:\n" in prompt else "The answer is 4."
            return {"status": "SUCCESS", "text": text}

        record = {"record_id": "log:0", "question": "What is 2 + 2?", "label_kind": "copy",
                  "status": "REJECTED_BY_FILTER", "ground_truth": "4"}
        with patch("src.evaluation.evaluate_single_answer_with_llm") as judge:
            judge.return_value = {"status": "SUCCESS", "is_correct": True}
            result = evaluate_question(record, generator, object(), {})
        self.assertEqual(judge.call_count, 1)
        self.assertEqual(summarize_evaluation([result])["solver"]["accuracy"]["adapted"], 1.0)

    def test_no_ground_truth_keeps_behavior_without_solver_calls(self):
        calls = []

        def generator(prompt, **kwargs):
            calls.append(prompt)
            return {"status": "SUCCESS", "text": "Q"}

        record = {"record_id": "log:2", "question": "Q", "label_kind": "copy",
                  "status": "SKIPPED_FAILSAFE", "ground_truth": None}
        result = evaluate_question(record, generator, None, {})
        self.assertEqual(result["solver_status"], "NO_GROUND_TRUTH")
        self.assertEqual(len(calls), 2)
        self.assertEqual(retryable_failures(result), [])

    def test_only_crashes_and_failed_judgments_are_retried(self):
        judged = {"status": "SUCCESS", "is_correct": False}
        case = {
            "direct": {"solution": {"status": "LENGTH_LIMIT", "text": "partial"}, "evaluation": None},
            "arms": {
                "base": {"simplification": {"status": "EMPTY", "text": ""}},
                "adapted": {"simplification": {"status": "SUCCESS", "text": "P"},
                            "proxy_solution": {"status": "SUCCESS", "text": "S"},
                            "original_solution": {"status": "CONTEXT_OVERFLOW", "text": ""}},
            },
        }
        self.assertEqual(retryable_failures(case), [], "Greedy decoding repeats these outcomes")
        case["arms"]["base"]["simplification"] = {"status": "GENERATION_FAILED", "text": ""}
        case["arms"]["adapted"]["original_solution"] = {"status": "SUCCESS", "text": "A"}
        case["arms"]["adapted"]["evaluation"] = {"status": "API_ERROR", "is_correct": None}
        self.assertEqual(retryable_failures(case), [
            "base.simplification: GENERATION_FAILED", "adapted.evaluation: API_ERROR",
        ])
        case["arms"]["base"]["simplification"] = {"status": "SUCCESS", "text": "Q"}
        case["arms"]["adapted"]["evaluation"] = judged
        self.assertEqual(retryable_failures(case), [])


class HeldoutPopulationTests(unittest.TestCase):
    def test_test_split_keeps_labels_log_rows_and_numina_judging(self):
        prepared = {"splits": {
            "train": [{"record_id": "log:1", "question": "Train?", "label_kind": "copy"}],
            "validation": [],
            "test": [
                {"record_id": "log:7", "question": "Hard?", "label_kind": "simplify", "ground_truth": "\\boxed{3}"},
                {"record_id": "log:9", "question": "Copy?", "label_kind": "copy", "ground_truth": None},
            ],
        }}
        config = {"TARGET_BENCHMARK": "math500", "_TARGET_BENCHMARK_FOR_QUERY": "math500"}
        records, audit, evaluator_config = load_heldout_population(prepared, config)
        self.assertEqual([(r["benchmark_index"], r["label_kind"]) for r in records], [(7, "simplify"), (9, "copy")])
        self.assertTrue(all(r["source_benchmark"] == HELDOUT_POPULATION for r in records))
        self.assertEqual(audit["label_counts"], {"simplify": 1, "copy": 1})
        self.assertEqual(audit["with_ground_truth"], 1)
        self.assertEqual(evaluator_config["_TARGET_BENCHMARK_FOR_QUERY"], "numina_hard")
        self.assertEqual(config["_TARGET_BENCHMARK_FOR_QUERY"], "math500", "Input config must not change")


class AdapterIdentityTests(unittest.TestCase):
    manifest = {"sources": [{"path": "/a/log.json", "sha256": "digest", "rows": 3}],
                "instruction": "Simplify.", "seed": 42, "split_ratios": [0.8, 0.1, 0.1]}

    def test_training_identity_ignores_paths_and_tracks_settings(self):
        first = training_identity(self.manifest, QLoRAConfig(output_dir="/kaggle/working/a"))
        moved = {**self.manifest, "sources": [{**self.manifest["sources"][0], "path": "/cache/other/log.json"}]}
        self.assertEqual(first, training_identity(moved, QLoRAConfig(output_dir="/elsewhere", gpu_index=1)))
        self.assertNotEqual(first, training_identity(self.manifest, QLoRAConfig(epochs=1.0)))
        self.assertNotIn("output_dir", json.dumps(first))

    def test_adapter_training_source_must_exist_and_match(self):
        expected = training_identity(self.manifest, QLoRAConfig())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "cannot be verified"):
                verify_adapter_training_source(directory, expected)
            (Path(directory) / "training_source.json").write_text(json.dumps(expected), encoding="utf-8")
            verify_adapter_training_source(directory, expected)
            changed = training_identity(self.manifest, QLoRAConfig(lora_rank=8))
            with self.assertRaisesRegex(ValueError, "different qlora.*HF_MODEL_REPO_ID"):
                verify_adapter_training_source(directory, changed)


class RepositoryNameTests(unittest.TestCase):
    def test_simplification_uses_its_own_adapter_and_evaluation_repositories(self):
        files = {"adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"}

        def download(**kwargs):
            for name in files:
                (Path(kwargs["local_dir"]) / name).write_text("test")

        created = []
        api = SimpleNamespace(
            whoami=lambda: {"name": "researcher"},
            create_repo=lambda **kwargs: created.append(kwargs["repo_id"]),
            upload_folder=lambda **kwargs: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            restored = download_adapter_from_hub(
                root / "hub", "token", api=FakeHub(files), download_fn=download,
                default_repo_name="simplification-qwen3-4b-qlora",
            )
            self.assertEqual(restored["repo_id"], "researcher/simplification-qwen3-4b-qlora")
            adapter = Path(restored["adapter_path"])
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            upload_adapter_to_hub(adapter, "token", api=api, default_repo_name="simplification-qwen3-4b-qlora")
            upload_adapter_to_hub(adapter, "token", api=api)
            self.assertEqual(created, ["researcher/simplification-qwen3-4b-qlora",
                                       "researcher/merging-qwen3-4b-qlora"])

            for checkpoint_class, repo, label in (
                (SimplificationEvaluationCheckpoint, "researcher/simplification-qwen3-4b-evaluation", "simplification"),
                (MergingEvaluationCheckpoint, "researcher/merging-qwen3-4b-evaluation", "merging"),
            ):
                hub = FakeDatasetHub()
                checkpoint_class(root / label, {"benchmark": "math500"}, token="token",
                                 upload_enabled=True, api=hub, download_fn=hub.download)
                self.assertEqual(hub.creates[0]["repo_id"], repo)
                self.assertEqual(hub.uploads[0]["commit_message"], f"Checkpoint {label} evaluation math500")


if __name__ == "__main__":
    unittest.main()
