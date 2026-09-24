import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.merging_finetuning import tokenize_splits
from src.prompts import create_core_simp_zero_shot_prompt
from src.simplification_finetuning import (
    copy_metrics,
    evaluate_question,
    load_simplification_data,
    prepare_simplification_data,
    recover_original_question,
    summarize_evaluation,
)


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


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if add_generation_prompt:
            return [10, 20]
        return [10, 20, 30, 31]


class SimplificationDataTests(unittest.TestCase):
    def test_targets_exclusions_and_donor_cross_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            donors = [{
                "status": "SUCCESS", "original_index": 0,
                "original_question": "Hard question", "proxy_question": "Easy question",
                "ground_truth": "42",
            }]
            logs = donors + [
                {"status": "REJECTED_BY_FILTER", "original_index": 1,
                 "original_question": "Reject question", "proxy_question": "Bad proxy",
                 "ground_truth": "5"},
                failsafe("Keep  all  spaces?", 2),
                {"status": "SKIPPED_PERFECT_BASELINE", "original_index": 3,
                 "original_question": "Already easy"},
                {"status": "FAILURE", "original_index": 4},
            ]
            save(root / "donors.json", donors)
            save(root / "logs.json", logs)
            loaded = load_simplification_data(root / "donors.json", root / "logs.json")
            by_status = {row["status"]: row for row in loaded["records"]}
            self.assertEqual(by_status["SUCCESS"]["label"], "Easy question")
            self.assertEqual(by_status["REJECTED_BY_FILTER"]["label"], "Reject question")
            self.assertEqual(by_status["SKIPPED_FAILSAFE"]["label"], "Keep  all  spaces?")
            self.assertIsNone(by_status["SKIPPED_FAILSAFE"]["ground_truth"])
            self.assertEqual(len(loaded["records"]), 3)
            self.assertIn("excluded_status:FAILURE", [a["reason"] for a in loaded["audit"]])

    def test_missing_or_ambiguous_prompt_is_not_guessed(self):
        self.assertIsNone(recover_original_question({"status": "SKIPPED_FAILSAFE", "trace": []}))
        prompt = create_core_simp_zero_shot_prompt("Q")
        row = {"trace": [{"sub_step": "generate_proxy", "input_context": {
            "prompt": prompt + "\nOriginal Question:\nsecond"
        }}]}
        self.assertIsNone(recover_original_question(row))
        self.assertIsNone(recover_original_question({
            "original_question": "A", "target_query_text": "B"
        }))

    def test_conflicts_duplicates_and_donor_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            donors = [{"status": "SUCCESS", "original_index": 0,
                       "original_question": "Q", "proxy_question": "P"}]
            logs = [donors[0], {"status": "REJECTED_BY_FILTER", "original_question": "Q"},
                    {"status": "REJECTED_BY_FILTER", "original_question": "Unique"},
                    {"status": "REJECTED_BY_FILTER", "original_question": "Unique"},
                    {"status": "SUCCESS", "original_index": 5,
                     "original_question": "Missing donor", "proxy_question": "P"}]
            save(root / "donors.json", donors)
            save(root / "logs.json", logs)
            loaded = load_simplification_data(root / "donors.json", root / "logs.json")
            self.assertEqual([r["question"] for r in loaded["records"]], ["Unique"])
            reasons = [a["reason"] for a in loaded["audit"]]
            self.assertIn("conflicting_outcomes", reasons)
            self.assertIn("duplicate_question", reasons)
            self.assertIn("donor_mismatch", reasons)

    def test_grouped_split_and_completion_only_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = [
                {"status": "REJECTED_BY_FILTER", "original_index": i,
                 "original_question": f"Question {i}"} for i in range(10)
            ]
            save(root / "donors.json", [])
            save(root / "logs.json", logs)
            prepared = prepare_simplification_data(
                root / "donors.json", root / "logs.json", root / "output", seed=9
            )
            groups = [
                {r["question_id"] for r in rows} for rows in prepared["splits"].values()
            ]
            self.assertFalse(groups[0] & groups[1])
            self.assertFalse(groups[0] & groups[2])
            self.assertFalse(groups[1] & groups[2])
            tokenized, _ = tokenize_splits(prepared["splits"], FakeTokenizer(), 100)
            self.assertEqual(tokenized["train"][0]["labels"], [-100, -100, 30, 31])
            self.assertTrue((root / "output" / "simplification_data_manifest.json").is_file())

    def test_generation_failure_inside_rejected_log_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "donors.json", [])
            save(root / "logs.json", [{
                "status": "REJECTED_BY_FILTER", "original_question": "Q",
                "trace": [{"sub_step": "baseline_solve_attempt_1",
                           "output_result": {"status": "FAILURE"}}],
            }])
            loaded = load_simplification_data(root / "donors.json", root / "logs.json")
            self.assertEqual(loaded["records"], [])
            self.assertIn("generation_failure_in_trace", [a["reason"] for a in loaded["audit"]])


class EvaluationTests(unittest.TestCase):
    def test_exact_copy_is_distinct_from_normalized_copy(self):
        record = {"question": "What is 2 + 2?"}
        metric = copy_metrics(record, {"status": "SUCCESS", "text": "what  is 2 + 2?"})
        self.assertFalse(metric["exact_copy"])
        self.assertTrue(metric["normalized_copy"])

    def test_copy_reuses_one_direct_solution_and_one_judgment(self):
        calls = []

        def generator(prompt, *, use_adapter, seed, temperature, top_p, max_new_tokens):
            calls.append((prompt, use_adapter))
            if "Original question:\n" in prompt:
                return {"status": "SUCCESS", "text": "What is 2 + 2?"}
            return {"status": "SUCCESS", "text": "The answer is 4."}

        record = {"record_id": "log:0", "question": "What is 2 + 2?",
                  "label_kind": "copy", "status": "REJECTED_BY_FILTER",
                  "ground_truth": "4"}
        with patch("src.evaluation.evaluate_single_answer_with_llm") as judge:
            judge.return_value = {"status": "SUCCESS", "is_correct": True}
            result = evaluate_question(record, generator, object(), {})
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(not used for prompt, used in calls if "Original question:\n" not in prompt))
        self.assertEqual(judge.call_count, 1)
        self.assertEqual(result["arms"]["base"]["solver_status"], "REUSED_DIRECT")
        self.assertEqual(result["arms"]["adapted"]["solver_status"], "REUSED_DIRECT")
        summary = summarize_evaluation([result])
        self.assertEqual(summary["solver"]["paired_judged"], 1)
        self.assertEqual(summary["solver"]["accuracy"]["adapted"], 1.0)

    def test_changed_proxy_uses_base_solver_for_proxy_and_original(self):
        calls = []

        def generator(prompt, *, use_adapter, seed, temperature, top_p, max_new_tokens):
            calls.append((prompt, use_adapter))
            if "Original question:\n" in prompt:
                return {"status": "SUCCESS", "text": "Easier Q" if use_adapter else "Original Q"}
            return {"status": "SUCCESS", "text": "Solved."}

        record = {"record_id": "log:0", "question": "Original Q",
                  "label_kind": "simplify", "status": "SUCCESS", "ground_truth": "42"}
        with patch("src.evaluation.evaluate_single_answer_with_llm") as judge:
            judge.return_value = {"status": "SUCCESS", "is_correct": True}
            result = evaluate_question(record, generator, object(), {})
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(not adapted for prompt, adapted in calls if "Original question:\n" not in prompt))
        self.assertEqual(judge.call_count, 2)
        self.assertEqual(result["arms"]["adapted"]["solver_status"], "EVALUATED")
        self.assertEqual(result["arms"]["base"]["solver_status"], "REUSED_DIRECT")

    def test_no_ground_truth_keeps_behavior_without_solver_calls(self):
        calls = []

        def generator(prompt, *, use_adapter, seed, temperature, top_p, max_new_tokens):
            calls.append((prompt, use_adapter))
            return {"status": "SUCCESS", "text": "Q"}

        record = {"record_id": "log:2", "question": "Q", "label_kind": "copy",
                  "status": "SKIPPED_FAILSAFE", "ground_truth": None}
        result = evaluate_question(record, generator, None, {})
        self.assertEqual(result["solver_status"], "NO_GROUND_TRUTH")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("direct", result)


if __name__ == "__main__":
    unittest.main()
