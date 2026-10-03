"""Offline regression checks for question duplicates in evaluation populations."""

from unittest import TestCase

from src.benchmark_data import load_target_benchmarks
from src.merging_finetuning import build_evaluation_populations, normalize_question


def record(question, record_id):
    return {"question": question, "question_id": normalize_question(question), "record_id": record_id}


class EvaluationPopulationTests(TestCase):
    def setUp(self):
        self.prepared = {"splits": {"train": [], "validation": [], "test": []}}

    def test_distinct_numina_indices_can_repeat_question_text(self):
        questions, answers, _ = load_target_benchmarks(
            {"TARGET_BENCHMARK": "numina_hard", "HARD_QUESTIONS_INDICES_PATH": "indices.json"},
            {"questions": ["Find x.", "Other?", " FIND  x. "], "solutions": ["x=1", "2", "x=1"]},
            load_json_fn=lambda path: [0, 1, 2],
        )
        audit = {}
        populations = build_evaluation_populations(self.prepared, questions, answers, audit=audit)
        rows = populations["remaining_benchmark"]
        self.assertEqual([row["benchmark_index"] for row in rows], [0, 1])
        self.assertEqual(rows[0]["benchmark_indices"], [0, 2])
        self.assertEqual(audit["duplicate_benchmark_rows_removed"], 1)
        self.assertTrue(audit["duplicate_benchmark_rows"][0]["reference_matches_canonical"])

    def test_reference_variants_keep_first_and_are_audited(self):
        audit = {}
        rows = build_evaluation_populations(
            self.prepared, ["Q?", "q?", " Q? "], ["First proof", "Another proof", "First  proof"], audit=audit,
        )["remaining_benchmark"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ground_truth"], "First proof")
        self.assertEqual(rows[0]["benchmark_indices"], [0, 1, 2])
        self.assertEqual(audit["duplicate_rows_with_different_reference_text"], 1)
        self.assertEqual(audit["duplicate_benchmark_rows"][0]["ground_truth"], "Another proof")

    def test_every_accepted_split_excludes_all_occurrences_from_remainder(self):
        self.prepared["splits"] = {
            "train": [record("Train?", "train")],
            "validation": [record("Validate?", "validation")],
            "test": [record("Test?", "test")],
        }
        populations = build_evaluation_populations(
            self.prepared,
            ["Train?", "Validate?", "Test?", "train?", "validate?", "test?", "New?", "new?"],
            ["answer"] * 8,
        )
        self.assertEqual([row["question"] for row in populations["remaining_benchmark"]], ["New?"])
        self.assertEqual(populations["heldout_accepted"][0]["benchmark_indices"], [2, 5])

    def test_multiple_heldout_records_evaluate_once_per_question(self):
        self.prepared["splits"]["test"] = [record("Q?", "first"), record("q?", "second")]
        audit = {}
        populations = build_evaluation_populations(self.prepared, ["Q?"], ["answer"], audit=audit)
        self.assertEqual(len(populations["heldout_accepted"]), 1)
        self.assertEqual(populations["heldout_accepted"][0]["record_id"], "first")
        self.assertEqual(populations["remaining_benchmark"], [])
        self.assertEqual(audit["duplicate_heldout_records_removed"], 1)

    def test_unique_rows_preserve_input_order_and_indices(self):
        rows = build_evaluation_populations(self.prepared, ["B?", "A?"], ["b", "a"])["remaining_benchmark"]
        self.assertEqual([(row["question"], row["ground_truth"], row["benchmark_index"]) for row in rows],
                         [("B?", "b", 0), ("A?", "a", 1)])

    def test_misaligned_benchmark_still_fails(self):
        with self.assertRaisesRegex(ValueError, "must align"):
            build_evaluation_populations(self.prepared, ["Q?"], [])

    def test_missing_heldout_question_still_fails(self):
        self.prepared["splits"]["test"] = [record("Missing?", "missing")]
        with self.assertRaisesRegex(ValueError, "absent from benchmark"):
            build_evaluation_populations(self.prepared, ["Q?"], ["answer"])
