"""Focused offline coverage for ordered benchmark target selection."""

import sys
import types
import unittest

from src.benchmark_data import (
    benchmark_name_for_target_index,
    load_target_benchmarks,
    selected_benchmark_names,
)


class CombinedBenchmarkSelectionTests(unittest.TestCase):
    def setUp(self):
        self.exemplar_data = {
            "questions": ["numina zero", "numina one", "numina two"],
            "solutions": ["solution zero", "solution one", "solution two"],
        }
        self.datasets = {
            "HuggingFaceH4/MATH-500": {
                "problem": ["m0", "m1"], "answer": ["a0", "a1"],
            },
            "openai/gsm8k": {
                "question": ["g0", "g1"], "answer": ["work #### 4", "why #### 5"],
            },
        }

    def fake_load_dataset(self, path, *args, **kwargs):
        return self.datasets[path]

    def test_combined_blocks_keep_configuration_order_and_ignore_cap(self):
        config = {
            "TARGET_BENCHMARK": "numina_hard",
            "TARGET_BENCHMARKS": [" Math500 ", "GSM8K"],
            "BENCHMARK_MAX_QUESTIONS": 1,
        }
        questions, answers, names = load_target_benchmarks(
            config, self.exemplar_data, load_json_fn=lambda _: [2],
            load_dataset_fn=self.fake_load_dataset,
        )
        self.assertEqual(questions, ["m0", "m1", "g0", "g1"])
        self.assertEqual(answers, ["a0", "a1", "work\n\nFinal Answer: \\boxed{4}", "why\n\nFinal Answer: \\boxed{5}"])
        self.assertEqual(names, ["math500", "math500", "gsm8k", "gsm8k"])

    def test_numina_hard_keeps_json_index_order_and_rejects_bad_indices(self):
        config = {"TARGET_BENCHMARKS": ["numina_hard"]}
        questions, answers, names = load_target_benchmarks(
            config, self.exemplar_data, load_json_fn=lambda _: [2, 0]
        )
        self.assertEqual(questions, ["numina two", "numina zero"])
        self.assertEqual(answers, ["solution two", "solution zero"])
        self.assertEqual(names, ["numina_hard", "numina_hard"])

        with self.assertRaises(ValueError):
            load_target_benchmarks(config, self.exemplar_data, load_json_fn=lambda _: [1, 1])
        with self.assertRaises(IndexError):
            load_target_benchmarks(config, self.exemplar_data, load_json_fn=lambda _: [3])
        with self.assertRaises(ValueError):
            load_target_benchmarks(config, self.exemplar_data, load_json_fn=lambda _: [])

    def test_scalar_mode_and_cap_remain_legacy_compatible(self):
        config = {"TARGET_BENCHMARK": " gsm8k ", "TARGET_BENCHMARKS": [], "BENCHMARK_MAX_QUESTIONS": 1}
        questions, answers, names = load_target_benchmarks(
            config, self.exemplar_data, load_json_fn=lambda _: [0],
            load_dataset_fn=self.fake_load_dataset,
        )
        self.assertEqual(questions, ["g0"])
        self.assertEqual(names, ["gsm8k"])
        self.assertEqual(len(answers), 1)

    def test_selection_rejects_invalid_and_repeated_combined_names(self):
        with self.assertRaises(ValueError):
            selected_benchmark_names({"TARGET_BENCHMARKS": ["gsm8k", " GSM8K "]})
        with self.assertRaises(ValueError):
            selected_benchmark_names({"TARGET_BENCHMARKS": ["unknown"]})
        with self.assertRaises(ValueError):
            selected_benchmark_names({"TARGET_BENCHMARKS": "gsm8k"})

    def test_per_index_benchmark_metadata_preserves_mixed_evaluation_modes(self):
        config = {"TARGET_BENCHMARK": "numina_hard", "TARGET_BENCHMARK_BY_INDEX": ["math500", "gsm8k"]}
        self.assertEqual(benchmark_name_for_target_index(config, 0), "math500")
        self.assertEqual(benchmark_name_for_target_index(config, 1), "gsm8k")


class DuplicateQuestionRetrievalTests(unittest.TestCase):
    def test_all_duplicate_self_matches_are_excluded(self):
        import numpy as np
        # Retrieval is otherwise dependency-free; provide the optional type
        # import so this unit test stays offline in a minimal test environment.
        if "sentence_transformers" not in sys.modules:
            sys.modules["sentence_transformers"] = types.SimpleNamespace(
                SentenceTransformer=type("SentenceTransformer", (), {})
            )
        from src import pipeline_steps

        original_generator = pipeline_steps._generate_embeddings
        pipeline_steps._generate_embeddings = lambda *_args, **_kwargs: np.array([[1.0, 0.0]])
        try:
            result = pipeline_steps.retrieve(
                "same", None, ["same", "other", "same"],
                np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
                top_k=3, question_to_index_map={"same": [0, 2]},
            )
        finally:
            pipeline_steps._generate_embeddings = original_generator

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["retrieved_indices"], [1])


if __name__ == "__main__":
    unittest.main()
