import importlib.util
import json
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock


def _install_optional_dependency_stubs():
    if "sklearn" not in sys.modules and importlib.util.find_spec("sklearn") is None:
        sklearn_stub = types.ModuleType("sklearn")
        metrics_stub = types.ModuleType("sklearn.metrics")
        pairwise_stub = types.ModuleType("sklearn.metrics.pairwise")
        pairwise_stub.cosine_similarity = lambda *_args, **_kwargs: None
        sklearn_stub.metrics = metrics_stub
        metrics_stub.pairwise = pairwise_stub
        sys.modules["sklearn"] = sklearn_stub
        sys.modules["sklearn.metrics"] = metrics_stub
        sys.modules["sklearn.metrics.pairwise"] = pairwise_stub

    if (
        "sentence_transformers" not in sys.modules
        and importlib.util.find_spec("sentence_transformers") is None
    ):
        sentence_transformers_stub = types.ModuleType("sentence_transformers")
        sentence_transformers_stub.SentenceTransformer = object
        sys.modules["sentence_transformers"] = sentence_transformers_stub

    if "src.evaluation" not in sys.modules:
        evaluation_stub = types.ModuleType("src.evaluation")
        evaluation_stub.evaluate_single_answer_with_llm = lambda *_args, **_kwargs: {
            "status": "SUCCESS",
            "is_correct": True,
        }
        sys.modules["src.evaluation"] = evaluation_stub


_install_optional_dependency_stubs()

import src.merging_dataset_builder as merging_dataset_builder
import src.pipeline_steps as pipeline_steps


class TrackingManager:
    def __init__(self, response_factory=None, delay=0.03):
        self.response_factory = response_factory or (
            lambda prompt: {"status": "SUCCESS", "text": f"merged:{prompt}"}
        )
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.lock = threading.Lock()

    def generate_content(self, prompt, _model, _temperature):
        with self.lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            return self.response_factory(prompt)
        finally:
            with self.lock:
                self.active -= 1


class MergingParallelTests(unittest.TestCase):
    def test_merging_dataset_internal_calls_run_in_parallel(self):
        def response_factory(prompt):
            if prompt == "few-shot":
                text = "few-shot-correct"
            elif prompt == "base":
                text = "base-wrong"
            else:
                text = f"solution:{prompt}"
            return {"status": "SUCCESS", "text": text}

        manager = TrackingManager(response_factory=response_factory)
        exemplar_data = {
            "questions": ["e0", "e1"],
            "solutions": ["s0", "s1"],
            "embeddings": [object(), object()],
        }
        config = {
            "GEMINI_MODEL_NAME_FINAL_SOLVER": "test-model",
            "MERGING_DS_CCS_N": 3,
            "MERGING_DS_TEMPERATURE": 0.5,
            "MERGING_DS_BASE_CHECK": True,
            "QUESTION_PARALLEL_API_ENABLED": True,
            "QUESTION_PARALLEL_MAX_WORKERS": 3,
        }

        def final_prompt(_target, exemplars, _config):
            if len(exemplars) == 2 and "Question: e0" in exemplars[0]:
                return "few-shot"
            if len(exemplars) == 1:
                return "r1" if "Question: e0" in exemplars[0] else "r2"
            return "dataset-input"

        with mock.patch.object(
            merging_dataset_builder, "GeminiAPIManager", TrackingManager
        ), mock.patch.object(
            merging_dataset_builder,
            "retrieve",
            return_value={"status": "SUCCESS", "retrieved_indices": [0, 1]},
        ), mock.patch.object(
            merging_dataset_builder,
            "create_final_reasoning_prompt",
            side_effect=final_prompt,
        ), mock.patch.object(
            merging_dataset_builder,
            "create_final_reasoning_prompt_simple",
            return_value="base",
        ), mock.patch.object(
            merging_dataset_builder,
            "evaluate_single_answer_with_llm",
            side_effect=lambda answer, *_args, **_kwargs: {
                "status": "SUCCESS",
                "is_correct": answer == "few-shot-correct",
            },
        ):
            result = merging_dataset_builder.process_single_question(
                target_query="target",
                ground_truth="ground-truth",
                exemplar_data=exemplar_data,
                embedding_model=object(),
                api_manager_solve=manager,
                api_manager_eval=object(),
                config=config,
            )

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["few_shot_ccs"], 1.0)
        self.assertEqual(result["base_ccs"], 0.0)
        self.assertEqual(manager.calls, 8)  # 3 few-shot + 3 base + R1 + R2
        self.assertGreaterEqual(manager.max_active, 3)

    def test_standard_merge_parallelizes_independent_pairs(self):
        parallel_manager = TrackingManager()
        parallel_config = {
            "APPLY_MERGING": True,
            "TARGET_ADAPTED_SAMPLES_MERGING": 3,
            "GEMINI_MODEL_NAME_ADAPTATION": "test-model",
            "DEFAULT_ADAPTATION_TEMPERATURE": 0.2,
            "QUESTION_PARALLEL_API_ENABLED": True,
            "QUESTION_PARALLEL_MAX_WORKERS": 3,
        }

        with mock.patch.object(
            pipeline_steps, "GeminiAPIManager", TrackingManager
        ), mock.patch.object(
            pipeline_steps,
            "create_merging_prompt",
            side_effect=lambda _target, pair: "+".join(pair),
        ):
            parallel_result = pipeline_steps.merge(
                target_query="target",
                adapted_texts=["a", "b", "c", "d", "e", "f"],
                embedding_model=object(),
                api_manager=parallel_manager,
                config=parallel_config,
            )

            sequential_manager = TrackingManager()
            sequential_config = dict(parallel_config)
            sequential_config["QUESTION_PARALLEL_API_ENABLED"] = False
            sequential_result = pipeline_steps.merge(
                target_query="target",
                adapted_texts=["a", "b", "c", "d", "e", "f"],
                embedding_model=object(),
                api_manager=sequential_manager,
                config=sequential_config,
            )

        self.assertEqual(len(parallel_result["merged_texts"]), 3)
        self.assertEqual(parallel_result["merged_texts"], sequential_result["merged_texts"])
        self.assertGreaterEqual(parallel_manager.max_active, 3)
        self.assertEqual(sequential_manager.max_active, 1)

    def test_merging_dataset_batch_parallelizes_questions_and_resumes(self):
        activity = TrackingManager(delay=0.03)

        def fake_process(target_query, ground_truth, **_kwargs):
            with activity.lock:
                activity.active += 1
                activity.calls += 1
                activity.max_active = max(activity.max_active, activity.active)
            try:
                time.sleep(activity.delay)
                return {
                    "status": "SUCCESS",
                    "target_query": target_query,
                    "few_shot_ccs": 1.0,
                    "base_ccs": 0.0,
                    "dataset_entry": {
                        "input_prompt": f"input:{target_query}",
                        "label": ground_truth,
                    },
                }
            finally:
                with activity.lock:
                    activity.active -= 1

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "RESULTS_DIR": temp_dir,
                "experiment_name": "merging_parallel_test",
                "MERGING_DS_OUTPUT_FILENAME": "merging_dataset.json",
                "API_PROVIDER_SOLVER": "gemini",
                "API_PROVIDER_EVALUATOR": "gemini",
                "BATCH_PROCESSING_ENABLED": True,
                "BATCH_SIZE": 4,
                "BATCH_MAX_WORKERS": 4,
                "BATCH_FAILURE_POLICY": "halt_after_failed_batch",
                "BATCH_WRITE_JOURNALS": False,
            }
            with mock.patch.object(
                merging_dataset_builder,
                "process_single_question",
                side_effect=fake_process,
            ), mock.patch.object(
                merging_dataset_builder, "periodic_batch_sync_check"
            ):
                first_results = merging_dataset_builder.build_merging_dataset(
                    hard_questions=["q0", "q1", "q2", "q3"],
                    hard_solutions=["a0", "a1", "a2", "a3"],
                    exemplar_data={},
                    embedding_model=object(),
                    api_managers={"gemini": object()},
                    config=config,
                )
                first_call_count = activity.calls
                second_results = merging_dataset_builder.build_merging_dataset(
                    hard_questions=["q0", "q1", "q2", "q3"],
                    hard_solutions=["a0", "a1", "a2", "a3"],
                    exemplar_data={},
                    embedding_model=object(),
                    api_managers={"gemini": object()},
                    config=config,
                )

            dataset = json.loads(
                Path(temp_dir, "merging_dataset.json").read_text(encoding="utf-8")
            )
            run_log = json.loads(
                Path(temp_dir, "merging_parallel_test_run_log.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertGreaterEqual(activity.max_active, 2)
        self.assertEqual(first_call_count, 4)
        self.assertEqual(activity.calls, first_call_count)
        self.assertEqual(first_results, second_results)
        self.assertEqual(len(dataset), 4)
        self.assertEqual(len(run_log), 4)
        self.assertEqual(
            [entry["metadata"]["original_index"] for entry in dataset],
            [0, 1, 2, 3],
        )


if __name__ == "__main__":
    unittest.main()
