import json
import importlib.util
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock


def _import_core_simplification():
    """Import the target module when optional analytics dependencies are absent."""
    try:
        import src.core_simplification as module

        return module
    except ModuleNotFoundError as exc:
        if exc.name != "sklearn":
            raise

    # core_simplification only needs one function from the much larger
    # evaluation module. Keep this unit test runnable in minimal environments.
    sys.modules.pop("src.core_simplification", None)
    evaluation_stub = types.ModuleType("src.evaluation")
    evaluation_stub.evaluate_single_answer_with_llm = lambda *_args, **_kwargs: {
        "status": "SUCCESS",
        "is_correct": True,
    }
    sys.modules["src.evaluation"] = evaluation_stub
    import src.core_simplification as module

    return module


core_simplification = _import_core_simplification()


def _import_core_simplification_phase2():
    """Import Phase 2 with lightweight stand-ins for optional ML packages."""
    if importlib.util.find_spec("sklearn") is None:
        sklearn_stub = types.ModuleType("sklearn")
        metrics_stub = types.ModuleType("sklearn.metrics")
        pairwise_stub = types.ModuleType("sklearn.metrics.pairwise")
        pairwise_stub.cosine_similarity = lambda *_args, **_kwargs: None
        sklearn_stub.metrics = metrics_stub
        metrics_stub.pairwise = pairwise_stub
        sys.modules["sklearn"] = sklearn_stub
        sys.modules["sklearn.metrics"] = metrics_stub
        sys.modules["sklearn.metrics.pairwise"] = pairwise_stub

    if importlib.util.find_spec("sentence_transformers") is None:
        sentence_transformers_stub = types.ModuleType("sentence_transformers")
        sentence_transformers_stub.SentenceTransformer = object
        sys.modules["sentence_transformers"] = sentence_transformers_stub

    import src.core_simplification_layer2 as module

    return module


core_simplification_phase2 = _import_core_simplification_phase2()


class TrackingManager:
    def __init__(self, delay=0.03):
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.lock = threading.Lock()

    def generate_content(self, _prompt, _model, _temperature):
        with self.lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            return {"status": "SUCCESS", "text": "answer"}
        finally:
            with self.lock:
                self.active -= 1


class CoreSimplificationParallelTests(unittest.TestCase):
    def _run_attempts(self, parallel_enabled):
        manager = TrackingManager()
        trace = []
        config = {
            "GEMINI_MODEL_NAME_FINAL_SOLVER": "test-model",
            "QUESTION_PARALLEL_API_ENABLED": parallel_enabled,
            "QUESTION_PARALLEL_MAX_WORKERS": 2,
        }
        with mock.patch.object(core_simplification, "GeminiAPIManager", TrackingManager), mock.patch.object(
            core_simplification,
            "evaluate_single_answer_with_llm",
            return_value={"status": "SUCCESS", "is_correct": True},
        ):
            score, attempts = core_simplification._solve_and_evaluate(
                target_query="q",
                ground_truth="gt",
                prompt="prompt",
                api_manager_solve=manager,
                api_manager_eval=object(),
                config=config,
                n_attempts=4,
                temp=0.5,
                trace_name_prefix="test",
                local_trace=trace,
            )
        return manager, score, attempts, trace

    def test_repeated_attempts_run_in_parallel_and_keep_trace_order(self):
        manager, score, attempts, trace = self._run_attempts(True)

        self.assertGreaterEqual(manager.max_active, 2)
        self.assertEqual(score, 1.0)
        self.assertEqual(attempts, ["answer"] * 4)
        self.assertEqual(
            [entry["sub_step"] for entry in trace],
            [f"test_attempt_{index}" for index in range(1, 5)],
        )

    def test_repeated_attempts_remain_sequential_when_disabled(self):
        manager, score, attempts, trace = self._run_attempts(False)

        self.assertEqual(manager.max_active, 1)
        self.assertEqual(score, 1.0)
        self.assertEqual(len(attempts), 4)
        self.assertEqual(len(trace), 4)

    def test_phase1_batch_execution_parallelizes_questions_and_resumes(self):
        activity = TrackingManager(delay=0.03)

        def fake_single_question(target_query, ground_truth, **_kwargs):
            with activity.lock:
                activity.active += 1
                activity.calls += 1
                activity.max_active = max(activity.max_active, activity.active)
            try:
                time.sleep(activity.delay)
                return {
                    "status": "SUCCESS",
                    "original_question": target_query,
                    "ground_truth": ground_truth,
                }
            finally:
                with activity.lock:
                    activity.active -= 1

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "RESULTS_DIR": temp_dir,
                "experiment_name": "phase1_parallel_test",
                "CORE_SIMP_DATASET_NAME": "donors.json",
                "BATCH_PROCESSING_ENABLED": True,
                "BATCH_SIZE": 4,
                "BATCH_MAX_WORKERS": 4,
                "BATCH_FAILURE_POLICY": "halt_after_failed_batch",
                "BATCH_WRITE_JOURNALS": False,
            }
            with mock.patch.object(
                core_simplification,
                "run_core_simplification_phase1",
                side_effect=fake_single_question,
            ), mock.patch.object(
                core_simplification, "periodic_batch_sync_check"
            ):
                first_results = core_simplification.execute_core_simplification_phase1(
                    hard_questions=["q0", "q1", "q2", "q3"],
                    hard_solutions=["a0", "a1", "a2", "a3"],
                    exemplar_data={},
                    api_manager_solve=object(),
                    api_manager_eval=object(),
                    config=config,
                )
                first_call_count = activity.calls
                second_results = core_simplification.execute_core_simplification_phase1(
                    hard_questions=["q0", "q1", "q2", "q3"],
                    hard_solutions=["a0", "a1", "a2", "a3"],
                    exemplar_data={},
                    api_manager_solve=object(),
                    api_manager_eval=object(),
                    config=config,
                )

            self.assertGreaterEqual(activity.max_active, 2)
            self.assertEqual(first_call_count, 4)
            self.assertEqual(activity.calls, first_call_count)
            self.assertEqual([item["original_index"] for item in first_results], [0, 1, 2, 3])
            self.assertEqual(second_results, first_results)

            donors = json.loads(Path(temp_dir, "donors.json").read_text(encoding="utf-8"))
            run_log = json.loads(
                Path(temp_dir, "phase1_parallel_test_run_log.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(donors), 4)
            self.assertEqual(len(run_log), 4)

    def test_phase2_batch_execution_parallelizes_test_questions(self):
        activity = TrackingManager(delay=0.03)
        test_suite = [
            {
                "test_idx": index,
                "test_question": f"q{index}",
                "ground_truth": f"a{index}",
                "linked_donor_original_idx": 100 + index,
            }
            for index in range(4)
        ]

        def fake_branches(test_item, *_args, **_kwargs):
            with activity.lock:
                activity.active += 1
                activity.calls += 1
                activity.max_active = max(activity.max_active, activity.active)
            try:
                time.sleep(activity.delay)
                return {"status": "SUCCESS", "test_idx": test_item["test_idx"]}
            finally:
                with activity.lock:
                    activity.active -= 1

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "RESULTS_DIR": temp_dir,
                "experiment_name": "phase2_parallel_test",
                "CORE_SIMP_LAYER2_RESULTS_NAME": "phase2_results.json",
                "API_PROVIDER_SOLVER": "gemini",
                "API_PROVIDER_EVALUATOR": "gemini",
                "BATCH_PROCESSING_ENABLED": True,
                "BATCH_SIZE": 4,
                "BATCH_MAX_WORKERS": 4,
                "BATCH_FAILURE_POLICY": "halt_after_failed_batch",
                "BATCH_WRITE_JOURNALS": False,
            }
            with mock.patch.object(
                core_simplification_phase2,
                "build_paired_test_suite",
                return_value=test_suite,
            ), mock.patch.object(
                core_simplification_phase2,
                "run_parallel_evaluation_branches",
                side_effect=fake_branches,
            ), mock.patch.object(
                core_simplification_phase2, "periodic_batch_sync_check"
            ):
                results = core_simplification_phase2.execute_core_simplification_phase2(
                    hard_questions=[],
                    hard_solutions=[],
                    exemplar_data={},
                    embedding_model=object(),
                    api_managers={"gemini": object()},
                    config=config,
                )

            self.assertIs(
                core_simplification_phase2._solve_and_evaluate,
                core_simplification._solve_and_evaluate,
            )
            self.assertGreaterEqual(activity.max_active, 2)
            self.assertEqual(activity.calls, 4)
            self.assertEqual([item["test_idx"] for item in results], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
