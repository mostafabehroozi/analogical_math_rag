import sys
import threading
import time
import types
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.batching import BatchCoordinator, QuestionWorkItem
from src.context_logger import ctx_batch_id, ctx_query_idx
from src.parallel_utils import run_parallel_api_calls


# Keep these focused tests runnable in lightweight maintenance environments.
# Production still imports the real optional package when it is installed.
try:
    __import__("sentence_transformers")
except ModuleNotFoundError:
    sentence_transformers_stub = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        pass

    sentence_transformers_stub.SentenceTransformer = SentenceTransformer
    sys.modules["sentence_transformers"] = sentence_transformers_stub


class ParallelHelperTests(unittest.TestCase):
    def test_parallel_helper_bounds_concurrency_and_preserves_order(self):
        lock = threading.Lock()
        active = 0
        peak = 0

        def make_task(index):
            def task():
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.01 * (4 - index % 4))
                with lock:
                    active -= 1
                return index

            return task

        results = run_parallel_api_calls(
            [make_task(index) for index in range(9)],
            {
                "QUESTION_PARALLEL_API_ENABLED": True,
                "QUESTION_PARALLEL_MAX_WORKERS": 3,
            },
        )

        self.assertEqual(results, list(range(9)))
        self.assertEqual(peak, 3)

    def test_parallel_helper_keeps_sequential_fallback(self):
        calls = []
        results = run_parallel_api_calls(
            [lambda index=index: calls.append(index) or index for index in range(4)],
            {"QUESTION_PARALLEL_API_ENABLED": False},
        )

        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertEqual(results, [0, 1, 2, 3])

    def test_parallel_helper_starts_new_work_when_one_call_is_slow(self):
        later_task_started = threading.Event()

        def slow_first_task():
            return later_task_started.wait(timeout=0.5)

        def short_task(value):
            time.sleep(0.01)
            return value

        def release_task():
            later_task_started.set()
            return 3

        results = run_parallel_api_calls(
            [
                slow_first_task,
                lambda: short_task(1),
                lambda: short_task(2),
                release_task,
            ],
            {
                "QUESTION_PARALLEL_API_ENABLED": True,
                "QUESTION_PARALLEL_MAX_WORKERS": 3,
            },
        )

        self.assertEqual(results, [True, 1, 2, 3])


class BatchCoordinatorTests(unittest.TestCase):
    def test_batch_context_contains_real_question_and_batch_ids(self):
        contexts = {}
        commits = []

        def worker(item):
            contexts[item.index] = (ctx_query_idx.get(), ctx_batch_id.get())
            return {"pipeline_status": "SUCCESS"}

        def commit(results, batch_id, batch_number):
            commits.append((batch_id, [result.item.index for result in results]))

        coordinator = BatchCoordinator(
            {
                "BATCH_SIZE": 2,
                "BATCH_MAX_WORKERS": 5,
                "BATCH_WRITE_JOURNALS": False,
            },
            "test",
            "phase",
        )
        coordinator.run(
            [QuestionWorkItem(index, f"q{index}") for index in range(3)],
            worker,
            commit,
        )

        self.assertEqual(contexts[0], (0, "phase-0001-0-1"))
        self.assertEqual(contexts[1], (1, "phase-0001-0-1"))
        self.assertEqual(contexts[2], (2, "phase-0002-2-2"))
        self.assertEqual(
            commits,
            [
                ("phase-0001-0-1", [0, 1]),
                ("phase-0002-2-2", [2]),
            ],
        )


class WorkflowConcurrencyTests(unittest.TestCase):
    def test_layer1_keeps_step_order_without_nested_stage_pools(self):
        from src import layer1_base_execution as layer1

        calls = []

        def record(name, result):
            def side_effect(*args, **kwargs):
                calls.append(name)
                return result

            return side_effect

        with TemporaryDirectory() as temp_dir, patch.object(
            layer1, "_load_cached_state", return_value=None
        ), patch.object(
            layer1, "print", return_value=None
        ), patch.object(
            layer1,
            "_execute_retrieval",
            side_effect=record(
                "A",
                {
                    "status": "SUCCESS",
                    "retrieved_indices": [7],
                    "retrieval_data": [{"corpus_index": 7}],
                    "trace": [],
                },
            ),
        ), patch.object(
            layer1,
            "_execute_candidate_generation",
            side_effect=record(
                "B",
                {
                    "status": "SUCCESS",
                    "candidates": {
                        7: {
                            "candidate_id": 7,
                            "candidate_text": "candidate",
                            "generation_status": "SUCCESS",
                        }
                    },
                    "generated_count": 1,
                },
            ),
        ), patch.object(
            layer1,
            "_execute_baseline_calculation",
            side_effect=record(
                "C", {"status": "SUCCESS", "intrinsic_baselines": {7: 1.0}}
            ),
        ), patch.object(
            layer1,
            "_execute_cross_evaluation",
            side_effect=record(
                "D", {"status": "SUCCESS", "cross_evaluation_matrix": {7: {7: 1.0}}}
            ),
        ), patch.object(
            layer1,
            "_execute_ground_truth_evaluation",
            side_effect=record(
                "E",
                {
                    "status": "SUCCESS",
                    "ground_truth_labels": {7: {"is_correct": True}},
                    "successful_evals": 1,
                },
            ),
        ):
            state = layer1.run_layer1_base_execution(
                target_query_index=0,
                target_query="question",
                ground_truth_answer="answer",
                embedding_model=None,
                exemplar_questions=["example"],
                exemplar_solutions=["solution"],
                embedded_exemplars=None,
                exemplar_data={"questions": ["example"], "solutions": ["solution"]},
                api_manager_solve=object(),
                api_manager_eval=object(),
                config={
                    "LAYER1_CACHE_DIR": temp_dir,
                    "TOP_N_CANDIDATES_RETRIEVAL": 1,
                    "LAYER1_ONE_SHOT_CANDIDATES_N": 1,
                },
                experiment_name="test",
            )

        self.assertEqual(calls, ["A", "B", "C", "D", "E"])
        self.assertEqual(state["overall_status"], "SUCCESS")
        self.assertEqual(state["candidate_set"][7]["candidate_text"], "candidate")

    def test_layer1_rejects_stale_zero_shot_cache_entries(self):
        from src import layer1_base_execution as layer1

        legacy_state = {
            "metadata": {"config_snapshot": {}},
            "candidate_set": {
                7: {"candidate_text": "one-shot"},
                "zs_0": {"candidate_text": "zero-0"},
                "zs_1": {"candidate_text": "zero-1"},
            },
        }
        current_state = {
            "metadata": {
                "config_snapshot": {"LAYER1_ZERO_SHOT_CANDIDATES_N": 3}
            },
            "candidate_set": {},
        }

        self.assertTrue(layer1._cache_matches_zero_shot_config(legacy_state, 2))
        self.assertFalse(layer1._cache_matches_zero_shot_config(legacy_state, 3))
        self.assertTrue(layer1._cache_matches_zero_shot_config(current_state, 3))

    def test_standard_solver_parallelizes_pass_attempts_in_stable_order(self):
        from src import pipeline_steps

        class DummyManager:
            def __init__(self):
                self.lock = threading.Lock()
                self.active = 0
                self.peak = 0

            def generate_content(self, prompt, model_name, temperature):
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(0.01)
                with self.lock:
                    self.active -= 1
                return {"status": "SUCCESS", "text": "answer"}

        manager = DummyManager()
        config = {
            "N_PASS_ATTEMPTS": 3,
            "AVALAI_MODEL_NAME_FINAL_SOLVER": "model",
            "DEFAULT_PASS_N_SOLVER_TEMPERATURE": 1.0,
            "PROMPT_TEMPLATE_FINAL_SOLVER": "final_solver_v2",
            "QUESTION_PARALLEL_API_ENABLED": True,
            "QUESTION_PARALLEL_MAX_WORKERS": 3,
        }

        with patch.object(pipeline_steps, "AvalAIAPIManager", DummyManager), patch.object(
            pipeline_steps,
            "create_final_reasoning_prompt_simple",
            return_value="prompt",
        ):
            result = pipeline_steps.solve("question", [], manager, config)

        self.assertEqual(result["solution_attempts"], ["answer", "answer", "answer"])
        self.assertEqual(
            [entry["sub_step"] for entry in result["trace"]],
            ["attempt_1", "attempt_2", "attempt_3"],
        )
        self.assertEqual(manager.peak, 3)

    def test_layer2_uses_one_manager_owned_retry_chain_per_attempt(self):
        from src import layer2_analysis

        class DummyManager:
            def __init__(self):
                self.lock = threading.Lock()
                self.active = 0
                self.peak = 0
                self.calls = 0

            def generate_content(self, prompt, model_name, temperature):
                with self.lock:
                    self.calls += 1
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(0.01)
                with self.lock:
                    self.active -= 1
                return {"status": "SUCCESS", "text": "answer"}

        solver = DummyManager()
        evaluator = DummyManager()
        config = {
            "AVALAI_MODEL_NAME_FINAL_SOLVER": "model",
            "DEFAULT_PASS_N_SOLVER_TEMPERATURE": 1.0,
            "QUESTION_PARALLEL_API_ENABLED": True,
            "QUESTION_PARALLEL_MAX_WORKERS": 2,
        }

        with patch.object(layer2_analysis, "AvalAIAPIManager", DummyManager), patch.object(
            layer2_analysis,
            "create_final_reasoning_prompt_simple",
            return_value="prompt",
        ), patch.object(
            layer2_analysis,
            "evaluate_single_answer_with_llm",
            return_value={"status": "SUCCESS", "is_correct": True},
        ):
            engine = layer2_analysis.ActiveInferenceEngine(solver, evaluator, config)
            _, executions, pass_at_k = engine.execute_and_evaluate(
                target_query="question",
                ground_truth="answer",
                context_texts=[],
                n_attempts=3,
            )

        self.assertEqual(solver.calls, 3)
        self.assertEqual(solver.peak, 2)
        self.assertEqual(
            [execution["attempt_index"] for execution in executions], [1, 2, 3]
        )
        self.assertEqual(pass_at_k, {1: 1.0, 2: 1.0, 3: 1.0})


if __name__ == "__main__":
    unittest.main()
