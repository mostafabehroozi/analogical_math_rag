"""Offline checks that a refused provider batch stops the dedicated workflows.

Without BATCH_PROCESSING_ENABLED, each dedicated workflow runs its questions
one by one and saves every result, which the resume then counts as done.  A
provider that refuses to batch the solver model fails every call, so each loop
must raise before it saves anything for that question.
"""

from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from src import api_batching, api_manager


def _import_workflows():
    # Only sentence_transformers is stubbed. The workflow modules stay in
    # sys.modules, so Layer 2 can pickle its ExperimentResult checkpoint.
    stubbed = "sentence_transformers" not in sys.modules
    if stubbed:
        sentence_transformers = ModuleType("sentence_transformers")
        sentence_transformers.SentenceTransformer = type("SentenceTransformer", (), {})
        sys.modules["sentence_transformers"] = sentence_transformers
    try:
        from src import (
            core_simplification,
            core_simplification_layer2,
            layer2_analysis,
            merging_dataset_builder,
            transformation_dataset_builder,
        )
    finally:
        if stubbed:
            del sys.modules["sentence_transformers"]
    return SimpleNamespace(
        phase1=core_simplification,
        phase2=core_simplification_layer2,
        layer2=layer2_analysis,
        merging=merging_dataset_builder,
        transformation=transformation_dataset_builder,
    )


WORKFLOWS = _import_workflows()
UNBATCHABLE = "meta-llama/llama-3.1-8b-instruct"
QUESTIONS = ["q0", "q1"]
SOLUTIONS = ["a0", "a1"]


def solved_once(manager, config):
    """One solver call: a refused batch only fails this call's response."""
    model = api_manager.model_name_for(manager, config, "final_solver")
    return manager.generate_content("Solve 1 + 1.", model)["status"] == "SUCCESS"


def status(solved):
    return "SUCCESS" if solved else "FAILURE"


class RefusedBatchStopsTheDedicatedWorkflowsTests(unittest.TestCase):
    """The 2026-10-18 Kaggle refusal, in each workflow's query-by-query loop."""

    def setUp(self):
        self.workflows = WORKFLOWS
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.posts = 0

        def refuse(manager, method, path, api_key, payload=None):
            self.posts += 1
            return 400, {"error": {"message": f"Model '{UNBATCHABLE}' does not have a :batch endpoint"}}

        # Direct (non-batch) chat completions answer every call.
        completion = SimpleNamespace(model=UNBATCHABLE, model_extra={},
                                     choices=[SimpleNamespace(message=SimpleNamespace(content="2"))])
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: completion)))
        fake_openai = SimpleNamespace(OpenAI=lambda **kwargs: client,
                                      RateLimitError=Exception, APIStatusError=Exception)
        for target, name, value in (
            (api_manager, "openai", fake_openai),
            (api_manager.OpenRouterAPIManager, "_batch_http", refuse),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for target, name in ((api_manager, "tprint"), (api_batching, "tprint"),
                             (api_manager.provider_pacer, "pace")):
            patcher = patch.object(target, name)
            patcher.start()
            self.addCleanup(patcher.stop)

    def start(self, workflow):
        """A fresh results folder, configuration and OpenRouter manager."""
        self.results_dir = Path(self.temporary.name) / workflow
        self.results_dir.mkdir()
        self.posts = 0
        self.config = {
            "RESULTS_DIR": str(self.results_dir),
            "DISTRIBUTED_EXECUTION_ENABLED": False,
            "BATCH_PROCESSING_ENABLED": False,
            "ENABLE_API_RETRY": False,
            "GLOBAL_CONSECUTIVE_ERROR_LIMIT": 0,
            "GLOBAL_PERIODIC_PAUSE_INTERVAL_MINUTES": 0,
            "API_BATCHING_ENABLED": True,
            "API_BATCH_COLLECT_SECONDS": 0.0,
            "API_PROVIDER_ADAPTATION": "openrouter",
            "API_PROVIDER_SOLVER": "openrouter",
            "API_PROVIDER_EVALUATOR": "openrouter",
            "OPENROUTER_MODEL_NAME_FINAL_SOLVER": UNBATCHABLE,
            "OPENROUTER_MODEL_ROUTING": {UNBATCHABLE: {"only": ["novita/fp8"]}},
        }
        return api_manager.OpenRouterAPIManager("test-key", {"default": {"rpm": 1000}}, self.config)

    def run_phase1(self, manager):
        """Returns the indices of the saved run-log entries."""
        workflow = self.workflows.phase1

        def simplify(target_query, ground_truth, api_manager_solve, api_manager_eval, config):
            return {"status": status(solved_once(api_manager_solve, config))}

        with patch.object(workflow, "run_core_simplification_phase1", simplify):
            logs = workflow.execute_core_simplification_phase1(
                QUESTIONS, SOLUTIONS, {}, manager, manager, self.config)
        return [log["original_index"] for log in logs if log["status"] == "SUCCESS"]

    def run_phase2(self, manager):
        workflow = self.workflows.phase2
        tests = [{"test_idx": index, "test_question": question}
                 for index, question in enumerate(QUESTIONS)]

        def branches(test_item, solver_mgr, eval_mgr, config):
            return {"test_idx": test_item["test_idx"], "solved": solved_once(solver_mgr, config)}

        with patch.object(workflow, "build_paired_test_suite", lambda *args, **kwargs: tests), \
                patch.object(workflow, "run_parallel_evaluation_branches", branches):
            results = workflow.execute_core_simplification_phase2(
                QUESTIONS, SOLUTIONS, {}, None, {"openrouter": manager}, self.config)
        return [result["test_idx"] for result in results if result["solved"]]

    def run_merging(self, manager):
        workflow = self.workflows.merging

        def process(target_query, ground_truth, exemplar_data, embedding_model,
                    api_manager_solve, api_manager_eval, config):
            return {"status": status(solved_once(api_manager_solve, config)),
                    "dataset_entry": {"prompt": target_query}, "few_shot_ccs": 1.0, "base_ccs": 0.0}

        with patch.object(workflow, "process_single_question", process):
            logs = workflow.build_merging_dataset(
                QUESTIONS, SOLUTIONS, {}, None, {"openrouter": manager}, self.config)
        return [log["original_index"] for log in logs if log["status"] == "SUCCESS"]

    def run_transformation(self, manager):
        workflow = self.workflows.transformation

        def process(target, exemplar_data, embedding_model, adapt_manager, solve_manager,
                    eval_manager, config):
            return {"original_index": target["index"],
                    "status": status(solved_once(solve_manager, config)),
                    "dataset_entry": {"prompt": target["question"]}}

        with patch.object(workflow, "process_single_question", process):
            logs = workflow.build_transformation_dataset(
                QUESTIONS, SOLUTIONS, {}, None, {"openrouter": manager}, self.config)
        return [log["original_index"] for log in logs if log["status"] == "SUCCESS"]

    def run_layer2(self, manager):
        workflow = self.workflows.layer2

        def run_single_query(orchestrator, layer1_state):
            solved = solved_once(orchestrator.api_manager_solve, orchestrator.global_config)
            result = workflow.ExperimentResult(
                target_query_idx=layer1_state["target_query_idx"],
                target_query_text=layer1_state["target_query_text"],
                ground_truth_answer=SOLUTIONS[layer1_state["target_query_idx"]],
                utility_calibration="Baseline", evaluator_setting="Baseline",
                scoring_strategy="", application="Block_Zero_Baseline",
                pass_at_k_metrics={1: float(solved)},
            )
            orchestrator.all_results.append(result)
            return [result]

        states = [{"target_query_idx": index, "target_query_text": question}
                  for index, question in enumerate(QUESTIONS)]
        with patch.object(workflow.Layer2Orchestrator, "run_single_query", run_single_query):
            results, _report = workflow.run_layer2_experiments(
                states, workflow.Layer2Config(layer2_config_name="refused_batch", global_pass_at_N=1),
                str(self.results_dir), manager, manager, self.config)
        return [result.target_query_idx for result in results if result.pass_at_k_metrics[1]]

    def workflow_runs(self):
        return {
            "core simplification phase 1": self.run_phase1,
            "core simplification phase 2": self.run_phase2,
            "merging dataset": self.run_merging,
            "transformation dataset": self.run_transformation,
            "layer 2 analysis": self.run_layer2,
        }

    def test_a_refused_batch_raises_before_the_question_is_saved(self):
        for name, run in self.workflow_runs().items():
            with self.subTest(name):
                manager = self.start(name)
                with self.assertRaises(api_batching.BatchRefusedError) as raised:
                    run(manager)
                message = str(raised.exception)
                self.assertIn(f"OpenRouter cannot batch {UNBATCHABLE}", message)
                self.assertIn("API_BATCHING_ENABLED=True", message)
                # No run log, dataset row, result file or checkpoint.
                self.assertEqual(list(self.results_dir.iterdir()), [])
                # The run stopped after the first question.
                self.assertEqual(self.posts, 1)

    def test_a_fixed_rerun_in_the_same_kernel_saves_every_question(self):
        for name, run in self.workflow_runs().items():
            with self.subTest(name):
                manager = self.start(name)
                with self.assertRaises(api_batching.BatchRefusedError):
                    run(manager)
                self.config["API_BATCHING_ENABLED"] = {"openrouter": False}
                self.assertEqual(run(manager), [0, 1])
                self.assertNotEqual(list(self.results_dir.iterdir()), [])
                self.assertEqual(self.posts, 1)


if __name__ == "__main__":
    unittest.main()
