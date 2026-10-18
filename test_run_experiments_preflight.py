"""Offline checks that run_experiments refuses runs that cannot do any work."""

from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from src import api_batching, api_manager


def _import_orchestration():
    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = type("SentenceTransformer", (), {})
    with patch.dict(sys.modules, {"sentence_transformers": sentence_transformers}):
        from src import orchestration
    return orchestration


# The Kaggle grouping experiment that produced an empty run log.
GROUPING_EXPERIMENT = {
    "experiment_name": "Layer1_Grouping_K5_Sizes2",
    "USE_RETRIEVAL": True,
    "TOP_N_CANDIDATES_RETRIEVAL": 5,
    "APPLY_LAYER1_GROUPING": True,
    "LAYER1_GROUPING_ONLY_MODE": True,
    "LAYER1_GROUP_SIZES": [2],
    "API_PROVIDER_ADAPTATION": "openrouter",
    "API_PROVIDER_SOLVER": "openrouter",
    "API_PROVIDER_EVALUATOR": "kaggle",
    "API_PROVIDER_SIMPLIFICATION": "openrouter",
}


class RunExperimentsPreflightTests(unittest.TestCase):
    def setUp(self):
        self.orchestration = _import_orchestration()
        self.temporary = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.temporary.name)
        self.config = {
            "RESULTS_DIR": str(self.results_dir),
            "DISTRIBUTED_EXECUTION_ENABLED": False,
            "BATCH_PROCESSING_ENABLED": False,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def run_log_path(self):
        return self.results_dir / f"{GROUPING_EXPERIMENT['experiment_name']}_run_log.json"

    def test_empty_question_list_raises_instead_of_saving_an_empty_log(self):
        managers = {"openrouter": object(), "kaggle": object()}
        with self.assertRaisesRegex(ValueError, "no target questions"):
            self.orchestration.run_experiments(
                [GROUPING_EXPERIMENT], self.config, [], None, {}, managers, [],
            )
        self.assertFalse(self.run_log_path().exists())

    def test_missing_provider_managers_raise_before_any_query(self):
        managers = {"gemini": object(), "avalai": object()}
        with patch.object(self.orchestration, "run_pipeline_for_single_query") as pipeline:
            with self.assertRaises(ValueError) as raised:
                self.orchestration.run_experiments(
                    [GROUPING_EXPERIMENT], self.config, ["q0", "q1"], None, {},
                    managers, ["a0", "a1"],
                )
        message = str(raised.exception)
        self.assertIn("API_PROVIDER_SOLVER='openrouter'", message)
        self.assertIn("API_PROVIDER_EVALUATOR='kaggle'", message)
        self.assertIn("['avalai', 'gemini']", message)
        pipeline.assert_not_called()
        self.assertFalse(self.run_log_path().exists())

    def test_matching_managers_process_every_question(self):
        managers = {"openrouter": object(), "kaggle": object()}

        def fake_pipeline(hard_list_idx, target_query, **kwargs):
            return {
                "target_query_original_hard_list_idx": hard_list_idx,
                "target_query_text": target_query,
                "pipeline_status": "SUCCESS",
            }

        with patch.object(
            self.orchestration, "run_pipeline_for_single_query", side_effect=fake_pipeline
        ), patch.object(self.orchestration, "periodic_sync_check"):
            results = self.orchestration.run_experiments(
                [GROUPING_EXPERIMENT], self.config, ["q0", "q1"], None, {},
                managers, ["a0", "a1"],
            )
        logs = results[GROUPING_EXPERIMENT["experiment_name"]]
        self.assertEqual([log["target_query_original_hard_list_idx"] for log in logs], [0, 1])
        self.assertTrue(self.run_log_path().exists())

    def test_dedicated_workflows_are_not_held_to_per_query_managers(self):
        builder = ModuleType("src.merging_dataset_builder")
        builder.build_merging_dataset = lambda **kwargs: ["built"]
        experiment = {
            "experiment_name": "Merging_Dataset",
            "APPLY_MERGING_DATASET_CONSTRUCTION": True,
            "API_PROVIDER_SOLVER": "openrouter",
        }
        with patch.dict(sys.modules, {"src.merging_dataset_builder": builder}):
            results = self.orchestration.run_experiments(
                [experiment], self.config, ["q0"], None, {}, {}, ["a0"],
            )
        self.assertEqual(results, {"Merging_Dataset": ["built"]})


UNBATCHABLE = "meta-llama/llama-3.1-8b-instruct"


def grouping_with_one_solver_call(api_manager_solve, config, **kwargs):
    """Stand-in for run_layer1_grouping: a failed call only fails its state."""
    model = api_manager.model_name_for(api_manager_solve, config, "final_solver")
    response = api_manager_solve.generate_content("Solve 1 + 1.", model)
    return {"overall_status": "SUCCESS" if response["status"] == "SUCCESS" else "FAILURE"}


class RefusedBatchStopsTheRunTests(unittest.TestCase):
    """The 2026-10-18 Kaggle run: OpenRouter refused to batch the solver model."""

    def setUp(self):
        self.orchestration = _import_orchestration()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.results_dir = Path(self.temporary.name)
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
            (self.orchestration, "run_layer1_grouping", grouping_with_one_solver_call),
            (self.orchestration, "periodic_sync_check", lambda *args, **kwargs: None),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for target, name in ((api_manager, "tprint"), (api_batching, "tprint"),
                             (api_manager.provider_pacer, "pace")):
            patcher = patch.object(target, name)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.config = {
            "RESULTS_DIR": str(self.results_dir),
            "DISTRIBUTED_EXECUTION_ENABLED": False,
            "BATCH_PROCESSING_ENABLED": False,
            "BATCH_WRITE_JOURNALS": False,
            "ENABLE_API_RETRY": False,
            "GLOBAL_CONSECUTIVE_ERROR_LIMIT": 0,
            "GLOBAL_PERIODIC_PAUSE_INTERVAL_MINUTES": 0,
            "API_BATCHING_ENABLED": True,
            "API_BATCH_COLLECT_SECONDS": 0.0,
            "OPENROUTER_MODEL_NAME_FINAL_SOLVER": UNBATCHABLE,
            "OPENROUTER_MODEL_ROUTING": {UNBATCHABLE: {"only": ["novita/fp8"]}},
        }

    def managers(self):
        openrouter = api_manager.OpenRouterAPIManager("test-key", {"default": {"rpm": 1000}}, self.config)
        return {"openrouter": openrouter, "kaggle": object()}

    def run_grouping(self, managers, **overrides):
        return self.orchestration.run_experiments(
            [{**GROUPING_EXPERIMENT, **overrides}], self.config, ["q0", "q1"], None, {},
            managers, ["a0", "a1"],
        )

    def test_a_refused_batch_raises_before_the_question_log_is_saved(self):
        modes = {
            "query by query": {},
            "batch processing": {"BATCH_PROCESSING_ENABLED": True, "BATCH_SIZE": 2,
                                 "BATCH_MAX_WORKERS": 1},
            "deferred solve": {"DEFER_SOLVE_STEP": True},
        }
        log_path = self.results_dir / f"{GROUPING_EXPERIMENT['experiment_name']}_run_log.json"
        for mode, overrides in modes.items():
            with self.subTest(mode):
                self.posts = 0
                with self.assertRaises(api_batching.BatchRefusedError) as raised:
                    self.run_grouping(self.managers(), **overrides)
                message = str(raised.exception)
                self.assertIn(f"OpenRouter cannot batch {UNBATCHABLE}", message)
                self.assertIn("API_BATCHING_ENABLED=True", message)
                self.assertFalse(log_path.exists())
                # In batch mode the second question failed without a request.
                self.assertEqual(self.posts, 1)

    def test_an_experiment_switch_that_overrides_config_is_named(self):
        # CONFIG asks for Kaggle-only batching, but the experiment dict still
        # says True, as in the notebook copy run on 2026-10-18.
        self.config["API_BATCHING_ENABLED"] = {"kaggle": True, "openrouter": False}
        with self.assertLogs("src.orchestration", level="WARNING") as logs, \
                self.assertRaises(api_batching.BatchRefusedError) as raised:
            self.run_grouping(self.managers(), API_BATCHING_ENABLED=True)
        setting = (
            f"API_BATCHING_ENABLED=True (set by experiment {GROUPING_EXPERIMENT['experiment_name']!r}; "
            "CONFIG has {'kaggle': True, 'openrouter': False})"
        )
        self.assertIn(f"{setting} batches this model", str(raised.exception))
        warning = "\n".join(logs.output)
        self.assertIn(setting, warning)
        self.assertIn(f"final_solver openrouter/{UNBATCHABLE} batched", warning)

    def test_the_config_switch_alone_sends_the_model_directly(self):
        self.config["API_BATCHING_ENABLED"] = {"kaggle": True, "openrouter": False}
        logs = self.run_grouping(self.managers())[GROUPING_EXPERIMENT["experiment_name"]]
        self.assertEqual({log["pipeline_status"] for log in logs}, {"SUCCESS"})
        self.assertEqual(self.posts, 0)

    def test_a_misspelled_provider_fails_before_any_question(self):
        self.config["API_BATCHING_ENABLED"] = {"kaggle": True, "open_router": False}
        log_path = self.results_dir / f"{GROUPING_EXPERIMENT['experiment_name']}_run_log.json"
        with patch.object(self.orchestration, "run_pipeline_for_single_query") as pipeline, \
                self.assertRaisesRegex(ValueError, "open_router"):
            self.run_grouping(self.managers())
        pipeline.assert_not_called()
        self.assertFalse(log_path.exists())

    def test_a_fixed_rerun_in_the_same_kernel_runs_every_question(self):
        managers = self.managers()
        with self.assertRaises(api_batching.BatchRefusedError):
            self.run_grouping(managers)
        self.config["API_BATCHING_ENABLED"] = {"kaggle": True, "openrouter": False}
        logs = self.run_grouping(managers)[GROUPING_EXPERIMENT["experiment_name"]]
        self.assertEqual([log["target_query_original_hard_list_idx"] for log in logs], [0, 1])
        self.assertEqual({log["pipeline_status"] for log in logs}, {"SUCCESS"})
        self.assertEqual(self.posts, 1)


if __name__ == "__main__":
    unittest.main()
