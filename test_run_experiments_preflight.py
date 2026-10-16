"""Offline checks that run_experiments refuses runs that cannot do any work."""

from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch


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


if __name__ == "__main__":
    unittest.main()
