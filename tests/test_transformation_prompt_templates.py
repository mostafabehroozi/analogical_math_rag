import sys
import types
import unittest
from unittest.mock import patch

# The helper under test does not use sentence-transformers. Stub this optional
# runtime dependency so the focused configuration tests stay lightweight.
if "sentence_transformers" not in sys.modules:
    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = object
    sys.modules["sentence_transformers"] = sentence_transformers

from src.transformation_dataset_builder import (
    _transformation_candidate_specs,
    process_single_question,
)


class _FakeAdaptationManager:
    def __init__(self):
        self.calls = []

    def generate_content(self, prompt, model_name, temperature, **kwargs):
        self.calls.append((prompt, model_name, temperature, kwargs))
        return {
            "status": "SUCCESS",
            "text": f"transformed-sample-{len(self.calls)}",
        }


class TransformationPromptTemplateTests(unittest.TestCase):
    def test_dictionary_expands_templates_in_order_with_repeated_samples(self):
        specs = _transformation_candidate_specs(
            {
                "TRANSFORMATION_DS_PROMPT_TEMPLATES": {
                    "transformation_shallow": 2,
                    "transformation_complete": 1,
                }
            }
        )

        self.assertEqual(
            specs,
            [
                {
                    "candidate_index": 0,
                    "template_name": "transformation_shallow",
                    "template_sample_index": 0,
                },
                {
                    "candidate_index": 1,
                    "template_name": "transformation_shallow",
                    "template_sample_index": 1,
                },
                {
                    "candidate_index": 2,
                    "template_name": "transformation_complete",
                    "template_sample_index": 0,
                },
            ],
        )

    def test_explicit_legacy_fallback_remains_supported(self):
        specs = _transformation_candidate_specs(
            {
                "TRANSFORMATION_DS_PROMPT_TEMPLATES": None,
                "TRANSFORMATION_DS_PROMPT_TEMPLATE": "transformation_v1",
                "TRANSFORMATION_DS_N_CANDIDATES": 2,
            }
        )

        self.assertEqual(
            [spec["template_name"] for spec in specs],
            ["transformation_v1", "transformation_v1"],
        )
        self.assertEqual(
            [spec["template_sample_index"] for spec in specs],
            [0, 1],
        )

    def test_invalid_template_dictionary_fails_before_api_execution(self):
        invalid_configs = [
            {"TRANSFORMATION_DS_PROMPT_TEMPLATES": []},
            {"TRANSFORMATION_DS_PROMPT_TEMPLATES": {}},
            {"TRANSFORMATION_DS_PROMPT_TEMPLATES": {"missing_template": 1}},
            {"TRANSFORMATION_DS_PROMPT_TEMPLATES": {"transformation_v1": 0}},
            {"TRANSFORMATION_DS_PROMPT_TEMPLATES": {"transformation_v1": True}},
        ]

        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    _transformation_candidate_specs(config)

    @patch(
        "src.transformation_dataset_builder.retrieve",
        return_value={
            "status": "SUCCESS",
            "retrieved_indices": [0],
            "retrieved_similarity_scores": [0.9],
        },
    )
    @patch("src.transformation_dataset_builder._ccs")
    def test_process_uses_every_configured_template_and_sample_count(
        self, mock_ccs, _mock_retrieve
    ):
        mock_ccs.side_effect = lambda *args: (
            (0.0, []) if args[-1] in {"zero_shot", "one_shot_base"} else (1.0, [])
        )
        adaptation_manager = _FakeAdaptationManager()
        config = {
            "TRANSFORMATION_DS_PROMPT_TEMPLATES": {
                "transformation_shallow": 2,
                "transformation_complete": 1,
            },
            "TRANSFORMATION_DS_TEMPERATURE": 0.7,
            "TRANSFORMATION_DS_CCS_N": 1,
            "QUESTION_PARALLEL_API_ENABLED": False,
            "API_PROVIDER_ADAPTATION": "avalai",
            "AVALAI_MODEL_NAME_ADAPTATION": "adaptation-model",
        }

        result = process_single_question(
            target={
                "index": 7,
                "question": "Target question",
                "ground_truth": "Target answer",
                "source": "hard_questions",
            },
            exemplar_data={
                "questions": ["Retrieved question"],
                "solutions": ["Retrieved solution"],
                "embeddings": [[1.0]],
            },
            embedding_model=object(),
            api_manager_adapt=adaptation_manager,
            api_manager_solve=object(),
            api_manager_eval=object(),
            config=config,
        )

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(len(adaptation_manager.calls), 3)
        self.assertEqual(
            [item["transformation_template"] for item in result["candidate_evaluations"]],
            [
                "transformation_shallow",
                "transformation_shallow",
                "transformation_complete",
            ],
        )
        self.assertEqual(
            [item["template_sample_index"] for item in result["candidate_evaluations"]],
            [0, 1, 0],
        )
        self.assertEqual(
            result["dataset_entry"]["metadata"]["transformation_template"],
            "transformation_shallow",
        )
        self.assertEqual(
            result["dataset_entry"]["metadata"]["selected_template_sample_index"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
