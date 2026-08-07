import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import src.transformation_dataset_builder as builder


class FakeManager:
    def __init__(self, response_factory):
        self.response_factory = response_factory
        self.calls = []

    def generate_content(self, prompt, model, temperature):
        self.calls.append((prompt, model, temperature))
        return self.response_factory(prompt)


def _config(**overrides):
    config = {
        "GEMINI_MODEL_NAME_ADAPTATION": "adapt-model",
        "GEMINI_MODEL_NAME_FINAL_SOLVER": "solve-model",
        "GEMINI_MODEL_NAME_EVALUATOR": "eval-model",
        "API_PROVIDER_ADAPTATION": "gemini",
        "API_PROVIDER_SOLVER": "gemini",
        "API_PROVIDER_EVALUATOR": "gemini",
        "TRANSFORMATION_DS_CCS_N": 4,
        "TRANSFORMATION_DS_N_CANDIDATES": 2,
        "TRANSFORMATION_DS_PROMPT_TEMPLATE": "transformation_v1",
        "TRANSFORMATION_DS_TEMPERATURE": 0.0,
        "TRANSFORMATION_DS_SOLVER_TEMPERATURE": 1.0,
        "QUESTION_PARALLEL_API_ENABLED": False,
    }
    config.update(overrides)
    return config


class TransformationDatasetBuilderTests(unittest.TestCase):
    def setUp(self):
        self.exemplar_data = {
            "questions": ["retrieved question"],
            "solutions": ["retrieved solution"],
            "embeddings": object(),
            "question_to_index": {},
        }

    def _run_process(self, answer_correct):
        def response_factory(prompt):
            if prompt.startswith("transform-0"):
                return {"status": "SUCCESS", "text": "transformed-0"}
            if prompt.startswith("transform-1"):
                return {"status": "SUCCESS", "text": "transformed-1"}
            return {"status": "SUCCESS", "text": prompt}

        manager = FakeManager(response_factory)

        def evaluate(answer, *_args, **_kwargs):
            return {"status": "SUCCESS", "is_correct": answer_correct(answer)}

        transform_counter = {"value": 0}

        def transformation_prompt(target, _sample, _config, _key):
            index = transform_counter["value"]
            transform_counter["value"] += 1
            return f"transform-{index}"

        def final_prompt(_target, examples, _config):
            text = examples[0]
            if "transformed-0" in text:
                return "aug-0"
            if "transformed-1" in text:
                return "aug-1"
            return "one"

        with mock.patch.object(builder, "GeminiAPIManager", FakeManager), \
             mock.patch.object(builder, "AvalAIAPIManager", FakeManager), \
             mock.patch.object(builder, "OllamaAPIManager", FakeManager), \
             mock.patch.object(builder, "retrieve", return_value={
                 "status": "SUCCESS",
                 "retrieved_indices": [0],
                 "retrieved_similarity_scores": [0.91],
             }), \
             mock.patch.object(builder, "create_transformation_prompt", side_effect=transformation_prompt), \
             mock.patch.object(builder, "create_final_reasoning_prompt_simple", return_value="zero"), \
             mock.patch.object(builder, "create_final_reasoning_prompt", side_effect=final_prompt), \
             mock.patch.object(builder, "evaluate_single_answer_with_llm", side_effect=evaluate):
            result = builder.process_single_question(
                target={"index": 0, "question": "target", "ground_truth": "answer", "source": "hard_questions"},
                exemplar_data=self.exemplar_data,
                embedding_model=object(),
                api_manager_adapt=manager,
                api_manager_solve=manager,
                api_manager_eval=manager,
                config=_config(),
            )
        return result, manager

    def test_selects_highest_candidate_and_stores_prompt_output(self):
        eval_counts = {"aug-0": 0}

        def correct(answer):
            if answer == "aug-0":
                eval_counts["aug-0"] += 1
                return eval_counts["aug-0"] <= 2
            return answer == "aug-1"

        result, _manager = self._run_process(correct)

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["selected_candidate_index"], 1)
        self.assertEqual(result["zero_shot_base_ccs"], 0.0)
        self.assertEqual(result["one_shot_base_ccs"], 0.0)
        self.assertEqual(result["selected_transformed_aug_ccs"], 1.0)
        self.assertEqual(result["dataset_entry"]["input"], "transform-1")
        self.assertEqual(result["dataset_entry"]["output"], "transformed-1")
        self.assertEqual(result["dataset_entry"]["metadata"]["retrieved_index"], 0)

    def test_strict_gate_rejects_candidate_equal_to_one_shot_baseline(self):
        def correct(answer):
            return answer in {"one", "aug-0", "aug-1"}

        result, _manager = self._run_process(correct)

        self.assertEqual(result["status"], "REJECTED_BY_FILTER")
        self.assertIsNone(result["dataset_entry"])
        self.assertEqual(result["one_shot_base_ccs"], 1.0)

    def test_failed_candidate_is_skipped(self):
        def response_factory(prompt):
            if prompt.startswith("transform-0"):
                return {"status": "FAILURE", "error_message": "temporary failure"}
            if prompt.startswith("transform-1"):
                return {"status": "SUCCESS", "text": "transformed-1"}
            return {"status": "SUCCESS", "text": prompt}

        manager = FakeManager(response_factory)
        transform_counter = {"value": 0}

        def transformation_prompt(*_args):
            index = transform_counter["value"]
            transform_counter["value"] += 1
            return f"transform-{index}"

        with mock.patch.object(builder, "GeminiAPIManager", FakeManager), \
             mock.patch.object(builder, "AvalAIAPIManager", FakeManager), \
             mock.patch.object(builder, "OllamaAPIManager", FakeManager), \
             mock.patch.object(builder, "retrieve", return_value={"status": "SUCCESS", "retrieved_indices": [0]}), \
             mock.patch.object(builder, "create_transformation_prompt", side_effect=transformation_prompt), \
             mock.patch.object(builder, "create_final_reasoning_prompt_simple", return_value="zero"), \
             mock.patch.object(builder, "create_final_reasoning_prompt", return_value="aug-1"), \
             mock.patch.object(builder, "evaluate_single_answer_with_llm", return_value={"status": "SUCCESS", "is_correct": True}):
            result = builder.process_single_question(
                target={"index": 0, "question": "target", "ground_truth": "answer", "source": "hard_questions"},
                exemplar_data=self.exemplar_data,
                embedding_model=object(),
                api_manager_adapt=manager,
                api_manager_solve=manager,
                api_manager_eval=manager,
                config=_config(),
            )

        self.assertEqual(result["status"], "REJECTED_BY_FILTER")
        self.assertEqual(result["candidate_evaluations"][0]["status"], "FAILED")
        self.assertEqual(result["candidate_evaluations"][1]["status"], "EVALUATED")

    def test_builder_resumes_completed_targets_without_duplicate_work(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(
                RESULTS_DIR=temp_dir,
                experiment_name="transformation_resume",
                TRANSFORMATION_DS_OUTPUT_FILENAME="transformed.json",
            )
            fake_entry = {
                "input": "prompt",
                "output": "transformed",
                "metadata": {"original_index": 0},
            }
            fake_process = mock.Mock(return_value={
                "status": "SUCCESS",
                "original_index": 0,
                "dataset_entry": fake_entry,
            })
            with mock.patch.object(builder, "process_single_question", fake_process), \
                 mock.patch.object(builder, "periodic_sync_check"):
                first = builder.build_transformation_dataset(
                    ["target"], ["answer"], self.exemplar_data, object(), {}, config
                )
                second = builder.build_transformation_dataset(
                    ["target"], ["answer"], self.exemplar_data, object(), {}, config
                )

            self.assertEqual(fake_process.call_count, 1)
            self.assertEqual(first, second)
            saved = json.loads(Path(temp_dir, "transformed.json").read_text())
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["metadata"]["original_index"], 0)


if __name__ == "__main__":
    unittest.main()
