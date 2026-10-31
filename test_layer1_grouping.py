"""Offline checks of few-shot (group) candidates measured like Layer-1 candidates."""

import sys
import tempfile
import threading
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np


def _import_without_sentence_transformers(*modules):
    # Only sentence_transformers is stubbed: patch.dict(sys.modules) would drop
    # the src modules imported inside it, and grouping imports pipeline_steps
    # again when it cross-evaluates (see test_deferred_solve_resume.py).
    stubbed = "sentence_transformers" not in sys.modules
    if stubbed:
        sentence_transformers = ModuleType("sentence_transformers")
        sentence_transformers.SentenceTransformer = type("SentenceTransformer", (), {})
        sys.modules["sentence_transformers"] = sentence_transformers
    try:
        return [__import__(name, fromlist=["_"]) for name in modules]
    finally:
        if stubbed:
            del sys.modules["sentence_transformers"]


orchestration, error_handler, layer1_grouping, pipeline_steps, layer1_base_execution = (
    _import_without_sentence_transformers(
        "src.orchestration", "src.error_handler", "src.layer1_grouping",
        "src.pipeline_steps", "src.layer1_base_execution",
    )
)
from src import distributed_execution
from src.prompts import PROMPT_TEMPLATES

TARGET = "What is 3 + 4?"
QUESTIONS = [f"Exemplar question {i}?" for i in range(6)]
SOLUTIONS = [f"Exemplar solution {i}. Final Answer: {i}" for i in range(6)]
EXEMPLAR_DATA = {"questions": QUESTIONS, "solutions": SOLUTIONS,
                 "embeddings": np.zeros((6, 2), np.float32)}
RETRIEVED = [4, 2, 0, 3, 1]   # descending similarity
RETRIEVED_SET = [{"corpus_index": index, "question": QUESTIONS[index], "similarity_score": score}
                 for index, score in zip(RETRIEVED, (0.9, 0.8, 0.7, 0.6, 0.5))]
MULTI_SHOT = "mirror_hypothesis_gen_multi_shot_v1"


class FakeSolver:
    """Every answer is unique, so a mirror prompt names the candidate it verifies."""

    provider_name = "openrouter"

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def generate_content(self, prompt, model_name, temperature=None, avalai_role=None, **kwargs):
        with self.lock:
            self.calls.append((prompt, temperature))
            text = f"<<answer {len(self.calls)}>> Final Answer: 7"
        return {"status": "SUCCESS", "text": text}


def judge(model_answer, ground_truth, api_manager, config, target_index=None):
    return {"status": "SUCCESS", "is_correct": True}


def config(**overrides):
    return {
        "OPENROUTER_MODEL_NAME_FINAL_SOLVER": "solver-model",
        "API_PROVIDER_SOLVER": "openrouter",
        "API_PROVIDER_EVALUATOR": "openrouter",
        "API_PROVIDER_ADAPTATION": "openrouter",
        "TOP_N_CANDIDATES_RETRIEVAL": 5,
        "MIRROR_N_OPTIMIZATION": 2,
        "LAYER1_GROUP_SIZES": [2],
        "LAYER1_GROUPING_PROMPT_TEMPLATE": MULTI_SHOT,
        "LAYER1_GROUPING_MIRROR_EVALUATION": True,
        **overrides,
    }


class GroupCandidateTests(unittest.TestCase):
    def setUp(self):
        for module in (pipeline_steps, layer1_grouping, layer1_base_execution):
            patcher = patch.object(module, "evaluate_single_answer_with_llm", judge)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.solver = FakeSolver()

    def group(self, **overrides):
        return layer1_grouping.run_layer1_grouping(
            target_query_index=0, target_query=TARGET, ground_truth_answer="7",
            embedding_model=None, exemplar_data=EXEMPLAR_DATA, api_manager_solve=self.solver,
            api_manager_eval=object(), config=config(**overrides), retrieved_set=RETRIEVED_SET,
        )

    def test_a_one_example_group_prompt_is_the_one_shot_prompt(self):
        prompt = layer1_grouping.build_group_prompt(
            TARGET, [3], EXEMPLAR_DATA, {"LAYER1_GROUPING_PROMPT_TEMPLATE": MULTI_SHOT})
        one_shot = PROMPT_TEMPLATES["mirror_hypothesis_gen_v1"].format(
            exemplar_question=QUESTIONS[3], exemplar_solution=SOLUTIONS[3], target_query=TARGET)
        self.assertEqual(prompt, one_shot)

    def test_a_pair_prompt_holds_both_examples_in_the_one_shot_layout(self):
        prompt = layer1_grouping.build_group_prompt(
            TARGET, [4, 2], EXEMPLAR_DATA, {"LAYER1_GROUPING_PROMPT_TEMPLATE": MULTI_SHOT})
        examples = (f"question:\n{QUESTIONS[4]}\n\nSolution:\n{SOLUTIONS[4]}\n\n"
                    f"question:\n{QUESTIONS[2]}\n\nSolution:\n{SOLUTIONS[2]}")
        self.assertIn(f"<Solved Examples>\n{examples}\n</Solved Examples>", prompt)
        # The companion runs' default template keeps its own example layout.
        default = layer1_grouping.build_group_prompt(TARGET, [4, 2], EXEMPLAR_DATA, {})
        self.assertIn(f"Question: {QUESTIONS[4]}\nRationale and Answer: {SOLUTIONS[4]}", default)

    def test_every_pair_is_cross_evaluated_by_every_retrieved_sample(self):
        state = self.group()
        self.assertEqual(state["overall_status"], "SUCCESS")
        self.assertEqual(state["step_statuses"], {
            "retrieval": "REUSED", "group_generation": "SUCCESS",
            "ground_truth_evaluation": "SUCCESS", "cross_evaluation": "SUCCESS"})
        candidates = state["candidate_set"]
        self.assertEqual(len(candidates), 10)
        matrix = state["cross_evaluation_matrix"]
        self.assertEqual(set(matrix), set(candidates))
        for row in matrix.values():
            self.assertEqual(row, {index: 1.0 for index in RETRIEVED})
        # Generation at the one-shot temperature, then MIRROR_N_OPTIMIZATION
        # verification runs (temperature 1.0) per candidate and evaluator.
        generations = [call for call in self.solver.calls if call[1] == 0.0]
        self.assertEqual(len(generations), 10)
        for candidate in candidates.values():
            verifications = [prompt for prompt, temperature in self.solver.calls
                             if temperature == 1.0 and candidate["candidate_text"] in prompt]
            self.assertEqual(len(verifications), 5 * 2)
            for index in RETRIEVED:
                self.assertEqual(sum(QUESTIONS[index] in prompt for prompt in verifications), 2)
        self.assertEqual(len(self.solver.calls), 10 + 10 * 5 * 2)
        snapshot = state["metadata"]["config_snapshot"]
        self.assertTrue(snapshot["LAYER1_GROUPING_MIRROR_EVALUATION"])
        self.assertEqual(snapshot["MIRROR_N_OPTIMIZATION"], 2)

    def test_without_mirror_evaluation_the_state_is_unchanged(self):
        state = self.group(LAYER1_GROUPING_MIRROR_EVALUATION=False)
        self.assertEqual(state["overall_status"], "SUCCESS")
        self.assertNotIn("cross_evaluation_matrix", state)
        self.assertNotIn("cross_evaluation", state["step_statuses"])
        self.assertEqual(len(self.solver.calls), 10)

    def test_a_failed_cross_evaluation_leaves_a_partial_state(self):
        with patch.object(pipeline_steps, "_evaluate_mirror_consistency",
                          side_effect=RuntimeError("proxy down")):
            state = self.group()
        self.assertEqual(state["step_statuses"]["cross_evaluation"], "FAILURE")
        self.assertEqual(state["overall_status"], "PARTIAL")
        self.assertEqual(len(state["ground_truth_labels"]), 10)


class CombinedExperimentTests(unittest.TestCase):
    """Zero-, one-, and two-shot candidates in one experiment, as in the README example."""

    def setUp(self):
        for module in (pipeline_steps, layer1_grouping, layer1_base_execution):
            patcher = patch.object(module, "evaluate_single_answer_with_llm", judge)
            patcher.start()
            self.addCleanup(patcher.stop)
        retrieval = {"status": "SUCCESS", "retrieved_indices": RETRIEVED,
                     "retrieved_similarity_scores": [0.9, 0.8, 0.7, 0.6, 0.5], "trace": []}
        patcher = patch.object(layer1_base_execution, "retrieve", return_value=retrieval)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.solver = FakeSolver()

    def test_group_candidates_join_the_layer1_run_without_entering_its_candidate_set(self):
        run_config = config(
            experiment_name="combined", RESULTS_DIR=self.temporary.name,
            LAYER1_CACHE_DIR=self.temporary.name, DISTRIBUTED_EXECUTION_ENABLED=False,
            APPLY_LAYER1_BASE_EXECUTION=True, LAYER1_ONLY_MODE=True,
            LAYER1_ZERO_SHOT_CANDIDATES_N=3, LAYER1_ONE_SHOT_CANDIDATES_N=5,
            APPLY_LAYER1_GROUPING=True,
        )
        log = orchestration.run_pipeline_for_single_query(
            hard_list_idx=0, target_query=TARGET, config=run_config, embedding_model=None,
            exemplar_data=EXEMPLAR_DATA, api_managers={"openrouter": self.solver},
            hard_solutions=["7"],
        )
        self.assertEqual(log["execution_mode"], "layer1_and_grouping_only")
        self.assertEqual(log["pipeline_status"], "SUCCESS")
        layer1 = log["layer1_base_execution_state"]
        self.assertEqual({str(cid) for cid in layer1["candidate_set"]},
                         {"zs_0", "zs_1", "zs_2", *map(str, RETRIEVED)})
        grouping = log["layer1_grouping_state"]
        self.assertEqual(grouping["overall_status"], "SUCCESS")
        self.assertEqual([row["corpus_index"] for row in grouping["retrieved_set"]], RETRIEVED)
        self.assertEqual(len(grouping["cross_evaluation_matrix"]), 10)
        # Pairs use the one-shot candidates' evaluators and repeat count.
        self.assertEqual({key for row in grouping["cross_evaluation_matrix"].values() for key in row},
                         {key for row in layer1["cross_evaluation_matrix"].values() for key in row})
        nearest_pair = layer1_grouping.build_group_prompt(TARGET, [4, 2], EXEMPLAR_DATA, run_config)
        self.assertIn((nearest_pair, 0.0), self.solver.calls)
        # A generation retry rebuilds the experiment from these flags.
        flags = log["config_flags_used"]
        self.assertEqual(flags["LAYER1_ZERO_SHOT_CANDIDATES_N"], 3)
        self.assertEqual(flags["MIRROR_N_OPTIMIZATION"], 2)
        self.assertTrue(flags["LAYER1_GROUPING_MIRROR_EVALUATION"])


class IncompleteGroupingIsRetriedTests(unittest.TestCase):
    """A run that continues past grouping reports the standard pipeline's status."""

    def logs(self, grouping_status):
        return {"pipeline_status": "SUCCESS", "layer1_grouping_state": {"overall_status": grouping_status}}

    def test_generation_retry_targets_an_incomplete_grouping_state(self):
        self.assertTrue(error_handler._log_needs_generation_retry(self.logs("PARTIAL")))
        self.assertTrue(error_handler._log_needs_generation_retry(
            {"pipeline_status": "SUCCESS", "layer1_grouping_state": None}))
        self.assertFalse(error_handler._log_needs_generation_retry(self.logs("SUCCESS")))
        self.assertFalse(error_handler._log_needs_generation_retry({"pipeline_status": "SUCCESS"}))

    def test_a_distributed_worker_does_not_count_it_as_done(self):
        self.assertFalse(distributed_execution.run_log_successful(self.logs("FAILURE")))
        self.assertTrue(distributed_execution.run_log_successful(self.logs("SUCCESS")))
        self.assertTrue(distributed_execution.run_log_successful({"pipeline_status": "SUCCESS"}))


if __name__ == "__main__":
    unittest.main()
