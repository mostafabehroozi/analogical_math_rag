from unittest.mock import patch

import numpy as np
import pytest

from src.layer1_grouping import (
    build_group_prompt,
    enumerate_retrieval_groups,
    normalize_group_sizes,
    run_layer1_grouping,
)


def test_group_sizes_and_all_combinations_are_deterministic():
    retrieved = [{"corpus_index": index} for index in range(4)]
    sizes = normalize_group_sizes([3, 2, 2], len(retrieved))
    assert sizes == [2, 3]
    groups = enumerate_retrieval_groups(retrieved, sizes)
    assert groups == [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3),
    ]


@pytest.mark.parametrize("value", [[], [1], [4], "2", [True]])
def test_invalid_group_sizes_fail_before_api_calls(value):
    with pytest.raises(ValueError):
        normalize_group_sizes(value, 3)


def test_group_prompt_contains_exact_selected_examples():
    exemplar_data = {
        "questions": ["Q0", "Q1", "Q2"],
        "solutions": ["S0", "S1", "S2"],
    }
    prompt = build_group_prompt(
        "TARGET",
        [0, 2],
        exemplar_data,
        {"LAYER1_GROUPING_PROMPT_TEMPLATE": "layer1_grouping_solver_v1"},
    )
    assert "Q0" in prompt and "S0" in prompt
    assert "Q2" in prompt and "S2" in prompt
    assert "Q1" not in prompt and "S1" not in prompt
    assert "TARGET" in prompt


class _SolveManager:
    def __init__(self):
        self.prompts = []

    def generate_content(self, prompt, model, temperature, avalai_role):
        self.prompts.append(prompt)
        return {"status": "SUCCESS", "text": f"answer-{len(self.prompts)}"}


def test_grouping_runner_makes_one_candidate_per_combination_without_ccs():
    retrieved = [
        {"corpus_index": 0, "question": "Q0", "similarity_score": 0.9},
        {"corpus_index": 1, "question": "Q1", "similarity_score": 0.8},
        {"corpus_index": 2, "question": "Q2", "similarity_score": 0.7},
    ]
    config = {
        "LAYER1_GROUP_SIZES": [2, 3],
        "LAYER1_GROUPING_PROMPT_TEMPLATE": "layer1_grouping_solver_v1",
        "LAYER1_GROUPING_TEMPERATURE": 0.0,
        "API_PROVIDER_SOLVER": "gemini",
        "GEMINI_MODEL_NAME_FINAL_SOLVER": "solver",
        "QUESTION_PARALLEL_API_ENABLED": False,
    }
    manager = _SolveManager()
    with patch(
        "src.layer1_grouping.evaluate_single_answer_with_llm",
        return_value={"is_correct": True, "status": "SUCCESS"},
    ):
        state = run_layer1_grouping(
            target_query_index=7,
            target_query="TARGET",
            ground_truth_answer="GT",
            embedding_model=object(),
            exemplar_data={
                "questions": ["Q0", "Q1", "Q2"],
                "solutions": ["S0", "S1", "S2"],
                "embeddings": np.zeros((3, 2)),
            },
            api_manager_solve=manager,
            api_manager_eval=object(),
            config=config,
            retrieved_set=retrieved,
        )

    assert state["overall_status"] == "SUCCESS"
    assert len(state["candidate_set"]) == 4
    assert len(manager.prompts) == 4
    assert set(state) >= {"candidate_set", "ground_truth_labels", "retrieved_set"}
    assert "cross_evaluation_matrix" not in state
    assert state["candidate_set"]["group_s2_p0-1"]["similarity_sum"] == pytest.approx(1.7)
