import copy

import pytest

from layer1_grouping_context_selector import build_question_options


def _records():
    retrieved = [
        {"corpus_index": 10, "similarity_score": 0.9},
        {"corpus_index": 11, "similarity_score": 0.8},
        {"corpus_index": 12, "similarity_score": 0.7},
    ]
    layer1_state = {
        "target_query_data": {"query_text": "target"},
        "retrieved_set": copy.deepcopy(retrieved),
        "candidate_set": {
            "zs_0": {"candidate_text": "zero", "source_exemplar_idx": -1},
            "one10": {"candidate_text": "one10", "source_exemplar_idx": 10},
            "one11": {"candidate_text": "one11", "source_exemplar_idx": 11},
            "one12": {"candidate_text": "one12", "source_exemplar_idx": 12},
        },
        "ground_truth_labels": {
            "zs_0": {"is_correct": False},
            "one10": {"is_correct": False},
            "one11": {"is_correct": True},
            "one12": {"is_correct": True},
        },
        "intrinsic_baselines": {"10": 0.1, "11": 0.2, "12": 0.3},
        "cross_evaluation_matrix": {
            "zs_0": {"10": 0.1, "11": 0.2, "12": 0.3},
            "one10": {"10": 0.4, "11": 0.5, "12": 0.6},
            "one11": {"10": 0.5, "11": 0.6, "12": 0.7},
            "one12": {"10": 0.6, "11": 0.7, "12": 0.8},
        },
    }
    grouping_state = {
        "target_query_data": {"query_text": "target"},
        "retrieved_set": copy.deepcopy(retrieved),
        "candidate_set": {
            "g01": {
                "candidate_text": "group01",
                "source_exemplar_indices": [10, 11],
                "source_retrieval_positions": [0, 1],
            },
            "g02": {
                "candidate_text": "group02",
                "source_exemplar_indices": [10, 12],
                "source_retrieval_positions": [0, 2],
            },
        },
        "ground_truth_labels": {
            "g01": {"is_correct": True},
            "g02": {"is_correct": True},
        },
    }
    return (
        {"state": layer1_state, "query_text": "target"},
        {"state": grouping_state, "query_text": "target"},
    )


def test_teacher_uses_smallest_correct_size_then_highest_similarity_sum():
    layer1, grouping = _records()
    built = build_question_options(layer1, grouping, max_k=3)
    teacher = built["options"][built["teacher_index"]]
    assert teacher["option_id"] == "one::one11"
    assert len(teacher["features"]) == 7 + 4 * 3


def test_teacher_uses_nearest_group_when_only_groups_are_correct():
    layer1, grouping = _records()
    layer1["state"]["ground_truth_labels"]["one11"]["is_correct"] = False
    layer1["state"]["ground_truth_labels"]["one12"]["is_correct"] = False
    built = build_question_options(layer1, grouping, max_k=3)
    teacher = built["options"][built["teacher_index"]]
    assert teacher["option_id"] == "group::g01"
    assert teacher["similarity_sum"] == pytest.approx(1.7)


def test_zero_shot_is_teacher_when_correct_or_when_everything_is_wrong():
    layer1, grouping = _records()
    layer1["state"]["ground_truth_labels"]["zs_0"]["is_correct"] = True
    built = build_question_options(layer1, grouping, max_k=3)
    assert built["teacher_index"] == 0

    for label in layer1["state"]["ground_truth_labels"].values():
        label["is_correct"] = False
    for label in grouping["state"]["ground_truth_labels"].values():
        label["is_correct"] = False
    built = build_question_options(layer1, grouping, max_k=3)
    assert built["teacher_index"] == 0
    assert built["has_correct_option"] is False


def test_first_answered_zero_shot_skips_failed_first_generation():
    layer1, grouping = _records()
    layer1["state"]["candidate_set"]["zs_0"]["candidate_text"] = None
    layer1["state"]["candidate_set"]["zs_1"] = {
        "candidate_text": "second zero",
        "source_exemplar_idx": -1,
    }
    layer1["state"]["ground_truth_labels"]["zs_1"] = {"is_correct": False}
    layer1["state"]["cross_evaluation_matrix"]["zs_1"] = {}
    built = build_question_options(layer1, grouping, max_k=3)
    assert built["options"][0]["option_id"] == "zero::zs_1"
