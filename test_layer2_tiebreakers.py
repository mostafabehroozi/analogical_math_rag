import sys
from types import SimpleNamespace
import numpy as np

# Ensure the repo root is on sys.path for local imports.
sys.path.append('.')

from src.layer2_analysis import BlockC


class DummyPTUEngine:
    def compute_holistic_score(self, ptu_matrix, weight_taker, weight_maker):
        # simplest holistic score for test purposes
        return np.array([1.0, 2.0, 3.0])


def test_candidate_centric_tiebreaker_uses_current_pool():
    ptu_engine = DummyPTUEngine()
    config = SimpleNamespace(block_C_tiebreaker_weight_taker=1.0, block_C_tiebreaker_weight_maker=1.0)
    block_c = BlockC(ptu_engine, config)

    tied_candidates = [0, 1]
    score_take = np.array([100.0, 50.0])
    target_query_embedding_similarity = {0: 0.1, 1: 0.2}
    already_selected_candidates = {1}
    ptu_matrix = np.zeros((2, 2))

    selected = block_c._apply_hierarchical_tiebreaker_candidate_centric(
        tied_candidates,
        ptu_matrix,
        eval_idx=0,
        score_take=score_take,
        target_query_embedding_similarity=target_query_embedding_similarity,
        already_selected_candidates=already_selected_candidates,
    )

    assert selected == 1, f"Expected selected candidate 1, got {selected}"


def test_evaluator_centric_tiebreaker_uses_current_pool():
    ptu_engine = DummyPTUEngine()
    config = SimpleNamespace(block_C_tiebreaker_weight_taker=1.0, block_C_tiebreaker_weight_maker=1.0)
    block_c = BlockC(ptu_engine, config)

    tied_evaluators = [0, 1]
    score_make = np.array([10.0, 20.0])
    target_query_embedding_similarity = {0: 0.1, 1: 0.2}
    already_selected_evaluators = {1}
    ptu_matrix = np.zeros((2, 2))

    evaluator_max_counts = {
        0: 0,
        1: 0,
    }
    selected = block_c._apply_hierarchical_tiebreaker_evaluator_centric(
        tied_evaluators,
        ptu_matrix,
        cand_idx=0,
        score_make=score_make,
        target_query_embedding_similarity=target_query_embedding_similarity,
        evaluator_max_counts=evaluator_max_counts,
    )

    assert selected == 1, f"Expected selected evaluator 1, got {selected}"


def test_evaluator_centric_tiebreaker_uses_global_maximum_counts():
    ptu_engine = DummyPTUEngine()
    config = SimpleNamespace(block_C_tiebreaker_weight_taker=1.0, block_C_tiebreaker_weight_maker=1.0)
    block_c = BlockC(ptu_engine, config)

    tied_evaluators = [0, 1]
    score_make = np.array([10.0, 20.0])
    target_query_embedding_similarity = {0: 0.1, 1: 0.2}
    ptu_matrix = np.array([
        [1.0, 1.0],
        [0.0, 2.0],
        [0.0, 3.0],
    ])
    evaluator_max_counts = {
        0: 1,
        1: 3,
    }

    selected = block_c._apply_hierarchical_tiebreaker_evaluator_centric(
        tied_evaluators,
        ptu_matrix,
        cand_idx=0,
        score_make=score_make,
        target_query_embedding_similarity=target_query_embedding_similarity,
        evaluator_max_counts=evaluator_max_counts,
    )

    assert selected == 1, f"Expected selected evaluator 1 based on global counts, got {selected}"


if __name__ == '__main__':
    test_candidate_centric_tiebreaker_uses_current_pool()
    test_evaluator_centric_tiebreaker_uses_current_pool()
    print('Tie-breaker regression tests passed.')
