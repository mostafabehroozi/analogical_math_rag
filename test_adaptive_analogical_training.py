"""Offline checks for the three supervised adaptive acquisition stages."""
import ast
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

import adaptive_analogical_training as a


def raw_record(index, cfg, prefix="train", all_wrong=False):
    rng = np.random.RandomState(index + 810)
    eids = [str(100+j) for j in range(cfg.k)]
    cids = [f"zs_{i}" for i in range(cfg.zero_shots)] + eids
    labels = rng.randint(0, 2, cfg.n)
    if all_wrong:
        labels[:] = 0
    candidates = {cid: {"source_exemplar_idx": -1 if i < cfg.zero_shots else int(cid),
                        "candidate_text": f"Reasoning {i}", "generation_status": "SUCCESS"}
                  for i, cid in enumerate(cids)}
    state = {
        "target_query_data": {"query_text": f"{prefix} unique mathematical question {index}"},
        "retrieved_set": [{"corpus_index": int(e), "similarity_score": .9-.1*j}
                          for j,e in enumerate(eids)],
        "candidate_set": candidates,
        "ground_truth_labels": {cid: {"is_correct": bool(labels[i]), "evaluation_status": "SUCCESS"}
                                for i,cid in enumerate(cids)},
        "intrinsic_baselines": {e: float(rng.randint(cfg.repeats+1)/cfg.repeats)
                                for e in eids},
        "cross_evaluation_matrix": {cid: {e: float(rng.randint(cfg.repeats+1)/cfg.repeats)
                                          for e in eids} for cid in cids},
    }
    return {"target_query_original_hard_list_idx": index,
            "target_query_text": state["target_query_data"]["query_text"],
            "layer1_base_execution_state": state}


def record(cfg, index=0, all_wrong=False):
    return a.parse_record(raw_record(index, cfg, all_wrong=all_wrong), index, "train", cfg)


def fake_predictor(cfg):
    torch.set_num_threads(1)
    a.seed_everything(7)
    model = a.ResNet(cfg.input_dim, 3*cfg.n+2, cfg)
    return a.FrozenPredictor({"weights": a.cpu_state(model),
                              "temperatures": np.ones(5)}, cfg, torch.device("cpu"))


def test_retrieval_sorted_by_similarity_and_safe_prefix():
    cfg = a.Config()
    raw = raw_record(0, cfg)
    raw["layer1_base_execution_state"]["retrieved_set"].reverse()
    r = a.parse_record(raw, 0, "train", cfg)
    assert np.all(np.diff(r.similarity) <= 0)
    assert r.evaluator_ids == [str(100+i) for i in range(cfg.k)]
    scores = np.arange(cfg.n, 0, -1, dtype=np.float32)
    truth = np.array([1, 1, 0, 1, 1, 1, 1, 1], np.float32)
    safe, maximum = a.teacher_labels(scores, truth, cfg)
    assert np.flatnonzero(safe).tolist() == [0, 1]
    assert np.flatnonzero(maximum).tolist() == [0]
    truth[0] = 0
    safe, maximum = a.teacher_labels(scores, truth, cfg)
    assert not safe.any() and not maximum.any()


def test_retrieval_ties_keep_log_order_and_missing_similarity_is_auditable():
    cfg = a.Config()
    raw = raw_record(0, cfg)
    samples = raw["layer1_base_execution_state"]["retrieved_set"]
    samples.reverse()
    for sample in samples:
        sample["similarity_score"] = .8
    r = a.parse_record(raw, 0, "train", cfg)
    assert r.evaluator_ids == [str(sample["corpus_index"]) for sample in samples]
    del samples[0]["similarity_score"]
    with pytest.raises(ValueError, match="missing_or_invalid_measurement"):
        a.parse_record(raw, 0, "train", cfg)


def test_initial_state_retrieval_order_one_shot_dependency_and_cost():
    cfg = a.Config()
    assert cfg.full_cost == 233
    assert a.state_cost(a.State(), cfg) == 1
    assert len(a.reachable_states(cfg)) == 729
    assert np.flatnonzero(a.valid_actions(a.State(), cfg)).tolist() == [0, 1, 2, 3, 4, 5]
    first = a.advance(a.State(), 0, cfg)
    assert first.evaluator_mask == 1
    assert 6 in np.flatnonzero(a.valid_actions(first, cfg))
    with pytest.raises(ValueError, match="Unavailable"):
        a.advance(a.State(), 6, cfg)
    assert a.fixed_order_actions(cfg) == [0, 6, 5, 1, 7, 5, 2, 8, 3, 9, 4, 10]


def test_exhaustive_catalog_matches_independent_enumerator_and_transition_graph():
    cfg = a.Config()
    independent = {(candidate_mask, evaluator_mask)
                   for candidate_mask in range(1, 1 << cfg.n)
                   for evaluator_mask in range(1 << cfg.k)
                   if not candidate_mask >> cfg.zero_shots & ~evaluator_mask}
    states, graph = a.structural_graph(cfg.zero_shots, cfg.k)
    assert len(independent) == len(states) == 1912
    assert {(s.candidate_mask, s.evaluator_mask) for s in states} == independent
    assert len(states) * 6 == 11472
    for state, edges in zip(states, graph):
        assert len(edges) == len(set(edges.values()))
        for action, successor in edges.items():
            nxt = states[successor]
            assert nxt == a.raw_next_state(state, action, cfg)
            assert nxt.candidate_mask | state.candidate_mask == nxt.candidate_mask
            assert nxt.evaluator_mask | state.evaluator_mask == nxt.evaluator_mask


def test_arbitrary_evaluator_subsets_zero_shot_and_exact_budget_cost():
    cfg = a.Config(k=3, zero_shots=3, max_cost=17)
    state = a.State(1, 4)  # ZS1 with only R3 active.
    assert a.state_cost(state, cfg) == 11
    assert np.flatnonzero(a.valid_actions(state, cfg)).tolist() == [3, 6]
    zs = a.advance(state, 3, cfg)
    os = a.advance(state, 6, cfg)
    assert zs == a.State(3, 4) and os == a.State(1 | (1 << 5), 4)
    snapshot = a.as_snapshot_state(os, cfg)
    assert snapshot.baseline_attempted == snapshot.baseline_observed == 4
    assert snapshot.ccs_attempted == snapshot.ccs_observed == (4 | (4 << (5*cfg.k)))
    with pytest.raises(ValueError, match="without retrieved source"):
        a.as_snapshot_state(a.State(1 << 5, 0), cfg)
    assert a.state_cost(zs, cfg) - a.state_cost(state, cfg) == 6
    assert a.state_cost(os, cfg) - a.state_cost(state, cfg) == 6
    with pytest.raises(ValueError, match="Unavailable"):
        a.advance(state, 4, cfg)  # R1 is not active.
    cfg.max_cost = 16
    assert not a.valid_actions(state, cfg).any()
    assert a.state_cost(a.State((1 << a.Config().n)-1, (1 << a.Config().k)-1), a.Config()) == 233


def test_independent_masks_include_training_only_hidden_source():
    cfg = a.Config(candidate_mask_fraction=0, evaluator_mask_fraction=1,
                   hidden_source_fraction=1)
    r = record(cfg)
    state = a.complete_snapshot_state((1 << cfg.n)-1, (1 << cfg.k)-1,
                                       (1 << cfg.k)-1, cfg)
    hidden = a.augment_structure(state, cfg, np.random.RandomState(3))
    assert hidden.retrieved == hidden.evaluators == 0
    assert hidden.candidates & (1 << cfg.zero_shots)
    with pytest.raises(ValueError, match="without retrieved source"):
        a.validate_snapshot_state(hidden, cfg)
    x = a.observation(r, hidden, cfg, allow_hidden_source=True)
    assert x.shape == (cfg.input_dim,)
    assert np.isfinite(x).all()
    assert hidden.ccs_observed == 0 and hidden.baseline_observed == 0


def test_first_and_second_training_share_full_and_augmented_inputs():
    cfg = a.Config(snapshots_per_query=6, candidate_mask_fraction=.5,
                   evaluator_mask_fraction=.5)
    r = record(cfg)
    teacher = {"safe": np.ones((1, cfg.n), np.float32),
               "maximum": np.eye(1, cfg.n, 0).astype(np.float32)}
    first_x, first_y = a.build_snapshots([r], [0], None, cfg, 6, 1, augmented=True)
    second_x, second_y = a.build_snapshots([r], [0], teacher, cfg, 6, 1, augmented=True)
    full = a.observation(r, a.State((1 << cfg.n)-1, (1 << cfg.k)-1), cfg)
    assert first_x.shape == second_x.shape == (6, cfg.input_dim)
    assert first_y.shape == (6, 2*cfg.n)
    assert second_y.shape == (6, 4*cfg.n+2)
    assert np.array_equal(first_x[0], full)
    assert np.array_equal(second_x[0], full)
    assert np.array_equal(first_x, second_x)
    assert np.array_equal(first_y[0, cfg.n:], np.ones(cfg.n))


@pytest.mark.parametrize("count,batch", [(3, 2), (9, 4), (129, 128)])
def test_training_batches_keep_every_example_without_singletons(count, batch):
    x = np.arange(count, dtype=np.float32)[:, None]
    batches = list(a.make_loader(x, x.copy(), batch))
    assert min(len(bx) for bx, _ in batches) >= 2
    assert sorted(torch.cat([bx[:, 0] for bx, _ in batches]).tolist()) == list(range(count))


def test_reference_score_ties_preserve_max_identity_after_slot_permutation():
    cfg = a.Config(k=1, zero_shots=2)
    r = record(cfg)
    scores = np.array([.8, .8, .1])
    safe, maximum = a.teacher_labels(scores, np.array([1, 0, 0]), cfg)
    shuffled, _, _, new_max = a.permute_zero_shots(r, scores, safe, maximum, (1, 0), cfg)
    order = a.permuted_reference_order(scores, (1, 0), cfg)
    assert shuffled.candidate_ids[order[0]] == r.candidate_ids[0]
    assert new_max[order[0]] == 1
    # With no acquired evaluator, the head has no evidence about candidate identity.
    assert a.visible_state_key(r, a.State(), cfg) == a.visible_state_key(shuffled, a.State(), cfg)


def test_slot_permutation_keeps_measurements_provenance_and_labels_aligned():
    cfg = a.Config()
    r = record(cfg)
    safe = np.arange(cfg.n, dtype=np.float32)
    maximum = np.arange(cfg.n, dtype=np.float32) + 10
    state = a.complete_snapshot_state((1 << cfg.n)-1, (1 << cfg.k)-1,
                                       (1 << cfg.k)-1, cfg)
    permuted, new_state, new_safe, new_max = a.permute_snapshot(
        r, state, safe, maximum, cfg, np.random.RandomState(19))
    original_candidates = {cid: i for i, cid in enumerate(r.candidate_ids)}
    original_evaluators = {eid: j for j, eid in enumerate(r.evaluator_ids)}
    for i, cid in enumerate(permuted.candidate_ids):
        source_i = original_candidates[cid]
        assert permuted.labels[i] == r.labels[source_i]
        assert new_safe[i] == safe[source_i] and new_max[i] == maximum[source_i]
        for j, eid in enumerate(permuted.evaluator_ids):
            assert permuted.ccs[i,j] == r.ccs[source_i, original_evaluators[eid]]
    assert a.observation(permuted, new_state, cfg).shape == (cfg.input_dim,)


def test_hidden_future_measurements_and_labels_do_not_change_observation_or_action():
    cfg = a.Config(device="cpu", hidden_dim=8, residual_blocks=1)
    r, state = record(cfg), a.State()
    changed = copy.deepcopy(r)
    changed.ccs[:] = .123
    changed.baseline[:] = .8
    changed.similarity[1:] = -.9
    changed.labels[:] = 1 - changed.labels
    assert np.array_equal(a.observation(r, state, cfg), a.observation(changed, state, cfg))
    predictor = fake_predictor(cfg)
    h1, p1 = predictor.predict(r, state)
    h2, p2 = predictor.predict(changed, state)
    assert np.array_equal(h1, h2) and np.array_equal(p1, p2)
    head = a.SupervisedActionHead(cfg)
    valid = a.valid_actions(state, cfg)
    assert a.greedy_action(head, h1, valid, "cpu") == a.greedy_action(head, h2, valid, "cpu")


def test_no_max_fallback_and_strict_optional_later_filter():
    cfg = a.Config(k=1, zero_shots=2)
    order = [0, 1, 2]
    p = np.zeros(3*cfg.n+2, np.float32)
    full = np.array([.3, .7, .5], np.float32)
    state = a.State(1, 0)
    empty = np.zeros(cfg.n, np.float32)
    p[0] = .7
    assert not a.goal_condition(0, order, empty, empty, p, full, state, cfg)
    p[0] = .70001
    assert a.goal_condition(0, order, empty, empty, p, full, state, cfg)
    present = a.State(3, 0)
    p[1] = .5
    assert not a.goal_condition(1, order, empty, empty, p, full, present, cfg)
    disabled = copy.deepcopy(cfg)
    disabled.later_rank_filter = False
    assert a.goal_condition(1, order, empty, empty, p, full, present, disabled)
    p[0] = .2
    assert not a.goal_condition(0, order, empty, empty, p, full, state, disabled)


def test_recognition_can_require_evaluator_or_prerequisite_path():
    cfg = a.Config(k=1, zero_shots=2, hidden_dim=4)
    r = record(cfg)
    class EvaluatorRecognition:
        def predict(self, rec, state, cfg_override=None):
            p = np.zeros(3*cfg.n+2, np.float32)
            if state.evaluator_mask:
                p[[cfg.n, 2*cfg.n, 3*cfg.n, 3*cfg.n+1]] = .9
            return np.zeros(cfg.hidden_dim, np.float32), p
        def predict_many(self, rec, states, allow_hidden_source=False):
            values = [self.predict(rec, state) for state in states]
            return np.stack([x[0] for x in values]), np.stack([x[1] for x in values])
    states = a.structural_states(cfg.zero_shots, cfg.k)
    safe = maximum = np.array([1, 0, 0], np.float32)
    scenario = a.decision_scenario(r, np.array([.9, .8, .7]), safe, maximum,
                                   EvaluatorRecognition(), cfg, states)
    root = next(j for j, state in enumerate(states) if state == a.State())
    assert not scenario["goal"][root, 0]  # Presence alone is insufficient.
    assert scenario["goal"][scenario["index"][a.State(1, 1)], 0]
    rows, cases = a.aggregate_decision_rows([scenario], cfg)
    case = next(c for c in cases if c["candidate_mask"] == 1 and c["evaluator_mask"] == 0)
    assert case["status"] == "ACTION"
    assert rows[case["training_row"]]["target"][0] == 1  # Only R1 is needed.

    # Rank one is now the one-shot candidate, so source acquisition precedes generation.
    one_shot_safe = one_shot_max = np.array([0, 0, 1], np.float32)
    class OneShotRecognition(EvaluatorRecognition):
        def predict(self, rec, state, cfg_override=None):
            p = np.zeros(3*cfg.n+2, np.float32)
            if state.candidate_mask & 4 and state.evaluator_mask:
                p[[cfg.n+2, 2*cfg.n+2, 3*cfg.n, 3*cfg.n+1]] = .9
            return np.zeros(cfg.hidden_dim, np.float32), p
    scenario = a.decision_scenario(r, np.array([.8, .7, .9]), one_shot_safe,
                                   one_shot_max, OneShotRecognition(), cfg, states)
    rows, cases = a.aggregate_decision_rows([scenario], cfg)
    case = next(c for c in cases if c["candidate_mask"] == 1 and c["evaluator_mask"] == 0)
    assert case["status"] == "ACTION"
    assert rows[case["training_row"]]["target"][0] == 1
    assert not a.valid_actions(a.State(), cfg)[2]
    assert a.valid_actions(a.State(1, 1), cfg)[2]


def test_explicit_status_precedence_and_case_reconciliation():
    cfg = a.Config(k=1, zero_shots=2, hidden_dim=4)
    r = record(cfg)
    scores = np.array([.9, .7, .3])
    empty = np.zeros(cfg.n, np.float32)
    class Scripted:
        def __init__(self, values):
            self.values = values
        def predict(self, rec, state, cfg_override=None):
            p = np.zeros(3*cfg.n+2, np.float32)
            for i in range(cfg.n):
                if state.candidate_mask & (1 << i):
                    p[i] = self.values[i]
            return np.zeros(cfg.hidden_dim, np.float32), p
        def predict_many(self, rec, states, allow_hidden_source=False):
            values = [self.predict(rec, state) for state in states]
            return np.stack([x[0] for x in values]), np.stack([x[1] for x in values])
    rows, cases = a.build_decision_rows(r, scores, empty, empty, Scripted([.9, .7, .3]), cfg)
    assert len(cases) == 20
    assert a.Counter(c["status"] for c in cases)["COMPLETE"] > 0
    rank_by_mask = {c["candidate_mask"]: c for c in cases
                    if c["scenario"] == 0 and c["evaluator_mask"] == 0}
    assert rank_by_mask[1]["goal_rank"] == 1 and rank_by_mask[1]["status"] == "ACTION"
    assert rank_by_mask[3]["goal_rank"] == 2 and rank_by_mask[3]["status"] == "ACTION"
    assert next(c for c in cases if c["scenario"] == 0 and c["candidate_mask"] == 7
                and c["evaluator_mask"] == 1)["status"] == "COMPLETE"
    assert all((c["training_row"] is not None) == (c["status"] == "ACTION") for c in cases)
    assert all(np.isclose(row["target"].sum(), 1) and not row["target"][~row["valid"]].any()
               for row in rows)
    rows, cases = a.build_decision_rows(r, scores, empty, empty, Scripted([0, 0, 0]), cfg)
    assert a.Counter(c["status"] for c in cases)["EXHAUSTED"] == 2
    assert a.Counter(c["status"] for c in cases)["UNREACHABLE"] > 0
    assert not rows
    cfg.max_cost = 1
    _, cases = a.build_decision_rows(r, scores, empty, empty, Scripted([0, 0, 0]), cfg)
    counts = a.Counter(c["status"] for c in cases)
    assert counts["OUT_OF_BUDGET"] > 0 and counts["BUDGET"] > 0
    assert counts["EXHAUSTED"] == 0  # Full state exceeds the budget first.
    assert sum(counts.values()) == 20


def test_max_requires_all_four_correct_lights_and_previous_goal_is_preserved():
    cfg = a.Config(k=1, zero_shots=2)
    order = [0, 1, 2]
    safe = np.array([1, 0, 0], np.float32)
    maximum = np.array([1, 0, 0], np.float32)
    p = np.zeros(3*cfg.n+2, np.float32)
    for col in (cfg.n, 2*cfg.n, 3*cfg.n, 3*cfg.n+1):
        p[col] = .5
    assert a.goal_condition(0, order, safe, maximum, p, p, a.State(), cfg)
    p[2*cfg.n] = .4999
    assert not a.goal_condition(0, order, safe, maximum, p, p, a.State(), cfg)

    class Scripted:
        device = torch.device("cpu")
        def predict(self, rec, state, cfg_override=None):
            q = np.zeros(3*cfg.n+2, np.float32)
            if state.evaluator_mask:
                q[[cfg.n, 2*cfg.n, 3*cfg.n, 3*cfg.n+1]] = .9
            if state.candidate_mask & 2:
                q[1] = .8
            if state.candidate_mask not in (1, 7):
                q[2*cfg.n] = .1  # Either intermediate addition loses MAX; full state restores it.
            return np.zeros(cfg.hidden_dim, np.float32), q
        def predict_many(self, rec, states, allow_hidden_source=False):
            values = [self.predict(rec, state) for state in states]
            return np.stack([v[0] for v in values]), np.stack([v[1] for v in values])
    r = record(cfg)
    states = a.reachable_states(cfg)
    scenario = a.decision_scenario(
        r, np.array([.9,.8,.7]), safe, maximum, Scripted(), cfg, states)
    after_first = states.index(a.State(1, 1))
    assert scenario["goal"][after_first].tolist() == [True, False, False]
    assert scenario["goal"][states.index(a.State(7, 1)), 0]
    rows, cases = a.aggregate_decision_rows([scenario], cfg)
    assert next(case for case in cases if case["candidate_mask"] == 1 and case["evaluator_mask"] == 1)["status"] == "UNREACHABLE"


def test_planner_shares_future_actions_until_zero_shot_evidence_is_observed():
    cfg = a.Config(k=1, zero_shots=2, repeats=1, hidden_dim=4)
    r = record(cfg)
    r.ccs[:, 0] = [1, 0, 0]
    states = a.structural_states(cfg.zero_shots, cfg.k)
    scores = np.array([.9, .7, .3])
    safe = maximum = np.array([1, 0, 0], np.float32)

    class Scripted:
        def predict(self, rec, state, cfg_override=None):
            p = np.zeros(3*cfg.n+2, np.float32)
            if state.evaluator_mask:
                for i in range(cfg.n):
                    if state.candidate_mask & (1 << i) and rec.ccs[i, 0] == 1:
                        p[[cfg.n+i, 2*cfg.n+i, 3*cfg.n, 3*cfg.n+1]] = .9
            return np.zeros(cfg.hidden_dim, np.float32), p
        def predict_many(self, rec, states, allow_hidden_source=False):
            values = [self.predict(rec, state) for state in states]
            return np.stack([v[0] for v in values]), np.stack([v[1] for v in values])

    scenarios = []
    for order in ((0, 1), (1, 0)):
        rec, s, sf, mx = a.permute_zero_shots(r, scores, safe, maximum, order, cfg)
        scenarios.append(a.decision_scenario(rec, s, sf, mx, Scripted(), cfg, states,
                          reference_order=a.permuted_reference_order(scores, order, cfg)))
    rows, cases = a.aggregate_decision_rows(scenarios, cfg)
    roots = [case for case in cases if case["candidate_mask"] == 1 and case["evaluator_mask"] == 0]
    assert len(roots) == 2 and all(case["status"] == "ACTION" for case in roots)
    assert roots[0]["training_row"] == roots[1]["training_row"]
    row = rows[roots[0]["training_row"]]
    assert np.isclose(row["target"].sum(), 1)
    assert np.isclose(row["reach"], 1)
    assert not row["target"][~row["valid"]].any()
    # Before the evaluator is acquired, swapping unseen answer identities leaves the input unchanged.
    assert scenarios[0]["keys"][scenarios[0]["index"][a.State()]] == scenarios[1]["keys"][scenarios[1]["index"][a.State()]]
    future = [case for case in cases if case["candidate_mask"] == 3 and case["evaluator_mask"] == 0]
    assert len(future) == 2 and future[0]["training_row"] == future[1]["training_row"]


def test_future_group_stays_shared_when_only_one_scenario_has_reached_its_goal():
    cfg = a.Config(k=1, zero_shots=2, hidden_dim=4)
    r = record(cfg)
    states = a.structural_states(cfg.zero_shots, cfg.k)
    safe = maximum = np.array([1, 0, 0], np.float32)
    class Scripted:
        def predict(self, rec, state, cfg_override=None):
            p = np.zeros(3*cfg.n+2, np.float32)
            if state.candidate_mask & 3 == 3:
                p[[cfg.n, 2*cfg.n, 3*cfg.n, 3*cfg.n+1]] = .9
                if state.evaluator_mask:
                    p[[cfg.n+1, 2*cfg.n+1]] = .9
            return np.zeros(cfg.hidden_dim, np.float32), p
        def predict_many(self, rec, states, allow_hidden_source=False):
            values = [self.predict(rec, state) for state in states]
            return np.stack([x[0] for x in values]), np.stack([x[1] for x in values])
    scenarios = []
    scores = np.array([.9, .8, .3])
    for order in ((0, 1), (1, 0)):
        rec, s, sf, mx = a.permute_zero_shots(r, scores, safe, maximum, order, cfg)
        scenarios.append(a.decision_scenario(rec, s, sf, mx, Scripted(), cfg, states,
                          reference_order=a.permuted_reference_order(scores, order, cfg)))
    rows, cases = a.aggregate_decision_rows(scenarios, cfg)
    future = [case for case in cases if case["candidate_mask"] == 3 and case["evaluator_mask"] == 0]
    assert len(future) == 2
    assert sorted(case["goal_rank"] for case in future) == [0, 1]
    assert future[0]["training_row"] == future[1]["training_row"]
    assert rows[future[0]["training_row"]]["target"][0] == 1


def test_fixed_policy_does_not_skip_an_unaffordable_step():
    cfg = a.Config(max_cost=29)
    class NoStop:
        def predict(self, rec, state, cfg_override=None):
            return np.zeros(cfg.hidden_dim), np.zeros(3*cfg.n+2)
    row = a.rollout(record(cfg), np.zeros(cfg.n), NoStop(), cfg, policy="fixed", trace=True)
    assert [step["action"] for step in row["trajectory"] if step["action"] is not None] == [0, 6, 5]
    assert row["cost"] == 23 and row["reason"] == "budget"


def test_no_max_evaluation_keeps_false_predicted_max_and_separate_stratum():
    rows = [dict(uid="a", correct=0, exact_max=None, has_max=False, cost=1,
                 saved_fraction=.9, reason="predicted_max"),
            dict(uid="b", correct=1, exact_max=True, has_max=True, cost=10,
                 saved_fraction=0, reason="predicted_max")]
    summary = a.policy_summary(rows)
    assert summary["n"] == 2 and summary["top1_correct"] == .5
    assert summary["exact_max_recovery"] == 1
    assert summary["no_max_false_predicted_max"] == 1


def test_continuation_counts_all_orders_and_rejects_recovery_after_a_broken_goal(monkeypatch):
    cfg = a.Config(k=1, zero_shots=2, hidden_dim=4)
    class Predictor:
        device = "cpu"
        def predict(self, rec, state, cfg_override=None):
            return np.zeros(cfg.hidden_dim, np.float32), np.zeros(3*cfg.n+2)
    class Head(torch.nn.Module):
        def forward(self, hidden):
            return torch.tensor([[0., 1., 2.]]).repeat(len(hidden), 1)
    # Initial rank recognized; adding ZS2 loses it; the final pool would restore all goals.
    def flags(rank, order, safe, maximum, probabilities, full_probabilities, state, config):
        return state.candidate_mask == 7 or (rank == 0 and state.candidate_mask == 1)
    monkeypatch.setattr(a, "goal_condition", flags)
    teacher = {"scores": np.array([[.9, .8, .7]]),
               "safe": np.array([[1, 0, 0]]), "maximum": np.array([[1, 0, 0]])}
    result = a.continuation_diagnostic([record(cfg)], [0], teacher, Predictor(), cfg, Head())
    assert result["orders_per_question"] == 2
    assert result["mean_rank_prefix_reached"] == 1
    assert result["per_rank_reach"] == [1., 0., 0.]
    assert result["prior_goal_loss_rate"] == 1
    assert result["questions_with_prior_goal_lost"] == 1


def test_end_to_end_checkpoint_resume_splits_and_frozen_bundle(tmp_path):
    train, external = tmp_path/"train.json", tmp_path/"external.json"
    cfg = a.Config(train_file=str(train), test_files=[str(external)],
                   output_dir=str(tmp_path/"run"), device="cpu", cpu_threads=1,
                   k=2, zero_shots=2, teacher_folds=2, hidden_dim=8, residual_blocks=1,
                   batch_size=16, teacher_epochs=2, snapshot_epochs=2, decision_epochs=2,
                   patience=2, snapshots_per_query=4, dev_snapshots_per_query=4,
                   decision_batch_size=16, recognition_threshold=.1,
                   bootstrap_samples=20)
    train.write_text(json.dumps([raw_record(i,cfg) for i in range(40)]))
    external.write_text(json.dumps([raw_record(i,cfg,prefix="external",all_wrong=(i==0))
                                    for i in range(3)]))
    work = a.run_pipeline(cfg)
    assert set(work.results["external"]["adaptive"]["policies"]) == {
        "full", "fixed", "supervised"}
    assert work.results["external"]["adaptive"]["policies"]["supervised"]["n"] == 3
    assert (tmp_path/"run"/"decision_head_completed.pt").exists()
    manifest = json.loads((tmp_path/"run"/"decision_dataset_manifest.json").read_text())
    assert manifest["schema_version"] == 3
    policy_shards = [a.load_checkpoint(tmp_path/"run"/path) for path in manifest["roles"]["policy"]]
    saved_rows = [row for shard in policy_shards for row in shard["rows"]]
    assert saved_rows and {"uid", "state_key", "h", "target", "valid", "goal_rank"} <= set(saved_rows[0])
    assert not {"truth", "teacher_scores", "future_measurements"} & set(saved_rows[0])
    assert all("weight" not in row for row in saved_rows)
    assert all(np.isclose(row["target"].sum(), 1) and not row["target"][~row["valid"]].any()
               for row in saved_rows)
    assert all(len(shard["cases"]) == len(a.structural_states(cfg.zero_shots, cfg.k)) * 2
               for shard in policy_shards)
    assert all(sum(a.Counter(case["status"] for case in shard["cases"]).values()) == shard["expected_cases"]
               for shard in policy_shards)
    role_uids = {role: {work.records[i].uid for i in ids} for role, ids in work.splits.items()}
    assert {row["uid"] for row in saved_rows} <= role_uids["policy"]
    dev_rows = [row for path in manifest["roles"]["dev"]
                for row in a.load_checkpoint(tmp_path/"run"/path)["rows"]]
    assert {row["uid"] for row in dev_rows} <= role_uids["dev"]
    for role, uids in role_uids.items():
        assert all(not uids & other for name, other in role_uids.items() if name != role)
    assert (tmp_path/"run"/"results.json").exists()
    assert (tmp_path/"run"/"final_trajectories.jsonl").exists()
    for fold in work.teacher["folds"]:
        assert not set(fold["heldout_ids"]) & (set(fold["train_ids"]) |
                                               set(fold["validation_ids"]))
    frozen = a.cpu_state(work.predictor.model)
    assert not any(p.requires_grad for p in work.predictor.model.parameters())
    restored_cfg, predictor, head = a.load_inference_bundle(tmp_path/"run"/"inference_bundle.pt")
    assert all(torch.equal(frozen[k], v.cpu()) for k,v in predictor.model.state_dict().items())
    assert not predictor.model.training
    assert np.array_equal(predictor.temperatures, work.predictor.temperatures)
    r = work.records[work.splits["external"][0]]
    maximum = work.teacher["maximum"][work.splits["external"][0]]
    assert a.rollout(r,maximum,predictor,restored_cfg,head=head)["selected"] == (
        a.rollout(r,maximum,work.predictor,cfg,head=work.head)["selected"])
    resumed = a.Workflow(cfg).prepare().train_teacher().train_snapshot().train_decision_head()
    assert resumed.decision_checkpoint["best_epoch"] == work.decision_checkpoint["best_epoch"]
    # Interrupted shard construction rebuilds only the missing question.
    keep = tmp_path/"run"/manifest["roles"]["policy"][0]
    missing = tmp_path/"run"/manifest["roles"]["policy"][1]
    kept_mtime = keep.stat().st_mtime_ns
    missing.unlink()
    (tmp_path/"run"/"decision_head_completed.pt").unlink()
    repaired = a.Workflow(cfg).prepare().train_teacher().train_snapshot().train_decision_head()
    assert missing.exists() and keep.stat().st_mtime_ns == kept_mtime
    assert repaired.decision_checkpoint["coverage"]["policy"]["state_order_cases"] == (
        len(a.structural_states(cfg.zero_shots, cfg.k)) * 2 * len(work.splits["policy"]))
    shard = a.load_checkpoint(keep)
    shard["schema_version"] = 2
    a.atomic_torch(keep, shard)
    (tmp_path/"run"/"decision_head_completed.pt").unlink()
    with pytest.raises(ValueError, match="Incompatible decision shard"):
        a.Workflow(cfg).prepare().train_teacher().train_snapshot().train_decision_head()
    shard["schema_version"] = 3
    a.atomic_torch(keep, shard)
    contract_path = tmp_path/"run"/"contract.json"
    contract = json.loads(contract_path.read_text())
    assert contract.pop("implementation_revision") == a.PIPELINE_REVISION
    contract_path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="different config/data contract"):
        a.Workflow(cfg)
    contract["implementation_revision"] = a.PIPELINE_REVISION
    contract_path.write_text(json.dumps(contract))
    changed = copy.deepcopy(cfg)
    changed.recognition_threshold = .5
    with pytest.raises(ValueError, match="different config"):
        a.Workflow(changed)


def test_notebook_is_self_contained_compiles_and_matches_module():
    root = Path(__file__).parent
    nb = json.loads((root/"adaptive_analogical_training_kaggle.ipynb").read_text(encoding="utf-8"))
    embedded = []
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, f"notebook_cell_{index}", "exec")
        if cell.get("metadata", {}).get("jupyter", {}).get("source_hidden"):
            embedded.append(source)
    original = (root/"adaptive_analogical_training.py").read_text(encoding="utf-8").split("# %% CLI")[0]
    assert ast.dump(ast.parse("\n".join(embedded))) == ast.dump(ast.parse(original))
    assert not any(cell.get("outputs") for cell in nb["cells"])
