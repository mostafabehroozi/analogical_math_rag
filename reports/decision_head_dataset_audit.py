"""Small offline counterexamples for the October 2026 decision-head audit.

Run from repository root: python -B reports/decision_head_dataset_audit.py
These are synthetic checks of code semantics, not measurements of real runs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adaptive_analogical_training as a


def terminal_mismatch():
    cfg = a.Config(k=1, zero_shots=2, hidden_dim=4)
    n = cfg.n
    probabilities = np.zeros(3*n+2, np.float32)
    probabilities[:n] = [.6, .9, .2]
    probabilities[[n, 2*n, 3*n, 3*n+1]] = .9
    safe = maximum = np.array([1, 0, 0], np.float32)
    state = a.State(3, 0)
    return {
        "teacher_max_slot": 0,
        "correctness_probabilities": probabilities[:n].tolist(),
        "planner_rank0_goal_attained": a.goal_condition(
            0, [0, 1, 2], safe, maximum, probabilities, probabilities, state, cfg),
        "runtime_selected_slot": a.selected_candidate(probabilities, state, cfg),
        "runtime_stopping_reason": a.stopping_reason(probabilities, state, cfg),
    }


def cross_question_cost_regret():
    cfg = a.Config(k=2, zero_shots=1, repeats=1, hidden_dim=4)
    def make_record(uid, baseline):
        return a.Record(uid, uid, "synthetic", ["ZS1", "OS1", "OS2"],
                        ["R1", "R2"], np.full(2, .5, np.float32),
                        np.full(2, baseline, np.float32),
                        np.zeros((3, 2), np.float32),
                        np.array([1, 0, 0], np.float32))
    records = [make_record("A", 0.), make_record("B", 1.)]
    scores = np.array([.9, .7, .3], np.float32)
    safe = maximum = np.array([1, 0, 0], np.float32)

    class ScriptedPredictor:
        device = torch.device("cpu")
        def predict(self, record, state, cfg_override=None):
            p = np.zeros(3*cfg.n+2, np.float32)
            p[:cfg.n] = [.9, .4, .3]
            visible = np.flatnonzero(a.bit_array(state.evaluator_mask, cfg.k))
            baseline = float(record.baseline[visible[0]]) if len(visible) else 0.
            is_b = baseline > .5
            lit = (bool(state.evaluator_mask & 1) if not is_b else
                   bool(state.evaluator_mask & 2) and
                   (not state.evaluator_mask & 1 or bool(state.candidate_mask & 2)))
            if lit:
                p[[cfg.n, 2*cfg.n, 3*cfg.n, 3*cfg.n+1]] = .9
            present = a.present_mask(state, cfg)
            p[:3*cfg.n].reshape(3, cfg.n)[:, ~present] = 0
            h = np.array([state.candidate_mask, state.evaluator_mask, baseline, 0], np.float32)
            return h, p
        def predict_many(self, record, states, allow_hidden_source=False):
            values = [self.predict(record, state) for state in states]
            return (np.stack([v[0] for v in values]),
                    np.stack([v[1] for v in values]))

    predictor = ScriptedPredictor()
    states = a.structural_states(cfg.zero_shots, cfg.k)
    scenarios = [a.decision_scenario(record, scores, safe, maximum,
                                     predictor, cfg, states) for record in records]

    def root(rows, cases):
        case = next(case for case in cases if case["scenario"] == 0 and
                    case["candidate_mask"] == 1 and case["evaluator_mask"] == 0)
        row = rows[case["training_row"]]
        return {
            "target": row["target"].tolist(),
            "action_reach_R1_R2": row["action_reach"][:2].tolist(),
            "incremental_goal_cost_R1_R2": row["action_expected_cost"][:2].tolist(),
        }

    local = [a.aggregate_decision_rows([scenario], cfg) for scenario in scenarios]
    joint = a.aggregate_decision_rows(scenarios, cfg)
    root_keys = [scenario["keys"][scenario["index"][a.State()]] for scenario in scenarios]
    lookup = {tuple(row["h"]): int(np.argmax(row["target"]))
              for rows, _ in local for row in rows}

    class RootThenLocalHead(torch.nn.Module):
        def __init__(self, root_action):
            super().__init__()
            self.root_action = root_action
        def forward(self, hidden):
            actions = [self.root_action if int(h[0]) == 1 and int(h[1]) == 0 else
                       lookup[tuple(h)] for h in hidden.cpu().numpy()]
            logits = torch.zeros((len(actions), cfg.action_count))
            logits[torch.arange(len(actions)), actions] = 10
            return logits

    runtime = {}
    for root_action in (0, 1):
        rows = [a.rollout(record, maximum, predictor, cfg, "supervised",
                          RootThenLocalHead(root_action), trace=True) for record in records]
        runtime[a.action_names(cfg)[root_action]] = {
            "per_question_solver_calls": [row["cost"] for row in rows],
            "mean_solver_calls": float(np.mean([row["cost"] for row in rows])),
            "per_question_stop_reason": [row["reason"] for row in rows],
        }
    return {
        "initial_observation_keys_equal": root_keys[0] == root_keys[1],
        "per_question_planner_A": root(*local[0]),
        "per_question_planner_B": root(*local[1]),
        "cross_question_joint_planner": root(*joint),
        "local_target_CE_mixture_R1_R2": [.5, .5],
        "greedy_tie_break_action": a.action_names(cfg)[0],
        "runtime_with_identical_root_and_locally_optimal_continuations": runtime,
        "note": "Goal costs are incremental; runtime costs include initial ZS1. Both runtime branches stop on the true selected MAX.",
    }


def structural_counts_and_temperature():
    cfg = a.Config()
    n = cfg.n
    logits = np.linspace(-5, 5, 3*n+2)
    state = a.State(3, 0)
    thresholds, selected = [], []
    for temperature in (.25, 1., 4.):
        p = 1/(1+np.exp(-logits/temperature))
        thresholds.append((p >= .5).tolist())
        selected.append(a.selected_candidate(p, state, cfg))
    return {
        "default_all_structural_states": len(a.structural_states(cfg.zero_shots, cfg.k)),
        "default_initial_reachable_states": len(a.reachable_states(cfg)),
        "threshold_point5_bits_unchanged_by_positive_temperature": all(x == thresholds[0] for x in thresholds),
        "selected_correctness_rank_unchanged": len(set(selected)) == 1,
    }


def early_state_sampling_and_invariance():
    cfg = a.Config()
    records = [a.Record(str(i), str(i), "synthetic", list(range(cfg.n)),
                        list(range(cfg.k)), np.full(cfg.k, i, np.float32),
                        np.full(cfg.k, i, np.float32),
                        np.full((cfg.n, cfg.k), i, np.float32),
                        np.full(cfg.n, i, np.float32)) for i in (0, 1)]
    states = a.reachable_states(cfg)
    evaluator_free = [state for state in states if state.evaluator_mask == 0]
    # Exactly reproduce the unaugmented dev sampler's state schedule.
    sampled = []
    for rank in range(500):
        sampled.append(a.State((1 << cfg.n)-1, (1 << cfg.k)-1))
        for j in range(cfg.dev_snapshots_per_query - 1):
            position = rank * (cfg.dev_snapshots_per_query - 1) + j
            sampled.append(states[(position * 53) % len(states)])
    p = np.zeros(3*cfg.n+2, np.float32)
    p[0] = .8
    threshold_outcomes = {}
    for value in (.499, .501):
        p[[cfg.n, 2*cfg.n, 3*cfg.n, 3*cfg.n+1]] = value
        threshold_outcomes[str(value)] = a.stopping_reason(p, a.State(), cfg)
    # Demonstrate the finite-precision exception to monotone sigmoid scaling.
    saturation = {}
    for temperature in (.25, 1.):
        raw = np.array([5., 6.], np.float32)
        probabilities = 1 / (1 + np.exp(-np.clip(raw / temperature, -40, 40)))
        saturation[str(temperature)] = {
            "probabilities": probabilities.tolist(), "argmax": int(probabilities.argmax())}
    return {
        "evaluator_free_states": [
            {"candidate_mask": state.candidate_mask, "cost": a.state_cost(state, cfg),
             "different_records_have_identical_observations": np.array_equal(
                 a.observation(records[0], state, cfg), a.observation(records[1], state, cfg))}
            for state in evaluator_free],
        "hypothetical_500_question_dev_sampler": {
            "total_snapshots": len(sampled),
            "initial_snapshots": sum(state == a.State() for state in sampled),
            "evaluator_free_snapshots": sum(state.evaluator_mask == 0 for state in sampled),
            "full_snapshots": sum(state == a.State(255, 31) for state in sampled)},
        "four_signal_threshold_cliff": threshold_outcomes,
        "float32_sigmoid_saturation": saturation,
        "default_call_counts": {
            f"candidates_{n}_evaluators_{k}": a.acquisition_call_counts(n, k, cfg)
            for n, k in ((1, 0), (2, 0), (1, 1), (8, 5))},
    }


def denominator_provenance_gap():
    cfg = a.Config(k=1, zero_shots=1, repeats=5)
    item = {
        "target_query_text": "Synthetic audit question",
        "metadata": {"config_snapshot": {"MIRROR_N_OPTIMIZATION": 3}},
        "retrieved_set": [{"corpus_index": 10, "similarity_score": .5}],
        "candidate_set": {
            "zs_0": {"source_exemplar_idx": -1, "generation_status": "SUCCESS", "candidate_text": "A"},
            "os_0": {"source_exemplar_idx": 10, "generation_status": "SUCCESS", "candidate_text": "B"}},
        "ground_truth_labels": {cid: {"evaluation_status": "SUCCESS", "is_correct": True}
                                for cid in ("zs_0", "os_0")},
        "intrinsic_baselines": {"10": 0.},
        "cross_evaluation_matrix": {cid: {"10": 1.} for cid in ("zs_0", "os_0")},
    }
    parsed = a.parse_record(item, 0, "synthetic", cfg)
    return {"stored_repeats": 3, "configured_repeats": cfg.repeats,
            "mismatched_log_accepted": parsed.uid == "synthetic::0",
            "reason": "Rates 0 and 1 satisfy both denominator grids; stored metadata is ignored."}


if __name__ == "__main__":
    print(json.dumps({"evidence_type": "synthetic offline code-semantic probes",
                      "terminal_mismatch": terminal_mismatch(),
                      "cross_question_cost_regret": cross_question_cost_regret(),
                      "structural_and_calibration": structural_counts_and_temperature(),
                      "early_states": early_state_sampling_and_invariance(),
                      "denominator_provenance": denominator_provenance_gap()}, indent=2))
