"""Checks for lossless Stage 3 observation and frozen-inference batching."""
import numpy as np
import pytest
import torch

import adaptive_analogical_training as a


def make_record(cfg, seed=7):
    rng = np.random.RandomState(seed)
    return a.Record(str(seed), str(seed), "synthetic", list(range(cfg.n)),
                    list(range(cfg.k)), rng.rand(cfg.k), rng.rand(cfg.k),
                    rng.rand(cfg.n, cfg.k), np.ones(cfg.n, np.float32))


@pytest.mark.parametrize("geometry", [(1, 1), (2, 2), (3, 5)])
@pytest.mark.parametrize("cost_unit,max_cost", [("solver_calls", None),
                                               ("solver_calls", 9.),
                                               ("total_calls", 9.)])
def test_batched_structural_observations_preserve_every_scalar_byte(geometry, cost_unit, max_cost):
    zero_shots, k = geometry
    cfg = a.Config(zero_shots=zero_shots, k=k, cost_unit=cost_unit, max_cost=max_cost)
    record = make_record(cfg)
    # Include exhausted and out-of-budget states, and a noncatalog order.
    states = tuple(reversed(a.structural_states(zero_shots, k)))
    expected = np.stack([a.observation(record, state, cfg) for state in states])
    actual = a.observation_many(record, states, cfg)
    assert actual.dtype == np.float32
    assert actual.tobytes() == expected.tobytes()


def test_batched_observations_index_only_visible_values_and_validate_hidden_sources():
    cfg = a.Config(k=2, zero_shots=2)
    record = make_record(cfg)
    record.similarity[:] = np.nan
    record.baseline[:] = np.nan
    record.ccs[:] = np.nan
    states = [a.State(1, 0), a.State(2, 0)]
    expected = np.stack([a.observation(record, state, cfg) for state in states])
    assert a.observation_many(record, states, cfg).tobytes() == expected.tobytes()
    with pytest.raises(ValueError, match="Invalid snapshot observation"):
        a.observation_many(record, [a.State(1, 1)], cfg)
    hidden = a.State(1 << cfg.zero_shots, 0)
    with pytest.raises(ValueError, match="one-shot"):
        a.observation_many(record, [hidden], cfg)
    assert a.observation_many(record, [hidden], cfg, True).tobytes() == a.observation(record, hidden, cfg, True).tobytes()


def test_augmented_snapshot_batches_keep_the_scalar_observation_contract():
    cfg = a.Config(k=2, zero_shots=2)
    record = make_record(cfg)
    full = a.complete_snapshot_state((1 << cfg.n)-1, 3, 3, cfg)
    missing = a.augment_measurements(full, cfg, np.random.RandomState(28))
    states = [a.State(), missing, full]
    expected = np.stack([a.observation(record, state, cfg, True) for state in states])
    assert a.observation_many(record, states, cfg, True).tobytes() == expected.tobytes()


def test_structural_templates_are_reused_across_questions_without_mutating_inputs(monkeypatch):
    cfg = a.Config(k=2, zero_shots=2)
    states = a.structural_states(cfg.zero_shots, cfg.k)
    a.structural_observation_template.cache_clear()
    original = a.observation
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(a, "observation", counted)
    first = a.observation_many(make_record(cfg), states, cfg)
    assert len(calls) == len(states)
    before = first.copy()
    second = a.observation_many(make_record(cfg, 9), states, cfg)
    assert len(calls) == len(states)
    np.testing.assert_array_equal(first, before)
    assert not np.array_equal(first, second)
    cfg.max_cost = 9.
    a.observation_many(make_record(cfg), states, cfg)
    assert len(calls) == 2*len(states)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_frozen_batch_probabilities_preserve_scalar_masking_exactly(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this test runtime")
    cfg = a.Config(k=5, zero_shots=3, hidden_dim=8, residual_blocks=1)
    record = make_record(cfg)
    model = a.ResNet(cfg.input_dim, 3*cfg.n+2, cfg)
    predictor = a.FrozenPredictor({"weights": a.cpu_state(model),
                                   "temperatures": np.array([.5, 1., 2., .75, 1.5])}, cfg, device)
    states = a.structural_states(cfg.zero_shots, cfg.k)
    xs = np.stack([a.observation(record, state, cfg) for state in states])
    hidden_parts, logits_parts = [], []
    with torch.no_grad():
        for start in range(0, len(xs), 1024):
            h = predictor.model.encode(torch.as_tensor(xs[start:start+1024], device=device))
            hidden_parts.append(h.cpu().numpy())
            logits_parts.append(predictor.model.output_layer(h).cpu().numpy())
    hidden, logits = np.concatenate(hidden_parts), np.concatenate(logits_parts)
    expected = 1 / (1 + np.exp(-np.clip(logits / predictor.temperatures, -40, 40)))
    for row, state in enumerate(states):
        for block in range(3):
            expected[row, block*cfg.n:(block+1)*cfg.n][~a.present_mask(state, cfg)] = 0
    actual_hidden, actual = predictor.predict_many(record, states)
    np.testing.assert_array_equal(actual_hidden, hidden)
    np.testing.assert_array_equal(actual, expected)
    assert not predictor.model.training
    assert all(not parameter.requires_grad for parameter in predictor.model.parameters())


@pytest.mark.parametrize("left,right,accepted", [(3., 3.00001, True),
                                                  (3., 3.001, False),
                                                  (np.inf, np.inf, True),
                                                  (np.nan, np.nan, False)])
def test_visible_feature_batch_validation_retains_tolerance_and_nonfinite_rules(left, right, accepted):
    cfg = a.Config(k=1, zero_shots=1, hidden_dim=4)
    state = a.State((1 << cfg.n)-1, (1 << cfg.k)-1)
    record = make_record(cfg)
    key = a.visible_state_key(record, state, cfg)
    scenarios = [{"record": record, "states": [state],
                  "hidden": np.full((1, cfg.hidden_dim), value),
                  "goal": np.ones((1, cfg.n), bool), "keys": [key],
                  "index": {state: 0}} for value in (left, right)]
    if accepted:
        rows, cases = a.aggregate_decision_rows(scenarios, cfg)
        assert rows == [] and all(case["status"] == "COMPLETE" for case in cases)
    else:
        with pytest.raises(ValueError, match="inconsistent frozen features"):
            a.aggregate_decision_rows(scenarios, cfg)
