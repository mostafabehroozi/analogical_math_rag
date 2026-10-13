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
def test_batched_observations_preserve_every_scalar_byte(geometry, cost_unit, max_cost):
    zero_shots, k = geometry
    cfg = a.Config(zero_shots=zero_shots, k=k, cost_unit=cost_unit, max_cost=max_cost)
    record = make_record(cfg)
    # Include exhausted and out-of-budget states, and a noncatalog order.
    states = tuple(reversed(a.presence_states(zero_shots, k)))
    expected = np.stack([a.observation(record, state, cfg) for state in states])
    actual = a.observation_many(record, states, cfg)
    assert actual.dtype == np.float32
    assert actual.tobytes() == expected.tobytes()


def test_batched_observations_index_only_visible_values_and_accept_a_one_shot_without_its_evaluator():
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
    # OS1 without R1 is an ordinary state: it shows the candidate's source slot and no measurement
    # of that source, so even NaN measurements stay invisible.
    lone = a.State(1 << cfg.zero_shots, 0)
    assert a.observation_many(record, [lone], cfg).tobytes() == a.observation(record, lone, cfg).tobytes()
    assert np.isfinite(a.observation(record, lone, cfg)).all()


def test_augmented_snapshot_batches_keep_the_scalar_observation_contract():
    cfg = a.Config(k=2, zero_shots=2)
    record = make_record(cfg)
    full = a.complete_snapshot_state((1 << cfg.n)-1, 3, 3, cfg)
    # One failed baseline and two unattempted cross evaluations.
    missing = a.SnapshotState(full.candidates, full.retrieved, full.evaluators, full.similarity_observed,
                              full.baseline_attempted, full.baseline_observed & ~1,
                              full.ccs_attempted & ~0b101, full.ccs_observed & ~0b101)
    states = [a.State(), missing, full]
    expected = np.stack([a.observation(record, state, cfg) for state in states])
    assert a.observation_many(record, states, cfg).tobytes() == expected.tobytes()


def test_state_templates_are_reused_across_questions_without_mutating_inputs(monkeypatch):
    cfg = a.Config(k=2, zero_shots=2)
    states = a.presence_states(cfg.zero_shots, cfg.k)
    a.state_observation_template.cache_clear()
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
    states = a.presence_states(cfg.zero_shots, cfg.k)
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
    features, actual = predictor.predict_many(record, states)
    # The head's input is the frozen hidden vector followed by the acquired-item masks.
    assert features.shape == (len(states), cfg.head_input_dim)
    np.testing.assert_array_equal(features[:, :cfg.hidden_dim], hidden)
    np.testing.assert_array_equal(actual, expected)
    assert not predictor.model.training
    assert all(not parameter.requires_grad for parameter in predictor.model.parameters())


def test_head_features_append_exactly_the_acquired_candidate_and_evaluator_masks():
    cfg = a.Config(k=3, zero_shots=2, hidden_dim=8, residual_blocks=1)
    record = make_record(cfg)
    predictor = a.FrozenPredictor({"weights": a.cpu_state(a.ResNet(cfg.input_dim, 3*cfg.n+2, cfg)),
                                   "temperatures": np.ones(5)}, cfg, "cpu")
    states = a.presence_states(cfg.zero_shots, cfg.k)
    xs = a.observation_many(record, states, cfg)
    features, _ = predictor.predict_many(record, states, observations=xs)
    masks = np.stack([np.r_[a.bit_array(s.candidate_mask, cfg.n), a.bit_array(s.evaluator_mask, cfg.k)]
                      for s in states]).astype(np.float32)
    np.testing.assert_array_equal(features[:, cfg.hidden_dim:], masks)
    # features() on the device agrees with the batched path, and predict() returns one row of it.
    np.testing.assert_allclose(predictor.features(xs).numpy(), features, rtol=1e-6, atol=1e-6)
    # A single-row batch may round differently from a full batch, but never the mask columns.
    single = predictor.predict(record, states[100])[0]
    np.testing.assert_allclose(single, features[100], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(single[cfg.hidden_dim:], features[100, cfg.hidden_dim:])
    assert a.SupervisedActionHead(cfg).net[0].in_features == cfg.head_input_dim == features.shape[1]


@pytest.mark.parametrize("left,right,accepted", [(3., 3.00001, True),
                                                  (3., 3.001, False),
                                                  (np.inf, np.inf, True),
                                                  (np.nan, np.nan, False)])
def test_visible_feature_batch_validation_retains_tolerance_and_nonfinite_rules(left, right, accepted):
    cfg = a.Config(k=1, zero_shots=2, hidden_dim=4)
    record = make_record(cfg)

    class Scripted:
        """The first zero-shot order gets `left`, the second `right`, at every state."""
        def __init__(self):
            self.calls = 0
        def predict(self, rec, state, cfg_override=None):
            return np.zeros(cfg.head_input_dim, np.float32), np.zeros(3*cfg.n+2, np.float32)
        def predict_many(self, rec, states):
            value = (left, right)[self.calls]
            self.calls += 1
            return (np.full((len(states), cfg.head_input_dim), value, np.float32),
                    np.zeros((len(states), 3*cfg.n+2), np.float32))

    scores, flags = np.array([.9, .8, .7], np.float32), np.zeros(cfg.n, np.float32)
    # Without an evaluator the two orders look identical, so their features must agree.
    if accepted:
        plan = a.question_plan(record, scores, flags, flags, Scripted(), cfg)
        assert plan["leader"][1, 0] == 0
    else:
        with pytest.raises(ValueError, match="inconsistent frozen features"):
            a.question_plan(record, scores, flags, flags, Scripted(), cfg)
