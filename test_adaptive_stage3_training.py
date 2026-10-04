"""Offline checks for bounded Stage 3 storage and unchanged training updates."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import adaptive_analogical_training as a


def packed_rows(cfg, count, seed=17):
    rng = np.random.RandomState(seed)
    valid = rng.rand(count, cfg.action_count) > .4
    valid[:, 0] = True
    target = valid.astype(np.float32)
    target /= target.sum(axis=1, keepdims=True)
    return {"h": torch.from_numpy(rng.randn(count, cfg.hidden_dim).astype(np.float32)),
            "target": torch.from_numpy(target), "valid": torch.from_numpy(valid)}


def training_files(tmp_path, cfg, policy_count=1, dev_count=1):
    paths = {"policy": [], "dev": []}
    for role, count in (("policy", policy_count), ("dev", dev_count)):
        for index in range(count):
            path = tmp_path / f"{role}_{index}.training.pt"
            a.atomic_torch(path, packed_rows(cfg, 9, index + 17))
            paths[role].append(path)
    return paths


def track_training_loads(monkeypatch):
    calls = []
    original = a.load_training_checkpoint

    def tracked(path):
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(a, "load_training_checkpoint", tracked)
    return calls


def test_training_loader_maps_only_packed_cpu_tensors(tmp_path, monkeypatch):
    cfg = a.Config(hidden_dim=4)
    expected = packed_rows(cfg, 13)
    path = tmp_path / "question.training.pt"
    a.atomic_torch(path, expected)
    kwargs_seen = []
    original = torch.load

    def tracked(*args, **kwargs):
        kwargs_seen.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(a.torch, "load", tracked)
    actual = a.load_training_checkpoint(path)
    assert len(kwargs_seen) == 1
    assert kwargs_seen[0]["mmap"] is True
    assert kwargs_seen[0]["weights_only"] is True
    assert str(kwargs_seen[0]["map_location"]) == "cpu"
    assert set(actual) == {"h", "target", "valid"}
    assert all(value.device.type == "cpu" and torch.equal(value, expected[key])
               for key, value in actual.items())


def test_identical_training_companion_is_reused_without_rewriting(tmp_path):
    cfg = a.Config(hidden_dim=4)
    packed = packed_rows(cfg, 13)
    path = tmp_path / "question.training.pt"
    assert a.ensure_training_companion(path, packed) is False
    original_bytes, original_mtime = path.read_bytes(), path.stat().st_mtime_ns
    assert a.ensure_training_companion(path, packed) is True
    assert path.read_bytes() == original_bytes and path.stat().st_mtime_ns == original_mtime


@pytest.mark.parametrize("damage", ["corrupt_archive", "wrong_dtype", "changed_target", "missing_field"])
def test_invalid_training_companion_is_rebuilt_from_validated_rows(tmp_path, damage):
    cfg = a.Config(hidden_dim=4)
    packed = packed_rows(cfg, 13)
    path = tmp_path / "question.training.pt"
    if damage == "corrupt_archive":
        path.write_bytes(b"interrupted cache write")
    else:
        bad = {key: value.clone() for key, value in packed.items()}
        if damage == "wrong_dtype":
            bad["h"] = bad["h"].double()
        elif damage == "changed_target":
            bad["target"][0, 0] += .25
        else:
            bad.pop("valid")
        a.atomic_torch(path, bad)
    assert a.ensure_training_companion(path, packed) is False
    restored = a.load_training_checkpoint(path)
    assert set(restored) == set(packed)
    assert all(value.dtype == packed[key].dtype and torch.equal(value, packed[key])
               for key, value in restored.items())


@pytest.mark.parametrize("available", [None, 0])
def test_unknown_or_unavailable_ram_streams_without_retention(tmp_path, monkeypatch, available):
    cfg = a.Config(hidden_dim=4, decision_cache_mb=4096)
    paths = training_files(tmp_path, cfg)
    monkeypatch.setattr(a, "available_host_memory", lambda: available)
    calls = track_training_loads(monkeypatch)
    cache = a.PackedDecisionCache(paths, cfg)
    assert calls == []
    path = paths["policy"][0]
    first, second = cache.get(path), cache.get(path)
    assert calls == [path, path]
    assert first is not second and cache.bytes == 0
    assert all(torch.equal(value, second[key]) for key, value in first.items())


def test_zero_cache_budget_streams_even_when_ram_is_available(tmp_path, monkeypatch):
    cfg = a.Config(hidden_dim=4, decision_cache_mb=0)
    paths = training_files(tmp_path, cfg)
    monkeypatch.setattr(a, "available_host_memory", lambda: 1024**4)
    calls = track_training_loads(monkeypatch)
    cache = a.PackedDecisionCache(paths, cfg)
    assert calls == [] and cache.bytes == 0
    for _ in range(2):
        cache.get(paths["dev"][0])
    assert calls == [paths["dev"][0]] * 2


def test_half_available_ram_cap_prioritizes_complete_policy_role(tmp_path, monkeypatch):
    cfg = a.Config(hidden_dim=4, decision_cache_mb=4096)
    paths = training_files(tmp_path, cfg)
    policy, dev = paths["policy"][0], paths["dev"][0]
    budget = policy.stat().st_size
    monkeypatch.setattr(a, "available_host_memory", lambda: budget * 2)
    calls = track_training_loads(monkeypatch)
    cache = a.PackedDecisionCache(paths, cfg)
    assert calls == [policy]
    assert cache.get(policy) is cache.get(policy)
    assert calls == [policy]
    for _ in range(2):
        cache.get(dev)
    assert calls == [policy, dev, dev]
    policy_arrays = cache.get(policy)
    assert cache.bytes == sum(value.numel() * value.element_size() for value in policy_arrays.values())
    assert cache.bytes <= budget


def test_oversized_policy_role_is_not_partially_cached_and_dev_can_fit(tmp_path, monkeypatch):
    cfg = a.Config(hidden_dim=4, decision_cache_mb=4096)
    paths = training_files(tmp_path, cfg, policy_count=2)
    budget = max(path.stat().st_size for role in paths.values() for path in role)
    assert sum(path.stat().st_size for path in paths["policy"]) > budget
    monkeypatch.setattr(a, "available_host_memory", lambda: budget * 2)
    calls = track_training_loads(monkeypatch)
    cache = a.PackedDecisionCache(paths, cfg)
    assert calls == paths["dev"]
    dev = paths["dev"][0]
    assert cache.get(dev) is cache.get(dev)
    policy = paths["policy"][0]
    cache.get(policy)
    cache.get(policy)
    assert calls == [dev, policy, policy]
    assert cache.bytes <= budget


def test_explicit_cache_budget_bounds_both_roles_together(tmp_path, monkeypatch):
    cfg = a.Config(hidden_dim=4, decision_cache_mb=1)
    paths = training_files(tmp_path, cfg)
    policy = paths["policy"][0]
    assert sum(path.stat().st_size for role in paths.values() for path in role) < 1024**2
    monkeypatch.setattr(a, "available_host_memory", lambda: 1024**4)
    calls = track_training_loads(monkeypatch)
    cache = a.PackedDecisionCache(paths, cfg)
    assert calls == paths["policy"] + paths["dev"]
    assert cache.bytes <= 1024**2
    cache.get(policy)
    cache.get(paths["dev"][0])
    assert calls == paths["policy"] + paths["dev"]


def test_cached_questions_preserve_row_order_rng_and_every_adam_update(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg = a.Config(hidden_dim=4, decision_hidden=4, decision_batch_size=3, decision_cache_mb=1)
    paths = training_files(tmp_path, cfg, policy_count=2)
    monkeypatch.setattr(a, "available_host_memory", lambda: 1024**4)
    torch.manual_seed(11)
    reference = a.SupervisedActionHead(cfg)
    optimized = copy.deepcopy(reference)
    old_opt = torch.optim.Adam(reference.parameters(), lr=cfg.decision_lr, weight_decay=cfg.weight_decay)
    new_opt = torch.optim.Adam(optimized.parameters(), lr=cfg.decision_lr, weight_decay=cfg.weight_decay)
    cache = a.PackedDecisionCache(paths, cfg)
    old_rng, new_rng = np.random.RandomState(31), np.random.RandomState(31)
    for _ in range(2):
        for head, opt, rng, loader in ((reference, old_opt, old_rng, a.load_checkpoint),
                                       (optimized, new_opt, new_rng, cache.get)):
            for path in rng.permutation(paths["policy"]):
                rows = loader(path)
                total, count = a.masked_action_loss(head, rows, rng.permutation(len(rows["h"])),
                                                     cfg, "cpu", opt)
                assert np.isfinite(total) and count == 9
    old_state, new_state = old_rng.get_state(), new_rng.get_state()
    assert old_state[0] == new_state[0] and old_state[2:] == new_state[2:]
    np.testing.assert_array_equal(old_state[1], new_state[1])
    assert all(torch.equal(value, optimized.state_dict()[key])
               for key, value in reference.state_dict().items())


def test_larger_evaluation_batches_preserve_all_rows_loss_and_weights():
    torch.set_num_threads(1)
    cfg = a.Config(hidden_dim=4, decision_hidden=4, decision_batch_size=3,
                   decision_eval_batch_size=4096)
    rows = packed_rows(cfg, 17)
    model = a.SupervisedActionHead(cfg)
    before = a.cpu_state(model)
    batches = []
    hook = model.register_forward_hook(lambda module, inputs, output: batches.append(len(inputs[0])))
    with torch.no_grad():
        old_total, old_count = a.masked_action_loss(model, rows, range(17), cfg, "cpu")
        assert batches == [3, 3, 3, 3, 3, 2]
        batches.clear()
        total, count = a.masked_action_loss(model, rows, range(17), cfg, "cpu",
                                          batch_size=cfg.decision_eval_batch_size)
    hook.remove()
    assert batches == [17] and old_count == count == 17
    assert total == pytest.approx(old_total, rel=1e-6, abs=1e-6)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert cfg.decision_batch_size == 3


def test_evaluation_batch_override_cannot_change_training_optimizer_steps():
    cfg = a.Config(hidden_dim=4, decision_batch_size=3)
    rows = packed_rows(cfg, 17)
    model = a.SupervisedActionHead(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.decision_lr)
    before = a.cpu_state(model)
    with pytest.raises(ValueError, match="preserve optimizer steps"):
        a.masked_action_loss(model, rows, range(17), cfg, "cpu", optimizer, batch_size=4096)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_runtime_storage_controls_preserve_dataset_fingerprint():
    cfg = a.Config(hidden_dim=4)
    records = [a.Record(str(i), str(i), "synthetic", list(range(cfg.n)), list(range(cfg.k)),
                        np.ones(cfg.k, np.float32), np.ones(cfg.k, np.float32),
                        np.ones((cfg.n, cfg.k), np.float32), np.ones(cfg.n, np.float32))
               for i in range(2)]
    teacher = {name: np.ones((2, cfg.n), np.float32) for name in ("scores", "safe", "maximum")}
    predictor = SimpleNamespace(model=torch.nn.Linear(4, 4), temperatures=np.ones(5))
    splits = {"policy": [0], "dev": [1]}
    first = a.decision_dataset_fingerprint(records, splits, teacher, predictor, cfg)
    changed = copy.deepcopy(cfg)
    changed.decision_cache_mb = 0
    changed.decision_eval_batch_size = 128
    assert a.decision_dataset_fingerprint(records, splits, teacher, predictor, changed) == first


def test_runtime_storage_controls_preserve_existing_workflow_contract(tmp_path):
    train = tmp_path / "train.json"
    train.write_text("[]", encoding="utf-8")
    cfg = a.Config(train_file=str(train), test_files=[], use_test_files_for_dev_and_audit=False,
                   output_dir=str(tmp_path / "run"), device="cpu", cpu_threads=1,
                   decision_cache_mb=0, decision_eval_batch_size=128)
    a.Workflow(cfg)
    contract = json.loads((tmp_path / "run" / "contract.json").read_text(encoding="utf-8"))
    assert not {"decision_cache_mb", "decision_eval_batch_size"} & set(contract["config"])
    cfg.decision_cache_mb = 4096
    cfg.decision_eval_batch_size = 4096
    a.Workflow(cfg)


def small_stage3_problem(**overrides):
    torch.set_num_threads(1)
    cfg = a.Config(k=2, zero_shots=2, hidden_dim=4, residual_blocks=1,
                   decision_hidden=4, decision_epochs=3, patience=3,
                   decision_batch_size=4, decision_cache_mb=0,
                   recognition_threshold=.1, **overrides)
    records = []
    for index in range(2):
        rng = np.random.RandomState(index + 40)
        records.append(a.Record(str(index), str(index), "synthetic",
                                ["zs_0", "zs_1", "r0", "r1"], ["r0", "r1"],
                                np.array([.9, .8], np.float32),
                                np.array([.4, .6], np.float32),
                                (rng.randint(6, size=(cfg.n, cfg.k)) / 5).astype(np.float32),
                                np.ones(cfg.n, np.float32)))
    scores = np.tile(np.arange(cfg.n, 0, -1, dtype=np.float32), (2, 1))
    maximum = np.zeros_like(scores)
    maximum[:, 0] = 1
    teacher = {"scores": scores, "safe": np.ones_like(scores), "maximum": maximum}
    torch.manual_seed(7)
    model = a.ResNet(cfg.input_dim, 3 * cfg.n + 2, cfg)
    predictor = a.FrozenPredictor({"weights": a.cpu_state(model), "temperatures": np.ones(5)},
                                  cfg, torch.device("cpu"))
    return cfg, records, {"supervised": [0], "policy": [0], "dev": [1]}, teacher, predictor


def assert_matching_training_checkpoints(reference, actual):
    assert actual["history"] == reference["history"]
    assert actual["best_epoch"] == reference["best_epoch"]
    assert actual["dev_loss"] == reference["dev_loss"]
    assert actual["coverage"] == reference["coverage"]
    assert actual["dataset_fingerprint"] == reference["dataset_fingerprint"]
    assert all(torch.equal(value, actual["weights"][key])
               for key, value in reference["weights"].items())


def test_interrupted_head_resumes_last_complete_epoch_with_identical_adam_and_shuffle(
        tmp_path, monkeypatch):
    cfg, records, splits, teacher, predictor = small_stage3_problem()
    complete, interrupted = tmp_path / "complete", tmp_path / "interrupted"
    complete.mkdir()
    interrupted.mkdir()
    expected = a.train_decision_head(records, splits, teacher, predictor, cfg, "cpu", complete)
    original = a.masked_action_loss
    train_calls = 0

    def stop_during_second_epoch(model, rows, order, config, device, optimizer=None, **kwargs):
        nonlocal train_calls
        if optimizer is not None:
            train_calls += 1
            if train_calls == 2:
                # Apply one partial-epoch update; resume must discard it and
                # restore the previous committed optimizer and shuffle states.
                original(model, rows, order[:config.decision_batch_size], config, device, optimizer)
                raise KeyboardInterrupt("simulated interrupted Stage 3 minibatch")
        return original(model, rows, order, config, device, optimizer, **kwargs)

    monkeypatch.setattr(a, "masked_action_loss", stop_during_second_epoch)
    with pytest.raises(KeyboardInterrupt, match="simulated interrupted"):
        a.train_decision_head(records, splits, teacher, predictor, cfg, "cpu", interrupted)
    progress = a.load_checkpoint(interrupted / "decision_head_progress.pt")
    assert progress["schema_version"] == 1 and progress["epoch"] == 1
    assert len(progress["history"]) == 1 and progress["optimizer"]["state"]
    monkeypatch.setattr(a, "masked_action_loss", original)
    monkeypatch.setattr(a, "build_decision_rows", lambda *args, **kwargs: pytest.fail("Saved labels must be reused"))
    actual = a.train_decision_head(records, splits, teacher, predictor, cfg, "cpu", interrupted)
    assert_matching_training_checkpoints(expected, actual)
    assert a.load_checkpoint(interrupted / "decision_head_progress.pt")["epoch"] == 3


@pytest.mark.parametrize("field,value,message", [
    ("schema_version", 99, "incompatible training contract"),
    ("fingerprint", "stale dataset", "incompatible training contract"),
    ("epoch", 99, "inconsistent epoch bookkeeping"),
])
def test_incompatible_or_inconsistent_head_progress_is_rejected(tmp_path, monkeypatch, field, value, message):
    cfg, records, splits, teacher, predictor = small_stage3_problem()
    a.train_decision_head(records, splits, teacher, predictor, cfg, "cpu", tmp_path)
    path = tmp_path / "decision_head_progress.pt"
    progress = a.load_checkpoint(path)
    progress[field] = value
    a.atomic_torch(path, progress)
    monkeypatch.setattr(a, "masked_action_loss", lambda *args, **kwargs: pytest.fail("Invalid progress must not train"))
    with pytest.raises(ValueError, match=message):
        a.train_decision_head(records, splits, teacher, predictor, cfg, "cpu", tmp_path)


def test_resuming_patience_exhausted_head_skips_every_training_epoch(tmp_path, monkeypatch):
    cfg, records, splits, teacher, predictor = small_stage3_problem()
    cfg.patience = 1
    original = a.masked_action_loss

    def constant_dev(model, rows, order, config, device, optimizer=None, **kwargs):
        if optimizer is None:
            return float(len(order)), len(order)
        return original(model, rows, order, config, device, optimizer, **kwargs)

    monkeypatch.setattr(a, "masked_action_loss", constant_dev)
    expected = a.train_decision_head(records, splits, teacher, predictor, cfg, "cpu", tmp_path)
    assert len(expected["history"]) == 2 < cfg.decision_epochs
    progress = a.load_checkpoint(tmp_path / "decision_head_progress.pt")
    assert progress["stale"] == cfg.patience
    monkeypatch.setattr(a, "masked_action_loss", lambda *args, **kwargs: pytest.fail("Stopped head must not restart epochs"))
    monkeypatch.setattr(a, "build_decision_rows", lambda *args, **kwargs: pytest.fail("Saved labels must be reused"))
    actual = a.train_decision_head(records, splits, teacher, predictor, cfg, "cpu", tmp_path)
    assert_matching_training_checkpoints(expected, actual)
