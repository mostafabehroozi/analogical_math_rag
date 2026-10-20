"""Offline checks for two-phase Stage 3 labels, their storage, and resumable head training."""
import copy
import itertools
import json
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
    return {"h": torch.from_numpy(rng.randn(count, cfg.head_input_dim).astype(np.float32)),
            "target": torch.from_numpy(target), "valid": torch.from_numpy(valid)}


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
    changed.decision_eval_batch_size = 128
    changed.continue_after_max = False
    assert a.decision_dataset_fingerprint(records, splits, teacher, predictor, changed) == first
    # Every training setting is part of the identity, including the fine-tuning ones.
    for name, value in (("decision_finetune_lr", 1e-3), ("decision_epochs", 7), ("recognition_threshold", .4)):
        changed = copy.deepcopy(cfg)
        setattr(changed, name, value)
        assert a.decision_dataset_fingerprint(records, splits, teacher, predictor, changed) != first
    # The fine-tuning labels also depend on the pretrained head they were rolled out from.
    head = a.SupervisedActionHead(cfg)
    other = a.SupervisedActionHead(cfg)
    assert a.finetune_fingerprint(first, a.cpu_state(head)) == a.finetune_fingerprint(first, a.cpu_state(head))
    assert a.finetune_fingerprint(first, a.cpu_state(head)) != a.finetune_fingerprint(first, a.cpu_state(other))


def test_runtime_storage_controls_preserve_existing_workflow_contract(tmp_path):
    train = tmp_path / "train.json"
    train.write_text("[]", encoding="utf-8")
    cfg = a.Config(train_file=str(train), test_files=[], use_test_files_for_dev_and_audit=False,
                   output_dir=str(tmp_path / "run"), device="cpu", cpu_threads=1,
                   decision_eval_batch_size=128)
    a.Workflow(cfg)
    contract = json.loads((tmp_path / "run" / "contract.json").read_text(encoding="utf-8"))
    assert not {"decision_eval_batch_size", "continue_after_max"} & set(contract["config"])
    assert {"decision_finetune_epochs", "decision_finetune_lr"} <= set(contract["config"])
    cfg.decision_eval_batch_size = 4096
    cfg.continue_after_max = False
    a.Workflow(cfg)


def small_stage3_problem(questions=2, **overrides):
    torch.set_num_threads(1)
    settings = dict(k=2, zero_shots=2, hidden_dim=4, residual_blocks=1, decision_hidden=4, decision_epochs=3,
                    decision_finetune_epochs=3, patience=3, decision_batch_size=4, recognition_threshold=.1)
    cfg = a.Config(**{**settings, **overrides})
    records = []
    for index in range(questions):
        rng = np.random.RandomState(index + 40)
        records.append(a.Record(str(index), str(index), "synthetic",
                                ["zs_0", "zs_1", "r0", "r1"], ["r0", "r1"],
                                np.array([.9, .8], np.float32),
                                np.array([.4, .6], np.float32),
                                (rng.randint(6, size=(cfg.n, cfg.k)) / 5).astype(np.float32),
                                np.ones(cfg.n, np.float32)))
    scores = np.tile(np.arange(cfg.n, 0, -1, dtype=np.float32), (questions, 1))
    maximum = np.zeros_like(scores)
    maximum[:, 0] = 1
    teacher = {"scores": scores, "safe": np.ones_like(scores), "maximum": maximum}
    torch.manual_seed(7)
    model = a.ResNet(cfg.input_dim, 3 * cfg.n + 2, cfg)
    predictor = a.FrozenPredictor({"weights": a.cpu_state(model), "temperatures": np.ones(5)},
                                  cfg, torch.device("cpu"))
    training = list(range(questions - 1))
    splits = {"supervised": training, "policy": training.copy(), "dev": [questions - 1]}
    return cfg, records, splits, teacher, predictor


def assert_matching_training_checkpoints(reference, actual):
    assert actual["history"] == reference["history"]
    assert actual["best_epoch"] == reference["best_epoch"]
    assert actual["dev_loss"] == reference["dev_loss"]
    assert actual["coverage"] == reference["coverage"]
    assert actual["dataset_fingerprint"] == reference["dataset_fingerprint"]
    assert all(torch.equal(value, actual["weights"][key])
               for key, value in reference["weights"].items())


PHASE_FILES = {"pretrain": "decision_head_pretrain_progress.pt", "finetune": "decision_head_finetune_progress.pt"}


def train_phase(phase, problem, out, pretrained=None, sync=None):
    """Run one Stage 3 phase; fine-tuning needs a pretrained checkpoint."""
    cfg, records, splits, teacher, predictor = problem
    if phase == "pretrain":
        return a.train_pretrained_head(records, splits, teacher, predictor, cfg, "cpu", out, sync)
    return a.train_finetuned_head(records, splits, teacher, predictor, pretrained, cfg, "cpu", out, sync)


@pytest.mark.parametrize("phase", ["pretrain", "finetune"])
@pytest.mark.parametrize("earlier_code", [False, True])
def test_interrupted_head_resumes_last_complete_epoch_with_identical_adam_and_shuffle(
        tmp_path, monkeypatch, phase, earlier_code):
    problem = small_stage3_problem()
    complete, interrupted = tmp_path / "complete", tmp_path / "interrupted"
    complete.mkdir()
    interrupted.mkdir()
    pretrained = train_phase("pretrain", problem, complete)
    expected = pretrained if phase == "pretrain" else train_phase("finetune", problem, complete, pretrained)
    if phase == "finetune":
        train_phase("pretrain", problem, interrupted)
        (interrupted / PHASE_FILES["pretrain"]).unlink()
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
        train_phase(phase, problem, interrupted, pretrained)
    progress = a.load_checkpoint(interrupted / PHASE_FILES[phase])
    assert progress["schema_version"] == a.HEAD_PROGRESS_SCHEMA and progress["phase"] == phase
    assert progress["epoch"] == 1 and progress["partial"] is None  # No timed save fell inside an epoch.
    assert len(progress["history"]) == 1 and progress["optimizer"]["state"]
    if earlier_code:  # A run that began before epochs could be resumed inside: completed epochs only.
        progress["schema_version"] = 2
        del progress["partial"]
        a.atomic_torch(interrupted / PHASE_FILES[phase], progress)
    monkeypatch.setattr(a, "masked_action_loss", original)
    for builder in ("build_route_labels", "build_finetune_labels"):
        monkeypatch.setattr(a, builder, lambda *args, **kwargs: pytest.fail("Saved labels must be reused"))
    actual = train_phase(phase, problem, interrupted, pretrained)
    assert_matching_training_checkpoints(expected, actual)
    assert a.load_checkpoint(interrupted / PHASE_FILES[phase])["epoch"] == 3


class RecordingSync:
    """Stands in for the Hub mirror: records each upload's message and whether it was forced."""
    def __init__(self):
        self.pushes = []

    def push(self, message, *, force=False, required=False):
        self.pushes.append((message, force))
        return True


def mid_epoch_problem():
    """Three training and two development questions, so both passes of an epoch have an inside."""
    cfg, records, splits, teacher, predictor = small_stage3_problem(questions=5)
    splits = {"supervised": [0, 1, 2], "policy": [0, 1, 2], "dev": [3, 4]}
    return cfg, records, splits, teacher, predictor


def save_at_every_question(monkeypatch, cfg):
    """A clock that advances a minute per reading, so every question boundary is a timed save."""
    clock = itertools.count()
    monkeypatch.setattr(a.time, "monotonic", lambda: 60. * next(clock))
    cfg.hf_sync_minutes = 1


def end_session_at_question(monkeypatch, stop):
    """The session ends as the head starts its `stop`-th question, training and development counted."""
    original, calls = a.masked_action_loss, 0

    def session_ends(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == stop:
            raise KeyboardInterrupt("simulated session end inside an epoch")
        return original(*args, **kwargs)

    monkeypatch.setattr(a, "masked_action_loss", session_ends)
    return original


@pytest.mark.parametrize("phase", ["pretrain", "finetune"])
@pytest.mark.parametrize("stop,epoch,train_done,dev_done", [
    (2, 0, 1, 0),  # Second training question of the first epoch: nothing completed yet.
    (5, 0, 3, 1),  # Second development question of the first epoch.
    (7, 1, 1, 0),  # Second training question of the second epoch.
])
def test_head_interrupted_inside_an_epoch_resumes_at_the_next_question(
        tmp_path, monkeypatch, capsys, phase, stop, epoch, train_done, dev_done):
    problem = mid_epoch_problem()
    complete, interrupted = tmp_path / "complete", tmp_path / "interrupted"
    complete.mkdir()
    interrupted.mkdir()
    pretrained = train_phase("pretrain", problem, complete)
    expected = pretrained if phase == "pretrain" else train_phase("finetune", problem, complete, pretrained)
    if phase == "finetune":
        train_phase("pretrain", problem, interrupted)
    save_at_every_question(monkeypatch, problem[0])
    original = end_session_at_question(monkeypatch, stop)
    sync = RecordingSync()
    with pytest.raises(KeyboardInterrupt, match="inside an epoch"):
        train_phase(phase, problem, interrupted, pretrained, sync)
    progress = a.load_checkpoint(interrupted / PHASE_FILES[phase])
    assert progress["schema_version"] == a.HEAD_PROGRESS_SCHEMA and progress["epoch"] == epoch
    assert (progress["partial"]["train_done"], progress["partial"]["dev_done"]) == (train_done, dev_done)
    # Each timed save is uploaded at once; the last names the pass and question it reached.
    title = a.HEAD_PHASES[phase]["title"]
    saves = [push for push in sync.pushes if "epoch" in push[0]]
    pass_name, done, total = ("development", dev_done, 2) if dev_done else ("training", train_done, 3)
    assert saves[-1] == (f"{title}: epoch {epoch + 1} {pass_name}, {done}/{total} questions", True)
    assert all(force for _, force in saves)
    monkeypatch.setattr(a, "masked_action_loss", original)
    for builder in ("build_route_labels", "build_finetune_labels"):
        monkeypatch.setattr(a, builder, lambda *args, **kwargs: pytest.fail("Saved labels must be reused"))
    capsys.readouterr()
    actual = train_phase(phase, problem, interrupted, pretrained)
    assert (f"{title} head resume: {epoch} completed epochs and epoch {epoch + 1} at {train_done}/3 training "
            f"and {dev_done}/2 development questions") in capsys.readouterr().out
    assert_matching_training_checkpoints(expected, actual)
    assert a.load_checkpoint(interrupted / PHASE_FILES[phase])["partial"] is None


@pytest.mark.parametrize("field,value", [
    ("train_done", 4),                    # More questions than the training role has.
    ("dev_done", 1),                      # Development questions before training finished.
    ("order", np.array([0, 0, 1])),       # Not an order of the training questions.
])
def test_inconsistent_epoch_in_progress_is_rejected(tmp_path, monkeypatch, field, value):
    problem = mid_epoch_problem()
    save_at_every_question(monkeypatch, problem[0])
    end_session_at_question(monkeypatch, 2)
    with pytest.raises(KeyboardInterrupt, match="inside an epoch"):
        train_phase("pretrain", problem, tmp_path)
    path = tmp_path / PHASE_FILES["pretrain"]
    progress = a.load_checkpoint(path)
    progress["partial"][field] = value
    a.atomic_torch(path, progress)
    monkeypatch.setattr(a, "masked_action_loss", lambda *args, **kwargs: pytest.fail("Invalid progress must not train"))
    with pytest.raises(ValueError, match="inconsistent epoch bookkeeping"):
        train_phase("pretrain", problem, tmp_path)


@pytest.mark.parametrize("phase", ["pretrain", "finetune"])
@pytest.mark.parametrize("field,value,message", [
    ("schema_version", 99, "incompatible training contract"),
    ("phase", "other", "incompatible training contract"),
    ("fingerprint", "stale dataset", "incompatible training contract"),
    ("epoch", 99, "inconsistent epoch bookkeeping"),
])
def test_incompatible_or_inconsistent_head_progress_is_rejected(tmp_path, monkeypatch, phase, field, value, message):
    problem = small_stage3_problem()
    pretrained = train_phase("pretrain", problem, tmp_path)
    if phase == "finetune":
        train_phase("finetune", problem, tmp_path, pretrained)
    path = tmp_path / PHASE_FILES[phase]
    progress = a.load_checkpoint(path)
    progress[field] = value
    a.atomic_torch(path, progress)
    monkeypatch.setattr(a, "masked_action_loss", lambda *args, **kwargs: pytest.fail("Invalid progress must not train"))
    with pytest.raises(ValueError, match=message):
        train_phase(phase, problem, tmp_path, pretrained)


@pytest.mark.parametrize("phase", ["pretrain", "finetune"])
def test_resuming_patience_exhausted_head_skips_every_training_epoch(tmp_path, monkeypatch, phase):
    problem = small_stage3_problem()
    problem[0].patience = 1
    original = a.masked_action_loss

    def constant_dev(model, rows, order, config, device, optimizer=None, **kwargs):
        if optimizer is None:
            return float(len(order)), len(order)
        return original(model, rows, order, config, device, optimizer, **kwargs)

    monkeypatch.setattr(a, "masked_action_loss", constant_dev)
    pretrained = train_phase("pretrain", problem, tmp_path)
    expected = pretrained if phase == "pretrain" else train_phase("finetune", problem, tmp_path, pretrained)
    assert len(expected["history"]) == 2 < problem[0].decision_epochs
    assert a.load_checkpoint(tmp_path / PHASE_FILES[phase])["stale"] == problem[0].patience
    monkeypatch.setattr(a, "masked_action_loss", lambda *args, **kwargs: pytest.fail("Stopped head must not restart epochs"))
    for builder in ("build_route_labels", "build_finetune_labels"):
        monkeypatch.setattr(a, builder, lambda *args, **kwargs: pytest.fail("Saved labels must be reused"))
    assert_matching_training_checkpoints(expected, train_phase(phase, problem, tmp_path, pretrained))


def test_stage3_logs_progress_and_reuses_saved_labels_on_resume(tmp_path, monkeypatch, capsys):
    problem = small_stage3_problem()
    cfg, records, splits, teacher, predictor = problem
    cfg.decision_epochs = 1
    first = train_phase("pretrain", problem, tmp_path)
    log = capsys.readouterr().out
    assert "Stage 3a dev labels: 1/1" in log and "Stage 3a epoch 1 train: 1/1" in log
    assert "Stage 3a epoch 1 dev: 1/1" in log and "Stage 3a epoch 1/1: train loss" in log
    assert "dev loss" in log and "ETA" in log and "head rows" in log and "cheapest first steps" in log
    # Resume saved role labels: no relabeling, and an interrupted write is cleaned up.
    interrupted = a.decision_role_path(tmp_path, "pretrain", "dev").with_suffix(".pt.tmp")
    interrupted.write_bytes(b"leftover from failed dev label write")
    monkeypatch.setattr(a, "build_route_labels", lambda *args: pytest.fail("Should reuse saved labels"))
    resumed = train_phase("pretrain", problem, tmp_path)
    assert not interrupted.exists()
    assert "built 0, reused 1" in capsys.readouterr().out
    assert first["coverage"] == resumed["coverage"] and first["history"] == resumed["history"]
    assert all(torch.equal(value, resumed["weights"][key]) for key, value in first["weights"].items())
    coverage = json.loads((tmp_path / "decision_label_coverage.json").read_text(encoding="utf-8"))
    assert set(coverage["pretrain"]) == {"policy", "dev"}


def test_finetuning_starts_from_the_pretrained_weights_and_is_bound_to_that_head(tmp_path):
    problem = small_stage3_problem(decision_finetune_epochs=1, decision_finetune_lr=1e-12)
    cfg, records, splits, teacher, predictor = problem
    pretrained = train_phase("pretrain", problem, tmp_path)
    # A negligible learning rate leaves the weights where fine-tuning started: at the pretrained head.
    nearly_unchanged = train_phase("finetune", problem, tmp_path, pretrained)
    assert all(torch.allclose(value, pretrained["weights"][key], atol=1e-6)
               for key, value in nearly_unchanged["weights"].items())
    assert nearly_unchanged["phase"] == "finetune" and pretrained["phase"] == "pretrain"
    assert nearly_unchanged["pretrained_fingerprint"] == a.head_fingerprint(pretrained["weights"])
    # Saved fine-tuning labels were rolled out from that head and are rejected for another.
    other = copy.deepcopy(pretrained)
    other["weights"] = {key: value + 1e-3 for key, value in other["weights"].items()}
    with pytest.raises(ValueError, match="Incompatible decision labels"):
        train_phase("finetune", problem, tmp_path, other)
    foreign = dict(pretrained, dataset_fingerprint="another dataset")
    with pytest.raises(ValueError, match="different dataset or action contract"):
        train_phase("finetune", problem, tmp_path, foreign)


def test_finetuning_labels_cover_exactly_the_pretraining_rows_and_choose_allowed_actions(tmp_path):
    problem = small_stage3_problem(questions=3)
    cfg, records, splits, teacher, predictor = problem
    pretrained = train_phase("pretrain", problem, tmp_path)
    finetuned = train_phase("finetune", problem, tmp_path, pretrained)
    for role in ("policy", "dev"):
        options = a.load_label_file(a.decision_role_path(tmp_path, "pretrain", role))["options"]
        choice = a.load_label_file(a.decision_role_path(tmp_path, "finetune", role))["choice"]
        for position in range(len(options)):
            rows = a.label_rows(options[position].numpy(), "pretrain")
            assert np.array_equal(rows, a.label_rows(choice[position].numpy(), "finetune")) and len(rows)
            assert a.decision_valid(rows, cfg)[np.arange(len(rows)), choice[position].numpy()[rows]].all()
        assert finetuned["coverage"][role]["action_rows"] == pretrained["coverage"][role]["action_rows"]
        coverage = finetuned["coverage"][role]
        # Several cheapest first steps are what Stage 3b decides; "changed" also counts head mistakes.
        assert 0 <= coverage["tied_rows_moved"] <= coverage["tied_rows"] <= coverage["informative_rows"]
        assert coverage["tied_rows_moved"] <= coverage["changed_rows"] <= coverage["action_rows"]
        assert coverage["no_thread_reaches_rows"] <= coverage["informative_rows"]
        assert coverage["tied_rows"] >= sum(count for options, count in coverage["option_counts"].items()
                                            if int(options) > 1)
        assert "tied_rows" not in pretrained["coverage"][role]


def one_question_labels(problem, index=0):
    cfg, records, splits, teacher, predictor = problem
    args = (records[index], teacher["scores"][index], teacher["safe"][index], teacher["maximum"][index], predictor)
    return (*args, cfg), a.build_route_labels(*args, cfg)


def test_pretraining_labels_rebuild_exact_targets_valid_masks_and_features():
    problem = small_stage3_problem()
    cfg, records, _, teacher, predictor = problem
    args, labels = one_question_labels(problem)
    record = records[0]
    assert set(labels) == {"options", "observation_sha256", "coverage"}
    states, _ = a.budget_state_index(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    orders = list(itertools.permutations(range(cfg.zero_shots)))
    assert labels["options"].dtype == torch.uint8
    assert labels["options"].shape == (len(orders) * len(states), (cfg.action_count + 7) // 8)
    source = a.check_route_labels(record, labels, cfg)
    assert len(source) == labels["coverage"]["action_rows"]
    blank = np.zeros(cfg.n, np.float32)
    observed = a.decision_row_observations(record, source, cfg)
    for x, value in zip(observed, source.tolist()):
        scenario, position = divmod(value, len(states))
        permuted = a.permute_zero_shots(record, blank, blank, blank, orders[scenario], cfg)[0]
        # The scalar observation is an independent oracle for the batched rebuild.
        np.testing.assert_array_equal(x, a.observation(permuted, states[position], cfg))
    plan = a.question_plan(*args)
    rebuilt = a.decision_training_rows(record, source, labels["options"].numpy()[source], "pretrain", predictor, cfg)
    torch.testing.assert_close(rebuilt["h"].cpu(), torch.from_numpy(plan["features"].reshape(-1, cfg.head_input_dim)[source]),
                               rtol=1e-5, atol=1e-5)
    # Targets and masks are decoded bit for bit: an even split over each row's cheapest first steps.
    targets = plan["targets"].reshape(-1, cfg.action_count)[source]
    expected = (targets / targets.sum(1, keepdims=True)).astype(np.float32)
    np.testing.assert_array_equal(rebuilt["target"].numpy(), expected)
    np.testing.assert_array_equal(rebuilt["valid"].numpy(), plan["graph"].legal[source % len(states)])
    assert not rebuilt["target"].numpy()[~rebuilt["valid"].numpy()].any()
    # Fine-tuning rows are one-hot at the chosen action.
    chosen = np.arange(len(source)) % cfg.action_count
    one_hot = a.decision_targets(chosen, "finetune", cfg)
    assert one_hot.shape == (len(source), cfg.action_count) and (one_hot.sum(1) == 1).all()
    assert (one_hot.argmax(1) == chosen).all()


def test_saved_labels_reject_observations_that_changed_after_labeling():
    problem = small_stage3_problem()
    cfg, records, _, teacher, predictor = problem
    _, labels = one_question_labels(problem)
    a.check_route_labels(records[0], labels, cfg)
    changed = copy.deepcopy(records[0])
    changed.ccs[0, 0] = 1 - changed.ccs[0, 0]
    with pytest.raises(ValueError, match="no longer reproduce"):
        a.check_route_labels(changed, labels, cfg)
    with pytest.raises(ValueError, match="inconsistent shapes"):
        a.check_route_labels(records[0], dict(labels, options=labels["options"][:-1]), cfg)
    with pytest.raises(ValueError, match="inconsistent shapes"):
        a.check_route_labels(records[0], dict(labels, options=labels["options"].to(torch.int16)), cfg)
    padded = labels["options"].clone()
    padded[0, -1] |= 1  # Bit 7 of the only byte is not an action.
    with pytest.raises(ValueError, match="inconsistent shapes"):
        a.check_route_labels(records[0], dict(labels, options=padded), cfg)
    # A bit for an action that is not allowed in that state, and a lost row.
    graph = a.decision_graph(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    state, action = np.argwhere(~graph.legal & graph.legal.any(1)[:, None])[0]
    bits = np.unpackbits(labels["options"].numpy(), axis=1)
    bits[state, action] = 1
    with pytest.raises(ValueError, match="invalid actions"):
        a.check_route_labels(records[0], dict(labels, options=torch.from_numpy(np.packbits(bits, axis=1))), cfg)
    lost = labels["options"].clone()
    lost[np.flatnonzero(lost.numpy().any(1))[0]] = 0
    with pytest.raises(ValueError, match="invalid actions"):
        a.check_route_labels(records[0], dict(labels, options=lost), cfg)


def test_saved_fine_tuning_labels_reject_invalid_choices_and_misaligned_rows():
    problem = small_stage3_problem()
    cfg, records, _, teacher, predictor = problem
    args, route = one_question_labels(problem)
    head = a.SupervisedActionHead(cfg)
    labels = a.build_finetune_labels(*args[:5], head, cfg)
    assert set(labels) == {"choice", "coverage"} and labels["choice"].dtype == torch.uint8
    a.check_finetune_labels(records[0], labels, cfg, route["options"])
    with pytest.raises(ValueError, match="inconsistent shapes"):
        a.check_finetune_labels(records[0], dict(labels, choice=labels["choice"][:-1]), cfg)
    rows = np.flatnonzero(labels["choice"].numpy() != a.NO_CHOICE)
    for bad in (cfg.action_count, 200):
        invalid = labels["choice"].clone()
        invalid[rows[0]] = bad
        with pytest.raises(ValueError, match="choose invalid actions"):
            a.check_finetune_labels(records[0], dict(labels, choice=invalid), cfg)
    # A choice the state does not allow.
    legal = a.decision_valid(rows, cfg)
    row, action = next((r, int(np.flatnonzero(~mask)[0])) for r, mask in enumerate(legal) if not mask.all())
    illegal = labels["choice"].clone()
    illegal[rows[row]] = action
    with pytest.raises(ValueError, match="choose invalid actions"):
        a.check_finetune_labels(records[0], dict(labels, choice=illegal), cfg)
    # Rows that Stage 3a does not have.
    moved = labels["choice"].clone()
    moved[rows[0]] = a.NO_CHOICE
    moved[np.flatnonzero(labels["choice"].numpy() == a.NO_CHOICE)[0]] = 0
    with pytest.raises(ValueError, match="choose invalid actions|do not cover"):
        a.check_finetune_labels(records[0], dict(labels, choice=moved), cfg, route["options"])


def test_default_geometry_question_labels_stay_small(tmp_path):
    torch.set_num_threads(1)
    cfg = a.Config()
    rng = np.random.RandomState(3)
    record = a.Record("synthetic::0", "0", "synthetic",
                      [f"zs_{i}" for i in range(cfg.zero_shots)] + [f"os_{i}" for i in range(cfg.k)],
                      [str(i) for i in range(cfg.k)],
                      np.sort(rng.rand(cfg.k).astype(np.float32))[::-1].copy(),
                      (rng.randint(6, size=cfg.k) / 5).astype(np.float32),
                      (rng.randint(6, size=(cfg.n, cfg.k)) / 5).astype(np.float32),
                      (rng.rand(cfg.n) > .5).astype(np.float32))
    scores = rng.rand(cfg.n).astype(np.float32)
    safe, maximum = a.teacher_labels(scores, record.labels, cfg)
    torch.manual_seed(7)
    model = a.ResNet(cfg.input_dim, 3 * cfg.n + 2, cfg)
    predictor = a.FrozenPredictor({"weights": a.cpu_state(model), "temperatures": np.ones(5)},
                                  cfg, torch.device("cpu"))
    route = a.build_route_labels(record, scores, safe, maximum, predictor, cfg)
    fine = a.build_finetune_labels(record, scores, safe, maximum, predictor, a.SupervisedActionHead(cfg), cfg)
    cases = a.decision_case_count(cfg)
    assert cases == 6 * 8160  # Every nonempty candidate subset with every evaluator subset.
    paths = {}
    for name, labels in (("route", route), ("fine", fine)):
        paths[name] = tmp_path / f"{name}.pt"
        a.atomic_torch(paths[name], {"schema_version": a.DECISION_LABEL_SCHEMA, "fingerprint": "f" * 64,
                                     "uid": record.uid, "phase": name, **labels})
    # Pickled audit rows once took about 1 MiB per question and stored features 8 MiB, which filled
    # Kaggle's disk. Now a case costs two bytes of first-step bits plus one byte of choice.
    rows = route["coverage"]["action_rows"]
    assert 30000 < rows < cases  # Only the complete state, and unlabeled group mates, lack a row.
    assert route["options"].shape == (cases, 2) and fine["choice"].shape == (cases,)
    assert paths["route"].stat().st_size < 2 * cases + 4 * 1024
    assert paths["fine"].stat().st_size < cases + 4 * 1024
    assert route["coverage"]["state_order_cases"] == cases == fine["coverage"]["state_order_cases"]
    assert fine["coverage"]["action_rows"] == rows
    a.check_route_labels(record, route, cfg)
    a.check_finetune_labels(record, fine, cfg, route["options"])


def count_builds(monkeypatch, name):
    built = []
    original = getattr(a, name)

    def counted(record, *rest, **kwargs):
        built.append(record.uid)
        return original(record, *rest, **kwargs)

    monkeypatch.setattr(a, name, counted)
    return built, original


def test_interrupted_label_build_reuses_completed_questions_then_merges_each_role(tmp_path, monkeypatch, capsys):
    problem = small_stage3_problem(questions=4)
    cfg, records, splits, teacher, predictor = problem
    expected = train_phase("pretrain", problem, tmp_path / "complete")
    run = tmp_path / "interrupted"
    original = a.build_route_labels
    calls = 0

    def stop_on_second_question(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("simulated interrupted Stage 3 label build")
        return original(*args, **kwargs)

    monkeypatch.setattr(a, "build_route_labels", stop_on_second_question)
    with pytest.raises(KeyboardInterrupt, match="simulated interrupted"):
        train_phase("pretrain", problem, run)
    assert len(list((run / "decision_labels" / "pretrain_policy").glob("*.pt"))) == 1
    assert not a.decision_role_path(run, "pretrain", "policy").exists()
    monkeypatch.setattr(a, "build_route_labels", original)
    capsys.readouterr()
    built, _ = count_builds(monkeypatch, "build_route_labels")
    actual = train_phase("pretrain", problem, run)
    log = capsys.readouterr().out
    assert built == [records[i].uid for i in splits["policy"][1:] + splits["dev"]]
    assert "Stage 3a policy labels: 3/3 questions" in log and "built 2, reused 1" in log
    # Completed roles are single files; per-question files are removed after merging.
    assert sorted(path.name for path in (run / "decision_labels").iterdir()) == ["pretrain_dev.pt", "pretrain_policy.pt"]
    store = a.load_label_file(a.decision_role_path(run, "pretrain", "policy"))
    assert store["uids"] == [records[i].uid for i in splits["policy"]]
    rows = sum(len(a.label_rows(options.numpy(), "pretrain")) for options in store["options"])
    assert rows == expected["coverage"]["policy"]["action_rows"]
    assert_matching_training_checkpoints(expected, actual)
    # Stage 3b labels are built the same way, one question at a time.
    built, _ = count_builds(monkeypatch, "build_finetune_labels")
    train_phase("finetune", problem, run, actual)
    assert built == [records[i].uid for i in splits["policy"] + splits["dev"]]
    assert sorted(path.name for path in (run / "decision_labels").iterdir()) == [
        "finetune_dev.pt", "finetune_policy.pt", "pretrain_dev.pt", "pretrain_policy.pt"]


def test_incompatible_label_manifest_or_store_is_never_reused(tmp_path):
    problem = small_stage3_problem()
    cfg, records, splits, teacher, predictor = problem
    train_phase("pretrain", problem, tmp_path)
    manifest_path = tmp_path / "decision_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == a.DECISION_LABEL_SCHEMA and manifest["implementation_revision"] == a.PIPELINE_REVISION
    manifest["fingerprint"] = "stale teacher or predictor"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible teacher, predictor"):
        train_phase("pretrain", problem, tmp_path)
    manifest["fingerprint"] = a.decision_dataset_fingerprint(records, splits, teacher, predictor, cfg)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    path = a.decision_role_path(tmp_path, "pretrain", "dev")
    store = a.load_label_file(path)
    for field, value in (("phase", "finetune"), ("role", "policy"), ("uids", ["another question"]),
                         ("schema_version", 5), ("fingerprint", "stale")):
        a.atomic_torch(path, dict(store, **{field: value}))
        with pytest.raises(ValueError, match="Incompatible decision labels"):
            train_phase("pretrain", problem, tmp_path)
    a.atomic_torch(path, store)
    train_phase("pretrain", problem, tmp_path)


def test_question_audit_recomputes_the_saved_cases(tmp_path):
    problem = small_stage3_problem()
    cfg, records, splits, teacher, predictor = problem
    checkpoint = train_phase("pretrain", problem, tmp_path)
    audit = a.decision_question_audit(records, splits, teacher, predictor, cfg, tmp_path, "dev", 0)
    args = (records[1], teacher["scores"][1], teacher["safe"][1], teacher["maximum"][1], predictor, cfg)
    plan = a.question_plan(*args)
    assert audit["uid"] == records[1].uid and len(audit["cases"]) == plan["status"].size
    assert sum(case["has_row"] for case in audit["cases"]) == plan["rows"].sum()
    assert audit["coverage"]["action_rows"] == checkpoint["coverage"]["dev"]["action_rows"]
    names = a.action_names(cfg)
    case = next(case for case in audit["cases"] if case["has_row"])
    assert set(case["targets"]) <= set(names) and case["targets"]
    assert next(case for case in audit["cases"] if case["status"] == "EXHAUSTED")["has_row"] is False
    path = a.decision_role_path(tmp_path, "pretrain", "dev")
    store = a.load_label_file(path)
    store["options"] = torch.zeros_like(store["options"])
    a.atomic_torch(path, store)
    with pytest.raises(ValueError, match="differ from the saved labels"):
        a.decision_question_audit(records, splits, teacher, predictor, cfg, tmp_path, "dev", 0)


def test_workflow_reclaims_interrupted_stage3_writes_before_stage0_writes(tmp_path, capsys):
    train = tmp_path / "train.json"
    train.write_text("[]", encoding="utf-8")
    cfg = a.Config(train_file=str(train), test_files=[], use_test_files_for_dev_and_audit=False,
                   output_dir=str(tmp_path / "run"), device="cpu", cpu_threads=1)
    a.Workflow(cfg)
    out = tmp_path / "run"
    # Interrupted atomic writes from a run that filled its disk.
    derived = [out / "decision_labels" / "pretrain_policy" / "000002_abc.pt.tmp",
               out / "decision_labels" / "pretrain_policy.pt.tmp",
               out / "decision_labels" / "finetune_dev.pt.tmp",
               out / "decision_head_pretrain_progress.pt.tmp",
               out / "compact_records.pt.tmp"]
    # Completed labels and checkpoints hold finished work and stay.
    kept = [out / "decision_labels" / "pretrain_policy" / "000003_abc.pt",
            out / "decision_labels" / "pretrain_dev.pt",
            out / "decision_labels" / "finetune_policy.pt",
            out / "decision_head_pretrain_progress.pt",
            out / "decision_head_finetune_progress.pt"]
    for path in derived + kept:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 1024)
    capsys.readouterr()
    a.Workflow(cfg)
    log = capsys.readouterr().out
    assert f"Reclaimed {len(derived) * 1024 / 1024**2:.2f} MiB" in log and "before resuming" in log
    assert not any(path.exists() for path in derived)
    assert all(path.exists() for path in kept)
    assert a.reclaim_derived_stage3_files(out) == 0
    capsys.readouterr()
    a.Workflow(cfg)
    assert "Reclaimed" not in capsys.readouterr().out


def test_stage3_fails_loudly_when_no_state_has_an_affordable_acquisition(tmp_path):
    # A budget of one call leaves no acquisition anywhere, so there is nothing to train on.
    problem = small_stage3_problem(max_cost=1.)
    with pytest.raises(ValueError, match="No actionable Stage 3 rows"):
        train_phase("pretrain", problem, tmp_path)


def test_reused_labels_are_checked_against_the_current_rules_and_data(tmp_path):
    problem = small_stage3_problem()
    cfg, records, splits, teacher, predictor = problem
    train_phase("pretrain", problem, tmp_path)
    path = a.decision_role_path(tmp_path, "pretrain", "dev")
    store = a.load_label_file(path)
    graph = a.decision_graph(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    state, action = np.argwhere(~graph.legal & graph.legal.any(1)[:, None])[0]
    bits = np.unpackbits(store["options"][0].numpy(), axis=1)
    bits[state, action] = 1  # An acquisition the rules do not allow in that state.
    tampered = dict(store, options=store["options"].clone())
    tampered["options"][0] = torch.from_numpy(np.packbits(bits, axis=1))
    a.atomic_torch(path, tampered)
    with pytest.raises(ValueError, match="invalid actions"):
        train_phase("pretrain", problem, tmp_path)
    # Changed measurements change the dataset identity, which the manifest refuses before any label is read.
    a.atomic_torch(path, store)
    changed = copy.deepcopy(records)
    changed[1].ccs[0, 0] = 1 - changed[1].ccs[0, 0]
    with pytest.raises(ValueError, match="incompatible teacher, predictor, split, or action contract"):
        a.train_pretrained_head(changed, splits, teacher, predictor, cfg, "cpu", tmp_path)
    train_phase("pretrain", problem, tmp_path)  # The untouched labels are reused as before.
