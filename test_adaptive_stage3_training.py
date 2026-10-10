"""Offline checks for compact Stage 3 labels, migration, and unchanged training updates."""
import copy
import gzip
import itertools
import json
import math
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


def test_runtime_storage_controls_preserve_existing_workflow_contract(tmp_path):
    train = tmp_path / "train.json"
    train.write_text("[]", encoding="utf-8")
    cfg = a.Config(train_file=str(train), test_files=[], use_test_files_for_dev_and_audit=False,
                   output_dir=str(tmp_path / "run"), device="cpu", cpu_threads=1,
                   decision_eval_batch_size=128)
    a.Workflow(cfg)
    contract = json.loads((tmp_path / "run" / "contract.json").read_text(encoding="utf-8"))
    assert not {"decision_eval_batch_size", "continue_after_max"} & set(contract["config"])
    cfg.decision_eval_batch_size = 4096
    cfg.continue_after_max = False
    a.Workflow(cfg)


def small_stage3_problem(questions=2, **overrides):
    torch.set_num_threads(1)
    cfg = a.Config(k=2, zero_shots=2, hidden_dim=4, residual_blocks=1,
                   decision_hidden=4, decision_epochs=3, patience=3,
                   decision_batch_size=4,
                   recognition_threshold=.1, **overrides)
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
    monkeypatch.setattr(a, "build_decision_labels", lambda *args, **kwargs: pytest.fail("Saved labels must be reused"))
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
    monkeypatch.setattr(a, "build_decision_labels", lambda *args, **kwargs: pytest.fail("Saved labels must be reused"))
    actual = a.train_decision_head(records, splits, teacher, predictor, cfg, "cpu", tmp_path)
    assert_matching_training_checkpoints(expected, actual)


def question_labels(record, rows, cases, cfg):
    return {"schema_version": a.DECISION_LABEL_SCHEMA, "fingerprint": "f" * 64, "uid": record.uid,
            "coverage": a.label_coverage(rows, cases, cfg),
            **a.compact_decision_labels(record, rows, cases, cfg)}


def test_compact_labels_rebuild_exact_targets_valid_masks_and_features():
    cfg, records, _, teacher, predictor = small_stage3_problem()
    record = records[0]
    rows, cases = a.build_decision_rows(record, teacher["scores"][0], teacher["safe"][0],
                                        teacher["maximum"][0], predictor, cfg)
    labels = a.compact_decision_labels(record, rows, cases, cfg)
    assert set(labels) == {"preferred", "first_cases", "observation_sha256"}
    assert labels["preferred"].dtype == labels["first_cases"].dtype == torch.uint8
    states, _ = a.budget_state_index(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    orders = list(itertools.permutations(range(cfg.zero_shots)))
    assert labels["preferred"].shape == (len(rows), 1)
    assert labels["first_cases"].shape == ((len(orders) * len(states) + 7) // 8,)
    # Rows keep planning order, so the bitmap of first cases alone rebuilds every source.
    first = {}
    for case in cases:
        if case["training_row"] is not None:
            first.setdefault(case["training_row"], case["scenario"] * len(states) + states.index(
                a.State(case["candidate_mask"], case["evaluator_mask"])))
    source = a.decision_row_sources(labels["first_cases"].numpy(), cfg)
    assert source.tolist() == [first[row] for row in range(len(rows))]
    blank = np.zeros(cfg.n, np.float32)
    observed = a.decision_row_observations(record, source, cfg)
    for x, value in zip(observed, source.tolist()):
        scenario, position = divmod(value, len(states))
        permuted = a.permute_zero_shots(record, blank, blank, blank, orders[scenario], cfg)[0]
        # The scalar observation is an independent oracle for the batched rebuild.
        np.testing.assert_array_equal(x, a.observation(permuted, states[position], cfg))
    rebuilt = a.decision_training_rows(record, source, labels["preferred"].numpy(), predictor, cfg)
    stored = a.pack_decision_rows(rows, cfg)
    torch.testing.assert_close(rebuilt["h"], stored["h"], rtol=1e-6, atol=1e-6)
    # Targets and masks are decoded bit for bit, not approximately.
    assert all(rebuilt[key].dtype == stored[key].dtype and torch.equal(rebuilt[key], stored[key])
               for key in ("target", "valid"))


def test_compact_labels_reject_rows_that_do_not_reproduce_their_state_or_target():
    cfg, records, _, teacher, predictor = small_stage3_problem()
    rows, cases = a.build_decision_rows(records[0], teacher["scores"][0], teacher["safe"][0],
                                        teacher["maximum"][0], predictor, cfg)
    tampered = copy.deepcopy(rows)
    tampered[0]["state_key"] = (*tampered[0]["state_key"][:2], "0" * 64)
    with pytest.raises(ValueError, match="do not reproduce"):
        a.compact_decision_labels(records[0], tampered, cases, cfg)
    # Another question's measurements cannot stand in for this question's rows.
    with pytest.raises(ValueError, match="do not reproduce"):
        a.compact_decision_labels(records[1], rows, cases, cfg)
    orphaned = [dict(case, training_row=None) if case["training_row"] == 0 else case for case in cases]
    with pytest.raises(ValueError, match="without an in-budget visible case"):
        a.compact_decision_labels(records[0], rows, orphaned, cfg)
    uneven = copy.deepcopy(rows)
    row = next(row for row in uneven if row["valid"].sum() >= 2)
    support = np.flatnonzero(row["valid"])
    row["target"] = np.zeros(cfg.action_count, np.float32)
    row["target"][support[:2]] = [.75, .25]
    with pytest.raises(ValueError, match="split evenly"):
        a.compact_decision_labels(records[0], uneven, cases, cfg)


def test_saved_labels_reject_observations_that_changed_after_labeling():
    cfg, records, _, teacher, predictor = small_stage3_problem()
    rows, cases = a.build_decision_rows(records[0], teacher["scores"][0], teacher["safe"][0],
                                        teacher["maximum"][0], predictor, cfg)
    labels = question_labels(records[0], rows, cases, cfg)
    a.check_saved_labels(records[0], labels, cfg)
    changed = copy.deepcopy(records[0])
    changed.ccs[0, 0] = 1 - changed.ccs[0, 0]
    with pytest.raises(ValueError, match="no longer reproduce"):
        a.check_saved_labels(changed, labels, cfg)
    truncated = dict(labels, preferred=labels["preferred"][:-1])
    with pytest.raises(ValueError, match="inconsistent shapes"):
        a.check_saved_labels(records[0], truncated, cfg)
    with pytest.raises(ValueError, match="inconsistent shapes"):
        a.check_saved_labels(records[0], dict(labels, first_cases=labels["first_cases"][:-1]), cfg)
    # Moving a row to another case changes its rebuilt observation.
    bits = np.unpackbits(labels["first_cases"].numpy())
    bits[[np.flatnonzero(bits)[0], np.flatnonzero(~bits.astype(bool))[0]]] ^= 1
    with pytest.raises(ValueError, match="no longer reproduce|invalid actions"):
        a.check_saved_labels(records[0], dict(labels, first_cases=torch.from_numpy(np.packbits(bits))), cfg)


@pytest.mark.parametrize("overrides", [{}, {"max_cost": 60.}, {"later_rank_filter": False}])
def test_lean_labels_equal_compacted_audit_rows_and_coverage(overrides):
    cfg, records, _, teacher, predictor = small_stage3_problem(questions=3, **overrides)
    for idx, record in enumerate(records):
        args = (record, teacher["scores"][idx], teacher["safe"][idx], teacher["maximum"][idx], predictor, cfg)
        rows, cases = a.build_decision_rows(*args)
        labels = a.build_decision_labels(*args)
        # Same plan: the array path never builds the audit dictionaries it matches.
        assert labels.pop("coverage") == a.label_coverage(rows, cases, cfg)
        compact = a.compact_decision_labels(record, rows, cases, cfg)
        assert labels.keys() == compact.keys()
        assert all(torch.equal(labels[key], compact[key]) for key in ("preferred", "first_cases"))
        assert labels["observation_sha256"] == compact["observation_sha256"]
        if overrides.get("max_cost"):
            assert {case["status"] for case in cases} >= {"OUT_OF_BUDGET", "BUDGET"}


def test_first_case_bitmaps_round_trip_planning_order_and_reject_corruption():
    cfg = a.Config()
    cases = a.decision_case_count(cfg)
    assert cases == 6 * 1912
    sizes = a.decision_state_sizes(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    rng = np.random.RandomState(5)
    chosen = np.flatnonzero(rng.rand(cases) < .6)
    # Planning order: larger structures first, ties in case order.
    source = chosen[np.argsort(-sizes[chosen % len(sizes)], kind="stable")]
    first_cases = a.decision_first_cases(source, cfg)
    assert first_cases.shape == (cases // 8,)
    np.testing.assert_array_equal(a.decision_row_sources(first_cases, cfg), source)
    np.testing.assert_array_equal(a.schema4_first_cases(source, cfg, "q").numpy(), first_cases)
    with pytest.raises(ValueError, match="not in planning order"):
        a.schema4_first_cases(source[::-1], cfg, "q")
    with pytest.raises(ValueError, match="distinct in-budget"):
        a.decision_first_cases(np.r_[source, source[:1]], cfg)
    with pytest.raises(ValueError, match="distinct in-budget"):
        a.decision_first_cases([cases], cfg)
    tiny = a.Config(k=1, zero_shots=1)
    assert a.decision_case_count(tiny) == 4
    # Cases 1 and 3 hold one- and three-item structures: the larger one is planned first.
    np.testing.assert_array_equal(a.decision_row_sources(np.array([0b01010000], np.uint8), tiny), [3, 1])
    for corrupt in (np.array([0b01011000], np.uint8), np.zeros(2, np.uint8)):
        with pytest.raises(ValueError, match="wrong size or padding"):
            a.decision_row_sources(corrupt, tiny)


def test_default_geometry_question_labels_need_about_one_byte_per_row(tmp_path):
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
    labels = a.build_decision_labels(record, scores, safe, maximum, predictor, cfg)
    path = tmp_path / "question.pt"
    a.atomic_torch(path, {"schema_version": a.DECISION_LABEL_SCHEMA, "fingerprint": "f" * 64,
                          "uid": record.uid, **labels})
    # Pickled audit rows took about 1 MiB per question (about 8 MiB while they
    # stored frozen features), which filled Kaggle's disk part-way through Stage 3;
    # schema 4 still kept an int32 source per row, about 5 bytes per row.
    rows = labels["coverage"]["action_rows"]
    assert rows > 5000
    assert path.stat().st_size < rows + a.decision_case_count(cfg) // 8 + 4 * 1024
    audit_rows, cases = a.build_decision_rows(record, scores, safe, maximum, predictor, cfg)
    compact = a.compact_decision_labels(record, audit_rows, cases, cfg)
    assert all(torch.equal(labels[key], compact[key]) for key in ("preferred", "first_cases"))
    assert labels["observation_sha256"] == compact["observation_sha256"]
    assert labels["coverage"] == a.label_coverage(audit_rows, cases, cfg)


def run_with_build_counter(monkeypatch, *args):
    built = []
    original = a.build_decision_labels

    def counted(record, *rest, **kwargs):
        built.append(record.uid)
        return original(record, *rest, **kwargs)

    monkeypatch.setattr(a, "build_decision_labels", counted)
    try:
        return a.train_decision_head(*args), built
    finally:
        monkeypatch.setattr(a, "build_decision_labels", original)


def test_interrupted_label_build_reuses_completed_questions_then_merges_each_role(tmp_path, monkeypatch, capsys):
    cfg, records, splits, teacher, predictor = small_stage3_problem(questions=4)
    expected = a.train_decision_head(records, splits, teacher, predictor, cfg, "cpu", tmp_path / "complete")
    run = tmp_path / "interrupted"
    original = a.build_decision_labels
    calls = 0

    def stop_on_second_question(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("simulated interrupted Stage 3 label build")
        return original(*args, **kwargs)

    monkeypatch.setattr(a, "build_decision_labels", stop_on_second_question)
    with pytest.raises(KeyboardInterrupt, match="simulated interrupted"):
        a.train_decision_head(records, splits, teacher, predictor, cfg, "cpu", run)
    assert len(list((run / "decision_labels" / "policy").glob("*.pt"))) == 1
    assert not a.decision_role_path(run, "policy").exists()
    monkeypatch.setattr(a, "build_decision_labels", original)
    capsys.readouterr()
    actual, built = run_with_build_counter(monkeypatch, records, splits, teacher, predictor, cfg, "cpu", run)
    log = capsys.readouterr().out
    assert built == [records[i].uid for i in splits["policy"][1:] + splits["dev"]]
    assert "Stage 3 policy labels: 3/3 questions" in log and "built 2, reused 1" in log
    # Completed roles are single files; per-question files are removed after merging.
    assert sorted(path.name for path in (run / "decision_labels").iterdir()) == ["dev.pt", "policy.pt"]
    store = a.load_label_file(a.decision_role_path(run, "policy"))
    assert store["uids"] == [records[i].uid for i in splits["policy"]]
    assert store["offsets"].tolist()[-1] == len(store["preferred"]) == expected["coverage"]["policy"]["action_rows"]
    assert_matching_training_checkpoints(expected, actual)


def write_legacy_layout(out, records, splits, teacher, predictor, cfg, layout):
    """Schema-3 shards as earlier revision-5 code wrote them, with their companions."""
    fingerprint = a.decision_dataset_fingerprint(records, splits, teacher, predictor, cfg)
    paths = a.decision_shard_paths(records, splits, out)
    structures = len(a.structural_states(cfg.zero_shots, cfg.k))
    for role in ("policy", "dev"):
        for idx, path in zip(splits[role], paths[role]):
            rows, cases = a.build_decision_rows(records[idx], teacher["scores"][idx], teacher["safe"][idx],
                                                teacher["maximum"][idx], predictor, cfg)
            shard = {"schema_version": 3, "implementation_revision": a.PIPELINE_REVISION,
                     "fingerprint": fingerprint, "uid": records[idx].uid,
                     "structural_states": structures,
                     "expected_cases": structures * math.factorial(cfg.zero_shots),
                     "rows": rows, "cases": cases}
            path.parent.mkdir(parents=True, exist_ok=True)
            companion = path.with_suffix(".training.pt")
            if layout == "reencoded_features":
                # Rows without "h", pickle protocol 4, compact companions.
                shard["rows"] = [{key: value for key, value in row.items() if key != "h"} for row in rows]
                with path.open("wb") as handle:
                    with gzip.GzipFile(fileobj=handle, mode="wb", compresslevel=1, mtime=0) as zipped:
                        torch.save(shard, zipped, pickle_protocol=4)
                a.atomic_torch(companion, {**a.pack_decision_rows(rows, cfg, ("target", "valid")),
                                           "scenario": torch.zeros(len(rows), dtype=torch.int32),
                                           "state": torch.zeros(len(rows), dtype=torch.int32)})
            elif layout == "stored_features":
                # The full-disk run: protocol-2 gzip rows with "h", companions with "h".
                with path.open("wb") as handle:
                    with gzip.GzipFile(fileobj=handle, mode="wb", compresslevel=1, mtime=0) as zipped:
                        torch.save(shard, zipped)
                a.atomic_torch(companion, a.pack_decision_rows(rows, cfg))
            else:
                a.atomic_torch(path, shard)
    a.atomic_json(out / "decision_dataset_manifest.json", {
        "schema_version": 3, "implementation_revision": a.PIPELINE_REVISION, "fingerprint": fingerprint,
        "roles": {role: [str(path.relative_to(out)) for path in paths[role]] for role in ("policy", "dev")}})
    return [path for role in ("policy", "dev") for path in paths[role]]


@pytest.mark.parametrize("layout", ["stored_features", "reencoded_features", "uncompressed"])
def test_schema3_shards_migrate_without_relabeling_and_are_deleted(tmp_path, monkeypatch, capsys, layout):
    cfg, records, splits, teacher, predictor = small_stage3_problem(questions=3)
    fresh, legacy = tmp_path / "fresh", tmp_path / "legacy"
    expected = a.train_decision_head(records, splits, teacher, predictor, cfg, "cpu", fresh)
    shards = write_legacy_layout(legacy, records, splits, teacher, predictor, cfg, layout)
    companions = [path.with_suffix(".training.pt") for path in shards]
    interrupted = shards[0].with_suffix(".pt.tmp")
    interrupted.write_bytes(b"leftover from the failed full-disk write")
    writes = []
    original = a.atomic_torch

    def track_writes(path, value):
        # Derived companions and failed writes are reclaimed before anything is written.
        writes.append((Path(path), [p for p in companions + [interrupted] if p.exists()]))
        return original(path, value)

    monkeypatch.setattr(a, "atomic_torch", track_writes)
    capsys.readouterr()
    actual, built = run_with_build_counter(monkeypatch, records, splits, teacher, predictor, cfg, "cpu", legacy)
    log = capsys.readouterr().out
    assert built == [] and writes and all(not remaining for _, remaining in writes)
    assert "built 0, reused 0, migrated 2" in log and "built 0, reused 0, migrated 1" in log
    assert not (legacy / "decision_shards").exists()
    manifest = json.loads((legacy / "decision_dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest == json.loads((fresh / "decision_dataset_manifest.json").read_text(encoding="utf-8"))
    for role in ("policy", "dev"):
        migrated, built_fresh = (a.load_label_file(a.decision_role_path(run, role)) for run in (legacy, fresh))
        assert migrated.keys() == built_fresh.keys()
        assert all(torch.equal(value, built_fresh[key]) if isinstance(value, torch.Tensor) else value == built_fresh[key]
                   for key, value in migrated.items())
    assert_matching_training_checkpoints(expected, actual)


def test_full_disk_run_migrates_completed_shards_and_builds_only_the_rest(tmp_path, monkeypatch, capsys):
    # The v5 run stopped part-way through policy labels, before any dev question.
    cfg, records, splits, teacher, predictor = small_stage3_problem(questions=4)
    expected = a.train_decision_head(records, splits, teacher, predictor, cfg, "cpu", tmp_path / "fresh")
    run = tmp_path / "full_disk"
    shards = write_legacy_layout(run, records, splits, teacher, predictor, cfg, "stored_features")
    for path in shards[2:]:
        path.unlink()
        path.with_suffix(".training.pt").unlink()
    capsys.readouterr()
    actual, built = run_with_build_counter(monkeypatch, records, splits, teacher, predictor, cfg, "cpu", run)
    log = capsys.readouterr().out
    assert built == [records[splits["policy"][2]].uid, records[splits["dev"][0]].uid]
    assert "Stage 3 policy labels: 3/3 questions" in log and "built 1, reused 0, migrated 2" in log
    assert not (run / "decision_shards").exists()
    assert_matching_training_checkpoints(expected, actual)


def write_schema4_layout(run, records, splits, cfg, merged):
    """Rewrite finished labels as schema-4 code left them: an int32 source per row."""
    manifest_path = run / "decision_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    a.atomic_json(manifest_path, dict(manifest, schema_version=4))
    for role in ("policy", "dev"):
        role_path = a.decision_role_path(run, role)
        store = a.load_label_file(role_path)
        offsets = store["offsets"].tolist()
        sources = [torch.from_numpy(a.decision_row_sources(first.numpy(), cfg).astype(np.int32))
                   for first in store["first_cases"]]
        if merged:
            schema4 = {key: value for key, value in store.items() if key != "first_cases"}
            a.atomic_torch(role_path, dict(schema4, schema_version=4, source=torch.cat(sources)))
            continue
        role_path.unlink()
        paths = a.question_paths(records, splits[role], run / "decision_labels" / role)
        for position, (path, source) in enumerate(zip(paths, sources)):
            start, stop = offsets[position:position+2]
            path.parent.mkdir(parents=True, exist_ok=True)
            a.atomic_torch(path, {"schema_version": 4, "fingerprint": store["fingerprint"],
                                  "uid": store["uids"][position], "coverage": store["coverage"][position],
                                  "source": source, "preferred": store["preferred"][start:stop],
                                  "observation_sha256": store["observation_sha256"][position]})


@pytest.mark.parametrize("merged", [True, False])
def test_schema4_labels_convert_in_place_without_relabeling(tmp_path, monkeypatch, capsys, merged):
    cfg, records, splits, teacher, predictor = small_stage3_problem(questions=3)
    fresh, run = tmp_path / "fresh", tmp_path / "schema4"
    expected = a.train_decision_head(records, splits, teacher, predictor, cfg, "cpu", fresh)
    a.train_decision_head(records, splits, teacher, predictor, cfg, "cpu", run)
    # Retrain from the converted labels alone.
    (run / "decision_head_progress.pt").unlink()
    write_schema4_layout(run, records, splits, cfg, merged)
    capsys.readouterr()
    actual, built = run_with_build_counter(monkeypatch, records, splits, teacher, predictor, cfg, "cpu", run)
    log = capsys.readouterr().out
    assert built == []
    if merged:
        assert "converting the schema-4 role file to schema 5 without relabeling" in log
        assert "built 0, reused 2" in log and "built 0, reused 1" in log
    else:
        assert "built 0, reused 0, migrated 2" in log and "built 0, reused 0, migrated 1" in log
    manifest = json.loads((run / "decision_dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest == json.loads((fresh / "decision_dataset_manifest.json").read_text(encoding="utf-8"))
    assert sorted(path.name for path in (run / "decision_labels").iterdir()) == ["dev.pt", "policy.pt"]
    for role in ("policy", "dev"):
        converted, built_fresh = (a.load_label_file(a.decision_role_path(path, role)) for path in (run, fresh))
        assert converted.keys() == built_fresh.keys()
        assert all(torch.equal(value, built_fresh[key]) if isinstance(value, torch.Tensor)
                   else value == built_fresh[key] for key, value in converted.items())
    assert_matching_training_checkpoints(expected, actual)


def test_incompatible_schema3_manifest_or_shard_is_never_migrated(tmp_path, monkeypatch):
    cfg, records, splits, teacher, predictor = small_stage3_problem()
    shards = write_legacy_layout(tmp_path, records, splits, teacher, predictor, cfg, "uncompressed")
    shard = a.load_checkpoint(shards[0])
    shard["uid"] = "another question"
    a.atomic_torch(shards[0], shard)
    with pytest.raises(ValueError, match="Incompatible decision shard"):
        a.train_decision_head(records, splits, teacher, predictor, cfg, "cpu", tmp_path)
    assert shards[0].exists()
    manifest_path = tmp_path / "decision_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fingerprint"] = "stale teacher or predictor"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible teacher, predictor"):
        a.train_decision_head(records, splits, teacher, predictor, cfg, "cpu", tmp_path)
    assert all(path.exists() for path in shards)


def test_question_audit_recomputes_the_saved_rows_and_cases(tmp_path):
    cfg, records, splits, teacher, predictor = small_stage3_problem()
    checkpoint = a.train_decision_head(records, splits, teacher, predictor, cfg, "cpu", tmp_path)
    audit = a.decision_question_audit(records, splits, teacher, predictor, cfg, tmp_path, "dev", 0)
    rows, cases = a.build_decision_rows(records[1], teacher["scores"][1], teacher["safe"][1],
                                        teacher["maximum"][1], predictor, cfg)
    assert audit["uid"] == records[1].uid and audit["cases"] == cases
    assert [row["state_key"] for row in audit["rows"]] == [row["state_key"] for row in rows]
    assert {"goal_rank", "reach", "action_expected_cost", "objective"} <= set(audit["rows"][0])
    assert audit["coverage"]["action_rows"] == checkpoint["coverage"]["dev"]["action_rows"]
    # A head completed under schema 4 never rewrote its role files; the audit reads them too.
    write_schema4_layout(tmp_path, records, splits, cfg, merged=True)
    schema4_audit = a.decision_question_audit(records, splits, teacher, predictor, cfg, tmp_path, "dev", 0)
    assert schema4_audit["cases"] == audit["cases"] and schema4_audit["coverage"] == audit["coverage"]
    path = a.decision_role_path(tmp_path, "dev")
    store = a.load_label_file(path)
    store["preferred"] = torch.zeros_like(store["preferred"])
    a.atomic_torch(path, store)
    with pytest.raises(ValueError, match="differ from the saved labels"):
        a.decision_question_audit(records, splits, teacher, predictor, cfg, tmp_path, "dev", 0)


def test_workflow_reclaims_derived_stage3_leftovers_before_stage0_writes(tmp_path, capsys):
    train = tmp_path / "train.json"
    train.write_text("[]", encoding="utf-8")
    cfg = a.Config(train_file=str(train), test_files=[], use_test_files_for_dev_and_audit=False,
                   output_dir=str(tmp_path / "run"), device="cpu", cpu_threads=1)
    a.Workflow(cfg)
    out = tmp_path / "run"
    # The full-disk v5 layout: derived companions and interrupted atomic writes.
    derived = [out / "decision_shards" / "policy" / "000000_abc.training.pt",
               out / "decision_shards" / "dev" / "000000_def.training.pt.tmp",
               out / "decision_shards" / "policy" / "000001_abc.pt.tmp",
               out / "decision_labels" / "policy" / "000002_abc.pt.tmp",
               out / "decision_labels" / "policy.pt.tmp",
               out / "compact_records.pt.tmp"]
    # Completed shards, labels, and checkpoints hold finished work and stay.
    kept = [out / "decision_shards" / "policy" / "000000_abc.pt",
            out / "decision_labels" / "policy" / "000003_abc.pt",
            out / "decision_labels" / "dev.pt",
            out / "decision_head_progress.pt"]
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
