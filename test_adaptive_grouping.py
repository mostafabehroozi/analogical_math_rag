"""Offline checks of the optional grouping phase (Stage 1G): grouping-log parsing and joining, group
targets and loss, the grouping teacher's folds, the report's selections, and the workflow."""
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import adaptive_analogical_training as a
from test_adaptive_analogical_training import raw_record, record


def label_of(index, positions):
    """A deterministic recorded label for the group of these true retrieval positions."""
    return (index + sum(positions)) % 3 != 0


def layer1_item(index, cfg, prefix="train", reverse=False):
    item = raw_record(index, cfg, prefix)
    if reverse:
        # Layer-1 logs written before 2026-09-21 list each similarity score at the mirrored position.
        rows = item["layer1_base_execution_state"]["retrieved_set"]
        for row, score in zip(rows, [row["similarity_score"] for row in rows][::-1]):
            row["similarity_score"] = score
    return item


def grouping_item(layer1, cfg, index=None, label=label_of):
    """The grouping run's entry for a Layer-1 entry: its retrieval in true order with correct scores."""
    state = layer1["layer1_base_execution_state"]
    idx = layer1["target_query_original_hard_list_idx"]
    scores = sorted((row["similarity_score"] for row in state["retrieved_set"]), reverse=True)
    retrieved = [{"corpus_index": row["corpus_index"], "question": "", "similarity_score": score,
                  "retrieval_position": position}
                 for position, (row, score) in enumerate(zip(state["retrieved_set"], scores))]
    candidates, outcomes = {}, {}
    for size in cfg.grouping_sizes:
        for positions in itertools.combinations(range(cfg.k), size):
            cid = f"group_s{size}_p" + "-".join(map(str, positions))
            members = [retrieved[p] for p in positions]
            candidates[cid] = {"candidate_id": cid, "candidate_text": f"Answer of {cid}",
                               "source_exemplar_indices": [m["corpus_index"] for m in members],
                               "source_retrieval_positions": list(positions), "group_size": size,
                               "similarity_scores": [m["similarity_score"] for m in members],
                               "similarity_sum": sum(m["similarity_score"] for m in members),
                               "generation_status": "SUCCESS"}
            outcomes[cid] = {"is_correct": bool(label(idx, positions)), "evaluation_status": "SUCCESS"}
    text = layer1["target_query_text"]
    return {"target_query_original_hard_list_idx": idx if index is None else index, "target_query_text": text,
            "pipeline_status": "SUCCESS", "execution_mode": "layer1_grouping_only",
            "batch_metadata": {"batch_id": "full-0001", "phase": "full", "terminal_status": "SUCCESS"},
            "layer1_grouping_state": {
                "metadata": {"query_index": idx, "schema_version": "1.0",
                             "config_snapshot": {"TOP_N_CANDIDATES_RETRIEVAL": cfg.k,
                                                 "LAYER1_GROUP_SIZES": list(cfg.grouping_sizes),
                                                 "LAYER1_GROUPING_PROMPT_TEMPLATE": "layer1_grouping_solver_v1"},
                             "effective_group_sizes": list(cfg.grouping_sizes),
                             "expected_candidate_count": len(candidates)},
                "target_query_data": {"query_text": text, "ground_truth_answer": "42", "query_index": idx},
                "retrieved_set": retrieved, "candidate_set": candidates, "ground_truth_labels": outcomes,
                "execution_trace": [],
                "step_statuses": {"retrieval": "SUCCESS", "group_generation": "SUCCESS",
                                  "ground_truth_evaluation": "SUCCESS"},
                "overall_status": "SUCCESS"}}


def expected_labels(layer1, parsed, cfg):
    """Each group output's label through corpus ids: record slots -> true positions in the Layer-1 list."""
    order = [str(row["corpus_index"]) for row in layer1["layer1_base_execution_state"]["retrieved_set"]]
    idx = layer1["target_query_original_hard_list_idx"]
    return [int(label_of(idx, sorted(order.index(parsed.evaluator_ids[j]) for j in np.flatnonzero(row))))
            for row in cfg.group_members]


COMBINED = "Layer1_Grouping_K3_Size2_run_log"
# The grouping phase on top of the teacher and the snapshot predictor (Stage 4 with the fixed orders).
GROUPING_STAGES = ("teacher", "snapshot", "grouping")


def grouping_config(root, combined=True, **overrides):
    """Layer-1 logs of a training file and a test file, and the grouping run's log(s) of their questions.

    The grouping experiment writes one run log holding every benchmark's questions in the combined
    target order (the training file's first; `combined`). The other variant has one grouping log per
    Layer-1 log.
    """
    files = {name: root / f"{name}.json" for name in ("train", "external", "train_grouping", "external_grouping",
                                                        COMBINED)}
    grouping = [files[COMBINED]] if combined else [files["train_grouping"], files["external_grouping"]]
    settings = dict(train_file=str(files["train"]), test_files=[str(files["external"])], stages=GROUPING_STAGES,
                    grouping_files=[str(path) for path in grouping],
                    use_test_files_for_dev_and_audit=True, output_dir=str(root / "run"), device="cpu",
                    cpu_threads=1, k=3, zero_shots=2, teacher_folds=2, hidden_dim=8, residual_blocks=1,
                    batch_size=256, teacher_epochs=1, snapshot_epochs=1, dev_checks_per_epoch=1,
                    view_patience=2, bootstrap_samples=20)
    settings.update(overrides)
    cfg = a.Config(**settings)
    if not files["train"].exists():
        # Global indices as in the combined run: the training file's questions, then the test file's.
        train = [layer1_item(i, cfg, reverse=i % 2 == 0) for i in range(24)]
        external = [layer1_item(24 + i, cfg, prefix="external", reverse=True) for i in range(4)]
        for name, items in (("train", train), ("external", external)):
            files[name].write_text(json.dumps(items))
            files[f"{name}_grouping"].write_text(json.dumps([grouping_item(item, cfg) for item in items]))
        files[COMBINED].write_text(json.dumps([grouping_item(item, cfg) for item in train + external]))
    return cfg


def test_group_members_follow_the_configured_sizes_in_combination_order():
    assert a.group_members(3, (2,)).tolist() == [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    members = a.Config(k=4, grouping_sizes=(2, 3)).group_members
    assert members.shape == (6 + 4, 4) and members.sum(1).tolist() == [2] * 6 + [3] * 4
    assert a.Config(k=5).group_count == 10  # The companion run's K=5, sizes [2]: every pair.


def test_grouping_settings_are_validated_and_stay_outside_the_run_contract():
    one = ["/logs/Layer1_Grouping_K5_Sizes2_run_log.json"]  # The grouping run's single log.
    ON = ("teacher", "grouping")
    with pytest.raises(ValueError, match="grouping_files"):
        a.Config(stages=ON).validate()
    with pytest.raises(ValueError, match="grouping_files"):
        a.Config(stages=ON, grouping_files=one[0]).validate()  # A list, not a bare path.
    with pytest.raises(ValueError, match="unique filename"):
        a.Config(stages=ON, grouping_files=["/a/run_log.json", "/b/run_log.json"]).validate()
    for sizes in ((1,), (6,), (3, 2), (2, 2), ()):
        with pytest.raises(ValueError, match="grouping_sizes"):
            a.Config(stages=ON, grouping_files=one, grouping_sizes=sizes).validate()
    with pytest.raises(AssertionError):
        a.Config(stages=ON, grouping_files=one, grouping_weight=-1.).validate()
    a.Config(stages=ON, grouping_files=[f"/logs/{name}_grouping.json" for name in ("numina", "gsm8k")]).validate()
    on = a.Config(stages=ON, grouping_files=one, grouping_sizes=(2, 3), grouping_weight=.5,
                  grouping_shuffle_control=True)
    on.validate()
    # Off or on, the run contract (and so the Stage 3 identity) holds the same settings.
    assert a.contract_settings(on) == a.contract_settings(a.Config())
    assert not set(a.GROUPING_SETTINGS) & set(a.contract_settings(on))


@pytest.mark.parametrize("reverse", [False, True])
def test_group_labels_join_by_corpus_id_even_when_the_layer1_scores_are_mirrored(reverse):
    cfg = a.Config(k=3, zero_shots=2, grouping_sizes=(2,))
    layer1 = layer1_item(5, cfg, reverse=reverse)
    parsed = a.parse_record(layer1, 5, "train", cfg)
    entry = grouping_item(layer1_item(5, cfg), cfg)  # The grouping run's retrieval is correct either way.
    joined, labels, true_rank, notes = a.parse_grouping(entry, {parsed.group: parsed}, cfg)
    assert joined is parsed
    assert labels.tolist() == expected_labels(layer1, parsed, cfg)
    order = [str(row["corpus_index"]) for row in layer1["layer1_base_execution_state"]["retrieved_set"]]
    assert true_rank.tolist() == [order.index(e) for e in parsed.evaluator_ids]
    # A mirrored Layer-1 log makes record slot 1 the farthest source.
    assert true_rank.tolist() == ([2, 1, 0] if reverse else [0, 1, 2])
    assert notes == {"index_differs": False, "layer1_scores_misattached": reverse}
    # The join is by question text, so another index still joins and is only noted.
    assert a.parse_grouping(grouping_item(layer1, cfg, index=999), {parsed.group: parsed}, cfg)[3]["index_differs"]


def test_tied_similarity_scores_are_never_reported_as_misattached():
    cfg = a.Config(k=3, zero_shots=2, grouping_sizes=(2,))
    layer1 = layer1_item(6, cfg)
    for row in layer1["layer1_base_execution_state"]["retrieved_set"]:
        row["similarity_score"] = .8  # Duplicate exemplars: every order of them is the true order.
    parsed = a.parse_record(layer1, 6, "train", cfg)
    _, labels, true_rank, notes = a.parse_grouping(grouping_item(layer1, cfg), {parsed.group: parsed}, cfg)
    assert true_rank.tolist() == [0, 1, 2] and not notes["layer1_scores_misattached"]
    assert labels.tolist() == expected_labels(layer1, parsed, cfg)


def test_every_grouping_exclusion_and_unknown_label_is_reported_never_counted_wrong():
    cfg = a.Config(k=3, zero_shots=2, grouping_sizes=(2,))
    layer1 = layer1_item(2, cfg)
    parsed = a.parse_record(layer1, 2, "train", cfg)
    by_group = {parsed.group: parsed}
    good = grouping_item(layer1, cfg)

    def reason(change):
        entry = copy.deepcopy(good)
        change(entry, entry["layer1_grouping_state"])
        with pytest.raises(ValueError) as error:
            a.parse_grouping(entry, by_group, cfg)
        return str(error.value)

    failed = {"target_query_original_hard_list_idx": 2, "target_query_text": good["target_query_text"],
              "pipeline_status": "FAILED_WORKER_EXCEPTION", "batch_error": {"message": "boom"}}
    with pytest.raises(ValueError, match="missing_grouping_state"):
        a.parse_grouping(failed, by_group, cfg)
    assert reason(lambda e, s: e.update(target_query_text="another question")) == "not_in_layer1"
    assert reason(lambda e, s: s["retrieved_set"][0].update(corpus_index=999)) == "retrieval_mismatch"
    assert reason(lambda e, s: s["retrieved_set"].pop()) == "retrieval_mismatch"
    assert reason(lambda e, s: s["retrieved_set"].reverse()) == "grouping_retrieval_not_descending"
    assert reason(lambda e, s: s["retrieved_set"][0].update(similarity_score=.95)) == "similarity_mismatch"
    assert reason(lambda e, s: s["metadata"].update(effective_group_sizes=[2, 3])) == "group_sizes_mismatch"
    assert reason(lambda e, s: s["candidate_set"]["group_s2_p0-2"].update(
        source_exemplar_indices=s["candidate_set"]["group_s2_p0-1"]["source_exemplar_indices"])) == "duplicate_group"
    # Failed generations or gradings and missing candidates leave that group's label unknown (-1).
    entry = copy.deepcopy(good)
    state = entry["layer1_grouping_state"]
    state["candidate_set"]["group_s2_p0-1"]["generation_status"] = "FAILURE"
    state["ground_truth_labels"]["group_s2_p0-2"] = {"is_correct": None, "evaluation_status": "EMPTY_CANDIDATE"}
    del state["candidate_set"]["group_s2_p1-2"]
    _, labels, _, _ = a.parse_grouping(entry, by_group, cfg)
    assert labels.tolist() == [-1, -1, -1]


def test_one_grouping_log_of_every_benchmark_splits_by_each_questions_layer1_log(tmp_path):
    cfg = a.Config(k=3, zero_shots=2, grouping_sizes=(2,))
    items = [layer1_item(i, cfg, reverse=i < 2) for i in range(4)]
    # Three questions of the training file and one of a test file, as one grouping run writes them.
    records = [a.parse_record(item, i, "numina_hard_run_log" if i < 3 else "gsm8k_run_log", cfg)
               for i, item in enumerate(items)]
    entries = [grouping_item(item, cfg) for item in items]
    entries[3]["layer1_grouping_state"]["ground_truth_labels"]["group_s2_p0-1"]["evaluation_status"] = "API_FAILURE"
    entries.append(copy.deepcopy(entries[0]))  # A second entry of a joined question.
    entries.append({"target_query_original_hard_list_idx": 9, "target_query_text": "unknown",
                    "pipeline_status": "FAILED_WORKER_EXCEPTION"})
    entries.append(grouping_item(layer1_item(9, cfg, prefix="unlisted"), cfg))  # In no Layer-1 log.
    path = tmp_path / "Layer1_Grouping_K5_Sizes2_run_log.json"
    path.write_text(json.dumps(entries))
    joined, counts = a.read_grouping_log(path, records, cfg)
    assert set(joined) == {r.uid for r in records}
    assert counts == {"total": 7, "joined": 4, "eligible": 3, "incomplete_labels": 1, "duplicate": 1,
                      "missing_grouping_state": 1, "not_in_layer1": 1, "layer1_scores_misattached": 2,
                      "group_labels_known": 11, "group_labels_correct": counts["group_labels_correct"],
                      "joined_by_layer1_log": {"numina_hard_run_log": 3, "gsm8k_run_log": 1}}
    grouping = a.grouping_arrays(records, {"joined": {uid: (l.tolist(), r.tolist()) for uid, (l, r) in joined.items()}},
                                 cfg)
    audit = a.grouping_benchmark_audit(records, grouping, "numina_hard_run_log")
    assert {name: (row["role"], row["layer1_questions"], row["joined"], row["eligible"])
            for name, row in audit.items()} == {"numina_hard_run_log": ("train", 3, 3, 3),
                                               "gsm8k_run_log": ("test", 1, 1, 0)}


def test_views_append_group_targets_equal_to_the_scalar_reference_for_every_order_and_state():
    cfg = a.Config(k=3, zero_shots=2, grouping_sizes=(2,))
    records = [record(cfg, i) for i in range(3)]
    labels = np.asarray([[1, 0, -1], [0, 1, 1], [-1, -1, 0]], np.int8)
    plain = a.SnapshotViews(records, [2, 0], cfg, "cpu")
    grouped = a.SnapshotViews(records, [2, 0], cfg, "cpu", groups={"labels": labels})
    _, x1, y1 = next(plain.ordered(plain.rows))
    _, x2, y2 = next(grouped.ordered(grouped.rows))
    # Stage 1's inputs and targets are untouched; the group targets follow them.
    assert torch.equal(x1, x2) and torch.equal(y1, y2[:, :2*cfg.n])
    states = a.presence_states(cfg.zero_shots, cfg.k)
    orders = list(itertools.permutations(range(cfg.zero_shots)))
    expected = [a.group_target(labels[idx], state, cfg) for idx in (2, 0) for _ in orders for state in states]
    assert np.stack(expected).tobytes() == y2[:, 2*cfg.n:].numpy().tobytes()
    assert plain.digest() != grouped.digest()
    # A group is present once each of its sources has its one-shot candidate or its evaluator.
    state = a.State(candidate_mask=0b1 | 1 << cfg.zero_shots, evaluator_mask=0b010)  # OS of source 0, R of source 1.
    assert a.group_target(np.asarray([1, 1, 1]), state, cfg).tolist() == [1, 0, 0, 1, 0, 0, 1, 1, 1]
    full = a.State((1 << cfg.n) - 1, (1 << cfg.k) - 1)
    assert a.group_target(labels[0], full, cfg).tolist() == [1, 0, 0, 1, 1, 1, 1, 1, 0]


def test_group_loss_trains_present_groups_to_labels_absent_ones_to_zero_and_skips_unknown():
    cfg = a.Config(k=3, zero_shots=2, grouping_sizes=(2,), absent_weight=.5, grouping_weight=2.)
    n, groups = cfg.n, cfg.group_count
    candidates = torch.cat([torch.ones(1, n), torch.ones(1, n)], 1)
    label, present, known = torch.tensor([[1., 1, 0]]), torch.tensor([[1., 0, 0]]), torch.tensor([[1., 0, 1]])
    logits = torch.zeros(1, n + groups, requires_grad=True)
    a.grouping_teacher_loss(logits, torch.cat([candidates, label * present, present, known], 1), cfg).backward()
    assert logits.grad[0, :n].tolist() == pytest.approx([-.5 / n] * n)  # Stage 1's loss is unchanged.
    # Present and known: toward its label; unknown: untrained; absent and known: toward 0, at absent_weight.
    assert logits.grad[0, n:].tolist() == pytest.approx([2 * -.5, 0., 2 * .5 * .5])
    # In a complete state every group is present: the plain mean BCE over the known groups.
    logits = torch.randn(4, n + groups)
    labels = torch.randint(0, 2, (4, groups)).float()
    complete = torch.cat([candidates.repeat(4, 1), labels, torch.ones(4, groups), torch.ones(4, groups)], 1)
    expected = (a.teacher_loss(logits[:, :n], candidates.repeat(4, 1), cfg) + 2 *
                torch.nn.functional.binary_cross_entropy_with_logits(logits[:, n:], labels))
    assert float(a.grouping_teacher_loss(logits, complete, cfg)) == pytest.approx(float(expected), rel=1e-6)


def small_teacher_cfg():
    return a.Config(k=3, zero_shots=2, grouping_sizes=(2,), hidden_dim=8, residual_blocks=1, batch_size=128,
                    device="cpu", cpu_threads=1, teacher_epochs=1, dev_checks_per_epoch=1, view_patience=10,
                    teacher_folds=2)


def test_the_grouping_teacher_uses_stage_1_folds_and_scores_groups_out_of_fold(tmp_path, monkeypatch):
    cfg = small_teacher_cfg()
    records = [record(cfg, i) for i in range(12)]
    splits = {"supervised": list(range(9)), "dev": [9, 10, 11]}
    labels = np.random.RandomState(1).randint(0, 2, (12, cfg.group_count)).astype(np.int8)
    labels[0, 1] = -1
    out = tmp_path / "run"
    out.mkdir()
    base = a.fit_teachers(records, splits, cfg, "cpu", out)
    identity = a.grouping_identity(labels, cfg)
    grouped = a.fit_teachers(records, splits, cfg, "cpu", out, groups={"labels": labels}, prefix="grouping_teacher",
                             stage="Stage 1G", title="Grouping teacher", identity=identity)
    assert grouped["folds"] == base["folds"]
    assert grouped["scores"].shape == (12, cfg.n) and grouped["group_scores"].shape == (12, cfg.group_count)
    assert "group_scores" not in base
    folds = [a.load_checkpoint(out / f"grouping_teacher_fold_{f}.pt") for f in (1, 2)]
    predictions = [a.teacher_predict(fold, records, list(range(12)), cfg, "cpu") for fold in folds]
    for fold, prediction in zip(folds, predictions):
        assert fold["grouping"] == identity["grouping"]
        held = [i for i, r in enumerate(records) if r.uid in fold["heldout_ids"]]
        assert np.allclose(grouped["group_scores"][held], prediction[held, cfg.n:])
        assert np.allclose(grouped["scores"][held], prediction[held, :cfg.n])
    mean = np.mean([prediction[9:] for prediction in predictions], 0)
    assert np.allclose(np.c_[grouped["scores"][9:], grouped["group_scores"][9:]], mean, atol=1e-6)
    # Its saved folds are reused for the same labels and settings only.
    monkeypatch.setattr(a, "train_teacher", lambda *args, **kwargs: pytest.fail("a completed fold retrained"))
    again = a.fit_teachers(records, splits, cfg, "cpu", out, groups={"labels": labels}, prefix="grouping_teacher",
                           stage="Stage 1G", title="Grouping teacher", identity=identity)
    assert np.array_equal(again["group_scores"], grouped["group_scores"])
    other = {key: folds[0][key] for key in ("train_ids", "validation_ids", "heldout_ids")}
    assert a.completed_teacher(out / "grouping_teacher_fold_1.pt", {**other, **identity}, cfg, "x") is not None
    for changed in (a.grouping_identity(labels[::-1].copy(), cfg),
                    a.grouping_identity(labels, a.Config(k=3, zero_shots=2, grouping_weight=.5)),
                    a.grouping_identity(labels, cfg, control=True)):
        assert a.completed_teacher(out / "grouping_teacher_fold_1.pt", {**other, **changed}, cfg, "x") is None
    # The control's labels are each question's labels in another order.
    shuffled = a.shuffled_group_labels(labels, 3)
    assert (np.sort(shuffled, 1) == np.sort(labels, 1)).all() and not np.array_equal(shuffled, labels)


def test_selections_keep_families_apart_and_break_ties_toward_fewer_and_nearer_examples():
    cfg = a.Config(k=3, zero_shots=2, grouping_sizes=(2,))
    parsed = record(cfg)
    parsed.labels[:] = [0, 1, 1, 0, 1]          # zs_0, zs_1, then the one-shots of record slots 0, 1, 2.
    true_rank = np.asarray([2, 0, 1], np.int8)   # Slot 1 is truly the nearest source, slot 2 the second.
    group_labels = np.asarray([0, 1, 1], np.int8)  # Groups: slots {0,1}, {0,2}, {1,2}.
    base = np.asarray([.95, .1, .3, .6, .2])
    grouped = (np.asarray([.2, .4, .5, .1, .3]), np.asarray([.7, .2, .9]))
    heuristics = {"H": np.asarray([0., 0., .5, .5, .1])}
    correct, choices, levels = a.grouping_question(parsed, group_labels, true_rank, base, grouped, heuristics, cfg)
    n = cfg.n
    assert correct.tolist() == [0, 1, 1, 0, 1, 0, 1, 1]
    assert choices["Similarity (no model)"] == {"top0": 0, "top1": 3, "top2": n + 2}
    # The best-scored candidate is a zero-shot, yet top-1 is still the best-scored one-shot.
    assert choices["Stage 1 teacher rerank"] == {"top0": 0, "top1": 3, "top2": n + 0}
    assert choices["Grouping teacher rerank"] == {"top0": 1, "top1": 2, "top2": n + 1}
    assert choices["Grouping teacher: best group output"] == {"top2": n + 2}
    # Tied scores go to the lower zero-shot index and to the nearer source.
    assert choices["Heuristic H rerank"]["top0"] == 0 and choices["Heuristic H rerank"]["top1"] == 3
    assert choices["Stage 1 member pairs (mean)"] == choices["Stage 1 member pairs (min)"] == {"top2": n + 0}
    assert choices["Stage 1 member pairs (max)"] == {"top2": n + 2}  # {0,1} and {1,2} tie; {1,2} is nearer.
    assert choices["Stage 1 member pairs (noisy-OR)"] == {"top2": n + 0}
    assert (levels[0]["Grouping teacher"], levels[1]["Grouping teacher"], levels[2]["Grouping teacher"]) == (1, 2, n + 2)
    assert (levels[0]["Stage 1 teacher"], levels[2]["Stage 1 teacher"]) == (0, 0)
    assert (levels[0]["random"], levels[0]["oracle"]) == (.5, 1.)
    assert levels[1]["random"] == pytest.approx(.6) and levels[2]["random"] == pytest.approx(correct.mean())
    # A tie across families goes to the fewer examples.
    tied = (np.asarray([.5, .1, .5, .1, .1]), np.asarray([.5, .1, .1]))
    assert a.grouping_question(parsed, group_labels, true_rank, base, tied, heuristics, cfg)[2][2]["Grouping teacher"] == 0


def test_grouping_tables_are_paired_add_up_and_are_json_ready():
    cfg = a.Config(k=3, zero_shots=2, grouping_sizes=(2,), bootstrap_samples=10)
    records = [record(cfg, i) for i in range(8)]
    rng = np.random.RandomState(3)
    labels = rng.randint(0, 2, (8, cfg.group_count)).astype(np.int8)
    labels[7, 1] = -1
    grouping = {"labels": labels, "true_rank": np.stack([rng.permutation(cfg.k) for _ in range(8)]).astype(np.int8),
                "eligible": (labels >= 0).all(1)}
    scores = lambda: {"scores": rng.rand(8, cfg.n), "group_scores": rng.rand(8, cfg.group_count)}
    teacher, control, base = scores(), scores(), rng.rand(8, cfg.n)
    heuristics = {"Absolute": ("Absolute", "All", "Holistic")}
    result = a.grouping_tables(records, list(range(8)), grouping, base, teacher, heuristics, cfg, .4, control)
    assert (result["questions"], result["excluded_without_group_labels"]) == (7, 1)
    top, levels = result["top_k"], result["levels"]
    assert top["Similarity (no model)"]["top0"] == pytest.approx(np.mean([records[i].labels[0] for i in range(7)]))
    assert set(top) >= {"Similarity (no model)", "Stage 1 teacher rerank", "Grouping teacher rerank",
                        "Shuffled-label control rerank", "Heuristic Absolute (All/Holistic) rerank",
                        "Grouping teacher: best group output", "Stage 1 member pairs (mean)"}
    assert levels["0"]["oracle"] <= levels["1"]["oracle"] <= levels["2"]["oracle"]
    comparison = result["comparisons"]["Grouping teacher level-2 vs level-1"]
    assert comparison["delta"] == pytest.approx(levels["2"]["Grouping teacher"] - levels["1"]["Grouping teacher"])
    assert "Grouping teacher level-2 vs shuffled-label control" in result["comparisons"]
    assert sum(row["questions"] for row in result["switch"].values() if isinstance(row, dict)) == 7
    strata = result["strata"]
    assert strata["one-shots mixed"]["questions"] + strata["one-shots unanimous"]["questions"] == 7
    for shares in result["level_family_share"]["2"].values():
        assert sum(shares.values()) == pytest.approx(1)
    calls = result["calls"]["top_k"]
    assert calls["Similarity (no model)"] == {"top0": 1., "top1": 1., "top2": 1.}
    assert (calls["Stage 1 teacher rerank"]["top0"], calls["Stage 1 teacher rerank"]["top2"]) == (
        cfg.full_cost, cfg.full_cost + 1)
    quality = result["group_outputs"]["Grouping teacher"]
    assert 0 <= quality["ap"] <= 1 and quality["prior"] == .4
    assert sum(row["count"] for row in result["calibration"]["group"]) == 7 * cfg.group_count
    json.dumps(result, allow_nan=False)
    assert a.grouping_tables(records, [7], grouping, base, teacher, heuristics, cfg, .4) == {
        "questions": 0, "excluded_without_group_labels": 1}


def test_the_grouping_phase_joins_trains_and_reports_and_can_join_a_finished_run(tmp_path, monkeypatch, capsys):
    off = grouping_config(tmp_path, stages=("teacher", "snapshot"))
    work = a.Workflow(off).build_datasets().prepare().train_teacher()
    assert work.grouping_report() is None and "Grouping phase is OFF" in capsys.readouterr().out
    contract = (tmp_path / "run" / "contract.json").read_bytes()
    assert not any(path.name.startswith("grouping") for path in (tmp_path / "run").iterdir())
    # Turning the phase on in the same folder keeps the contract and reuses the completed Stage 1.
    cfg = grouping_config(tmp_path)
    prefixes = []
    original = a.fit_teachers

    def fit(*args, **kwargs):
        prefixes.append(kwargs.get("prefix", "teacher"))
        return original(*args, **kwargs)

    monkeypatch.setattr(a, "fit_teachers", fit)
    work = a.Workflow(cfg).build_datasets().prepare()
    assert "Grouping phase ON (Stage 1G)" in capsys.readouterr().out
    # The grouping data phase reads logs only: no network is constructed.
    with monkeypatch.context() as patch:
        patch.setattr(torch.nn.Module, "__init__", lambda *args, **kwargs: pytest.fail("a model was built"))
        work.build_grouping_dataset()
    results = work.grouping_report()
    out = capsys.readouterr().out
    assert prefixes == ["grouping_teacher"]
    assert (tmp_path / "run" / "contract.json").read_bytes() == contract
    run = tmp_path / "run"
    for name in ("grouping_records.pt", "grouping_manifest.json", "grouping_data_audit.json",
                 "grouping_teacher_fold_1.pt", "grouping_teacher_fold_2.pt", "grouping_teacher_completed.pt",
                 "grouping_results.json"):
        assert (run / name).exists(), name
    assert not (run / a.GROUPING_PROGRESS_DIR).exists() and not (run / a.PROGRESS_DIR).exists()
    # The grouping run's single log is split by each question's own Layer-1 log.
    audit = json.loads((run / "grouping_data_audit.json").read_text())
    combined = audit["grouping_logs"][f"{COMBINED}.json"]
    assert combined["total"] == combined["joined"] == combined["eligible"] == 28
    assert combined["joined_by_layer1_log"] == {"train": 24, "external": 4}
    assert combined["layer1_scores_misattached"] == 12 + 4
    assert {name: (row["role"], row["layer1_questions"], row["eligible"])
            for name, row in audit["layer1_logs"].items()} == {"train": ("train", 24, 24), "external": ("test", 4, 4)}
    assert "train (train)" in out and "external (test)" in out
    assert set(results["reports"]) == {"supervised", "audit", "external"}
    assert results["reports"]["supervised"]["questions"] == 24 and results["reports"]["external"]["questions"] == 4
    assert json.loads((run / "grouping_results.json").read_text()) == json.loads(json.dumps(results))
    for table in ("Best-of-N level", "Paired difference", "group outputs: BCE", "one-shots mixed"):
        assert table in out
    # Every label reached its output through corpus ids, mirrored Layer-1 logs included.
    for i, parsed in enumerate(work.records):
        source = json.loads(Path(cfg.train_file if parsed.benchmark == "train" else cfg.test_files[0]).read_text())
        item = next(x for x in source if x["target_query_text"].endswith(f" {parsed.uid.split('::')[1]}"))
        assert work.grouping["labels"][i].tolist() == expected_labels(item, parsed, cfg)
    # A new session reuses the grouping data and teacher.
    monkeypatch.setattr(a, "fit_teachers", lambda *args, **kwargs: pytest.fail("retrained"))
    monkeypatch.setattr(a, "read_grouping_log", lambda *args, **kwargs: pytest.fail("read again"))
    again = a.Workflow(cfg).build_datasets().prepare().grouping_report()
    assert again["reports"] == results["reports"]
    # Other grouping logs are refused rather than mixed in.
    log = Path(cfg.grouping_files[0])
    entries = json.loads(log.read_text())
    entries[-1]["layer1_grouping_state"]["ground_truth_labels"]["group_s2_p0-1"]["is_correct"] ^= True
    log.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="other grouping logs"):
        a.Workflow(cfg).build_datasets().build_grouping_dataset()


def test_one_combined_grouping_log_and_one_log_per_benchmark_give_the_same_labels(tmp_path):
    combined = a.Workflow(grouping_config(tmp_path, output_dir=str(tmp_path / "combined"))).build_datasets()
    split = a.Workflow(grouping_config(tmp_path, combined=False, output_dir=str(tmp_path / "split"))).build_datasets()
    combined.build_grouping_dataset()
    split.build_grouping_dataset()
    for key in ("labels", "true_rank", "eligible"):
        assert np.array_equal(combined.grouping[key], split.grouping[key])
    # The grouping teacher needs the training file's questions; a log of the test file alone is refused.
    cfg = grouping_config(tmp_path, grouping_files=[str(tmp_path / "external_grouping.json")],
                          output_dir=str(tmp_path / "tests_only"))
    work = a.Workflow(cfg).build_datasets().prepare().train_teacher()
    with pytest.raises(ValueError, match="No question of train_file"):
        work.train_grouping()
    assert int(work.grouping["eligible"].sum()) == 4


def test_an_interrupted_grouping_data_build_continues_with_the_logs_it_had_not_finished(tmp_path, monkeypatch):
    cfg = grouping_config(tmp_path, combined=False)
    original, reads = a.read_grouping_log, []

    def flaky(path, *args):
        reads.append(Path(path).name)
        if len(reads) == 2:
            raise RuntimeError("session ended")
        return original(path, *args)

    monkeypatch.setattr(a, "read_grouping_log", flaky)
    work = a.Workflow(cfg).build_datasets()
    with pytest.raises(RuntimeError, match="session ended"):
        work.build_grouping_dataset()
    progress = tmp_path / "run" / a.GROUPING_PROGRESS_DIR
    assert sorted(path.name for path in progress.iterdir()) == ["01_train_grouping.pt"]
    work = a.Workflow(cfg).build_datasets().build_grouping_dataset()
    assert reads == ["train_grouping.json", "external_grouping.json", "external_grouping.json"]
    assert not progress.exists() and int(work.grouping["eligible"].sum()) == 28


def test_the_shuffled_label_control_trains_beside_the_grouping_teacher(tmp_path, capsys):
    cfg = grouping_config(tmp_path, grouping_shuffle_control=True)
    results = a.Workflow(cfg).build_datasets().prepare().grouping_report()
    run = tmp_path / "run"
    assert (run / "grouping_shuffle_teacher_completed.pt").exists()
    control = a.load_checkpoint(run / "grouping_shuffle_teacher_completed.pt")
    assert control["grouping"]["control"] is True
    report = results["reports"]["audit"]
    assert "Shuffled-label control" in report["group_outputs"] and "Shuffled-label control" in report["levels"]["2"]
    assert "shuffled-label control" in capsys.readouterr().out.lower()
