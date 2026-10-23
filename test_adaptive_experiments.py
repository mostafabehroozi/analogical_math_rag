"""Offline checks of the stages setting and of experiment suites: several configurations, each running its
own chain of stages in its own folder and mirror, then compared with paired differences."""
import json
import shutil
from itertools import permutations

import numpy as np
import pytest

import adaptive_analogical_training as a
from test_adaptive_analogical_training import fake_predictor, record
from test_adaptive_session_resume import FakeHub, small_config


def test_stages_form_a_chain_and_stay_outside_the_run_contract():
    cfg = a.Config()
    assert cfg.stages == ("teacher", "snapshot", "decision_heads")
    assert (cfg.use_snapshot, cfg.use_decision_heads, cfg.grouping) == (True, True, False)
    teacher = a.Config(stages=("teacher",))
    assert (teacher.use_snapshot, teacher.use_decision_heads) == (False, False)
    grouping = a.Config(stages=("teacher", "snapshot", "decision_heads", "grouping"))
    assert grouping.grouping and not grouping.use_decision_heads  # Grouping has priority.
    assert a.stage_chain(grouping) == "teacher -> grouping -> snapshot"
    assert a.heads_off_reason(grouping) == "'grouping' is in stages and has priority over 'decision_heads'"
    for stages, message in ((("snapshot",), "'snapshot' needs 'teacher'"),
                            (("teacher", "decision_heads"), "'decision_heads' needs 'snapshot'"),
                            (("grouping",), "'grouping' needs 'teacher'"),
                            (("teacher", "teacher"), "distinct phases"), (("teacher", "planner"), "distinct phases"),
                            ("teacher", "distinct phases"), ((), "distinct phases")):
        with pytest.raises(ValueError, match=message):
            a.Config(stages=stages).validate()
    a.Config(stages=["teacher", "snapshot"]).validate()  # A list works too.
    # Which stages run changes no stage's output: one run contract.
    assert a.contract_settings(teacher) == a.contract_settings(cfg)
    # Saved configurations from before `stages` (inference bundles) keep their meaning.
    old = {key: value for key, value in a.asdict(cfg).items() if key != "stages"}
    assert a.saved_config({**old, "decision_heads": True, "grouping": False}).stages == cfg.stages
    assert a.saved_config({**old, "decision_heads": False}).stages == ("teacher", "snapshot")
    assert a.saved_config({**old, "grouping": True}).stages == ("teacher", "snapshot", "decision_heads", "grouping")
    assert a.saved_config(a.asdict(teacher)).stages == ("teacher",)


def test_the_plan_checks_every_experiment_before_anything_runs(tmp_path):
    base = a.Config(output_dir=str(tmp_path / "run"), hf_repo="user/run")
    plan = a.experiment_plan(base, [{"name": "a", "stages": ["teacher"]},
                                    {"name": "b.1", "teacher_complete_share": .01, "test_files": ("x/t.json",)}])
    (name, cfg, changes), (_, other, other_changes) = plan
    assert (name, cfg.output_dir, cfg.hf_repo) == ("a", str(tmp_path / "run_experiments" / "a"), "user/run-a")
    assert cfg.stages == ("teacher",) and changes == {"stages": ["teacher"]}
    assert other.teacher_complete_share == .01 and other.test_files == ["x/t.json"] and other.stages == base.stages
    assert base.output_dir == str(tmp_path / "run") and base.teacher_complete_share == .15  # Untouched.
    folder = str(tmp_path / "run_experiments" / "a")
    for experiments, message in (
            ([], "No experiments"), ([{"stages": ("teacher",)}], "must be a dict with a 'name'"),
            ([{"name": "a"}, {"name": "a"}], "Two experiments are named 'a'"),
            ([{"name": "bad name"}], "Experiment name"), ([{"name": "a--b"}], "Experiment name"),
            ([{"name": "a", "teacher_complete_shar": .1}], "did you mean 'teacher_complete_share'"),
            ([{"name": "a", "stages": ("snapshot",)}], "'snapshot' needs 'teacher'"),
            ([{"name": "a", "teacher_complete_share": 1.5}], "Experiment 'a' has an invalid configuration"),
            ([{"name": "a"}, {"name": "b", "output_dir": folder, "hf_repo": "user/run-a", "seed": 1}],
             "share .* but not its run contract"),
            ([{"name": "a"}, {"name": "b", "hf_repo": "user/run-a"}], "use the Hub mirror user/run-a")):
        with pytest.raises(ValueError, match=message):
            a.experiment_plan(base, experiments)
    # A grouping experiment may join a finished run's folder: grouping stays outside the run contract.
    a.experiment_plan(base, [{"name": "a"}, {"name": "g", "output_dir": folder, "hf_repo": "user/run-a",
                                              "stages": ("teacher", "grouping"), "grouping_files": ["/g.json"]}])
    (tmp_path / "single").mkdir()
    (tmp_path / "single" / "contract.json").write_text("{}")
    with pytest.raises(ValueError, match="holds a single run"):
        a.experiment_plan(base, [{"name": "a"}], tmp_path / "single")


def brute_fidelity(rec, scores, safe, maximum, predictor, states, cfg):
    """fidelity_sums of one question, one state and zero-shot order at a time."""
    sums = dict.fromkeys(a.FIDELITY_FIELDS, 0.)
    for order in permutations(range(cfg.zero_shots)):
        permuted, listed, permuted_safe, permuted_max = a.permute_zero_shots(rec, scores, safe, maximum, order, cfg)
        _, probabilities = predictor.predict_many(permuted, states)
        for state, p in zip(states, probabilities):
            present = np.flatnonzero(a.present_mask(state, cfg))
            mine, theirs = present[np.argmax(p[present])], present[np.argmax(listed[present])]
            pairs = [(i, j) for i in present for j in present if i < j and listed[i] != listed[j]]
            sums["views"] += 1
            sums["agree"] += mine == theirs
            if pairs:
                sums["tau"] += np.mean([np.sign(p[i] - p[j]) * np.sign(listed[i] - listed[j]) for i, j in pairs])
                sums["tau_views"] += 1
            sums["snapshot_top1"] += permuted.labels[mine]
            sums["list_top1"] += permuted.labels[theirs]
            for name, column, target in (("max", 3*cfg.n+1, permuted_max), ("safe", 3*cfg.n, permuted_safe)):
                predicted, true = p[column] >= cfg.recognition_threshold, bool(target[present].any())
                sums[f"{name}_predicted"] += predicted
                sums[f"{name}_true_positive"] += predicted and true
                sums[f"{name}_true"] += true
    return np.asarray([sums[name] for name in a.FIDELITY_FIELDS])


def test_snapshot_fidelity_counts_agreement_with_the_teachers_list_view_by_view():
    cfg = a.Config(k=2, zero_shots=2, hidden_dim=8, residual_blocks=1, device="cpu", recognition_threshold=.5)
    records = [record(cfg, i) for i in range(4)]
    predictor = fake_predictor(cfg)
    rng = np.random.RandomState(3)
    scores = rng.rand(len(records), cfg.n).astype(np.float32)
    scores[0, 1] = scores[0, 2]  # A tie in the teacher's list is not an ordered pair.
    safe, maximum = zip(*(a.teacher_labels(s, r.labels, cfg) for s, r in zip(scores, records)))
    teacher = {"scores": scores, "safe": np.stack(safe), "maximum": np.stack(maximum)}
    states = a.reachable_states(cfg)
    views = a.SnapshotViews(records, range(len(records)), cfg, "cpu", teacher, states)
    sums = a.fidelity_sums(predictor, views, scores, cfg)
    for i, rec in enumerate(records):
        expected = brute_fidelity(rec, scores[i], teacher["safe"][i], teacher["maximum"][i], predictor, states, cfg)
        assert np.allclose(sums[i], expected, atol=1e-6), (i, sums[i], expected)
    result = a.snapshot_fidelity(records, {"dev": [0, 1, 2, 3], "one": [2]}, teacher, predictor, cfg)
    reachable = result["sets"]["dev"]["reachable"]
    assert reachable["views"] == len(records) * len(states) * 2
    assert reachable["agree_top1"] == pytest.approx(sums[:, 1].sum() / sums[:, 0].sum())
    assert result["sets"]["one"]["complete"]["views"] == 2  # One question, two zero-shot orders.
    assert result["per_question"]["one"] == {"uids": [records[2].uid], "agree_top1": [sums[2, 1] / sums[2, 0]]}


def test_a_complete_share_of_one_trains_the_teacher_on_complete_views_with_an_all_state_schedule(tmp_path, capsys):
    cfg = small_config(tmp_path, teacher_complete_share=1., stages=("teacher",))
    work = a.run_pipeline(cfg)
    out = capsys.readouterr().out
    assert "teacher_complete_share=1: the teacher trains on complete views only" in out
    assert "x 1 presence states" in out and "one all-state epoch is" in out
    fold = a.load_checkpoint(tmp_path / "run" / "teacher_fold_1.pt")
    count, every = len(fold["train_ids"]), len(a.presence_states(cfg.zero_shots, cfg.k)) * 2
    # An epoch of complete views takes as many optimizer steps as one all-state epoch.
    assert fold["schedule_scale"] == a.epoch_steps(every, count, cfg.batch_size) / a.epoch_steps(2, count, cfg.batch_size)
    assert fold["schedule_scale"] > 1
    assert 0 < a.model_training(tmp_path / "run" / "teacher_fold_1.pt")["best_epoch"] <= cfg.teacher_epochs
    assert work.results["supervised"]["ranking"]["ResNet teacher"]["n"] == len(work.splits["supervised"])


def test_a_suite_runs_each_configuration_through_its_own_stages_and_compares_them(tmp_path, capsys, monkeypatch):
    base = small_config(tmp_path)
    root = tmp_path / "suite"
    experiments = [{"name": "teacher_only", "stages": ("teacher",)},
                   {"name": "complete_only", "stages": ("teacher",), "teacher_complete_share": 1.},
                   {"name": "with_snapshot", "stages": ("teacher", "snapshot")},
                   {"name": "full_chain", "stages": ("teacher", "snapshot", "decision_heads")}]
    result = a.run_experiments(base, experiments, root)
    out = capsys.readouterr().out
    summaries = {summary["name"]: summary for summary in result["experiments"]}
    assert list(summaries) == [entry["name"] for entry in experiments]
    for name in summaries:
        assert (root / name / a.experiment_file(name)).exists()
    assert not (root / "teacher_only" / "snapshot_completed.pt").exists()
    assert (root / "with_snapshot" / "snapshot_completed.pt").exists()
    assert not (root / "with_snapshot" / "decision_head_pretrained.pt").exists()
    assert (root / "full_chain" / "inference_bundle.pt").exists()
    assert "No 'snapshot' in stages: Stages 2-4 are skipped" in out
    assert "STAGE 3 | Skipped: 'decision_heads' is not in stages" in out
    # The suite prints compact summaries, not the per-set report tables.
    assert "1) API calls versus accuracy" not in out and "TEACHER DEVELOPMENT REPORT" not in out
    for title in ("EXPERIMENTS | 4 configurations", "SUMMARY | teacher_only | teacher",
                  "SUMMARY | full_chain | teacher -> snapshot -> decision_heads", "Snapshot vs the teacher's list",
                  "EXPERIMENT COMPARISON | 4 configurations", "Stage 4 on audit"):
        assert title in out, title
    teacher_only, snapshot, chain = summaries["teacher_only"], summaries["with_snapshot"], summaries["full_chain"]
    assert list(teacher_only["teacher"]["sets"]) == ["supervised", "dev", "external"]
    assert teacher_only["snapshot"] is None and teacher_only["acquisition"] is None
    assert snapshot["decision_heads"] is None
    assert snapshot["acquisition"]["methods"] == ["full", *base.fixed_acquisition_orders]
    assert chain["acquisition"]["methods"] == ["full", *base.fixed_acquisition_orders, *a.LEARNED_METHODS]
    assert set(chain["decision_heads"]) == set(a.LEARNED_METHODS)
    assert set(snapshot["snapshot"]["sets"]["dev"]) == {"reachable", "complete"}
    assert summaries["complete_only"]["teacher"]["training"][0]["best_epoch"] <= base.teacher_epochs
    comparison = result["comparison"]
    assert comparison["teacher"]["reference"] == "teacher_only"
    assert set(comparison["teacher"]["deltas"]) == {"complete_only", "with_snapshot", "full_chain"}
    # Identical Stage 1 settings train identical teachers: the paired difference is exactly zero.
    for name in ("with_snapshot", "full_chain"):
        assert comparison["teacher"]["deltas"][name]["ap"]["delta"] == 0
    assert comparison["snapshot"]["reference"] == "with_snapshot"
    assert comparison["snapshot"]["deltas"]["full_chain"]["agree_top1"]["delta"] == 0
    # Stage 4 compares each method with the first experiment that ran it.
    assert comparison["acquisition"]["reference"] == {"full": "with_snapshot", "fixed": "with_snapshot",
                                                      "fixed_evaluators_first": "with_snapshot",
                                                      "fixed_zero_shots_first": "with_snapshot",
                                                      "supervised_pretrained": "full_chain", "supervised": "full_chain"}
    learned = comparison["acquisition"]["deltas"]["full_chain"]
    assert learned["full"]["top1"]["delta"] == 0 and "supervised" not in learned
    saved = json.loads((root / "experiments_summary.json").read_text(encoding="utf-8"))
    assert [item["name"] for item in saved["experiments"]] == list(summaries)
    assert "per_question" not in saved["experiments"][0]

    # A new session: every experiment finished, so nothing trains and the comparison is the same.
    monkeypatch.setattr(a, "fit_teachers", lambda *args, **kwargs: pytest.fail("a finished experiment trained"))
    again = a.run_experiments(base, experiments, root)
    out = capsys.readouterr().out
    assert out.count("Finished in an earlier session with these settings") == 4
    assert again["comparison"] == comparison
    # Changing an experiment's settings under the same name runs it, and its run contract objects.
    changed = [{**experiments[0], "teacher_complete_share": .5}]
    with pytest.raises(ValueError, match="different config/data contract"):
        a.run_experiments(base, changed, root)
    assert "was written with other settings or stages" in capsys.readouterr().out


def test_a_finished_experiment_is_read_from_its_mirror_without_downloading_the_run(tmp_path, monkeypatch, capsys):
    hub = FakeHub()
    monkeypatch.setattr(a, "hub_api", lambda: hub)
    base = small_config(tmp_path, hf_repo="user/run", stages=("teacher",))
    root = tmp_path / "suite"
    experiments = [{"name": "one"}, {"name": "two", "teacher_complete_share": .5}]
    first = a.run_experiments(base, experiments, root)
    assert {"user/run-one", "user/run-two"} <= set(hub.repos)
    assert a.experiment_file("two") in hub.repos["user/run-two"]
    shutil.rmtree(root)  # A new Kaggle session: only the mirrors remain.
    monkeypatch.setattr(a, "fit_teachers", lambda *args, **kwargs: pytest.fail("a finished experiment trained"))
    capsys.readouterr()
    again = a.run_experiments(base, experiments, root)
    assert capsys.readouterr().out.count("Finished in an earlier session") == 2
    assert again["comparison"] == first["comparison"]
    # Only the summaries were downloaded, outside the output folders, which stay absent until a run needs them.
    assert not (root / "one").exists() and not (root / "two").exists()
    assert (root / ".summaries" / "one" / a.experiment_file("one")).is_file()


def test_a_grouping_experiment_tests_the_grouping_phase_without_the_snapshot(tmp_path, monkeypatch, capsys):
    from test_adaptive_grouping import grouping_config
    base = grouping_config(tmp_path, stages=("teacher",))
    monkeypatch.setattr(a, "fit_snapshot", lambda *args, **kwargs: pytest.fail("Stage 2 ran"))
    result = a.run_experiments(base, [{"name": "ranking"}, {"name": "grouping", "stages": ("teacher", "grouping")}],
                               tmp_path / "suite")
    out = capsys.readouterr().out
    ranking, grouping = result["experiments"]
    assert ranking["grouping"] is None and grouping["grouping"]["sets"]["audit"]["questions"] > 0
    assert grouping["snapshot"] is None and grouping["stages"] == "teacher -> grouping"
    assert "Grouping teacher:" in out and "Grouping on audit (pass@1)" in out
    assert (tmp_path / "suite" / "grouping" / a.GROUPING_RESULTS).exists()
