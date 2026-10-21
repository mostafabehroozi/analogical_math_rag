"""Offline checks of the decision_heads switch: with the heads off (or the grouping phase on, which has
priority) Stage 3 never runs, and Stage 4 reports the full pool and the fixed orders, with the learned
heads' rows as zeros."""
import json
from pathlib import Path

import pytest

import adaptive_analogical_training as a
from test_adaptive_grouping import grouping_config
from test_adaptive_session_resume import small_config

STAGE3 = ("train_pretrained_head", "train_finetuned_head", "finetune_labels", "fit_action_head",
          "build_route_labels", "build_finetune_labels")


def forbid_stage3(monkeypatch):
    for name in STAGE3:
        monkeypatch.setattr(a, name, lambda *args, _name=name, **kwargs: pytest.fail(f"Stage 3 ran: {_name}"))


def all_zero(value):
    if isinstance(value, dict):
        return all(all_zero(item) for item in value.values())
    if isinstance(value, list):
        return all(all_zero(item) for item in value)
    return isinstance(value, str) or value == 0


def test_the_grouping_phase_has_priority_over_the_decision_heads():
    assert a.Config().use_decision_heads is True
    assert a.Config(decision_heads=False).use_decision_heads is False
    assert a.Config(grouping=True).use_decision_heads is False
    assert a.Config(grouping=True, decision_heads=True).use_decision_heads is False
    # The switch changes no stage's output, so it stays out of the run contract.
    assert a.contract_settings(a.Config(decision_heads=False)) == a.contract_settings(a.Config())
    assert a.evaluation_methods(a.Config(decision_heads=False)) == ("full", *a.Config().fixed_acquisition_orders)
    assert a.reported_methods(a.Config(decision_heads=False)) == a.evaluation_methods(a.Config())
    with pytest.raises(AssertionError):
        a.Config(decision_heads=1).validate()


def check_heads_off_report(work, cfg, run):
    results = work.results
    assert results["decision_heads"] is False and work.heads == {}
    assert not (run / "decision_labels").exists()
    for name in ("decision_head_pretrained.pt", "decision_head_completed.pt", "inference_bundle.pt"):
        assert not (run / name).exists(), name
    assert all_zero(results["decision_training"]["supervised"])
    fixed = list(cfg.fixed_acquisition_orders)
    for name in ["audit"] + [Path(path).stem for path in cfg.test_files]:
        adaptive = results[name]["adaptive"]
        policies = adaptive["policies"]
        assert list(policies) == ["full", *fixed, *a.LEARNED_METHODS]
        for method in a.LEARNED_METHODS:
            assert all_zero(policies[method]) and all_zero(adaptive["recognition_outcomes"][method])
            assert all_zero(adaptive["continuation"][method])
        assert policies["full"]["mean_cost"] > 0 and all(policies[method]["mean_cost"] > 0 for method in fixed)
        assert all(all_zero(item) for pair, item in adaptive["comparisons"].items() if "supervised" in pair)
        assert any(not all_zero(item) for pair, item in adaptive["comparisons"].items() if "supervised" not in pair)
        assert all_zero(results[name]["paired_top1"])
    assert set(results["external_macro"]["methods"]) == set(a.reported_methods(cfg))
    assert all_zero(results["external_macro"]["methods"]["supervised"])
    methods = {json.loads(line)["method"] for line in (run / a.ROLLOUT_FILE).read_text().splitlines()}
    assert methods == {"full", *fixed}  # The learned heads were never rolled out.


def test_with_the_decision_heads_off_stage_4_reports_full_and_fixed_orders_and_zero_learned_heads(
        tmp_path, monkeypatch, capsys):
    cfg = small_config(tmp_path, decision_heads=False)
    forbid_stage3(monkeypatch)
    work = a.run_pipeline(cfg)
    out = capsys.readouterr().out
    assert "Decision heads OFF (decision_heads=False)" in out and "STAGE 3 | Skipped: decision_heads=False" in out
    assert "were not trained or rolled out and are shown as zeros" in out
    check_heads_off_report(work, cfg, tmp_path / "run")
    # rollout_audit replays only the methods that ran.
    audit = work.rollout_audit("audit", 0)
    assert {row["method"] for row in audit["rows"]} == {"full", *cfg.fixed_acquisition_orders}


def test_turning_the_grouping_phase_on_switches_the_decision_heads_off(tmp_path, monkeypatch, capsys):
    cfg = grouping_config(tmp_path, decision_heads=True)
    forbid_stage3(monkeypatch)
    work = a.run_pipeline(cfg)
    out = capsys.readouterr().out
    assert "grouping=True has priority over decision_heads" in out
    assert "STAGE 3 | Skipped: grouping=True" in out
    assert (tmp_path / "run" / "grouping_results.json").exists()
    check_heads_off_report(work, cfg, tmp_path / "run")
