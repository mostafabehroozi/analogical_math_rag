"""Offline checks that a deferred solve run (DEFER_SOLVE_STEP) resumes from its run log.

Each rerun must redo nothing that the run log already holds: phase 1 skips every
question with a log, and phase 2 solves only the pending INTERMEDIATE_COMPLETE logs,
which stay in the run log until their solve step replaces them.
"""

import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import pytest


def _import_orchestration():
    # Only sentence_transformers is stubbed. patch.dict(sys.modules) would also
    # drop every src module first imported inside it, and a later test that
    # pickles one of their classes (Layer 2's ExperimentResult checkpoint)
    # would then meet a second copy of that module.
    stubbed = "sentence_transformers" not in sys.modules
    if stubbed:
        sentence_transformers = ModuleType("sentence_transformers")
        sentence_transformers.SentenceTransformer = type("SentenceTransformer", (), {})
        sys.modules["sentence_transformers"] = sentence_transformers
    try:
        from src import orchestration
    finally:
        if stubbed:
            del sys.modules["sentence_transformers"]
    return orchestration


orchestration = _import_orchestration()

EXPERIMENT_NAME = "Deferred_Resume"
QUESTIONS = ["q0", "q1", "q2"]
PENDING = "INTERMEDIATE_COMPLETE"


class Interrupted(BaseException):
    """Stands in for a notebook interruption: not caught by the batch worker guard."""


class FakePipeline:
    """Returns fake logs per run_mode and records every (run_mode, index) call."""

    def __init__(self, fail_solve=(), interrupt_solve=(), raise_solve=()):
        self.fail_solve = set(fail_solve)
        self.interrupt_solve = set(interrupt_solve)
        self.raise_solve = set(raise_solve)
        self.calls = []

    def __call__(self, hard_list_idx, target_query, run_mode, existing_log=None, **kwargs):
        self.calls.append((run_mode, hard_list_idx))
        if run_mode == "intermediate":
            return {
                "target_query_original_hard_list_idx": hard_list_idx,
                "target_query_text": target_query,
                "pipeline_status": PENDING,
                "intermediate_result": f"retrieved for {target_query}",
            }
        assert run_mode == "solve_only"
        assert existing_log["pipeline_status"] == PENDING
        assert existing_log["intermediate_result"] == f"retrieved for {target_query}"
        if hard_list_idx in self.interrupt_solve:
            raise Interrupted()
        if hard_list_idx in self.raise_solve:
            # The real solve step updates its log in place before it can raise.
            existing_log["pipeline_status"] = "PENDING_SOLVE"
            existing_log["partial_solve"] = "half done"
            raise RuntimeError(f"solve failed for {target_query}")
        status ="FAILURE" if hard_list_idx in self.fail_solve else "SUCCESS"
        return {**existing_log, "pipeline_status": status}


@pytest.fixture(params=["cross_experiment", "single_experiment"])
def branch(request):
    return request.param


@pytest.fixture(params=["per_query", "batch"])
def mode(request):
    return request.param


def _configs(results_dir, branch, mode):
    global_config = {
        "RESULTS_DIR": str(results_dir),
        "DISTRIBUTED_EXECUTION_ENABLED": False,
        "BATCH_PROCESSING_ENABLED": mode == "batch",
        "BATCH_SIZE": 1,
        "BATCH_MAX_WORKERS": 1,
        "BATCH_WRITE_JOURNALS": False,
        "BATCH_FAILURE_POLICY": "halt_after_failed_batch",
    }
    experiment = {"experiment_name": EXPERIMENT_NAME}
    # The cross-experiment branch runs when an experiment sets the flag itself;
    # the single-experiment branch when only the global config sets it.
    if branch == "cross_experiment":
        experiment["DEFER_SOLVE_STEP"] = True
    else:
        global_config["DEFER_SOLVE_STEP"] = True
    return [experiment], global_config


def _run(results_dir, branch, mode, pipeline):
    experiments, global_config = _configs(results_dir, branch, mode)
    with patch.object(orchestration, "run_pipeline_for_single_query", side_effect=pipeline), \
            patch.object(orchestration, "periodic_sync_check"), \
            patch.object(orchestration, "periodic_batch_sync_check"):
        return orchestration.run_experiments(
            experiments, global_config, list(QUESTIONS), None, {}, {"gemini": object()},
            ["a0", "a1", "a2"],
        )


def _saved_logs(results_dir):
    return json.loads((Path(results_dir) / f"{EXPERIMENT_NAME}_run_log.json").read_text())


def _statuses(logs):
    return [(log["target_query_original_hard_list_idx"], log["pipeline_status"]) for log in logs]


def test_rerun_after_a_complete_run_redoes_nothing(tmp_path, branch, mode):
    first = FakePipeline()
    _run(tmp_path, branch, mode, first)
    assert sorted(first.calls) == sorted(
        [("intermediate", index) for index in range(3)]
        + [("solve_only", index) for index in range(3)]
    )
    saved = _saved_logs(tmp_path)
    assert _statuses(saved) == [(0, "SUCCESS"), (1, "SUCCESS"), (2, "SUCCESS")]

    rerun = FakePipeline()
    results = _run(tmp_path, branch, mode, rerun)

    assert rerun.calls == []
    assert _saved_logs(tmp_path) == saved
    assert _statuses(results[EXPERIMENT_NAME]) == _statuses(saved)


@pytest.mark.parametrize(
    "mode, stop, rerun_solves, stopped_statuses",
    [
        # The notebook is interrupted while solving question 1.
        ("per_query", {"interrupt_solve": {1}}, [1, 2],
         [(0, "SUCCESS"), (1, PENDING), (2, PENDING)]),
        ("batch", {"interrupt_solve": {1}}, [1, 2],
         [(0, "SUCCESS"), (1, PENDING), (2, PENDING)]),
        # BATCH_FAILURE_POLICY=halt_after_failed_batch stops after question 1 fails.
        ("batch", {"fail_solve": {1}}, [2],
         [(0, "SUCCESS"), (1, "FAILURE"), (2, PENDING)]),
    ],
)
def test_rerun_after_a_partial_solve_phase_solves_only_pending_logs(
    tmp_path, branch, mode, stop, rerun_solves, stopped_statuses
):
    stopped = FakePipeline(**stop)
    if stop.get("interrupt_solve"):
        with pytest.raises(Interrupted):
            _run(tmp_path, branch, mode, stopped)
    else:
        _run(tmp_path, branch, mode, stopped)

    # The pending intermediate logs survive the early stop.
    assert _statuses(_saved_logs(tmp_path)) == stopped_statuses

    rerun = FakePipeline()
    results = _run(tmp_path, branch, mode, rerun)

    assert rerun.calls == [("solve_only", index) for index in rerun_solves]
    final = [(0, "SUCCESS"), (1, "FAILURE" if stop.get("fail_solve") else "SUCCESS"), (2, "SUCCESS")]
    assert _statuses(_saved_logs(tmp_path)) == final
    assert _statuses(results[EXPERIMENT_NAME]) == final


def test_raised_batch_solve_keeps_the_pending_log_for_a_retry(tmp_path, branch):
    # The batch worker catches the exception, and BATCH_FAILURE_POLICY then
    # stops before question 2.
    stopped = FakePipeline(raise_solve={1})
    _run(tmp_path, branch, "batch", stopped)

    assert stopped.calls == (
        [("intermediate", index) for index in range(3)]
        + [("solve_only", 0), ("solve_only", 1)]
    )
    saved = _saved_logs(tmp_path)
    assert _statuses(saved) == [(0, "SUCCESS"), (1, PENDING), (2, PENDING)]
    # Question 1 keeps its intermediate results, not the partial solve, and
    # records the error.
    failed = saved[1]
    assert failed["intermediate_result"] == "retrieved for q1"
    assert "partial_solve" not in failed
    assert failed["batch_error"]["type"] == "RuntimeError"
    assert failed["batch_error"]["message"] == "solve failed for q1"
    assert failed["batch_metadata"]["phase"] == "solve"
    assert failed["batch_metadata"]["terminal_status"] == "FAILED"

    rerun = FakePipeline()
    results = _run(tmp_path, branch, "batch", rerun)

    assert rerun.calls == [("solve_only", 1), ("solve_only", 2)]
    final = [(0, "SUCCESS"), (1, "SUCCESS"), (2, "SUCCESS")]
    assert _statuses(_saved_logs(tmp_path)) == final
    assert _statuses(results[EXPERIMENT_NAME]) == final


def test_pending_duplicate_of_a_solved_question_is_not_solved_again(tmp_path, branch, mode):
    # Older code reran phase 1 for solved questions and, per query, appended a
    # second, pending log for the same index.
    solved = {
        "target_query_original_hard_list_idx": 0, "target_query_text": "q0",
        "pipeline_status": "SUCCESS", "intermediate_result": "retrieved for q0",
    }
    pending = {
        "target_query_original_hard_list_idx": 1, "target_query_text": "q1",
        "pipeline_status": PENDING, "intermediate_result": "retrieved for q1",
    }
    stale_duplicate = {**solved, "pipeline_status": PENDING}
    (tmp_path / f"{EXPERIMENT_NAME}_run_log.json").write_text(
        json.dumps([solved, pending, stale_duplicate])
    )

    rerun = FakePipeline()
    results = _run(tmp_path, branch, mode, rerun)

    assert rerun.calls == [("intermediate", 2), ("solve_only", 1), ("solve_only", 2)]
    final = [(0, "SUCCESS"), (1, "SUCCESS"), (2, "SUCCESS")]
    assert _statuses(_saved_logs(tmp_path)) == final
    assert _statuses(results[EXPERIMENT_NAME]) == final
