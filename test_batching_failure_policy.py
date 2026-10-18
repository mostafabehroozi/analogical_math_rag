import threading

import pytest

from src.api_batching import BatchRefusedError, batch_refusals
from src.batching import BatchCoordinator, QuestionWorkItem


@pytest.mark.parametrize(
    "first_status, expected_indices",
    [
        ("SKIPPED_FAILSAFE", [0, 1]),
        ("FAILURE", [0]),
        ("PARSING_FAILED", [0]),
        ("API_ERROR", [0]),
    ],
)
def test_halt_policy_continues_after_failsafe_skip_only(first_status, expected_indices):
    coordinator = BatchCoordinator(
        {
            "BATCH_SIZE": 1,
            "BATCH_MAX_WORKERS": 1,
            "BATCH_WRITE_JOURNALS": False,
            "BATCH_FAILURE_POLICY": "halt_after_failed_batch",
            "DISTRIBUTED_EXECUTION_ENABLED": False,
        },
        "simplification",
        "core_simp_phase1",
    )
    committed = []

    def worker(item):
        return {"status": first_status if item.index == 0 else "SUCCESS"}

    def commit(results, batch_id, batch_number):
        committed.extend(results)

    results = coordinator.run(
        [QuestionWorkItem(0, "first"), QuestionWorkItem(1, "second")],
        worker,
        commit,
    )

    assert [result.item.index for result in results] == expected_indices
    assert [result.item.index for result in committed] == expected_indices
    # The stored skip remains a skip; the coordinator only classifies execution.
    assert results[0].value["status"] == first_status


def _distributed_coordinator():
    # Distributed workers continue after a failed batch by default.
    return BatchCoordinator(
        {"BATCH_SIZE": 2, "BATCH_MAX_WORKERS": 2, "BATCH_WRITE_JOURNALS": False,
         "DISTRIBUTED_EXECUTION_ENABLED": True},
        "grouping",
        "full",
    )


def test_refused_provider_batch_stops_before_the_commit():
    committed = []

    def worker(item):
        # The refused call already failed; the pipeline step reports FAILURE.
        batch_refusals.record("OpenRouter cannot batch author/model with provider routing {}.")
        return {"pipeline_status": "FAILURE"}

    with pytest.raises(BatchRefusedError, match="cannot batch author/model"):
        _distributed_coordinator().run(
            [QuestionWorkItem(0, "q0"), QuestionWorkItem(1, "q1"), QuestionWorkItem(2, "q2")],
            worker,
            lambda results, batch_id, batch_number: committed.extend(results),
        )
    assert committed == []


def test_refusal_recorded_before_the_batch_does_not_stop_it():
    batch_refusals.record("A refusal from an earlier run in this kernel.")
    committed = []
    _distributed_coordinator().run(
        [QuestionWorkItem(0, "q0")],
        lambda item: {"pipeline_status": "SUCCESS"},
        lambda results, batch_id, batch_number: committed.extend(results),
    )
    assert [result.item.index for result in committed] == [0]


def test_watchdog_prints_where_a_stalled_question_is(capsys):
    release = threading.Event()

    def stalled_step():
        release.wait(10)  # Stands in for a call that never returns.

    def worker(item):
        if item.index == 1:
            stalled_step()
        return {"pipeline_status": "SUCCESS"}

    coordinator = BatchCoordinator(
        {"BATCH_SIZE": 2, "BATCH_MAX_WORKERS": 2, "BATCH_WRITE_JOURNALS": False,
         "BATCH_WATCHDOG_MINUTES": 0.002},  # 0.12 s
        "grouping",
        "full",
    )
    timer = threading.Timer(0.5, release.set)
    timer.start()
    try:
        results = coordinator.run(
            [QuestionWorkItem(0, "q0"), QuestionWorkItem(1, "q1")],
            worker,
            lambda results, batch_id, batch_number: None,
        )
    finally:
        release.set()
        timer.cancel()
    output = capsys.readouterr().out
    assert [result.item.index for result in results] == [0, 1]
    # One full report naming the stalled question and its frame, then one-line reminders.
    assert output.count("No question finished for 0.002 min; still running: Q#1.") == 1
    assert "--- thread question" in output and "in stalled_step" in output
    assert "Still no question finished; running: Q#1." in output


def test_pipeline_failure_takes_precedence_over_skip():
    from src.batching import _terminal_status

    assert _terminal_status(
        {"pipeline_status": "FAILED_WORKER_EXCEPTION", "status": "SKIPPED_FAILSAFE"}
    ) == "FAILED"
