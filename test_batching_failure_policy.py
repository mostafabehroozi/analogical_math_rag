import pytest

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


def test_pipeline_failure_takes_precedence_over_skip():
    from src.batching import _terminal_status

    assert _terminal_status(
        {"pipeline_status": "FAILED_WORKER_EXCEPTION", "status": "SKIPPED_FAILSAFE"}
    ) == "FAILED"
