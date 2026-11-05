"""Offline checks for the API failures of the Kaggle run of 2026-11-02.

One proxy 403 disabled the only Kaggle pseudo-key, every later call failed at
once without a retry, and the 658 questions those calls belonged to were saved
as SUCCESS. A call that fails for good now fails its question, which the batch
policy stops on and a resumed run redoes.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src import api_manager, context_logger
from src.batching import BatchCoordinator, QuestionWorkItem
from src.parallel_utils import run_parallel_api_calls
from test_batch_providers import BATCH_CONFIG, _Quiet, concurrently, fake_kbench
from test_deferred_solve_resume import FakePipeline, orchestration

DENIED = (
    "Traceback (most recent call last):\n"
    '  File "/benchmarks/.venv/lib/python3.12/site-packages/openai/_base_client.py", line 1047, in request\n'
    "    raise self._make_status_error_from_response(err.response) from None\n"
    "                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
    "openai.PermissionDeniedError: Error code: 403 - {'error': {'message': 'denied'}}"
)
NO_QUOTA = {"status": "RATE_LIMITED", "text": None, "error_type": "ProactiveRateLimit",
            "error_message": "No eligible Kaggle proxy quota for m."}
NO_RETRY = {"ENABLE_API_RETRY": False, "GLOBAL_CONSECUTIVE_ERROR_LIMIT": 0,
            "GLOBAL_PERIODIC_PAUSE_INTERVAL_MINUTES": 0}


class KaggleProxyDenialTests(_Quiet):
    def setUp(self):
        super().setUp()
        self.kbench, self.state = fake_kbench()
        patcher = patch.dict("sys.modules", {"kaggle_benchmarks": self.kbench})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.denied = {"once"}
        llm = self.kbench.llms["google/gemini-2.5-flash"]

        def prompt(message, **options):
            self.state["prompts"].append(message)
            if message in self.denied:
                self.denied.discard(message)
                raise RuntimeError(DENIED)
            return f"answer:{message}"

        llm.prompt = prompt

    def manager(self, **overrides):
        config = {**BATCH_CONFIG, "API_BATCHING_ENABLED": False, "ENABLE_API_RETRY": True,
                  "MAX_API_RETRIES": 3, "API_RETRY_DELAY_SECONDS": 0, **overrides}
        return api_manager.KaggleBenchmarksAPIManager({"default": {"rpm": 1000}}, config)

    def test_a_denied_request_is_retried_and_the_shared_key_stays_usable(self):
        manager = self.manager()
        self.assertEqual(manager.generate_content("once", "google/gemini-2.5-flash")["text"], "answer:once")
        self.assertEqual(self.state["prompts"], ["once", "once"])
        self.assertEqual(manager.scheduler.snapshot()["disabled_keys"], {})
        self.assertEqual(manager.generate_content("later", "google/gemini-2.5-flash")["text"], "answer:later")

    def test_a_denied_batch_row_leaves_quota_for_its_siblings_and_later_calls(self):
        manager = self.manager(API_BATCHING_ENABLED=True, ENABLE_API_RETRY=False)
        results = concurrently(*[
            (lambda prompt=prompt: manager.generate_content(prompt, "google/gemini-2.5-flash"))
            for prompt in ("a", "once", "c")
        ])
        self.assertEqual([result["status"] for result in results], ["SUCCESS", "ERROR", "SUCCESS"])
        self.assertEqual(results[1]["error_type"], "ProxyPermissionDeniedError")
        self.assertIn("ProxyPermissionDeniedError", api_manager.RETRYABLE_ERROR_TYPES)
        self.assertEqual(manager.generate_content("later", "google/gemini-2.5-flash")["text"], "answer:later")

    def test_the_error_message_leads_with_the_exception_line(self):
        failure = self.manager()._failure(DENIED, {}, "m")
        self.assertTrue(failure["error_message"].startswith(
            "openai.PermissionDeniedError: Error code: 403"), failure["error_message"])
        self.assertIn("_base_client.py", failure["error_message"])  # The traceback follows.
        one_line = self.manager()._failure("RateLimitError: 429 quota", {}, "m")["error_message"]
        self.assertEqual(one_line, "RateLimitError: 429 quota")

    def test_the_exception_line_decides_over_names_in_the_frames(self):
        message = 'File "x.py", in _retry_after_RateLimitError\n' + DENIED
        self.assertEqual(api_manager._kaggle_error_type(message), "PermissionDeniedError")


def failing_call(config, response=NO_QUOTA):
    return api_manager.execute_with_retry(config, "kaggle", "m", "prompt", lambda: dict(response))


@pytest.fixture(autouse=True)
def quiet_api_logs():
    with patch.object(api_manager, "tprint"):
        yield


def test_a_failed_call_is_printed_in_full():
    detail = "openai.PermissionDeniedError: Error code: 403 - {'code': '', 'message': '" + "x" * 500 + "'}"
    message = DENIED.rsplit("\n", 1)[0] + "\n" + detail
    with patch.object(api_manager, "tprint") as printed:
        failing_call(NO_RETRY, {"status": "ERROR", "error_type": "PermissionDeniedError", "error_message": message})
    attempt_log, final_line = (call.args[0] for call in printed.call_args_list)
    assert detail in attempt_log and "_base_client.py" in attempt_log
    assert final_line.endswith(detail)
    ok = api_manager._format_api_log("p", "m", "k", 0.1, 1, 1, "SUCCESS", "prompt", "y" * 500, {})
    assert "y" * 71 not in ok  # Successful replies stay one short line.


def test_a_call_no_key_could_take_does_not_pause_every_api_call():
    pause = api_manager._GlobalAPIPauseManager()
    config = {**NO_RETRY, "GLOBAL_CONSECUTIVE_ERROR_LIMIT": 1, "GLOBAL_CONSECUTIVE_ERROR_PAUSE_SECONDS": 600}
    with patch.object(api_manager, "global_pause_manager", pause):
        failing_call(config)
        assert pause.pause_remaining("kaggle") == 0.0
        failing_call(config, {"status": "ERROR", "error_type": "APIConnectionError", "error_message": "down"})
        assert pause.pause_remaining("kaggle") > 0.0
        # The backoff pauses the failing provider only.
        assert pause.pause_remaining("openrouter") == 0.0


def _coordinator(**overrides):
    config = {"BATCH_SIZE": 1, "BATCH_MAX_WORKERS": 1, "BATCH_WRITE_JOURNALS": False,
              "BATCH_FAILURE_POLICY": "halt_after_failed_batch", "DISTRIBUTED_EXECUTION_ENABLED": False,
              **overrides}
    return BatchCoordinator(config, "experiment", "full")


def test_a_question_whose_call_failed_for_good_fails_and_stops_the_run():
    def worker(item):
        if item.index == 0:
            # The failing call runs in the question's API pool, as in the pipeline.
            run_parallel_api_calls([lambda: failing_call(NO_RETRY)], {"QUESTION_PARALLEL_API_ENABLED": True})
        return {"pipeline_status": "SUCCESS"}

    results = _coordinator().run([QuestionWorkItem(0, "q0"), QuestionWorkItem(1, "q1")], worker,
                                 lambda results, batch_id, batch_number: None)

    assert [(result.item.index, result.terminal_status) for result in results] == [(0, "FAILED")]
    (failure,) = results[0].value["api_failures"]
    assert failure["error_type"] == "ProactiveRateLimit"
    assert failure["error_message"] == "No eligible Kaggle proxy quota for m."
    assert failure["provider"] == "kaggle"


@pytest.mark.parametrize("error_type", sorted(api_manager.REQUEST_OUTCOME_ERROR_TYPES))
def test_a_reply_that_is_the_requests_own_outcome_does_not_fail_the_question(error_type):
    def worker(item):
        failing_call(NO_RETRY, {"status": "BLOCKED", "error_type": error_type, "error_message": "x"})
        return {"pipeline_status": "SUCCESS", "api_failures": ["from an earlier attempt"]}

    (result,) = _coordinator().run([QuestionWorkItem(0, "q0")], worker, lambda *args: None)
    assert result.terminal_status == "SUCCESS"
    assert "api_failures" not in result.value


def test_a_failure_the_caller_recovers_from_does_not_fail_the_question():
    def worker(item):
        with context_logger.tolerated_api_failures():
            failing_call(NO_RETRY)
        return {"pipeline_status": "SUCCESS"}

    (result,) = _coordinator().run([QuestionWorkItem(0, "q0")], worker, lambda *args: None)
    assert result.terminal_status == "SUCCESS"


def test_calls_outside_a_batch_question_record_nothing():
    assert failing_call(NO_RETRY)["error_type"] == "ProactiveRateLimit"
    assert context_logger.ctx_api_failures.get() is None


class ApiFailurePipeline(FakePipeline):
    """The fake pipeline, where some questions make a call that fails for good."""

    def __init__(self, fail_calls=(), fail_solve_calls=(), **kwargs):
        super().__init__(**kwargs)
        self.fail_calls = set(fail_calls)
        self.fail_solve_calls = set(fail_solve_calls)

    def __call__(self, hard_list_idx, target_query, run_mode, existing_log=None, **kwargs):
        if hard_list_idx in self.fail_calls or (
            run_mode == "solve_only" and hard_list_idx in self.fail_solve_calls
        ):
            failing_call(NO_RETRY)
        if run_mode == "full":
            self.calls.append((run_mode, hard_list_idx))
            return {"target_query_original_hard_list_idx": hard_list_idx,
                    "target_query_text": target_query, "pipeline_status": "SUCCESS"}
        return super().__call__(hard_list_idx, target_query, run_mode, existing_log, **kwargs)


def _run(results_dir, pipeline, defer=False):
    global_config = {
        "RESULTS_DIR": str(results_dir), "DISTRIBUTED_EXECUTION_ENABLED": False,
        "BATCH_PROCESSING_ENABLED": True, "BATCH_SIZE": 1, "BATCH_MAX_WORKERS": 1,
        "BATCH_WRITE_JOURNALS": False, "BATCH_FAILURE_POLICY": "halt_after_failed_batch",
        "DEFER_SOLVE_STEP": defer,
    }
    with patch.object(orchestration, "run_pipeline_for_single_query", side_effect=pipeline), \
            patch.object(orchestration, "periodic_sync_check"), \
            patch.object(orchestration, "periodic_batch_sync_check"):
        orchestration.run_experiments(
            [{"experiment_name": "Api_Failures"}], global_config, ["q0", "q1", "q2"], None, {},
            {"gemini": object()}, ["a0", "a1", "a2"],
        )
    logs = json.loads((Path(results_dir) / "Api_Failures_run_log.json").read_text())
    return {log["target_query_original_hard_list_idx"]: log for log in logs}


def test_a_resumed_run_redoes_a_question_saved_with_api_failures(tmp_path):
    stopped = ApiFailurePipeline(fail_calls={1})
    logs = _run(tmp_path, stopped)
    assert stopped.calls == [("full", 0), ("full", 1)]  # The batch policy stops after Q#1.
    assert logs[0]["pipeline_status"] == "SUCCESS"
    assert logs[1]["pipeline_status"] == orchestration.API_FAILURE_STATUS
    assert logs[1]["api_failures"][0]["error_type"] == "ProactiveRateLimit"
    assert logs[1]["batch_metadata"]["terminal_status"] == "FAILED"

    rerun = ApiFailurePipeline()
    logs = _run(tmp_path, rerun)
    assert rerun.calls == [("full", 1), ("full", 2)]
    assert {index: log["pipeline_status"] for index, log in logs.items()} == {
        0: "SUCCESS", 1: "SUCCESS", 2: "SUCCESS"}
    assert "api_failures" not in logs[1]


def test_a_deferred_solve_with_api_failures_stays_pending(tmp_path):
    stopped = ApiFailurePipeline(fail_calls={1})
    logs = _run(tmp_path, stopped, defer=True)
    # Phase 1 stops at Q#1's failed call; phase 2 solves only Q#0.
    assert stopped.calls == [("intermediate", 0), ("intermediate", 1), ("solve_only", 0)]
    assert logs[1]["pipeline_status"] == orchestration.API_FAILURE_STATUS

    rerun = ApiFailurePipeline(fail_solve_calls={2})
    logs = _run(tmp_path, rerun, defer=True)
    # Q#1's intermediate step is redone; Q#2's solve call fails, so its
    # intermediate log stays pending for the next run.
    assert rerun.calls == [("intermediate", 1), ("intermediate", 2), ("solve_only", 1), ("solve_only", 2)]
    assert logs[2]["pipeline_status"] == "INTERMEDIATE_COMPLETE"
    assert logs[2]["api_failures"][0]["error_type"] == "ProactiveRateLimit"

    final = ApiFailurePipeline()
    logs = _run(tmp_path, final, defer=True)
    assert final.calls == [("solve_only", 2)]
    assert {index: log["pipeline_status"] for index, log in logs.items()} == {
        0: "SUCCESS", 1: "SUCCESS", 2: "SUCCESS"}
    assert "api_failures" not in logs[2]
