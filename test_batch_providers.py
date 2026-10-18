"""Offline checks for the OpenRouter Batch API and Kaggle Benchmarks providers."""

import contextlib
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

from src import api_batching, api_manager

UNBATCHABLE = "meta-llama/llama-3.1-8b-instruct"
BATCH_CONFIG = {
    "API_BATCHING_ENABLED": True,
    "API_BATCH_MAX_SIZE": 10,
    "API_BATCH_COLLECT_SECONDS": 0.3,
    "API_BATCH_POLL_SECONDS": 0.01,
    "API_BATCH_TIMEOUT_SECONDS": 5,
    "ENABLE_API_RETRY": False,
    "GLOBAL_CONSECUTIVE_ERROR_LIMIT": 0,
    "GLOBAL_PERIODIC_PAUSE_INTERVAL_MINUTES": 0,
}


def concurrently(*calls):
    results = [None] * len(calls)

    def run(index):
        results[index] = calls[index]()

    threads = [threading.Thread(target=run, args=(index,)) for index in range(len(calls))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    return results


class _Quiet(unittest.TestCase):
    def setUp(self):
        for target, name in ((api_batching, "tprint"), (api_manager, "tprint"), (api_manager.provider_pacer, "pace")):
            patcher = patch.object(target, name)
            patcher.start()
            self.addCleanup(patcher.stop)


class OpenRouterBatchTests(_Quiet):
    def setUp(self):
        super().setUp()
        fake_openai = types.SimpleNamespace(OpenAI=lambda **kwargs: object(),
                                            RateLimitError=Exception, APIStatusError=Exception)
        patcher = patch.object(api_manager, "openai", fake_openai)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.config = {
            **BATCH_CONFIG,
            "OPENROUTER_PROVIDER_ROUTING": {"only": ["provider-a"], "allow_fallbacks": False},
            "OPENROUTER_REASONING_EFFORT": "low",
        }
        self.manager = api_manager.OpenRouterAPIManager("test-key", {"default": {"rpm": 1000}}, self.config)
        self.http_calls = []
        self.polls = 0

        def fake_http(manager, method, path, api_key, payload=None):
            self.http_calls.append((method, path, payload))
            if method == "POST":
                return 202, {"id": "batch-1", "status": "validating"}
            self.polls += 1
            if self.polls == 1:
                return 200, {"id": "batch-1", "status": "in_progress"}
            # Results arrive out of order; they are matched by custom_id.
            return 200, {"id": "batch-1", "status": "completed", "results": [
                {"custom_id": "req-1", "response": {"status_code": 400, "body": {"error": {"message": "bad"}}}},
                {"custom_id": "req-0", "response": {"status_code": 200, "body": {
                    "model": "author/model", "choices": [{"message": {"content": "first answer"}}]}}},
            ]}

        patcher = patch.object(api_manager.OpenRouterAPIManager, "_batch_http", fake_http)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_concurrent_calls_become_one_batch_with_per_item_results(self):
        bound = api_manager.bind_api_managers({"openrouter": self.manager}, self.config)["openrouter"]
        first, second = concurrently(
            lambda: bound.generate_content("p0", "author/model", temperature=0.0),
            # A short head start keeps custom_id order deterministic.
            lambda: (threading.Event().wait(0.1), bound.generate_content("p1", "author/model", temperature=1.0))[1],
        )
        self.assertEqual(first["status"], "SUCCESS")
        self.assertEqual(first["text"], "first answer")
        self.assertEqual(first["request_meta"]["openrouter_batch_id"], "batch-1")
        self.assertEqual(second["error_type"], "InvalidArgument")
        posts = [call for call in self.http_calls if call[0] == "POST"]
        self.assertEqual(len(posts), 1)
        request = posts[0][2]
        self.assertEqual(list(request)[-1], "requests")
        self.assertEqual(request["model"], "author/model")
        self.assertEqual(request["provider"], {"only": ["provider-a"], "allow_fallbacks": False})
        self.assertEqual(request["requests"][0]["body"], {
            "messages": [{"role": "user", "content": "p0"}], "temperature": 0.0,
            "reasoning": {"effort": "low"},
        })

    def test_routing_the_batch_api_rejects_fails_before_any_request_and_stops_the_run(self):
        config = {**self.config, "OPENROUTER_PROVIDER_ROUTING": {"order": ["provider-a"]}}
        mark = api_batching.batch_refusals.mark()
        result = self.manager.for_config(config).generate_content("p", "author/model")
        self.assertEqual(result["error_type"], "BatchRefused")
        self.assertIn("provider routing keys ['order']", result["error_message"])
        self.assertEqual(self.http_calls, [])
        with self.assertRaisesRegex(api_batching.BatchRefusedError, "route this model with only"):
            api_batching.batch_refusals.raise_if_refused(since=mark)

    def test_batching_off_keeps_direct_requests(self):
        config = {**self.config, "API_BATCHING_ENABLED": False}
        with patch.object(api_manager.OpenRouterAPIManager, "_submit_batch") as submit:
            with patch.object(self.manager.batcher, "call") as call:
                self.manager.for_config(config).generate_content("p", "author/model")
        call.assert_not_called()
        submit.assert_not_called()


def fake_kbench():
    """A minimal stand-in for the kaggle_benchmarks API surface the manager uses."""
    state = {"prompts": [], "chats": 0, "evaluate": [], "annotation": None}

    class FakeLLM:
        def prompt(self, message, **options):
            state["prompts"].append((message, options))
            if message == "fail":
                raise RuntimeError("RateLimitError: 429 quota")
            return f"answer:{message}"

    @contextlib.contextmanager
    def new(name, orphan=False):
        state["chats"] += 1
        assert orphan
        yield

    def task(name=None, store_task=True, store_run=True):
        assert not store_task and not store_run

        def decorate(func):
            state["annotation"] = func.__annotations__.get("return")

            def evaluate(llm, evaluation_data, n_jobs, timeout, on_failure, remove_run_files):
                state["evaluate"].append({"n_jobs": n_jobs, "rows": len(evaluation_data)})
                runs = []
                rows = evaluation_data.reset_index().rename(columns={"index": "_id"})
                for _, row in rows.iterrows():
                    try:
                        result, status, error = func(llm[0], row["prompt"], row["options"]), "success", None
                    except Exception as exc:  # The real library records a traceback.
                        result, status, error = None, "failed", f"Traceback ...\n{exc}"
                    runs.append(types.SimpleNamespace(param_id=row["_id"], status=status,
                                                      result=result, error_message=error))
                return list(reversed(runs))

            return types.SimpleNamespace(evaluate=evaluate)
        return decorate

    module = types.SimpleNamespace(
        kaggle=types.SimpleNamespace(is_configured=lambda: True, load_model=lambda name: FakeLLM()),
        llms={"google/gemini-2.5-flash": FakeLLM()},
        chats=types.SimpleNamespace(new=new),
        task=task,
    )
    return module, state


class KaggleBenchmarksTests(_Quiet):
    def setUp(self):
        super().setUp()
        self.kbench, self.state = fake_kbench()
        patcher = patch.dict(sys.modules, {"kaggle_benchmarks": self.kbench})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.config = {**BATCH_CONFIG, "API_BATCHING_ENABLED": False, "KAGGLE_REASONING_EFFORT": "low",
                       "KAGGLE_REASONING_EFFORT_EVALUATOR": "none", "API_BATCH_KAGGLE_N_JOBS": 3}
        self.manager = api_manager.KaggleBenchmarksAPIManager({"default": {"rpm": 1000}}, self.config)

    def test_direct_call_uses_an_isolated_chat_and_role_options(self):
        result = self.manager.generate_content("q", "google/gemini-2.5-flash", temperature=0.0, avalai_role="evaluator")
        self.assertEqual(result["text"], "answer:q")
        self.assertEqual(self.state["chats"], 1)
        self.assertEqual(self.state["prompts"], [("q", {"temperature": 0.0, "reasoning": "none"})])
        self.assertEqual(api_manager.model_name_for(self.manager, {"KAGGLE_MODEL_NAME_EVALUATOR": "m"}, "evaluator"), "m")

    def test_direct_failure_is_classified(self):
        result = self.manager.generate_content("fail", "other/model")
        self.assertEqual(result["error_type"], "RateLimitError")

    def test_batched_calls_run_one_evaluate_and_map_rows_back(self):
        bound = self.manager.for_config({**self.config, "API_BATCHING_ENABLED": True})
        results = concurrently(*[
            (lambda prompt=prompt: bound.generate_content(prompt, "google/gemini-2.5-flash", temperature=1.0))
            for prompt in ("a", "fail", "c")
        ])
        self.assertEqual(self.state["evaluate"], [{"n_jobs": 3, "rows": 3}])
        self.assertIs(self.state["annotation"], dict)
        self.assertEqual(results[0]["text"], "answer:a")
        self.assertEqual(results[1]["error_type"], "RateLimitError")
        self.assertEqual(results[2]["text"], "answer:c")
        self.assertTrue(all(("low" == options["reasoning"]) for _, options in self.state["prompts"]))

    def test_unconfigured_proxy_fails_at_setup(self):
        self.kbench.kaggle.is_configured = lambda: False
        with self.assertRaisesRegex(RuntimeError, "MODEL_PROXY_URL"):
            api_manager.KaggleBenchmarksAPIManager({}, {})


class SwitchTests(unittest.TestCase):
    def test_switch_accepts_bool_or_provider_names(self):
        enabled = api_batching.batching_enabled
        self.assertTrue(enabled({"API_BATCHING_ENABLED": True}, "openrouter"))
        self.assertFalse(enabled({}, "kaggle"))
        self.assertTrue(enabled({"API_BATCHING_ENABLED": ["kaggle"]}, "kaggle"))
        self.assertFalse(enabled({"API_BATCHING_ENABLED": ["kaggle"]}, "openrouter"))
        self.assertTrue(enabled({"API_BATCHING_ENABLED": "Kaggle"}, "kaggle"))

    def test_switch_accepts_per_provider_model_rules(self):
        enabled = api_batching.batching_enabled
        config = {"API_BATCHING_ENABLED": {"Kaggle": True, "openrouter": ["a/batchable"]}}
        self.assertTrue(enabled(config, "kaggle", "any/model"))
        self.assertTrue(enabled(config, "openrouter", "a/batchable"))
        self.assertFalse(enabled(config, "openrouter", "meta-llama/llama-3.1-8b-instruct"))
        self.assertTrue(enabled({"API_BATCHING_ENABLED": {"openrouter": "a/one"}}, "openrouter", "a/one"))
        self.assertFalse(enabled({"API_BATCHING_ENABLED": {"kaggle": True}}, "openrouter", "a/one"))
        with self.assertRaises(ValueError):
            enabled({"API_BATCHING_ENABLED": {"openrouter": 3}}, "openrouter", "a/one")

    def test_each_route_is_logged_once_while_the_switch_is_set(self):
        config = {"API_BATCHING_ENABLED": {"kaggle": True, "openrouter": False}}
        with patch.object(api_batching, "_announced_routes", set()), \
                patch.object(api_batching, "tprint") as log:
            for _ in range(3):
                api_batching.batching_enabled(config, "openrouter", "meta-llama/llama-3.1-8b-instruct")
                api_batching.batching_enabled(config, "kaggle", "google/x")
            api_batching.batching_enabled({}, "openrouter", "other/model")
        messages = [call.args[0] for call in log.call_args_list]
        self.assertEqual(len(messages), 2)
        self.assertIn("meta-llama/llama-3.1-8b-instruct: sent directly", messages[0])
        self.assertIn("google/x: batched", messages[1])


class OpenRouterRefusalTests(_Quiet):
    """A model OpenRouter refuses to batch stops the run, or goes direct when allowed."""

    def setUp(self):
        super().setUp()
        self.direct_calls = []

        def create(**kwargs):
            self.direct_calls.append(kwargs["model"])
            return types.SimpleNamespace(
                model=kwargs["model"], model_extra={},
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="direct answer"))],
            )

        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        fake_openai = types.SimpleNamespace(OpenAI=lambda **kwargs: client,
                                            RateLimitError=FakeError, APIStatusError=FakeError)
        patcher = patch.object(api_manager, "openai", fake_openai)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.creates = 0
        # The response OpenRouter gave on Kaggle on 2026-10-18.
        self.refusal = (400, f"Model '{UNBATCHABLE}' does not have a :batch endpoint")

        def fake_http(manager, method, path, api_key, payload=None):
            self.creates += 1
            code, message = self.refusal
            return code, {"error": {"message": message}}

        patcher = patch.object(api_manager.OpenRouterAPIManager, "_batch_http", fake_http)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.config = {**BATCH_CONFIG, "API_BATCH_COLLECT_SECONDS": 0.0,
                       "OPENROUTER_MODEL_ROUTING": {UNBATCHABLE: {"only": ["novita/fp8"]}}}
        self.manager = api_manager.OpenRouterAPIManager("test-key", {"default": {"rpm": 1000}}, self.config)

    def warnings(self):
        return [call.args[0] for call in api_manager.tprint.call_args_list
                if call.args[0].startswith("[API BATCH openrouter]")]

    def assert_run_stopped(self, mark):
        with self.assertRaises(api_batching.BatchRefusedError) as raised:
            api_batching.batch_refusals.raise_if_refused(since=mark)
        return str(raised.exception)

    def test_without_fallback_later_calls_fail_at_once_and_the_run_stops(self):
        mark = api_batching.batch_refusals.mark()
        with patch.object(api_manager.global_pause_manager, "record_result") as record_result:
            results = [self.manager.generate_content(prompt, UNBATCHABLE) for prompt in ("p", "q")]
        self.assertEqual([result["error_type"] for result in results], ["BatchRefused", "BatchRefused"])
        self.assertEqual(self.creates, 1)  # The second call sent no batch request.
        self.assertEqual(self.direct_calls, [])
        record_result.assert_not_called()  # A refusal never triggers the global error pause.
        # The call log truncates the message, so it is printed in full once.
        self.assertEqual(len(self.warnings()), 1)
        message = self.assert_run_stopped(mark)
        self.assertIn(message, self.warnings()[0])
        for expected in (
            f"OpenRouter cannot batch {UNBATCHABLE} with provider routing " + '{"only": ["novita/fp8"]}',
            "HTTP 400: Model 'meta-llama/llama-3.1-8b-instruct' does not have a :batch endpoint.",
            "no provider pin can help", "API_BATCHING_ENABLED=True",
            "{'kaggle': True, 'openrouter': False}", "https://openrouter.ai/models?variant=batch",
            "API_BATCH_FALLBACK_TO_DIRECT=True",
        ):
            self.assertIn(expected, message)

    def test_the_error_names_the_experiments_own_switch_value(self):
        batching = {"kaggle": True, "openrouter": [UNBATCHABLE]}
        mark = api_batching.batch_refusals.mark()
        self.manager.for_config({**self.config, "API_BATCHING_ENABLED": batching}).generate_content("p", UNBATCHABLE)
        self.assertIn(f"API_BATCHING_ENABLED={batching!r}", self.assert_run_stopped(mark))

    def test_404_points_at_the_provider_pin(self):
        self.refusal = (404, "No batch endpoint matches the requested providers")
        mark = api_batching.batch_refusals.mark()
        self.manager.generate_content("p", UNBATCHABLE)
        message = self.assert_run_stopped(mark)
        self.assertIn("HTTP 404", message)
        self.assertIn("provider.only ['novita/fp8'] matches none of them", message)
        self.assertIn("pin a provider that serves the model's batch endpoint", message)
        self.assertNotIn("no provider pin can help", message)

    def test_calls_queued_behind_the_refused_batch_send_no_request(self):
        bound = self.manager.for_config({**self.config, "API_BATCH_MAX_CONCURRENT_BATCHES": 1})
        posted, release = threading.Event(), threading.Event()

        def held_refusal(manager, method, path, api_key, payload=None):
            self.creates += 1
            posted.set()
            release.wait(5)
            return 400, {"error": {"message": "no :batch endpoint"}}

        results = {}
        with patch.object(api_manager.OpenRouterAPIManager, "_batch_http", held_refusal):
            threads = [threading.Thread(target=lambda prompt=prompt: results.update(
                {prompt: bound.generate_content(prompt, UNBATCHABLE)})) for prompt in ("p", "q")]
            threads[0].start()
            self.assertTrue(posted.wait(5))
            threads[1].start()  # Waits in the queue: the only batch slot is busy.
            deadline = time.monotonic() + 5
            while not bound.batcher._pending and time.monotonic() < deadline:
                time.sleep(0.01)
            release.set()
            for thread in threads:
                thread.join(timeout=10)
        self.assertEqual(self.creates, 1)
        self.assertEqual({result["error_type"] for result in results.values()}, {"BatchRefused"})

    def test_an_earlier_refusal_does_not_stop_a_run_that_sends_the_model_directly(self):
        self.manager.generate_content("p", UNBATCHABLE)
        mark = api_batching.batch_refusals.mark()
        fixed = self.manager.for_config({**self.config, "API_BATCHING_ENABLED": {"kaggle": True, "openrouter": False}})
        self.assertEqual(fixed.generate_content("q", UNBATCHABLE)["text"], "direct answer")
        api_batching.batch_refusals.raise_if_refused(since=mark)

    def test_fallback_sends_the_model_directly_and_remembers_it(self):
        bound = self.manager.for_config({**self.config, "API_BATCH_FALLBACK_TO_DIRECT": True})
        mark = api_batching.batch_refusals.mark()
        first = bound.generate_content("p", UNBATCHABLE)
        second = bound.generate_content("q", UNBATCHABLE)
        self.assertEqual((first["text"], second["text"]), ("direct answer", "direct answer"))
        self.assertEqual(self.creates, 1)  # The second call never tried a batch.
        self.assertEqual(len(self.direct_calls), 2)
        self.assertEqual(len(self.warnings()), 1)
        self.assertIn("API_BATCH_FALLBACK_TO_DIRECT is on", self.warnings()[0])
        api_batching.batch_refusals.raise_if_refused(since=mark)  # The run goes on.

    def test_fallback_turned_on_after_a_refusal_sends_directly(self):
        self.manager.generate_content("p", UNBATCHABLE)
        bound = self.manager.for_config({**self.config, "API_BATCH_FALLBACK_TO_DIRECT": True})
        mark = api_batching.batch_refusals.mark()
        self.assertEqual(bound.generate_content("q", UNBATCHABLE)["text"], "direct answer")
        self.assertEqual(self.creates, 1)
        api_batching.batch_refusals.raise_if_refused(since=mark)

    def test_model_left_out_of_the_switch_goes_direct(self):
        config = {**self.config, "API_BATCHING_ENABLED": {"kaggle": True, "openrouter": ["other/model"]}}
        result = self.manager.for_config(config).generate_content("p", UNBATCHABLE)
        self.assertEqual(result["text"], "direct answer")
        self.assertEqual(self.creates, 0)

    def test_fallback_also_covers_routing_the_batch_api_rejects(self):
        config = {**self.config, "API_BATCH_FALLBACK_TO_DIRECT": True,
                  "OPENROUTER_MODEL_ROUTING": {UNBATCHABLE: {"order": ["novita/fp8"]}}}
        result = self.manager.for_config(config).generate_content("p", UNBATCHABLE)
        self.assertEqual(result["text"], "direct answer")
        self.assertEqual(self.creates, 0)


class FakeError(Exception):
    status_code = 500


class BindingTests(unittest.TestCase):
    def test_every_manager_is_bound_to_experiment_options(self):
        class Manager(api_manager._ConfigBindable):
            config = {"AVALAI_REASONING_EFFORT": None}

        manager = Manager()
        bound = api_manager.bind_api_managers({"avalai": manager, "other": object()},
                                              {"AVALAI_REASONING_EFFORT": "low"})
        self.assertEqual(bound["avalai"].config["AVALAI_REASONING_EFFORT"], "low")
        self.assertIsNone(manager.config["AVALAI_REASONING_EFFORT"])


if __name__ == "__main__":
    unittest.main()
