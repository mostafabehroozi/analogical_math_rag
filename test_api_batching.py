"""Offline checks for provider batch coalescing and its retry contract."""

import threading
import time
import unittest
from unittest.mock import patch

from src import api_batching, api_manager
from src.api_batching import APIBatchDispatcher, BatchSettings
from src.distributed_execution import sanitize_scientific_config


def settings(**overrides):
    values = dict(max_size=4, collect_seconds=0.3, max_concurrent_batches=1,
                  poll_seconds=0.01, timeout_seconds=5.0)
    values.update(overrides)
    return BatchSettings(**values)


def call_concurrently(dispatcher, payloads, batch_settings, group_key="model"):
    results = [None] * len(payloads)

    def run(index):
        results[index] = dispatcher.call(payloads[index], group_key, batch_settings)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(len(payloads))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    return results


class DispatcherTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(api_batching, "tprint")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_concurrent_calls_share_one_batch_and_keep_their_own_responses(self):
        submitted = []

        def submit(items, _settings):
            submitted.append([item.payload["prompt"] for item in items])
            return [{"status": "SUCCESS", "text": item.payload["prompt"].upper()} for item in items]

        dispatcher = APIBatchDispatcher("test", submit)
        results = call_concurrently(dispatcher, [{"prompt": p} for p in "abc"], settings())
        self.assertEqual([r["text"] for r in results], ["A", "B", "C"])
        self.assertEqual(len(submitted), 1)
        self.assertEqual(sorted(submitted[0]), ["a", "b", "c"])
        self.assertTrue(all(r["request_meta"]["api_batch_size"] == 3 for r in results))

    def test_max_size_splits_and_groups_never_mix(self):
        submitted = []

        def submit(items, _settings):
            submitted.append({item.group_key[0] for item in items} | {len(items)})
            return [{"status": "SUCCESS", "text": "ok"} for _ in items]

        dispatcher = APIBatchDispatcher("test", submit)
        batch_settings = settings(max_size=2, max_concurrent_batches=2)
        threads = [
            threading.Thread(target=dispatcher.call, args=({"prompt": str(i)}, f"model-{i % 2}", batch_settings))
            for i in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(sorted(map(sorted, map(lambda s: [str(x) for x in s], submitted))),
                         [["2", "model-0"], ["2", "model-1"]])

    def test_batch_failure_fails_every_item_without_hanging(self):
        def submit(items, _settings):
            raise ConnectionError("down")

        results = call_concurrently(APIBatchDispatcher("test", submit), [{"prompt": "a"}, {"prompt": "b"}], settings())
        self.assertEqual({r["error_type"] for r in results}, {"BatchError"})
        self.assertIn("down", results[0]["error_message"])

    def test_busy_slot_accumulates_the_next_batch(self):
        release = threading.Event()
        sizes = []

        def submit(items, _settings):
            sizes.append(len(items))
            if len(sizes) == 1:
                release.wait(5)
            return [{"status": "SUCCESS", "text": "ok"} for _ in items]

        dispatcher = APIBatchDispatcher("test", submit)
        batch_settings = settings(collect_seconds=0.0, max_size=10)
        first = threading.Thread(target=dispatcher.call, args=({"prompt": "first"}, "m", batch_settings))
        first.start()
        time.sleep(0.2)
        later = [threading.Thread(target=dispatcher.call, args=({"prompt": str(i)}, "m", batch_settings)) for i in range(3)]
        for thread in later:
            thread.start()
        time.sleep(0.2)
        release.set()
        for thread in [first, *later]:
            thread.join(timeout=10)
        self.assertEqual(sizes, [1, 3])

    def test_settings_validation_and_runtime_classification(self):
        with self.assertRaises(ValueError):
            BatchSettings.from_config({"API_BATCH_MAX_SIZE": 0})
        config = {"API_PROVIDER_SOLVER": "gemini", "API_BATCHING_ENABLED": True,
                  "API_BATCH_MAX_SIZE": 7, "API_BATCH_KAGGLE_N_JOBS": 5,
                  "KAGGLE_MODEL_NAME_FINAL_SOLVER": "google/x"}
        self.assertEqual(sanitize_scientific_config(config), {"API_PROVIDER_SOLVER": "gemini"})
        active = sanitize_scientific_config({**config, "API_PROVIDER_SOLVER": "kaggle"})
        self.assertEqual(active["KAGGLE_MODEL_NAME_FINAL_SOLVER"], "google/x")
        self.assertNotIn("API_BATCHING_ENABLED", active)


class RetryThroughBatchTests(unittest.TestCase):
    """A failed batch item goes back through execute_with_retry on its own."""

    def setUp(self):
        for target in (api_batching, api_manager):
            patcher = patch.object(target, "tprint")
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_failed_item_is_retried_in_a_later_batch(self):
        attempts = {"a": 0, "b": 0}

        def submit(items, _settings):
            responses = []
            for item in items:
                prompt = item.payload["prompt"]
                attempts[prompt] += 1
                if prompt == "b" and attempts[prompt] == 1:
                    responses.append({"status": "ERROR", "text": None, "error_type": "APIConnectionError"})
                else:
                    responses.append({"status": "SUCCESS", "text": prompt})
            return responses

        dispatcher = APIBatchDispatcher("test", submit)
        config = {"ENABLE_API_RETRY": True, "MAX_API_RETRIES": 3, "API_RETRY_DELAY_SECONDS": 0,
                  "GLOBAL_CONSECUTIVE_ERROR_LIMIT": 0, "GLOBAL_PERIODIC_PAUSE_INTERVAL_MINUTES": 0}
        results = {}

        def run(prompt):
            results[prompt] = api_manager.execute_with_retry(
                config, "test", "m", prompt,
                lambda: dispatcher.call({"prompt": prompt}, "m", settings(collect_seconds=0.2)),
            )

        threads = [threading.Thread(target=run, args=(p,)) for p in "ab"]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(results["a"]["text"], "a")
        self.assertEqual(results["b"]["text"], "b")
        self.assertEqual(attempts, {"a": 1, "b": 2})


if __name__ == "__main__":
    unittest.main()
