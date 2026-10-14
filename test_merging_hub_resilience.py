"""Transient Hub errors are retried; silent restarts from zero are explained."""

import copy
import functools
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import requests
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from src.hub_resilience import call_with_retries, is_transient_hub_error
from src.merging_evaluation_checkpoints import (
    MergingEvaluationCheckpoint, differing_identity_keys,
)
from src.merging_finetuning import download_adapter_from_hub, upload_adapter_to_hub
from test_merging_evaluation_checkpoints import FakeDatasetHub, hub_error


class FlakyDatasetHub(FakeDatasetHub):
    """A dataset hub whose next calls fail with the queued errors, then succeed."""

    def __init__(self, files=None):
        super().__init__(files)
        self.failures = {"upload": [], "info": [], "download": []}

    def upload_folder(self, **kwargs):
        if self.failures["upload"]:
            raise self.failures["upload"].pop(0)
        return super().upload_folder(**kwargs)

    def repo_info(self, **kwargs):
        if self.failures["info"]:
            raise self.failures["info"].pop(0)
        return super().repo_info(**kwargs)

    def download(self, **kwargs):
        if self.failures["download"]:
            raise self.failures["download"].pop(0)
        return super().download(**kwargs)


class TransientClassificationTests(TestCase):
    def test_rate_limits_server_errors_and_network_failures_are_transient(self):
        for status in (408, 409, 412, 425, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                self.assertTrue(is_transient_hub_error(hub_error(HfHubHTTPError, status)))
        for error in (ConnectionError("reset"), TimeoutError("slow"), requests.exceptions.ConnectionError("down"),
                      requests.exceptions.ReadTimeout("slow")):
            with self.subTest(error=type(error).__name__):
                self.assertTrue(is_transient_hub_error(error))

    def test_authentication_missing_repositories_and_local_errors_are_not(self):
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status):
                self.assertFalse(is_transient_hub_error(hub_error(HfHubHTTPError, status)))
        self.assertFalse(is_transient_hub_error(hub_error(RepositoryNotFoundError, 404)))
        for error in (OSError("disk full"), ValueError("bad"), RuntimeError("bug"), KeyError("x")):
            with self.subTest(error=type(error).__name__):
                self.assertFalse(is_transient_hub_error(error))


class RetryTests(TestCase):
    def test_success_needs_no_wait(self):
        sleeps = []
        self.assertEqual(call_with_retries(lambda: "done", "Doing it", sleep=sleeps.append), "done")
        self.assertEqual(sleeps, [])

    def test_transient_failures_wait_the_configured_backoff_then_succeed(self):
        errors = [hub_error(HfHubHTTPError, 503), hub_error(HfHubHTTPError, 429)]
        sleeps, messages = [], []

        def operation():
            if errors:
                raise errors.pop(0)
            return "recovered"

        self.assertEqual(call_with_retries(operation, "Uploading", delays=(1, 2, 3), sleep=sleeps.append,
                                           log=messages.append), "recovered")
        self.assertEqual(sleeps, [1.0, 2.0])
        self.assertIn("Uploading failed transiently", messages[0])
        self.assertIn("retry 1/3 in 1s", messages[0])

    def test_the_last_error_propagates_unchanged_when_retries_run_out(self):
        error = hub_error(HfHubHTTPError, 503)
        calls, sleeps = [], []

        def operation():
            calls.append(1)
            raise error

        with self.assertRaises(HfHubHTTPError) as caught:
            call_with_retries(operation, "Uploading", delays=(1, 2), sleep=sleeps.append, log=lambda m: None)
        self.assertIs(caught.exception, error)
        self.assertEqual((len(calls), sleeps), (3, [1.0, 2.0]))

    def test_non_transient_errors_and_interrupts_are_never_retried(self):
        for error in (hub_error(HfHubHTTPError, 401), ValueError("bug"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                calls = []

                def operation():
                    calls.append(1)
                    raise error

                with self.assertRaises(type(error)):
                    call_with_retries(operation, "Doing it", delays=(1, 2), sleep=lambda s: self.fail("waited"))
                self.assertEqual(len(calls), 1)


class EvaluationCheckpointResilienceTests(TestCase):
    identity = {"benchmark": "math500", "adapter": "weights-v1", "generation": {"seed": 42},
                "code_sha256": {"merging_finetuning.py": "aaa", "utils.py": "bbb"}}

    def make(self, directory, hub, identity=None, **kwargs):
        self.sleeps = getattr(self, "sleeps", [])
        options = dict(token="secret", repo_id="researcher/evaluations", upload_enabled=True,
                       restore_enabled=True, api=hub, download_fn=hub.download, upload_every=1,
                       retry_delays=(1.0, 2.0, 3.0), sleep=self.sleeps.append)
        options.update(kwargs)
        return MergingEvaluationCheckpoint(directory, copy.deepcopy(identity or self.identity), **options)

    def finish_questions(self, manager, count):
        for index in range(count):
            manager.complete_question(
                manager.load_question(index),
                [{"benchmark_index": index, "benchmark": "math500", "phase": 1, "status": "SUCCESS"}], [])

    def test_a_transient_upload_failure_is_retried_and_the_checkpoint_still_lands(self):
        hub = FlakyDatasetHub()
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()) as output:
            manager = self.make(directory, hub)
            hub.failures["upload"] = [hub_error(HfHubHTTPError, 503)]
            self.finish_questions(manager, 1)
            self.assertEqual(self.sleeps, [1.0])
            self.assertIn("question-00000000.json", "".join(hub.files))
            self.assertIn("Uploading the evaluation checkpoint failed transiently", output.getvalue())

    def test_persistent_failure_still_stops_the_run_with_progress_kept_locally(self):
        hub = FlakyDatasetHub()
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            manager = self.make(directory, hub)
            hub.failures["upload"] = [hub_error(HfHubHTTPError, 503)] * 4
            with self.assertRaises(HfHubHTTPError):
                self.finish_questions(manager, 1)
            self.assertEqual(self.sleeps, [1.0, 2.0, 3.0])
            restored = self.make(directory, FakeDatasetHub(), upload_enabled=False, restore_enabled=False)
            self.assertTrue(restored.load_question(0)["completed"], "local progress survives the failed upload")

    def test_authentication_failures_are_not_retried(self):
        hub = FlakyDatasetHub()
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            manager = self.make(directory, hub)
            hub.failures["upload"] = [hub_error(HfHubHTTPError, 401)]
            with self.assertRaises(HfHubHTTPError):
                self.finish_questions(manager, 1)
            self.assertEqual(self.sleeps, [])

    def test_restore_retries_reading_and_downloading_the_repository(self):
        hub = FlakyDatasetHub()
        with TemporaryDirectory() as first, TemporaryDirectory() as second, redirect_stdout(StringIO()):
            self.finish_questions(self.make(first, hub), 2)
            self.sleeps = []
            hub.failures["info"] = [hub_error(HfHubHTTPError, 502)]
            hub.failures["download"] = [requests.exceptions.ConnectionError("dropped")]
            restored = self.make(second, hub, upload_enabled=False)
            self.assertTrue(restored.load_question(1)["completed"])
            self.assertEqual(self.sleeps, [1.0, 1.0])

    def test_a_missing_repository_is_still_a_fresh_start_not_a_retry(self):
        hub = FlakyDatasetHub()
        self.assertFalse(hub.exists)
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()) as output:
            self.make(directory, hub)
            self.assertEqual(self.sleeps, [])
            self.assertIn("starting a fresh run", output.getvalue())


class OtherIdentityDiagnosticTests(TestCase):
    base = EvaluationCheckpointResilienceTests.identity

    def checkpoint(self, directory, hub, identity, **kwargs):
        options = dict(token="secret", repo_id="researcher/evaluations", upload_enabled=True,
                       restore_enabled=True, api=hub, download_fn=hub.download, retry_delays=(), sleep=lambda s: None)
        options.update(kwargs)
        return MergingEvaluationCheckpoint(directory, copy.deepcopy(identity), **options)

    def saved_run(self, hub, directory, identity, questions=3):
        manager = self.checkpoint(directory, hub, identity)
        for index in range(questions):
            manager.complete_question(
                manager.load_question(index),
                [{"benchmark_index": index, "benchmark": identity["benchmark"], "phase": 1, "status": "SUCCESS"}], [])
        return manager

    def test_a_changed_source_file_is_named_instead_of_silently_restarting(self):
        hub = FakeDatasetHub()
        changed = copy.deepcopy(self.base)
        changed["code_sha256"]["merging_finetuning.py"] = "edited"
        changed["adapter"] = "weights-v2"
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            with redirect_stdout(StringIO()):
                old = self.saved_run(hub, first, self.base)
            output = StringIO()
            with redirect_stdout(output):
                new = self.checkpoint(second, hub, changed)
            text = output.getvalue()
            self.assertFalse(new.load_question(0)["completed"])
            self.assertIn("No matching Hugging Face evaluation checkpoint exists", text)
            self.assertIn("1 other identity; they are not reused", text)
            self.assertIn(f"{old.fingerprint[:12]}: 3 question files", text)
            self.assertIn("adapter", text)
            self.assertIn("code_sha256.merging_finetuning.py", text)
            self.assertNotIn("code_sha256.utils.py", text, "unchanged keys are not blamed")
            self.assertNotIn("secret", text)

    def test_the_matching_identity_is_not_reported_as_another_one(self):
        hub = FakeDatasetHub()
        with TemporaryDirectory() as first, TemporaryDirectory() as second, redirect_stdout(StringIO()):
            self.saved_run(hub, first, self.base)
        output = StringIO()
        with TemporaryDirectory() as second, redirect_stdout(output):
            restored = self.checkpoint(second, hub, self.base)
        self.assertTrue(restored.load_question(2)["completed"])
        self.assertNotIn("other identit", output.getvalue())

    def test_a_brand_new_benchmark_prints_nothing_extra(self):
        hub = FakeDatasetHub()
        other_benchmark = {**self.base, "benchmark": "gsm8k"}
        with TemporaryDirectory() as first, TemporaryDirectory() as second, redirect_stdout(StringIO()):
            self.saved_run(hub, first, other_benchmark)
        output = StringIO()
        with TemporaryDirectory() as second, redirect_stdout(output):
            self.checkpoint(second, hub, self.base)
        self.assertNotIn("other identit", output.getvalue())

    def test_an_unreadable_manifest_is_reported_without_stopping_the_run(self):
        hub = FakeDatasetHub()
        changed = {**self.base, "adapter": "weights-v2"}
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            with redirect_stdout(StringIO()):
                old = self.saved_run(hub, first, self.base)
            hub.files[f"{old.remote_prefix}/manifest.json"] = b"not json"
            hub.snapshots[hub.revision] = dict(hub.files)
            output = StringIO()
            with redirect_stdout(output):
                new = self.checkpoint(second, hub, changed)
            self.assertIn("Could not inspect other saved identities", output.getvalue())
            self.assertFalse(new.load_question(0)["completed"])

    def test_identity_difference_names_nested_added_and_removed_keys(self):
        self.assertEqual(differing_identity_keys({"a": 1, "b": {"c": 2, "d": 3}}, {"a": 1, "b": {"c": 2, "d": 4}}),
                         ["b.d"])
        self.assertEqual(differing_identity_keys({"a": 1}, {"a": 1, "z": 9}), ["z"])
        self.assertEqual(differing_identity_keys({"a": 1}, {"a": 1}), [])
        self.assertEqual(differing_identity_keys([1], [2]), ["<identity>"])


class AdapterRetryTests(TestCase):
    def patched_retries(self, sleeps):
        return patch("src.merging_finetuning.call_with_retries",
                     functools.partial(call_with_retries, delays=(1.0, 2.0), sleep=sleeps.append, log=lambda m: None))

    def test_adapter_download_retries_transient_hub_errors_but_not_a_missing_repository(self):
        files = ["adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"]

        class Hub:
            def __init__(self):
                self.failures = [hub_error(HfHubHTTPError, 503)]

            def whoami(self):
                return {"name": "researcher"}

            def model_info(self, repo_id):
                if self.failures:
                    raise self.failures.pop(0)
                return SimpleNamespace(sha="rev", siblings=[SimpleNamespace(rfilename=n) for n in files])

        downloads = []

        def download(**kwargs):
            downloads.append(1)
            if len(downloads) == 1:
                raise requests.exceptions.ReadTimeout("slow")
            for name in files:
                (Path(kwargs["local_dir"]) / name).write_text("x")

        sleeps = []
        with TemporaryDirectory() as directory, self.patched_retries(sleeps):
            result = download_adapter_from_hub(Path(directory) / "adapter", "t", api=Hub(), download_fn=download)
        self.assertEqual(result["revision"], "rev")
        self.assertEqual(sleeps, [1.0, 1.0])

        class Missing(Hub):
            def model_info(self, repo_id):
                raise hub_error(RepositoryNotFoundError, 404)

        sleeps.clear()
        with TemporaryDirectory() as directory, self.patched_retries(sleeps):
            self.assertIsNone(download_adapter_from_hub(Path(directory) / "adapter", "t", api=Missing(),
                                                        download_fn=lambda **k: self.fail("downloaded")))
        self.assertEqual(sleeps, [])

    def test_adapter_upload_retries_transient_hub_errors(self):
        class Hub:
            def __init__(self):
                self.failures = [hub_error(HfHubHTTPError, 429)]
                self.uploads = 0

            def whoami(self):
                return {"name": "researcher"}

            def create_repo(self, **kwargs):
                pass

            def upload_folder(self, **kwargs):
                if self.failures:
                    raise self.failures.pop(0)
                self.uploads += 1

        sleeps, hub = [], Hub()
        with TemporaryDirectory() as directory, self.patched_retries(sleeps):
            (Path(directory) / "adapter_config.json").write_text("{}")
            (Path(directory) / "adapter_model.safetensors").write_bytes(b"w")
            result = upload_adapter_to_hub(directory, "t", api=hub)
        self.assertEqual((hub.uploads, sleeps), (1, [1.0]))
        self.assertEqual(result["repo_id"], "researcher/merging-qwen3-4b-qlora")
