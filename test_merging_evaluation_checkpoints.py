"""Offline recovery checks for costly merging evaluation inference and Hub state."""

import copy
from fnmatch import fnmatchcase
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from requests import Response

from src.merging_evaluation_checkpoints import (
    MergingEvaluationCheckpoint,
    canonical_fingerprint,
    fingerprint_merging_adapter,
)
from src.merging_finetuning import run_merging_tree


def hub_error(error_type, status):
    response = Response()
    response.status_code = status
    response.url = "https://huggingface.co/datasets/researcher/evaluations"
    return error_type("fake Hub failure", response=response)


class FakeDatasetHub:
    """Keep immutable snapshots so a moving main branch cannot hide a bad restore."""

    def __init__(self, files=None):
        self.files = dict(files or {})
        self.exists = files is not None
        self.revision = "revision-0"
        self.snapshots = {self.revision: dict(self.files)}
        self.creates = []
        self.uploads = []
        self.infos = []
        self.downloads = []
        self.upload_error = None
        self.info_error = None

    def whoami(self):
        return {"name": "researcher"}

    def create_repo(self, **kwargs):
        self.creates.append(kwargs)
        self.exists = True
        return SimpleNamespace(repo_id=kwargs["repo_id"])

    def repo_info(self, **kwargs):
        self.infos.append(kwargs)
        if self.info_error is not None:
            raise self.info_error
        if not self.exists:
            raise hub_error(RepositoryNotFoundError, 404)
        return SimpleNamespace(
            sha=self.revision,
            siblings=[SimpleNamespace(rfilename=name) for name in self.files],
        )

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)
        if self.upload_error is not None:
            raise self.upload_error
        folder = Path(kwargs["folder_path"])
        prefix = kwargs["path_in_repo"].rstrip("/")
        for path in folder.glob("*.json"):
            if not any(fnmatchcase(path.name, pattern) for pattern in kwargs["allow_patterns"]):
                continue
            self.files[f"{prefix}/{path.name}"] = path.read_bytes()
        self.revision = f"revision-{len(self.uploads)}"
        self.snapshots[self.revision] = dict(self.files)
        return SimpleNamespace(oid=self.revision)

    def download(self, **kwargs):
        self.downloads.append(kwargs)
        snapshot = self.snapshots[kwargs["revision"]]
        patterns = kwargs["allow_patterns"]
        for name, data in snapshot.items():
            if not any(fnmatchcase(name, pattern) for pattern in patterns):
                continue
            destination = Path(kwargs["local_dir"]) / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        return str(kwargs["local_dir"])

    def mutate_snapshot(self, name, payload):
        self.files[name] = json.dumps(payload).encode("utf-8")
        self.snapshots[self.revision] = dict(self.files)


class MergingEvaluationCheckpointTests(TestCase):
    identity = {
        "benchmark": "math500", "questions": [{"index": 0, "question": "Find x."}],
        "adapter": "weights-v1", "generation": {"seed": 42, "max_new_tokens": 100},
    }

    def manager(self, directory, **kwargs):
        return MergingEvaluationCheckpoint(directory, copy.deepcopy(self.identity), **kwargs)

    def run_result(self, phase, **kwargs):
        return {"benchmark_index": 0, "benchmark": "math500", "phase": phase,
                "status": "SUCCESS", **kwargs}

    def call(self, wrapped, prompt="question", **overrides):
        options = {
            "use_adapter": False, "seed": 42, "temperature": 0.7,
            "top_p": 0.8, "max_new_tokens": 100,
        }
        options.update(overrides)
        return wrapped(prompt, **options)

    def test_partial_tree_resume_reuses_success_before_interruption(self):
        with TemporaryDirectory() as directory:
            manager = self.manager(directory)
            state = manager.load_question(0)
            calls = []

            def interrupted(prompt, **kwargs):
                calls.append(kwargs["seed"])
                if len(calls) == 2:
                    raise RuntimeError("runtime interrupted")
                return {"status": "SUCCESS", "text": f"solution-{kwargs['seed']}"}

            with self.assertRaisesRegex(RuntimeError, "runtime interrupted"):
                run_merging_tree(
                    "Find x.", manager.wrap_generator(interrupted, state),
                    "zero_shot", zero_shot_n=2,
                )

            resumed = self.manager(directory)
            saved = resumed.load_question(0)
            fresh_calls = []

            def finish(prompt, **kwargs):
                fresh_calls.append(kwargs["seed"])
                return {"status": "SUCCESS", "text": f"solution-{kwargs['seed']}"}

            result = run_merging_tree(
                "Find x.", resumed.wrap_generator(finish, saved),
                "zero_shot", zero_shot_n=2,
            )
            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(fresh_calls, [43, 142])
            self.assertEqual(result["trace"][0]["text"], "solution-42")
            self.assertEqual(len(saved["generations"]), 3)

    def test_generator_cache_separates_prompt_adapter_and_generation_settings(self):
        with TemporaryDirectory() as directory:
            manager = self.manager(directory)
            state = manager.load_question(0)
            calls = []

            def generate(prompt, **kwargs):
                calls.append((prompt, kwargs))
                return {"status": "SUCCESS", "text": str(len(calls))}

            wrapped = manager.wrap_generator(generate, state)
            first = self.call(wrapped)
            self.assertEqual(self.call(wrapped), first)
            self.call(wrapped, prompt="other question")
            self.call(wrapped, use_adapter=True)
            self.call(wrapped, max_new_tokens=200)
            self.call(wrapped, seed=43)
            self.assertEqual(len(calls), 5)
            self.assertEqual(len(state["generations"]), 5)

    def test_failed_generations_are_retried(self):
        with TemporaryDirectory() as directory:
            manager = self.manager(directory)
            state = manager.load_question(0)
            responses = iter([
                {"status": "FAILED", "text": ""},
                {"status": "SUCCESS", "text": "recovered"},
            ])
            wrapped = manager.wrap_generator(lambda *args, **kwargs: next(responses), state)
            self.assertEqual(self.call(wrapped)["status"], "FAILED")
            self.assertEqual(self.call(wrapped)["text"], "recovered")
            self.assertEqual(self.call(wrapped)["text"], "recovered")
            self.assertEqual(len(state["generations"]), 1)

    def test_completed_question_recovers_judgments_retrieval_and_phase_runs(self):
        with TemporaryDirectory() as directory:
            manager = self.manager(directory)
            state = manager.load_question(0)
            state["judgments"][canonical_fingerprint({"answer": "answer-id"})] = {
                "status": "SUCCESS", "judgment": "CORRECT", "score": 1,
            }
            state["retrieved"] = [{"index": 7, "question": "Example", "solution": "Proof"}]
            phase_1 = [self.run_result(1, setting="zero_shot", base={"status": "SUCCESS"})]
            phase_2 = [self.run_result(2, setting="fusion", adapted={"status": "SUCCESS"})]
            manager.complete_question(state, phase_1, phase_2)
            restored = self.manager(directory).load_question(0)
            self.assertTrue(restored["completed"])
            self.assertEqual(restored["benchmark_index"], 0)
            self.assertEqual(restored["phase_1_runs"], phase_1)
            self.assertEqual(restored["phase_2_runs"], phase_2)
            self.assertEqual(restored["judgments"], state["judgments"])
            self.assertEqual(restored["retrieved"], state["retrieved"])

    def test_different_benchmark_adapter_or_generation_identity_cannot_reuse_state(self):
        with TemporaryDirectory() as directory:
            manager = self.manager(directory)
            manager.complete_question(manager.load_question(0), [self.run_result(1)], [self.run_result(2)])
            for key, value in (("benchmark", "gsm8k"), ("adapter", "weights-v2"),
                               ("generation", {"seed": 43})):
                with self.subTest(key=key):
                    changed = copy.deepcopy(self.identity)
                    changed[key] = value
                    separate = MergingEvaluationCheckpoint(directory, changed)
                    self.assertNotEqual(separate.fingerprint, manager.fingerprint)
                    self.assertFalse(separate.load_question(0)["completed"])

    def test_upload_and_clean_restore_use_dataset_repo_scoped_path_and_pinned_revision(self):
        hub = FakeDatasetHub()
        with TemporaryDirectory() as directory:
            first = self.manager(
                Path(directory) / "first", token="fake-secret", repo_id="researcher/evaluations",
                upload_enabled=True, upload_every=10, api=hub,
            )
            first.complete_question(
                first.load_question(0), [self.run_result(1, accuracy=1)], [self.run_result(2, accuracy=0)],
            )
            pinned_revision = hub.revision
            prefix = first.remote_prefix
            hub.files["unrelated/private.json"] = b"{}"
            hub.snapshots[pinned_revision] = dict(hub.files)
            restored = self.manager(
                Path(directory) / "clean", token="fake-secret", repo_id="researcher/evaluations",
                restore_enabled=True, api=hub, download_fn=hub.download,
            )
            self.assertTrue(restored.load_question(0)["completed"])
            self.assertEqual(hub.creates[0]["repo_type"], "dataset")
            self.assertTrue(hub.creates[0]["private"])
            self.assertTrue(hub.creates[0]["exist_ok"])
            self.assertEqual(hub.uploads[0]["repo_type"], "dataset")
            self.assertEqual(hub.uploads[0]["path_in_repo"], prefix)
            self.assertIn("/math500/", prefix)
            self.assertIn(first.fingerprint, prefix)
            self.assertEqual(hub.downloads[0]["repo_type"], "dataset")
            self.assertEqual(hub.downloads[0]["revision"], pinned_revision)
            self.assertEqual(hub.downloads[0]["allow_patterns"], [f"{prefix}/*.json"])
            self.assertEqual(hub.downloads[0]["token"], "fake-secret")
            self.assertFalse((Path(directory) / "clean" / "unrelated" / "private.json").exists())
            manifest_name = f"{prefix}/manifest.json"
            self.assertEqual(json.loads(hub.files[manifest_name])["identity"], self.identity)
            self.assertNotIn("fake-secret", hub.files[manifest_name].decode("utf-8"))

    def test_restore_merges_remote_without_discarding_newer_local_generations(self):
        hub = FakeDatasetHub()
        with TemporaryDirectory() as directory:
            manager = self.manager(
                directory, repo_id="researcher/evaluations", upload_enabled=True,
                upload_every=100, api=hub,
            )
            state = manager.load_question(0)
            generator = manager.wrap_generator(
                lambda prompt, **kwargs: {"status": "SUCCESS", "text": prompt}, state,
            )
            self.call(generator, prompt="remote saved answer")
            manager.sync()
            self.call(generator, prompt="newer local answer")
            restored = self.manager(
                directory, repo_id="researcher/evaluations", restore_enabled=True,
                api=hub, download_fn=hub.download,
            )
            merged = restored.load_question(0)
            wrapped = restored.wrap_generator(
                lambda *args, **kwargs: self.fail("successful inference was repeated"), merged,
            )
            self.assertEqual(self.call(wrapped, prompt="remote saved answer")["text"], "remote saved answer")
            self.assertEqual(self.call(wrapped, prompt="newer local answer")["text"], "newer local answer")
            self.assertEqual(len(merged["generations"]), 2)

    def test_remote_identity_or_state_tampering_fails_closed(self):
        for target in ("manifest", "question", "empty_completed_row", "wrong_run_benchmark", "wrong_run_index"):
            with self.subTest(target=target), TemporaryDirectory() as directory:
                hub = FakeDatasetHub()
                manager = self.manager(
                    Path(directory) / "first", repo_id="researcher/evaluations",
                    upload_enabled=True, api=hub,
                )
                manager.complete_question(manager.load_question(0), [self.run_result(1)], [self.run_result(2)])
                name = f"{manager.remote_prefix}/" + (
                    "manifest.json" if target == "manifest" else "question-00000000.json"
                )
                payload = json.loads(hub.files[name])
                if target == "manifest":
                    payload["identity"]["adapter"] = "other weights"
                elif target == "question":
                    payload["benchmark_index"] = "not an integer"
                elif target == "empty_completed_row":
                    payload["phase_1_runs"] = []
                    payload["phase_2_runs"] = []
                elif target == "wrong_run_benchmark":
                    payload["phase_1_runs"][0]["benchmark"] = "gsm8k"
                else:
                    payload["phase_1_runs"][0]["benchmark_index"] = 17
                hub.mutate_snapshot(name, payload)
                with self.assertRaises((ValueError, RuntimeError)):
                    restored = self.manager(
                        Path(directory) / "clean", repo_id="researcher/evaluations",
                        restore_enabled=True, api=hub, download_fn=hub.download,
                    )
                    restored.load_question(0)

    def test_download_missing_an_advertised_question_fails_instead_of_restarting_it(self):
        with TemporaryDirectory() as directory:
            hub = FakeDatasetHub()
            manager = self.manager(
                Path(directory) / "first", repo_id="researcher/evaluations",
                upload_enabled=True, api=hub,
            )
            manager.complete_question(manager.load_question(0), [self.run_result(1)], [self.run_result(2)])

            def incomplete_download(**kwargs):
                result = hub.download(**kwargs)
                (Path(kwargs["local_dir"]) / manager.remote_prefix / "question-00000000.json").unlink()
                return result

            with self.assertRaises((ValueError, RuntimeError)):
                self.manager(
                    Path(directory) / "clean", repo_id="researcher/evaluations",
                    restore_enabled=True, api=hub, download_fn=incomplete_download,
                )

    def test_upload_network_or_auth_failure_preserves_success_and_stops(self):
        for error in (OSError("network unavailable"), hub_error(HfHubHTTPError, 401)):
            with self.subTest(error=type(error).__name__), TemporaryDirectory() as directory:
                hub = FakeDatasetHub()
                manager = self.manager(
                    directory, repo_id="researcher/evaluations", upload_enabled=True,
                    upload_every=1, api=hub,
                )
                hub.upload_error = error
                state = manager.load_question(0)
                wrapped = manager.wrap_generator(
                    lambda *args, **kwargs: {"status": "SUCCESS", "text": "costly answer"}, state,
                )
                with self.assertRaises((OSError, RuntimeError, HfHubHTTPError)):
                    self.call(wrapped)
                local = self.manager(directory)
                restored = local.load_question(0)
                self.assertEqual(len(restored["generations"]), 1)
                no_repeat = local.wrap_generator(
                    lambda *args, **kwargs: self.fail("persisted answer was regenerated"), restored,
                )
                self.assertEqual(self.call(no_repeat)["text"], "costly answer")

    def test_missing_remote_repo_or_matching_path_starts_fresh(self):
        for files in (None, {"other/run/manifest.json": b"{}"}):
            with self.subTest(files=files), TemporaryDirectory() as directory:
                hub = FakeDatasetHub(files)
                manager = self.manager(
                    directory, repo_id="researcher/evaluations", restore_enabled=True,
                    upload_enabled=True,
                    api=hub, download_fn=hub.download,
                )
                self.assertFalse(manager.load_question(0)["completed"])
                self.assertEqual(hub.downloads, [])

    def test_restore_auth_error_is_not_treated_as_missing_repo(self):
        with TemporaryDirectory() as directory:
            hub = FakeDatasetHub()
            for error_type in (HfHubHTTPError, RepositoryNotFoundError):
                with self.subTest(error_type=error_type.__name__):
                    hub.info_error = hub_error(error_type, 401)
                    with self.assertRaises((HfHubHTTPError, RuntimeError)):
                        self.manager(
                            directory, repo_id="researcher/evaluations", restore_enabled=True,
                            api=hub, download_fn=hub.download,
                        )

    def test_explicit_restore_only_missing_repo_stops_instead_of_repeating_inference(self):
        with TemporaryDirectory() as directory:
            hub = FakeDatasetHub()
            with self.assertRaises((HfHubHTTPError, RuntimeError)):
                self.manager(
                    directory, repo_id="researcher/evaluations", restore_enabled=True,
                    api=hub, download_fn=hub.download,
                )

    def test_upload_authentication_is_checked_before_inference_starts(self):
        with TemporaryDirectory() as directory:
            hub = FakeDatasetHub()
            hub.upload_error = hub_error(HfHubHTTPError, 401)
            with self.assertRaises((HfHubHTTPError, RuntimeError)):
                self.manager(directory, upload_enabled=True, api=hub)
            self.assertEqual(len(hub.uploads), 1)
            self.assertFalse(any(Path(directory).rglob("question-*.json")))

    def test_remote_completed_row_replaces_local_incomplete_phase_and_failed_judgment(self):
        with TemporaryDirectory() as directory:
            hub = FakeDatasetHub()
            key = canonical_fingerprint({"answer": "same answer"})
            remote = self.manager(
                Path(directory) / "remote", repo_id="researcher/evaluations",
                upload_enabled=True, api=hub,
            )
            remote_state = remote.load_question(0)
            remote_state["judgments"][key] = {"status": "SUCCESS", "judgment": "CORRECT"}
            phases = ([self.run_result(1)], [self.run_result(2)])
            remote.complete_question(remote_state, *phases)

            local_path = Path(directory) / "local"
            local = self.manager(local_path)
            local_state = local.load_question(0)
            local_state["judgments"][key] = {"status": "FAILED", "error": "temporary failure"}
            local_state["phase_1_runs"] = [self.run_result(1, status="INCOMPLETE")]
            local.save_question(local_state)

            merged = self.manager(
                local_path, repo_id="researcher/evaluations", restore_enabled=True,
                api=hub, download_fn=hub.download,
            ).load_question(0)
            self.assertTrue(merged["completed"])
            self.assertEqual(merged["judgments"][key], remote_state["judgments"][key])
            self.assertEqual(merged["phase_1_runs"], phases[0])
            self.assertEqual(merged["phase_2_runs"], phases[1])

    def test_conflicting_successful_local_and_remote_output_fails_closed(self):
        with TemporaryDirectory() as directory:
            hub = FakeDatasetHub()
            manager = self.manager(
                directory, repo_id="researcher/evaluations", upload_enabled=True, api=hub,
            )
            state = manager.load_question(0)
            self.call(manager.wrap_generator(
                lambda *args, **kwargs: {"status": "SUCCESS", "text": "original answer"}, state,
            ))
            manager.sync()
            name = f"{manager.remote_prefix}/question-00000000.json"
            payload = json.loads(hub.files[name])
            key = next(iter(payload["generations"]))
            payload["generations"][key]["text"] = "conflicting successful answer"
            hub.mutate_snapshot(name, payload)
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                self.manager(
                    directory, repo_id="researcher/evaluations", restore_enabled=True,
                    api=hub, download_fn=hub.download,
                )

    def test_periodic_and_final_sync_save_progress_before_question_completion(self):
        with TemporaryDirectory() as directory:
            hub = FakeDatasetHub()
            manager = self.manager(
                directory, repo_id="researcher/evaluations", upload_enabled=True,
                upload_every=2, api=hub,
            )
            initial_uploads = len(hub.uploads)
            state = manager.load_question(0)
            wrapped = manager.wrap_generator(
                lambda prompt, **kwargs: {"status": "SUCCESS", "text": prompt}, state,
            )
            self.call(wrapped, prompt="one")
            self.assertEqual(len(hub.uploads), initial_uploads)
            self.call(wrapped, prompt="two")
            self.assertEqual(len(hub.uploads), initial_uploads + 1)
            self.call(wrapped, prompt="three")
            manager.sync()
            self.assertEqual(len(hub.uploads), initial_uploads + 2)
            saved = json.loads(hub.files[f"{manager.remote_prefix}/question-00000000.json"])
            self.assertFalse(saved["completed"])
            self.assertEqual(len(saved["generations"]), 3)


class MergingEvaluationFingerprintTests(TestCase):
    def test_canonical_identity_does_not_depend_on_mapping_order(self):
        self.assertEqual(
            canonical_fingerprint({"benchmark": "math500", "settings": {"b": 2, "a": 1}}),
            canonical_fingerprint({"settings": {"a": 1, "b": 2}, "benchmark": "math500"}),
        )

    def test_adapter_fingerprint_uses_content_and_detects_weights_changes(self):
        with TemporaryDirectory() as directory:
            first = Path(directory) / "local-training"
            second = Path(directory) / "downloaded-from-hub" / "revision"
            for path in (first, second):
                path.mkdir(parents=True)
                (path / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
                (path / "adapter_model.safetensors").write_bytes(b"test weights v1")
                (path / "tokenizer_config.json").write_text('{"padding_side": "left"}', encoding="utf-8")
                (path / "tokenizer.json").write_text('{"vocab": {"hello": 1}}', encoding="utf-8")
            original = fingerprint_merging_adapter(first)
            self.assertEqual(original, fingerprint_merging_adapter(second))
            (second / "adapter_model.safetensors").write_bytes(b"test weights v2")
            self.assertNotEqual(original, fingerprint_merging_adapter(second))

    def test_adapter_chat_templates_have_the_same_identity_after_download_and_detect_edits(self):
        with TemporaryDirectory() as directory:
            first = Path(directory) / "trained"
            downloaded = Path(directory) / "hub-cache" / "pinned-revision"
            for path in (first, downloaded):
                (path / "chat_templates").mkdir(parents=True)
                (path / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
                (path / "adapter_model.safetensors").write_bytes(b"identical weights")
                (path / "chat_template.jinja").write_text("{{ messages[0]['content'] }}", encoding="utf-8")
                (path / "chat_templates" / "default.jinja").write_text("{{ messages }}", encoding="utf-8")
            original = fingerprint_merging_adapter(first)
            self.assertEqual(original, fingerprint_merging_adapter(downloaded))
            for relative in (Path("chat_template.jinja"), Path("chat_templates") / "default.jinja"):
                with self.subTest(template=str(relative)):
                    template = downloaded / relative
                    contents = template.read_text(encoding="utf-8")
                    template.write_text(contents + " extra model instruction", encoding="utf-8")
                    self.assertNotEqual(original, fingerprint_merging_adapter(downloaded))
                    template.write_text(contents, encoding="utf-8")
                    self.assertEqual(original, fingerprint_merging_adapter(downloaded))
