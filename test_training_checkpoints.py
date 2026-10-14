"""Offline checks for uploading Trainer checkpoints to the Hub and resuming from them."""

import json
from dataclasses import replace
from fnmatch import fnmatchcase
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from requests import Response

from src.merging_finetuning import QLoRAConfig
from src.training_checkpoints import (
    HubTrainingCheckpoints, describe_hub_progress, fingerprint_tokenized_rows,
    merging_training_source, training_checkpoint_identity, verify_adapter_training_source,
)


def hub_error(error_type, status):
    response = Response()
    response.status_code = status
    response.url = "https://huggingface.co/datasets/researcher/training-state"
    return error_type("fake Hub failure", response=response)


class FakeTrainingHub:
    """A dataset repo with atomic commits and immutable snapshots, like the real Hub."""

    def __init__(self, exists=True):
        self.exists = exists
        self.files = {}
        self.revision = "revision-0"
        self.snapshots = {self.revision: {}}
        self.commits = []
        self.creates = []
        self.downloads = []
        self.commit_errors = []  # raised by successive create_commit calls, then success
        self.info_error = None
        self.adapters = {}  # model repo id -> file names; an absent repo raises RepositoryNotFoundError

    def whoami(self):
        return {"name": "researcher"}

    def model_info(self, repo_id):
        if repo_id not in self.adapters:
            raise hub_error(RepositoryNotFoundError, 404)
        return SimpleNamespace(sha="abcdef1234567",
                               siblings=[SimpleNamespace(rfilename=name) for name in self.adapters[repo_id]])

    def create_repo(self, **kwargs):
        self.creates.append(kwargs)
        self.exists = True

    def repo_info(self, **kwargs):
        if self.info_error is not None:
            raise self.info_error
        if not self.exists:
            raise hub_error(RepositoryNotFoundError, 404)
        return SimpleNamespace(sha=self.revision,
                               siblings=[SimpleNamespace(rfilename=name) for name in self.files])

    def create_commit(self, **kwargs):
        if self.commit_errors:
            raise self.commit_errors.pop(0)
        operations = kwargs["operations"]
        staged = dict(self.files)
        for operation in operations:
            if isinstance(operation, CommitOperationAdd):
                source = operation.path_or_fileobj
                staged[operation.path_in_repo] = source if isinstance(source, bytes) else Path(source).read_bytes()
            elif isinstance(operation, CommitOperationDelete):
                folder = operation.path_in_repo.rstrip("/") + "/"
                staged = {name: data for name, data in staged.items()
                          if not (name.startswith(folder) if operation.is_folder else name == operation.path_in_repo)}
        self.files = staged
        self.commits.append({"operations": operations, "message": kwargs["commit_message"]})
        self.revision = f"revision-{len(self.commits)}"
        self.snapshots[self.revision] = dict(self.files)
        return SimpleNamespace(oid=self.revision)

    def download(self, **kwargs):
        self.downloads.append(kwargs)
        for name, data in self.snapshots[kwargs["revision"]].items():
            if any(fnmatchcase(name, pattern) for pattern in kwargs["allow_patterns"]):
                destination = Path(kwargs["local_dir"]) / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
        return str(kwargs["local_dir"])

    def names_under(self, prefix):
        return sorted(name for name in self.files if name.startswith(prefix))


def make_checkpoint(output_dir, step, best=None, max_steps=100, payload=None):
    """Write a folder shaped like one Trainer saves for a PEFT model."""
    folder = Path(output_dir) / f"checkpoint-{step}"
    folder.mkdir(parents=True)
    best_name = best or f"checkpoint-{step}"
    (folder / "trainer_state.json").write_text(json.dumps({
        "global_step": step, "max_steps": max_steps, "epoch": step / 10,
        "best_metric": 0.5, "best_model_checkpoint": str(Path(output_dir) / best_name),
        "log_history": [],
    }), encoding="utf-8")
    (folder / "adapter_model.safetensors").write_bytes(payload or f"adapter weights at {step}".encode())
    (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
    (folder / "optimizer.pt").write_bytes(f"optimizer at {step}".encode())
    (folder / "scheduler.pt").write_bytes(b"scheduler")
    (folder / "rng_state.pth").write_bytes(b"rng")
    return folder


IDENTITY = {"format_version": 1, "workflow": "test", "qlora": {"epochs": 3, "learning_rate": 1e-4},
            "data": {"train": {"rows": 2, "sha256": "a"}, "validation": {"rows": 1, "sha256": "b"}},
            "extra": {}}


class IdentityTests(TestCase):
    rows = {"train": [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}],
            "validation": [{"input_ids": [4, 5], "labels": [-100, 5]}]}

    def identity(self, config=None, rows=None, workflow="merging"):
        return training_checkpoint_identity(config or QLoRAConfig(), rows or self.rows, workflow=workflow)

    def test_paths_gpu_and_save_cadence_do_not_change_the_identity(self):
        original = self.identity()
        for changed in (replace(QLoRAConfig(), output_dir="/elsewhere"), replace(QLoRAConfig(), gpu_index=1),
                        replace(QLoRAConfig(), checkpoint_steps=5)):
            self.assertEqual(self.identity(changed), original)

    def test_training_settings_data_and_workflow_each_change_the_identity(self):
        original = self.identity()
        for changed in (replace(QLoRAConfig(), epochs=1.0), replace(QLoRAConfig(), learning_rate=2e-4),
                        replace(QLoRAConfig(), lora_rank=8), replace(QLoRAConfig(), seed=7),
                        replace(QLoRAConfig(), model_name="other/model")):
            self.assertNotEqual(self.identity(changed), original)
        edited = {**self.rows, "train": [{"input_ids": [1, 2, 9], "labels": [-100, 2, 9]}]}
        self.assertNotEqual(self.identity(rows=edited), original)
        self.assertNotEqual(self.identity(workflow="simplification"), original)

    def test_row_boundaries_and_order_are_part_of_the_data_fingerprint(self):
        a, b = {"input_ids": [1, 2], "labels": [1, 2]}, {"input_ids": [3], "labels": [3]}
        merged = {"input_ids": [1, 2, 3], "labels": [1, 2, 3]}
        self.assertNotEqual(fingerprint_tokenized_rows([a, b]), fingerprint_tokenized_rows([merged]))
        self.assertNotEqual(fingerprint_tokenized_rows([a, b]), fingerprint_tokenized_rows([b, a]))

    def test_merging_adapter_source_tracks_data_and_settings(self):
        manifest = {"sources": [{"sha256": "x"}], "split_question_ids": {"train": ["q"]},
                    "seed": 42, "split_ratios": [0.8, 0.1, 0.1]}
        source = merging_training_source(manifest, QLoRAConfig())
        self.assertEqual(source, merging_training_source(manifest, replace(QLoRAConfig(), output_dir="/other")))
        self.assertNotEqual(source, merging_training_source(manifest, replace(QLoRAConfig(), epochs=1.0)))
        self.assertNotEqual(source, merging_training_source({**manifest, "split_question_ids": {"train": ["z"]}},
                                                            QLoRAConfig()))


class AdapterSourceTests(TestCase):
    def test_matching_source_passes_and_mismatch_names_the_changed_field(self):
        source = {"workflow": "merging", "qlora": {"epochs": 3}}
        with TemporaryDirectory() as directory:
            (Path(directory) / "training_source.json").write_text(json.dumps(source), encoding="utf-8")
            self.assertTrue(verify_adapter_training_source(directory, source))
            with self.assertRaisesRegex(ValueError, "different qlora"):
                verify_adapter_training_source(directory, {**source, "qlora": {"epochs": 1}})

    def test_adapter_without_a_recorded_source_is_a_warning_only_when_allowed(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "cannot be verified"):
                verify_adapter_training_source(directory, {"a": 1})
            messages = []
            self.assertFalse(verify_adapter_training_source(
                directory, {"a": 1}, allow_unrecorded=True, log=messages.append))
            self.assertIn("cannot be verified", messages[0])


class StoreCase(TestCase):
    def setUp(self):
        self.hub = FakeTrainingHub()
        self.messages = []
        self.sleeps = []

    def store(self, output_dir, identity=None, **kwargs):
        options = dict(token="fake-secret", repo_id="researcher/training-state", upload_enabled=True,
                       restore_enabled=True, api=self.hub, download_fn=self.hub.download,
                       sleep=self.sleeps.append, log=self.messages.append)
        options.update(kwargs)
        return HubTrainingCheckpoints(output_dir, identity or IDENTITY, **options)

    def logged(self):
        return "\n".join(self.messages)


class UploadTests(StoreCase):
    def test_one_commit_carries_the_checkpoint_manifest_and_pointer(self):
        with TemporaryDirectory() as directory:
            store = self.store(directory)
            store.restore()
            folder = make_checkpoint(directory, 20)
            self.assertTrue(store.on_checkpoint_saved(folder, "checkpoint-20"))
            names = self.hub.names_under(store.remote_root + "/")
            self.assertEqual([name.split(store.remote_root + "/")[1] for name in names], [
                "checkpoint-20/adapter_config.json", "checkpoint-20/adapter_model.safetensors",
                "checkpoint-20/optimizer.pt", "checkpoint-20/rng_state.pth", "checkpoint-20/scheduler.pt",
                "checkpoint-20/trainer_state.json", "latest.json", "manifest.json",
            ])
            pointer = json.loads(self.hub.files[store.remote_root + "/latest.json"])
            self.assertEqual((pointer["latest"], pointer["best"], pointer["global_step"], pointer["max_steps"]),
                             ("checkpoint-20", "checkpoint-20", 20, 100))
            self.assertEqual(pointer["files"]["checkpoint-20"]["optimizer.pt"], len(b"optimizer at 20"))
            self.assertTrue(self.hub.creates[0]["private"])
            self.assertEqual(self.hub.creates[0]["repo_type"], "dataset")
            self.assertNotIn("fake-secret", json.dumps(self.hub.files, default=str))

    def test_new_checkpoint_replaces_old_ones_in_the_same_commit_but_keeps_the_best(self):
        with TemporaryDirectory() as directory:
            store = self.store(directory)
            store.restore()
            root = store.remote_root + "/"
            store.on_checkpoint_saved(make_checkpoint(directory, 20), "checkpoint-20")
            store.on_checkpoint_saved(make_checkpoint(directory, 40, best="checkpoint-20"), "checkpoint-20")
            self.assertEqual({n.split("/")[-2] for n in self.hub.names_under(root + "checkpoint-")},
                             {"checkpoint-20", "checkpoint-40"})
            commit = self.hub.commits[-1]["operations"]
            uploaded = {op.path_in_repo for op in commit if isinstance(op, CommitOperationAdd)}
            self.assertFalse(any("checkpoint-20/" in path for path in uploaded), "the best folder is not re-sent")
            store.on_checkpoint_saved(make_checkpoint(directory, 60, best="checkpoint-20"), "checkpoint-20")
            self.assertEqual({n.split("/")[-2] for n in self.hub.names_under(root + "checkpoint-")},
                             {"checkpoint-20", "checkpoint-60"})
            deletes = [op for op in self.hub.commits[-1]["operations"] if isinstance(op, CommitOperationDelete)]
            self.assertEqual([op.path_in_repo for op in deletes], [root + "checkpoint-40"])
            pointer = json.loads(self.hub.files[root + "latest.json"])
            self.assertEqual((pointer["latest"], pointer["best"]), ("checkpoint-60", "checkpoint-20"))

    def test_a_failed_commit_leaves_the_previous_checkpoint_intact(self):
        with TemporaryDirectory() as directory:
            store = self.store(directory, retry_delays=())
            store.restore()
            store.on_checkpoint_saved(make_checkpoint(directory, 20), "checkpoint-20")
            self.hub.commit_errors = [hub_error(HfHubHTTPError, 503)]
            self.assertFalse(store.on_checkpoint_saved(make_checkpoint(directory, 40), "checkpoint-40"))
            pointer = json.loads(self.hub.files[store.remote_root + "/latest.json"])
            self.assertEqual(pointer["latest"], "checkpoint-20")
            self.assertEqual(len(self.hub.names_under(store.remote_root + "/checkpoint-20/")), 6)

    def test_upload_failure_warns_and_training_continues_until_the_next_save(self):
        with TemporaryDirectory() as directory:
            store = self.store(directory, retry_delays=())
            store.restore()
            self.hub.commit_errors = [hub_error(HfHubHTTPError, 401)]
            self.assertFalse(store.on_checkpoint_saved(make_checkpoint(directory, 20), "checkpoint-20"))
            self.assertEqual(store.upload_failures, 1)
            self.assertIn("WARNING: could not upload checkpoint-20", self.logged())
            self.assertTrue(store.on_checkpoint_saved(make_checkpoint(directory, 40), "checkpoint-40"))
            self.assertEqual(json.loads(self.hub.files[store.remote_root + "/latest.json"])["latest"], "checkpoint-40")

    def test_strict_mode_makes_an_upload_failure_fatal(self):
        with TemporaryDirectory() as directory:
            store = self.store(directory, retry_delays=(), raise_on_upload_error=True)
            store.restore()
            self.hub.commit_errors = [hub_error(HfHubHTTPError, 401)]
            with self.assertRaises(HfHubHTTPError):
                store.on_checkpoint_saved(make_checkpoint(directory, 20), "checkpoint-20")

    def test_transient_hub_errors_are_retried_with_backoff(self):
        with TemporaryDirectory() as directory:
            store = self.store(directory, retry_delays=(1.0, 2.0, 3.0))
            store.restore()
            self.hub.commit_errors = [hub_error(HfHubHTTPError, 503), hub_error(HfHubHTTPError, 429)]
            self.assertTrue(store.on_checkpoint_saved(make_checkpoint(directory, 20), "checkpoint-20"))
            self.assertEqual(self.sleeps, [1.0, 2.0])
            self.assertEqual(store.upload_failures, 0)

    def test_upload_interval_skips_saves_and_flush_publishes_the_last_one(self):
        with TemporaryDirectory() as directory:
            store = self.store(directory, upload_every=2)
            store.restore()
            commits = len(self.hub.commits)
            self.assertFalse(store.on_checkpoint_saved(make_checkpoint(directory, 20), "checkpoint-20"))
            self.assertTrue(store.on_checkpoint_saved(make_checkpoint(directory, 40, best="checkpoint-20"), "checkpoint-20"))
            self.assertEqual(len(self.hub.commits), commits + 1)
            self.assertFalse(store.on_checkpoint_saved(make_checkpoint(directory, 60, best="checkpoint-20"), "checkpoint-20"))
            self.assertTrue(store.flush())
            self.assertEqual(json.loads(self.hub.files[store.remote_root + "/latest.json"])["latest"], "checkpoint-60")
            self.assertFalse(store.flush(), "nothing new to publish")

    def test_incomplete_local_checkpoint_is_refused_before_anything_is_sent(self):
        with TemporaryDirectory() as directory:
            store = self.store(directory, raise_on_upload_error=True)
            store.restore()
            folder = make_checkpoint(directory, 20)
            (folder / "adapter_model.safetensors").unlink()
            commits = len(self.hub.commits)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                store.on_checkpoint_saved(folder, "checkpoint-20")
            self.assertEqual(len(self.hub.commits), commits)

    def test_write_access_is_proven_before_training_starts(self):
        with TemporaryDirectory() as directory:
            store = self.store(directory, retry_delays=())
            self.hub.commit_errors = [hub_error(HfHubHTTPError, 403)]
            with self.assertRaises(HfHubHTTPError):
                store.restore()
            self.assertEqual(self.hub.creates[0]["repo_id"], "researcher/training-state")

    def test_uploading_can_be_disabled(self):
        with TemporaryDirectory() as directory:
            store = self.store(directory, upload_enabled=False)
            self.assertIsNone(store.restore())
            self.assertFalse(store.on_checkpoint_saved(make_checkpoint(directory, 20), "checkpoint-20"))
            self.assertEqual(self.hub.commits, [])


class RestoreTests(StoreCase):
    def publish(self, steps=(20, 40), best="checkpoint-20", identity=None):
        with TemporaryDirectory() as directory:
            store = self.store(Path(directory) / "first", identity)
            store.restore()
            for step in steps:
                chosen = best if best and step > int(best.split("-")[1]) else f"checkpoint-{step}"
                store.on_checkpoint_saved(make_checkpoint(Path(directory) / "first", step, best=chosen), chosen)
        return store

    def test_a_fresh_session_restores_latest_and_best_at_a_pinned_revision(self):
        published = self.publish()
        pinned = self.hub.revision
        self.hub.files["unrelated/file.json"] = b"{}"
        self.hub.snapshots[self.hub.revision] = dict(self.hub.files)
        with TemporaryDirectory() as directory:
            store = self.store(directory)
            resume = store.restore()
            self.assertEqual(Path(resume).name, "checkpoint-40")
            self.assertEqual((Path(directory) / "checkpoint-20" / "adapter_model.safetensors").read_bytes(),
                             b"adapter weights at 20")
            self.assertEqual((Path(resume) / "optimizer.pt").read_bytes(), b"optimizer at 40")
            self.assertTrue(all(call["revision"] == pinned for call in self.hub.downloads))
            self.assertEqual(self.hub.downloads[-1]["repo_type"], "dataset")
            self.assertEqual(self.hub.downloads[-1]["token"], "fake-secret")
            self.assertFalse((Path(directory) / "unrelated").exists())
            self.assertFalse(list(Path(directory).glob(".restore-*")), "staging folder is removed")
            self.assertEqual(published.fingerprint, store.fingerprint)

    def test_best_checkpoint_path_is_rewritten_for_this_machine(self):
        self.publish()
        with TemporaryDirectory() as directory:
            resume = self.store(directory).restore()
            state = json.loads((Path(resume) / "trainer_state.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(state["best_model_checkpoint"]), Path(directory).resolve() / "checkpoint-20")
            self.assertTrue(Path(state["best_model_checkpoint"]).is_dir())
            self.assertEqual(state["global_step"], 40)

    def test_nothing_on_the_hub_means_training_starts_from_the_beginning(self):
        for hub in (FakeTrainingHub(exists=False), FakeTrainingHub()):
            self.hub = hub
            with TemporaryDirectory() as directory:
                self.assertIsNone(self.store(directory).restore())
                self.assertEqual(hub.downloads, [])

    def test_a_manifest_only_identity_is_not_an_error(self):
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            self.store(first).restore()  # writes the manifest commit only
            self.assertIsNone(self.store(second).restore())
            self.assertIn("no checkpoint yet", self.logged())

    def test_explicit_missing_repo_without_upload_stops_instead_of_silently_restarting(self):
        self.hub = FakeTrainingHub(exists=False)
        with TemporaryDirectory() as directory:
            with self.assertRaises(RepositoryNotFoundError):
                self.store(directory, upload_enabled=False).restore()

    def test_auth_error_is_not_treated_as_a_missing_repository(self):
        self.hub.info_error = hub_error(HfHubHTTPError, 401)
        with TemporaryDirectory() as directory, self.assertRaises(HfHubHTTPError):
            self.store(directory, retry_delays=()).restore()

    def test_another_training_identity_is_not_restored_and_the_difference_is_named(self):
        self.publish()
        other = {**IDENTITY, "qlora": {"epochs": 1, "learning_rate": 1e-4}}
        with TemporaryDirectory() as directory:
            self.assertIsNone(self.store(directory, other).restore())
            self.assertIn("1 other configuration", self.logged())
            self.assertIn("step 40", self.logged())
            self.assertIn("qlora.epochs", self.logged())

    def test_a_newer_local_checkpoint_wins_over_the_hub(self):
        self.publish()
        with TemporaryDirectory() as directory:
            store = self.store(directory)
            store._adopt_local_directory()
            local = make_checkpoint(directory, 60, best="checkpoint-60")
            self.assertEqual(store.restore(), str(local))
            self.assertIn("at least as new", self.logged())
            self.assertFalse((Path(directory) / "checkpoint-40").exists())

    def test_a_newer_hub_checkpoint_wins_over_an_older_local_one(self):
        self.publish()
        with TemporaryDirectory() as directory:
            self.store(directory)._adopt_local_directory()
            make_checkpoint(directory, 20)
            self.assertEqual(Path(self.store(directory).restore()).name, "checkpoint-40")

    def test_truncated_or_missing_downloads_fail_instead_of_resuming_from_a_broken_folder(self):
        self.publish()
        for damage in ("truncate", "delete"):
            with self.subTest(damage=damage), TemporaryDirectory() as directory:
                def damaged(**kwargs):
                    self.hub.download(**kwargs)
                    for path in Path(kwargs["local_dir"]).rglob("optimizer.pt"):
                        path.write_bytes(b"x") if damage == "truncate" else path.unlink()
                with self.assertRaisesRegex(ValueError, "not restored completely"):
                    self.store(directory, download_fn=damaged).restore()
                self.assertFalse((Path(directory) / "checkpoint-40").exists())

    def test_tampered_manifest_or_pointer_fails_closed(self):
        published = self.publish()
        root = published.remote_root + "/"
        original = dict(self.hub.files)
        cases = {
            "manifest": (root + "manifest.json", lambda p: p["identity"].update(workflow="other")),
            "pointer_fingerprint": (root + "latest.json", lambda p: p.update(fingerprint="0" * 64)),
            "pointer_step": (root + "latest.json", lambda p: p.update(global_step=41)),
            "pointer_best": (root + "latest.json", lambda p: p.update(best="checkpoint-99")),
            "pointer_escape": (root + "latest.json",
                               lambda p: p["files"]["checkpoint-40"].update({"../evil": 1})),
            "pointer_incomplete": (root + "latest.json",
                                   lambda p: p["files"]["checkpoint-40"].pop("trainer_state.json")),
        }
        for label, (name, change) in cases.items():
            with self.subTest(case=label), TemporaryDirectory() as directory:
                self.hub.files = dict(original)
                payload = json.loads(self.hub.files[name])
                change(payload)
                self.hub.files[name] = json.dumps(payload).encode()
                self.hub.snapshots[self.hub.revision] = dict(self.hub.files)
                with self.assertRaises(ValueError):
                    self.store(directory).restore()

    def test_checkpoint_files_without_a_pointer_are_rejected(self):
        published = self.publish()
        del self.hub.files[published.remote_root + "/latest.json"]
        self.hub.snapshots[self.hub.revision] = dict(self.hub.files)
        with TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "no pointer"):
            self.store(directory).restore()

    def test_restoring_can_be_disabled(self):
        self.publish()
        with TemporaryDirectory() as directory:
            self.assertIsNone(self.store(directory, restore_enabled=False).restore())
            self.assertEqual(self.hub.downloads, [])


class LocalTrustTests(StoreCase):
    def test_local_checkpoints_of_the_same_run_are_resumed_without_the_hub(self):
        with TemporaryDirectory() as directory:
            first = self.store(directory, upload_enabled=False, restore_enabled=False)
            first.restore()
            folder = make_checkpoint(directory, 20)
            again = self.store(directory, upload_enabled=False, restore_enabled=False)
            self.assertEqual(again.restore(), str(folder))

    def test_checkpoints_from_another_configuration_are_set_aside_not_resumed(self):
        with TemporaryDirectory() as directory:
            old = self.store(directory, {**IDENTITY, "qlora": {"epochs": 9}}, upload_enabled=False, restore_enabled=False)
            old.restore()
            make_checkpoint(directory, 80)
            new = self.store(directory, upload_enabled=False, restore_enabled=False)
            self.assertIsNone(new.restore())
            self.assertEqual(new.local_checkpoints(), [])
            moved = list((Path(directory) / "_untrusted_checkpoints").rglob("checkpoint-80"))
            self.assertEqual(len(moved), 1)
            self.assertTrue((moved[0] / "optimizer.pt").is_file(), "nothing is deleted")
            self.assertIn("Moved 1 local checkpoint", self.logged())

    def test_unrecorded_legacy_checkpoints_are_set_aside_too(self):
        with TemporaryDirectory() as directory:
            make_checkpoint(directory, 80)
            store = self.store(directory, upload_enabled=False, restore_enabled=False)
            self.assertIsNone(store.restore())
            self.assertTrue(list((Path(directory) / "_untrusted_checkpoints" / "unrecorded").glob("checkpoint-80")))


class CallbackAndStatusTests(StoreCase):
    def test_trainer_callback_uploads_on_save_and_flushes_at_the_end(self):
        try:
            import transformers  # noqa: F401
        except ImportError:
            self.skipTest("transformers is not installed")
        with TemporaryDirectory() as directory:
            store = self.store(directory, upload_every=2)
            store.restore()
            callback = store.trainer_callback()
            for step, best in ((20, "checkpoint-20"), (40, "checkpoint-20"), (60, "checkpoint-20")):
                make_checkpoint(directory, step, best=best)
                callback.on_save(SimpleNamespace(output_dir=directory),
                                 SimpleNamespace(global_step=step, best_model_checkpoint=str(Path(directory) / best)),
                                 SimpleNamespace())
            callback.on_train_end(SimpleNamespace(output_dir=directory), SimpleNamespace(), SimpleNamespace())
            self.assertEqual(json.loads(self.hub.files[store.remote_root + "/latest.json"])["latest"], "checkpoint-60")

    def test_status_reports_this_configuration_and_the_others(self):
        with TemporaryDirectory() as directory:
            other = self.store(Path(directory) / "other", {**IDENTITY, "qlora": {"epochs": 1}})
            other.restore()
            other.on_checkpoint_saved(make_checkpoint(Path(directory) / "other", 20), "checkpoint-20")
            mine = self.store(Path(directory) / "mine")
            self.assertIsNone(mine.status()["matching"])
            mine.restore()
            mine.on_checkpoint_saved(make_checkpoint(Path(directory) / "mine", 40), "checkpoint-40")
            status = mine.status()
            self.assertEqual((status["matching"]["latest"], status["matching"]["global_step"]), ("checkpoint-40", 40))
            self.assertEqual(status["others"][0]["global_step"], 20)
            self.assertIn("qlora.epochs", status["others"][0]["differs"])
            self.hub.exists = False
            self.assertFalse(mine.status()["exists"])

    def test_progress_report_covers_training_adapter_and_evaluation(self):
        hub = FakeTrainingHub()
        hub.adapters["researcher/adapter"] = ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json"]
        with TemporaryDirectory() as directory:
            self.hub = hub
            store = self.store(directory)
            store.restore()
            store.on_checkpoint_saved(make_checkpoint(directory, 40), "checkpoint-40")
            hub.files["evaluations/math500/aaaaaaaaaaaa/manifest.json"] = b"{}"
            for index in range(3):
                hub.files[f"evaluations/math500/aaaaaaaaaaaa/question-{index:08d}.json"] = b"{}"
            lines = []
            report = describe_hub_progress(
                "fake-secret", training=store, api=hub, log=lines.append,
                adapter={"repo_id": None, "default_repo_name": "adapter"},
                evaluation={"repo_id": "researcher/training-state", "prefix": "evaluations",
                            "populations": ["math500", "gsm8k"]},
            )
            text = "\n".join(lines)
            self.assertIn("checkpoint-40 (step 40 of 100) matches this configuration -> training resumes", text)
            self.assertIn("researcher/adapter@abcdef1: complete -> training is skipped", text)
            self.assertIn("no training_source.json", text)
            self.assertIn("math500: 1 saved identity (aaaaaaaaaaaa: 3 question files)", text)
            self.assertIn("gsm8k: nothing saved yet", text)
            self.assertNotIn("fake-secret", text)
            self.assertEqual(report["adapter"]["revision"], "abcdef1234567")
            missing = describe_hub_progress("t", api=hub, log=lines.append,
                                            adapter={"repo_id": "researcher/missing", "default_repo_name": "x"})
            self.assertFalse(missing["adapter"]["exists"])
            self.assertIn("not found -> training runs", lines[-1])

    def test_progress_report_never_raises_when_a_stage_cannot_be_checked(self):
        hub = FakeTrainingHub()
        hub.info_error = hub_error(HfHubHTTPError, 401)
        lines = []
        report = describe_hub_progress(
            "t", api=hub, log=lines.append,
            evaluation={"repo_id": "researcher/e", "prefix": "evaluations", "populations": ["math500"]},
        )
        self.assertIn("could not check", lines[-1])
        self.assertIn("error", report["evaluation"])
