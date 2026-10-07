"""Offline checks for resuming a distributed run after worker code changed.

The immutable manifest keeps the run's code identity.  Edits outside the
worker's import closure never block a resume.  Edits inside it block a resume
unless the operator sets ``DISTRIBUTED_ALLOW_WORKER_CODE_CHANGE``; the accepted
change is then recorded as provenance in worker status and run logs.
"""

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from src.distributed_code_compatibility import (
    ALLOW_WORKER_CODE_CHANGE_KEY,
    WORKER_CODE_CHANGE_STATE_KEY,
    accept_worker_code_change,
)
from src.distributed_execution import (
    DistributedManifestMismatch,
    build_run_manifest,
    configure_worker_paths,
    record_worker_code_provenance,
    write_worker_status,
)
from src.hf_sync import (
    DistributedManifestMismatchError,
    _initialize_distributed_workspace,
    ensure_distributed_manifest,
)


CHANGED = (False, "Worker source changed in: src/batching.py.")


def _orchestration():
    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = type("SentenceTransformer", (), {})
    with patch.dict(sys.modules, {"sentence_transformers": sentence_transformers}):
        from src import orchestration
    return orchestration


class WorkerCodeChangeTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "DISTRIBUTED_EXECUTION_ENABLED": True,
            "DISTRIBUTED_RUN_ID": "grouping-run",
            "DISTRIBUTED_WORKER_ID": 0,
            "DISTRIBUTED_WORKER_COUNT": 2,
            "HF_HUB_USERNAME": "example",
            "HF_HUB_REPO_NAME": "results",
            "PERSIST_RESULTS_ONLINE": True,
            "HF_SYNC_TOKEN": "test-token",
        }
        self.experiment = {
            "experiment_name": "Layer1_Grouping_K5_Sizes_2",
            "APPLY_LAYER1_GROUPING": True,
            "LAYER1_GROUP_SIZES": [2],
            "TOP_N_CANDIDATES_RETRIEVAL": 5,
        }

    def manifest(self, code, *, experiment=None, config=None):
        return build_run_manifest(
            config or self.config,
            [experiment or self.experiment],
            ["question one", "question two"],
            ["answer one", "answer two"],
            code_fingerprint=code,
            exemplar_fingerprint="corpus-a",
        )

    def assert_recorded(self, config):
        record = config[WORKER_CODE_CHANGE_STATE_KEY]
        self.assertEqual(record["manifest_code_fingerprint"], "worker-code")
        self.assertEqual(record["runtime_code_fingerprint"], "current-code")
        self.assertEqual(record["authorized_by"], ALLOW_WORKER_CODE_CHANGE_KEY)
        self.assertIn("src/batching.py", record["reason"])

    def test_opt_in_is_required_and_recorded_once(self):
        config = {}
        self.assertFalse(accept_worker_code_change(
            config, saved_fingerprint="a", runtime_fingerprint="b",
            reason="changed", source="manifest",
        ))
        self.assertNotIn(WORKER_CODE_CHANGE_STATE_KEY, config)
        config[ALLOW_WORKER_CODE_CHANGE_KEY] = True
        for _ in range(2):
            self.assertTrue(accept_worker_code_change(
                config, saved_fingerprint="a", runtime_fingerprint="b",
                reason="changed", source="manifest",
            ))
        self.assertEqual(config[WORKER_CODE_CHANGE_STATE_KEY]["checked_against"], "manifest")

    def test_remote_manifest_resumes_changed_worker_code_only_with_opt_in(self):
        saved = self.manifest("worker-code")
        for allowed in (False, True):
            with self.subTest(allowed=allowed), tempfile.TemporaryDirectory() as temporary:
                config = {**self.config, ALLOW_WORKER_CODE_CHANGE_KEY: allowed}
                manifest_path = Path(temporary) / "manifest.json"
                manifest_path.write_text(json.dumps(self.manifest("current-code")), encoding="utf-8")
                with patch("src.hf_sync.HfApi") as api, patch(
                    "src.hf_sync._remote_manifest_at_revision", return_value=(saved, "remote-hash")
                ), patch(
                    "src.distributed_execution.resolve_code_fingerprint", return_value="current-code"
                ), patch("src.hf_sync.diagnose_worker_code_compatibility", return_value=CHANGED):
                    api.return_value.repo_info.return_value = SimpleNamespace(sha="repo-head")
                    if not allowed:
                        with self.assertRaisesRegex(
                            DistributedManifestMismatchError, ALLOW_WORKER_CODE_CHANGE_KEY
                        ):
                            ensure_distributed_manifest(config, manifest_path)
                        self.assertNotIn("DISTRIBUTED_CODE_FINGERPRINT", config)
                        self.assertNotIn(WORKER_CODE_CHANGE_STATE_KEY, config)
                        continue
                    self.assertEqual(ensure_distributed_manifest(config, manifest_path), "remote-hash")
                    api.return_value.create_commit.assert_not_called()
                self.assertEqual(config["DISTRIBUTED_CODE_FINGERPRINT"], "worker-code")
                self.assert_recorded(config)

    def test_opt_in_never_hides_scientific_changes(self):
        saved = self.manifest("worker-code")
        changed = {**self.experiment, "TOP_N_CANDIDATES_RETRIEVAL": 4}
        config = {**self.config, ALLOW_WORKER_CODE_CHANGE_KEY: True}
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(
                json.dumps(self.manifest("current-code", experiment=changed)), encoding="utf-8"
            )
            with patch("src.hf_sync.HfApi") as api, patch(
                "src.hf_sync._remote_manifest_at_revision", return_value=(saved, "remote-hash")
            ), patch(
                "src.distributed_execution.resolve_code_fingerprint", return_value="current-code"
            ), patch("src.hf_sync.diagnose_worker_code_compatibility", return_value=CHANGED):
                api.return_value.repo_info.return_value = SimpleNamespace(sha="repo-head")
                with self.assertRaisesRegex(DistributedManifestMismatchError, "different inputs/config"):
                    ensure_distributed_manifest(config, manifest_path)
                api.return_value.create_commit.assert_not_called()
            self.assertNotIn("DISTRIBUTED_CODE_FINGERPRINT", config)

    def test_workspace_restore_honors_opt_in(self):
        for allowed in (False, True):
            with self.subTest(allowed=allowed), tempfile.TemporaryDirectory() as temporary:
                config = {
                    **self.config, "BASE_OUTPUT_DIR": temporary,
                    ALLOW_WORKER_CODE_CHANGE_KEY: allowed,
                }
                paths = configure_worker_paths(config)
                expected = self.manifest("worker-code", config=config)
                remote_bytes = json.dumps(expected).encode("utf-8")

                def stage_remote(**kwargs):
                    remote_manifest = Path(kwargs["local_dir"]) / "distributed_runs/grouping-run/manifest.json"
                    remote_manifest.parent.mkdir(parents=True)
                    remote_manifest.write_bytes(remote_bytes)

                with patch("src.hf_sync.snapshot_download", side_effect=stage_remote), patch(
                    "src.distributed_execution.resolve_code_fingerprint", return_value="current-code"
                ), patch("src.hf_sync.diagnose_worker_code_compatibility", return_value=CHANGED):
                    if not allowed:
                        with self.assertRaisesRegex(DistributedManifestMismatchError, "incompatible"):
                            _initialize_distributed_workspace(config, expected_manifest=expected)
                        self.assertFalse(Path(paths["manifest_path"]).exists())
                        continue
                    _initialize_distributed_workspace(config, expected_manifest=expected)
                self.assertEqual(Path(paths["manifest_path"]).read_bytes(), remote_bytes)
                self.assert_recorded(config)

    def test_local_manifest_pin_honors_opt_in(self):
        orchestration = _orchestration()
        saved = self.manifest("worker-code")
        for allowed in (False, True):
            with self.subTest(allowed=allowed), tempfile.TemporaryDirectory() as temporary:
                config = {
                    **self.config, ALLOW_WORKER_CODE_CHANGE_KEY: allowed,
                    "_DISTRIBUTED_RUNTIME_CODE_FINGERPRINT": "current-code",
                }
                manifest_path = Path(temporary) / "manifest.json"
                manifest_path.write_text(json.dumps(saved), encoding="utf-8")
                with patch.object(
                    orchestration, "diagnose_worker_code_compatibility", return_value=CHANGED
                ):
                    if not allowed:
                        with self.assertRaisesRegex(
                            DistributedManifestMismatch, ALLOW_WORKER_CODE_CHANGE_KEY
                        ):
                            orchestration._auto_pin_local_legacy_code_fingerprint(
                                config, str(manifest_path)
                            )
                        self.assertNotIn("DISTRIBUTED_CODE_FINGERPRINT", config)
                        continue
                    self.assertTrue(orchestration._auto_pin_local_legacy_code_fingerprint(
                        config, str(manifest_path)
                    ))
                self.assertEqual(config["DISTRIBUTED_CODE_FINGERPRINT"], "worker-code")
                self.assert_recorded(config)

    def test_worker_startup_discards_a_stale_acceptance_record(self):
        orchestration = _orchestration()
        self.config[WORKER_CODE_CHANGE_STATE_KEY] = {"stale": True}
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            orchestration, "_validate_distributed_scope"
        ), patch.object(
            orchestration, "configure_worker_paths",
            return_value={"manifest_path": str(Path(temporary) / "manifest.json")},
        ), patch.object(
            orchestration, "resolve_code_fingerprint", return_value="current-code"
        ), patch(
            "src.distributed_execution.resolve_code_fingerprint", return_value="current-code"
        ), patch.object(
            orchestration, "write_or_validate_manifest", side_effect=RuntimeError("stop")
        ):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                orchestration._prepare_distributed_worker(
                    self.config, [self.experiment], ["question"], ["answer"], {}
                )
        self.assertNotIn(WORKER_CODE_CHANGE_STATE_KEY, self.config)

    def test_status_and_run_logs_record_runtime_code_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = {
                **self.config, "BASE_OUTPUT_DIR": temporary,
                "_DISTRIBUTED_RUNTIME_CODE_FINGERPRINT": "current-code",
            }
            configure_worker_paths(config)
            manifest = self.manifest("worker-code", config=config)

            run_log = {}
            record_worker_code_provenance(run_log, config, "full")
            record_worker_code_provenance(run_log, config, "full")
            self.assertEqual(run_log["worker_code_history"], [
                {"run_mode": "full", "runtime_code_fingerprint": "current-code"},
            ])
            status = json.loads(write_worker_status(config, "RUNNING", manifest=manifest).read_text())
            self.assertEqual(status["worker_code"], {
                "manifest_code_fingerprint": "worker-code",
                "runtime_code_fingerprint": "current-code",
                "accepted_change": None,
            })

            config[ALLOW_WORKER_CODE_CHANGE_KEY] = True
            accept_worker_code_change(
                config, saved_fingerprint="worker-code", runtime_fingerprint="current-code",
                reason=CHANGED[1], source="manifest",
            )
            record_worker_code_provenance(run_log, config, "solve_only")
            self.assertEqual(len(run_log["worker_code_history"]), 2)
            self.assertEqual(
                run_log["worker_code_history"][-1]["accepted_worker_code_change"]["reason"],
                CHANGED[1],
            )
            status = json.loads(write_worker_status(config, "RUNNING", manifest=manifest).read_text())
            self.assertEqual(status["worker_code"]["accepted_change"]["reason"], CHANGED[1])

    def test_non_distributed_run_logs_are_untouched(self):
        run_log = {}
        record_worker_code_provenance(run_log, {}, "full")
        self.assertEqual(run_log, {})


if __name__ == "__main__":
    unittest.main()
