import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.distributed_execution import (
    DistributedManifestMismatch,
    build_run_manifest,
    merge_distributed_run,
    pending_indices_for_worker,
    record_avalai_model_provenance,
    validate_manifest_compatibility,
    write_or_validate_manifest,
    write_worker_status,
)
from src.hf_sync import ensure_distributed_manifest


class DistributedModelRotationTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "DISTRIBUTED_EXECUTION_ENABLED": True,
            "DISTRIBUTED_RUN_ID": "model-rotation-test",
            "DISTRIBUTED_WORKER_COUNT": 2,
            "DISTRIBUTED_WORKER_ID": 0,
            "DISTRIBUTED_HF_REVISION": "main",
            "HF_HUB_USERNAME": "example-user",
            "HF_HUB_REPO_NAME": "example-results",
            "HF_SYNC_TOKEN": "test-token",
            "PERSIST_RESULTS_ONLINE": True,
            "AVALAI_MODEL_NAME_ADAPTATION": "meta/llama-3.1-8b-instruct",
            "AVALAI_MODEL_NAME_FINAL_SOLVER": "meta/llama-3.1-8b-instruct",
            "AVALAI_MODEL_NAME_EVALUATOR": "openai/gpt-oss-20b",
            "AVALAI_REASONING_EFFORT": "low",
            "API_PROVIDER_ADAPTATION": "avalai",
            "API_PROVIDER_SOLVER": "avalai",
            "API_PROVIDER_EVALUATOR": "avalai",
        }
        self.experiments = [{
            "experiment_name": "keep-original-experiment-name",
            "APPLY_LAYER1_BASE_EXECUTION": False,
            "TOP_N_CANDIDATES_RETRIEVAL": 5,
        }]
        self.questions = ["q0", "q1"]
        self.answers = ["a0", "a1"]

    def manifest(self, config=None, *, code_fingerprint="fixed-code"):
        return build_run_manifest(
            config or self.config,
            self.experiments,
            self.questions,
            self.answers,
            code_fingerprint=code_fingerprint,
            exemplar_fingerprint="fixed-exemplars",
        )

    def rotated_config(self):
        rotated = dict(self.config)
        rotated.update({
            "DISTRIBUTED_ALLOW_MODEL_ROTATION": True,
            "AVALAI_MODEL_NAME_ADAPTATION": "meta/llama-3.2-11b-vision-instruct",
            "AVALAI_MODEL_NAME_FINAL_SOLVER": "meta/llama-3.2-11b-vision-instruct",
            "AVALAI_MODEL_NAME_EVALUATOR": "openai/gpt-oss-120b",
        })
        return rotated

    def test_default_mode_rejects_model_change(self):
        with self.assertRaises(DistributedManifestMismatch):
            validate_manifest_compatibility(
                self.manifest(),
                self.manifest(self.rotated_config()),
            )

    def test_opt_in_accepts_only_allowed_model_fields(self):
        changes = validate_manifest_compatibility(
            self.manifest(),
            self.manifest(self.rotated_config()),
            allow_model_rotation=True,
        )
        self.assertEqual(
            set(changes["keep-original-experiment-name"]),
            {
                "AVALAI_MODEL_NAME_ADAPTATION",
                "AVALAI_MODEL_NAME_FINAL_SOLVER",
                "AVALAI_MODEL_NAME_EVALUATOR",
            },
        )

    def test_opt_in_still_rejects_other_scientific_change(self):
        changed = self.rotated_config()
        changed["AVALAI_REASONING_EFFORT"] = "high"
        with self.assertRaises(DistributedManifestMismatch):
            validate_manifest_compatibility(
                self.manifest(),
                self.manifest(changed),
                allow_model_rotation=True,
            )

    def test_code_fingerprint_remains_strict_and_legacy_pin_matches(self):
        existing = self.manifest(code_fingerprint="original-code")
        with self.assertRaises(DistributedManifestMismatch):
            validate_manifest_compatibility(
                existing,
                self.manifest(self.rotated_config(), code_fingerprint="patched-code"),
                allow_model_rotation=True,
            )
        changes = validate_manifest_compatibility(
            existing,
            self.manifest(self.rotated_config(), code_fingerprint="original-code"),
            allow_model_rotation=True,
        )
        self.assertTrue(changes)

    def test_local_authoritative_manifest_is_not_overwritten(self):
        existing = self.manifest()
        requested = self.manifest(self.rotated_config())
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "manifest.json"
            path.write_text(json.dumps(existing, sort_keys=True), encoding="utf-8")
            original_bytes = path.read_bytes()
            write_or_validate_manifest(
                path,
                requested,
                allow_model_rotation=True,
            )
            self.assertEqual(path.read_bytes(), original_bytes)

    @patch("src.hf_sync._remote_manifest_at_revision")
    @patch("src.hf_sync.HfApi")
    def test_remote_authoritative_manifest_is_not_committed_over(
        self, mock_hf_api, mock_remote_manifest
    ):
        existing = self.manifest()
        requested = self.manifest(self.rotated_config())
        mock_remote_manifest.return_value = (existing, "remote-canonical-hash")
        api = MagicMock()
        api.repo_info.return_value.sha = "parent-commit"
        mock_hf_api.return_value = api
        config = self.rotated_config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "manifest.json"
            path.write_text(json.dumps(requested), encoding="utf-8")
            result = ensure_distributed_manifest(config, path)
        self.assertEqual(result, "remote-canonical-hash")
        api.create_commit.assert_not_called()

    def test_status_uses_original_manifest_hash_and_records_runtime_models(self):
        manifest = self.manifest()
        config = self.rotated_config()
        config["_DISTRIBUTED_RUNTIME_CODE_FINGERPRINT"] = "actual-patched-code"
        with tempfile.TemporaryDirectory() as temporary_directory:
            status_path = Path(temporary_directory) / "status.json"
            config["_DISTRIBUTED_WORKER_STATUS_PATH"] = str(status_path)
            write_worker_status(config, "RUNNING", manifest=manifest)
            status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["manifest_sha256"], manifest["manifest_sha256"])
        self.assertEqual(
            status["model_rotation"]["runtime_code_fingerprint"],
            "actual-patched-code",
        )
        self.assertEqual(
            status["model_rotation"]["active_avalai_models"][
                "AVALAI_MODEL_NAME_FINAL_SOLVER"
            ],
            "meta/llama-3.2-11b-vision-instruct",
        )

    def test_full_and_solve_only_provenance_history(self):
        run_log = {}
        record_avalai_model_provenance(run_log, self.config, "full")
        record_avalai_model_provenance(run_log, self.rotated_config(), "solve_only")
        self.assertEqual(len(run_log["avalai_model_config_history"]), 2)
        self.assertEqual(
            run_log["config_flags_used"]["AVALAI_MODEL_NAME_FINAL_SOLVER"],
            "meta/llama-3.2-11b-vision-instruct",
        )

    def test_successful_questions_stay_complete_and_failed_questions_stay_pending(self):
        config = dict(self.config)
        config["DISTRIBUTED_WORKER_COUNT"] = 1
        manifest = self.manifest(config)
        logs = [
            {
                "target_query_original_hard_list_idx": 0,
                "target_query_text": "q0",
                "pipeline_status": "SUCCESS",
            },
            {
                "target_query_original_hard_list_idx": 1,
                "target_query_text": "q1",
                "pipeline_status": "FAILURE",
            },
        ]
        self.assertEqual(
            pending_indices_for_worker(
                logs,
                manifest,
                0,
                require_layer1=False,
            ),
            [1],
        )

    def test_mixed_model_logs_merge_under_original_manifest_hash(self):
        manifest = self.manifest()
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_root = Path(temporary_directory)
            (run_root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            for worker_id, (question, model_name) in enumerate((
                ("q0", "meta/llama-3.1-8b-instruct"),
                ("q1", "meta/llama-3.2-11b-vision-instruct"),
            )):
                worker_root = run_root / "workers" / f"worker-{worker_id:03d}"
                results_root = worker_root / "results"
                results_root.mkdir(parents=True)
                status = {
                    "run_id": manifest["run_id"],
                    "worker_id": worker_id,
                    "worker_count": 2,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "assigned_indices": [worker_id],
                    "state": "COMPLETE",
                }
                (worker_root / "status.json").write_text(
                    json.dumps(status), encoding="utf-8"
                )
                log = {
                    "target_query_original_hard_list_idx": worker_id,
                    "target_query_text": question,
                    "pipeline_status": "SUCCESS",
                    "avalai_model_config_history": [{
                        "run_mode": "full",
                        "models": {"AVALAI_MODEL_NAME_FINAL_SOLVER": model_name},
                    }],
                }
                (results_root / "keep-original-experiment-name_run_log.json").write_text(
                    json.dumps([log]), encoding="utf-8"
                )

            result = merge_distributed_run(
                run_root,
                output_dir=run_root / "merged",
            )
            merged_logs = json.loads(
                (run_root / "merged" / "results" / "keep-original-experiment-name_run_log.json")
                .read_text(encoding="utf-8")
            )
        self.assertEqual(result["manifest"]["manifest_sha256"], manifest["manifest_sha256"])
        self.assertEqual(len(merged_logs), 2)


if __name__ == "__main__":
    unittest.main()
