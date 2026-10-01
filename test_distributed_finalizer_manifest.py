"""Offline checks for distributed code compatibility and manifest identity."""

from pathlib import Path
import json
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from src.distributed_code_compatibility import (
    _committed_python,
    _legacy_source_hash,
    diagnose_worker_code_compatibility,
    worker_code_unchanged_since_manifest,
)
from src.distributed_execution import (
    DistributedManifestMismatch,
    build_run_manifest,
    fingerprint_exemplar_data,
    validate_manifest_compatibility,
)
from src.hf_sync import DistributedManifestMismatchError, ensure_distributed_manifest


class FinalizerManifestTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "DISTRIBUTED_EXECUTION_ENABLED": True,
            "DISTRIBUTED_RUN_ID": "grouping-run",
            "DISTRIBUTED_WORKER_ID": 0,
            "DISTRIBUTED_WORKER_COUNT": 2,
            "HF_HUB_USERNAME": "example",
            "HF_HUB_REPO_NAME": "results",
        }
        self.experiment = {
            "experiment_name": "Layer1_Grouping_K5_Sizes_2",
            "APPLY_LAYER1_GROUPING": True,
            "LAYER1_GROUP_SIZES": [2],
            "TOP_N_CANDIDATES_RETRIEVAL": 5,
        }

    def manifest(self, code, *, experiment=None, questions=None, corpus="corpus-a"):
        return build_run_manifest(
            self.config,
            [experiment or self.experiment],
            questions or ["question one", "question two"],
            ["answer one", "answer two"],
            code_fingerprint=code,
            exemplar_fingerprint=corpus,
        )

    def test_worker_is_strict_and_finalizer_retains_saved_manifest_identity(self):
        saved = self.manifest("worker-code")
        current = self.manifest("finalizer-code")
        with self.assertRaises(DistributedManifestMismatch):
            validate_manifest_compatibility(saved, current)

        finalizer_candidate = self.manifest(saved["code_fingerprint"])
        self.assertEqual(validate_manifest_compatibility(saved, finalizer_candidate), {})
        self.assertEqual(saved["manifest_sha256"], finalizer_candidate["manifest_sha256"])

    def test_finalizer_still_rejects_scientific_and_corpus_changes(self):
        saved = self.manifest("worker-code")
        changed_experiment = {**self.experiment, "TOP_N_CANDIDATES_RETRIEVAL": 4}
        for current in (
            self.manifest("worker-code", experiment=changed_experiment),
            self.manifest("worker-code", corpus="corpus-b"),
            self.manifest("worker-code", questions=["different", "question two"]),
        ):
            with self.subTest(current=current["manifest_sha256"]):
                with self.assertRaises(DistributedManifestMismatch):
                    validate_manifest_compatibility(saved, current)

    def test_fresh_worker_validates_remote_manifest_before_pinning_code(self):
        saved = self.manifest("worker-code")
        self.config.update({"PERSIST_RESULTS_ONLINE": True, "HF_SYNC_TOKEN": "test-token"})
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest("current-code")), encoding="utf-8")
            with patch("src.hf_sync.HfApi") as api, patch(
                "src.hf_sync._remote_manifest_at_revision", return_value=(saved, "remote-hash")
            ), patch(
                "src.hf_sync.diagnose_worker_code_compatibility", return_value=(True, "match")
            ):
                api.return_value.repo_info.return_value = SimpleNamespace(sha="repo-head")
                self.assertEqual(
                    ensure_distributed_manifest(self.config, manifest_path), "remote-hash"
                )
                api.return_value.create_commit.assert_not_called()
            self.assertEqual(self.config["DISTRIBUTED_CODE_FINGERPRINT"], "worker-code")
            self.assertTrue(
                self.config["_DISTRIBUTED_MODEL_ROTATION_CODE_FINGERPRINT_AUTO_PINNED"]
            )

    def test_fresh_worker_rejects_remote_scientific_change(self):
        saved = self.manifest("worker-code")
        changed = {**self.experiment, "TOP_N_CANDIDATES_RETRIEVAL": 4}
        self.config.update({"PERSIST_RESULTS_ONLINE": True, "HF_SYNC_TOKEN": "test-token"})
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(
                json.dumps(self.manifest("current-code", experiment=changed)),
                encoding="utf-8",
            )
            with patch("src.hf_sync.HfApi") as api, patch(
                "src.hf_sync._remote_manifest_at_revision", return_value=(saved, "remote-hash")
            ), patch(
                "src.hf_sync.diagnose_worker_code_compatibility", return_value=(True, "match")
            ):
                api.return_value.repo_info.return_value = SimpleNamespace(sha="repo-head")
                with self.assertRaises(DistributedManifestMismatchError):
                    ensure_distributed_manifest(self.config, manifest_path)
            self.assertNotIn("DISTRIBUTED_CODE_FINGERPRINT", self.config)

    def test_model_rotation_flag_cannot_hide_changed_worker_code(self):
        saved = self.manifest("worker-code")
        self.config.update({
            "PERSIST_RESULTS_ONLINE": True,
            "HF_SYNC_TOKEN": "test-token",
            "DISTRIBUTED_ALLOW_MODEL_ROTATION": True,
            "AVALAI_MODEL_NAME_FINAL_SOLVER": "rotated-model",
        })
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest("current-code")), encoding="utf-8")
            with patch("src.hf_sync.HfApi") as api, patch(
                "src.hf_sync._remote_manifest_at_revision", return_value=(saved, "remote-hash")
            ), patch(
                "src.hf_sync.diagnose_worker_code_compatibility", return_value=(False, "changed")
            ):
                api.return_value.repo_info.return_value = SimpleNamespace(sha="repo-head")
                with self.assertRaises(DistributedManifestMismatchError):
                    ensure_distributed_manifest(self.config, manifest_path)
            self.assertNotIn("DISTRIBUTED_CODE_FINGERPRINT", self.config)

    def test_matching_manifests_do_not_authenticate_a_manual_code_pin(self):
        saved = self.manifest("worker-code")
        self.config.update({
            "PERSIST_RESULTS_ONLINE": True,
            "HF_SYNC_TOKEN": "test-token",
            "DISTRIBUTED_CODE_FINGERPRINT": "worker-code",
        })
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(saved), encoding="utf-8")
            with patch("src.hf_sync.HfApi") as api, patch(
                "src.hf_sync._remote_manifest_at_revision", return_value=(saved, "remote-hash")
            ), patch(
                "src.distributed_execution.resolve_code_fingerprint", return_value="changed-runtime-code"
            ), patch(
                "src.hf_sync.diagnose_worker_code_compatibility",
                return_value=(False, "worker source changed"),
            ):
                api.return_value.repo_info.return_value = SimpleNamespace(sha="repo-head")
                with self.assertRaisesRegex(
                    DistributedManifestMismatchError, "current worker source"
                ):
                    ensure_distributed_manifest(self.config, manifest_path)
                api.return_value.create_commit.assert_not_called()

    def test_worker_drops_manual_code_pin_before_building_manifest(self):
        sentence_transformers = ModuleType("sentence_transformers")
        sentence_transformers.SentenceTransformer = type("SentenceTransformer", (), {})
        with patch.dict(sys.modules, {"sentence_transformers": sentence_transformers}):
            from src import orchestration

        self.config["DISTRIBUTED_CODE_FINGERPRINT"] = "old-code"
        self.config["_DISTRIBUTED_MODEL_ROTATION_CODE_FINGERPRINT_AUTO_PINNED"] = True
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
        ) as write_manifest:
            with self.assertRaisesRegex(RuntimeError, "stop"):
                orchestration._prepare_distributed_worker(
                    self.config, [self.experiment], ["question"], ["answer"], {}
                )
        self.assertEqual(write_manifest.call_args.args[1]["code_fingerprint"], "current-code")
        self.assertNotIn("DISTRIBUTED_CODE_FINGERPRINT", self.config)

    def test_existing_local_worker_manifest_pins_only_authenticated_code(self):
        sentence_transformers = ModuleType("sentence_transformers")
        sentence_transformers.SentenceTransformer = type("SentenceTransformer", (), {})
        with patch.dict(sys.modules, {"sentence_transformers": sentence_transformers}):
            from src import orchestration

        saved = self.manifest("worker-code")
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(saved), encoding="utf-8")
            with patch.object(
                orchestration, "diagnose_worker_code_compatibility", return_value=(True, "match")
            ):
                self.assertTrue(orchestration._auto_pin_local_legacy_code_fingerprint(
                    self.config, str(manifest_path)
                ))
            self.assertEqual(self.config["DISTRIBUTED_CODE_FINGERPRINT"], "worker-code")
            validate_manifest_compatibility(
                saved, self.manifest(self.config["DISTRIBUTED_CODE_FINGERPRINT"])
            )

    def test_existing_local_manifest_rejects_stale_manual_pin(self):
        sentence_transformers = ModuleType("sentence_transformers")
        sentence_transformers.SentenceTransformer = type("SentenceTransformer", (), {})
        with patch.dict(sys.modules, {"sentence_transformers": sentence_transformers}):
            from src import orchestration

        self.config["DISTRIBUTED_CODE_FINGERPRINT"] = "worker-code"
        self.config["_DISTRIBUTED_RUNTIME_CODE_FINGERPRINT"] = "changed-runtime-code"
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest("worker-code")), encoding="utf-8")
            with patch.object(
                orchestration, "diagnose_worker_code_compatibility",
                return_value=(False, "worker source changed"),
            ):
                with self.assertRaisesRegex(
                    DistributedManifestMismatch, "worker source changed"
                ):
                    orchestration._auto_pin_local_legacy_code_fingerprint(
                        self.config, str(manifest_path)
                    )
        self.assertEqual(self.config["DISTRIBUTED_CODE_FINGERPRINT"], "worker-code")

    def test_finalizer_uses_saved_code_identity_and_current_runtime_provenance(self):
        sentence_transformers = ModuleType("sentence_transformers")
        sentence_transformers.SentenceTransformer = type("SentenceTransformer", (), {})
        with patch.dict(sys.modules, {"sentence_transformers": sentence_transformers}):
            from src import orchestration

        questions = ["question one", "question two"]
        answers = ["answer one", "answer two"]
        saved = build_run_manifest(
            self.config, [self.experiment], questions, answers,
            code_fingerprint="worker-code",
            exemplar_fingerprint=fingerprint_exemplar_data({}),
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            (run_root / "manifest.json").write_text(json.dumps(saved), encoding="utf-8")
            with patch.object(orchestration, "_validate_distributed_scope"), patch.object(
                orchestration, "configure_worker_paths", return_value={}
            ), patch.object(
                orchestration, "download_distributed_run_for_merge", return_value=str(run_root)
            ), patch.object(
                orchestration, "resolve_code_fingerprint", return_value="finalizer-code"
            ), patch.object(
                orchestration, "merge_distributed_run", return_value={"merged": True}
            ) as merge, patch.object(orchestration, "sync_distributed_merged_results"):
                result = orchestration.finalize_distributed_experiments(
                    experiment_configs=[self.experiment],
                    global_config=self.config,
                    hard_questions=questions,
                    hard_solutions=answers,
                    exemplar_data={},
                    api_managers={},
                    run_layer2=False,
                )
            merge.assert_called_once()
            self.assertEqual(result["merge"], {"merged": True})
            self.assertEqual(
                self.config["_DISTRIBUTED_RUNTIME_CODE_FINGERPRINT"], "finalizer-code"
            )
            self.assertEqual(
                json.loads((run_root / "manifest.json").read_text())["code_fingerprint"],
                "worker-code",
            )


class WorkerCodeCompatibilityTests(unittest.TestCase):
    def test_authenticated_infrastructure_change_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "src" / "orchestration.py").write_text(
                "def run_experiments():\n    return 1\n\n"
                "def _prepare_distributed_worker():\n    return 1\n\n"
                "def finalize_distributed_experiments():\n    return 1\n",
                encoding="utf-8",
            )
            (root / "src" / "distributed_execution.py").write_text(
                "def validate_manifest_compatibility():\n    return 1\n\n"
                "def build_run_manifest():\n    return 1\n",
                encoding="utf-8",
            )
            (root / "src" / "pipeline_steps.py").write_text(
                "def solve():\n    return 1\n", encoding="utf-8"
            )
            (root / "src" / "merging_finetuning.py").write_text(
                "def train():\n    return 1\n", encoding="utf-8"
            )
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                 "commit", "-qm", "original"],
                cwd=root, check=True,
            )
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            saved = f"git:{revision}:source:{_legacy_source_hash(_committed_python(root, revision))}"

            (root / "src" / "orchestration.py").write_text(
                "def run_experiments():\n    return 1\n\n"
                "def _prepare_distributed_worker():\n    return 2\n\n"
                "def finalize_distributed_experiments():\n    return 2\n",
                encoding="utf-8",
            )
            (root / "src" / "distributed_execution.py").write_text(
                "def validate_manifest_compatibility():\n    return 2\n\n"
                "def build_run_manifest():\n    return 1\n",
                encoding="utf-8",
            )
            (root / "test_new.py").write_text("assert True\n", encoding="utf-8")
            (root / "src" / "merging_finetuning.py").write_text(
                "def train():\n    return 2\n", encoding="utf-8"
            )
            self.assertTrue(worker_code_unchanged_since_manifest(root, saved))
            wrong_hash = saved[:-1] + ("0" if saved[-1] != "0" else "1")
            self.assertFalse(worker_code_unchanged_since_manifest(root, wrong_hash))

            (root / "src" / "pipeline_steps.py").write_text(
                "def solve():\n    return 2\n", encoding="utf-8"
            )
            self.assertFalse(worker_code_unchanged_since_manifest(root, saved))
            self.assertIn(
                "src/pipeline_steps.py",
                diagnose_worker_code_compatibility(root, saved)[1],
            )


if __name__ == "__main__":
    unittest.main()
