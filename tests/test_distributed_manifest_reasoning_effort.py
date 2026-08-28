import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.distributed_execution import (
    DistributedManifestMismatch,
    build_run_manifest,
    validate_manifest_compatibility,
    write_or_validate_manifest,
)


ROLE_EFFORT_KEYS = (
    "AVALAI_REASONING_EFFORT_ADAPTATION",
    "AVALAI_REASONING_EFFORT_FINAL_SOLVER",
    "AVALAI_REASONING_EFFORT_EVALUATOR",
)


def _sha256_json(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resign_manifest(manifest):
    manifest["scientific_config_sha256"] = _sha256_json(
        {"experiments": manifest["experiments"]}
    )
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _sha256_json(unsigned)


class DistributedReasoningEffortManifestTests(unittest.TestCase):
    def _config(self, *, models=("old-a", "old-s", "old-e")):
        return {
            "DISTRIBUTED_RUN_ID": "reasoning-effort-compatibility-test",
            "DISTRIBUTED_WORKER_COUNT": 2,
            "DISTRIBUTED_WORKER_ID": 0,
            "HF_HUB_USERNAME": "owner",
            "HF_HUB_REPO_NAME": "repo",
            "AVALAI_REASONING_EFFORT": "low",
            "AVALAI_MODEL_NAME_ADAPTATION": models[0],
            "AVALAI_MODEL_NAME_FINAL_SOLVER": models[1],
            "AVALAI_MODEL_NAME_EVALUATOR": models[2],
        }

    def _manifest(self, config, *, code_fingerprint):
        return build_run_manifest(
            config,
            [{"experiment_name": "experiment"}],
            ["question-0", "question-1"],
            ["answer-0", "answer-1"],
            code_fingerprint=code_fingerprint,
            exemplar_fingerprint="fixed-exemplar",
        )

    def test_inherited_none_role_efforts_are_omitted_but_other_none_is_retained(self):
        config = self._config()
        config.update({key: None for key in ROLE_EFFORT_KEYS})
        config["UNRELATED_SCIENTIFIC_OPTION"] = None

        manifest = self._manifest(config, code_fingerprint="code")
        scientific_config = manifest["experiments"][0]["config"]

        for key in ROLE_EFFORT_KEYS:
            self.assertNotIn(key, scientific_config)
        self.assertIn("UNRELATED_SCIENTIFIC_OPTION", scientific_config)
        self.assertIsNone(scientific_config["UNRELATED_SCIENTIFIC_OPTION"])

    def test_legacy_nulls_allow_only_authorized_model_rotation(self):
        existing = self._manifest(
            self._config(),
            code_fingerprint="legacy-code",
        )
        for key in ROLE_EFFORT_KEYS:
            existing["experiments"][0]["config"][key] = None
        _resign_manifest(existing)

        expected = self._manifest(
            self._config(models=("new-a", "new-s", "new-e")),
            code_fingerprint="current-code",
        )

        changes = validate_manifest_compatibility(
            existing,
            expected,
            allow_model_rotation=True,
            allow_legacy_code_fingerprint=True,
        )

        self.assertEqual(
            set(changes["experiment"]),
            {
                "AVALAI_MODEL_NAME_ADAPTATION",
                "AVALAI_MODEL_NAME_FINAL_SOLVER",
                "AVALAI_MODEL_NAME_EVALUATOR",
            },
        )

    def test_pre_role_manifest_matches_current_inherited_none_configuration(self):
        existing = self._manifest(
            self._config(),
            code_fingerprint="legacy-code",
        )
        expected_config = self._config(models=("new-a", "new-s", "new-e"))
        expected_config.update({key: None for key in ROLE_EFFORT_KEYS})
        expected = self._manifest(
            expected_config,
            code_fingerprint="current-code",
        )

        changes = validate_manifest_compatibility(
            existing,
            expected,
            allow_model_rotation=True,
            allow_legacy_code_fingerprint=True,
        )

        self.assertEqual(len(changes["experiment"]), 3)

    def test_local_legacy_manifest_is_validated_without_being_rewritten(self):
        existing = self._manifest(
            self._config(),
            code_fingerprint="legacy-code",
        )
        for key in ROLE_EFFORT_KEYS:
            existing["experiments"][0]["config"][key] = None
        _resign_manifest(existing)
        expected = self._manifest(
            self._config(models=("new-a", "new-s", "new-e")),
            code_fingerprint="legacy-code",
        )

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            original_bytes = manifest_path.read_bytes()

            write_or_validate_manifest(
                manifest_path,
                expected,
                allow_model_rotation=True,
            )

            self.assertEqual(manifest_path.read_bytes(), original_bytes)

    def test_explicit_role_effort_change_remains_rejected(self):
        existing = self._manifest(
            self._config(),
            code_fingerprint="legacy-code",
        )
        expected_config = self._config(models=("new-a", "new-s", "new-e"))
        expected_config["AVALAI_REASONING_EFFORT_ADAPTATION"] = "high"
        expected = self._manifest(
            expected_config,
            code_fingerprint="current-code",
        )

        with self.assertRaises(DistributedManifestMismatch):
            validate_manifest_compatibility(
                existing,
                expected,
                allow_model_rotation=True,
                allow_legacy_code_fingerprint=True,
            )

    def test_unrelated_scientific_change_remains_rejected(self):
        existing = self._manifest(
            self._config(),
            code_fingerprint="legacy-code",
        )
        expected_config = copy.deepcopy(
            self._config(models=("new-a", "new-s", "new-e"))
        )
        expected_config["UNRELATED_SCIENTIFIC_OPTION"] = None
        expected = self._manifest(
            expected_config,
            code_fingerprint="current-code",
        )

        with self.assertRaises(DistributedManifestMismatch):
            validate_manifest_compatibility(
                existing,
                expected,
                allow_model_rotation=True,
                allow_legacy_code_fingerprint=True,
            )


if __name__ == "__main__":
    unittest.main()
