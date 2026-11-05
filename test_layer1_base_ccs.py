"""Offline checks of the LAYER1_BASE_CCS_ENABLED switch (Layer 1, Step C)."""

import sys
from types import ModuleType
import unittest
from unittest.mock import patch


def _import_without_sentence_transformers(name):
    stubbed = "sentence_transformers" not in sys.modules
    if stubbed:
        sentence_transformers = ModuleType("sentence_transformers")
        sentence_transformers.SentenceTransformer = type("SentenceTransformer", (), {})
        sys.modules["sentence_transformers"] = sentence_transformers
    try:
        return __import__(name, fromlist=["_"])
    finally:
        if stubbed:
            del sys.modules["sentence_transformers"]


layer1 = _import_without_sentence_transformers("src.layer1_base_execution")
from config import CONFIG


class NoCallManager:
    def generate_content(self, *args, **kwargs):
        raise AssertionError("Base CCS is off: no API call may be made")


class BaseCcsSwitchTest(unittest.TestCase):
    def baseline(self, **config):
        return layer1._execute_baseline_calculation(
            [4, 2, 0], {"questions": [], "solutions": []},
            NoCallManager(), NoCallManager(), {"MIRROR_N_OPTIMIZATION": 5, **config}, [],
        )

    def test_default_is_off(self):
        self.assertFalse(CONFIG["LAYER1_BASE_CCS_ENABLED"])

    def test_off_assumes_zero_without_calls(self):
        result = self.baseline()
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["intrinsic_baselines"], {4: 0.0, 2: 0.0, 0: 0.0})

    def test_on_calculates(self):
        with patch.object(layer1, "_calculate_baseline_difficulty", return_value={4: 0.4}) as calculate:
            result = self.baseline(LAYER1_BASE_CCS_ENABLED=True)
        calculate.assert_called_once()
        self.assertEqual(result["intrinsic_baselines"], {4: 0.4})

    def run_with_cache(self, cached_enabled, enabled):
        cached = {"metadata": {"last_completed_step": 4, "config_snapshot": {
            "LAYER1_ZERO_SHOT_CANDIDATES_N": 0, "LAYER1_BASE_CCS_ENABLED": cached_enabled}}}
        with patch.object(layer1, "_load_cached_state", return_value=cached), \
                patch.object(layer1, "_execute_retrieval", return_value={"status": "FAILURE"}):
            state = layer1.run_layer1_base_execution(
                0, "q", "a", None, [], [], None, {"question_to_index": {}}, None, None,
                {"LAYER1_ZERO_SHOT_CANDIDATES_N": 0, "LAYER1_BASE_CCS_ENABLED": enabled},
            )
        return state is cached, state

    def test_cache_with_assumed_zeros_is_not_reused_when_on(self):
        reused, state = self.run_with_cache(cached_enabled=False, enabled=True)
        self.assertFalse(reused)
        self.assertTrue(state["metadata"]["config_snapshot"]["LAYER1_BASE_CCS_ENABLED"])

    def test_cache_is_reused_otherwise(self):
        self.assertTrue(self.run_with_cache(cached_enabled=False, enabled=False)[0])
        self.assertTrue(self.run_with_cache(cached_enabled=True, enabled=False)[0])
        self.assertTrue(self.run_with_cache(cached_enabled=True, enabled=True)[0])


if __name__ == "__main__":
    unittest.main()
