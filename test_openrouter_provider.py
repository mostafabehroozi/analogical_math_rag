"""Offline checks for OpenRouter request construction and provider integration."""

import copy
import ast
import json
import logging
from pathlib import Path
import types
import unittest
from unittest.mock import patch

from src import api_manager
from src.distributed_execution import (
    DistributedManifestMismatch,
    build_run_manifest,
    sanitize_scientific_config,
    validate_manifest_compatibility,
)


class FakeStatusError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


class FakeRateLimitError(Exception):
    pass


class FakeClient:
    instances = []
    responses = []

    def __init__(self, **kwargs):
        self.options = kwargs
        self.calls = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
        self.instances.append(self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def completion(model="author/model", text="answer", metadata=None):
    return types.SimpleNamespace(
        model=model,
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))],
        model_extra={"openrouter_metadata": metadata} if metadata else {},
    )


class OpenRouterProviderTests(unittest.TestCase):
    def setUp(self):
        FakeClient.instances = []
        FakeClient.responses = []
        fake_openai = types.SimpleNamespace(
            OpenAI=FakeClient,
            RateLimitError=FakeRateLimitError,
            APIStatusError=FakeStatusError,
        )
        self.openai_patch = patch.object(api_manager, "openai", fake_openai)
        self.pacer_patch = patch.object(api_manager.provider_pacer, "pace")
        self.log_patch = patch.object(api_manager, "tprint")
        self.openai_patch.start()
        self.pacer = self.pacer_patch.start()
        self.log_patch.start()
        self.addCleanup(self.openai_patch.stop)
        self.addCleanup(self.pacer_patch.stop)
        self.addCleanup(self.log_patch.stop)
        self.config = {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
            "OPENROUTER_PROVIDER_ROUTING": {"order": ["provider-a", "provider-b"], "allow_fallbacks": True},
            "OPENROUTER_MODEL_ROUTING": {},
            "OPENROUTER_MODEL_FALLBACKS": {"author/model": ["other/model"]},
            "OPENROUTER_REASONING_EFFORT": "low",
            "OPENROUTER_REASONING_EFFORT_FINAL_SOLVER": "high",
            "OPENROUTER_HTTP_REFERER": "https://example.org",
            "OPENROUTER_APP_TITLE": "Research",
            "GLOBAL_API_CALL_DELAY_SECONDS": {"openrouter": 2},
            "ENABLE_API_RETRY": False,
            "GLOBAL_CONSECUTIVE_ERROR_LIMIT": 0,
            "GLOBAL_PERIODIC_PAUSE_INTERVAL_MINUTES": 0,
        }
        self.manager = api_manager.OpenRouterAPIManager(
            "test-key", {"default": {"rpm": 1000}}, self.config
        )

    def test_request_routing_reasoning_and_metadata(self):
        FakeClient.responses = [completion(
            model="other/model",
            metadata={"endpoints": {"available": [{"provider": "provider-b", "selected": True}]}},
        )]
        result = self.manager.generate_content("prompt", "author/model", temperature=0.4, avalai_role="final_solver")
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["request_meta"]["requested_model"], "author/model")
        self.assertEqual(result["request_meta"]["returned_model"], "other/model")
        self.assertEqual(result["request_meta"]["routed_provider"], "provider-b")
        request = FakeClient.instances[0].calls[0]
        self.assertEqual(request["extra_body"], {
            "provider": {"order": ["provider-a", "provider-b"], "allow_fallbacks": True},
            "models": ["other/model"],
            "reasoning": {"effort": "high"},
        })
        self.assertEqual(request["temperature"], 0.4)
        self.assertEqual(FakeClient.instances[0].options["default_headers"]["HTTP-Referer"], "https://example.org")
        self.assertEqual(FakeClient.instances[0].options["max_retries"], 0)
        self.pacer.assert_called_with("openrouter", 2.0)

    def test_experiment_bindings_are_isolated(self):
        first_config = copy.deepcopy(self.config)
        second_config = copy.deepcopy(self.config)
        first_config["OPENROUTER_MODEL_ROUTING"] = {"author/model": {"only": ["provider-a"]}}
        second_config["OPENROUTER_MODEL_ROUTING"] = {"author/model": {"only": ["provider-b"]}}
        first = api_manager.bind_api_managers({"openrouter": self.manager}, first_config)["openrouter"]
        second = api_manager.bind_api_managers({"openrouter": self.manager}, second_config)["openrouter"]
        self.assertIs(first.scheduler, second.scheduler)
        self.assertIs(first.clients, second.clients)
        self.assertEqual(first._extra_body("author/model", None)["provider"], {"only": ["provider-a"]})
        self.assertEqual(second._extra_body("author/model", None)["provider"], {"only": ["provider-b"]})
        self.assertEqual(self.manager._extra_body("author/model", None)["provider"], self.config["OPENROUTER_PROVIDER_ROUTING"])
        FakeClient.responses = [completion(), completion()]
        first.generate_content("first", "author/model")
        second.generate_content("second", "author/model")
        calls = FakeClient.instances[0].calls
        self.assertEqual(calls[0]["extra_body"]["provider"], {"only": ["provider-a"]})
        self.assertEqual(calls[1]["extra_body"]["provider"], {"only": ["provider-b"]})

    def test_terminal_payment_error_is_not_retried(self):
        self.config.update(ENABLE_API_RETRY=True, MAX_API_RETRIES=5, RETRY_ALL_API_ERRORS=True)
        FakeClient.responses = [FakeStatusError(402)]
        result = self.manager.generate_content("prompt", "author/model")
        self.assertEqual(result["error_type"], "PaymentRequired")
        self.assertEqual(len(FakeClient.instances[0].calls), 1)

    def test_rate_limit_retries_and_empty_text_is_reported(self):
        self.config.update(ENABLE_API_RETRY=True, MAX_API_RETRIES=2, RETRY_ALL_API_ERRORS=True,
                           API_RETRY_DELAY_SECONDS=0, API_KEY_ERROR_COOLDOWN_SECONDS=0)
        FakeClient.responses = [FakeRateLimitError("limit"), completion()]
        result = self.manager.generate_content("prompt", "author/model")
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(len(FakeClient.instances[0].calls), 2)
        FakeClient.responses = [completion(text="")]
        self.config["ENABLE_API_RETRY"] = False
        result = self.manager.generate_content("prompt", "author/model")
        self.assertEqual(result["error_type"], "NoChoices")

    def test_model_roles_legacy_delay_and_manifest(self):
        models = {
            "OPENROUTER_MODEL_NAME_ADAPTATION": "author/adapt",
            "OPENROUTER_MODEL_NAME_FINAL_SOLVER": "author/solve",
            "OPENROUTER_MODEL_NAME_EVALUATOR": "author/eval",
            "OPENROUTER_MODEL_NAME_SIMPLIFICATION": "author/simplify",
        }
        for role, expected in zip(("adaptation", "solver", "evaluator", "simplification"), models.values()):
            self.assertEqual(api_manager.model_name_for(self.manager, models, role), expected)
        api_manager._pace_provider_call({"GLOBAL_API_CALL_DELAY_SECONDS": 3}, "openrouter")
        self.pacer.assert_called_with("openrouter", 3.0)
        inactive = sanitize_scientific_config({"API_PROVIDER_SOLVER": "gemini", **models,
                                               "OPENROUTER_MODEL_ROUTING": {"author/solve": {"only": ["provider-a"]}}})
        active = sanitize_scientific_config({"API_PROVIDER_SOLVER": "openrouter", **models,
                                             "OPENROUTER_MODEL_ROUTING": {"author/solve": {"only": ["provider-a"]}}})
        self.assertNotIn("OPENROUTER_MODEL_ROUTING", inactive)
        self.assertEqual(active["OPENROUTER_MODEL_ROUTING"], {"author/solve": {"only": ["provider-a"]}})

    def test_distributed_manifest_tracks_active_routing_and_model_rotation(self):
        base = {
            "DISTRIBUTED_RUN_ID": "router-test",
            "DISTRIBUTED_WORKER_ID": 0,
            "DISTRIBUTED_WORKER_COUNT": 1,
            "HF_HUB_USERNAME": "example",
            "HF_HUB_REPO_NAME": "results",
            "API_PROVIDER_SOLVER": "gemini",
        }

        def manifest(config):
            return build_run_manifest(
                config, [{"experiment_name": "test"}], ["question"],
                code_fingerprint="fixed-code", exemplar_fingerprint="fixed-corpus",
            )

        existing = manifest(base)
        unused = manifest({**base, "OPENROUTER_MODEL_NAME_FINAL_SOLVER": None,
                           "OPENROUTER_MODEL_ROUTING": {"author/model": {"only": ["provider-a"]}}})
        self.assertEqual(existing["scientific_config_sha256"], unused["scientific_config_sha256"])

        active = {**base, "API_PROVIDER_SOLVER": "openrouter",
                  "OPENROUTER_MODEL_NAME_FINAL_SOLVER": "author/model",
                  "OPENROUTER_MODEL_ROUTING": {"author/model": {"only": ["provider-a"]}}}
        first = manifest(active)
        changed_routing = manifest({**active, "OPENROUTER_MODEL_ROUTING": {
            "author/model": {"only": ["provider-b"]}}})
        with self.assertRaises(DistributedManifestMismatch):
            validate_manifest_compatibility(first, changed_routing)
        changed_model = manifest({**active, "OPENROUTER_MODEL_NAME_FINAL_SOLVER": "other/model"})
        changes = validate_manifest_compatibility(first, changed_model)
        self.assertIn("OPENROUTER_MODEL_NAME_FINAL_SOLVER", changes["test"])

    def test_notebook_initializes_selected_openrouter_and_checks_config(self):
        notebook = json.loads(Path("main_experiment.ipynb").read_text(encoding="utf-8"))
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if any("def initialize_required_api_managers" in line for line in cell.get("source", [])))
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "initialize_required_api_managers")
        module = ast.Module(body=[function], type_ignores=[])
        namespace = {
            "SUPPORTED_API_PROVIDERS": {"gemini", "avalai", "openrouter", "ollama"},
            "PROVIDER_CONFIG_KEYS": (
                "API_PROVIDER_ADAPTATION", "API_PROVIDER_SOLVER",
                "API_PROVIDER_EVALUATOR", "API_PROVIDER_SIMPLIFICATION",
            ),
            "_usable_notebook_credential": lambda value: bool(value) and "YOUR_" not in str(value),
            "OpenRouterAPIManager": api_manager.OpenRouterAPIManager,
            "startup_progress": lambda message: None,
            "logger": logging.getLogger("openrouter-test"),
        }
        exec(compile(module, "notebook-setup", "exec"), namespace)
        initialize = namespace["initialize_required_api_managers"]
        config = copy.deepcopy(self.config)
        config.update({
            "API_PROVIDER_ADAPTATION": "openrouter",
            "API_PROVIDER_SOLVER": "openrouter",
            "API_PROVIDER_EVALUATOR": "openrouter",
            "API_PROVIDER_SIMPLIFICATION": "openrouter",
            "OPENROUTER_MODEL_QUOTAS": {"default": {"rpm": 1000}},
            "OPENROUTER_MODEL_NAME_ADAPTATION": "author/adapt",
            "OPENROUTER_MODEL_NAME_FINAL_SOLVER": "author/solve",
            "OPENROUTER_MODEL_NAME_EVALUATOR": "author/eval",
            "OPENROUTER_MODEL_NAME_SIMPLIFICATION": "author/simplify",
        })
        self.assertIsInstance(initialize(config)["openrouter"], api_manager.OpenRouterAPIManager)
        missing_model = {**config, "OPENROUTER_MODEL_NAME_EVALUATOR": None}
        with self.assertRaisesRegex(ValueError, "OPENROUTER_MODEL_NAME_EVALUATOR"):
            initialize(missing_model)
        missing_key = {**config, "OPENROUTER_API_KEY": ""}
        with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
            initialize(missing_key)


if __name__ == "__main__":
    unittest.main()
