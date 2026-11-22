"""Offline checks of Kaggle model proxy keys minted from a Kaggle API token (src/kaggle_proxy_auth.py)."""

import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from src import api_manager
from src import kaggle_proxy_auth as kpa
from test_batch_providers import BATCH_CONFIG, _Quiet, concurrently, fake_kbench

TOKEN = "KGAT_example_token_value"


class FakeResponse:
    def __init__(self, status, data=None):
        self.status_code = status
        self._data = data

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


def recording_post(responses):
    """A requests.post stand-in returning ``responses`` in turn and recording each call."""
    calls = []

    def post(url, json=None, headers=None, auth=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "auth": auth})
        return responses[min(len(calls), len(responses)) - 1]

    return post, calls


class CredentialResolutionTest(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.home, ignore_errors=True))
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
            os.environ.pop(name, None)

    def resolve(self, config=None, secret=None):
        return kpa.resolve_kaggle_credential(config or {}, secret_fn=lambda name: secret, home=self.home)

    def write(self, name, text):
        os.makedirs(os.path.join(self.home, ".kaggle"), exist_ok=True)
        path = os.path.join(self.home, ".kaggle", name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def test_sources_in_order(self):
        self.write("kaggle.json", json.dumps({"username": "u", "key": "k"}))
        legacy = self.resolve()
        self.assertEqual((legacy.username, legacy.key, legacy.token), ("u", "k", None))
        self.assertEqual(legacy.request_auth(), ({}, ("u", "k")))

        self.write("access_token", f"{TOKEN}\n")
        self.assertEqual(self.resolve().source, "~/.kaggle/access_token")

        self.assertEqual(self.resolve(secret="from-secret").token, "from-secret")

        os.environ["KAGGLE_API_TOKEN"] = "from-env"
        self.assertEqual(self.resolve(secret="from-secret").token, "from-env")

        chosen = self.resolve({"KAGGLE_API_TOKEN": "from-config"}, secret="from-secret")
        self.assertEqual(chosen.token, "from-config")
        self.assertEqual(chosen.request_auth(), ({"Authorization": "Bearer from-config"}, None))

    def test_token_may_be_a_file_path_and_is_never_shown(self):
        path = self.write("my_token", TOKEN)
        credential = self.resolve({"KAGGLE_API_TOKEN": path})
        self.assertEqual(credential.token, TOKEN)
        self.assertNotIn(TOKEN, repr(credential))

    def test_no_token_anywhere_is_a_clear_error(self):
        with self.assertRaisesRegex(kpa.KaggleProxyAuthError, "Kaggle Secret named\\s+KAGGLE_API_TOKEN"):
            self.resolve()
        with self.assertRaises(ValueError):
            self.resolve({"KAGGLE_API_TOKEN": 123})


class MintTest(unittest.TestCase):
    credential = kpa.KaggleCredential("test", token=TOKEN)

    def test_mint_calls_the_cli_endpoint_and_parses_the_key(self):
        post, calls = recording_post([FakeResponse(200, {
            "token": "proxy-key", "baseUri": "https://mp.kaggle.net/models/openapi",
            "expiryTime": "2030-01-01T00:00:00.123456789Z"})])
        key = kpa.mint_proxy_key(self.credential, post_fn=post)
        self.assertEqual(calls[0]["url"],
                         "https://api.kaggle.com/v1/models.ModelProxyApiService/CreateDefaultModelProxyToken")
        self.assertEqual(calls[0]["headers"], {"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual((key.url, key.api_key), ("https://mp.kaggle.net/models/openapi", "proxy-key"))
        self.assertAlmostEqual(key.expires_at, 1893456000.123456, places=3)
        self.assertNotIn("proxy-key", repr(key))

    def test_refusals_explain_themselves(self):
        for status, pattern in ((403, "verification"), (401, "wrong or expired"), (404, "beta"), (500, "HTTP 500")):
            post, _ = recording_post([FakeResponse(status, {})])
            with self.assertRaisesRegex(kpa.KaggleProxyAuthError, pattern):
                kpa.mint_proxy_key(self.credential, post_fn=post)
        post, _ = recording_post([FakeResponse(200, {"token": ""})])
        with self.assertRaisesRegex(kpa.KaggleProxyAuthError, "no model proxy key"):
            kpa.mint_proxy_key(self.credential, post_fn=post)

    def test_quota(self):
        post, calls = recording_post([FakeResponse(200, {"quotaBalances": []})])
        self.assertEqual(kpa.fetch_proxy_quota(self.credential, post_fn=post), {"quotaBalances": []})
        self.assertTrue(calls[0]["url"].endswith("/GetModelProxyQuotas"))


class FreshKeyTest(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.minted = []
        self.environ = {}

    def mint(self, credential):
        self.minted.append(self.now)
        return kpa.ProxyKey(url="https://proxy", api_key=f"key{len(self.minted)}", expires_at=self.now + 3600)

    def credentials(self, refresh=300, default_model=None):
        return kpa.KaggleProxyCredentials(kpa.KaggleCredential("test", token=TOKEN), refresh, default_model,
                                          mint_fn=self.mint, clock=lambda: self.now, environ=self.environ)

    def test_mints_once_then_before_expiry(self):
        creds = self.credentials(default_model="openai/gpt-6-luna")
        self.assertEqual(creds.ensure_fresh(), 1)
        self.assertEqual(self.environ["MODEL_PROXY_API_KEY"], "key1")
        self.assertEqual(self.environ["MODEL_PROXY_URL"], "https://proxy")
        self.assertEqual(self.environ["LLM_DEFAULT"], "openai/gpt-6-luna")
        self.assertIn("MODEL_PROXY_EXPIRY_TIME", self.environ)
        self.now += 3000  # 600 s left: still fine
        self.assertEqual(creds.ensure_fresh(), 1)
        self.now += 400   # 200 s left: renewed
        self.assertEqual(creds.ensure_fresh(), 2)
        self.assertEqual(self.environ["MODEL_PROXY_API_KEY"], "key2")
        self.assertEqual(len(self.minted), 2)
        self.assertNotIn("key2", creds.describe())

    def test_a_rejection_mints_one_new_key_per_generation(self):
        creds = self.credentials()
        creds.ensure_fresh()
        for _ in range(5):  # five calls failed on key 1
            creds.invalidate(1)
        self.assertEqual(creds.ensure_fresh(), 2)
        creds.invalidate(1)  # a late report about the old key
        self.assertEqual(creds.ensure_fresh(), 2)
        self.assertEqual(len(self.minted), 2)

    def test_an_existing_default_model_is_kept_and_settings_are_checked(self):
        self.environ["LLM_DEFAULT"] = "notebook/model"
        self.credentials().ensure_fresh()
        self.assertEqual(self.environ["LLM_DEFAULT"], "notebook/model")
        with self.assertRaises(ValueError):
            kpa.api_token_enabled({"KAGGLE_PROXY_USE_API_TOKEN": "yes"})
        with self.assertRaises(ValueError):
            kpa.KaggleProxyCredentials.from_config({"KAGGLE_API_TOKEN": TOKEN,
                                                    "KAGGLE_PROXY_TOKEN_REFRESH_SECONDS": "300"})


class ManagerWithApiTokenTest(_Quiet):
    """The Kaggle provider with KAGGLE_PROXY_USE_API_TOKEN=True, on a fake kaggle_benchmarks."""

    def setUp(self):
        super().setUp()
        self.kbench, self.state = fake_kbench()
        self.built_with = []
        self.keys = iter(f"key{n}" for n in range(1, 100))
        kbench_llm = type(self.kbench.llms["google/gemini-2.5-flash"])

        def load_model(name):  # records the proxy key each client is built with
            self.built_with.append((name, os.environ["MODEL_PROXY_API_KEY"]))
            return kbench_llm()

        self.kbench.kaggle.load_model = load_model
        for patcher in (patch.dict(sys.modules, {"kaggle_benchmarks": self.kbench}),
                        patch.dict(os.environ, {}, clear=False),
                        patch.object(kpa, "mint_proxy_key", lambda credential: kpa.ProxyKey(
                            "https://proxy", next(self.keys), None))):
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ("MODEL_PROXY_URL", "MODEL_PROXY_API_KEY", "LLM_DEFAULT"):
            os.environ.pop(name, None)
        self.config = {**BATCH_CONFIG, "API_BATCHING_ENABLED": False, "KAGGLE_PROXY_USE_API_TOKEN": True,
                       "KAGGLE_API_TOKEN": TOKEN, "KAGGLE_MODEL_NAME_FINAL_SOLVER": "openai/gpt-6-luna"}
        with patch("builtins.print"):
            self.manager = api_manager.KaggleBenchmarksAPIManager({"default": {"rpm": 1000}}, self.config)

    def test_key_is_installed_before_the_sdk_is_used(self):
        self.assertEqual(os.environ["MODEL_PROXY_API_KEY"], "key1")
        self.assertEqual(os.environ["LLM_DEFAULT"], "openai/gpt-6-luna")
        result = self.manager.generate_content("q", "google/gemini-2.5-flash")
        self.assertEqual(result["text"], "answer:q")
        # The import-time kbench.llms client is skipped: it may carry another key.
        self.assertEqual(self.built_with, [("google/gemini-2.5-flash", "key1")])

    def test_a_rejected_key_is_replaced_and_clients_rebuilt(self):
        self.manager.generate_content("q", "m")
        denied = self.manager._failure("openai.AuthenticationError: 401 expired", {}, "m", generation=1)
        self.assertEqual(denied["error_type"], "ProxyAuthenticationError")  # retried, key not disabled
        self.manager.generate_content("q", "m")
        self.assertEqual(self.built_with, [("m", "key1"), ("m", "key2")])
        self.manager.generate_content("q", "m")
        self.assertEqual(len(self.built_with), 2, "one client per key")

    def test_batched_calls_use_the_minted_key(self):
        bound = self.manager.for_config({**self.config, "API_BATCHING_ENABLED": True,
                                         "API_BATCH_KAGGLE_N_JOBS": 2})
        results = concurrently(*[(lambda prompt=prompt: bound.generate_content(prompt, "m"))
                                 for prompt in ("a", "b")])
        self.assertEqual([r["text"] for r in results], ["answer:a", "answer:b"])
        self.assertEqual(self.built_with, [("m", "key1")])

    def test_off_by_default_and_kept_out_of_the_run_manifest(self):
        from config import CONFIG
        from src.distributed_execution import sanitize_scientific_config

        self.assertFalse(CONFIG["KAGGLE_PROXY_USE_API_TOKEN"])
        self.assertEqual(CONFIG["KAGGLE_API_TOKEN"], "")
        # Where the key comes from never changes a result: a run keeps its manifest
        # (and so its run ID) when the switch is turned on to resume it elsewhere.
        on = {**CONFIG, "API_PROVIDER_FINAL_SOLVER": "kaggle", **self.config,
              "KAGGLE_PROXY_TOKEN_REFRESH_SECONDS": 60}
        off = {key: value for key, value in on.items() if not key.startswith(("KAGGLE_PROXY_", "KAGGLE_API_TOKEN"))}
        self.assertEqual(sanitize_scientific_config(on), sanitize_scientific_config(off))
        self.assertEqual(sanitize_scientific_config(on)["API_PROVIDER_FINAL_SOLVER"], "kaggle")


if __name__ == "__main__":
    unittest.main()
