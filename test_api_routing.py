"""Per-provider/per-model API limits and multi-endpoint provider pools."""

import threading
import time
from unittest.mock import patch

import pytest

from src import api_manager, api_routing
from src.context_logger import track_api_failures
from src.distributed_execution import sanitize_scientific_config

PROVIDERS = api_manager.SUPPORTED_API_PROVIDERS


@pytest.fixture(autouse=True)
def quiet_and_fresh_state():
    with patch.object(api_manager, "tprint"), \
            patch.object(api_manager, "global_pause_manager", api_manager._GlobalAPIPauseManager()), \
            patch.object(api_manager, "provider_pacer", api_manager._ProviderRatePacer()), \
            patch.object(api_manager, "concurrency_limiter", api_manager._ConcurrencyLimiter()):
        yield


# --- Scoped settings -------------------------------------------------------

def test_a_plain_value_applies_to_every_provider_and_model():
    assert api_routing.resolve_scoped(5.0, "gemini", "m") == (5.0, False)
    assert api_routing.resolve_scoped(None, "gemini") == (None, False)


def test_the_most_specific_entry_wins():
    value = {"openrouter": {"default": 1.0, "fast/model": 3.0}, "gemini": 5.0, "default": 0.5}
    assert api_routing.resolve_scoped(value, "openrouter", "fast/model") == (3.0, True)
    assert api_routing.resolve_scoped(value, "openrouter", "other/model") == (1.0, False)
    assert api_routing.resolve_scoped(value, "gemini", "any") == (5.0, False)
    assert api_routing.resolve_scoped(value, "kaggle", "any") == (0.5, False)
    # A provider dict without its own default falls back to the top-level one.
    assert api_routing.resolve_scoped({"openrouter": {"a": 2}, "default": 7}, "openrouter", "b") == (7, False)
    # The former per-provider dict keeps its meaning: a missing provider gets the built-in default.
    assert api_routing.scoped_value({"X": {"gemini": 5.0}}, "X", "kaggle", default=0.0) == 0.0


def test_scoped_settings_reject_unknown_providers_and_misplaced_model_entries():
    api_routing.validate_scoped_settings(
        {"GLOBAL_API_CALL_DELAY_SECONDS": {"gemini": 5, "openrouter": {"default": 1, "m": 2}, "default": 0},
         "API_REQUEST_TIMEOUT_SECONDS": {"kaggle": 120}, "MAX_API_RETRIES": 200},
        PROVIDERS,
    )
    with pytest.raises(ValueError, match="unknown provider 'openroutr'"):
        api_routing.validate_scoped_settings({"GLOBAL_API_CALL_DELAY_SECONDS": {"openroutr": 1}}, PROVIDERS)
    with pytest.raises(ValueError, match="per provider, not per model"):
        api_routing.validate_scoped_settings({"API_REQUEST_TIMEOUT_SECONDS": {"kaggle": {"m": 1}}}, PROVIDERS)
    with pytest.raises(ValueError, match="must be numbers"):
        api_routing.validate_scoped_settings({"MAX_API_RETRIES": {"kaggle": {"m": "5"}}}, PROVIDERS)


def test_a_model_with_its_own_gap_gets_its_own_clock():
    config = {"GLOBAL_API_CALL_DELAY_SECONDS": {"openrouter": {"default": 4.0, "slow/model": 9.0}}}
    with patch.object(api_manager.provider_pacer, "pace") as pace:
        api_manager._pace_provider_call(config, "openrouter", "slow/model")
        api_manager._pace_provider_call(config, "openrouter", "other/model")
        api_manager._pace_provider_call({"GLOBAL_API_CALL_DELAY_SECONDS": 3}, "gemini", "m")
        api_manager._pace_provider_call({"GLOBAL_API_CALL_DELAY_SECONDS": 0.1, "ENABLE_MIN_TIME_BETWEEN_API_CALLS": True,
                                         "MIN_TIME_BETWEEN_API_CALLS_SECONDS": 0.5}, "kaggle", "m")
    assert [call.args for call in pace.call_args_list] == [
        (("openrouter", "slow/model"), 9.0), ("openrouter", 4.0), ("gemini", 3.0), ("kaggle", 0.5),
    ]


def test_two_providers_are_paced_on_separate_clocks():
    config = {"GLOBAL_API_CALL_DELAY_SECONDS": 0.4}
    api_manager._pace_provider_call(config, "gemini")
    started = time.monotonic()
    api_manager._pace_provider_call(config, "openrouter")  # Another provider: no wait.
    assert time.monotonic() - started < 0.2
    api_manager._pace_provider_call(config, "gemini")  # Same provider: waits out the gap.
    assert time.monotonic() - started >= 0.3


def _peak_concurrency(config, routes, hold=0.15):
    lock, running, peak = threading.Lock(), {}, {}

    def call(provider, model):
        with api_manager.concurrency_limiter.slot(config, provider, model):
            with lock:
                running[provider] = running.get(provider, 0) + 1
                peak[provider] = max(peak.get(provider, 0), running[provider])
            time.sleep(hold)
            with lock:
                running[provider] -= 1

    threads = [threading.Thread(target=call, args=route) for route in routes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return peak


def test_max_concurrent_calls_limit_one_provider_without_blocking_another():
    config = {"API_MAX_CONCURRENT_CALLS": {"gemini": 1}}
    peak = _peak_concurrency(config, [("gemini", "m")] * 3 + [("openrouter", "m")] * 3)
    assert peak == {"gemini": 1, "openrouter": 3}
    assert _peak_concurrency({"API_MAX_CONCURRENT_CALLS": None}, [("gemini", "m")] * 3) == {"gemini": 3}


def test_retry_settings_are_read_per_provider():
    config = {"ENABLE_API_RETRY": True, "MAX_API_RETRIES": {"default": 4, "openrouter": 2},
              "API_RETRY_DELAY_SECONDS": 0.0, "RETRY_ALL_API_ERRORS": True}
    for provider, expected in (("openrouter", 2), ("gemini", 4)):
        calls = []
        api_manager.execute_with_retry(config, provider, "m", "p", lambda: calls.append(1) or
                                       {"status": "ERROR", "error_type": "APIConnectionError", "error_message": "down"})
        assert len(calls) == expected


# --- Provider pools --------------------------------------------------------

class FakeManager(api_manager._ConfigBindable):
    """A provider whose replies are scripted per model; it records each call."""

    def __init__(self, provider, script=None, hold=None):
        self.provider_name = provider
        self.config = {}
        self.script = script or {}
        self.calls = []
        self.hold = hold

    def generate_content(self, prompt, model_name, temperature=None, avalai_role=None):
        def attempt():
            self.calls.append(model_name)
            if self.hold:
                self.hold.wait(timeout=2)
            replies = self.script.get(model_name)
            reply = replies.pop(0) if replies else {"status": "SUCCESS", "text": f"{self.provider_name}:{model_name}"}
            return {**reply, "request_meta": {"provider": self.provider_name, "model": model_name}}
        return api_manager.execute_with_retry(self.config, self.provider_name, model_name, prompt, attempt)


DOWN = {"status": "ERROR", "error_type": "APIConnectionError", "error_message": "down"}
BASE = {"ENABLE_API_RETRY": True, "MAX_API_RETRIES": 3, "API_RETRY_DELAY_SECONDS": 0.0,
        "RETRY_ALL_API_ERRORS": True, "API_PROVIDER_EVALUATOR": "judges"}


def _pool(endpoints, routing="spread", **config):
    managers = {"gemini": FakeManager("gemini"), "openrouter": FakeManager("openrouter")}
    full = {**BASE, "API_PROVIDER_POOLS": {"judges": {"routing": routing, "endpoints": endpoints}}, **config}
    api_manager.attach_provider_pools(managers, full)
    return managers["judges"].for_config(full), managers


ENDPOINTS = [{"provider": "gemini", "model": "g"}, {"provider": "openrouter", "model": "o"}]


def test_a_role_selects_a_pool_like_a_provider_and_logs_its_label():
    pool, managers = _pool(ENDPOINTS)
    assert isinstance(managers["judges"], api_manager.ProviderPool)
    assert api_manager.model_name_for(pool, pool.config, "evaluator") == "pool:judges"
    response = pool.generate_content("p", "pool:judges", 0.0, avalai_role="evaluator")
    assert response["status"] == "SUCCESS"
    assert response["request_meta"] == {"provider": "gemini", "model": "g", "pool": "judges"}


def test_spread_rotates_calls_by_weight():
    pool, managers = _pool([{"provider": "gemini", "model": "g", "weight": 2}, {"provider": "openrouter", "model": "o"}])
    served = [pool.generate_content("p")["request_meta"]["provider"] for _ in range(6)]
    assert served.count("gemini") == 4 and served.count("openrouter") == 2


def test_concurrent_calls_spread_over_providers():
    release = threading.Event()
    pool, managers = _pool(ENDPOINTS)
    for manager in (managers["gemini"], managers["openrouter"]):
        manager.hold = release
    threads = [threading.Thread(target=pool.generate_content, args=("p",)) for _ in range(2)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 2
    while len(managers["gemini"].calls) + len(managers["openrouter"].calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    release.set()
    for thread in threads:
        thread.join()
    assert managers["gemini"].calls == ["g"] and managers["openrouter"].calls == ["o"]


def test_failover_uses_the_first_endpoint_while_it_works():
    pool, managers = _pool(ENDPOINTS, routing="failover")
    for _ in range(3):
        pool.generate_content("p")
    assert managers["gemini"].calls == ["g"] * 3 and managers["openrouter"].calls == []


def test_a_failed_attempt_is_retried_on_another_endpoint_and_does_not_fail_the_question():
    pool, managers = _pool(ENDPOINTS, routing="failover")
    managers["gemini"].script = {"g": [dict(DOWN)]}
    with track_api_failures() as failures:
        response = pool.generate_content("p")
    assert response["status"] == "SUCCESS" and response["request_meta"]["provider"] == "openrouter"
    assert failures == []
    assert managers["gemini"].calls == ["g"]  # Retried elsewhere, not on gemini again.


def test_a_failing_endpoint_is_avoided_for_its_retry_delay():
    pool, managers = _pool(ENDPOINTS, routing="failover", API_RETRY_DELAY_SECONDS={"gemini": 60.0, "default": 0.0})
    managers["gemini"].script = {"g": [dict(DOWN)]}
    for _ in range(3):
        pool.generate_content("p")
    assert managers["gemini"].calls == ["g"] and managers["openrouter"].calls == ["o"] * 3


def test_every_endpoint_spends_its_own_retry_budget_before_the_call_fails():
    pool, managers = _pool(ENDPOINTS, MAX_API_RETRIES={"gemini": 2, "openrouter": 3})
    managers["gemini"].script = {"g": [dict(DOWN) for _ in range(5)]}
    managers["openrouter"].script = {"o": [dict(DOWN) for _ in range(5)]}
    with track_api_failures() as failures:
        response = pool.generate_content("p")
    assert response["status"] == "ERROR" and response["retry_exhausted"]
    assert len(managers["gemini"].calls) == 2 and len(managers["openrouter"].calls) == 3
    assert len(failures) == 1 and failures[0]["error_type"] == "APIConnectionError"


def test_an_endpoint_that_cannot_serve_the_call_is_retired_for_it():
    pool, managers = _pool(ENDPOINTS, routing="failover")
    managers["gemini"].script = {"g": [{"status": "ERROR", "error_type": "InvalidModel", "error_message": "no such model"}]}
    response = pool.generate_content("p")
    assert response["status"] == "SUCCESS" and managers["gemini"].calls == ["g"]


def test_pools_are_validated_before_the_run():
    managers = {"gemini": FakeManager("gemini")}
    with pytest.raises(ValueError, match=r"provider\(s\) \['openrouter'\] that have no API manager"):
        api_manager.attach_provider_pools(managers, {"API_PROVIDER_POOLS": {"judges": ENDPOINTS},
                                                     "API_PROVIDER_EVALUATOR": "judges"})
    for pools, message in (
        ({"gemini": ENDPOINTS}, "has a provider's name"),
        ({"judges": [{"provider": "gemini"}]}, "needs a 'model'"),
        ({"judges": [{"provider": "gemnii", "model": "g"}]}, "unknown provider 'gemnii'"),
        ({"judges": {"routing": "random", "endpoints": ENDPOINTS}}, "routing must be one of"),
        ({"judges": [ENDPOINTS[0], ENDPOINTS[0]]}, "repeats gemini/g"),
    ):
        with pytest.raises(ValueError, match=message):
            api_manager.attach_provider_pools({}, {"API_PROVIDER_POOLS": pools, "API_PROVIDER_EVALUATOR": "judges"})
    # A pool no role selects is not checked and needs no managers.
    assert "judges" not in api_manager.attach_provider_pools(dict(managers), {"API_PROVIDER_POOLS": {"judges": ENDPOINTS}})


def test_pool_members_count_as_used_providers():
    config = {"API_PROVIDER_SOLVER": "gemini", "API_PROVIDER_EVALUATOR": "judges",
              "API_PROVIDER_POOLS": {"judges": [{"provider": "openrouter", "model": "o"}, {"provider": "kaggle", "model": "k"}]}}
    assert api_routing.providers_in_use(config) == {"gemini", "openrouter", "kaggle"}
    sanitized = sanitize_scientific_config({**config, "OPENROUTER_MODEL_ROUTING": {"o": {"only": ["x"]}},
                                            "KAGGLE_REASONING_EFFORT": "low"})
    assert sanitized["OPENROUTER_MODEL_ROUTING"] == {"o": {"only": ["x"]}}
    assert sanitized["KAGGLE_REASONING_EFFORT"] == "low"
    assert sanitized["API_PROVIDER_POOLS"]["judges"][0] == {"model": "o", "provider": "openrouter"}


def test_new_settings_leave_existing_manifests_unchanged():
    before = {"API_PROVIDER_SOLVER": "gemini", "GEMINI_MODEL_NAME_FINAL_SOLVER": "m"}
    after = {**before, "API_PROVIDER_POOLS": {}, "API_MAX_CONCURRENT_CALLS": {"gemini": 2},
             "LAYER1_GROUPING_TOP_N": None}
    assert sanitize_scientific_config(after) == sanitize_scientific_config(before)
    kaggle = {"API_PROVIDER_SOLVER": "kaggle", "KAGGLE_REASONING_EFFORT": None}
    assert (sanitize_scientific_config({**kaggle, "KAGGLE_REASONING_EFFORT_BY_MODEL": {}})
            == sanitize_scientific_config(kaggle))
    # Set values are scientific settings.
    assert sanitize_scientific_config({**before, "LAYER1_GROUPING_TOP_N": 3})["LAYER1_GROUPING_TOP_N"] == 3
    limited = sanitize_scientific_config({**kaggle, "KAGGLE_REASONING_EFFORT_BY_MODEL": {"openai/": "none"}})
    assert limited["KAGGLE_REASONING_EFFORT_BY_MODEL"] == {"openai/": "none"}


def test_the_run_preflight_registers_pools_and_reports_their_batching(capsys):
    import sys
    from types import ModuleType

    stub = ModuleType("sentence_transformers")
    stub.SentenceTransformer = type("SentenceTransformer", (), {})
    with patch.dict(sys.modules, {"sentence_transformers": stub}):
        from src import orchestration

    managers = {"gemini": FakeManager("gemini"), "openrouter": FakeManager("openrouter")}
    config = {**BASE, "API_PROVIDER_POOLS": {"judges": ENDPOINTS}, "API_BATCHING_ENABLED": {"openrouter": True}}
    orchestration._check_api_routing([{"experiment_name": "e"}], config, managers)
    orchestration._check_api_batching([{"experiment_name": "e"}], config, managers)
    printed = capsys.readouterr().out
    assert "evaluator -> 'judges' (spread): gemini/g, openrouter/o" in printed
    assert "evaluator openrouter/o batched" in printed
    with pytest.raises(ValueError, match="Experiment 'bad'.*unknown provider 'gemni'"):
        orchestration._check_api_routing(
            [{"experiment_name": "bad", "GLOBAL_API_CALL_DELAY_SECONDS": {"gemni": 1}}], config, managers)
    with pytest.raises(ValueError, match="Experiment 'k'.*KAGGLE_REASONING_EFFORT_BY_MODEL"):
        orchestration._check_api_routing(
            [{"experiment_name": "k", "API_PROVIDER_SOLVER": "kaggle",
              "KAGGLE_REASONING_EFFORT_BY_MODEL": {"openai/": "nope"}}], config, managers)
