"""Thread-safe API managers with per-key rolling RPM scheduling.

Every remote request is reserved by a shared scheduler before it is sent.  The
scheduler is scoped to ``(provider, key, model)`` so a temporary failure on one
credential never blocks healthy credentials.  The public ``generate_content``
interface remains compatible with the rest of the pipeline.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple, TypedDict, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from src.context_logger import ctx_query_idx, ctx_batch_id, tprint

try:  # Keep scheduler/batch tests importable without optional provider SDKs.
    import ollama
except ModuleNotFoundError:  # pragma: no cover - depends on deployment extras
    ollama = None
try:
    import openai
except ModuleNotFoundError:  # pragma: no cover - depends on deployment extras
    openai = None


RETRYABLE_ERROR_TYPES = {
    "APITimeoutError", "APIConnectionError", "ResourceExhausted",
    "OllamaConnectionError", "UnknownError", "APIStatusError",
    "RateLimitError", "ProactiveRateLimit",
}
NON_RETRYABLE_ERROR_TYPES = {"AuthenticationError", "InvalidArgument", "Safety", "NoChoices", "ModelMismatch"}


class APIResponse(TypedDict, total=False):
    status: str
    text: Optional[str]
    error_type: Optional[str]
    error_message: Optional[str]
    error_details: Optional[Any]
    request_meta: Dict[str, Any]


def _mask_key(api_key: str) -> str:
    return f"…{api_key[-4:]}" if api_key else "<missing>"


@dataclass(frozen=True)
class KeyLease:
    api_key: str
    model_name: str
    rolling_requests: int
    daily_requests: int


class RPMKeyScheduler:
    """A condition-protected, fair rolling-60-second scheduler for API keys."""

    WINDOW_SECONDS = 60.0

    def __init__(
        self,
        api_keys: List[str],
        quota_for_key: Callable[[str, str], Dict[str, Any]],
        config: Dict[str, Any],
        provider_name: str,
    ) -> None:
        self.api_keys = list(api_keys)
        self.quota_for_key = quota_for_key
        self.config = config
        self.provider_name = provider_name
        self._condition = threading.Condition(threading.RLock())
        self._recent: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)
        self._daily: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self._cooldown_until: Dict[Tuple[str, str], float] = defaultdict(float)
        self._disabled: Dict[str, str] = {}
        self._tie_breaker = 0

    @staticmethod
    def _today() -> str:
        return datetime.utcnow().strftime("%Y-%m-%d")

    def _rpm(self, quota: Dict[str, Any]) -> int:
        rpm = quota.get("rpm", self.config.get("DEFAULT_API_RPM"))
        if rpm is None:
            # Legacy configurations used delay_seconds.  This conversion is only
            # a compatibility fallback; new batch configurations should set rpm.
            delay = float(quota.get("delay_seconds", 1.0))
            rpm = max(1, int(self.WINDOW_SECONDS / max(delay, 1.0)))
        return max(1, int(rpm))

    def acquire(self, model_name: str) -> Optional[KeyLease]:
        """Reserve the least-used ready key, waiting only for the necessary key."""
        while True:
            with self._condition:
                now = time.monotonic()
                today = self._today()
                ready: List[Tuple[int, int, int, str, int]] = []
                waits: List[float] = []

                for position, api_key in enumerate(self.api_keys):
                    if api_key in self._disabled:
                        continue
                    quota = self.quota_for_key(model_name, api_key)
                    key_model = (api_key, model_name)
                    timestamps = self._recent[key_model]
                    while timestamps and now - timestamps[0] >= self.WINDOW_SECONDS:
                        timestamps.popleft()

                    rpd = quota.get("rpd", float("inf"))
                    daily_key = (api_key, model_name, today)
                    if self._daily[daily_key] >= rpd:
                        continue

                    cooldown = self._cooldown_until[key_model]
                    rpm = self._rpm(quota)
                    if now < cooldown:
                        waits.append(cooldown - now)
                        continue
                    if len(timestamps) >= rpm:
                        waits.append(max(0.001, timestamps[0] + self.WINDOW_SECONDS - now))
                        continue

                    # Favour low rolling usage first, then low daily usage. The
                    # rotating final key prevents one identical key from winning ties.
                    tie = (position - self._tie_breaker) % max(1, len(self.api_keys))
                    ready.append((len(timestamps), self._daily[daily_key], tie, api_key, position))

                if ready:
                    ready.sort()
                    rolling, daily, _, selected_key, position = ready[0]
                    self._recent[(selected_key, model_name)].append(now)
                    self._daily[(selected_key, model_name, today)] += 1
                    self._tie_breaker = (position + 1) % max(1, len(self.api_keys))
                    return KeyLease(selected_key, model_name, rolling + 1, daily + 1)

                if not waits:
                    return None  # Every key is disabled or has exhausted RPD.

                self._condition.wait(timeout=max(0.001, min(waits)))

    def cooldown(self, api_key: str, model_name: str, seconds: Optional[float] = None) -> None:
        delay = float(seconds if seconds is not None else self.config.get("API_KEY_ERROR_COOLDOWN_SECONDS", 20.0))
        with self._condition:
            key = (api_key, model_name)
            self._cooldown_until[key] = max(self._cooldown_until[key], time.monotonic() + max(0.0, delay))
            # BUG FIX: Removed self._condition.notify_all() here. 
            # We don't wake up threads when taking a resource away!

    def disable(self, api_key: str, reason: str) -> None:
        with self._condition:
            self._disabled[api_key] = reason

    def snapshot(self) -> Dict[str, Any]:
        with self._condition:
            now = time.monotonic()
            return {
                "provider": self.provider_name,
                "disabled_keys": {_mask_key(k): v for k, v in self._disabled.items()},
                "cooldowns": {
                    f"{_mask_key(key)}:{model}": round(max(0.0, until - now), 3)
                    for (key, model), until in self._cooldown_until.items() if until > now
                },
            }


def execute_with_retry(config: Dict[str, Any], api_call_func: Callable[[], APIResponse]) -> APIResponse:
    """Retry without terminating the process; managers re-enter their scheduler each time."""
    logger = logging.getLogger(__name__)
    if not config.get("ENABLE_API_RETRY", False):
        return api_call_func()

    max_attempts = max(1, int(config.get("MAX_API_RETRIES", 3)))
    retry_all = config.get("RETRY_ALL_API_ERRORS", True)
    last_response: Optional[APIResponse] = None
    for attempt in range(1, max_attempts + 1):
        response = api_call_func()
        if response["status"] == "SUCCESS":
            return response

        error_type = response.get("error_type") or response.get("status", "ERROR")
        retryable = error_type not in NON_RETRYABLE_ERROR_TYPES and (retry_all or error_type in RETRYABLE_ERROR_TYPES)
        if not retryable or attempt == max_attempts:
            response["retry_exhausted"] = attempt == max_attempts
            return response

        # NEW: Uses tprint so the warning shows up on the console with the Q# tag!
        tprint(f"API attempt {attempt}/{max_attempts} failed ({error_type}); retrying with next available key.", level="WARNING")
        # In batch mode a failed key was put into its own cooldown by the manager.
        # Do not impose the old global retry sleep on unrelated keys.
        if not config.get("BATCH_PROCESSING_ENABLED", False):
            time.sleep(float(config.get("API_RETRY_DELAY_SECONDS", 5.0)))
        last_response = response
    return last_response or {"status": "ERROR", "text": None, "error_type": "RetryError", "error_message": "No API attempt was made", "error_details": None}


class _KeyedAPIManager:
    provider_name = "remote"

    def __init__(self, api_keys: List[str], model_quotas: Dict[str, Any], config: Optional[Dict[str, Any]]) -> None:
        if not api_keys:
            raise ValueError("API keys list cannot be empty.")
        self.logger = logging.getLogger(__name__)
        self.api_keys_list = list(api_keys)
        self.model_quotas = model_quotas or {}
        self.config = config or {}
        self.print_details = self.config.get("PRINT_API_CALL_DETAILS", False)
        self.truncation_length = self.config.get("API_RESPONSE_TRUNCATION_LENGTH", 50)
        self.scheduler = RPMKeyScheduler(self.api_keys_list, self._get_quota_for_key, self.config, self.provider_name)

    def _get_quota_for_key(self, model_name: str, api_key: str) -> Dict[str, Any]:
        model_entry = self.model_quotas.get(model_name, self.model_quotas.get("default", {}))
        if isinstance(model_entry, dict):
            return model_entry
        if isinstance(model_entry, list):
            default: Dict[str, Any] = {}
            for quota in model_entry:
                target = quota.get("api_key")
                if target == api_key or (isinstance(target, (list, tuple, set)) and api_key in target):
                    return quota
                if target is None:
                    default = quota
            return default
        return {}

    def _request_meta(self, lease: KeyLease) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": lease.model_name,
            "key": _mask_key(lease.api_key),
            "rolling_requests": lease.rolling_requests,
            "daily_requests": lease.daily_requests,
            # NEW: Grab the query ID from the context engine
            "query_idx": ctx_query_idx.get(),
            "batch_id": ctx_batch_id.get(),
        }

    def _print(self, message: str) -> None:
        if self.print_details:
            # NEW: Use tprint with DEBUG level. 
            # This hides the spam from the console but saves it to the log file!
            tprint(f"[API {self.provider_name}] {message}", level="DEBUG")


class GeminiAPIManager(_KeyedAPIManager):
    """Gemini manager using REST requests so every call is bound to its selected key."""

    provider_name = "gemini"

    def __init__(self, api_keys: List[str], model_quotas: Dict[str, Any], global_delay_seconds: int = 0, config: Optional[Dict[str, Any]] = None):
        super().__init__(api_keys, model_quotas, config)
        self.timeout_seconds = float(self.config.get("API_REQUEST_TIMEOUT_SECONDS", 180.0))
        if global_delay_seconds:
            self.logger.info("GLOBAL_API_CALL_DELAY_SECONDS is ignored by the RPM scheduler.")

    def _generation_config(self, model_name: str, temperature: Optional[float]) -> Dict[str, Any]:
        max_tokens: Optional[int] = None
        if model_name == self.config.get("GEMINI_MODEL_NAME_FINAL_SOLVER"):
            max_tokens = self.config.get("DEFAULT_FINAL_SOLVER_MAX_TOKENS", 8192)
        elif model_name == self.config.get("GEMINI_MODEL_NAME_ADAPTATION"):
            max_tokens = self.config.get("DEFAULT_ADAPTATION_MAX_TOKENS", 2048)
        elif model_name == self.config.get("GEMINI_MODEL_NAME_EVALUATOR"):
            max_tokens = self.config.get("DEFAULT_EVALUATOR_MAX_TOKENS", 512)
        result: Dict[str, Any] = {}
        if temperature is not None:
            result["temperature"] = temperature
        if max_tokens is not None:
            result["maxOutputTokens"] = max_tokens
        return result

    def _call_rest(self, api_key: str, prompt: str, model_name: str, generation_config: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": generation_config}
        request = Request(
            url, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return int(response.status), json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            try:
                return error.code, json.loads(body)
            except json.JSONDecodeError:
                return error.code, {"error": {"message": body}}
        except (URLError, TimeoutError) as error:
            return 0, {"error": {"message": str(error), "type": "ConnectionError"}}

    def generate_content(self, prompt: str, model_name: str, temperature: Optional[float] = None) -> APIResponse:
        self._print(f"request model={model_name} prompt={prompt[:self.truncation_length]!r}")

        def attempt() -> APIResponse:
            lease = self.scheduler.acquire(model_name)
            if lease is None:
                return {"status": "RATE_LIMITED", "text": None, "error_type": "ProactiveRateLimit", "error_message": f"No eligible Gemini key for {model_name}.", "error_details": self.scheduler.snapshot()}
            meta = self._request_meta(lease)
            status_code, payload = self._call_rest(lease.api_key, prompt, model_name, self._generation_config(model_name, temperature))
            if 200 <= status_code < 300:
                candidates = payload.get("candidates", [])
                parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
                text = "".join(part.get("text", "") for part in parts)
                if text:
                    self._print(f"success model={model_name} key={meta['key']}")
                    return {"status": "SUCCESS", "text": text, "error_type": None, "error_message": None, "error_details": None, "request_meta": meta}
                return {"status": "BLOCKED", "text": None, "error_type": "Safety", "error_message": "Gemini returned no text.", "error_details": payload.get("promptFeedback"), "request_meta": meta}

            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            message = error.get("message", str(payload))
            if status_code == 429:
                self.scheduler.cooldown(lease.api_key, model_name)
                error_type, api_status = "ResourceExhausted", "RATE_LIMITED"
            elif status_code in (401, 403):
                self.scheduler.disable(lease.api_key, f"HTTP {status_code}")
                error_type, api_status = "AuthenticationError", "ERROR"
            elif status_code == 400:
                error_type, api_status = "InvalidArgument", "ERROR"
            else:
                self.scheduler.cooldown(lease.api_key, model_name)
                error_type, api_status = ("APIConnectionError" if status_code == 0 else "UnknownError"), "ERROR"
            self._print(f"failure model={model_name} key={meta['key']} type={error_type}")
            return {"status": api_status, "text": None, "error_type": error_type, "error_message": message, "error_details": payload, "request_meta": meta}

        return execute_with_retry(self.config, attempt)


class AvalAIAPIManager(_KeyedAPIManager):
    provider_name = "avalai"

    def __init__(self, api_key_or_list: Union[str, List[str]], base_url: str, model_quotas: Dict[str, Any], global_delay_seconds: int = 0, config: Optional[Dict[str, Any]] = None):
        if openai is None:
            raise ImportError("AvalAIAPIManager requires the optional 'openai' package.")
        keys = [api_key_or_list] if isinstance(api_key_or_list, str) else list(api_key_or_list)
        if not base_url:
            raise ValueError("base_url cannot be empty for AvalAIAPIManager.")
        super().__init__(keys, model_quotas, config)
        self.clients = {key: openai.OpenAI(api_key=key, base_url=base_url) for key in self.api_keys_list}
        if global_delay_seconds:
            self.logger.info("GLOBAL_API_CALL_DELAY_SECONDS is ignored by the RPM scheduler.")

    def generate_content(self, prompt: str, model_name: str, temperature: Optional[float] = None) -> APIResponse:
        self._print(f"request model={model_name} prompt={prompt[:self.truncation_length]!r}")

        def attempt() -> APIResponse:
            lease = self.scheduler.acquire(model_name)
            if lease is None:
                return {"status": "RATE_LIMITED", "text": None, "error_type": "ProactiveRateLimit", "error_message": f"No eligible AvalAI key for {model_name}.", "error_details": self.scheduler.snapshot()}
            meta = self._request_meta(lease)
            try:
                completion = self.clients[lease.api_key].chat.completions.create(
                    model=model_name, messages=[{"role": "user", "content": prompt}], temperature=temperature
                )
                returned_model = getattr(completion, "model", None)
                if returned_model and returned_model != model_name:
                    return {"status": "ERROR", "text": None, "error_type": "ModelMismatch", "error_message": f"Requested {model_name!r}; provider returned {returned_model!r}.", "error_details": {"requested": model_name, "returned": returned_model}, "request_meta": meta}
                choices = getattr(completion, "choices", [])
                text = choices[0].message.content if choices else None
                if not text:
                    return {"status": "BLOCKED", "text": None, "error_type": "NoChoices", "error_message": "Provider returned no completion choices.", "error_details": None, "request_meta": meta}
                self._print(f"success model={model_name} key={meta['key']}")
                return {"status": "SUCCESS", "text": text, "error_type": None, "error_message": None, "error_details": None, "request_meta": meta}
            except openai.RateLimitError as error:
                self.scheduler.cooldown(lease.api_key, model_name)
                return {"status": "RATE_LIMITED", "text": None, "error_type": "RateLimitError", "error_message": str(error), "error_details": repr(error), "request_meta": meta}
            except Exception as error:
                auth_errors = tuple(cls for cls in (getattr(openai, "AuthenticationError", None), getattr(openai, "PermissionDeniedError", None)) if cls)
                if auth_errors and isinstance(error, auth_errors):
                    self.scheduler.disable(lease.api_key, type(error).__name__)
                else:
                    self.scheduler.cooldown(lease.api_key, model_name)
                error_type = type(error).__name__
                return {"status": "ERROR", "text": None, "error_type": error_type, "error_message": str(error), "error_details": repr(error), "request_meta": meta}

        return execute_with_retry(self.config, attempt)


class OllamaAPIManager:
    """Local manager retaining the same response contract (no API-key scheduler)."""

    def __init__(self, config: Dict[str, Any]):
        if ollama is None:
            raise ImportError("OllamaAPIManager requires the optional 'ollama' package.")
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.print_details = config.get("PRINT_API_CALL_DETAILS", False)
        self.truncation_length = config.get("API_RESPONSE_TRUNCATION_LENGTH", 50)
        self.client = ollama.Client(host=config.get("OLLAMA_BASE_URL", "http://localhost:11434"))

    def generate_content(self, prompt: str, model_name: str, temperature: Optional[float] = None) -> APIResponse:
        def attempt() -> APIResponse:
            try:
                options: Dict[str, Any] = {}
                if temperature is not None:
                    options["temperature"] = temperature
                if self.config.get("OLLAMA_THINK_MODE"):
                    options["think"] = self.config["OLLAMA_THINK_MODE"]
                response = self.client.generate(model=model_name, prompt=prompt, options=options)
                return {"status": "SUCCESS", "text": response["response"], "error_type": None, "error_message": None, "error_details": None}
            except ollama.ResponseError as error:
                return {"status": "ERROR", "text": None, "error_type": "OllamaResponseError", "error_message": str(error), "error_details": repr(error)}
            except Exception as error:
                return {"status": "ERROR", "text": None, "error_type": "OllamaConnectionError", "error_message": str(error), "error_details": repr(error)}

        return execute_with_retry(self.config, attempt)
