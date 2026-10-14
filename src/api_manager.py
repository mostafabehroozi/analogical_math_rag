"""Thread-safe API managers with per-key rolling RPM scheduling.

Every remote request is reserved by a shared scheduler before it is sent.  The
scheduler is scoped to ``(provider, key, model)`` so a temporary failure on one
credential never blocks healthy credentials.  The public ``generate_content``
interface remains compatible with the rest of the pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
import inspect
import json
import os
import copy
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, TypedDict, Union
from src.api_batching import APIBatchDispatcher, BatchItem, BatchSettings, batching_enabled
from src.context_logger import ctx_query_idx, ctx_batch_id, tprint
from src.distributed_execution import api_deadline_due, remaining_api_seconds

try:  # Keep scheduler/batch tests importable without optional provider SDKs.
    import ollama
except ModuleNotFoundError:  # pragma: no cover - depends on deployment extras
    ollama = None
try:
    import openai
except ModuleNotFoundError:  # pragma: no cover - depends on deployment extras
    openai = None
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
except ImportError:  # The legacy ``google`` namespace may exist without google-genai.
    genai = None
    types = None
    APIError = Exception

RETRYABLE_ERROR_TYPES = {
    "APITimeoutError", "APIConnectionError", "ResourceExhausted",
    "OllamaConnectionError", "UnknownError", "APIStatusError",
    "RateLimitError", "BatchError", "BatchItemError", "BatchTimeout",
}
NON_RETRYABLE_ERROR_TYPES = {"AuthenticationError", "PermissionDeniedError", "PaymentRequired", "InvalidArgument", "InvalidModel", "Safety", "NoChoices", "ModelMismatch", "ProactiveRateLimit", "SessionDeadline"}

class APIResponse(TypedDict, total=False):
    status: str
    text: Optional[str]
    error_type: Optional[str]
    error_message: Optional[str]
    error_details: Optional[Any]
    request_meta: Dict[str, Any]


def _mask_key(api_key: str) -> str:
    return f"…{api_key[-4:]}" if api_key else "<missing>"


def _get_api_caller_location() -> str:
    """Dynamically finds the exact file and function that requested the API call for accurate error logging."""
    try:
        for frame_info in inspect.stack():
            filename = os.path.basename(frame_info.filename)
            # Skip the internal wrapper files to find the real caller (like pipeline_steps.py)
            if filename not in ("api_manager.py", "context_logger.py", "logging.py"):
                return f"{filename}:{frame_info.lineno} (in {frame_info.function})"
    except Exception:
        pass
    return "Unknown Location"

# Thread-safe Global API Pause & Backoff Manager
class _GlobalAPIPauseManager:
    """Tracks consecutive errors and periodic breaks across all threads."""
    def __init__(self):
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._consecutive_errors = 0
        self._pause_until = 0.0
        self._last_periodic_pause = time.monotonic()

    def wait_if_needed(self, config: Dict[str, Any]) -> None:
        """Pauses the calling thread if a global break or error backoff is active."""
        with self._condition:
            now = time.monotonic()
            
            # 1. Check if a periodic break is due
            interval_mins = config.get("GLOBAL_PERIODIC_PAUSE_INTERVAL_MINUTES")
            if interval_mins and interval_mins > 0:
                interval_secs = interval_mins * 60.0
                if now - self._last_periodic_pause >= interval_secs:
                    duration = config.get("GLOBAL_PERIODIC_PAUSE_DURATION_SECONDS", 15.0)
                    tprint(f"⏰ [GLOBAL PAUSE] Scheduled periodic break! Pausing all new API calls for {duration}s.", level="WARNING")
                    self._pause_until = max(self._pause_until, now + duration)
                    self._last_periodic_pause = now
            
            # 2. Put the thread to sleep until the pause expires
            while True:
                now = time.monotonic()
                if api_deadline_due(config, now):
                    return
                if now >= self._pause_until:
                    break
                wait_time = self._pause_until - now
                remaining = remaining_api_seconds(config, now)
                if remaining is not None:
                    wait_time = min(wait_time, remaining)
                self._condition.wait(timeout=wait_time)

    def record_result(self, is_success: bool, config: Dict[str, Any]) -> None:
        """Updates the consecutive error counter after an API call."""
        with self._condition:
            if is_success:
                self._consecutive_errors = 0
            else:
                self._consecutive_errors += 1
                limit = config.get("GLOBAL_CONSECUTIVE_ERROR_LIMIT")
                if limit and limit > 0 and self._consecutive_errors >= limit:
                    pause_secs = config.get("GLOBAL_CONSECUTIVE_ERROR_PAUSE_SECONDS", 60.0)
                    tprint(f"🛑 [GLOBAL PAUSE] {self._consecutive_errors} consecutive errors hit! Pausing all new API calls for {pause_secs}s to recover.", level="ERROR")
                    
                    now = time.monotonic()
                    self._pause_until = max(self._pause_until, now + pause_secs)
                    self._consecutive_errors = 0  # Reset so it doesn't trigger repeatedly
                    self._condition.notify_all()

# Create the single shared manager for the whole application
global_pause_manager = _GlobalAPIPauseManager()

# Per-provider rate pacer — enforces a minimum time gap between
#       consecutive API calls to the same provider, across all threads.
class _ProviderRatePacer:
    """Thread-safe minimum-interval enforcer for API calls per provider."""

    def __init__(self) -> None:
        self._creation_lock = threading.Lock()
        self._locks: Dict[str, threading.Lock] = {}
        self._last_call_time: Dict[str, float] = {}

    def _get_lock(self, provider_name: str) -> threading.Lock:
        """Double-checked-locking creation of a per-provider lock."""
        if provider_name not in self._locks:
            with self._creation_lock:
                if provider_name not in self._locks:
                    self._locks[provider_name] = threading.Lock()
        return self._locks[provider_name]

    def pace(self, provider_name: str, min_seconds: float) -> None:
        """Block until it is safe to make the next API call to *provider_name*."""
        if min_seconds <= 0:
            return
            
        lock = self._get_lock(provider_name)
        
        # Acquire the lock so only one thread per provider can check/wait at a time
        with lock:
            now = time.monotonic()
            last = self._last_call_time.get(provider_name, 0.0)
            elapsed = now - last
            
            if elapsed < min_seconds:
                wait_time = min_seconds - elapsed
                tprint(
                    f"⏳ [PACER] Provider '{provider_name}' pacing: "
                    f"sleeping {wait_time:.2f}s "
                    f"(min gap={min_seconds}s, elapsed={elapsed:.2f}s).",
                    level="DEBUG",
                )
                time.sleep(wait_time)
                
            # Record the time we release the lock (when the API call is about to start)
            self._last_call_time[provider_name] = time.monotonic()

# Global pacer instance shared across all threads / all API managers
provider_pacer = _ProviderRatePacer()


def _pace_provider_call(config: Dict[str, Any], provider_name: str) -> None:
    """Apply the configured gap between outgoing calls to this provider."""
    configured_delay = config.get("GLOBAL_API_CALL_DELAY_SECONDS", 0.0)
    # Accept the former single numeric value for existing callers.
    if isinstance(configured_delay, dict):
        configured_delay = configured_delay.get(provider_name, 0.0)
    delay = float(configured_delay)
    if config.get("ENABLE_MIN_TIME_BETWEEN_API_CALLS", False):
        delay = max(delay, float(config.get("MIN_TIME_BETWEEN_API_CALLS_SECONDS", 0.0)))
    provider_pacer.pace(provider_name, delay)


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
            delay = float(quota.get("delay_seconds", 1.0))
            rpm = max(1, int(self.WINDOW_SECONDS / max(delay, 1.0)))
        return max(1, int(rpm))

    def acquire(self, model_name: str) -> Optional[KeyLease]:
        """Reserve the least-used ready key, waiting only for the necessary key."""
        while True:
            with self._condition:
                now = time.monotonic()
                if api_deadline_due(self.config, now):
                    return None
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

                wait_seconds = max(0.001, min(waits))
                remaining = remaining_api_seconds(self.config, now)
                if remaining is not None:
                    wait_seconds = min(wait_seconds, max(0.001, remaining))
                self._condition.wait(timeout=wait_seconds)

    def cooldown(self, api_key: str, model_name: str, seconds: Optional[float] = None) -> None:
        delay = float(seconds if seconds is not None else self.config.get("API_KEY_ERROR_COOLDOWN_SECONDS", 20.0))
        with self._condition:
            key = (api_key, model_name)
            self._cooldown_until[key] = max(self._cooldown_until[key], time.monotonic() + max(0.0, delay))

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


def _format_api_log(provider: str, model: str, key: str, duration: float, attempt: int, max_attempts: int, status: str, prompt: str, response_text: str, config: Dict) -> str:
    """Formats a beautiful, multi-line string for console output."""
    trunc_len = config.get("API_RESPONSE_TRUNCATION_LENGTH", 70)
    
    # Safely truncate and remove newlines so it stays on one neat line
    in_trunc = (prompt[:trunc_len] + '...') if prompt and len(prompt) > trunc_len else str(prompt)
    in_trunc = in_trunc.replace('\n', ' ') 
    
    out_trunc = ""
    if response_text:
        out_trunc = (response_text[:trunc_len] + '...') if len(response_text) > trunc_len else str(response_text)
        out_trunc = out_trunc.replace('\n', ' ')
        
    status_icon = "🟢" if status == "SUCCESS" else "🔴"
    
    log_str = f"⏱️ {duration:.2f}s | [{provider} | {model} | Key: {key}] | Att: {attempt}/{max_attempts} | {status_icon} {status}\n"
    log_str += f"    ↳ IN : \"{in_trunc}\"\n"
    log_str += f"    ↳ OUT: \"{out_trunc}\""
    return log_str


def execute_with_retry(config: Dict[str, Any], provider_name: str, model_name: str, prompt: str, api_call_func: Callable[[], APIResponse]) -> APIResponse:
    """Retry logic that times the execution, respects global pauses, and prints clean logs."""
    caller_loc = _get_api_caller_location()
    
    # Determine max attempts based on config
    max_attempts = max(1, int(config.get("MAX_API_RETRIES", 3))) if config.get("ENABLE_API_RETRY", False) else 1
    retry_all = config.get("RETRY_ALL_API_ERRORS", True)
    last_response: Optional[APIResponse] = None

    def deadline_response() -> APIResponse:
        return {
            "status": "ERROR",
            "text": None,
            "error_type": "SessionDeadline",
            "error_message": (
                "Distributed session API deadline reached; this question will be "
                "checkpointed as incomplete and retried by the same worker next session."
            ),
        }
    
    for attempt in range(1, max_attempts + 1):
        if api_deadline_due(config):
            return deadline_response()
        
        # Check if the system is globally paused before making ANY request
        global_pause_manager.wait_if_needed(config)
        if api_deadline_due(config):
            return deadline_response()
        
        # 1. Start the Timer
        start_time = time.monotonic()
        
        # 2. Make the API Call
        try:
            response = api_call_func()
        except Exception as error:
            response = {
                "status": "ERROR", "text": None,
                "error_type": type(error).__name__,
                "error_message": str(error), "error_details": repr(error),
            }
        
        # 3. Stop the Timer
        duration = time.monotonic() - start_time
        
        # 4. Extract data for logging
        status = response.get("status", "ERROR")
        key = response.get("request_meta", {}).get("key", "Local")
        response_text = response.get("text") or response.get("error_message", "No text returned")
        
        # Record success or failure to the global pause manager
        is_success = (status == "SUCCESS")
        global_pause_manager.record_result(is_success, config)
        
        # 5. Format and print the beautiful log
        log_msg = _format_api_log(provider_name, model_name, key, duration, attempt, max_attempts, status, prompt, response_text, config)
        
        if is_success:
            tprint(log_msg, level="INFO")
            return response

        # If we reach here, it failed
        error_type = response.get("error_type") or status
        tprint(log_msg, level="WARNING")
        
        # A session deadline is a local stop signal, not a failed API call.
        if error_type == "SessionDeadline":
            retryable = False
        elif error_type in NON_RETRYABLE_ERROR_TYPES:
            retryable = False
        elif retry_all:
            retryable = True
        else:
            retryable = error_type in RETRYABLE_ERROR_TYPES and error_type not in NON_RETRYABLE_ERROR_TYPES

        if not retryable or attempt == max_attempts:
            response["retry_exhausted"] = attempt == max_attempts
            tprint(f"[API ERROR] Final failure. Type: {error_type} @ {caller_loc}", level="ERROR")
            return response
        
        # Wait before retrying
        retry_delay = float(config.get("API_RETRY_DELAY_SECONDS", 20.0))
        remaining = remaining_api_seconds(config)
        if remaining is not None:
            if remaining <= 0:
                return deadline_response()
            retry_delay = min(retry_delay, remaining)
        time.sleep(max(0.0, retry_delay))
            
        last_response = response
        
    final_resp = last_response or {"status": "ERROR", "text": None, "error_type": "RetryError", "error_message": "No API attempt made"}
    return final_resp


class _ConfigBindable:
    def for_config(self, config: Dict[str, Any]):
        """Use experiment options while sharing clients, quota and batch state."""
        bound = copy.copy(self)
        bound.config = config
        return bound


class _KeyedAPIManager(_ConfigBindable):
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
            "query_idx": ctx_query_idx.get(),
            "batch_id": ctx_batch_id.get(),
        }

    def _print(self, message: str) -> None:
        if self.print_details:
            tprint(f"[API {self.provider_name}] {message}", level="DEBUG")


_MODEL_ROLE_SUFFIX = {
    "adaptation": "ADAPTATION",
    "final_solver": "FINAL_SOLVER",
    "solver": "FINAL_SOLVER",
    "evaluator": "EVALUATOR",
    "simplification": "SIMPLIFICATION",
}


SUPPORTED_API_PROVIDERS = frozenset({"gemini", "avalai", "openrouter", "kaggle", "ollama"})


def model_name_for(manager: Any, config: Dict[str, Any], role: str) -> str:
    """Resolve a role's model for any configured API provider."""
    provider = getattr(manager, "provider_name", None)
    if provider not in SUPPORTED_API_PROVIDERS:
        raise TypeError(f"Unsupported API manager: {type(manager)}")
    suffix = _MODEL_ROLE_SUFFIX[role]
    prefix = provider.upper()
    model = config.get(f"{prefix}_MODEL_NAME_{suffix}")
    if role == "simplification" and not model:
        model = config.get(f"{prefix}_MODEL_NAME_ADAPTATION")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{prefix}_MODEL_NAME_{suffix} must be configured.")
    return model


class GeminiAPIManager(_KeyedAPIManager):
    """Gemini manager upgraded to use the new google-genai SDK with Thinking support."""

    provider_name = "gemini"

    def __init__(self, api_keys: List[str], model_quotas: Dict[str, Any], global_delay_seconds: Union[float, Dict[str, float]] = 0, config: Optional[Dict[str, Any]] = None):
        super().__init__(api_keys, model_quotas, config)
        if "GLOBAL_API_CALL_DELAY_SECONDS" not in self.config:
            self.config["GLOBAL_API_CALL_DELAY_SECONDS"] = global_delay_seconds
        if genai is None:
            raise ImportError("GeminiAPIManager requires the 'google-genai' package. Run: pip install google-genai")
        
        self.timeout_seconds = float(self.config.get("API_REQUEST_TIMEOUT_SECONDS", 300.0))
        self.clients = {
            key: genai.Client(
                api_key=key, 
                http_options={'timeout': int(self.timeout_seconds * 1000)}
            ) for key in self.api_keys_list
        }
        
    def _get_max_tokens(self, model_name: str) -> Optional[int]:
        if model_name == self.config.get("GEMINI_MODEL_NAME_FINAL_SOLVER"):
            return self.config.get("DEFAULT_FINAL_SOLVER_MAX_TOKENS", 8192)
        elif model_name == self.config.get("GEMINI_MODEL_NAME_ADAPTATION"):
            return self.config.get("DEFAULT_ADAPTATION_MAX_TOKENS", 2048)
        elif model_name == self.config.get("GEMINI_MODEL_NAME_EVALUATOR"):
            return self.config.get("DEFAULT_EVALUATOR_MAX_TOKENS", 512)
        return None

    def generate_content(self, prompt: str, model_name: str, temperature: Optional[float] = None, avalai_role: Optional[str] = None) -> APIResponse:
        self._print(f"request model={model_name} prompt={prompt[:self.truncation_length]!r}")

        def attempt() -> APIResponse:
            lease = self.scheduler.acquire(model_name)
            if lease is None:
                return {"status": "RATE_LIMITED", "text": None, "error_type": "ProactiveRateLimit", "error_message": f"No eligible Gemini key for {model_name}.", "error_details": self.scheduler.snapshot()}
            
            meta = self._request_meta(lease)
            config_kwargs = {}
            if temperature is not None:
                config_kwargs["temperature"] = temperature
                
            max_tokens = self._get_max_tokens(model_name)
            if max_tokens is not None:
                config_kwargs["max_output_tokens"] = max_tokens

            if self.config.get("GEMINI_ENABLE_THINKING", False):
                thinking_level_str = str(self.config.get("GEMINI_THINKING_LEVEL", "minimal")).lower()
                if thinking_level_str == "high":
                    t_level = types.ThinkingLevel.HIGH
                else:
                    t_level = types.ThinkingLevel.MINIMAL
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=t_level)

            gen_config = types.GenerateContentConfig(**config_kwargs)

            try:
                clean_model_name = model_name.replace("models/", "")
                client = self.clients[lease.api_key]
                _pace_provider_call(self.config, self.provider_name)
                response = client.models.generate_content(
                    model=clean_model_name,
                    contents=prompt,
                    config=gen_config
                )
                
                text = response.text
                if text:
                    self._print(f"success model={model_name} key={meta['key']}")
                    return {"status": "SUCCESS", "text": text, "error_type": None, "error_message": None, "error_details": None, "request_meta": meta}
                else:
                    return {"status": "BLOCKED", "text": None, "error_type": "Safety", "error_message": "Gemini returned no text (possibly blocked).", "error_details": None, "request_meta": meta}

            except APIError as error:
                status_code = error.code
                message = error.message

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
                    error_type, api_status = ("APIConnectionError", "ERROR")
                    
                self._print(f"failure model={model_name} key={meta['key']} type={error_type}")
                return {"status": api_status, "text": None, "error_type": error_type, "error_message": message, "error_details": str(error), "request_meta": meta}
            
            except Exception as error:
                self.scheduler.cooldown(lease.api_key, model_name)
                self._print(f"failure model={model_name} key={meta['key']} type=UnknownError")
                return {"status": "ERROR", "text": None, "error_type": "UnknownError", "error_message": str(error), "error_details": repr(error), "request_meta": meta}

        return execute_with_retry(self.config, self.provider_name, model_name, prompt, attempt)

class AvalAIAPIManager(_KeyedAPIManager):
    provider_name = "avalai"

    def __init__(self, api_key_or_list: Union[str, List[str]], base_url: str, model_quotas: Dict[str, Any], global_delay_seconds: Union[float, Dict[str, float]] = 0, config: Optional[Dict[str, Any]] = None):
        if openai is None:
            raise ImportError("AvalAIAPIManager requires the optional 'openai' package.")
        keys = [api_key_or_list] if isinstance(api_key_or_list, str) else list(api_key_or_list)
        if not base_url:
            raise ValueError("base_url cannot be empty for AvalAIAPIManager.")
        super().__init__(keys, model_quotas, config)
        if "GLOBAL_API_CALL_DELAY_SECONDS" not in self.config:
            self.config["GLOBAL_API_CALL_DELAY_SECONDS"] = global_delay_seconds
        
        timeout_seconds = float(self.config.get("API_REQUEST_TIMEOUT_SECONDS", 300.0))
        self.clients = {
            key: openai.OpenAI(
                api_key=key, 
                base_url=base_url,
                timeout=timeout_seconds 
            ) for key in self.api_keys_list
        }
        
    def generate_content(self, prompt: str, model_name: str, temperature: Optional[float] = None, avalai_role: Optional[str] = None) -> APIResponse:
        self._print(f"request model={model_name} prompt={prompt[:self.truncation_length]!r}")

        def attempt() -> APIResponse:
            lease = self.scheduler.acquire(model_name)
            if lease is None:
                return {"status": "RATE_LIMITED", "text": None, "error_type": "ProactiveRateLimit", "error_message": f"No eligible AvalAI key for {model_name}.", "error_details": self.scheduler.snapshot()}
            meta = self._request_meta(lease)
            try:
                kwargs = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature
                }
                role_effort_keys = {
                    "adaptation": "AVALAI_REASONING_EFFORT_ADAPTATION",
                    "final_solver": "AVALAI_REASONING_EFFORT_FINAL_SOLVER",
                    "evaluator": "AVALAI_REASONING_EFFORT_EVALUATOR",
                }
                role_effort_key = role_effort_keys.get(avalai_role)
                reasoning_effort = self.config.get(role_effort_key) if role_effort_key else None
                if reasoning_effort is None:
                    reasoning_effort = self.config.get("AVALAI_REASONING_EFFORT")
                if reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort
                
                enable_thinking = self.config.get("AVALAI_ENABLE_THINKING")
                if enable_thinking is None:
                    enable_thinking = self.config.get("AVALAI_CHAT_TEMPLATE_KWARGS_ENABLE_THINKING")
                if enable_thinking is not None:
                    kwargs["extra_body"] = {
                        "chat_template_kwargs": {
                            "enable_thinking": enable_thinking
                        }
                    }
                
                _pace_provider_call(self.config, self.provider_name)
                completion = self.clients[lease.api_key].chat.completions.create(**kwargs)
                returned_model = getattr(completion, "model", None)
                if returned_model and returned_model != model_name:
                    self.logger.debug(f"Provider returned model '{returned_model}' instead of requested '{model_name}'. Proceeding.")
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

        return execute_with_retry(self.config, self.provider_name, model_name, prompt, attempt)


class OpenRouterAPIManager(_KeyedAPIManager):
    """OpenRouter chat completions with per-model endpoint preferences."""

    provider_name = "openrouter"

    def __init__(self, api_key_or_list: Union[str, List[str]], model_quotas: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        if openai is None:
            raise ImportError("OpenRouterAPIManager requires the optional 'openai' package.")
        keys = [api_key_or_list] if isinstance(api_key_or_list, str) else list(api_key_or_list)
        if not keys or any(not isinstance(key, str) or not key.strip() or "YOUR_" in key.upper() for key in keys):
            raise ValueError("OPENROUTER_API_KEY must contain a usable key.")
        super().__init__(keys, model_quotas, config)
        base_url = self.config.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        if not base_url:
            raise ValueError("OPENROUTER_BASE_URL cannot be empty.")
        headers = {"X-OpenRouter-Metadata": "enabled"}
        if self.config.get("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = self.config["OPENROUTER_HTTP_REFERER"]
        if self.config.get("OPENROUTER_APP_TITLE"):
            headers["X-OpenRouter-Title"] = self.config["OPENROUTER_APP_TITLE"]
        self.base_url = base_url.rstrip("/")
        self.headers = headers
        self.timeout_seconds = float(self.config.get("API_REQUEST_TIMEOUT_SECONDS", 300.0))
        self.batcher = APIBatchDispatcher(self.provider_name, self._submit_batch)
        self.clients = {
            key: openai.OpenAI(
                api_key=key,
                base_url=base_url,
                timeout=float(self.config.get("API_REQUEST_TIMEOUT_SECONDS", 300.0)),
                default_headers=headers,
                max_retries=0,
            ) for key in self.api_keys_list
        }

    def _extra_body(self, model_name: str, role: Optional[str]) -> Dict[str, Any]:
        config = self.config
        model_routing = config.get("OPENROUTER_MODEL_ROUTING") or {}
        default_routing = config.get("OPENROUTER_PROVIDER_ROUTING") or {}
        model_fallbacks = config.get("OPENROUTER_MODEL_FALLBACKS") or {}
        if not isinstance(default_routing, dict) or not isinstance(model_routing, dict) or not isinstance(model_fallbacks, dict):
            raise ValueError("OpenRouter routing and fallback settings must be dictionaries.")
        provider = model_routing.get(model_name, default_routing)
        if not isinstance(provider, dict):
            raise ValueError(f"OpenRouter provider routing for {model_name!r} must be a dictionary.")
        body: Dict[str, Any] = {}
        if provider:
            body["provider"] = copy.deepcopy(provider)
        fallbacks = model_fallbacks.get(model_name)
        if fallbacks is not None:
            if not isinstance(fallbacks, list) or not all(isinstance(item, str) and item.strip() for item in fallbacks):
                raise ValueError(f"OpenRouter model fallbacks for {model_name!r} must be a list of model IDs.")
            if fallbacks:
                body["models"] = list(fallbacks)
        role_key = {
            "adaptation": "OPENROUTER_REASONING_EFFORT_ADAPTATION",
            "final_solver": "OPENROUTER_REASONING_EFFORT_FINAL_SOLVER",
            "evaluator": "OPENROUTER_REASONING_EFFORT_EVALUATOR",
        }.get(role)
        effort = config.get(role_key) if role_key else None
        if effort is None:
            effort = config.get("OPENROUTER_REASONING_EFFORT")
        if effort is not None:
            body["reasoning"] = {"effort": effort}
        return body

    def generate_content(self, prompt: str, model_name: str, temperature: Optional[float] = None, avalai_role: Optional[str] = None) -> APIResponse:
        self._print(f"request model={model_name} prompt={prompt[:self.truncation_length]!r}")
        extra_body = self._extra_body(model_name, avalai_role)
        if batching_enabled(self.config):
            return self._generate_batched(prompt, model_name, temperature, extra_body)

        def attempt() -> APIResponse:
            lease = self.scheduler.acquire(model_name)
            if lease is None:
                return {"status": "RATE_LIMITED", "text": None, "error_type": "ProactiveRateLimit", "error_message": f"No eligible OpenRouter key for {model_name}.", "error_details": self.scheduler.snapshot()}
            meta = self._request_meta(lease)
            meta["requested_model"] = model_name
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if extra_body:
                kwargs["extra_body"] = extra_body
            try:
                _pace_provider_call(self.config, self.provider_name)
                completion = self.clients[lease.api_key].chat.completions.create(**kwargs)
                meta["returned_model"] = getattr(completion, "model", None)
                router_meta = getattr(completion, "openrouter_metadata", None)
                if router_meta is None:
                    router_meta = (getattr(completion, "model_extra", None) or {}).get("openrouter_metadata")
                if hasattr(router_meta, "model_dump"):
                    router_meta = router_meta.model_dump()
                if isinstance(router_meta, dict) and router_meta:
                    endpoints = router_meta.get("endpoints") or {}
                    for endpoint in endpoints.get("available") or []:
                        if isinstance(endpoint, dict) and endpoint.get("selected"):
                            meta["routed_provider"] = endpoint.get("provider")
                            break
                choices = getattr(completion, "choices", [])
                content = choices[0].message.content if choices else None
                if not content:
                    return {"status": "BLOCKED", "text": None, "error_type": "NoChoices", "error_message": "OpenRouter returned no completion text.", "request_meta": meta}
                return {"status": "SUCCESS", "text": content, "error_type": None, "error_message": None, "request_meta": meta}
            except openai.RateLimitError as error:
                self.scheduler.cooldown(lease.api_key, model_name)
                return {"status": "RATE_LIMITED", "text": None, "error_type": "RateLimitError", "error_message": str(error), "request_meta": meta}
            except openai.APIStatusError as error:
                code = error.status_code
                if code in (401, 402, 403):
                    self.scheduler.disable(lease.api_key, f"HTTP {code}")
                elif code >= 500:
                    self.scheduler.cooldown(lease.api_key, model_name)
                error_type = {400: "InvalidArgument", 401: "AuthenticationError", 402: "PaymentRequired", 403: "PermissionDeniedError", 404: "InvalidModel", 422: "InvalidArgument"}.get(code, type(error).__name__)
                return {"status": "ERROR", "text": None, "error_type": error_type, "error_message": str(error), "request_meta": meta}
            except Exception as error:
                self.scheduler.cooldown(lease.api_key, model_name)
                return {"status": "ERROR", "text": None, "error_type": type(error).__name__, "error_message": str(error), "request_meta": meta}

        return execute_with_retry(self.config, self.provider_name, model_name, prompt, attempt)

    # --- Batch API (https://openrouter.ai/docs/batch-quickstart) ---
    # One batch has one model and one provider object; only provider.only and
    # provider.allow_fallbacks are accepted, and results come back inline once
    # the batch reaches a terminal status.

    _BATCH_PROVIDER_KEYS = frozenset({"only", "allow_fallbacks"})
    _BATCH_TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})

    def _generate_batched(self, prompt: str, model_name: str, temperature: Optional[float], extra_body: Dict[str, Any]) -> APIResponse:
        provider = extra_body.get("provider") or {}
        unsupported = sorted(set(provider) - self._BATCH_PROVIDER_KEYS)
        if unsupported or extra_body.get("models"):
            reason = (
                f"provider routing keys {unsupported}" if unsupported
                else f"OPENROUTER_MODEL_FALLBACKS for {model_name!r}"
            )
            return {
                "status": "ERROR", "text": None, "error_type": "InvalidArgument",
                "error_message": (
                    f"The OpenRouter Batch API does not accept {reason}; only provider "
                    "'only'/'allow_fallbacks' are allowed. Change the routing or set "
                    "API_BATCHING_ENABLED=False."
                ),
            }
        body: Dict[str, Any] = {"messages": [{"role": "user", "content": prompt}]}
        if temperature is not None:
            body["temperature"] = temperature
        if "reasoning" in extra_body:
            body["reasoning"] = extra_body["reasoning"]
        payload = {"model": model_name, "provider": provider, "body": body}
        group = (model_name, json.dumps(provider, sort_keys=True))
        settings = BatchSettings.from_config(self.config)
        return execute_with_retry(
            self.config, self.provider_name, model_name, prompt,
            lambda: self.batcher.call(payload, group, settings),
        )

    def _batch_http(self, method: str, path: str, api_key: str, payload: Optional[Dict[str, Any]] = None) -> Tuple[int, Any]:
        import httpx  # Installed with the openai SDK.

        response = httpx.request(
            method, f"{self.base_url}{path}", json=payload, timeout=self.timeout_seconds,
            headers={**self.headers, "Authorization": f"Bearer {api_key}"},
        )
        try:
            data = response.json()
        except ValueError:
            data = {"error": {"message": response.text[:500]}}
        return response.status_code, data

    def _http_error_type(self, code: int, api_key: str, model_name: str) -> str:
        if code in (401, 402, 403):
            self.scheduler.disable(api_key, f"HTTP {code}")
        elif code == 429 or code >= 500:
            self.scheduler.cooldown(api_key, model_name)
        return {400: "InvalidArgument", 401: "AuthenticationError", 402: "PaymentRequired",
                403: "PermissionDeniedError", 404: "InvalidModel", 422: "InvalidArgument",
                429: "RateLimitError"}.get(code, "APIStatusError")

    def _submit_batch(self, items: List[BatchItem], settings: BatchSettings) -> List[APIResponse]:
        model_name = items[0].payload["model"]
        provider = items[0].payload["provider"]

        def fail_all(error_type: str, message: str, meta: Optional[Dict[str, Any]] = None) -> List[APIResponse]:
            status = "RATE_LIMITED" if error_type in ("RateLimitError", "ProactiveRateLimit") else "ERROR"
            return [{"status": status, "text": None, "error_type": error_type, "error_message": message,
                     "request_meta": dict(meta or {})} for _ in items]

        lease = self.scheduler.acquire(model_name)
        if lease is None:
            return fail_all("ProactiveRateLimit", f"No eligible OpenRouter key for {model_name}.")
        meta = self._request_meta(lease)
        meta["requested_model"] = model_name
        request: Dict[str, Any] = {"endpoint": "/v1/chat/completions", "model": model_name}
        if provider:
            request["provider"] = provider
        request["completion_window"] = "24h"
        # The API requires "requests" to be serialized last.
        request["requests"] = [
            {"custom_id": f"req-{position}", "body": item.payload["body"]}
            for position, item in enumerate(items)
        ]
        _pace_provider_call(self.config, self.provider_name)
        code, batch = self._batch_http("POST", "/batches", lease.api_key, request)
        if code >= 400 or not isinstance(batch, dict) or not batch.get("id"):
            message = (batch.get("error") or {}).get("message") if isinstance(batch, dict) else None
            error_type = self._http_error_type(code, lease.api_key, model_name) if code >= 400 else "BatchError"
            return fail_all(error_type, f"Batch create failed (HTTP {code}): {message or batch}", meta)
        batch_id = batch["id"]
        meta["openrouter_batch_id"] = batch_id
        tprint(f"[API BATCH openrouter] {batch_id}: {len(items)} request(s) for {model_name} submitted.", level="INFO")

        deadline = time.monotonic() + settings.timeout_seconds
        while batch.get("status") not in self._BATCH_TERMINAL:
            if api_deadline_due(self.config):
                return fail_all("SessionDeadline", f"Session API deadline reached while waiting for batch {batch_id}.", meta)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return fail_all(
                    "BatchTimeout",
                    f"Batch {batch_id} was not finished after {settings.timeout_seconds:.0f}s "
                    f"(last status {batch.get('status')!r}); OpenRouter may still complete and bill it.",
                    meta,
                )
            time.sleep(min(settings.poll_seconds, remaining))
            try:
                code, polled = self._batch_http("GET", f"/batches/{batch_id}", lease.api_key)
            except Exception as error:  # A failed poll is retried on the next tick.
                tprint(f"[API BATCH openrouter] {batch_id}: poll failed ({error!r}).", level="DEBUG")
                continue
            if code < 400 and isinstance(polled, dict):
                batch = polled
                tprint(f"[API BATCH openrouter] {batch_id}: {batch.get('status')} {batch.get('request_counts')}", level="DEBUG")

        if batch.get("status") != "completed":
            return fail_all("BatchError", f"Batch {batch_id} ended {batch.get('status')!r}: {batch.get('errors') or batch.get('error')}", meta)

        by_id = {result.get("custom_id"): result for result in batch.get("results") or []}
        responses: List[APIResponse] = []
        for position in range(len(items)):
            item_meta = dict(meta)
            result = by_id.get(f"req-{position}")
            response = (result or {}).get("response") or {}
            body = response.get("body") or {}
            if result is None or result.get("error") or response.get("status_code") != 200:
                code = response.get("status_code")
                error = (result or {}).get("error") or body.get("error") or {}
                error_type = (
                    self._http_error_type(code, lease.api_key, model_name)
                    if isinstance(code, int) and code >= 400 else "BatchItemError"
                )
                message = error.get("message") if isinstance(error, dict) else error
                responses.append({"status": "ERROR", "text": None, "error_type": error_type,
                                  "error_message": str(message or "Missing batch result."), "request_meta": item_meta})
                continue
            item_meta["returned_model"] = body.get("model")
            choices = body.get("choices") or []
            content = ((choices[0] if choices else {}).get("message") or {}).get("content")
            if not content:
                responses.append({"status": "BLOCKED", "text": None, "error_type": "NoChoices",
                                  "error_message": "OpenRouter returned no completion text.", "request_meta": item_meta})
            else:
                responses.append({"status": "SUCCESS", "text": content, "error_type": None,
                                  "error_message": None, "request_meta": item_meta})
        return responses


def bind_api_managers(api_managers: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Bind request settings (reasoning, routing, batching) to one experiment.

    Bound copies share clients, key schedulers and batch dispatchers with the
    original manager, so no shared state is mutated.
    """
    return {
        name: manager.for_config(config) if isinstance(manager, _ConfigBindable) else manager
        for name, manager in api_managers.items()
    }


def _kaggle_error_type(message: str) -> str:
    """Classify a Kaggle model-proxy failure from its exception name or traceback."""
    for marker, error_type in (
        ("RateLimitError", "RateLimitError"),
        ("AuthenticationError", "AuthenticationError"),
        ("PermissionDeniedError", "PermissionDeniedError"),
        ("NotFoundError", "InvalidModel"),
        ("BadRequestError", "InvalidArgument"),
        ("UnprocessableEntityError", "InvalidArgument"),
    ):
        if marker in message:
            return error_type
    return "UnknownError"


class KaggleBenchmarksAPIManager(_KeyedAPIManager):
    """Kaggle Benchmarks model proxy (``kaggle_benchmarks``), free within the notebook quota.

    Direct calls use ``llm.prompt`` inside an isolated chat: outside a task the
    library shares one global chat, so concurrent prompts would overwrite each
    other.  Batched calls run one ``task.evaluate`` over a DataFrame of
    prompts; every row is its own task run with its own chat.
    The proxy has no per-user keys, so one pseudo-key carries the RPM quota.
    """

    provider_name = "kaggle"
    _PROXY_KEY = "model-proxy"

    def __init__(self, model_quotas: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        try:
            import kaggle_benchmarks as kbench
        except ImportError as error:
            raise ImportError(
                "KaggleBenchmarksAPIManager requires the 'kaggle-benchmarks' package "
                "(preinstalled in Kaggle notebooks; otherwise: pip install kaggle-benchmarks)."
            ) from error
        if not kbench.kaggle.is_configured():
            raise RuntimeError(
                "The Kaggle model proxy is not configured (MODEL_PROXY_URL and "
                "MODEL_PROXY_API_KEY are missing). Run inside a Kaggle notebook with "
                "benchmark models available, or authenticate with 'kaggle benchmarks auth'."
            )
        super().__init__([self._PROXY_KEY], model_quotas, config)
        self.kbench = kbench
        self._llms: Dict[str, Any] = {}
        self._llm_lock = threading.Lock()
        self._row_task = None
        self.batcher = APIBatchDispatcher(self.provider_name, self._submit_batch)

    def _llm(self, model_name: str) -> Any:
        with self._llm_lock:
            if model_name not in self._llms:
                llm = self.kbench.llms.get(model_name)
                self._llms[model_name] = llm if llm is not None else self.kbench.kaggle.load_model(model_name)
            return self._llms[model_name]

    def _prompt_options(self, temperature: Optional[float], role: Optional[str]) -> Dict[str, Any]:
        # The library currently drops temperature for proxy models; it is sent
        # anyway so it takes effect as soon as the proxy supports it.
        options: Dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        role_key = {
            "adaptation": "KAGGLE_REASONING_EFFORT_ADAPTATION",
            "final_solver": "KAGGLE_REASONING_EFFORT_FINAL_SOLVER",
            "evaluator": "KAGGLE_REASONING_EFFORT_EVALUATOR",
        }.get(role)
        effort = self.config.get(role_key) if role_key else None
        if effort is None:
            effort = self.config.get("KAGGLE_REASONING_EFFORT")
        if effort is not None:
            options["reasoning"] = effort
        return options

    def _failure(self, message: str, meta: Dict[str, Any], model_name: str) -> APIResponse:
        error_type = _kaggle_error_type(message)
        if error_type in ("AuthenticationError", "PermissionDeniedError"):
            self.scheduler.disable(self._PROXY_KEY, error_type)
        elif error_type in ("RateLimitError", "UnknownError"):
            self.scheduler.cooldown(self._PROXY_KEY, model_name)
        status = "RATE_LIMITED" if error_type == "RateLimitError" else "ERROR"
        return {"status": status, "text": None, "error_type": error_type,
                "error_message": message[-2000:], "request_meta": meta}

    def generate_content(self, prompt: str, model_name: str, temperature: Optional[float] = None, avalai_role: Optional[str] = None) -> APIResponse:
        self._print(f"request model={model_name} prompt={prompt[:self.truncation_length]!r}")
        options = self._prompt_options(temperature, avalai_role)
        if batching_enabled(self.config):
            payload = {
                "model": model_name, "prompt": prompt, "options": options,
                "n_jobs": max(1, int(self.config.get("API_BATCH_KAGGLE_N_JOBS", 5))),
            }
            settings = BatchSettings.from_config(self.config)
            return execute_with_retry(
                self.config, self.provider_name, model_name, prompt,
                lambda: self.batcher.call(payload, model_name, settings),
            )

        def attempt() -> APIResponse:
            lease = self.scheduler.acquire(model_name)
            if lease is None:
                return {"status": "RATE_LIMITED", "text": None, "error_type": "ProactiveRateLimit", "error_message": f"No eligible Kaggle proxy quota for {model_name}.", "error_details": self.scheduler.snapshot()}
            meta = self._request_meta(lease)
            try:
                llm = self._llm(model_name)
                _pace_provider_call(self.config, self.provider_name)
                with self.kbench.chats.new("analogical-math-rag", orphan=True):
                    text = llm.prompt(prompt, **options)
            except Exception as error:
                return self._failure(f"{type(error).__name__}: {error}", meta, model_name)
            if not text:
                return {"status": "BLOCKED", "text": None, "error_type": "NoChoices", "error_message": "Kaggle model proxy returned no text.", "request_meta": meta}
            return {"status": "SUCCESS", "text": text, "error_type": None, "error_message": None, "request_meta": meta}

        return execute_with_retry(self.config, self.provider_name, model_name, prompt, attempt)

    def _get_row_task(self) -> Any:
        with self._llm_lock:
            if self._row_task is None:
                def prompt_row(llm, prompt, options):
                    return {"text": llm.prompt(prompt, **json.loads(options))}

                # kbench infers the result type from the real annotation object;
                # this module's postponed annotations would only give a string.
                prompt_row.__annotations__ = {"return": dict}
                self._row_task = self.kbench.task(
                    name="analogical_math_rag_prompt", store_task=False, store_run=False
                )(prompt_row)
            return self._row_task

    def _submit_batch(self, items: List[BatchItem], settings: BatchSettings) -> List[APIResponse]:
        import pandas as pd

        model_name = items[0].payload["model"]
        responses: List[Optional[APIResponse]] = [None] * len(items)
        metas: List[Dict[str, Any]] = [{} for _ in items]
        runnable: List[int] = []
        for position in range(len(items)):
            # Every row is a real model call, so each one takes quota.
            lease = self.scheduler.acquire(model_name)
            if lease is None:
                responses[position] = {"status": "RATE_LIMITED", "text": None, "error_type": "ProactiveRateLimit",
                                       "error_message": f"No eligible Kaggle proxy quota for {model_name}."}
                continue
            metas[position] = self._request_meta(lease)
            runnable.append(position)
        if runnable:
            frame = pd.DataFrame({
                "prompt": [items[position].payload["prompt"] for position in runnable],
                # JSON keeps None and per-row options intact through pandas.
                "options": [json.dumps(items[position].payload["options"]) for position in runnable],
            })
            _pace_provider_call(self.config, self.provider_name)
            runs = self._get_row_task().evaluate(
                llm=[self._llm(model_name)],
                evaluation_data=frame,
                n_jobs=min(items[0].payload["n_jobs"], len(runnable)),
                timeout=settings.timeout_seconds,
                on_failure="continue",
                remove_run_files=True,
            )
            # evaluate() passes each row's DataFrame index as the run's param_id.
            by_row = {int(run.param_id): run for run in runs if run.param_id is not None}
            for row, position in enumerate(runnable):
                run = by_row.get(row)
                meta = metas[position]
                result = getattr(run, "result", None)
                if run is None or str(run.status) != "success" or not isinstance(result, dict):
                    message = getattr(run, "error_message", None) or "Kaggle task run produced no result."
                    responses[position] = self._failure(str(message), meta, model_name)
                elif not result.get("text"):
                    responses[position] = {"status": "BLOCKED", "text": None, "error_type": "NoChoices",
                                           "error_message": "Kaggle model proxy returned no text.", "request_meta": meta}
                else:
                    responses[position] = {"status": "SUCCESS", "text": result["text"], "error_type": None,
                                           "error_message": None, "request_meta": meta}
        return responses


class OllamaAPIManager(_ConfigBindable):
    """Local manager retaining the same response contract (no API-key scheduler)."""

    provider_name = "ollama"

    def __init__(self, config: Dict[str, Any]):
        if ollama is None:
            raise ImportError("OllamaAPIManager requires the optional 'ollama' package.")
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.print_details = config.get("PRINT_API_CALL_DETAILS", False)
        self.truncation_length = config.get("API_RESPONSE_TRUNCATION_LENGTH", 50)
        self.client = ollama.Client(host=config.get("OLLAMA_BASE_URL", "http://localhost:11434"))
        
        max_concurrent = config.get("OLLAMA_MAX_CONCURRENT", 1) 
        self.semaphore = threading.Semaphore(max_concurrent)

    def generate_content(self, prompt: str, model_name: str, temperature: Optional[float] = None, avalai_role: Optional[str] = None) -> APIResponse:
        def attempt() -> APIResponse:
            with self.semaphore:
                try:
                    options: Dict[str, Any] = {}
                    if temperature is not None:
                        options["temperature"] = temperature
                    if self.config.get("OLLAMA_THINK_MODE"):
                        options["think"] = self.config["OLLAMA_THINK_MODE"]
                    _pace_provider_call(self.config, "ollama")
                    response = self.client.generate(model=model_name, prompt=prompt, options=options)
                    return {"status": "SUCCESS", "text": response["response"], "error_type": None, "error_message": None, "error_details": None}
                except ollama.ResponseError as error:
                    return {"status": "ERROR", "text": None, "error_type": "OllamaResponseError", "error_message": str(error), "error_details": repr(error)}
                except Exception as error:
                    return {"status": "ERROR", "text": None, "error_type": "OllamaConnectionError", "error_message": str(error), "error_details": repr(error)}

        return execute_with_retry(self.config, "ollama", model_name, prompt, attempt)
