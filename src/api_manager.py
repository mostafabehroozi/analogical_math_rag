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
import inspect
import os
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
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
except ModuleNotFoundError:
    genai = None

RETRYABLE_ERROR_TYPES = {
    "APITimeoutError", "APIConnectionError", "ResourceExhausted",
    "OllamaConnectionError", "UnknownError", "APIStatusError",
    "RateLimitError", 
}
NON_RETRYABLE_ERROR_TYPES = {"AuthenticationError", "InvalidArgument", "Safety", "NoChoices", "ModelMismatch", "ProactiveRateLimit"}

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
                if now >= self._pause_until:
                    break
                wait_time = self._pause_until - now
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

                self._condition.wait(timeout=max(0.001, min(waits)))

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
    
    for attempt in range(1, max_attempts + 1):
        
        # Check if the system is globally paused before making ANY request
        global_pause_manager.wait_if_needed(config)
        
        # --- NEW: Enforce minimum time between consecutive API calls per provider ---
        if config.get("ENABLE_MIN_TIME_BETWEEN_API_CALLS", False):
            _min_gap = float(config.get("MIN_TIME_BETWEEN_API_CALLS_SECONDS", 0.0))
            if _min_gap > 0:
                provider_pacer.pace(provider_name, _min_gap)
        
        # 1. Start the Timer
        start_time = time.monotonic()
        
        # 2. Make the API Call
        response = api_call_func()
        
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
        
        if error_type in NON_RETRYABLE_ERROR_TYPES:
            retryable = False
        else:
            retryable = retry_all or (error_type in RETRYABLE_ERROR_TYPES)

        if not retryable or attempt == max_attempts:
            response["retry_exhausted"] = attempt == max_attempts
            tprint(f"[API ERROR] Final failure. Type: {error_type} @ {caller_loc}", level="ERROR")
            return response
        
        # Wait before retrying
        time.sleep(float(config.get("API_RETRY_DELAY_SECONDS", 20.0)))
            
        last_response = response
        
    final_resp = last_response or {"status": "ERROR", "text": None, "error_type": "RetryError", "error_message": "No API attempt made"}
    return final_resp


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
            "query_idx": ctx_query_idx.get(),
            "batch_id": ctx_batch_id.get(),
        }

    def _print(self, message: str) -> None:
        if self.print_details:
            tprint(f"[API {self.provider_name}] {message}", level="DEBUG")


class GeminiAPIManager(_KeyedAPIManager):
    """Gemini manager upgraded to use the new google-genai SDK with Thinking support."""

    provider_name = "gemini"

    def __init__(self, api_keys: List[str], model_quotas: Dict[str, Any], global_delay_seconds: int = 0, config: Optional[Dict[str, Any]] = None):
        super().__init__(api_keys, model_quotas, config)
        if genai is None:
            raise ImportError("GeminiAPIManager requires the 'google-genai' package. Run: pip install google-genai")
        
        self.timeout_seconds = float(self.config.get("API_REQUEST_TIMEOUT_SECONDS", 180.0))
        self.clients = {
            key: genai.Client(
                api_key=key, 
                http_options={'timeout': int(self.timeout_seconds * 1000)}
            ) for key in self.api_keys_list
        }
        
        if global_delay_seconds:
            self.logger.info("GLOBAL_API_CALL_DELAY_SECONDS is ignored by the RPM scheduler.")

    def _get_max_tokens(self, model_name: str) -> Optional[int]:
        if model_name == self.config.get("GEMINI_MODEL_NAME_FINAL_SOLVER"):
            return self.config.get("DEFAULT_FINAL_SOLVER_MAX_TOKENS", 8192)
        elif model_name == self.config.get("GEMINI_MODEL_NAME_ADAPTATION"):
            return self.config.get("DEFAULT_ADAPTATION_MAX_TOKENS", 2048)
        elif model_name == self.config.get("GEMINI_MODEL_NAME_EVALUATOR"):
            return self.config.get("DEFAULT_EVALUATOR_MAX_TOKENS", 512)
        return None

    def generate_content(self, prompt: str, model_name: str, temperature: Optional[float] = None) -> APIResponse:
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

    def __init__(self, api_key_or_list: Union[str, List[str]], base_url: str, model_quotas: Dict[str, Any], global_delay_seconds: int = 0, config: Optional[Dict[str, Any]] = None):
        if openai is None:
            raise ImportError("AvalAIAPIManager requires the optional 'openai' package.")
        keys = [api_key_or_list] if isinstance(api_key_or_list, str) else list(api_key_or_list)
        if not base_url:
            raise ValueError("base_url cannot be empty for AvalAIAPIManager.")
        super().__init__(keys, model_quotas, config)
        
        timeout_seconds = float(self.config.get("API_REQUEST_TIMEOUT_SECONDS", 180.0))
        self.clients = {
            key: openai.OpenAI(
                api_key=key, 
                base_url=base_url,
                timeout=timeout_seconds 
            ) for key in self.api_keys_list
        }
        
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
        
        import threading
        max_concurrent = config.get("OLLAMA_MAX_CONCURRENT", 1) 
        self.semaphore = threading.Semaphore(max_concurrent)

    def generate_content(self, prompt: str, model_name: str, temperature: Optional[float] = None) -> APIResponse:
        def attempt() -> APIResponse:
            with self.semaphore:
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

        return execute_with_retry(self.config, "ollama", model_name, prompt, attempt)