# src/api_manager.py

"""
API management module for interacting with various LLM providers.

This module contains the API manager classes responsible for:
- Managing API credentials, endpoints, and key rotation.
- Handling provider-specific and global rate limiting.
- Making API calls with robust, granular error handling.
- Returning a standardized, structured response for consistent control flow.
- Providing detailed, configurable console logging for all API calls.

This rewritten version includes critical efficiency fixes for error handling,
improved rate-limiting logic, and enhanced documentation for maintainability.

Currently supported providers:
- GeminiAPIManager: For Google's Gemini models.
- AvalAIAPIManager: For OpenAI-compatible endpoints like AvalAI.
"""

import time
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any, TypedDict

import openai
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# Formally define the structured API response using TypedDict for clarity and static analysis.
class APIResponse(TypedDict):
    """A standardized structure for all API call results."""
    status: str  # e.g., "SUCCESS", "ERROR", "BLOCKED", "RATE_LIMITED"
    text: Optional[str]
    error_type: Optional[str]
    error_message: Optional[str]
    error_details: Optional[Any]


class GeminiAPIManager:
    """
    Manages API keys, rate limits, and calls to the Google Gemini API with enhanced error handling.
    """
    def __init__(self, api_keys: List[str], model_quotas: Dict[str, Dict[str, Any]], global_delay_seconds: int = 0, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Gemini API manager.

        Args:
            api_keys (List[str]): A list of Gemini API keys for rotation.
            model_quotas (Dict): Defines per-model rate limits (delay_seconds, rpd).
            global_delay_seconds (int): A minimum delay between any two API calls, across all keys.
            config (Optional[Dict]): The main configuration dictionary to read control flags.
        """
        self.logger = logging.getLogger(__name__)

        if not api_keys:
            self.logger.critical("GeminiAPIManager initialized with an empty list of API keys. All calls will fail.")
            raise ValueError("API keys list cannot be empty for GeminiAPIManager.")

        self.api_keys_list: List[str] = api_keys
        self.model_quotas: Dict[str, Dict[str, Any]] = model_quotas
        self.global_delay_seconds: int = global_delay_seconds
        
        # Internal state for tracking key usage and rate limits
        self.key_usage_timestamps: Dict[Tuple[str, str], float] = {}
        self.key_daily_counts: Dict[Tuple[str, str, str], int] = {}
        self.last_global_call_timestamp: float = 0
        self.current_key_index: int = 0
        self._lock = False # A simple lock to prevent race conditions in key selection

        # Read control flags from config, with safe defaults
        cfg = config or {}
        self.print_details = cfg.get("PRINT_API_CALL_DETAILS", False)
        self.truncation_length = cfg.get("API_RESPONSE_TRUNCATION_LENGTH", 50)
        self._print_timing_checkpoints = cfg.get("PRINT_API_TIMING_CHECKPOINTS", False)
        self._last_checkpoint_timestamp: Optional[float] = None

        self.logger.info(f"GeminiAPIManager initialized with {len(self.api_keys_list)} keys.")
        try:
            genai.configure(api_key=self.api_keys_list[0])
            self.logger.info("Initial Gemini API configuration with the first key was successful.")
        except Exception as e:
            self.logger.error(f"Initial Gemini API configuration failed. Error: {e}", exc_info=True)

    def _get_current_date_str(self) -> str:
        """Returns the current UTC date as a formatted string ('YYYY-MM-DD')."""
        return datetime.utcnow().strftime('%Y-%m-%d')

    def _get_available_key_and_sleep_time(self, model_name: str) -> Tuple[Optional[str], float]:
        """
        Finds the next available API key and calculates the necessary sleep time.
        
        This method iterates through the key ring, checking both per-key/per-model
        delays and daily request quotas. It also respects the global delay.
        """
        if not self.api_keys_list:
            return None, 3600

        if self._lock:
            self.logger.warning("Key selection is locked; waiting.")
            return None, 5
        self._lock = True

        try:
            current_time = time.time()
            time_since_last_global = current_time - self.last_global_call_timestamp
            global_sleep_needed = max(0, self.global_delay_seconds - time_since_last_global)
            
            num_keys = len(self.api_keys_list)
            model_quotas = self.model_quotas.get(model_name, {})
            required_delay = model_quotas.get("delay_seconds", 1)
            max_rpd = model_quotas.get("rpd", float('inf'))

            for i in range(num_keys):
                key_idx = (self.current_key_index + i) % num_keys
                api_key = self.api_keys_list[key_idx]
                
                # Check daily quota
                date_str = self._get_current_date_str()
                daily_usage_key = (api_key, model_name, date_str)
                if self.key_daily_counts.get(daily_usage_key, 0) >= max_rpd:
                    continue # This key has hit its daily limit for this model

                # Check time-based delay
                last_call_key = (api_key, model_name)
                time_since_last_call = current_time - self.key_usage_timestamps.get(last_call_key, 0)
                per_key_sleep_needed = max(0, required_delay - time_since_last_call)
                
                # The final sleep time is the max of the global and per-key requirements
                final_sleep = max(global_sleep_needed, per_key_sleep_needed)
                self.current_key_index = (key_idx + 1) % num_keys
                return api_key, final_sleep

            self.logger.warning(f"All {num_keys} API keys are currently rate-limited for model '{model_name}'.")
            return None, 3600
        finally:
            self._lock = False

    def _record_api_call(self, api_key: str, model_name: str) -> None:
        """Records the timestamp and increments the daily count for a given key and model."""
        current_time = time.time()
        self.last_global_call_timestamp = current_time
        self.key_usage_timestamps[(api_key, model_name)] = current_time
        
        date_str = self._get_current_date_str()
        daily_usage_key = (api_key, model_name, date_str)
        self.key_daily_counts[daily_usage_key] = self.key_daily_counts.get(daily_usage_key, 0) + 1
        
    def generate_content(self, prompt: str, model_name: str, temperature: Optional[float] = None) -> APIResponse:
        """
        Generates content using the Gemini API, handling key selection, rate limiting, and errors.
        
        This method orchestrates the entire call process, from getting a valid key to
        handling specific exceptions and returning a standardized response.
        """
        if self.print_details:
            print("\n" + "--- [API Call Start: Gemini] ---")
            print(f"Model: {model_name}, Temperature: {temperature}")
            print("Prompt Sent (truncated):")
            print(f"{prompt[:self.truncation_length]}{'...' if len(prompt) > self.truncation_length else ''}")
            print("----------------------------------")

        api_key, sleep_time = self._get_available_key_and_sleep_time(model_name)

        if api_key is None:
            msg = f"All API keys are proactively rate-limited for model '{model_name}'."
            if self.print_details: print(f"\n!!! [API Call FAILED: Gemini] !!!\nError Type: ProactiveRateLimit\nDetails: {msg}\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            return {"status": "RATE_LIMITED", "text": None, "error_type": "ProactiveRateLimit", "error_message": msg, "error_details": None}

        if sleep_time > 0:
            self.logger.info(f"Rate limit requires sleeping for {sleep_time:.2f}s.")
            print(f"Sleeping for {sleep_time:.2f} seconds due to rate limiting.")
            time.sleep(sleep_time)

        # Timing checkpoint logic (occurs after sleep, before the call)
        call_start_time = time.time()
        if self._print_timing_checkpoints and self._last_checkpoint_timestamp is not None:
            elapsed = call_start_time - self._last_checkpoint_timestamp
            print(f"    [TIMING CHECKPOINT] Time since last API call started: {elapsed:.2f} seconds.")
        self._last_checkpoint_timestamp = call_start_time

        caught_exception = None
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            gen_config = genai.types.GenerationConfig(temperature=temperature) if temperature is not None else None
            
            self.logger.info(f"Calling Gemini model '{model_name}' with key ending in ...{api_key[-4:]}.")
            response = model.generate_content(prompt, generation_config=gen_config)
            
            # CRITICAL FIX: Record the call ONLY after a successful API response.
            self._record_api_call(api_key, model_name)

            if not response.parts:
                reason = "Unknown"
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason.name
                msg = f"Response was empty or blocked. Reason: {reason}."
                if self.print_details: print(f"\n!!! [API Call BLOCKED: Gemini] !!!\nReason: {reason}\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                return {"status": "BLOCKED", "text": None, "error_type": "Safety", "error_message": msg, "error_details": str(response.prompt_feedback)}
            
            response_text = response.text
            if self.print_details:
                print("--- [API Call SUCCESS: Gemini] ---")
                print(f"Response (truncated): {response_text[:self.truncation_length]}{'...' if len(response_text) > self.truncation_length else ''}")
                print("----------------------------------\n")
            
            return {"status": "SUCCESS", "text": response_text, "error_type": None, "error_message": None, "error_details": None}

        except google_exceptions.ResourceExhausted as e:
            status, etype, msg = "RATE_LIMITED", "ResourceExhausted", f"Gemini API rate limit exceeded: {e}"
            caught_exception = e
        except google_exceptions.InvalidArgument as e:
            status, etype, msg = "ERROR", "InvalidArgument", f"Invalid argument to Gemini API (check API key or prompt content): {e}"
            caught_exception = e
        except Exception as e:
            status, etype, msg = "ERROR", "UnknownError", f"An unexpected error occurred with the Gemini API: {e}"
            caught_exception = e
        
        self.logger.error(f"Gemini API call FAILED. Key: ...{api_key[-4:]}. Type: {etype}. Error: {msg}", exc_info=True)
        if self.print_details:
            print(f"\n!!! [API Call FAILED: Gemini] !!!")
            print(f"Model: {model_name}\nError Type: {etype}\nError Details:\n{repr(caught_exception)}")
            print("--- Prompt that caused the error ---\n" + prompt + "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        
        return {"status": status, "text": None, "error_type": etype, "error_message": msg, "error_details": repr(caught_exception)}


class AvalAIAPIManager:
    """
    Manages API calls to an OpenAI-compatible endpoint like AvalAI with robust error handling.
    """
    def __init__(self, api_key: str, base_url: str, model_quotas: Dict[str, Dict[str, Any]], global_delay_seconds: int = 0, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the OpenAI-compatible API manager.

        Args:
            api_key (str): The API key for the service.
            base_url (str): The base URL of the API endpoint.
            model_quotas (Dict): Configuration for rate limiting (e.g., delay_seconds).
            global_delay_seconds (int): A minimum delay between any two API calls.
            config (Optional[Dict]): The main configuration dictionary to read control flags.
        """
        self.logger = logging.getLogger(__name__)

        if not api_key or "YOUR_AVALAI_API_KEY" in api_key or not base_url:
            self.logger.critical("AvalAIAPIManager initialized with missing or placeholder API key/base URL.")
            raise ValueError("API key and base URL must be valid for AvalAIAPIManager.")
        
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_quotas = model_quotas
        self.global_delay_seconds = global_delay_seconds
        
        # Internal state for rate limiting
        self.last_global_call_timestamp: float = 0
        self.last_model_call_timestamps: Dict[str, float] = {}
        
        cfg = config or {}
        self.print_details = cfg.get("PRINT_API_CALL_DETAILS", False)
        self.truncation_length = cfg.get("API_RESPONSE_TRUNCATION_LENGTH", 50)
        self._print_timing_checkpoints = cfg.get("PRINT_API_TIMING_CHECKPOINTS", False)
        self._last_checkpoint_timestamp: Optional[float] = None
        
        self.logger.info(f"AvalAIAPIManager initialized for endpoint: {base_url}")

    def _apply_rate_limit_and_record(self, model_name: str) -> None:
        """
        Calculates and applies the necessary sleep time, then records timestamps for the upcoming call.
        """
        time_before_sleep = time.time()

        # Calculate required sleep based on time *before* sleeping
        global_sleep = max(0, self.global_delay_seconds - (time_before_sleep - self.last_global_call_timestamp))
        
        model_quotas = self.model_quotas.get(model_name, self.model_quotas.get("default", {}))
        model_delay = model_quotas.get("delay_seconds", 1)
        model_sleep = max(0, model_delay - (time_before_sleep - self.last_model_call_timestamps.get(model_name, 0)))
        
        final_sleep = max(global_sleep, model_sleep)
        
        if final_sleep > 0:
            self.logger.info(f"Rate limit requires sleeping for {final_sleep:.2f}s (Global: {global_sleep:.2f}s, Model: {model_sleep:.2f}s).")
            print(f"Sleeping for {final_sleep:.2f} seconds due to rate limiting.")
            time.sleep(final_sleep)

        # Get the timestamp *after* any sleep to mark the true start of the call attempt
        call_start_time = time.time()
        
        # Timing checkpoint logic
        if self._print_timing_checkpoints and self._last_checkpoint_timestamp is not None:
            elapsed = call_start_time - self._last_checkpoint_timestamp
            print(f"    [TIMING CHECKPOINT] Time since last API call started: {elapsed:.2f} seconds.")
        
        # Record timestamps for rate-limiting *after* sleeping but *before* the API call.
        self._last_checkpoint_timestamp = call_start_time
        self.last_global_call_timestamp = call_start_time
        self.last_model_call_timestamps[model_name] = call_start_time

    def generate_content(self, prompt: str, model_name: str, temperature: Optional[float] = None) -> APIResponse:
        """
        Generates content using an OpenAI-compatible API, handling rate limiting and errors.
        """
        self._apply_rate_limit_and_record(model_name)

        if self.print_details:
            print("\n" + "--- [API Call Start: AvalAI] ---")
            print(f"Model: {model_name}, Temperature: {temperature}")
            print("Prompt Sent (truncated):")
            print(f"{prompt[:self.truncation_length]}{'...' if len(prompt) > self.truncation_length else ''}")
            print("----------------------------------")

        caught_exception = None
        try:
            self.logger.info(f"Calling OpenAI-compatible model '{model_name}'.")
            
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )

            if not completion.choices:
                msg = "Response was empty or blocked (no choices returned)."
                if self.print_details: print(f"\n!!! [API Call BLOCKED: AvalAI] !!!\nReason: {msg}\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                return {"status": "BLOCKED", "text": None, "error_type": "NoChoices", "error_message": msg, "error_details": None}

            response_text = completion.choices[0].message.content
            if self.print_details:
                print("--- [API Call SUCCESS: AvalAI] ---")
                print(f"Response (truncated): {response_text[:self.truncation_length]}{'...' if len(response_text) > self.truncation_length else ''}")
                print("----------------------------------\n")

            return {"status": "SUCCESS", "text": response_text, "error_type": None, "error_message": None, "error_details": None}

        except openai.RateLimitError as e:
            status, etype, msg = "RATE_LIMITED", "RateLimitError", f"OpenAI API rate limit exceeded: {e}"
            caught_exception = e
        except openai.APIStatusError as e:
            status, etype, msg = "ERROR", "APIStatusError", f"OpenAI API returned an error status {e.status_code}: {e.response}"
            caught_exception = e
        except openai.APITimeoutError as e:
            status, etype, msg = "ERROR", "APITimeoutError", f"OpenAI API request timed out: {e}"
            caught_exception = e
        except openai.APIConnectionError as e:
            status, etype, msg = "ERROR", "APIConnectionError", f"Failed to connect to OpenAI API: {e}"
            caught_exception = e
        except Exception as e:
            status, etype, msg = "ERROR", "UnknownError", f"An unexpected error occurred with the OpenAI API: {e}"
            caught_exception = e

        self.logger.error(f"OpenAI API call FAILED. Type: {etype}. Error: {msg}", exc_info=True)
        if self.print_details:
            print(f"\n!!! [API Call FAILED: AvalAI] !!!")
            print(f"Model: {model_name}\nError Type: {etype}\nError Details:\n{repr(caught_exception)}")
            print("--- Prompt that caused the error ---\n" + prompt + "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        
        return {"status": status, "text": None, "error_type": etype, "error_message": msg, "error_details": repr(caught_exception)}