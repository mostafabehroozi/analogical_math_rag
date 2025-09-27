# src/api_manager.py

"""
API management module for interacting with various LLM providers.

This file contains the API manager classes responsible for:
- Managing API credentials and endpoints.
- Handling rate limiting.
- Making API calls with robust error handling.
- Returning a standardized, structured response for consistent control flow.
- NEW: Provides detailed, configurable console logging for all API calls.

Currently supported providers:
- GeminiAPIManager: For Google's Gemini models.
- AvalAIAPIManager: For OpenAI-compatible endpoints like AvalAI.
"""

import time
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any

import openai
import google.generativeai as genai

# Define a generic type hint for the structured response for clarity.
APIResponse = Dict[str, Any]


class GeminiAPIManager:
    """
    Manages API keys, rate limits, and calls to the Google Gemini API.
    """
    # MODIFIED: Added config parameter to __init__
    def __init__(self, api_keys: List[str], model_quotas: Dict[str, Dict[str, Any]], global_delay_seconds: int = 0, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Gemini API manager.

        Args:
            api_keys (List[str]): A list of Gemini API keys.
            model_quotas (Dict): A dictionary defining per-model rate limits (delay, rpd).
            global_delay_seconds (int): A minimum delay between any two API calls.
            config (Optional[Dict]): The main configuration dictionary to read control flags.
        """
        self.logger = logging.getLogger(__name__)

        if not api_keys:
            self.logger.critical("GeminiAPIManager initialized with an empty list of API keys. All calls will fail.")
            raise ValueError("API keys list cannot be empty.")
        
        self.api_keys_list: List[str] = api_keys
        self.model_quotas: Dict[str, Dict[str, Any]] = model_quotas
        self.global_delay_seconds: int = global_delay_seconds
        
        # Internal state for tracking key usage
        self.key_usage_timestamps: Dict[Tuple[str, str], float] = {}
        self.key_daily_counts: Dict[Tuple[str, str, str], int] = {}
        self.last_global_call_timestamp: float = 0
        
        self.current_key_index: int = 0
        self._lock = False

        # NEW: Store the print details flag from config
        self.print_details = config.get("PRINT_API_CALL_DETAILS", False) if config else False

        self.logger.info(f"GeminiAPIManager initialized with {len(self.api_keys_list)} keys.")
        
        try:
            genai.configure(api_key=self.api_keys_list[0])
            self.logger.info("Initial Gemini API configuration with the first key was successful.")
        except Exception as e:
            self.logger.error(f"Initial Gemini API configuration failed. Error: {e}", exc_info=True)

    def _get_current_date_str(self) -> str:
        """Returns the current UTC date as a formatted string."""
        return datetime.utcnow().strftime('%Y-%m-%d')

    def _get_available_key_and_sleep_time(self, model_name: str) -> Tuple[Optional[str], float]:
        """Finds an available API key and calculates the necessary sleep time."""
        if not self.api_keys_list:
            return None, 3600

        if self._lock:
            self.logger.warning("Key selection is locked; waiting.")
            return None, 5

        self._lock = True
        current_time_val = time.time()
        time_since_last_global_call = current_time_val - self.last_global_call_timestamp
        global_sleep_needed = max(0, self.global_delay_seconds - time_since_last_global_call)
        
        num_keys = len(self.api_keys_list)
        start_index = self.current_key_index
        model_specific_quotas = self.model_quotas.get(model_name, {})
        required_per_key_delay = model_specific_quotas.get("delay_seconds", 1)
        max_rpd = model_specific_quotas.get("rpd", float('inf'))

        for i in range(num_keys):
            key_idx = (start_index + i) % num_keys
            current_api_key = self.api_keys_list[key_idx]
            
            current_date_str = self._get_current_date_str()
            daily_usage_key = (current_api_key, model_name, current_date_str)
            current_daily_calls = self.key_daily_counts.get(daily_usage_key, 0)
            if current_daily_calls >= max_rpd:
                continue

            last_call_timestamp_key = (current_api_key, model_name)
            last_call_time = self.key_usage_timestamps.get(last_call_timestamp_key, 0)
            time_since_last_call = current_time_val - last_call_time
            per_key_sleep_needed = max(0, required_per_key_delay - time_since_last_call)
            
            final_sleep_duration = max(global_sleep_needed, per_key_sleep_needed)
            self.current_key_index = (key_idx + 1) % num_keys
            self._lock = False
            return current_api_key, final_sleep_duration

        self._lock = False
        self.logger.warning(f"All {num_keys} API keys are rate-limited for model '{model_name}'.")
        return None, 3600

    def _record_api_call(self, api_key: str, model_name: str) -> None:
        """Records the timestamp and increments the daily count for a given key and model."""
        current_time = time.time()
        self.last_global_call_timestamp = current_time
        self.key_usage_timestamps[(api_key, model_name)] = current_time
        current_date_str = self._get_current_date_str()
        daily_usage_key = (api_key, model_name, current_date_str)
        self.key_daily_counts[daily_usage_key] = self.key_daily_counts.get(daily_usage_key, 0) + 1
        
    def generate_content(self, prompt: str, model_name: str, temperature: Optional[float] = None) -> APIResponse:
        """Generates content using the Gemini API, handling key selection, rate limiting, and errors."""
        
        # NEW: Print request details if enabled
        if self.print_details:
            print("\n" + "--- [API Call Start: Gemini] ---")
            print(f"Model: {model_name}, Temperature: {temperature}")
            print("Prompt Sent:")
            print(prompt)
            print("----------------------------------")

        api_key, sleep_time = self._get_available_key_and_sleep_time(model_name)

        if api_key is None:
            error_msg = f"All API keys are rate-limited for model '{model_name}'. Please wait."
            if self.print_details:
                print("\n" + "!!! [API Call FAILED: Gemini] !!!")
                print(f"Model: {model_name}")
                print(f"Error Type: Rate Limiting")
                print(f"Error Details: {error_msg}")
                print("--- Prompt that caused the error ---")
                print(prompt)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            return {"status": "FAILURE", "text": None, "error_message": error_msg}

        key_for_log = f"...{api_key[-4:]}"
        if sleep_time > 0:
            self.logger.info(f"Rate limit requires sleeping for {sleep_time:.2f}s.")
            print(f"Sleeping for {sleep_time:.2f} seconds due to rate limiting.")
            time.sleep(sleep_time)

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            generation_config = genai.types.GenerationConfig(temperature=temperature) if temperature is not None else None
            
            self.logger.info(f"Calling Gemini model '{model_name}' with key {key_for_log}.")
            response = model.generate_content(prompt, generation_config=generation_config)
            self._record_api_call(api_key, model_name)

            if not response.parts:
                block_reason = response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback, 'block_reason') else "Unknown"
                if self.print_details:
                    print("\n" + "!!! [API Call BLOCKED: Gemini] !!!")
                    print(f"Reason: {block_reason}")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                return {"status": "BLOCKED", "text": None, "error_message": f"Response was empty or blocked. Reason: {block_reason}."}
            
            if self.print_details:
                print("--- [API Call SUCCESS: Gemini] ---")
                print(f"Response (truncated): {response.text[:200]}...")
                print("----------------------------------\n")
            
            return {"status": "SUCCESS", "text": response.text, "error_message": None}

        except Exception as e:
            self.logger.error(f"Gemini API call with key {key_for_log} FAILED. Error: {e}", exc_info=True)
            self._record_api_call(api_key, model_name)
            if self.print_details:
                print("\n" + "!!! [API Call FAILED: Gemini] !!!")
                print(f"Model: {model_name}")
                print(f"Error Type: {type(e).__name__}")
                print(f"Error Details (Full, Un-truncated):\n{repr(e)}")
                print("--- Prompt that caused the error ---")
                print(prompt)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            return {"status": "ERROR", "text": None, "error_message": f"An API error occurred: {type(e).__name__} - {e}"}


# --- NEW: Manager for OpenAI-compatible APIs like AvalAI ---
class AvalAIAPIManager:
    """
    Manages API calls to an OpenAI-compatible endpoint like AvalAI.
    """
    # MODIFIED: Added config parameter to __init__
    def __init__(self, api_key: str, base_url: str, model_quotas: Dict[str, Dict[str, Any]], config: Optional[Dict[str, Any]] = None):
        """
        Initializes the OpenAI-compatible API manager.

        Args:
            api_key (str): The API key for the service.
            base_url (str): The base URL of the API endpoint.
            model_quotas (Dict): Configuration for rate limiting (e.g., delay).
            config (Optional[Dict]): The main configuration dictionary to read control flags.
        """
        self.logger = logging.getLogger(__name__)

        if not api_key or not base_url:
            self.logger.critical("AvalAIAPIManager initialized with missing API key or base URL.")
            raise ValueError("API key and base URL cannot be empty for AvalAIAPIManager.")
        
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_quotas = model_quotas
        self.last_call_timestamp: float = 0
        
        # NEW: Store the print details flag from config
        self.print_details = config.get("PRINT_API_CALL_DETAILS", False) if config else False
        
        self.logger.info(f"AvalAIAPIManager initialized for endpoint: {base_url}")

    def _apply_rate_limit(self) -> None:
        """Applies a simple delay-based rate limit."""
        delay_seconds = self.model_quotas.get("default", {}).get("delay_seconds", 1)
        
        time_since_last_call = time.time() - self.last_call_timestamp
        sleep_needed = max(0, delay_seconds - time_since_last_call)
        
        if sleep_needed > 0:
            self.logger.info(f"Rate limit requires sleeping for {sleep_needed:.2f}s.")
            print(f"Sleeping for {sleep_needed:.2f} seconds due to rate limiting.")
            time.sleep(sleep_needed)

    def generate_content(self, prompt: str, model_name: str, temperature: Optional[float] = None) -> APIResponse:
        """
        Generates content using an OpenAI-compatible API, handling rate limiting and errors.
        """
        self._apply_rate_limit()

        # NEW: Print request details if enabled
        if self.print_details:
            print("\n" + "--- [API Call Start: AvalAI] ---")
            print(f"Model: {model_name}, Temperature: {temperature}")
            print("Prompt Sent:")
            print(prompt)
            print("----------------------------------")

        try:
            self.logger.info(f"Calling OpenAI-compatible model '{model_name}'.")
            
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            
            self.last_call_timestamp = time.time()

            if not completion.choices:
                self.logger.warning(f"API call to '{model_name}' returned no choices.")
                if self.print_details:
                    print("\n" + "!!! [API Call BLOCKED: AvalAI] !!!")
                    print("Reason: No choices were returned by the API.")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                return {"status": "BLOCKED", "text": None, "error_message": "Response was empty or blocked (no choices returned)."}

            response_text = completion.choices[0].message.content
            if self.print_details:
                print("--- [API Call SUCCESS: AvalAI] ---")
                print(f"Response (truncated): {response_text[:200]}...")
                print("----------------------------------\n")

            return {"status": "SUCCESS", "text": response_text, "error_message": None}

        except openai.APIError as e:
            self.logger.error(f"OpenAI API call FAILED. Status: {e.status_code}, Response: {e.response}", exc_info=True)
            self.last_call_timestamp = time.time()
            if self.print_details:
                print("\n" + "!!! [API Call FAILED: AvalAI (APIError)] !!!")
                print(f"Model: {model_name}")
                print(f"Status Code: {e.status_code}")
                print(f"Error Details (Full, Un-truncated Response from Server):\n{e.response}")
                print("--- Prompt that caused the error ---")
                print(prompt)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            return {"status": "ERROR", "text": None, "error_message": f"An API error occurred: {type(e).__name__} - {e}"}
        
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during the OpenAI API call: {e}", exc_info=True)
            self.last_call_timestamp = time.time()
            if self.print_details:
                print("\n" + "!!! [API Call FAILED: AvalAI (General Error)] !!!")
                print(f"Model: {model_name}")
                print(f"Error Type: {type(e).__name__}")
                print(f"Error Details (Full, Un-truncated):\n{repr(e)}")
                print("--- Prompt that caused the error ---")
                print(prompt)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            return {"status": "ERROR", "text": None, "error_message": f"An unexpected error occurred: {type(e).__name__} - {e}"}