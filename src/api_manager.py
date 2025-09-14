# src/api_manager.py

"""
API management module for interacting with the Google Gemini API.

This file contains the GeminiAPIManager class, which is responsible for:
- Managing a pool of API keys.
- Handling rate limiting (per-key, daily, and global).
- Rotating keys in a round-robin fashion.
- Making API calls with robust error handling.
- Returning structured responses for better control flow in the pipeline.
"""

import time
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any
import google.generativeai as genai

# Define a type hint for the structured response for clarity.
GeminiResponse = Dict[str, Any]

class GeminiAPIManager:
    """
    Manages API keys, rate limits, and calls to the Gemini API.
    """
    def __init__(self, api_keys: List[str], model_quotas: Dict[str, Dict[str, Any]], global_delay_seconds: int = 0):
        """
        Initializes the API manager.

        Args:
            api_keys (List[str]): A list of Gemini API keys.
            model_quotas (Dict): A dictionary defining per-model rate limits (delay, rpd).
            global_delay_seconds (int): A minimum delay between any two API calls.
        """
        self.logger = logging.getLogger(__name__) # Get a logger instance for this module.

        if not api_keys:
            self.logger.critical("GeminiAPIManager initialized with an empty list of API keys. All calls will fail.")
            raise ValueError("API keys list cannot be empty.")
        
        self.api_keys_list: List[str] = api_keys
        self.model_quotas: Dict[str, Dict[str, Any]] = model_quotas
        self.global_delay_seconds: int = global_delay_seconds
        
        # Internal state for tracking key usage
        self.key_usage_timestamps: Dict[Tuple[str, str], float] = {}  # (api_key, model_name) -> last_call_timestamp
        self.key_daily_counts: Dict[Tuple[str, str, str], int] = {}    # (api_key, model_name, date_str) -> count
        self.last_global_call_timestamp: float = 0
        
        self.current_key_index: int = 0
        self._lock = False # A simple, non-thread-safe lock for key rotation.

        self.logger.info(f"GeminiAPIManager initialized with {len(self.api_keys_list)} keys.")
        
        # Perform an initial configuration test with the first key.
        try:
            genai.configure(api_key=self.api_keys_list[0])
            self.logger.info("Initial Gemini API configuration with the first key was successful.")
        except Exception as e:
            self.logger.error(f"Initial Gemini API configuration failed. Error: {e}", exc_info=True)
            # We don't raise an error here, allowing initialization, but calls will likely fail.

    def _get_current_date_str(self) -> str:
        """Returns the current UTC date as a formatted string."""
        return datetime.utcnow().strftime('%Y-%m-%d')

    def _get_available_key_and_sleep_time(self, model_name: str) -> Tuple[Optional[str], float]:
        """
        Finds an available API key and calculates the necessary sleep time.
        This is the core rate-limiting logic.
        """
        if not self.api_keys_list:
            return None, 3600 # No keys to use.

        if self._lock:
            self.logger.warning("Key selection is locked; another process is likely choosing a key. Waiting.")
            return None, 5 # Wait a short time if locked.

        self._lock = True
        
        current_time_val = time.time()
        
        # --- 1. Global Delay Calculation ---
        time_since_last_global_call = current_time_val - self.last_global_call_timestamp
        global_sleep_needed = max(0, self.global_delay_seconds - time_since_last_global_call)
        
        num_keys = len(self.api_keys_list)
        start_index = self.current_key_index

        model_specific_quotas = self.model_quotas.get(model_name, {})
        required_per_key_delay = model_specific_quotas.get("delay_seconds", 1)
        max_rpd = model_specific_quotas.get("rpd", float('inf'))

        # --- 2. Iterate through keys to find a valid one ---
        for i in range(num_keys):
            key_idx = (start_index + i) % num_keys
            current_api_key = self.api_keys_list[key_idx]
            
            # Check daily limit (RPD - Requests Per Day)
            current_date_str = self._get_current_date_str()
            daily_usage_key = (current_api_key, model_name, current_date_str)
            current_daily_calls = self.key_daily_counts.get(daily_usage_key, 0)

            if current_daily_calls >= max_rpd:
                self.logger.debug(f"Key ...{current_api_key[-4:]} has reached its daily limit for {model_name}.")
                continue # Try the next key.

            # Check per-key delay (RPM - Requests Per Minute, translated to seconds)
            last_call_timestamp_key = (current_api_key, model_name)
            last_call_time = self.key_usage_timestamps.get(last_call_timestamp_key, 0)
            time_since_last_call = current_time_val - last_call_time
            per_key_sleep_needed = max(0, required_per_key_delay - time_since_last_call)
            
            # The final sleep duration is the maximum of the global and per-key requirements.
            final_sleep_duration = max(global_sleep_needed, per_key_sleep_needed)

            self.current_key_index = (key_idx + 1) % num_keys # Rotate to the next key for the next call
            self._lock = False
            return current_api_key, final_sleep_duration

        # If the loop completes, no key was available.
        self._lock = False
        self.logger.warning(f"All {num_keys} API keys are currently rate-limited or have exceeded daily quotas for model '{model_name}'.")
        return None, 3600 # Return a long sleep time.

    def _record_api_call(self, api_key: str, model_name: str) -> None:
        """Records the timestamp and increments the daily count for a given key and model."""
        current_time = time.time()
        self.last_global_call_timestamp = current_time
        
        # Record timestamp for per-key delay
        self.key_usage_timestamps[(api_key, model_name)] = current_time
        
        # Increment daily count
        current_date_str = self._get_current_date_str()
        daily_usage_key = (api_key, model_name, current_date_str)
        self.key_daily_counts[daily_usage_key] = self.key_daily_counts.get(daily_usage_key, 0) + 1
        
    def generate_content(self, prompt: str, model_name: str, temperature: Optional[float] = None) -> GeminiResponse:
        """
        Generates content using the Gemini API, handling key selection, rate limiting, and errors.
        This is the primary public method for this class.

        Args:
            prompt (str): The prompt to send to the model.
            model_name (str): The name of the model to use.
            temperature (Optional[float]): The generation temperature.

        Returns:
            GeminiResponse: A structured dictionary containing the status, text, and any errors.
        """
        api_key, sleep_time = self._get_available_key_and_sleep_time(model_name)

        if api_key is None:
            return {
                "status": "FAILURE", "text": None,
                "error_message": f"All API keys are currently rate-limited for model '{model_name}'. Please wait.",
                "model_name": model_name, "api_key_used": None, "raw_response": None
            }

        key_for_log = f"...{api_key[-4:]}"
        self.logger.info(f"Selected API key {key_for_log} for model '{model_name}'.")

        if sleep_time > 0:
            self.logger.info(f"Rate limit requires sleeping for {sleep_time:.2f}s.")
            print(f"Sleeping for {sleep_time:.2f} seconds due to rate limiting.")
            time.sleep(sleep_time)

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)

            generation_config = genai.types.GenerationConfig(temperature=temperature) if temperature is not None else None
            
            self.logger.info(f"Calling model '{model_name}' with temp={temperature if temperature is not None else 'default'}.")
            response = model.generate_content(prompt, generation_config=generation_config)
            
            # An API call attempt was made, so record it.
            self._record_api_call(api_key, model_name)

            if not response.parts:
                block_reason = "Unknown"
                if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason'):
                    block_reason = response.prompt_feedback.block_reason.name
                self.logger.warning(f"API call with key {key_for_log} was BLOCKED. Reason: {block_reason}.")
                return {
                    "status": "BLOCKED", "text": None,
                    "error_message": f"Response was empty or blocked. Reason: {block_reason}.",
                    "model_name": model_name, "api_key_used": key_for_log, "raw_response": response
                }
            
            self.logger.info(f"API call with key {key_for_log} successful.")
            print(f"    LLM Response (truncated): {response.text[:100]}...")
            return {
                "status": "SUCCESS", "text": response.text, "error_message": None,
                "model_name": model_name, "api_key_used": key_for_log, "raw_response": response
            }

        except Exception as e:
            # --- NEW: Enhanced Error Printing ---
            print("\n" + "="*80)
            print("!!! API CALL ERROR !!!")
            print(f"    - Error Type: {type(e).__name__}")
            print(f"    - Error Message: {e}")
            print("    - Prompt that caused the error (first 500 chars):")
            print(f"    - \"\"\"{prompt[:500]}...\"\"\"")
            print("="*80 + "\n")
            # --- END NEW ---
            
            self.logger.error(f"API call with key {key_for_log} FAILED. Error: {type(e).__name__} - {e}", exc_info=True)
            # Record the attempt even if it failed to prevent hammering a failing key.
            self._record_api_call(api_key, model_name)
            return {
                "status": "ERROR", "text": None,
                "error_message": f"An API error occurred: {type(e).__name__} - {e}",
                "model_name": model_name, "api_key_used": key_for_log, "raw_response": None
            }