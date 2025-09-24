# src/avalai_manager.py

"""
API management module for interacting with OpenAI-compatible APIs like AvalAI.

This file contains the AvalAIManager class, which is responsible for:
- Managing a single API key and base URL.
- Handling rate limiting (per-model, daily, and global).
- Making API calls using the 'openai' library with robust error handling.
- Returning structured responses identical to GeminiAPIManager for seamless
  integration into the pipeline.
"""

import time
import logging
from datetime import datetime
from typing import Dict, Optional, Any

# The openai library is required for this manager.
# Ensure it's installed via: pip install openai
try:
    from openai import OpenAI, APIError
except ImportError:
    raise ImportError("The 'openai' library is not installed. Please install it with 'pip install openai' to use AvalAIManager.")

# Define a type hint for the structured response for clarity and consistency.
AvalAIResponse = Dict[str, Any]

class AvalAIManager:
    """
    Manages API key, rate limits, and calls to an OpenAI-compatible API.
    """
    def __init__(self, api_key: str, base_url: str, model_quotas: Dict[str, Dict[str, Any]], global_delay_seconds: int = 0):
        """
        Initializes the API manager for an OpenAI-compatible endpoint.

        Args:
            api_key (str): The API key for the service.
            base_url (str): The base URL of the API endpoint.
            model_quotas (Dict): A dictionary defining per-model rate limits (delay, rpd).
            global_delay_seconds (int): A minimum delay between any two API calls.
        """
        self.logger = logging.getLogger(__name__)

        if not api_key or not base_url:
            self.logger.critical("AvalAIManager initialized with an empty API key or base URL. All calls will fail.")
            raise ValueError("API key and base URL cannot be empty.")

        self.api_key = api_key
        self.base_url = base_url
        self.model_quotas = model_quotas
        self.global_delay_seconds = global_delay_seconds

        # Internal state for tracking usage
        self.model_usage_timestamps: Dict[str, float] = {}  # model_name -> last_call_timestamp
        self.model_daily_counts: Dict[tuple[str, str], int] = {}    # (model_name, date_str) -> count
        self.last_global_call_timestamp: float = 0

        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.logger.info(f"AvalAIManager initialized. Client configured for base URL: {self.base_url}")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client. Error: {e}", exc_info=True)
            self.client = None # Ensure client is None if initialization fails

    def _get_current_date_str(self) -> str:
        """Returns the current UTC date as a formatted string."""
        return datetime.utcnow().strftime('%Y-%m-%d')

    def _get_sleep_time(self, model_name: str) -> float:
        """Calculates the necessary sleep time based on rate limits."""
        current_time_val = time.time()

        # 1. Global Delay Calculation
        time_since_last_global_call = current_time_val - self.last_global_call_timestamp
        global_sleep_needed = max(0, self.global_delay_seconds - time_since_last_global_call)

        model_specific_quotas = self.model_quotas.get(model_name, {})
        required_per_model_delay = model_specific_quotas.get("delay_seconds", 1)
        max_rpd = model_specific_quotas.get("rpd", float('inf'))

        # 2. Check daily limit (RPD - Requests Per Day)
        current_date_str = self._get_current_date_str()
        daily_usage_key = (model_name, current_date_str)
        current_daily_calls = self.model_daily_counts.get(daily_usage_key, 0)

        if current_daily_calls >= max_rpd:
            self.logger.warning(f"Model '{model_name}' has reached its daily limit of {max_rpd} requests.")
            return 3600  # Return a long sleep time to signal limit reached

        # 3. Check per-model delay
        last_call_time = self.model_usage_timestamps.get(model_name, 0)
        time_since_last_call = current_time_val - last_call_time
        per_model_sleep_needed = max(0, required_per_model_delay - time_since_last_call)

        # The final sleep duration is the maximum of the global and per-model requirements.
        return max(global_sleep_needed, per_model_sleep_needed)

    def _record_api_call(self, model_name: str) -> None:
        """Records the timestamp and increments the daily count for a given model."""
        current_time = time.time()
        self.last_global_call_timestamp = current_time
        self.model_usage_timestamps[model_name] = current_time

        current_date_str = self._get_current_date_str()
        daily_usage_key = (model_name, current_date_str)
        self.model_daily_counts[daily_usage_key] = self.model_daily_counts.get(daily_usage_key, 0) + 1

    def generate_content(self, prompt: str, model_name: str, temperature: Optional[float] = None) -> AvalAIResponse:
        """
        Generates content using the OpenAI-compatible API, handling rate limiting and errors.
        This is the primary public method and mirrors GeminiAPIManager.

        Args:
            prompt (str): The prompt to send to the model.
            model_name (str): The name of the model to use.
            temperature (Optional[float]): The generation temperature.

        Returns:
            AvalAIResponse: A structured dictionary containing the status, text, and any errors.
        """
        if not self.client:
            return {"status": "FAILURE", "text": None, "error_message": "OpenAI client is not initialized."}

        sleep_time = self._get_sleep_time(model_name)

        if sleep_time >= 3600: # Check for the daily limit signal
            return {
                "status": "FAILURE", "text": None,
                "error_message": f"Daily request limit reached for model '{model_name}'. Please wait.",
                "model_name": model_name, "raw_response": None
            }

        if sleep_time > 0:
            self.logger.info(f"Rate limit requires sleeping for {sleep_time:.2f}s.")
            print(f"Sleeping for {sleep_time:.2f} seconds due to rate limiting.")
            time.sleep(sleep_time)

        try:
            messages = [{"role": "user", "content": prompt}]
            
            self.logger.info(f"Calling model '{model_name}' with temp={temperature if temperature is not None else 'default'}.")
            
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature
            )
            
            self._record_api_call(model_name)

            if not completion.choices or not completion.choices[0].message.content:
                finish_reason = completion.choices[0].finish_reason if completion.choices else "unknown"
                self.logger.warning(f"API call to '{model_name}' was successful but returned no content. Finish reason: {finish_reason}.")
                return {
                    "status": "BLOCKED", "text": None,
                    "error_message": f"Response was empty or blocked by content filters. Finish reason: {finish_reason}.",
                    "model_name": model_name, "raw_response": completion
                }

            response_text = completion.choices[0].message.content
            self.logger.info(f"API call to '{model_name}' successful.")
            print(f"    LLM Response (truncated): {response_text[:100]}...")
            return {
                "status": "SUCCESS", "text": response_text, "error_message": None,
                "model_name": model_name, "raw_response": completion
            }

        except APIError as e:
            error_message = f"An API error occurred: {e.status_code} - {e.message}"
            print("\n" + "="*80)
            print("!!! API CALL ERROR !!!")
            print(f"    - Error Type: {type(e).__name__}")
            print(f"    - Status Code: {e.status_code}")
            print(f"    - Error Message: {e.message}")
            print("    - Prompt that caused the error (first 500 chars):")
            print(f"    - \"\"\"{prompt[:500]}...\"\"\"")
            print("="*80 + "\n")
            
            self.logger.error(f"API call to '{model_name}' FAILED. Error: {error_message}", exc_info=True)
            self._record_api_call(model_name) # Record the attempt even if it failed
            return {
                "status": "ERROR", "text": None,
                "error_message": error_message,
                "model_name": model_name, "raw_response": e.response
            }
        except Exception as e:
            error_message = f"An unexpected error occurred: {type(e).__name__} - {e}"
            print("\n" + "="*80)
            print("!!! UNEXPECTED ERROR DURING API CALL !!!")
            print(f"    - Error Type: {type(e).__name__}")
            print(f"    - Error Message: {e}")
            print("="*80 + "\n")

            self.logger.error(f"API call to '{model_name}' FAILED. Error: {error_message}", exc_info=True)
            self._record_api_call(model_name)
            return {
                "status": "ERROR", "text": None,
                "error_message": error_message,
                "model_name": model_name, "raw_response": None
            }