# src/evaluation.py

"""
Evaluation module for the Analogical Reasoning RAG project.

This file provides functions to:
- Evaluate generated answers against ground truths using an LLM.
- Parse and analyze the detailed JSON logs from experiment runs.
- Calculate Pass@K metrics and generate a summary DataFrame.

This version is updated to be API provider-agnostic and to handle structured
error responses from the evaluator LLM calls, enabling more robust retries.
"""

import logging
import os
import re
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional, Union

# Import our custom modules
from src.prompts import create_evaluation_prompt
from src.utils import save_json, load_json
from src.hf_sync import periodic_sync_check
from src.api_manager import APIResponse # Import the structured response type

# Define a more specific type for the result of a single evaluation
EvaluationResult = Tuple[Optional[bool], str, Optional[APIResponse]]

def evaluate_single_answer_with_llm(
    model_answer: str,
    ground_truth: str,
    api_manager: Any,
    config: Dict[str, Any]
) -> EvaluationResult:
    """

    Evaluates a single model-generated answer against a ground truth using an LLM.

    Returns:
        A tuple containing:
        - Optional[bool]: True if correct, False if incorrect, None if evaluation failed.
        - str: A status message (e.g., 'SUCCESS', 'EMPTY_ANSWER', 'PARSING_FAILED', 'API_ERROR').
        - Optional[APIResponse]: The full, structured error response if an API call failed.
    """
    logger = logging.getLogger(__name__)

    # An empty or invalid answer is a distinct failure type, not an API error.
    if not model_answer or not isinstance(model_answer, str):
        return None, "EMPTY_ANSWER", None

    provider = config.get("API_PROVIDER", "gemini").lower()
    evaluator_model = config[f'{"AVALAI" if provider == "avalai" else "GEMINI"}_MODEL_NAME_EVALUATOR']
    evaluator_temp = config['DEFAULT_EVALUATOR_TEMPERATURE']

    prompt = create_evaluation_prompt(model_answer, ground_truth)
    
    print(f"      [API Context] Calling LLM for: Evaluation")
    response = api_manager.generate_content(prompt, evaluator_model, evaluator_temp)

    if response['status'] != 'SUCCESS':
        error_msg = response.get('error_message', 'Unknown API failure')
        logger.warning(f"LLM-based evaluation API call failed with status '{response['status']}': {error_msg}")
        # Propagate the specific status and the full error response
        return None, f"API_{response['status']}", response

    raw_text = response.get('text', '').strip()
    
    # MODIFIED: Truncate this print statement for consistency
    trunc_len = config.get("API_RESPONSE_TRUNCATION_LENGTH", 50)
    print(f"      Evaluator LLM Raw Output: {raw_text[:trunc_len]}{'...' if len(raw_text) > trunc_len else ''}") 
    logger.debug(f"Evaluator raw response: '{raw_text}'")

    eval_match = re.search(r"Evaluation:\s*(true|false)", raw_text, re.IGNORECASE)

    if eval_match:
        result_str = eval_match.group(1).lower()
        logger.info(f"Parsed evaluation result: {result_str}")
        return result_str == 'true', "SUCCESS", None
    else:
        logger.warning(f"Could not parse 'Evaluation:' line from LLM response. Treating as failure.")
        return None, "PARSING_FAILED", None

def analyze_experiment_logs(
    all_experiments_logs: Dict[str, List[Dict]],
    ground_truths: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Analyzes the complete logs from all experiments to calculate Pass@K metrics.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting analysis of all experiment logs.")
    analysis_summary = []

    pass_k_values_to_report = config.get("PASS_K_VALUES_TO_REPORT", [1])
    results_dir = config['RESULTS_DIR']

    for exp_name, query_logs in all_experiments_logs.items():
        logger.info(f"--- Analyzing Experiment: {exp_name} ---")
        if not query_logs:
            logger.warning(f"No logs found for experiment '{exp_name}'. Skipping.")
            continue

        exp_config = query_logs[0].get("config_flags_used", {})
        n_attempts_config = exp_config.get("N_PASS_ATTEMPTS", 1)
        pass_k_values = sorted([k for k in pass_k_values_to_report if 0 < k <= n_attempts_config])

        pass_k_counts = {k: 0 for k in pass_k_values}
        # Updated to handle more specific error statuses
        error_counts = {"API_BLOCKED": 0, "API_ERROR": 0, "API_RATE_LIMITED": 0, "PARSING_FAILED": 0, "EMPTY_ANSWER": 0}
        total_valid_queries = 0

        eval_file_path = os.path.join(results_dir, f"{exp_name}_detailed_eval.json")

        existing_evals = load_json(eval_file_path)
        detailed_evaluations = existing_evals if existing_evals else []
        evaluated_indices = {eval_log['hard_list_idx'] for eval_log in detailed_evaluations}
        logger.info(f"Loaded {len(detailed_evaluations)} existing evaluation results for '{exp_name}'.")

        logs_to_process = [log for log in query_logs if log["target_query_original_hard_list_idx"] not in evaluated_indices]

        if not logs_to_process:
            logger.info(f"All query logs for '{exp_name}' have already been evaluated. Moving to summary.")
        else:
            for loop_idx, log in enumerate(tqdm(logs_to_process, desc=f"Evaluating {exp_name}")):
                hard_list_idx = log["target_query_original_hard_list_idx"]
                ground_truth = ground_truths[hard_list_idx]
                
                # IMPORTANT: This now comes from the 'solving' step in the main run log
                solution_attempts = log.get("steps", {}).get("solving", {}).get("solution_attempts", [])

                is_correct_list = []
                status_list = []
                api_error_details = []

                for i, attempt in enumerate(solution_attempts):
                    # Check if the attempt was a successful generation (a string)
                    if isinstance(attempt, str):
                        print(f"    -> Evaluating attempt {i+1}/{len(solution_attempts)} for query #{hard_list_idx}")
                        is_correct, status, error_info = evaluate_single_answer_with_llm(attempt, ground_truth, api_manager, config)
                        if status != "SUCCESS":
                            print(f"       WARNING: Evaluation attempt failed with status: {status}")
                            logger.warning(f"Evaluation for query {hard_list_idx}, attempt {i+1} failed with status: {status}")
                    
                    # Check if the generation itself failed (a dict)
                    elif isinstance(attempt, dict) and attempt.get('status') == 'FAILURE':
                        is_correct, status, error_info = None, "GENERATION_FAILED", attempt.get("error_info")
                        print(f"    -> Skipping evaluation for attempt {i+1} (generation failed).")
                    
                    else: # Should not happen, but good to handle
                        is_correct, status, error_info = None, "INVALID_ATTEMPT_FORMAT", None

                    is_correct_list.append(is_correct)
                    status_list.append(status)
                    api_error_details.append(error_info)

                detailed_evaluations.append({
                    "hard_list_idx": hard_list_idx,
                    "is_correct_list": is_correct_list,
                    "evaluation_status_list": status_list,
                    "evaluation_error_details": api_error_details, # Store detailed errors
                    "attempts": solution_attempts # Store the original attempts (mix of str/dict)
                })

                save_json(detailed_evaluations, eval_file_path)
                periodic_sync_check(loop_idx, config)

        save_json(detailed_evaluations, eval_file_path)

        # Recalculate summary from the full (potentially updated) evaluation list
        for eval_result in detailed_evaluations:
            if eval_result.get("attempts"):
                total_valid_queries += 1
                is_correct_list = eval_result["is_correct_list"]
                status_list = eval_result["evaluation_status_list"]

                for status in status_list:
                    # Normalize API statuses for counting
                    if status.startswith("API_"):
                        error_key = status.replace("API_", "")
                        # Continue with the rest of the original code...
                        # (This part of the file was cut off in the original prompt,
                        # but the logic for the modification is complete.)  