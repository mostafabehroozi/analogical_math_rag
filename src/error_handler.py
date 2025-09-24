# src/error_handler.py

"""
Error handling and reporting module for the Analogical Reasoning RAG project.

This module provides functions to:
- Retry failed LLM-based evaluation attempts.
- Generate a detailed report of API errors that occurred during evaluation.

This version is updated to be provider-agnostic, working with any API manager
that follows the project's standard interface.
"""

import logging
import os
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

# Import our custom modules
from src.evaluation import evaluate_single_answer_with_llm
from src.utils import load_json, save_json
from src.hf_sync import periodic_sync_check

def retry_failed_evaluations(
    all_experiments_logs: Dict[str, List[Dict]],
    ground_truths: List[str],
    api_manager: Any,  # MODIFIED: Generic API manager
    config: Dict[str, Any]
) -> Dict[str, List[Dict]]:
    """
    Retries failed evaluation attempts (API_ERROR) for all experiments.
    This function is provider-agnostic.
    
    This function loads the detailed evaluation logs for each experiment,
    identifies evaluation attempts that previously failed with an 'API_ERROR',
    and re-runs the evaluation for those specific attempts using the provided
    api_manager. The logs are updated in place with the new results.

    Args:
        all_experiments_logs (Dict): The main dictionary of all experiment logs.
        ground_truths (List[str]): The list of ground truth solutions.
        api_manager (Any): The initialized, generic API manager (e.g., GeminiAPIManager or AvalAIManager).
        config (Dict): The main configuration dictionary.

    Returns:
        The original all_experiments_logs dictionary (though the underlying
        detailed log files are modified on disk).
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting the process to retry failed evaluation attempts.")
    results_dir = config['RESULTS_DIR']

    for exp_name, _ in all_experiments_logs.items():
        eval_file_path = os.path.join(results_dir, f"{exp_name}_detailed_eval.json")
        detailed_evaluations = load_json(eval_file_path)

        if not detailed_evaluations:
            logger.warning(f"No detailed evaluation log found for '{exp_name}'. Skipping retry.")
            continue

        logger.info(f"--- Retrying for Experiment: {exp_name} ---")
        
        evals_to_retry = [
            (idx, eval_log)
            for idx, eval_log in enumerate(detailed_evaluations)
            if "API_ERROR" in eval_log.get("evaluation_status_list", [])
        ]

        if not evals_to_retry:
            logger.info(f"No API_ERRORs found in '{exp_name}' to retry.")
            continue

        for loop_idx, (eval_idx, log) in enumerate(tqdm(evals_to_retry, desc=f"Retrying {exp_name}")):
            hard_list_idx = log["hard_list_idx"]
            ground_truth = ground_truths[hard_list_idx]
            solution_attempts = log["attempts"]
            
            for attempt_idx, status in enumerate(log["evaluation_status_list"]):
                if status == "API_ERROR":
                    print(f"    -> Retrying attempt {attempt_idx + 1} for query #{hard_list_idx}")
                    attempt_text = solution_attempts[attempt_idx]
                    
                    # MODIFIED: Pass the generic api_manager to the evaluation function
                    is_correct, new_status = evaluate_single_answer_with_llm(
                        attempt_text, ground_truth, api_manager, config
                    )
                    
                    detailed_evaluations[eval_idx]["is_correct_list"][attempt_idx] = is_correct
                    detailed_evaluations[eval_idx]["evaluation_status_list"][attempt_idx] = new_status
                    print(f"       New status for attempt {attempt_idx + 1}: {new_status}")

            save_json(detailed_evaluations, eval_file_path)
            periodic_sync_check(loop_idx, config)

        save_json(detailed_evaluations, eval_file_path)

    logger.info("Finished retrying all failed evaluations across all experiments.")
    return all_experiments_logs


def generate_error_report(
    all_experiments_logs: Dict[str, List[Dict]],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Generates a detailed report of all API_ERRORs from the evaluation logs.

    This function is inherently provider-agnostic as it only reads log files.

    Args:
        all_experiments_logs (Dict): The main dictionary of all experiment logs.
        config (Dict): The main configuration dictionary.

    Returns:
        A pandas DataFrame detailing each API error.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating a detailed report of API errors from evaluation logs.")
    error_records = []
    results_dir = config['RESULTS_DIR']

    for exp_name, _ in all_experiments_logs.items():
        eval_file_path = os.path.join(results_dir, f"{exp_name}_detailed_eval.json")
        detailed_evaluations = load_json(eval_file_path)

        if not detailed_evaluations:
            continue

        for log in detailed_evaluations:
            hard_list_idx = log["hard_list_idx"]
            status_list = log.get("evaluation_status_list", [])
            
            for attempt_idx, status in enumerate(status_list):
                if status == "API_ERROR":
                    error_records.append({
                        "experiment_name": exp_name,
                        "query_hard_list_idx": hard_list_idx,
                        "attempt_number": attempt_idx + 1,
                        "error_type": "API_ERROR",
                        "error_message": "An API error occurred during the evaluation call."
                    })

    if not error_records:
        logger.info("No API_ERRORs were found in any experiment logs.")
        return pd.DataFrame(columns=[
            "experiment_name", "query_hard_list_idx", "attempt_number", 
            "error_type", "error_message"
        ])

    return pd.DataFrame(error_records)