# src/error_handler.py

"""
Error handling, reporting, and retry module for the Analogical Reasoning RAG project.

This module provides functions to:
- NEW: Retry generation pipelines that failed midway through.
- Retry failed LLM-based evaluation attempts with more granularity.
- Generate a comprehensive report of all API and processing errors that occurred
  during both generation and evaluation phases.
"""

import logging
import os
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

# Import our custom modules
from src.evaluation import evaluate_single_answer_with_llm
from src.orchestration import run_pipeline_for_single_query # Used for retrying generation
from src.utils import load_json, save_json
from src.hf_sync import periodic_sync_check, sync_workspace_to_hub

# --- NEW: Generation Phase Retry Logic ---

def retry_failed_generation_pipelines(
    all_experiments_logs: Dict[str, List[Dict]],
    global_config: Dict[str, Any],
    hard_questions: List[str],
    embedding_model: Any,
    exemplar_data: Dict[str, Any],
    api_managers: Dict[str, Any]
) -> Dict[str, List[Dict]]:
    """
    Identifies and retries generation pipelines that failed partway through.

    It loads the run log for each experiment, finds entries with a 'FAILURE' status,
    and re-runs the `run_pipeline_for_single_query` function for them. The original
    log entry is then replaced with the new result.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting the process to retry failed generation pipelines.")
    
    for exp_name, original_logs in all_experiments_logs.items():
        exp_config_overrides = original_logs[0].get("config_flags_used", {})
        current_config = global_config.copy()
        current_config.update(exp_config_overrides)
        
        log_file_path = os.path.join(global_config['RESULTS_DIR'], f"{exp_name}_run_log.json")
        
        # We work directly with the list of logs for this experiment
        logs_to_process = original_logs
        
        # Find indices of logs that need retrying
        indices_to_retry = [
            i for i, log in enumerate(logs_to_process)
            if "FAILURE" in log.get("pipeline_status", "")
        ]

        if not indices_to_retry:
            logger.info(f"No failed generation pipelines found for experiment '{exp_name}'. Skipping.")
            continue

        logger.info(f"--- Retrying {len(indices_to_retry)} failed pipelines for Experiment: {exp_name} ---")

        for loop_idx, log_idx in enumerate(tqdm(indices_to_retry, desc=f"Retrying generation for {exp_name}")):
            failed_log = logs_to_process[log_idx]
            original_hard_list_idx = failed_log["target_query_original_hard_list_idx"]
            target_query = failed_log["target_query_text"]
            
            print(f"\nRetrying pipeline for Query #{original_hard_list_idx}...")
            
            # Re-run the entire pipeline logic for this single query
            new_run_log = run_pipeline_for_single_query(
                hard_list_idx=original_hard_list_idx,
                target_query=target_query,
                config=current_config,
                embedding_model=embedding_model,
                exemplar_data=exemplar_data,
                api_managers=api_managers
            )
            
            # Replace the old failed log with the new one
            logs_to_process[log_idx] = new_run_log
            
            # Save progress incrementally
            save_json(logs_to_process, log_file_path)
            periodic_sync_check(loop_idx, current_config)

        # Final save and sync for the experiment
        save_json(logs_to_process, log_file_path)
        sync_workspace_to_hub(current_config)

    logger.info("Finished retrying all failed generation pipelines.")
    return all_experiments_logs


# --- UPGRADED: Evaluation Phase Retry Logic ---

def retry_failed_evaluations(
    all_experiments_logs: Dict[str, List[Dict]],
    ground_truths: List[str],
    api_managers: Dict[str, Any],
    config: Dict[str, Any]
) -> None:
    """
    Retries failed evaluation attempts for all experiments.

    This function now retries any evaluation that did not achieve a 'SUCCESS' status,
    including API errors, parsing failures, etc. It updates the detailed evaluation
    logs in place.
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

        logger.info(f"--- Retrying evaluations for Experiment: {exp_name} ---")
        
        # MODIFIED: Select the correct API manager for evaluation based on the config
        provider_for_eval = config.get('API_PROVIDER_EVALUATOR', 'gemini')
        manager_for_eval = api_managers[provider_for_eval]

        # Find all attempts across all queries that did not succeed
        retries_needed = []
        for eval_idx, log in enumerate(detailed_evaluations):
            for attempt_idx, status in enumerate(log.get("evaluation_status_list", [])):
                if status != "SUCCESS":
                    retries_needed.append((eval_idx, attempt_idx))

        if not retries_needed:
            logger.info(f"No failed evaluations found in '{exp_name}' to retry.")
            continue

        for loop_idx, (eval_idx, attempt_idx) in enumerate(tqdm(retries_needed, desc=f"Retrying evaluations for {exp_name}")):
            log = detailed_evaluations[eval_idx]
            hard_list_idx = log["hard_list_idx"]
            ground_truth = ground_truths[hard_list_idx]
            
            # The attempt can be a string (successful generation) or a dict (failed generation)
            attempt_text_or_dict = log["attempts"][attempt_idx]

            # Only retry if the generation was successful in the first place
            if isinstance(attempt_text_or_dict, str):
                print(f"    -> Retrying evaluation for query #{hard_list_idx}, attempt #{attempt_idx + 1}")
                
                # MODIFIED: Use the selected manager for evaluation
                eval_result = evaluate_single_answer_with_llm(
                    attempt_text_or_dict, ground_truth, manager_for_eval, config
                )
                
                # Update the log in place
                log["is_correct_list"][attempt_idx] = eval_result["is_correct"]
                log["evaluation_status_list"][attempt_idx] = eval_result["status"]
                log["evaluation_error_details"][attempt_idx] = eval_result["error_details"]
                print(f"       New status: {eval_result['status']}")

            # Persist changes after each retry
            save_json(detailed_evaluations, eval_file_path)
            periodic_sync_check(loop_idx, config)

        # Final save and sync
        save_json(detailed_evaluations, eval_file_path)
        sync_workspace_to_hub(config)

    logger.info("Finished retrying all failed evaluations across all experiments.")


# --- UPGRADed: Comprehensive Error Reporting ---

def generate_error_report(
    all_experiments_logs: Dict[str, List[Dict]],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Generates a detailed report of all errors from both generation and evaluation logs.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating a comprehensive error report.")
    error_records = []
    results_dir = config['RESULTS_DIR']

    for exp_name, query_logs in all_experiments_logs.items():
        # --- 1. Scan Generation Logs for Errors ---
        for log in query_logs:
            hard_idx = log['target_query_original_hard_list_idx']
            if "FAILURE" in log.get("pipeline_status", ""):
                # Find the first failing step
                for step_name, step_result in log.get("steps", {}).items():
                    if "FAILURE" in step_result.get("status", "") or "PARTIAL_SUCCESS" in step_result.get("status", ""):
                        # Handle failures within steps that process multiple items, like adapt
                        failures = step_result.get("failed_adaptations", []) + step_result.get("failed_merges", [])
                        if failures:
                            for failure in failures:
                                error_info = failure.get("error_info", {})
                                error_records.append({
                                    "phase": "generation", "experiment_name": exp_name, "query_hard_list_idx": hard_idx,
                                    "pipeline_step": step_name, "attempt_number": failure.get("source_index", "N/A"),
                                    "error_type": error_info.get("error_type"), "error_message": error_info.get("error_message")
                                })
                        # Handle failures in the final solve step
                        elif step_name == "solving":
                            for i, attempt in enumerate(step_result.get("solution_attempts", [])):
                                if isinstance(attempt, dict) and "FAILURE" in attempt.get("status", ""):
                                    error_info = attempt.get("error_info", {})
                                    error_records.append({
                                        "phase": "generation", "experiment_name": exp_name, "query_hard_list_idx": hard_idx,
                                        "pipeline_step": "solve", "attempt_number": i + 1,
                                        "error_type": error_info.get("error_type"), "error_message": error_info.get("error_message")
                                    })
                        break # Found the first major failing step for this log

        # --- 2. Scan Evaluation Logs for Errors ---
        eval_file_path = os.path.join(results_dir, f"{exp_name}_detailed_eval.json")
        detailed_evaluations = load_json(eval_file_path)
        if detailed_evaluations:
            for log in detailed_evaluations:
                hard_idx = log["hard_list_idx"]
                for i, status in enumerate(log.get("evaluation_status_list", [])):
                    if status != "SUCCESS":
                        error_details = log.get("evaluation_error_details", [])[i] or {}
                        error_records.append({
                            "phase": "evaluation", "experiment_name": exp_name, "query_hard_list_idx": hard_idx,
                            "pipeline_step": "evaluate_answer", "attempt_number": i + 1,
                            "error_type": status, "error_message": error_details.get("error_message", "N/A")
                        })

    if not error_records:
        logger.info("No errors were found in any experiment logs.")
        return pd.DataFrame()

    return pd.DataFrame(error_records)