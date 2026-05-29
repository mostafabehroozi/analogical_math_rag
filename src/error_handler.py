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

import os
import logging
from tqdm import tqdm
from typing import List, Dict, Any
from src.orchestration import run_pipeline_for_single_query
from src.utils import load_json, save_json
from src.hf_sync import periodic_sync_check
from src.layer1_base_execution import _get_cache_filename, _get_cache_path

def retry_failed_generation_pipelines(
    all_experiments_logs: Dict[str, List[Dict]],
    global_config: Dict[str, Any],
    hard_questions: List[str],
    embedding_model: Any,
    exemplar_data: Dict[str, Any],
    api_managers: Dict[str, Any]
) -> Dict[str, List[Dict]]:
    
    logger = logging.getLogger(__name__)
    logger.info("Starting SMART retry process for failed/partial generation pipelines.")
    
    for exp_name, original_logs in all_experiments_logs.items():
        exp_config_overrides = original_logs[0].get("config_flags_used", {})
        current_config = global_config.copy()
        current_config.update(exp_config_overrides)
        
        log_file_path = os.path.join(global_config['RESULTS_DIR'], f"{exp_name}_run_log.json")
        logs_to_process = original_logs
        
        indices_to_retry = []
        
        # 1. Identify Bad Logs
        for i, log in enumerate(logs_to_process):
            status = log.get("pipeline_status", "")
            needs_retry = False
            
            if "FAILURE" in status or "PARTIAL" in status:
                needs_retry = True
            
            if "layer1_base_execution_state" in log:
                l1_state = log["layer1_base_execution_state"]
                if not l1_state.get("candidate_set") or not l1_state.get("cross_evaluation_matrix"):
                    needs_retry = True
            
            if needs_retry:
                indices_to_retry.append(i)

        if not indices_to_retry:
            continue

        # 2. Load Layer 1 Cache
        cache_dir = current_config.get("LAYER1_CACHE_DIR", current_config.get("RESULTS_DIR"))
        top_k = current_config.get("TOP_N_CANDIDATES_RETRIEVAL", 5)
        n_candidates = current_config.get("LAYER1_N_CANDIDATES") or top_k
        cache_filename = _get_cache_filename(top_k, n_candidates, exp_name)
        cache_path = _get_cache_path(cache_dir, cache_filename)
        
        l1_cache = load_json(cache_path) if os.path.exists(cache_path) else None

        for loop_idx, log_idx in enumerate(tqdm(indices_to_retry, desc=f"Retrying {exp_name}")):
            failed_log = logs_to_process[log_idx]
            original_hard_list_idx = failed_log["target_query_original_hard_list_idx"]
            target_query = failed_log["target_query_text"]
            
            # WIPE BAD CACHE
            if l1_cache and "queries" in l1_cache:
                query_key = str(original_hard_list_idx)
                if query_key in l1_cache["queries"]:
                    del l1_cache["queries"][query_key]
                    save_json(l1_cache, cache_path)
            
            # --- NEW: CRITICAL OVERRIDE ---
            # Force the pipeline to run all the way through, even if the experiment
            # originally deferred the solving step to phase 2.
            retry_config = current_config.copy()
            retry_config["DEFER_SOLVE_STEP"] = False
            
            # 3. Re-run Pipeline
            new_run_log = run_pipeline_for_single_query(
                hard_list_idx=original_hard_list_idx,
                target_query=target_query,
                config=retry_config,  # <-- Using the forced config
                embedding_model=embedding_model,
                exemplar_data=exemplar_data,
                api_managers=api_managers
            )
            
            logs_to_process[log_idx] = new_run_log
            save_json(logs_to_process, log_file_path)
            periodic_sync_check(loop_idx, current_config)

        save_json(logs_to_process, log_file_path)

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
