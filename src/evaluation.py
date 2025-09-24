# src/evaluation.py

"""
Evaluation module for the Analogical Reasoning RAG project.

This file provides functions to:
- Evaluate generated answers against ground truths using a generic LLM.
- Parse and analyze detailed JSON logs from experiment runs, accommodating
  both pre-computed online evaluations and traditional batch evaluations.
- Calculate Pass@K metrics and generate a summary DataFrame.

This version is updated to be provider-agnostic, working with any API manager
that follows the project's standard interface.
"""

import logging
import os
import re
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

# Import our custom modules
from src.prompts import create_evaluation_prompt
from src.utils import save_json, load_json
from src.hf_sync import periodic_sync_check

EvaluationResult = Tuple[Optional[bool], str]

def evaluate_single_answer_with_llm(
    model_answer: str,
    ground_truth: str,
    api_manager: Any,  # MODIFIED: Generic API manager
    config: Dict[str, Any]
) -> EvaluationResult:
    """
    Evaluates a single model-generated answer against a ground truth using an LLM.
    This function is provider-agnostic and can be called directly for online evaluation.

    Returns:
        A tuple containing:
        - Optional[bool]: True if correct, False if incorrect, None if evaluation failed.
        - str: A status message ('SUCCESS', 'EMPTY_ANSWER', 'API_BLOCKED', 'API_ERROR', 'PARSING_FAILED').
    """
    logger = logging.getLogger(__name__)

    if not model_answer or not isinstance(model_answer, str) or model_answer.startswith("Error:"):
        return None, "EMPTY_ANSWER"

    # MODIFIED: Use generic model name key
    evaluator_model = config.get('MODEL_NAME_EVALUATOR')
    evaluator_temp = config['DEFAULT_EVALUATOR_TEMPERATURE']

    prompt = create_evaluation_prompt(model_answer, ground_truth)
    # MODIFIED: Call the generic api_manager
    response = api_manager.generate_content(prompt, evaluator_model, evaluator_temp)

    if response['status'] != 'SUCCESS':
        print(f"--- ERROR during LLM-based EVALUATION ---")
        print(f"    - Error: {response.get('error_message', 'No error message provided.')}")
        print(f"-----------------------------------------")
        return None, response.get('status', 'API_ERROR')

    raw_text = response['text'].strip()
    eval_match = re.search(r"Evaluation:\s*(true|false)", raw_text, re.IGNORECASE)

    if eval_match:
        return eval_match.group(1).lower() == 'true', "SUCCESS"
    else:
        logger.warning(f"Could not parse 'true' or 'false' from evaluator output: {raw_text[:200]}")
        return None, "PARSING_FAILED"

def analyze_experiment_logs(
    all_experiments_logs: Dict[str, List[Dict]],
    ground_truths: List[str],
    api_manager: Any,  # MODIFIED: Generic API manager
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Analyzes the complete logs from all experiments to calculate Pass@K metrics.

    This function is dual-mode and provider-agnostic:
    1. If online evaluation was enabled, it directly uses the results from the logs.
    2. If online evaluation was disabled, it performs the evaluation in a batch using the provided api_manager.
    """
    logger = logging.getLogger(__name__)
    analysis_summary = []

    pass_k_values_to_report = config.get("PASS_K_VALUES_TO_REPORT", [1])
    results_dir = config['RESULTS_DIR']

    for exp_name, query_logs in all_experiments_logs.items():
        logger.info(f"--- Analyzing Experiment: {exp_name} ---")
        if not query_logs:
            continue

        exp_config = query_logs[0].get("config_flags_used", {})
        n_attempts = exp_config.get("N_PASS_ATTEMPTS", 1)
        pass_k_values = sorted([k for k in pass_k_values_to_report if 0 < k <= n_attempts])
        
        online_eval_was_enabled = exp_config.get("ONLINE_EVALUATION_ENABLED", False)
        
        eval_file_path = os.path.join(results_dir, f"{exp_name}_detailed_eval.json")
        detailed_evaluations = load_json(eval_file_path) or []

        if online_eval_was_enabled:
            logger.info("Online evaluation was enabled. Using pre-computed results from run logs.")
            detailed_evaluations = []
            for log in query_logs:
                hard_list_idx = log["target_query_original_hard_list_idx"]
                online_results = log.get("online_evaluation_results", [])
                
                detailed_evaluations.append({
                    "hard_list_idx": hard_list_idx,
                    "is_correct_list": [res.get("is_correct") for res in online_results],
                    "evaluation_status_list": [res.get("status") for res in online_results],
                    "attempts": log["llm_final_solution_attempts_texts"]
                })
            save_json(detailed_evaluations, eval_file_path)
        
        else:
            logger.info("Online evaluation was disabled. Performing batch evaluation.")
            evaluated_indices = {eval_log['hard_list_idx'] for eval_log in detailed_evaluations}
            logs_to_process = [log for log in query_logs if log["target_query_original_hard_list_idx"] not in evaluated_indices]

            if logs_to_process:
                for loop_idx, log in enumerate(tqdm(logs_to_process, desc=f"Batch Evaluating {exp_name}")):
                    hard_list_idx = log["target_query_original_hard_list_idx"]
                    ground_truth = ground_truths[hard_list_idx]
                    solution_attempts = log["llm_final_solution_attempts_texts"]
                    
                    is_correct_list, status_list = [], []
                    for attempt in solution_attempts:
                        # MODIFIED: Pass the generic api_manager
                        is_correct, status = evaluate_single_answer_with_llm(
                            attempt, ground_truth, api_manager, config
                        )
                        is_correct_list.append(is_correct)
                        status_list.append(status)

                    detailed_evaluations.append({
                        "hard_list_idx": hard_list_idx,
                        "is_correct_list": is_correct_list,
                        "evaluation_status_list": status_list,
                        "attempts": solution_attempts
                    })
                    save_json(detailed_evaluations, eval_file_path)
                    periodic_sync_check(loop_idx, config)

        # --- Metric Calculation (Unified for both modes) ---
        pass_k_counts = {k: 0 for k in pass_k_values}
        error_counts = {"API_BLOCKED": 0, "API_ERROR": 0, "PARSING_FAILED": 0, "EMPTY_ANSWER": 0}
        total_valid_queries = 0

        for eval_result in detailed_evaluations:
            if eval_result.get("attempts"):
                total_valid_queries += 1
                is_correct_list = eval_result["is_correct_list"]
                
                for status in eval_result["evaluation_status_list"]:
                    if status in error_counts:
                        error_counts[status] += 1
                
                for k in pass_k_values:
                    if any(res for res in is_correct_list[:k] if res is True):
                        pass_k_counts[k] += 1

        summary_row = {"experiment_name": exp_name, **exp_config}
        summary_row["total_queries_processed"] = len(detailed_evaluations)

        for k in pass_k_values:
            count = pass_k_counts[k]
            accuracy = (count / total_valid_queries) * 100 if total_valid_queries > 0 else 0
            summary_row[f"pass@{k}_count"] = count
            summary_row[f"pass@{k}_accuracy (%)"] = round(accuracy, 2)
        
        summary_row.update(error_counts)
        analysis_summary.append(summary_row)

    return pd.DataFrame(analysis_summary)