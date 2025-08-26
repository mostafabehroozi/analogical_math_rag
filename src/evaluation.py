# src/evaluation.py

"""
Evaluation module for the Analogical Reasoning RAG project.

This file provides functions to:
- Evaluate generated answers against ground truths using an LLM.
- Parse and analyze the detailed JSON logs from experiment runs.
- Calculate Pass@K metrics and generate a summary DataFrame.
"""

import logging
import os
import re
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

# Import our custom modules
from src.api_manager import GeminiAPIManager
from src.prompts import create_evaluation_prompt
from src.utils import save_json, load_json
# --- NEW: Import for periodic synchronization ---
from src.hf_sync import periodic_sync_check

def evaluate_single_answer_with_llm(
    model_answer: str,
    ground_truth: str,
    gemini_manager: GeminiAPIManager,
    config: Dict[str, Any]
) -> bool:
    """
    Evaluates a single model-generated answer against a ground truth using an LLM.
    """
    logger = logging.getLogger(__name__)
    
    if not model_answer or model_answer.startswith("Error:"):
        return False

    evaluator_model = config['GEMINI_MODEL_NAME_EVALUATOR']
    evaluator_temp = config['DEFAULT_EVALUATOR_TEMPERATURE']
    
    prompt = create_evaluation_prompt(model_answer, ground_truth)
    response = gemini_manager.generate_content(prompt, evaluator_model, evaluator_temp)

    if response['status'] != 'SUCCESS':
        logger.warning(f"LLM-based evaluation failed: {response['error_message']}")
        return False

    raw_text = response['text'].strip()
    print(f"    Evaluator LLM Raw Output: {raw_text}")
    logger.debug(f"Evaluator raw response: '{raw_text}'")
    
    eval_match = re.search(r"Evaluation:\s*(true|false)", raw_text, re.IGNORECASE)
    
    if eval_match:
        result_str = eval_match.group(1).lower()
        logger.info(f"Parsed evaluation result: {result_str}")
        return result_str == 'true'
    else:
        logger.warning(f"Could not parse 'Evaluation:' line from LLM response. Treating as False.")
        return False

def analyze_experiment_logs(
    all_experiments_logs: Dict[str, List[Dict]],
    ground_truths: List[str],
    gemini_manager: GeminiAPIManager,
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
        n_attempts = exp_config.get("N_PASS_ATTEMPTS", 1)
        pass_k_values = sorted([k for k in pass_k_values_to_report if 0 < k <= n_attempts])
        
        pass_k_counts = {k: 0 for k in pass_k_values}
        total_valid_queries = 0
        
        # --- MODIFIED: Pause and Resume logic for evaluations ---
        eval_file_path = os.path.join(results_dir, f"{exp_name}_detailed_eval.json")
        
        existing_evals = load_json(eval_file_path)
        if existing_evals:
            detailed_evaluations = existing_evals
            evaluated_indices = {eval_log['hard_list_idx'] for eval_log in detailed_evaluations}
            logger.info(f"Loaded {len(detailed_evaluations)} existing evaluation results for '{exp_name}'.")
        else:
            detailed_evaluations = []
            evaluated_indices = set()

        logs_to_process = [log for log in query_logs if log["target_query_original_hard_list_idx"] not in evaluated_indices]
        
        if not logs_to_process:
            logger.info(f"All query logs for '{exp_name}' have already been evaluated. Moving to summary.")
        else:
            # --- MODIFIED: Use enumerate to get a loop counter ---
            for loop_idx, log in enumerate(tqdm(logs_to_process, desc=f"Evaluating {exp_name}")):
                hard_list_idx = log["target_query_original_hard_list_idx"]
                ground_truth = ground_truths[hard_list_idx]
                solution_attempts = log["llm_final_solution_attempts_texts"]

                valid_attempts = [s for s in solution_attempts if isinstance(s, str) and not s.startswith("Error:")]
                if not valid_attempts:
                    continue
                
                is_correct_list = [
                    evaluate_single_answer_with_llm(attempt, ground_truth, gemini_manager, config)
                    for attempt in valid_attempts
                ]
                
                detailed_evaluations.append({
                    "hard_list_idx": hard_list_idx,
                    "is_correct_list": is_correct_list,
                    "attempts": valid_attempts
                })
                
                # --- NEW: Save progress locally and sync periodically ---
                save_json(detailed_evaluations, eval_file_path)
                periodic_sync_check(loop_idx, config)

        # Final local save
        save_json(detailed_evaluations, eval_file_path)
        
        # Calculate final metrics from the complete list of evaluations
        for eval_result in detailed_evaluations:
            if eval_result.get("attempts"):
                total_valid_queries += 1
                is_correct_list = eval_result["is_correct_list"]
                for k in pass_k_values:
                    if any(is_correct_list[:k]):
                        pass_k_counts[k] += 1
        
        summary_row = {"experiment_name": exp_name, **exp_config}
        summary_row["total_queries_processed"] = len(query_logs)
        summary_row["total_queries_with_valid_solutions"] = total_valid_queries

        for k in pass_k_values:
            count = pass_k_counts[k]
            accuracy = (count / total_valid_queries) * 100 if total_valid_queries > 0 else 0
            summary_row[f"pass@{k}_count"] = count
            summary_row[f"pass@{k}_accuracy (%)"] = round(accuracy, 2)
        
        analysis_summary.append(summary_row)
        
    return pd.DataFrame(analysis_summary)
