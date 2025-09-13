# src/orchestration.py

"""
Orchestration module for the Analogical Reasoning RAG project.

This module chains together the individual steps from `pipeline_steps.py` to
run the full RAG pipeline. It manages running experiments for multiple queries
and configurations, and handles the saving and loading of results for
pausing and resuming.
"""

import logging
from tqdm import tqdm
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np

# Import our custom modules
from src.pipeline_steps import retrieve, adapt, merge, solve
from src.api_manager import GeminiAPIManager
from src.utils import save_json, load_json
from src.hf_sync import periodic_sync_check

def run_pipeline_for_single_query(
    hard_list_idx: int,
    target_query: str,
    ground_truth: str, # NEW: Added ground_truth for online evaluation
    config: Dict[str, Any],
    embedding_model: SentenceTransformer,
    exemplar_data: Dict[str, Any],
    gemini_manager: GeminiAPIManager
) -> Dict[str, Any]:
    """
    Executes the full RAG pipeline for a single 'hard question'.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting pipeline for Query #{hard_list_idx}: '{target_query[:80]}...' ---")
    
    print("\n" + "="*80)
    print(f"Processing Query #{hard_list_idx}: '{target_query[:100]}...'")
    print("="*80)

    run_log = {
        "target_query_original_hard_list_idx": hard_list_idx,
        "target_query_text": target_query,
        "config_flags_used": {
            "USE_RETRIEVAL": config.get('USE_RETRIEVAL'),
            "APPLY_STANDARDIZATION": config.get('APPLY_STANDARDIZATION'),
            "APPLY_TRANSFORMATION": config.get('APPLY_TRANSFORMATION'),
            "APPLY_MERGING": config.get('APPLY_MERGING'),
            "TOP_N_CANDIDATES_RETRIEVAL": config.get('TOP_N_CANDIDATES_RETRIEVAL'),
            "N_PASS_ATTEMPTS": config.get('N_PASS_ATTEMPTS'),
            # NEW: Log online evaluation settings for clarity
            "ONLINE_EVALUATION_ENABLED": config.get('ONLINE_EVALUATION_ENABLED'),
            "STOP_ON_FIRST_SUCCESS": config.get('STOP_ON_FIRST_SUCCESS')
        },
        "pipeline_status": "PENDING",
        "steps": {}
    }

    # --- Conditional execution based on USE_RETRIEVAL flag ---
    if config.get('USE_RETRIEVAL', True):
        # Step 1: Retrieve, Adapt, Merge
        retrieval_result = retrieve(
            target_query=target_query,
            embedding_model=embedding_model,
            exemplar_questions=exemplar_data['questions'],
            embedded_exemplars=exemplar_data['embeddings'],
            top_k=config['TOP_N_CANDIDATES_RETRIEVAL']
        )
        run_log['steps']['retrieval'] = retrieval_result
        if retrieval_result['status'] == 'FAILURE':
            run_log['pipeline_status'] = "FAILURE: Retrieval failed."
            return run_log

        adapt_result = adapt(
            target_query=target_query,
            retrieved_indices=retrieval_result['retrieved_indices'],
            exemplar_questions=exemplar_data['questions'],
            exemplar_solutions=exemplar_data['solutions'],
            gemini_manager=gemini_manager,
            config=config
        )
        run_log['steps']['adaptation'] = adapt_result
        
        merge_result = merge(
            target_query=target_query,
            adapted_texts=adapt_result['adapted_texts'],
            embedding_model=embedding_model,
            gemini_manager=gemini_manager,
            config=config
        )
        run_log['steps']['merging'] = merge_result
        final_exemplars_for_solve = merge_result['merged_texts']
        
    else:
        # If retrieval is OFF, skip all intermediate steps.
        run_log['steps']['retrieval'] = {"status": "SKIPPED", "reason": "USE_RETRIEVAL is False"}
        run_log['steps']['adaptation'] = {"status": "SKIPPED", "reason": "USE_RETRIEVAL is False"}
        run_log['steps']['merging'] = {"status": "SKIPPED", "reason": "USE_RETRIEVAL is False"}
        final_exemplars_for_solve = []

    # --- Step 4: Solve (MODIFIED) ---
    print("\n[STEP 4] SOLVE")
    solve_result = solve(
        target_query=target_query,
        final_exemplars=final_exemplars_for_solve,
        ground_truth=ground_truth, # Pass the ground truth for online evaluation
        gemini_manager=gemini_manager,
        config=config
    )
    run_log['steps']['solving'] = solve_result
    
    # --- NEW: Handle the updated return structure from solve ---
    run_log['llm_final_solution_attempts_texts'] = solve_result.get('solution_attempts', [])
    
    if config.get("ONLINE_EVALUATION_ENABLED"):
        run_log['online_evaluation_results'] = solve_result.get('evaluation_results', [])

    # Check for the new 'UN-EVALUABLE' status from the solve step
    if solve_result.get('status') == 'UN-EVALUABLE':
        run_log['pipeline_status'] = f"UN-EVALUABLE: {solve_result.get('reason', 'Unknown error in solve step.')}"
        logger.error(run_log['pipeline_status'])
        print(f"  -> Pipeline HALTED. Reason: {run_log['pipeline_status']}")
    else:
        run_log['pipeline_status'] = "SUCCESS"
        logger.info(f"--- Pipeline finished successfully for Query #{hard_list_idx} ---")
        
    return run_log


def run_experiments(
    experiment_configs: List[Dict[str, Any]],
    global_config: Dict[str, Any],
    hard_questions: List[str],
    hard_questions_ground_truths: List[str], # NEW: Pass ground truths
    embedding_model: SentenceTransformer,
    exemplar_data: Dict[str, Any],
    gemini_manager: GeminiAPIManager
) -> Dict[str, List[Dict]]:
    """
    Orchestrates running multiple experiments with different configurations.
    """
    logger = logging.getLogger(__name__)
    all_results = {}

    for exp_overrides in experiment_configs:
        current_config = global_config.copy()
        current_config.update(exp_overrides)
        
        exp_name = current_config.get("experiment_name", "unnamed_experiment")
        logger.info(f"########## Starting Experiment: {exp_name} ##########")

        log_file_path = os.path.join(global_config['RESULTS_DIR'], f"{exp_name}_run_log.json")
        
        existing_logs = load_json(log_file_path)
        if existing_logs:
            completed_indices = {log['target_query_original_hard_list_idx'] for log in existing_logs}
            run_logs = existing_logs
        else:
            completed_indices = set()
            run_logs = []
            
        queries_to_process = [
            (idx, q) for idx, q in enumerate(hard_questions) if idx not in completed_indices
        ]
        
        if not queries_to_process:
            logger.info(f"All queries for '{exp_name}' are already processed. Skipping.")
            all_results[exp_name] = run_logs
            continue

        for loop_idx, (original_idx, query_text) in enumerate(tqdm(queries_to_process, desc=f"Running {exp_name}")):
            # Get the corresponding ground truth
            ground_truth_text = hard_questions_ground_truths[original_idx]
            
            single_run_log = run_pipeline_for_single_query(
                hard_list_idx=original_idx,
                target_query=query_text,
                ground_truth=ground_truth_text, # Pass it to the single query runner
                config=current_config,
                embedding_model=embedding_model,
                exemplar_data=exemplar_data,
                gemini_manager=gemini_manager
            )
            
            # --- NEW: Exclude un-evaluable questions from the final log ---
            # This ensures they don't skew results, as per the requirements.
            if single_run_log['pipeline_status'].startswith('UN-EVALUABLE'):
                print(f"\n--- Query #{original_idx} was un-evaluable and will be excluded from the final logs. ---")
                logger.warning(f"Query {original_idx} excluded from logs for experiment '{exp_name}' due to being un-evaluable.")
                # We save it to a separate file for debugging but don't append to the main run_logs
                error_log_path = os.path.join(global_config['LOGS_DIR'], f"{exp_name}_unevaluable_log.json")
                existing_errors = load_json(error_log_path) or []
                existing_errors.append(single_run_log)
                save_json(existing_errors, error_log_path)
                continue # Skip to the next query
            
            run_logs.append(single_run_log)
            save_json(run_logs, log_file_path)
            periodic_sync_check(loop_idx, current_config)

        save_json(run_logs, log_file_path)
        logger.info(f"########## Finished Experiment: {exp_name} ##########")
        all_results[exp_name] = run_logs
        
    return all_results