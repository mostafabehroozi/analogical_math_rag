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
# --- NEW: Import for periodic synchronization ---
from src.hf_sync import periodic_sync_check

def run_pipeline_for_single_query(
    hard_list_idx: int,
    target_query: str,
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
            "APPLY_STANDARDIZATION": config.get('APPLY_STANDARDIZATION'),
            "APPLY_TRANSFORMATION": config.get('APPLY_TRANSFORMATION'),
            "APPLY_MERGING": config.get('APPLY_MERGING'),
            "TOP_N_CANDIDATES_RETRIEVAL": config.get('TOP_N_CANDIDATES_RETRIEVAL'),
            "N_PASS_ATTEMPTS": config.get('N_PASS_ATTEMPTS'),
            "PASS_N_SOLVER_TEMPERATURE_USED": config.get('DEFAULT_PASS_N_SOLVER_TEMPERATURE')
        },
        "pipeline_status": "PENDING",
        "steps": {}
    }

    # Step 1: Retrieve
    print("\n[STEP 1] RETRIEVE")
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
        logger.error(run_log['pipeline_status'])
        print("  -> Retrieval FAILED.")
        return run_log
    print(f"  -> Retrieved indices: {retrieval_result['retrieved_indices']}")

    # Step 2: Adapt
    print("\n[STEP 2] ADAPT")
    adapt_result = adapt(
        target_query=target_query,
        retrieved_indices=retrieval_result['retrieved_indices'],
        exemplar_questions=exemplar_data['questions'],
        exemplar_solutions=exemplar_data['solutions'],
        gemini_manager=gemini_manager,
        config=config
    )
    run_log['steps']['adaptation'] = adapt_result
    for i, text in enumerate(adapt_result.get('adapted_texts', [])):
        print(f"  -> Adapted text #{i+1} (start): '{text[:120]}...'")
    
    # Step 3: Merge
    print("\n[STEP 3] MERGE")
    merge_result = merge(
        target_query=target_query,
        adapted_texts=adapt_result['adapted_texts'],
        embedding_model=embedding_model,
        gemini_manager=gemini_manager,
        config=config
    )
    run_log['steps']['merging'] = merge_result
    if merge_result['status'] == 'SKIPPED':
        print("  -> Merging was SKIPPED as per config.")
    for i, text in enumerate(merge_result.get('merged_texts', [])):
        print(f"  -> Final merged text #{i+1} (start): '{text[:120]}...'")

    # Step 4: Solve
    print("\n[STEP 4] SOLVE")
    solve_result = solve(
        target_query=target_query,
        final_exemplars=merge_result['merged_texts'],
        gemini_manager=gemini_manager,
        config=config
    )
    run_log['steps']['solving'] = solve_result
    run_log['llm_final_solution_attempts_texts'] = solve_result.get('solution_attempts', [])
    for i, text in enumerate(solve_result.get('solution_attempts', [])):
        print(f"  -> Solution attempt #{i+1} (start): '{text[:120]}...'")

    run_log['pipeline_status'] = "SUCCESS"
    logger.info(f"--- Pipeline finished successfully for Query #{hard_list_idx} ---")
    return run_log


def run_experiments(
    experiment_configs: List[Dict[str, Any]],
    global_config: Dict[str, Any],
    hard_questions: List[str],
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
            logger.info(f"Loaded {len(existing_logs)} existing results from {log_file_path}.")
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

        # --- MODIFIED: Use enumerate to get a loop counter for periodic sync ---
        for loop_idx, (original_idx, query_text) in enumerate(tqdm(queries_to_process, desc=f"Running {exp_name}")):
            single_run_log = run_pipeline_for_single_query(
                hard_list_idx=original_idx,
                target_query=query_text,
                config=current_config,
                embedding_model=embedding_model,
                exemplar_data=exemplar_data,
                gemini_manager=gemini_manager
            )
            run_logs.append(single_run_log)
            
            # Save progress locally
            save_json(run_logs, log_file_path)
            
            # --- NEW: Call the periodic sync check ---
            # This will trigger an upload to the Hub if the interval is reached.
            periodic_sync_check(loop_idx, current_config)

        # Final local save at the end of the experiment
        save_json(run_logs, log_file_path)
        logger.info(f"########## Finished Experiment: {exp_name} ##########")
        all_results[exp_name] = run_logs
        
    return all_results
