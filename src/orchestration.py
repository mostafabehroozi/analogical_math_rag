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

    Args:
        hard_list_idx (int): The index of the query in the original hard list.
        target_query (str): The text of the question to solve.
        config (Dict): The configuration for this specific run.
        embedding_model (SentenceTransformer): The initialized embedding model.
        exemplar_data (Dict): A dict containing questions, solutions, and embeddings for the corpus.
        gemini_manager (GeminiAPIManager): The initialized API manager.

    Returns:
        A dictionary containing a detailed log of the entire pipeline run for this query.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting pipeline for Query #{hard_list_idx}: '{target_query[:80]}...' ---")
    
    # --- ADDED FOR MONITORING ---
    print("\n" + "="*80)
    print(f"Processing Query #{hard_list_idx}: '{target_query[:100]}...'")
    print("="*80)


    # Log the specific configuration flags being used for this run
    run_log = {
        "target_query_original_hard_list_idx": hard_list_idx,
        "target_query_text": target_query,
        "config_flags_used": {
            "APPLY_TRANSFORMATION": config.get('APPLY_TRANSFORMATION'),
            "APPLY_SUMMARIZATION": config.get('APPLY_SUMMARIZATION'),
            "APPLY_MERGING": config.get('APPLY_MERGING'),
            "TOP_N_CANDIDATES_RETRIEVAL": config.get('TOP_N_CANDIDATES_RETRIEVAL'),
            "N_PASS_ATTEMPTS": config.get('N_PASS_ATTEMPTS'),
            "PASS_N_SOLVER_TEMPERATURE_USED": config.get('DEFAULT_PASS_N_SOLVER_TEMPERATURE')
        },
        "pipeline_status": "PENDING",
        "steps": {}
    }

    # --- Step 1: Retrieve ---
    # --- ADDED FOR MONITORING ---
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
        # --- ADDED FOR MONITORING ---
        print("  -> Retrieval FAILED.")
        return run_log
    # --- ADDED FOR MONITORING ---
    print(f"  -> Retrieved indices: {retrieval_result['retrieved_indices']}")


    # --- Step 2: Adapt ---
    # --- ADDED FOR MONITORING ---
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
    # --- ADDED FOR MONITORING ---
    for i, text in enumerate(adapt_result.get('adapted_texts', [])):
        print(f"  -> Adapted text #{i+1} (start): '{text[:120]}...'")
    
    # --- Step 3: Merge ---
    # --- ADDED FOR MONITORING ---
    print("\n[STEP 3] MERGE")
    merge_result = merge(
        target_query=target_query,
        adapted_texts=adapt_result['adapted_texts'],
        embedding_model=embedding_model, # Required for advanced merging strategies
        gemini_manager=gemini_manager,
        config=config
    )
    run_log['steps']['merging'] = merge_result
    # --- ADDED FOR MONITORING ---
    if merge_result['status'] == 'SKIPPED':
        print("  -> Merging was SKIPPED as per config.")
    for i, text in enumerate(merge_result.get('merged_texts', [])):
        print(f"  -> Final merged text #{i+1} (start): '{text[:120]}...'")


    # --- Step 4: Solve ---
    # --- ADDED FOR MONITORING ---
    print("\n[STEP 4] SOLVE")
    solve_result = solve(
        target_query=target_query,
        final_exemplars=merge_result['merged_texts'],
        gemini_manager=gemini_manager,
        config=config
    )
    run_log['steps']['solving'] = solve_result
    run_log['llm_final_solution_attempts_texts'] = solve_result.get('solution_attempts', [])
    # --- ADDED FOR MONITORING ---
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

    Manages loading previous results to allow for pausing and resuming.
    
    Returns:
        A dictionary containing the full logs for all completed experiments.
    """
    logger = logging.getLogger(__name__)
    all_results = {}

    for exp_overrides in experiment_configs:
        # Create a new config for this specific experiment
        current_config = global_config.copy()
        current_config.update(exp_overrides)
        
        exp_name = current_config.get("experiment_name", "unnamed_experiment")
        logger.info(f"########## Starting Experiment: {exp_name} ##########")

        # --- THIS IS THE KEY LINE FOR YOUR SECOND REQUEST ---
        # The log file path is created using the experiment name.
        # This acts as a checkpoint file for each specific experiment.
        log_file_path = os.path.join(global_config['RESULTS_DIR'], f"{exp_name}_run_log.json")
        
        # --- Pause and Resume Logic ---
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

        for idx, query_text in tqdm(queries_to_process, desc=f"Running {exp_name}"):
            single_run_log = run_pipeline_for_single_query(
                hard_list_idx=idx,
                target_query=query_text,
                config=current_config,
                embedding_model=embedding_model,
                exemplar_data=exemplar_data,
                gemini_manager=gemini_manager
            )
            run_logs.append(single_run_log)
            
            # Save progress periodically
            if (len(run_logs) % 5 == 0): # Save every 5 queries
                save_json(run_logs, log_file_path)
                logger.info(f"Saved progress for '{exp_name}' ({len(run_logs)} queries complete).")

        # Final save at the end of the experiment
        save_json(run_logs, log_file_path)
        logger.info(f"########## Finished Experiment: {exp_name} ##########")
        all_results[exp_name] = run_logs
        
    return all_results
