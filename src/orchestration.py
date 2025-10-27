# src/orchestration.py

"""
Orchestration module for the Analogical Reasoning RAG project.

This module chains together the individual steps from `pipeline_steps.py` to
run the full RAG pipeline. It manages running experiments for multiple queries
and configurations, and handles the saving and loading of results for
pausing and resuming.

This version is updated to be API provider-agnostic and to handle detailed
error states from the pipeline steps. It logs partial progress when failures
occur, preventing data loss and enabling targeted retries.

MODIFIED: This version now supports a deferred execution mode via the
`DEFER_SOLVE_STEP` config flag. When enabled, it runs all intermediate steps
(retrieve, adapt, merge) for all queries first, then runs the final solve
step for all queries in a second phase.

REWRITTEN: The `run_experiments` function now implements a global, cross-experiment
deferred execution. If ANY experiment has `DEFER_SOLVE_STEP` set to True, the
entire run switches to a two-phase model:
1. Phase 1: All intermediate steps for ALL experiments are completed.
2. Phase 2: All final solving steps for ALL experiments are completed.
This optimizes API usage by batching all expensive 'solve' calls together.
"""

import logging
from tqdm import tqdm
import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# Import our custom modules
from src.pipeline_steps import retrieve, adapt, merge, solve, generate_synthetic_samples # <-- MODIFIED IMPORT
from src.utils import save_json, load_json
from src.hf_sync import periodic_sync_check

def run_pipeline_for_single_query(
    hard_list_idx: int,
    target_query: str,
    config: Dict[str, Any],
    embedding_model: SentenceTransformer,
    exemplar_data: Dict[str, Any],
    api_managers: Dict[str, Any],
    run_mode: str = 'full',
    existing_log: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes the RAG pipeline for a single query, supporting different execution modes.

    Args:
        ... (standard arguments) ...
        run_mode (str): Controls execution flow.
            - 'full': Runs the entire pipeline from start to finish.
            - 'intermediate': Runs only retrieve, adapt, and merge steps.
            - 'solve_only': Runs only the solve step, using pre-computed intermediate results.
        existing_log (Optional[Dict]): A pre-existing log from the intermediate phase,
                                       required for 'solve_only' mode.
    """
    logger = logging.getLogger(__name__)
    
    # --- Log Initialization ---
    if run_mode == 'solve_only' and existing_log:
        run_log = existing_log
        # Reset status to ensure we correctly log the outcome of the solve step
        run_log['pipeline_status'] = "PENDING_SOLVE"
        print(f"\nResuming pipeline for Query #{hard_list_idx} (Solve Phase)...")
    else:
        # Standard initialization for 'full' or 'intermediate' modes
        print("\n" + "="*80)
        print(f"Processing Query #{hard_list_idx}: '{target_query[:100]}...'")
        print("="*80)
        logger.info(f"--- Starting pipeline for Query #{hard_list_idx}: '{target_query[:80]}...' ---")
        run_log = {
            "target_query_original_hard_list_idx": hard_list_idx,
            "target_query_text": target_query,
            "config_flags_used": {
                key: config.get(key) for key in [
                    "USE_RETRIEVAL", "SELF_SAMPLING", # <-- Added SELF_SAMPLING to logged configs
                    "APPLY_NORMALIZATION", "APPLY_TRANSFORMATION_1",
                    "APPLY_TRANSFORMATION_2", "APPLY_TRANSFORMATION_3", "APPLY_MERGING",
                    "DEFER_SOLVE_STEP", "TOP_N_CANDIDATES_RETRIEVAL", "N_PASS_ATTEMPTS",
                    "N_SELF_SAMPLES", # <-- Added N_SELF_SAMPLES to logged configs
                    "DEFAULT_PASS_N_SOLVER_TEMPERATURE"
                ]
            },
            "pipeline_status": "PENDING",
            "steps": {}
        }

    # --- API Manager Selection ---
    provider_for_adapt = config.get('API_PROVIDER_ADAPTATION', 'gemini')
    manager_for_adapt = api_managers[provider_for_adapt]
    provider_for_solve = config.get('API_PROVIDER_SOLVER', 'gemini')
    manager_for_solve = api_managers[provider_for_solve]
    
    # --- Pipeline Execution with Failure Checks ---
    final_exemplars_for_solve = []
    pipeline_halted = False

    # --- Phase 1: Intermediate Steps (Retrieve, Adapt, Merge, OR Self-Sample) ---
    if run_mode in ['full', 'intermediate']:
        if config.get('USE_RETRIEVAL', True):
            # Step 1: Retrieve
            print("\n[STEP 1] RETRIEVE")
            retrieval_result = retrieve(
                target_query=target_query, embedding_model=embedding_model,
                exemplar_questions=exemplar_data['questions'], embedded_exemplars=exemplar_data['embeddings'],
                top_k=config['TOP_N_CANDIDATES_RETRIEVAL']
            )
            run_log['steps']['retrieval'] = retrieval_result
            if retrieval_result['status'] == 'FAILURE':
                run_log['pipeline_status'] = "FAILURE: Retrieval failed."
                logger.error(run_log['pipeline_status'])
                print("  -> Retrieval FAILED. Halting pipeline for this query.")
                pipeline_halted = True
            else:
                print(f"  -> Retrieved indices: {retrieval_result['retrieved_indices']}")

            # Step 2: Adapt
            if not pipeline_halted:
                print("\n[STEP 2] ADAPT")
                adapt_result = adapt(
                    target_query=target_query, retrieved_indices=retrieval_result['retrieved_indices'],
                    exemplar_questions=exemplar_data['questions'], exemplar_solutions=exemplar_data['solutions'],
                    api_manager=manager_for_adapt, config=config
                )
                run_log['steps']['adaptation'] = adapt_result
                if adapt_result['status'] == 'FAILURE':
                    run_log['pipeline_status'] = "FAILURE: Adaptation failed for all exemplars."
                    logger.error(run_log['pipeline_status'])
                    print("  -> Adaptation FAILED for all exemplars. Halting pipeline for this query.")
                    pipeline_halted = True
                else:
                    for i, text in enumerate(adapt_result.get('adapted_texts', [])):
                        print(f"  -> Adapted text #{i+1} (start): '{text[:120]}...'")
                    if adapt_result['status'] == 'PARTIAL_SUCCESS':
                        print(f"  -> WARNING: Adaptation partially succeeded. {len(adapt_result.get('failed_adaptations', []))} exemplars failed.")
            
            # Step 3: Merge
            if not pipeline_halted:
                print("\n[STEP 3] MERGE")
                merge_result = merge(
                    target_query=target_query, adapted_texts=adapt_result['adapted_texts'],
                    embedding_model=embedding_model, api_manager=manager_for_adapt, config=config
                )
                run_log['steps']['merging'] = merge_result
                if merge_result['status'] == 'SKIPPED':
                    print("  -> Merging was SKIPPED as per config.")
                for i, text in enumerate(merge_result.get('merged_texts', [])):
                    print(f"  -> Final merged text #{i+1} (start): '{text[:120]}...'")
                final_exemplars_for_solve = merge_result['merged_texts']

        # <<< --- START OF NEW LOGIC --- >>>
        elif config.get('SELF_SAMPLING', False):
            print("\n[STEP 1] GENERATE SYNTHETIC SAMPLES (Self-Sampling Mode)")
            # Use the powerful solver's manager for this creative task
            sampling_result = generate_synthetic_samples(
                target_query=target_query,
                api_manager=manager_for_solve,
                config=config
            )
            run_log['steps']['self_sampling'] = sampling_result
            if "FAILURE" in sampling_result['status']:
                run_log['pipeline_status'] = "FAILURE: Self-sampling failed for all attempts."
                logger.error(run_log['pipeline_status'])
                print("  -> Self-Sampling FAILED for all attempts. Halting pipeline.")
                pipeline_halted = True
            else:
                final_exemplars_for_solve = sampling_result['synthetic_samples']
                print(f"  -> Generated {len(final_exemplars_for_solve)} synthetic samples.")
                if sampling_result['status'] == 'PARTIAL_SUCCESS':
                    print(f"  -> WARNING: {len(sampling_result['failed_generations'])} sample generations failed.")
        # <<< --- END OF NEW LOGIC --- >>>
        
        else:
            print("\n[STEP 1, 2, 3] SKIPPED (USE_RETRIEVAL and SELF_SAMPLING are False).")
            run_log['steps']['retrieval'] = {"status": "SKIPPED", "reason": "USE_RETRIEVAL is False"}
            run_log['steps']['adaptation'] = {"status": "SKIPPED", "reason": "USE_RETRIEVAL is False"}
            run_log['steps']['merging'] = {"status": "SKIPPED", "reason": "USE_RETRIEVAL is False"}
            final_exemplars_for_solve = []

    elif run_mode == 'solve_only':
        print("\n[STEP 1, 2, 3] SKIPPED (Running in solve_only mode). Loading intermediate results.")
        pipeline_halted = "FAILURE" in run_log.get("pipeline_status", "")
        if not pipeline_halted:
            # Handle loading exemplars from either RAG or Self-Sampling
            if 'merging' in run_log.get('steps', {}):
                final_exemplars_for_solve = run_log.get('steps', {}).get('merging', {}).get('merged_texts', [])
            elif 'self_sampling' in run_log.get('steps', {}):
                 final_exemplars_for_solve = run_log.get('steps', {}).get('self_sampling', {}).get('synthetic_samples', [])
            print(f"  -> Loaded {len(final_exemplars_for_solve)} exemplars for solving.")
        else:
            print("  -> Prior step failed, solve will be skipped.")

    # --- Phase 2: Final Solving Step ---
    if run_mode in ['full', 'solve_only']:
        if not pipeline_halted:
            print("\n[STEP 4] SOLVE")
            solve_result = solve(
                target_query=target_query, final_exemplars=final_exemplars_for_solve,
                api_manager=manager_for_solve, config=config
            )
            run_log['steps']['solving'] = solve_result
            solution_texts = [attempt for attempt in solve_result.get('solution_attempts', []) if isinstance(attempt, str)]
            failed_attempts = sum(1 for attempt in solve_result.get('solution_attempts', []) if isinstance(attempt, dict))
            for i, text in enumerate(solution_texts): print(f"  -> Solution attempt #{i+1} (start): '{text[:120]}...'")
            if failed_attempts > 0: print(f"  -> {failed_attempts} solution attempt(s) FAILED.")
            
            run_log['llm_final_solution_attempts_texts'] = solution_texts
            if "FAILURE" not in run_log['pipeline_status']:
                 run_log['pipeline_status'] = "SUCCESS"
        else:
            run_log['steps']['solving'] = {"status": "SKIPPED", "reason": "Pipeline halted due to critical failure in a prior step."}

    elif run_mode == 'intermediate':
        print("\n[STEP 4] SOLVE DEFERRED.")
        run_log['steps']['solving'] = {"status": "DEFERRED"}
        if not pipeline_halted:
             run_log['pipeline_status'] = "INTERMEDIATE_COMPLETE"

    logger.info(f"--- Pipeline finished for Query #{hard_list_idx} with status: {run_log['pipeline_status']} ---")
    return run_log


def run_experiments(
    experiment_configs: List[Dict[str, Any]],
    global_config: Dict[str, Any],
    hard_questions: List[str],
    embedding_model: SentenceTransformer,
    exemplar_data: Dict[str, Any],
    api_managers: Dict[str, Any]
) -> Dict[str, List[Dict]]:
    """
    Orchestrates running multiple experiments with different configurations.
    Supports both standard and cross-experiment deferred execution modes.
    """
    logger = logging.getLogger(__name__)
    all_results = {}

    # --- REWRITTEN LOGIC: Check for and handle cross-experiment deferred execution ---
    is_cross_experiment_defer_enabled = any(
        exp.get('DEFER_SOLVE_STEP', False) for exp in experiment_configs
    )

    if is_cross_experiment_defer_enabled:
        logger.info("Cross-experiment deferred mode is ENABLED. Running in two phases.")
        print("\n" + "#"*25 + " PHASE 1: EXECUTING INTERMEDIATE STEPS FOR ALL EXPERIMENTS " + "#"*25)
        
        # --- PHASE 1: Intermediate Steps for ALL experiments ---
        for exp_overrides in experiment_configs:
            current_config = global_config.copy()
            current_config.update(exp_overrides)
            exp_name = current_config.get("experiment_name", "unnamed_experiment")
            
            # Only run intermediate steps for experiments that are actually deferred
            if not current_config.get('DEFER_SOLVE_STEP', False):
                logger.warning(f"Experiment '{exp_name}' does not have DEFER_SOLVE_STEP enabled. It will be SKIPPED in this run.")
                continue

            logger.info(f"########## Starting Phase 1 (Intermediate) for Experiment: {exp_name} ##########")
            log_file_path = os.path.join(global_config['RESULTS_DIR'], f"{exp_name}_run_log.json")
            
            run_logs = load_json(log_file_path) or []
            completed_intermediate_indices = {log['target_query_original_hard_list_idx'] for log in run_logs if log.get('pipeline_status') == 'INTERMEDIATE_COMPLETE'}
            queries_to_process = [(idx, q) for idx, q in enumerate(hard_questions) if idx not in completed_intermediate_indices]

            if queries_to_process:
                for loop_idx, (original_idx, query_text) in enumerate(tqdm(queries_to_process, desc=f"{exp_name} - Phase 1: Intermediate")):
                    intermediate_log = run_pipeline_for_single_query(
                        hard_list_idx=original_idx, target_query=query_text, config=current_config,
                        embedding_model=embedding_model, exemplar_data=exemplar_data, api_managers=api_managers,
                        run_mode='intermediate'
                    )
                    run_logs.append(intermediate_log)
                    save_json(run_logs, log_file_path)
                    periodic_sync_check(loop_idx, current_config)
            else:
                logger.info(f"All intermediate steps for '{exp_name}' are already complete.")

        print("\n" + "#"*25 + " PHASE 1 COMPLETE " + "#"*25)
        print("\n" + "#"*25 + " PHASE 2: EXECUTING FINAL SOLVE STEPS FOR ALL EXPERIMENTS " + "#"*25)

        # --- PHASE 2: Final Solving Steps for ALL experiments ---
        for exp_overrides in experiment_configs:
            current_config = global_config.copy()
            current_config.update(exp_overrides)
            exp_name = current_config.get("experiment_name", "unnamed_experiment")

            # Only run solve steps for experiments that are deferred
            if not current_config.get('DEFER_SOLVE_STEP', False):
                continue
            
            logger.info(f"########## Starting Phase 2 (Solving) for Experiment: {exp_name} ##########")
            log_file_path = os.path.join(global_config['RESULTS_DIR'], f"{exp_name}_run_log.json")

            intermediate_logs = load_json(log_file_path) or []
            logs_to_solve = [log for log in intermediate_logs if log.get('pipeline_status') == 'INTERMEDIATE_COMPLETE']
            
            if logs_to_solve:
                completed_logs_map = {log['target_query_original_hard_list_idx']: log for log in intermediate_logs if log.get('pipeline_status') != 'INTERMEDIATE_COMPLETE'}

                for loop_idx, log_to_solve in enumerate(tqdm(logs_to_solve, desc=f"{exp_name} - Phase 2: Solving")):
                    original_idx = log_to_solve['target_query_original_hard_list_idx']
                    query_text = log_to_solve['target_query_text']
                    
                    completed_log = run_pipeline_for_single_query(
                        hard_list_idx=original_idx, target_query=query_text, config=current_config,
                        embedding_model=embedding_model, exemplar_data=exemplar_data, api_managers=api_managers,
                        run_mode='solve_only', existing_log=log_to_solve
                    )
                    completed_logs_map[original_idx] = completed_log
                    save_json(list(completed_logs_map.values()), log_file_path)
                    periodic_sync_check(loop_idx, current_config)
                
                final_logs = list(completed_logs_map.values())
            else:
                 logger.info(f"All solve steps for '{exp_name}' are already complete.")
                 final_logs = intermediate_logs
            
            save_json(final_logs, log_file_path)
            all_results[exp_name] = final_logs
            logger.info(f"########## Finished Experiment: {exp_name} ##########")

        print("\n" + "#"*25 + " PHASE 2 COMPLETE. ALL EXPERIMENTS FINISHED. " + "#"*25)

    else:
        # --- Original Mode: Run each experiment sequentially ---
        logger.info("Deferred mode is DISABLED. Running experiments sequentially.")
        for exp_overrides in experiment_configs:
            current_config = global_config.copy()
            current_config.update(exp_overrides)
            exp_name = current_config.get("experiment_name", "unnamed_experiment")
            logger.info(f"########## Starting Experiment: {exp_name} ##########")
            log_file_path = os.path.join(global_config['RESULTS_DIR'], f"{exp_name}_run_log.json")
            
            # This logic handles both standard (non-deferred) and single-experiment deferred runs
            if not current_config.get('DEFER_SOLVE_STEP', False):
                # --- Standard Mode: Run query-by-query ---
                logger.info(f"Running '{exp_name}' in standard (query-by-query) mode.")
                run_logs = load_json(log_file_path) or []
                completed_indices = {log['target_query_original_hard_list_idx'] for log in run_logs}
                queries_to_process = [(idx, q) for idx, q in enumerate(hard_questions) if idx not in completed_indices]
                
                if not queries_to_process:
                    logger.info(f"All queries for '{exp_name}' are already processed. Skipping.")
                    all_results[exp_name] = run_logs
                    continue

                for loop_idx, (original_idx, query_text) in enumerate(tqdm(queries_to_process, desc=f"Running {exp_name}")):
                    single_run_log = run_pipeline_for_single_query(
                        hard_list_idx=original_idx, target_query=query_text, config=current_config,
                        embedding_model=embedding_model, exemplar_data=exemplar_data, api_managers=api_managers,
                        run_mode='full'
                    )
                    run_logs.append(single_run_log)
                    save_json(run_logs, log_file_path)
                    periodic_sync_check(loop_idx, current_config)
            else:
                # --- Single-Experiment Deferred Mode ---
                logger.info(f"Running '{exp_name}' in single-experiment deferred solve mode.")
                
                # PHASE 1: Intermediate Steps
                print(f"\n--- {exp_name}: STARTING PHASE 1 of 2 (Intermediate Steps) ---")
                run_logs = load_json(log_file_path) or []
                completed_intermediate_indices = {log['target_query_original_hard_list_idx'] for log in run_logs if log.get('pipeline_status') == 'INTERMEDIATE_COMPLETE'}
                queries_to_process = [(idx, q) for idx, q in enumerate(hard_questions) if idx not in completed_intermediate_indices]

                if queries_to_process:
                    for loop_idx, (original_idx, query_text) in enumerate(tqdm(queries_to_process, desc=f"{exp_name} - Phase 1: Intermediate")):
                        intermediate_log = run_pipeline_for_single_query(
                            hard_list_idx=original_idx, target_query=query_text, config=current_config,
                            embedding_model=embedding_model, exemplar_data=exemplar_data, api_managers=api_managers,
                            run_mode='intermediate'
                        )
                        run_logs.append(intermediate_log)
                        save_json(run_logs, log_file_path)
                        periodic_sync_check(loop_idx, current_config)
                else:
                    logger.info(f"All intermediate steps for '{exp_name}' are already complete.")

                # PHASE 2: Final Solving Step
                print(f"\n--- {exp_name}: STARTING PHASE 2 of 2 (Final Solving) ---")
                intermediate_logs = load_json(log_file_path)
                logs_to_solve = [log for log in intermediate_logs if log.get('pipeline_status') == 'INTERMEDIATE_COMPLETE']
                
                if logs_to_solve:
                    completed_logs_map = {log['target_query_original_hard_list_idx']: log for log in intermediate_logs if log.get('pipeline_status') != 'INTERMEDIATE_COMPLETE'}

                    for loop_idx, log_to_solve in enumerate(tqdm(logs_to_solve, desc=f"{exp_name} - Phase 2: Solving")):
                        original_idx = log_to_solve['target_query_original_hard_list_idx']
                        query_text = log_to_solve['target_query_text']
                        
                        completed_log = run_pipeline_for_single_query(
                            hard_list_idx=original_idx, target_query=query_text, config=current_config,
                            embedding_model=embedding_model, exemplar_data=exemplar_data, api_managers=api_managers,
                            run_mode='solve_only', existing_log=log_to_solve
                        )
                        completed_logs_map[original_idx] = completed_log
                        save_json(list(completed_logs_map.values()), log_file_path)
                        periodic_sync_check(loop_idx, current_config)
                    
                    run_logs = list(completed_logs_map.values())
                else:
                     logger.info(f"All solve steps for '{exp_name}' are already complete.")
                     run_logs = intermediate_logs
            
            save_json(run_logs, log_file_path)
            logger.info(f"########## Finished Experiment: {exp_name} ##########")
            all_results[exp_name] = run_logs
        
    return all_results