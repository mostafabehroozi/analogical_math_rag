#======================================================================
#   File: src/orchestration.py
#======================================================================

# src/orchestration.py

"""
Orchestration module for the Analogical Reasoning RAG project.
This module chains together the individual steps from `pipeline_steps.py` to
run the full RAG pipeline.
It manages running experiments for multiple queries
and configurations, and handles the saving and loading of results for
pausing and resuming.
This version is updated to be API provider-agnostic and to handle detailed
error states from the pipeline steps.
It logs partial progress when failures
occur, preventing data loss and enabling targeted retries.
MODIFIED: This version now supports a deferred execution mode via the
`DEFER_SOLVE_STEP` config flag.
When enabled, it runs all intermediate steps
(retrieve, adapt, merge) for all queries first, then runs the final solve
step for all queries in a second phase.
REWRITTEN: The `run_experiments` function now implements a global, cross-experiment
deferred execution.
If ANY experiment has `DEFER_SOLVE_STEP` set to True, the
entire run switches to a two-phase model:
1. Phase 1: All intermediate steps for ALL experiments are completed.
2. Phase 2: All final solving steps for ALL experiments are completed.
This optimizes API usage by batching all expensive 'solve' calls together.
This version also integrates new, optional pipeline steps for self-sampling,
augmentation, analogical adaptation, the NEW Analogical Consistency check,
the Group-Based Self-Consistency Selection, and the NEW Hierarchical Augmentation.
PERFORMANCE FIX: The call to the `retrieve` function has been updated to pass
a pre-computed hash map, enabling O(1) self-match detection and resolving a
major performance bottleneck.
NEW FEATURE: Added `APPLY_FULL_PIPELINE_RETRY`. If True, the entire pipeline
(Retrieval -> Adaptation -> Merging -> Solving) is re-run N times, rather than
just retrying the final Solver step N times.
NEW FEATURE: Added Pipeline Simplification.
- Workflow A: Simplification of Retrieved Samples (replaces standard adaptation).
- Workflow B: Simplification of Main Question (alters solving strategy).
NEW FEATURE: Multi-Strategy Branching (Mirroring).
- Supports branching the pipeline after Mirroring to evaluate "Base" vs "Redundancy"
  filtering strategies in parallel.

NEW FEATURE: High-Resolution Execution Tracing.
- Collects granular prompt/response logs from all steps and aggregates them
  into a master `execution_trace` list.
"""

import logging
from tqdm import tqdm
import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# Import our custom modules
from src.pipeline_steps import (
    retrieve, adapt, merge, solve,
    self_sample, augment_question, select_augmented_questions, analogical_adapt,
    generate_reasoning_pathways, 
    solve_with_group_consistency, 
    solve_hierarchical_tree, 
    solve_with_analogical_consistency, 
    optimize_demonstrations_via_mirroring,
    simplify_retrieved_samples,
    solve_via_main_simplification
)
from src.utils import save_json, load_json
from src.hf_sync import periodic_sync_check
from src.prompts import EXEMPLAR_FORMAT, create_analogical_adaptation_prompt

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
    Executes the RAG pipeline for a single query, supporting different execution modes
    and new features like self-sampling and analogical adaptation.
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
        # Ensure execution_trace exists if resuming old logs
        if "execution_trace" not in run_log:
            run_log["execution_trace"] = []
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
                    # Core flags
                    "USE_RETRIEVAL", "APPLY_NORMALIZATION", "APPLY_TRANSFORMATION_1",
                    "APPLY_TRANSFORMATION_2", "APPLY_TRANSFORMATION_3", "APPLY_MERGING",
                    "DEFER_SOLVE_STEP", "TOP_N_CANDIDATES_RETRIEVAL", "N_PASS_ATTEMPTS",
                    # New Feature Flags
                    "APPLY_SELF_SAMPLING", "SELF_SAMPLING_N",
                    "APPLY_ANALOGICAL_ADAPTATION", "ANALOGICAL_GROUP_SETS",
                    "APPLY_SELF_SAMPLING_AUGMENTATION", "APPLY_ANALOGICAL_ADAPTATION_AUGMENTATION",
                    "ANALOGICAL_USE_MAIN_QUERY_AS_AUGMENTATION", # <--- NEW FLAG LOGGED
                    "SELECTIVE_AUGMENTATION_SAMPLING", "AUGMENT_K", "AUGMENT_N",
                    # Consistency Flags
                    "APPLY_CONSISTENCY_ANALOGICAL_CHECK", "CONSISTENCY_GENERATION_MODE",
                    "CONSISTENCY_PATHWAYS_K", "CONSISTENCY_SAMPLES_PER_PATHWAY_N",
                    # Group Consistency Flags
                    "APPLY_GROUP_CONSISTENCY_SELECTION", "GROUP_CONSISTENCY_CANDIDATES",
                    "GROUP_CONSISTENCY_SAMPLES_N",
                    # Hierarchical Augmentation Flags
                    "APPLY_HIERARCHICAL_AUGMENTATION", "HIERARCHICAL_TREE_DEPTH",
                    "HIERARCHICAL_BRANCHING_FACTOR", "HIERARCHICAL_LEAF_RETRIEVAL_ENABLED",
                    # Reverse Validation Flags
                    "APPLY_REVERSE_VALIDATION", "REVERSE_VALIDATION_CANDIDATES_N",
                    "REVERSE_VALIDATION_RETRIEVAL_K", "REVERSE_VALIDATION_ATTEMPTS_N",
                    # Simplification Flags
                    "APPLY_SIMPLIFICATION", "SIMPLIFY_RETRIEVED_SAMPLES", "SIMPLIFY_MAIN_QUESTION",
                    # Mirroring Flags
                    "APPLY_MIRROR_AS_EVALUATOR", "MIRROR_EVALUATE_BASE_FILTERING",
                    # Full Pipeline Retry Flag
                    "APPLY_FULL_PIPELINE_RETRY"
                ]
            },
            "pipeline_status": "PENDING",
            "execution_trace": [], # NEW: High-resolution trace log
            "steps": {},
            "steps_alternatives": {} # NEW: Store alternative strategies (Base vs Redundancy)
        }

    # --- API Manager Selection ---
    provider_for_adapt = config.get('API_PROVIDER_ADAPTATION', 'gemini')
    manager_for_adapt = api_managers[provider_for_adapt]
    provider_for_solve = config.get('API_PROVIDER_SOLVER', 'gemini')
    manager_for_solve = api_managers[provider_for_solve]
    
    # NEW: Specific Manager for Augmentation
    provider_for_aug = config.get('API_PROVIDER_AUGMENTATION', provider_for_adapt) # Fallback to adapt if not set
    manager_for_aug = api_managers[provider_for_aug]
    
    # NEW: Specific Manager for Evaluation (needed for Reverse Validation loop)
    provider_for_eval = config.get('API_PROVIDER_EVALUATOR', 'gemini')
    manager_for_eval = api_managers[provider_for_eval]

    # NEW: Specific Manager for Simplification
    provider_for_simp = config.get('API_PROVIDER_SIMPLIFICATION', provider_for_adapt)
    manager_for_simp = api_managers[provider_for_simp]

    # --- NEW: Branch for Reverse Validation (Analogical Consistency) ---
    if config.get('APPLY_REVERSE_VALIDATION', False):
        print("\n[MODE] ANALOGICAL CONSISTENCY (REVERSE VALIDATION) ACTIVATED")
        # This mode replaces the standard solve flow entirely.
        consistency_result = solve_with_analogical_consistency(
            target_query=target_query,
            exemplar_data=exemplar_data,
            embedding_model=embedding_model,
            api_manager_solve=manager_for_solve,
            api_manager_eval=manager_for_eval, # Needs eval manager to check validators
            config=config
        )
        
        # Aggregation: Extract trace
        if 'trace' in consistency_result:
            run_log['execution_trace'].extend(consistency_result.pop('trace'))
        
        run_log['steps']['solving'] = consistency_result
        
        if consistency_result['status'] == 'SUCCESS':
            # run_log['llm_final_solution_attempts_texts'] = [consistency_result['selected_candidate']]
            run_log['pipeline_status'] = "SUCCESS"
        else:
            run_log['pipeline_status'] = f"FAILURE: {consistency_result.get('error')}"
            
        return run_log

    # --- NEW: Branch for Hierarchical Augmentation (Tree-Based) ---
    if config.get('APPLY_HIERARCHICAL_AUGMENTATION', False):
        print("\n[MODE] HIERARCHICAL AUGMENTATION ACTIVATED")
        
        # Determine retry behavior
        is_full_retry = config.get('APPLY_FULL_PIPELINE_RETRY', False)
        n_passes = config.get("N_PASS_ATTEMPTS", 1)
        
        all_root_attempts = []
        last_hierarchical_result = None
        
        # --- SCENARIO A: Full Pipeline Retry (N Distinct Trees) ---
        if is_full_retry and n_passes > 1:
            logger.info(f"Full Pipeline Retry Enabled for Hierarchical Mode: Generating {n_passes} distinct trees.")
            
            # CRITICAL: Create a config copy that forces the INTERNAL solver 
            # to run only once per tree. We handle the looping here externally.
            single_pass_config = config.copy()
            single_pass_config['N_PASS_ATTEMPTS'] = 1
            
            full_pipeline_iterations_data = []

            for i in range(n_passes):
                print(f"\n[HIERARCHICAL ITERATION] {i+1}/{n_passes} (Building fresh tree)")
                
                # Run the full tree pipeline (Build -> Leaf Solve -> Root Solve)
                # This ensures a NEW simplification (Augmentation) happens every time.
                hierarchical_result = solve_hierarchical_tree(
                    target_query=target_query,
                    exemplar_data=exemplar_data,
                    embedding_model=embedding_model,
                    api_manager_adapt=manager_for_adapt,
                    api_manager_solve=manager_for_solve,
                    api_manager_augment=manager_for_aug,
                    config=single_pass_config # <--- Pass the N=1 config
                )
                
                # Aggregation: Extract trace
                if 'trace' in hierarchical_result:
                    run_log['execution_trace'].extend(hierarchical_result.pop('trace'))
                
                # Aggregate the single solution from this tree
                if hierarchical_result['status'] == 'SUCCESS':
                    sol = hierarchical_result.get('root_solution')
                    if sol:
                        all_root_attempts.append(sol)
                
                # Store debug data for this tree
                full_pipeline_iterations_data.append({
                    "iteration": i,
                    "tree_structure": hierarchical_result.get('tree_structure'),
                    "root_solution": hierarchical_result.get('root_solution')
                })
                
                last_hierarchical_result = hierarchical_result
            
            # Store iteration data
            run_log['full_pipeline_iterations_data'] = full_pipeline_iterations_data

        # --- SCENARIO B: Standard Pass@N (1 Tree, N Root Solves) ---
        else:
            # If retry is False, we pass the original config.
            # The internal logic in propagate_solutions_upward handles the N loops.
            logger.info(f"Standard Hierarchical Mode: 1 Tree, Pass@{n_passes} on Root.")
            
            last_hierarchical_result = solve_hierarchical_tree(
                target_query=target_query,
                exemplar_data=exemplar_data,
                embedding_model=embedding_model,
                api_manager_adapt=manager_for_adapt,
                api_manager_solve=manager_for_solve,
                api_manager_augment=manager_for_aug,
                config=config
            )
            
            # Aggregation: Extract trace
            if 'trace' in last_hierarchical_result:
                run_log['execution_trace'].extend(last_hierarchical_result.pop('trace'))
            
            if last_hierarchical_result['status'] == 'SUCCESS':
                # Grab the list generated internally
                all_root_attempts = last_hierarchical_result.get('root_solution_attempts', [])
                # Fallback if the list is empty but solution exists
                if not all_root_attempts and last_hierarchical_result.get('root_solution'):
                    all_root_attempts = [last_hierarchical_result['root_solution']]

        # --- Final Log Construction ---
        run_log['steps']['hierarchical_process'] = last_hierarchical_result
        
        # Populate the location the Evaluator looks for
        run_log['steps']['solving'] = {
            "status": "SUCCESS" if all_root_attempts else "FAILURE",
            "solution_attempts": all_root_attempts
        }
        
        # Also populate the top-level text list for the analysis script
        run_log['llm_final_solution_attempts_texts'] = all_root_attempts
        
        if all_root_attempts:
            run_log['pipeline_status'] = "SUCCESS"
        else:
            run_log['pipeline_status'] = "FAILURE: Hierarchical process failed to produce solutions."
        return run_log

    # --- NEW: Branch for Analogical Consistency Check (The "Pathway" approach - OLD VERSION) ---
    if config.get('APPLY_CONSISTENCY_ANALOGICAL_CHECK', False):
        print("\n[MODE] ANALOGICAL CONSISTENCY CHECK (PATHWAY) ACTIVATED")
        
        # 1. Generate Layer 1 (Reasoning Pathways / Exemplars)
        print(f"[LAYER 1] Generating {config.get('CONSISTENCY_PATHWAYS_K')} Reasoning Pathways...")
        # Use manager_for_aug for generating the pathways (augmentations)
        pathways_result = generate_reasoning_pathways(target_query, manager_for_aug, config)
        
        # Aggregation: Extract trace
        if 'trace' in pathways_result:
            run_log['execution_trace'].extend(pathways_result.pop('trace'))
        
        if pathways_result['status'] == 'FAILURE':
            run_log['pipeline_status'] = "FAILURE: Pathway generation failed."
            run_log['consistency_analysis_data'] = {"error": pathways_result.get("error_info")}
            return run_log
            
        generated_pathways = pathways_result['pathway_exemplars']
        print(f"  -> Generated {len(generated_pathways)} pathways.")
        
        # 2. Generate Layer 2 (Sampling Main Question using each Pathway)
        layer_1_data = []
        n_samples = config.get("CONSISTENCY_SAMPLES_PER_PATHWAY_N", 3)
        temp_layer_2 = config.get("CONSISTENCY_LAYER_2_TEMPERATURE", 0.7)
        model_name = config.get('GEMINI_MODEL_NAME_FINAL_SOLVER') # Default to Gemini for now, or adapt based on provider
        if config.get('API_PROVIDER_SOLVER') == 'avalai': model_name = config.get('AVALAI_MODEL_NAME_FINAL_SOLVER')
        if config.get('API_PROVIDER_SOLVER') == 'ollama': model_name = config.get('OLLAMA_MODEL_NAME_FINAL_SOLVER')

        for i, pathway_text in enumerate(generated_pathways):
            print(f"\n[LAYER 2] Stress Testing Pathway #{i+1} ({n_samples} samples)...")
            
            # Create a prompt using THIS specific pathway as the only exemplar
            # We use a list of 1 because the prompt function expects a list
            prompt = create_analogical_adaptation_prompt(target_query, [pathway_text], config)
            
            samples = []
            for j in range(n_samples):
                print(f"    -> Generating Sample {j+1}/{n_samples}")
                resp = manager_for_solve.generate_content(prompt, model_name, temp_layer_2)
                
                # Manually log trace for this explicit loop since it's inside orchestration
                from src.utils import create_trace_entry
                run_log['execution_trace'].append(create_trace_entry(
                    "pathway_consistency", f"layer_2_pathway_{i}_sample_{j}",
                    {"pathway_text": pathway_text, "prompt": prompt}, resp, {"model": model_name, "temp": temp_layer_2}
                ))

                if resp['status'] == 'SUCCESS':
                    samples.append(resp['text'])
                else:
                    samples.append({"error": resp})
            
            layer_1_data.append({
                "pathway_id": i,
                "exemplar_text": pathway_text,
                "layer_2_results": samples
            })
            
        run_log['consistency_analysis_data'] = {
            "layer_1_pathways": layer_1_data
        }
        run_log['pipeline_status'] = "SUCCESS"
        return run_log

    # --- Standard Pipeline Execution ---
    
    # --- SETUP FULL PIPELINE RETRY LOGIC ---
    # Determine if we run the pipeline once (standard) or N times (new feature)
    full_retry_mode = config.get('APPLY_FULL_PIPELINE_RETRY', False)
    
    if full_retry_mode and run_mode == 'full':
        # If enabled, disable DEFER_SOLVE_STEP as complex state saving is not supported
        if config.get("DEFER_SOLVE_STEP", False):
            logger.warning("Disabling DEFER_SOLVE_STEP because FULL_PIPELINE_RETRY is On.")
            config['DEFER_SOLVE_STEP'] = False
            
        n_pipeline_iterations = config.get("N_PASS_ATTEMPTS", 1)
        n_solver_attempts_per_pass = 1 # We solve 1 time per pipeline iteration
        logger.info(f"Full Pipeline Retry Enabled: Running complete pipeline {n_pipeline_iterations} times.")
    else:
        n_pipeline_iterations = 1
        n_solver_attempts_per_pass = config.get("N_PASS_ATTEMPTS", 1) # Standard Pass@N

    aggregated_solution_attempts = []
    iteration_details = []
    final_pipeline_status = "PENDING"

    # --- PIPELINE ITERATION LOOP ---
    for iteration_idx in range(n_pipeline_iterations):
        if full_retry_mode:
            print(f"\n[FULL PIPELINE ITERATION] {iteration_idx + 1}/{n_pipeline_iterations}")
            
        pipeline_halted = False
        
        # Local container for this iteration's logs (to avoid overwriting the main structure if iterating)
        # Structure: { "primary": {step: val}, "base_filtering": {step: val}, ... }
        iter_log_strategies = {} 

        # --- Phase 1: Intermediate Steps ---
        if run_mode in ['full', 'intermediate']:
            # -- Step 1: Retrieve --
            retrieved_indices_initial = []
            
            if config.get('USE_RETRIEVAL', True):
                print("\n[STEP 1] RETRIEVE")
                # ========================= START OF MODIFICATION =========================
                # Pass the pre-computed hash map to the retrieve function for O(1) lookup.
                # This is the core of the performance fix.
                retrieval_result = retrieve(
                    target_query=target_query, embedding_model=embedding_model,
                    exemplar_questions=exemplar_data['questions'], embedded_exemplars=exemplar_data['embeddings'],
                    top_k=config['TOP_N_CANDIDATES_RETRIEVAL'],
                    question_to_index_map=exemplar_data.get('question_to_index')
                )
                # ========================== END OF MODIFICATION ==========================
                
                # Aggregation: Extract trace
                if 'trace' in retrieval_result:
                    run_log['execution_trace'].extend(retrieval_result.pop('trace'))

                # Retrieval is common to all strategies
                if retrieval_result['status'] == 'FAILURE':
                    final_pipeline_status = "FAILURE: Retrieval failed."
                    logger.error(final_pipeline_status)
                    print("  -> Retrieval FAILED. Halting pipeline for this query.")
                    pipeline_halted = True
                else:
                    retrieved_indices_initial = retrieval_result['retrieved_indices']
                    print(f"  -> Retrieved indices: {retrieved_indices_initial}")

            else:
                print("\n[STEP 1] RETRIEVE SKIPPED (USE_RETRIEVAL is False).")

            # ============================================================================
            # [STEP 1.5] MIRROR OPTIMIZATION (Analogical Consistency)
            # ============================================================================
            # Returns a dictionary of strategies: { "strategy_name": [indices], ... }
            
            candidate_strategies = {} # Map: strategy_name -> indices
            
            if not pipeline_halted:
                if config.get("APPLY_MIRROR_AS_EVALUATOR", False) and retrieved_indices_initial:
                    print("\n[STEP 1.5] MIRROR OPTIMIZATION")
                    
                    # Use the Solver Manager (requires reasoning)
                    mirror_result = optimize_demonstrations_via_mirroring(
                        target_query=target_query,
                        retrieved_indices=retrieved_indices_initial,
                        exemplar_data=exemplar_data,
                        api_manager=manager_for_solve, 
                        config=config
                    )

                    # 1. Capture Trace
                    if 'trace' in mirror_result:
                        run_log['execution_trace'].extend(mirror_result.pop('trace'))
                    
                    run_log['steps']['mirror_optimization'] = mirror_result

                    # 2. Determine Strategies (Branching)
                    if mirror_result['status'] == "BRANCHED":
                        # We have multiple lists to solve (Base vs Redundancy)
                        candidate_strategies = mirror_result['strategies']
                        print(f"  -> Branching Active. Strategies: {list(candidate_strategies.keys())}")
                        
                    elif mirror_result['status'] == 'SUCCESS':
                        # Standard single list
                        optimized = mirror_result['optimized_indices']
                        candidate_strategies = {"primary": optimized}
                        print(f"  -> Mirroring Complete. Optimized to {len(optimized)} candidates.")
                        
                    else:
                        logger.warning("Mirroring optimization failed. Proceeding with original retrieved indices.")
                        candidate_strategies = {"fallback": retrieved_indices_initial}
                else:
                    # No Mirroring -> Single strategy using retrieved indices
                    candidate_strategies = {"primary": retrieved_indices_initial}

            # ============================================================================
            # [STEP 2-6] STRATEGY LOOP (Adapt -> Merge -> Solve)
            # ============================================================================
            # Iterate through each strategy (e.g., "primary", "base_filtering", "redundancy_filtering")
            
            # Define which keys are considered "Primary" for backward compatibility in logs
            primary_keys = ["primary", "redundancy_filtering", "fallback"]

            for strategy_name, indices in candidate_strategies.items():
                if pipeline_halted: break # Global halt
                
                print(f"\n--- Strategy: {strategy_name.upper()} (Candidates: {len(indices)}) ---")
                
                local_step_log = {} # Log for this strategy
                current_exemplars_for_step = [] # Will hold text
                
                # We need to manually populate retrieval info in local log if we want completeness
                if config.get('USE_RETRIEVAL'):
                    local_step_log['retrieval'] = {"retrieved_indices": indices}

                # --- BRANCH: Simplification vs Standard Adaptation ---
                if config.get('APPLY_SIMPLIFICATION', False) and config.get('SIMPLIFY_RETRIEVED_SAMPLES', False):
                    print(f"[{strategy_name}] SIMPLIFY RETRIEVED SAMPLES")
                    simp_result = simplify_retrieved_samples(
                        retrieved_indices=indices,
                        exemplar_questions=exemplar_data['questions'],
                        exemplar_solutions=exemplar_data['solutions'],
                        api_manager=manager_for_simp,
                        config=config
                    )
                    if 'trace' in simp_result:
                        run_log['execution_trace'].extend(simp_result.pop('trace'))

                    local_step_log['simplification_of_samples'] = simp_result
                    
                    if simp_result.get('simplified_exemplars'):
                        current_exemplars_for_step.extend(simp_result['simplified_exemplars'])
                    else:
                        print(f"  -> WARNING: Sample Simplification failed for {strategy_name}.")
                else:
                    # -- Step 2: Standard Adapt --
                    if config.get('USE_RETRIEVAL'):
                        print(f"[{strategy_name}] ADAPT (Standard Transformations)")
                        adapt_result = adapt(
                            target_query=target_query, 
                            retrieved_indices=indices,
                            exemplar_questions=exemplar_data['questions'], 
                            exemplar_solutions=exemplar_data['solutions'],
                            api_manager=manager_for_adapt, 
                            config=config
                        )
                        if 'trace' in adapt_result:
                            run_log['execution_trace'].extend(adapt_result.pop('trace'))

                        local_step_log['adaptation'] = adapt_result
                        current_exemplars_for_step.extend(adapt_result.get('adapted_texts', []))
                    else:
                        print(f"[{strategy_name}] RETRIEVE & ADAPT SKIPPED.")

                # -- Step 3: Analogical Adaptation --
                if config.get('APPLY_ANALOGICAL_ADAPTATION', False):
                    print(f"[{strategy_name}] ADAPT (Analogical Adaptation)")
                    # (Simplified logic: Assumes existing functions handle augmentation correctly)
                    # Note: We pass the indices for this strategy
                    
                    augmented_qs_for_aa = None
                    if config.get('APPLY_ANALOGICAL_ADAPTATION_AUGMENTATION') and not config.get('ANALOGICAL_USE_MAIN_QUERY_AS_AUGMENTATION', False):
                        k = config.get('AUGMENT_K', 10)
                        aug_result = augment_question(target_query, k, manager_for_aug, config)
                        if 'trace' in aug_result:
                            run_log['execution_trace'].extend(aug_result.pop('trace'))
                        if aug_result['status'] == 'SUCCESS':
                            augmented_qs_for_aa = aug_result['augmented_questions']
                            if config.get('SELECTIVE_AUGMENTATION_SAMPLING'):
                                retrieved_texts = [EXEMPLAR_FORMAT.format(question=exemplar_data['questions'][i], solution=exemplar_data['solutions'][i]) for i in indices]
                                augmented_qs_for_aa = select_augmented_questions(augmented_qs_for_aa, config, embedding_model, retrieved_texts)

                    aa_result = analogical_adapt(
                        target_query, indices, exemplar_data, 
                        api_manager=manager_for_adapt, api_manager_augment=manager_for_aug,
                        config=config, embedding_model=embedding_model, augmented_questions=augmented_qs_for_aa
                    )
                    if 'trace' in aa_result:
                        run_log['execution_trace'].extend(aa_result.pop('trace'))

                    local_step_log['analogical_adaptation'] = aa_result
                    if aa_result.get('analogically_adapted_texts'):
                         current_exemplars_for_step = aa_result['analogically_adapted_texts']

                # -- Step 4: Self-Sampling --
                if config.get('APPLY_SELF_SAMPLING', False):
                    print(f"[{strategy_name}] SELF-SAMPLE")
                    self_sampled_texts = []
                    # (Standard self-sampling logic)
                    ss_result = self_sample(target_query, manager_for_adapt, config)
                    if 'trace' in ss_result:
                        run_log['execution_trace'].extend(ss_result.pop('trace'))
                    self_sampled_texts.extend(ss_result.get('self_sampled_texts', []))
                    
                    local_step_log['self_sampling'] = ss_result
                    current_exemplars_for_step.extend(self_sampled_texts)

                # -- Step 5: Merge --
                print(f"[{strategy_name}] MERGE")
                merge_result = merge(
                    target_query=target_query, adapted_texts=current_exemplars_for_step,
                    embedding_model=embedding_model, api_manager=manager_for_adapt, config=config
                )
                if 'trace' in merge_result:
                    run_log['execution_trace'].extend(merge_result.pop('trace'))

                local_step_log['merging'] = merge_result
                final_exemplars_for_solve = merge_result['merged_texts']
                
                # STORE INTERMEDIATE LOG FOR THIS STRATEGY
                iter_log_strategies[strategy_name] = local_step_log
                iter_log_strategies[strategy_name]['final_exemplars'] = final_exemplars_for_solve

        elif run_mode == 'solve_only':
            print("\n[STEPS 1-5] SKIPPED (Running in solve_only mode). Loading intermediate results.")
            pipeline_halted = "FAILURE" in run_log.get("pipeline_status", "")
            
            # In solve_only, we check existing logs.
            # If 'steps_alternatives' exists, we respect branching.
            # If not, we assume standard 'primary' strategy from 'steps'.
            
            # Load Primary
            candidate_strategies = {}
            if not pipeline_halted:
                # Primary
                primary_exemplars = run_log.get('steps', {}).get('merging', {}).get('merged_texts', [])
                if primary_exemplars:
                    candidate_strategies["primary"] = primary_exemplars # Value is exemplars list here, not indices
                
                # Alternatives
                alts = run_log.get('steps_alternatives', {})
                for name, data in alts.items():
                    alt_exemplars = data.get('merging', {}).get('merged_texts', [])
                    if alt_exemplars:
                        candidate_strategies[name] = alt_exemplars
                
                # Note: In solve_only mode, 'candidate_strategies' values are TEXTS, not INDICES.
                # We need to handle this diff in the solving phase below.
            else:
                print("  -> Prior step failed, solve will be skipped.")

        # --- Phase 2: Final Solving Step (For EACH Strategy) ---
        
        # We loop again or continue. Since run_mode='full' populates 'iter_log_strategies' with keys and exemplars,
        # and 'solve_only' populates 'candidate_strategies' with keys and exemplars, we can unify.
        
        strategies_to_solve = {}
        if run_mode in ['full', 'intermediate']:
             for name, data in iter_log_strategies.items():
                 strategies_to_solve[name] = data['final_exemplars']
        elif run_mode == 'solve_only':
             strategies_to_solve = candidate_strategies # Already mapping name -> exemplars

        if run_mode in ['full', 'solve_only'] and not pipeline_halted:
            primary_keys = ["primary", "redundancy_filtering", "fallback"]

            for strategy_name, final_exemplars_for_solve in strategies_to_solve.items():
                print(f"\n[{strategy_name}] SOLVE PHASE")
                solve_result = {}
                
                # --- BRANCH: Main Question Simplification Workflow (Workflow B) ---
                if config.get('APPLY_SIMPLIFICATION', False) and config.get('SIMPLIFY_MAIN_QUESTION', False):
                    print(f"[{strategy_name}] SOLVE (Simplification of Main Question)")
                    solve_result = solve_via_main_simplification(
                        target_query=target_query,
                        final_exemplars=final_exemplars_for_solve,
                        api_manager=manager_for_simp,
                        config=config
                    )
                
                # --- BRANCH: Group-Based Self-Consistency Selection ---
                elif config.get('APPLY_GROUP_CONSISTENCY_SELECTION', False):
                    print(f"[{strategy_name}] SOLVE (Group-Based Consistency Mode)")
                    solve_result = solve_with_group_consistency(
                        target_query=target_query,
                        available_exemplars=final_exemplars_for_solve,
                        api_manager=manager_for_solve,
                        config=config
                    )
                
                # --- BRANCH: Standard Solving (Pass@N) ---
                else:
                    print(f"[{strategy_name}] SOLVE (Standard)")
                    current_solver_config = config.copy()
                    current_solver_config['N_PASS_ATTEMPTS'] = n_solver_attempts_per_pass
                    
                    solve_result = solve(
                        target_query=target_query, final_exemplars=final_exemplars_for_solve,
                        api_manager=manager_for_solve, config=current_solver_config
                    )

                # Aggregation: Extract trace
                if 'trace' in solve_result:
                    run_log['execution_trace'].extend(solve_result.pop('trace'))

                # --- STORE RESULTS ---
                # Check if this is a Primary strategy
                is_primary = (strategy_name in primary_keys)
                
                if is_primary:
                    # Update Main Status
                    if solve_result['status'] == 'SUCCESS':
                        final_pipeline_status = "SUCCESS"
                    else:
                        final_pipeline_status = "FAILURE"
                    
                    # Store in main 'steps' for backward compatibility
                    # Note: We assume only ONE primary strategy exists usually.
                    # If Mirroring returns 'redundancy_filtering' as primary, we log it here.
                    run_log['steps']['solving'] = solve_result
                    
                    # Collect attempts for global aggregation
                    if 'solution_attempts' in solve_result:
                         aggregated_solution_attempts.extend(solve_result['solution_attempts'])
                    elif 'group_consistency_results' in solve_result:
                        # For group consistency, we might assume the first group's first attempt 
                        # or similar for "aggregation", but usually it's handled by specific evaluators.
                        pass
                else:
                    # Store in 'steps_alternatives'
                    if 'steps_alternatives' not in run_log:
                        run_log['steps_alternatives'] = {}
                    
                    if strategy_name not in run_log['steps_alternatives']:
                        run_log['steps_alternatives'][strategy_name] = {}
                        
                    run_log['steps_alternatives'][strategy_name]['solving'] = solve_result
                    
                    # We also ensure the intermediate steps for this alternative are saved there
                    if run_mode in ['full', 'intermediate'] and strategy_name in iter_log_strategies:
                         run_log['steps_alternatives'][strategy_name].update(iter_log_strategies[strategy_name])

        elif run_mode == 'intermediate':
            print("\n[STEP 6] SOLVE DEFERRED.")
            solve_result = {"status": "DEFERRED"}
            if not pipeline_halted:
                 final_pipeline_status = "INTERMEDIATE_COMPLETE"

        # Update logs for this iteration (Full Retry Mode)
        if full_retry_mode:
            # In full retry mode, we store detailed steps for each iteration separately
            # We assume Primary strategy for retry logic simplicity currently
            iteration_details.append({
                "iteration": iteration_idx,
                "context_exemplars": strategies_to_solve.get("primary", []), # Fallback
                "steps": iter_log_strategies.get("primary", {}),
                "solve_result": run_log['steps'].get('solving', {})
            })
            # Also update the main run_log steps with the LAST iteration's details just so it isn't empty
            if "primary" in iter_log_strategies:
                run_log['steps'].update(iter_log_strategies["primary"])
        else:
            # In standard mode, we update the main steps directly
            if "primary" in iter_log_strategies:
                run_log['steps'].update(iter_log_strategies["primary"])
            elif "redundancy_filtering" in iter_log_strategies:
                 run_log['steps'].update(iter_log_strategies["redundancy_filtering"])
            elif "fallback" in iter_log_strategies:
                 run_log['steps'].update(iter_log_strategies["fallback"])

    # --- END PIPELINE ITERATION LOOP ---

    # Final Aggregation
    run_log['pipeline_status'] = final_pipeline_status
    if full_retry_mode:
        run_log['full_pipeline_iterations_data'] = iteration_details
    
    # Consolidate all solution attempts (strings) for the evaluator
    all_solution_texts = [attempt for attempt in aggregated_solution_attempts if isinstance(attempt, str)]
    run_log['llm_final_solution_attempts_texts'] = all_solution_texts
    
    # Ensure the solve step in run_log accurately reflects the aggregated results
    if 'solving' in run_log['steps'] and aggregated_solution_attempts:
        run_log['steps']['solving']['solution_attempts'] = aggregated_solution_attempts

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