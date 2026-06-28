# src/core_simplification_layer2.py

import os
import logging
import re
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.utils import load_json, save_json, create_trace_entry
from src.hf_sync import periodic_sync_check
from src.api_manager import GeminiAPIManager, AvalAIAPIManager, OllamaAPIManager

# Import shared prompts
from src.prompts import (
    create_final_reasoning_prompt_simple,
    create_core_simp_zero_shot_prompt,
    create_core_simp_few_shot_prompt,
    create_core_simp_augmented_solver_prompt
)

# Import shared helpers from Phase 1 to keep code DRY
from src.core_simplification import _solve_and_evaluate, _parse_simplification_trace

logger = logging.getLogger(__name__)

def build_paired_test_suite(
    hard_questions: List[str],
    hard_solutions: List[str],
    embedding_model: SentenceTransformer,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Implements the 'Inverted Retrieval Paradigm'.
    Finds Top-K unseen test questions for every Phase-1 Donor.
    """
    logger.info("Building strictly paired test suite via Inverted Retrieval...")
    
    dataset_filename = config.get("CORE_SIMP_DATASET_NAME", "core_simp_dataset.json")
    dataset_path = os.path.join(config['RESULTS_DIR'], dataset_filename)
    
    donors = load_json(dataset_path)
    if not donors:
        logger.error(f"Cannot build test suite: Donor dataset {dataset_filename} is missing or empty.")
        return []

    # 1. Identify Unseen Questions
    donor_original_indices = {d['original_index'] for d in donors if 'original_index' in d}
    
    unseen_indices = []
    unseen_questions = []
    for idx, q in enumerate(hard_questions):
        # Must not be a donor, and must have a valid ground truth to evaluate against
        if idx not in donor_original_indices and hard_solutions and idx < len(hard_solutions) and hard_solutions[idx]:
            unseen_indices.append(idx)
            unseen_questions.append(q)

    if not unseen_questions:
        logger.warning("No unseen questions left to test!")
        return []

    # 2. Embed Donors and Unseen Pool
    logger.info(f"Embedding {len(donors)} donors and {len(unseen_questions)} unseen test questions...")
    donor_texts = [d['original_question'] for d in donors]
    donor_embeddings = embedding_model.encode(donor_texts, convert_to_numpy=True, show_progress_bar=False)
    unseen_embeddings = embedding_model.encode(unseen_questions, convert_to_numpy=True, show_progress_bar=False)

    # 3. Calculate Cosine Similarity Matrix (Donors x Unseen)
    similarity_matrix = cosine_similarity(donor_embeddings, unseen_embeddings)
    
    k_retrieval = config.get("CORE_SIMP_LAYER2_K_RETRIEVAL", 3)
    paired_test_suite = []
    already_assigned_unseen_indices = set()

    # 4. Assign Test Questions to Donors
    for donor_idx_in_matrix, donor in enumerate(donors):
        similarities = similarity_matrix[donor_idx_in_matrix]
        
        # Sort unseen indices by highest similarity to this donor
        sorted_unseen_matrix_indices = np.argsort(similarities)[::-1]
        
        assigned_count = 0
        for matrix_idx in sorted_unseen_matrix_indices:
            if assigned_count >= k_retrieval:
                break
                
            actual_unseen_idx = unseen_indices[matrix_idx]
            
            # Deduplication: Ensure strictly 1-to-1 mapping
            if actual_unseen_idx not in already_assigned_unseen_indices:
                already_assigned_unseen_indices.add(actual_unseen_idx)
                
                # Construct a clean demonstration from the donor's original and proxy questions
                donor_original = donor.get('original_question', '')
                donor_proxy = donor.get('proxy_question', '')
                demo_text = f"Original Question:\n{donor_original}\n\nSimplified Question:\n{donor_proxy}"
                
                # Bundle the test question with its perfect donor demonstration
                paired_test_suite.append({
                    "test_idx": actual_unseen_idx,
                    "test_question": hard_questions[actual_unseen_idx],
                    "ground_truth": hard_solutions[actual_unseen_idx],
                    "linked_donor_original_idx": donor.get('original_index'),
                    "linked_donor_trace_text": demo_text  # We inject the clean demo text here!
                })
                assigned_count += 1

    logger.info(f"Successfully built paired test suite with {len(paired_test_suite)} target questions.")
    return paired_test_suite


def run_parallel_evaluation_branches(
    test_item: Dict[str, Any],
    api_manager_solve: Any,
    api_manager_eval: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Runs Branch A (Baseline), Branch B (Zero-Shot), and Branch C (Few-Shot Analogical).
    """
    t_q = test_item["test_question"]
    t_gt = test_item["ground_truth"]
    d_trace = test_item["linked_donor_trace_text"]
    
    n_attempts = config.get("CORE_SIMP_LAYER2_N_ATTEMPTS", 5)
    temp_gen = config.get("CORE_SIMP_TEMPERATURE_GEN", 0.3)
    temp_solve = config.get("CORE_SIMP_TEMPERATURE_SOLVE", 1.0)
    
    # Model Selection
    if isinstance(api_manager_solve, GeminiAPIManager):
        m_gen = config.get('GEMINI_MODEL_NAME_ADAPTATION')
        m_solve = config.get('GEMINI_MODEL_NAME_FINAL_SOLVER')
    elif isinstance(api_manager_solve, AvalAIAPIManager):
        m_gen = config.get('AVALAI_MODEL_NAME_ADAPTATION')
        m_solve = config.get('AVALAI_MODEL_NAME_FINAL_SOLVER')
    elif isinstance(api_manager_solve, OllamaAPIManager):
        m_gen = config.get('OLLAMA_MODEL_NAME_ADAPTATION')
        m_solve = config.get('OLLAMA_MODEL_NAME_FINAL_SOLVER')
    
    local_trace = []
    results = {"test_idx": test_item["test_idx"], "donor_idx": test_item["linked_donor_original_idx"]}
    
    print(f"\n" + "="*70)
    print(f"  [LAYER 2] Evaluating Test Q#{test_item['test_idx']} (Linked to Donor #{test_item['linked_donor_original_idx']})")
    print("="*70)

    # -------------------------------------------------------------------------
    # BRANCH A: Baseline (Direct Solve)
    # -------------------------------------------------------------------------
    print(f"  -> [Branch A] Baseline Direct Solve (N={n_attempts})...")
    prompt_a = create_final_reasoning_prompt_simple(t_q, config)
    
    # Note: target_correct_to_beat=-1 ensures no early stopping. We want full Pass@N data.
    score_a, attempts_a = _solve_and_evaluate(
        t_q, t_gt, prompt_a, api_manager_solve, api_manager_eval, config,
        n_attempts, temp_solve, "branch_a_baseline", local_trace, target_correct_to_beat=-1
    )
    results["branch_a_score"] = score_a
    results["branch_a_attempts"] = attempts_a
    print(f"     => Score A: {score_a:.2f}")

    # -------------------------------------------------------------------------
    # Helper Function for Branches B and C (Fully Optimized)
    # -------------------------------------------------------------------------
    def _execute_simplification_branch(branch_name: str, gen_prompt: str, match_proxy: str = None, match_results: tuple = None) -> tuple:
        print(f"  -> [{branch_name}] Generating Proxy...")
        
        # 1. Generate Proxy
        resp_gen = api_manager_solve.generate_content(gen_prompt, m_gen, temp_gen)
        local_trace.append(create_trace_entry("layer2", f"{branch_name}_generate", {"prompt": gen_prompt}, resp_gen, {"model": m_gen}))
        
        is_fallback = False
        proxy_q = ""
        
        if resp_gen['status'] != 'SUCCESS':
            is_fallback = True
            print(f"     => Proxy generation API failed. Triggering Fallback.")
        else:
            parsed = _parse_simplification_trace(resp_gen['text'])
            proxy_q = parsed.get("proxy_question", "")
            
            def clean_text(t): return re.sub(r'\W+', '', t.lower())
            
            # Failsafe Checks
            if not proxy_q or clean_text(proxy_q) == clean_text(t_q):
                is_fallback = True
                print(f"     => Failsafe triggered or parsing failed. Triggering Fallback.")
                
            # === NEW: CONVERGENCE OPTIMIZATION ===
            elif match_proxy and clean_text(proxy_q) == clean_text(match_proxy):
                print(f"     => [CONVERGENCE] Branch generated the exact same proxy as Branch B!")
                print(f"     => Recycling Branch B's solve and evaluation results to save API costs.")
                local_trace.append({
                    "step": "layer2", 
                    "sub_step": f"{branch_name}_convergence", 
                    "note": "Recycled previous branch results because generated proxy was identical."
                })
                # match_results contains (score_b, attempts_b)
                return match_results[0], match_results[1], resp_gen['text'], "CONVERGED_WITH_B", proxy_q
            # =====================================

        # 2. Execute solving logic
        if is_fallback:
            # OPTIMIZATION: Recycle Branch A's results
            print(f"     => Re-using Branch A baseline results to save API costs.")
            local_trace.append({"step": "layer2", "sub_step": f"{branch_name}_fallback_solve", "note": "Recycled Branch A results"})
            return score_a, attempts_a, resp_gen.get('text', 'API_FAILED'), "FALLBACK_TRIGGERED", proxy_q
        else:
            # Solve the Proxy
            print(f"  -> [{branch_name}] Solving Proxy...")
            prompt_proxy_solve = create_final_reasoning_prompt_simple(proxy_q, config)
            resp_proxy = api_manager_solve.generate_content(prompt_proxy_solve, m_solve, temp_solve)
            local_trace.append(create_trace_entry("layer2", f"{branch_name}_solve_proxy", {"prompt": prompt_proxy_solve}, resp_proxy, {"model": m_solve}))
            
            # === NEW: EMPTY RESPONSE FAST-FAIL OPTIMIZATION ===
            proxy_solution_text = resp_proxy.get('text', '').strip()
            
            if resp_proxy['status'] != 'SUCCESS' or not proxy_solution_text:
                error_reason = "API failed" if resp_proxy['status'] != 'SUCCESS' else "Empty/Blank response"
                print(f"     => Proxy solve failed ({error_reason}). Re-using Branch A baseline results.")
                local_trace.append({
                    "step": "layer2", 
                    "sub_step": f"{branch_name}_fallback_solve2", 
                    "note": f"Recycled Branch A results due to proxy solve failure: {error_reason}"
                })
                return score_a, attempts_a, resp_gen['text'], "FALLBACK_PROXY_SOLVE_FAILED", proxy_q
            # ==================================================
            
            # Augmented Solve
            solved_proxy_combined = f"Question: {proxy_q}\nRationale and Answer: {proxy_solution_text}"
            print(f"  -> [{branch_name}] Augmented Main Solve (N={n_attempts})...")
            prompt_aug = create_core_simp_augmented_solver_prompt(t_q, solved_proxy_combined)
            
            score, attempts = _solve_and_evaluate(
                t_q, t_gt, prompt_aug, api_manager_solve, api_manager_eval, config,
                n_attempts, temp_solve, f"{branch_name}_aug_solve", local_trace, target_correct_to_beat=-1
            )
            return score, attempts, resp_gen['text'], "SUCCESS", proxy_q

    # -------------------------------------------------------------------------
    # BRANCH B: Zero-Shot Simplification
    # -------------------------------------------------------------------------
    prompt_b = create_core_simp_zero_shot_prompt(t_q)
    score_b, attempts_b, trace_b, status_b, proxy_q_b = _execute_simplification_branch("Branch B (Zero-Shot)", prompt_b)
    
    results["branch_b_score"] = score_b
    results["branch_b_status"] = status_b
    results["branch_b_trace_text"] = trace_b
    results["branch_b_attempts"] = attempts_b
    print(f"     => Score B: {score_b:.2f} ({status_b})")

    # -------------------------------------------------------------------------
    # BRANCH C: Few-Shot Analogical Simplification (Core Innovation)
    # -------------------------------------------------------------------------
    prompt_c = create_core_simp_few_shot_prompt(t_q, d_trace)
    
    # Pass Branch B's data so Branch C can check for convergence
    match_data_b = (score_b, attempts_b)
    score_c, attempts_c, trace_c, status_c, proxy_q_c = _execute_simplification_branch(
        "Branch C (Few-Shot Analogical)", prompt_c, match_proxy=proxy_q_b, match_results=match_data_b
    )
    
    results["branch_c_score"] = score_c
    results["branch_c_status"] = status_c
    results["branch_c_trace_text"] = trace_c
    results["branch_c_attempts"] = attempts_c
    print(f"     => Score C: {score_c:.2f} ({status_c})")
    
    # Store all traces for debugging
    results["execution_trace"] = local_trace
    
    return results


def execute_core_simplification_phase2(
    hard_questions: List[str],
    hard_solutions: List[str],
    exemplar_data: Dict[str, Any],
    embedding_model: SentenceTransformer,
    api_managers: Dict[str, Any],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Main Orchestration Loop for Phase 2. Handles State, Resuming, and Saving.
    """
    results_filename = config.get("CORE_SIMP_LAYER2_RESULTS_NAME", "core_simp_layer2_results.json")
    results_path = os.path.join(config['RESULTS_DIR'], results_filename)
    
    solver_mgr = api_managers.get(config.get("API_PROVIDER_SOLVER", "gemini"))
    eval_mgr = api_managers.get(config.get("API_PROVIDER_EVALUATOR", "gemini"))
    
    # 1. Load Existing Results (Checkpointing)
    all_results = load_json(results_path) or []
    completed_test_indices = {res['test_idx'] for res in all_results if 'test_idx' in res}
    
    if completed_test_indices:
        logger.info(f"Phase 2 Resuming: Found {len(completed_test_indices)} previously evaluated test questions.")

    # 2. Build the strictly paired test suite
    test_suite = build_paired_test_suite(hard_questions, hard_solutions, embedding_model, config)
    
    if not test_suite:
        return all_results
        
    # 3. Execution Loop
    for loop_idx, test_item in enumerate(tqdm(test_suite, desc="Phase 2 Execution")):
        if test_item['test_idx'] in completed_test_indices:
            continue
            
        # Execute the 3 Branches
        branch_results = run_parallel_evaluation_branches(test_item, solver_mgr, eval_mgr, config)
        
        # Save immediately
        all_results.append(branch_results)
        save_json(all_results, results_path)
        
        periodic_sync_check(loop_idx, config)
        
    logger.info(f"Phase 2 successfully evaluated all {len(test_suite)} test questions.")
    return all_results