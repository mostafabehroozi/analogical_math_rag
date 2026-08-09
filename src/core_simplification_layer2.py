# src/core_simplification_layer2.py

import os
import logging
import re
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.utils import load_json, save_json, save_json_atomic, create_trace_entry
from src.hf_sync import periodic_sync_check, periodic_batch_sync_check
from src.batching import BatchCoordinator, QuestionWorkItem, QuestionResult
from src.api_manager import GeminiAPIManager, AvalAIAPIManager, OllamaAPIManager

# Import shared prompts
from src.prompts import (
    create_final_reasoning_prompt_simple,
    create_core_simp_zero_shot_prompt,
    create_core_simp_few_shot_prompt,
    create_core_simp_augmented_solver_prompt,
    create_core_simp_few_shot_short_prompt
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
                    "linked_donor_trace_text": demo_text,  # We inject the clean demo text here!
                    
                    "linked_donor_original_q": donor.get('original_question'),
                    "linked_donor_ground_truth": donor.get('ground_truth'),
                    "linked_donor_base_score": donor.get('base_score', 0.0)
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
    
    d_q = test_item.get("linked_donor_original_q", "")
    d_gt = test_item.get("linked_donor_ground_truth", "")
    d_base_score = test_item.get("linked_donor_base_score", 0.0)
    
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
    
    print("\n" + "="*70)
    print(f"  [LAYER 2] Evaluating Test Q#{test_item['test_idx']} (Linked to Donor #{test_item['linked_donor_original_idx']})")
    print("="*70)

    # -------------------------------------------------------------------------
    # BRANCH A: Baseline (Direct Solve)
    # -------------------------------------------------------------------------
    if config.get("CORE_SIMP_RUN_BRANCH_A", True):
        print(f"  -> [Branch A] Baseline Direct Solve (N={n_attempts})...")
        prompt_a = create_final_reasoning_prompt_simple(t_q, config)
        score_a, attempts_a = _solve_and_evaluate(
            t_q, t_gt, prompt_a, api_manager_solve, api_manager_eval, config,
            n_attempts, temp_solve, "branch_a_baseline", local_trace, target_correct_to_beat=-1
        )
        results["branch_a_score"] = score_a
        results["branch_a_attempts"] = attempts_a
        print(f"     => Score A: {score_a:.2f}")
    else:
        print("  -> [Branch A] SKIPPED via config.")
        score_a, attempts_a = 0.0, []  # Safe defaults in case fallback is needed
        results["branch_a_score"] = None
        results["branch_a_status"] = "SKIPPED"

    # -------------------------------------------------------------------------
    # Helper Function for Branches B, C, and D (Fully Optimized)
    # -------------------------------------------------------------------------
    def _execute_simplification_branch(branch_name: str, gen_prompt: str, match_proxy: str = None, match_results: tuple = None) -> tuple:
        print(f"  -> [{branch_name}] Generating Proxy...")
        
        # 1. Generate Proxy
        resp_gen = api_manager_solve.generate_content(gen_prompt, m_gen, temp_gen)
        local_trace.append(create_trace_entry("layer2", f"{branch_name}_generate", {"prompt": gen_prompt}, resp_gen, {"model": m_gen}))
        
        is_fallback = False
        proxy_q = ""
        fallback_reason = "FALLBACK_TRIGGERED"  # Default reason
        
        if resp_gen['status'] != 'SUCCESS':
            is_fallback = True
            fallback_reason = "API_FAILED"
            print("     => Proxy generation API failed. Triggering Fallback.")
        else:
            parsed = _parse_simplification_trace(resp_gen['text'])
            proxy_q = parsed.get("proxy_question", "")
            
            def clean_text(t): return re.sub(r'\W+', '', t.lower())
            
            # Failsafe Checks
            if not proxy_q or clean_text(proxy_q) == clean_text(t_q):
                is_fallback = True
                fallback_reason = "FAILSAFE_IDENTICAL_OR_EMPTY"
                print("     => Failsafe triggered or parsing failed. Triggering Fallback.")
                
            # === CONVERGENCE OPTIMIZATION ===
            elif match_proxy and clean_text(proxy_q) == clean_text(match_proxy):
                print("     => [CONVERGENCE] Branch generated the exact same proxy as a previous branch!")
                print("     => Recycling previous solve and evaluation results to save API costs.")
                local_trace.append({
                    "step": "layer2", 
                    "sub_step": f"{branch_name}_convergence", 
                    "note": "Recycled previous branch results because generated proxy was identical."
                })
                # match_results contains (score_b, attempts_b)
                return match_results[0], match_results[1], resp_gen['text'], "CONVERGED", proxy_q
            # =====================================

            # === NEW: BIDIRECTIONAL MIRROR FILTER (Steps 4 & 5) ===
            elif config.get("CORE_SIMP_ENABLE_MIRROR_FILTER", False) and "Few-Shot" in branch_name:
                print("     => [Mirror Filter] Verifying analogical symmetry (Test B -> Donor A)...")
                
                # A. Inverted Demo Creation
                inverted_demo = f"Original Question:\n{t_q}\n\nSimplified Question:\n{proxy_q}"
                
                # B. Mirror Generation (A_simp_mirrored)
                prompt_mirror_gen = create_core_simp_few_shot_prompt(d_q, inverted_demo, config)
                resp_mirror_gen = api_manager_solve.generate_content(prompt_mirror_gen, m_gen, temp_gen)
                local_trace.append(create_trace_entry("layer2", f"{branch_name}_mirror_gen", {"prompt": prompt_mirror_gen}, resp_mirror_gen, {"model": m_gen}))
                
                if resp_mirror_gen['status'] != 'SUCCESS':
                    print("        -> Mirror generation failed. Rejecting.")
                    is_fallback = True
                    fallback_reason = "REJECTED_MIRROR_GEN_FAIL"
                else:
                    parsed_mirror = _parse_simplification_trace(resp_mirror_gen['text'])
                    proxy_a = parsed_mirror.get("proxy_question", "")
                    
                    # C. Mirror Solve
                    prompt_mirror_solve = create_final_reasoning_prompt_simple(proxy_a, config)
                    resp_mirror_solve = api_manager_solve.generate_content(prompt_mirror_solve, m_solve, temp_solve)
                    local_trace.append(create_trace_entry("layer2", f"{branch_name}_mirror_solve", {"prompt": prompt_mirror_solve}, resp_mirror_solve, {"model": m_solve}))
                    
                    if resp_mirror_solve['status'] != 'SUCCESS' or not resp_mirror_solve['text'].strip():
                        print("        -> Mirror solve failed. Rejecting.")
                        is_fallback = True
                        fallback_reason = "REJECTED_MIRROR_SOLVE_FAIL"
                    else:
                        # D. Validation (Calculate CCS_mirrored)
                        mirrored_combined = f"Question: {proxy_a}\nRationale and Answer: {resp_mirror_solve['text']}"
                        prompt_mirror_aug = create_core_simp_augmented_solver_prompt(d_q, mirrored_combined)
                        
                        mirror_n = config.get("CORE_SIMP_MIRROR_N_ATTEMPTS", 3)
                        # Early-stopping optimization: mathematically must beat baseline
                        target_correct = int(d_base_score * mirror_n) 
                        
                        ccs_mirrored, _ = _solve_and_evaluate(
                            d_q, d_gt, prompt_mirror_aug, api_manager_solve, api_manager_eval, config,
                            mirror_n, temp_solve, f"{branch_name}_mirror_eval", local_trace, target_correct_to_beat=target_correct
                        )
                        
                        # E. The Filter Gate
                        print(f"        -> CCS_mirrored(A) = {ccs_mirrored:.2f} | CCS_base(A) = {d_base_score:.2f}")
                        if ccs_mirrored <= d_base_score:
                            print("     => [Mirror Filter] REJECTED. Symmetry broken (CCS_m <= CCS_b). Falling back to Branch A.")
                            local_trace.append({"step": "layer2", "sub_step": f"{branch_name}_mirror_filter", "note": f"Rejected: {ccs_mirrored:.2f} <= {d_base_score:.2f}"})
                            is_fallback = True
                            fallback_reason = "REJECTED_BY_MIRROR_FILTER"
                        else:
                            print("     => [Mirror Filter] PASSED! Analogical symmetry confirmed.")
                            local_trace.append({"step": "layer2", "sub_step": f"{branch_name}_mirror_filter", "note": f"Passed: {ccs_mirrored:.2f} > {d_base_score:.2f}"})
            # ========================================================

        # 2. Execute solving logic
        if is_fallback:
            if attempts_a:
                # OPTIMIZATION: Branch A ran, recycle its results
                print("     => Re-using Branch A baseline results to save API costs.")
                local_trace.append({"step": "layer2", "sub_step": f"{branch_name}_fallback_solve", "note": f"Recycled Branch A results due to: {fallback_reason}"})
                return score_a, attempts_a, resp_gen.get('text', 'API_FAILED'), fallback_reason, proxy_q
            else:
                # SAFEGUARD: Branch A was OFF. We must perform a fresh baseline solve.
                print("     => Branch A was OFF. Performing a fresh direct solve for fallback...")
                prompt_fallback = create_final_reasoning_prompt_simple(t_q, config)
                fb_score, fb_attempts = _solve_and_evaluate(
                    t_q, t_gt, prompt_fallback, api_manager_solve, api_manager_eval, config,
                    n_attempts, temp_solve, f"{branch_name}_fallback_solve_fresh", local_trace, target_correct_to_beat=-1
                )
                return fb_score, fb_attempts, resp_gen.get('text', 'API_FAILED'), fallback_reason, proxy_q
        else:
            # Solve the Proxy
            print(f"  -> [{branch_name}] Solving Proxy...")
            prompt_proxy_solve = create_final_reasoning_prompt_simple(proxy_q, config)
            resp_proxy = api_manager_solve.generate_content(prompt_proxy_solve, m_solve, temp_solve)
            local_trace.append(create_trace_entry("layer2", f"{branch_name}_solve_proxy", {"prompt": prompt_proxy_solve}, resp_proxy, {"model": m_solve}))
            
            # === EMPTY RESPONSE FAST-FAIL OPTIMIZATION ===
            proxy_solution_text = resp_proxy.get('text', '').strip()
            
            if resp_proxy['status'] != 'SUCCESS' or not proxy_solution_text:
                error_reason = "API failed" if resp_proxy['status'] != 'SUCCESS' else "Empty/Blank response"
                
                if attempts_a:
                    print(f"     => Proxy solve failed ({error_reason}). Re-using Branch A baseline results.")
                    local_trace.append({"step": "layer2", "sub_step": f"{branch_name}_fallback_solve2", "note": f"Recycled Branch A results due to: {error_reason}"})
                    return score_a, attempts_a, resp_gen['text'], "FALLBACK_PROXY_SOLVE_FAILED", proxy_q
                else:
                    print(f"     => Proxy solve failed ({error_reason}). Branch A was OFF, performing fresh direct solve...")
                    prompt_fallback = create_final_reasoning_prompt_simple(t_q, config)
                    fb_score, fb_attempts = _solve_and_evaluate(
                        t_q, t_gt, prompt_fallback, api_manager_solve, api_manager_eval, config,
                        n_attempts, temp_solve, f"{branch_name}_fallback_solve2_fresh", local_trace, target_correct_to_beat=-1
                    )
                    return fb_score, fb_attempts, resp_gen['text'], "FALLBACK_PROXY_SOLVE_FAILED", proxy_q
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
    if config.get("CORE_SIMP_RUN_BRANCH_B", True):
        prompt_b = create_core_simp_zero_shot_prompt(t_q)
        score_b, attempts_b, trace_b, status_b, proxy_q_b = _execute_simplification_branch("Branch B (Zero-Shot)", prompt_b)
        results["branch_b_score"] = score_b
        results["branch_b_status"] = status_b
        results["branch_b_trace_text"] = trace_b
        results["branch_b_attempts"] = attempts_b
        print(f"     => Score B: {score_b:.2f} ({status_b})")
    else:
        print("  -> [Branch B] SKIPPED via config.")
        score_b, attempts_b, proxy_q_b = 0.0, [], None # Safe defaults
        results["branch_b_score"] = None
        results["branch_b_status"] = "SKIPPED"

    # -------------------------------------------------------------------------
    # BRANCH C: Few-Shot Analogical Simplification (Complex Prompt)
    # -------------------------------------------------------------------------
    if config.get("CORE_SIMP_RUN_BRANCH_C", True):
        prompt_c = create_core_simp_few_shot_prompt(t_q, d_trace, config)
        match_data_b = (score_b, attempts_b)
        score_c, attempts_c, trace_c, status_c, proxy_q_c = _execute_simplification_branch(
            "Branch C (Few-Shot Complex)", prompt_c, match_proxy=proxy_q_b, match_results=match_data_b
        )
        results["branch_c_score"] = score_c
        results["branch_c_status"] = status_c
        results["branch_c_trace_text"] = trace_c
        results["branch_c_attempts"] = attempts_c
        print(f"     => Score C: {score_c:.2f} ({status_c})")
    else:
        print("  -> [Branch C] SKIPPED via config.")
        results["branch_c_score"] = None
        results["branch_c_status"] = "SKIPPED"

    # -------------------------------------------------------------------------
    # BRANCH D: Few-Shot Analogical Simplification (Concise Prompt)
    # -------------------------------------------------------------------------
    if config.get("CORE_SIMP_RUN_BRANCH_D", True):
        prompt_d = create_core_simp_few_shot_short_prompt(t_q, d_trace, config)
        # We also pass match_proxy so it saves API calls if it outputs the same proxy as Branch B
        match_data_b = (score_b, attempts_b) 
        score_d, attempts_d, trace_d, status_d, proxy_q_d = _execute_simplification_branch(
            "Branch D (Few-Shot Concise)", prompt_d, match_proxy=proxy_q_b, match_results=match_data_b
        )
        results["branch_d_score"] = score_d
        results["branch_d_status"] = status_d
        results["branch_d_trace_text"] = trace_d
        results["branch_d_attempts"] = attempts_d
        print(f"     => Score D: {score_d:.2f} ({status_d})")
    else:
        print("  -> [Branch D] SKIPPED via config.")
        results["branch_d_score"] = None
        results["branch_d_status"] = "SKIPPED"
    
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
        
    tests_to_process = [item for item in test_suite if item['test_idx'] not in completed_test_indices]
    
    if not tests_to_process:
        logger.info("All test items for Phase 2 are already processed. Skipping.")
        return all_results

    if config.get("BATCH_PROCESSING_ENABLED", False):
        results_by_index = {result.get("test_idx"): result for result in all_results if "test_idx" in result}
        tests_by_index = {item["test_idx"]: item for item in tests_to_process}
        def worker(item: QuestionWorkItem) -> Dict[str, Any]:
            return run_parallel_evaluation_branches(tests_by_index[item.index], solver_mgr, eval_mgr, config)
        def commit(results: List[QuestionResult], _batch_id: str, batch_number: int) -> None:
            for result in results:
                results_by_index[result.item.index] = result.value
            committed = [results_by_index[index] for index in sorted(results_by_index)]
            if not save_json_atomic(committed, results_path):
                raise RuntimeError("Failed to commit core simplification Phase 2 results")
            periodic_batch_sync_check(batch_number - 1, config)
        BatchCoordinator(config, config.get("experiment_name", "core_simp_phase2"), "core_simp_phase2").run(
            [QuestionWorkItem(index=item["test_idx"], question=item.get("test_question", "")) for item in tests_to_process], worker, commit
        )
        return [results_by_index[index] for index in sorted(results_by_index)]

    # 3. Execution Loop
    for loop_idx, test_item in enumerate(tqdm(tests_to_process, desc="Phase 2 Execution")):
        # Execute the 3 Branches
        branch_results = run_parallel_evaluation_branches(test_item, solver_mgr, eval_mgr, config)
        
        # Save immediately
        all_results.append(branch_results)
        save_json(all_results, results_path)
        
        periodic_sync_check(loop_idx, config)
        
    logger.info(f"Phase 2 successfully evaluated all {len(test_suite)} test questions.")
    return all_results
