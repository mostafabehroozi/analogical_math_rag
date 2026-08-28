#src/pipeline_steps.py

import logging
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union, Optional
import time
import threading

from src.prompts import (
    PROMPT_TEMPLATES,
    EXEMPLAR_FORMAT,
    create_normalization_prompt,
    create_transformation_prompt,
    create_final_reasoning_prompt,
    create_final_reasoning_prompt_simple,
    is_prompt_construction_error,
    create_duplicate_check_prompt,
    create_best_of_transformation_solver_prompt,
    create_reverse_validation_candidate_prompt,
    create_reverse_validation_prompt,
    create_simplification_prompt,
    create_simplified_sample_solver_prompt,
    create_main_from_simplified_proxy_prompt,
    create_reverse_transformation_main_to_exemplar_prompt,
    create_reverse_transformation_solve_transformed_prompt,
    create_reverse_transformation_final_solve_prompt,
    create_mirror_hypothesis_zeroshot_prompt,
)
from src.utils import create_trace_entry
from src.parallel_utils import run_parallel_api_calls
from src.api_manager import GeminiAPIManager, AvalAIAPIManager, OllamaAPIManager
from src.evaluation import evaluate_single_answer_with_llm
# Global cache for embedding norms to massively speed up retrieval and save RAM
_NORMS_CACHE = {}
# Protects local PyTorch and Numpy resources from being overwhelmed by batch threads
_LOCAL_COMPUTE_LOCK = threading.Lock()

from src.context_logger import tprint
import builtins

def silent_context_print(*args, **kwargs):
    """
    Secretly intercepts all print() calls in this file safely.
    Instead of spamming the console, it sends them to the log file.
    """
    if 'file' in kwargs or kwargs.get('end', '\n') != '\n':
        builtins.print(*args, **kwargs)
        return
    sep = kwargs.get('sep', ' ')
    message = sep.join(str(a) for a in args)
    tprint(message, level="DEBUG")

print = silent_context_print

def log_time_diagnostic(message: str, start_time: float, indent: int = 0) -> float:
    end_time = time.time()
    elapsed = end_time - start_time
    indent_str = "  " * indent
    if elapsed > 0.001: 
        print(f"{indent_str}⏱️  DIAGNOSTIC: {message} took {elapsed:.4f} seconds.")
    return end_time

def _generate_embeddings(
    texts: List[str],
    embedding_model: SentenceTransformer,
    batch_size: int = 32
) -> np.ndarray:
    if not isinstance(embedding_model, SentenceTransformer) or not texts:
        return np.array([])
    try:
        with _LOCAL_COMPUTE_LOCK: 
            return embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to generate embeddings: {e}", exc_info=True)
        return np.array([])

def retrieve(
    target_query: str,
    embedding_model: SentenceTransformer,
    exemplar_questions: List[str],
    embedded_exemplars: np.ndarray,
    top_k: int,
    question_to_index_map: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info(f"Starting retrieval for Top-{top_k} exemplars.")
    
    local_trace = []
    
    indent_level = 3
    
    print(f"{'  '*indent_level}--- STARTING DETAILED RETRIEVAL DIAGNOSTICS ---")
    print(f"{'  '*indent_level}Target Query (start): '{target_query[:100]}...'")
    print(f"{'  '*indent_level}Exemplar corpus size: {len(exemplar_questions)}")
    print(f"{'  '*indent_level}Embedded exemplars shape: {embedded_exemplars.shape}")
    print(f"{'  '*indent_level}Requested top_k: {top_k}")

    query_embedding_start_time = time.time()
    query_embedding = _generate_embeddings([target_query], embedding_model)
    log_time_diagnostic("Generate query embedding", query_embedding_start_time, indent=indent_level)
    print(f"{'  '*indent_level}Query embedding shape: {query_embedding.shape}")
    
    if query_embedding.size == 0:
        logger.error("Failed to generate embedding for the target query. Retrieval cannot proceed.")
        local_trace.append(create_trace_entry("retrieve", "embedding_generation", {"target": target_query}, {"error": "Failed to generate embedding"}, error_info={"msg": "Empty embedding"}))
        return {"status": "FAILURE", "retrieved_indices": [], "retrieved_exemplars": [], "trace": local_trace}
    
    cosine_similarity_start_time = time.time()
    print(f"{'  '*indent_level}Starting cosine similarity calculation (query_embedding shape: {query_embedding.shape}, embedded_exemplars shape: {embedded_exemplars.shape})...")

    # LOCKED BLOCK STARTS HERE
    with _LOCAL_COMPUTE_LOCK:
        q_vec = np.asarray(query_embedding[0], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        
        if q_norm < 1e-10:
            similarities = np.zeros(embedded_exemplars.shape[0], dtype=np.float32)
        else:
            q_vec_normalized = q_vec / q_norm
            # Fast dot product (much more memory efficient than sklearn)
            dot_prods = np.dot(embedded_exemplars, q_vec_normalized)
            
            # Cache the norms of the huge embedded_exemplars matrix
            arr_id = id(embedded_exemplars)
            if arr_id not in _NORMS_CACHE:
                logger.info("Computing and caching L2 norms for embedded_exemplars...")
                norms = np.linalg.norm(embedded_exemplars, axis=1)
                _NORMS_CACHE[arr_id] = np.maximum(norms, 1e-10).astype(np.float32)
                
            similarities = dot_prods / _NORMS_CACHE[arr_id]
        log_time_diagnostic("Calculate cosine_similarity", cosine_similarity_start_time, indent=indent_level)
        print(f"{'  '*indent_level}Similarities array shape: {similarities.shape}")
        
        self_match_start_time = time.time()

        if question_to_index_map is not None:
            query_indices_in_corpus = question_to_index_map.get(target_query)
            if query_indices_in_corpus is not None:
                # Legacy callers may supply a single index. New corpus maps
                # preserve every duplicate question so none can be retrieved.
                if isinstance(query_indices_in_corpus, (list, tuple, set, np.ndarray)):
                    indices_to_exclude = list(query_indices_in_corpus)
                else:
                    indices_to_exclude = [query_indices_in_corpus]
                similarities[indices_to_exclude] = -np.inf
                print(f"{'  '*indent_level}Self-match found at indices {indices_to_exclude}, set to -np.inf.")
            else:
                print(f"{'  '*indent_level}Target query not found in corpus (no self-match to remove).")
        else:
            print(f"{'  '*indent_level}Warning: question_to_index_map not provided. Skipping self-match check.")
            logger.warning("retrieve() called without question_to_index_map. Self-match detection skipped.")

        log_time_diagnostic("Handle self-match (O(1) lookup)", self_match_start_time, indent=indent_level)

        k_retrieve_start_time = time.time()
        eligible_count = int(np.isfinite(similarities).sum())
        if eligible_count == 0:
            logger.error("No eligible exemplars remain after exact-question exclusion.")
            return {
                "status": "FAILURE", "retrieved_indices": [], "retrieved_exemplars": [],
                "trace": local_trace,
                "error": "No eligible exemplars remain after exact-question exclusion.",
            }
        k_to_retrieve = min(top_k, eligible_count)
        log_time_diagnostic("Determine k_to_retrieve", k_retrieve_start_time, indent=indent_level)
        print(f"{'  '*indent_level}Effective k_to_retrieve: {k_to_retrieve}")

        print(f"{'  '*indent_level}Starting granular timing for top-k selection...")
            
        argpartition_start_time = time.time()
        print(f"{'  '*indent_level}  Calling np.argpartition on similarities array (shape: {similarities.shape}) for k={k_to_retrieve}...")
        
        partitioned_indices = np.argpartition(similarities, -k_to_retrieve)
        log_time_diagnostic("np.argpartition (full array)", argpartition_start_time, indent=indent_level)
        print(f"{'  '*indent_level}  Resulting partitioned_indices shape: {partitioned_indices.shape}")

        slice_partitioned_start_time = time.time()
        print(f"{'  '*indent_level}  Slicing to get the top {k_to_retrieve} indices from partitioned_indices...")
        
        top_k_indices_unsorted = partitioned_indices[-k_to_retrieve:]
        log_time_diagnostic("Slicing partitioned indices", slice_partitioned_start_time, indent=indent_level)
        print(f"{'  '*indent_level}  top_k_indices_unsorted shape: {top_k_indices_unsorted.shape}, Content (first 5): {top_k_indices_unsorted[:5]}...")

        argsort_slice_start_time = time.time()
        print(f"{'  '*indent_level}  Calling np.argsort on the {k_to_retrieve} selected indices based on their similarities...")
        print(f"{'  '*indent_level}  Accessing similarities values for sorting (similarities[top_k_indices_unsorted])...")

        relevant_similarities = similarities[top_k_indices_unsorted]
        
        sorted_order_in_slice = np.argsort(relevant_similarities)[::-1] 
        
        top_k_indices = top_k_indices_unsorted[sorted_order_in_slice]
    # --- LOCKED BLOCK ENDS HERE ---

    log_time_diagnostic("np.argsort & final sort (on small slice)", argsort_slice_start_time, indent=indent_level)
    print(f"{'  '*indent_level}  Final top_k_indices shape: {top_k_indices.shape}, Content: {top_k_indices.tolist()}")
    print(f"{'  '*indent_level}--- ENDING DETAILED RETRIEVAL DIAGNOSTICS ---")

    logger.info(f"Successfully retrieved indices: {top_k_indices.tolist()}")

    # Log the retrieval result as a trace event
    local_trace.append(create_trace_entry(
        step_name="retrieve",
        sub_step="cosine_similarity_search",
        input_context={"target_query": target_query, "top_k": top_k},
        output_result={"retrieved_indices": top_k_indices.tolist()}
    ))
    
    return {
        "status": "SUCCESS",
        "retrieved_indices": top_k_indices.tolist(),
        "retrieved_similarity_scores": relevant_similarities.tolist(),
        "trace": local_trace
    }


def _calculate_baseline_difficulty(
    retrieved_indices: List[int],
    exemplar_data: Dict[str, Any],
    api_manager_solve: Any,
    api_manager_eval: Any,
    config: Dict[str, Any],
    trace_accumulator: List[Dict]
) -> Dict[int, float]:
    """
    Phase 0: Calculates S_base for each retrieved sample (Zero-Shot).
    Returns a dict mapping {index: baseline_score}.
    """
    n_mirror = config.get("MIRROR_N_OPTIMIZATION", 3)
    
    # Determine Model Name
    # FIX: Changed 'api_manager' to 'api_manager_solve'
    if isinstance(api_manager_solve, GeminiAPIManager): model_name = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager_solve, AvalAIAPIManager): model_name = config['AVALAI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager_solve, OllamaAPIManager): model_name = config['OLLAMA_MODEL_NAME_FINAL_SOLVER']
    
    # Load Template
    template_name = config.get("PROMPT_TEMPLATE_MIRROR_BASELINE", "mirror_baseline_zero_shot_v1")
    base_template = PROMPT_TEMPLATES.get(template_name, "Problem: {question}\nSolve this step-by-step.\nFinal Answer:")
    
    baseline_scores = {}
    print(f"    -> [Mirror Phase 0] Calculating Baseline Difficulty for {len(retrieved_indices)} samples...")

    # --- NEW PARALLEL LOGIC ---
    def _baseline_task(idx, attempt_idx):
        """Task wrapper to run in parallel."""
        question = exemplar_data['questions'][idx]
        ground_truth = exemplar_data['solutions'][idx]
        
        prompt = base_template.format(question=question)
        
        resp = api_manager_solve.generate_content(
            prompt, model_name, temperature=1.0, avalai_role="final_solver"
        )
        
        trace_entry = create_trace_entry(
            "mirror_phase_0", f"baseline_idx_{idx}_attempt_{attempt_idx}",
            {"question": question, "ground_truth": ground_truth}, 
            resp, {"model": model_name}
        )
        
        is_correct = False
        if resp['status'] == 'SUCCESS':
            eval_res = evaluate_single_answer_with_llm(
                resp['text'], ground_truth, api_manager_eval, config
            )
            if eval_res['is_correct']:
                is_correct = True
                
        return idx, trace_entry, is_correct

    # Flatten the nested loops into a 1D list of tasks
    tasks = [
        lambda idx=idx, i=i: _baseline_task(idx, i) 
        for idx in retrieved_indices 
        for i in range(n_mirror)
    ]
    
    # Execute in parallel or sequential based on config
    results = run_parallel_api_calls(tasks, config)
    
    # Tally up the results safely in the main thread
    correct_counts = {idx: 0 for idx in retrieved_indices}
    for idx, trace_entry, is_correct in results:
        trace_accumulator.append(trace_entry)
        if is_correct:
            correct_counts[idx] += 1
            
    for idx in retrieved_indices:
        baseline_scores[idx] = correct_counts[idx] / n_mirror

    return baseline_scores

def _generate_hypotheses(
    target_query: str,
    candidate_indices: List[int], # Includes -1 for R0 if enabled
    exemplar_data: Dict[str, Any],
    api_manager: Any,
    config: Dict[str, Any],
    trace_accumulator: List[Dict]
) -> Dict[int, str]:
    """
    Phase 1: Generates a Hypothesis (H) for the target query using each candidate.
    Returns a dict mapping {candidate_index: hypothesis_text}.
    """
    if isinstance(api_manager, GeminiAPIManager): model_name = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, AvalAIAPIManager): model_name = config['AVALAI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, OllamaAPIManager): model_name = config['OLLAMA_MODEL_NAME_FINAL_SOLVER']
    
    hypotheses = {}
    
    # Load templates
    tmpl_zero = PROMPT_TEMPLATES.get("mirror_hypothesis_gen_zero_shot_v1", "{target_query}")
    tmpl_few = PROMPT_TEMPLATES.get("mirror_hypothesis_gen_v1", "{target_query}")

    print(f"    -> [Mirror Phase 1] Generating Hypotheses for {len(candidate_indices)} candidates...")

    # PARALLEL LOGIC 
    def _hypothesis_task(cand_idx):
        """Task wrapper to run in parallel."""
        if cand_idx == -1:
            # R0: Zero-Shot
            prompt = None
            try:
                prompt = create_mirror_hypothesis_zeroshot_prompt(target_query, config)
            except Exception:
                pass
                
            if prompt is None:
                try:
                    prompt = tmpl_zero.format(main_question_text=target_query)
                except Exception:
                    try:
                        prompt = tmpl_zero.format(target_query=target_query)
                    except Exception:
                        try:
                            prompt = tmpl_zero.format(question=target_query)
                        except Exception:
                            prompt = target_query
        else:
            # R_cand: Few-Shot
            q_ex = exemplar_data['questions'][cand_idx]
            s_ex = exemplar_data['solutions'][cand_idx]
            prompt = tmpl_few.format(
                exemplar_question=q_ex,
                exemplar_solution=s_ex,
                target_query=target_query
            )

        # Generate Hypothesis (Single attempt, Temp=0.0)
        resp = api_manager.generate_content(
            prompt, model_name, temperature=0.0, avalai_role="final_solver"
        )
        
        # Return data instead of appending to shared list
        trace_entry = create_trace_entry(
            "mirror_phase_1", f"hypothesis_cand_{cand_idx}",
            {"prompt": prompt}, resp, {"model": model_name}
        )
        return cand_idx, resp, trace_entry

    # Create list of tasks (using default args to prevent lambda late-binding bugs)
    tasks = [lambda c=cand_idx: _hypothesis_task(c) for cand_idx in candidate_indices]
    
    # Execute in parallel or sequential based on config
    results = run_parallel_api_calls(tasks, config)
    
    # Safely process results in the main thread
    for cand_idx, resp, trace_entry in results:
        trace_accumulator.append(trace_entry)
        if resp['status'] == 'SUCCESS':
            hypotheses[cand_idx] = resp['text']
        else:
            hypotheses[cand_idx] = ""

    return hypotheses

def _evaluate_mirror_consistency(
    target_query: str,
    hypotheses: Dict[int, str],
    validation_indices: List[int],
    exemplar_data: Dict[str, Any],
    api_manager_solve: Any,
    api_manager_eval: Any,
    config: Dict[str, Any],
    trace_accumulator: List[Dict]
) -> Dict[int, Dict[int, float]]:
    """
    Phase 2: Mirror Evaluation (Backward Pass).
    Tests if the Hypothesis (H) can solve the Validation Samples (R_val).
    """
    n_mirror = config.get("MIRROR_N_OPTIMIZATION", 3)
    
    # Determine Model Name
    # FIX: Changed 'api_manager' to 'api_manager_solve'
    if isinstance(api_manager_solve, GeminiAPIManager): model_name = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager_solve, AvalAIAPIManager): model_name = config['AVALAI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager_solve, OllamaAPIManager): model_name = config['OLLAMA_MODEL_NAME_FINAL_SOLVER']
    
    # Load Template
    tmpl_verify = PROMPT_TEMPLATES.get(
        config.get("PROMPT_TEMPLATE_MIRROR_VERIFICATION", "mirror_verification_v1"), 
        "Example Q:\n{hypothesis_question}\nExample A:\n{hypothesis_solution}\n\nProblem:\n{validation_question}\n\nFinal Answer:"
    )

    mirror_results = {}
    print(f"    -> [Mirror Phase 2] Running Consistency Check ({len(hypotheses)} Candidates x {len(validation_indices)} Validation Samples)...")

    # --- NEW PARALLEL LOGIC ---
    def _mirror_task(cand_id, val_idx, attempt_idx, formatted_hyp, val_q, val_gt):
        """Task wrapper to run in parallel."""
        prompt = tmpl_verify.format(
            hypothesis_question=target_query, 
            hypothesis_solution=formatted_hyp,
            validation_question=val_q
        )
        
        resp = api_manager_solve.generate_content(
            prompt, model_name, temperature=1.0, avalai_role="final_solver"
        )
        
        trace_entry = create_trace_entry(
            "mirror_phase_2", f"cand_{cand_id}_vs_val_{val_idx}_run_{attempt_idx}",
            {"prompt": prompt}, resp, {"model": model_name}
        )
        
        is_correct = False
        if resp['status'] == 'SUCCESS':
            eval_res = evaluate_single_answer_with_llm(
                resp['text'], val_gt, api_manager_eval, config
            )
            if eval_res['is_correct']:
                is_correct = True
                
        return cand_id, val_idx, trace_entry, is_correct

    tasks = []
    
    # First, handle empty hypotheses (must be done sequentially to populate dict)
    for cand_id, hypothesis_text in hypotheses.items():
        if not hypothesis_text:
            mirror_results[cand_id] = {v_idx: 0.0 for v_idx in validation_indices}
            continue
            
        mirror_results[cand_id] = {}
        formatted_hyp = f"{hypothesis_text}"
        
        # Build tasks for the non-empty hypotheses
        for val_idx in validation_indices:
            val_q = exemplar_data['questions'][val_idx]
            val_gt = exemplar_data['solutions'][val_idx]
            
            for i in range(n_mirror):
                tasks.append(
                    lambda c=cand_id, v=val_idx, i=i, fh=formatted_hyp, vq=val_q, vgt=val_gt: 
                    _mirror_task(c, v, i, fh, vq, vgt)
                )
                
    # Execute all tasks in parallel or sequential based on config
    results = run_parallel_api_calls(tasks, config)
    
    # Tally up the results safely in the main thread
    correct_counts = {}
    for cand_id, val_idx, trace_entry, is_correct in results:
        trace_accumulator.append(trace_entry)
        
        if cand_id not in correct_counts:
            correct_counts[cand_id] = {}
        if val_idx not in correct_counts[cand_id]:
            correct_counts[cand_id][val_idx] = 0
            
        if is_correct:
            correct_counts[cand_id][val_idx] += 1
            
    # Calculate final float scores
    for cand_id, vals in correct_counts.items():
        for val_idx, count in vals.items():
            mirror_results[cand_id][val_idx] = count / n_mirror

    return mirror_results

def _rank_and_filter_candidates(
    candidate_indices: list,
    candidate_scores: dict,
    candidate_contributions: dict,
    config: dict,
    trace_accumulator: list
) -> tuple:
    """
    Applies Usefulness Filtering (Score > 0) and Redundancy Filtering (Pairwise Dominance).
    Returns the Redundancy-Filtered list, Base (Noise-Filtered) list, and the Master Sorted list.
    """
    enable_filtering = config.get("MIRROR_ENABLE_FILTERING", True)
    enable_redundancy = config.get("MIRROR_ENABLE_REDUNDANCY_FILTER", True)


    sorted_candidates = sorted(candidate_indices, key=lambda x: (candidate_scores[x], -x), reverse=True)

    # 2. Base Filtering: Remove Useless Candidates (Score <= 0)
    if enable_filtering:
        base_indices = [c for c in sorted_candidates if candidate_scores[c] > 0]
        # Trace log
        trace_accumulator.append(f"   > Base Filtering: Kept {len(base_indices)}/{len(sorted_candidates)} (Score > 0)")
    else:
        base_indices = sorted_candidates
        trace_accumulator.append("   > Base Filtering: DISABLED (All candidates kept)")

    # 3. Redundancy Filtering: Remove Covered Candidates
    if not enable_redundancy or not enable_filtering:

        return base_indices, base_indices, sorted_candidates

    final_selection = []
    
    # We iterate through the sorted 'useful' candidates (highest score first)
    for current_cand in base_indices:
        is_covered = False
        
        # Check against already accepted candidates (which have higher/equal scores)
        for accepted_cand in final_selection:
            # Check if 'accepted_cand' covers 'current_cand'
            # Definition: accepted covers current if for ALL validation samples j:
            # contribution(accepted, j) >= contribution(current, j)
            
            accepted_contribs = candidate_contributions.get(accepted_cand, {})
            current_contribs = candidate_contributions.get(current_cand, {})
            
            covered_completely = True
            
            # We must check against ALL validation samples that 'current_cand' helped with
            for val_idx, score_gain in current_contribs.items():
                if score_gain > 0:
                    # If the accepted candidate didn't help here, or helped less/equal...
                    if accepted_contribs.get(val_idx, 0.0) < score_gain:
                        covered_completely = False
                        break
            
            if covered_completely:
                is_covered = True
                trace_accumulator.append(f"     - Dropped Candidate {current_cand} (Covered by {accepted_cand})")
                break
        
        if not is_covered:
            final_selection.append(current_cand)

    trace_accumulator.append(f"   > Redundancy Filtering: Kept {len(final_selection)}/{len(base_indices)}")

    # RETURN THREE LISTS: [Redundancy Filtered], [Base Filtered], [Master Sorted]
    return final_selection, base_indices, sorted_candidates


def simplify_retrieved_samples(
    retrieved_indices: List[int],
    exemplar_questions: List[str],
    exemplar_solutions: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info("Starting Simplification of Retrieved Samples.")
    
    local_trace = []
    successful_simplifications = []
    failed_indices = []

    if isinstance(api_manager, GeminiAPIManager):
        model_name = config.get('GEMINI_MODEL_NAME_SIMPLIFICATION', config['GEMINI_MODEL_NAME_ADAPTATION'])
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config.get('AVALAI_MODEL_NAME_SIMPLIFICATION', config['AVALAI_MODEL_NAME_ADAPTATION'])
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config.get('OLLAMA_MODEL_NAME_SIMPLIFICATION', config['OLLAMA_MODEL_NAME_ADAPTATION'])
    else:
        raise TypeError(f"Unsupported API manager type: {type(api_manager)}")
        
    temp = config.get("DEFAULT_SIMPLIFICATION_TEMPERATURE", 0.3)

    for idx in retrieved_indices:
        original_q = exemplar_questions[idx]
        original_s = exemplar_solutions[idx]
        original_exemplar_text = EXEMPLAR_FORMAT.format(question=original_q, solution=original_s)
        
        print(f"    -> Processing Sample #{idx} for Simplification...")

        # 1. Simplify Question
        prompt_simp = create_simplification_prompt(original_q, config)
        resp_simp = api_manager.generate_content(
            prompt_simp, model_name, temp, avalai_role="adaptation"
        )
        
        local_trace.append(create_trace_entry(
            "simplify_samples", "simplify_question", 
            {"original_idx": idx, "original_q": original_q, "prompt": prompt_simp},
            resp_simp, 
            {"model": model_name, "temp": temp}
        ))
        
        if resp_simp['status'] != 'SUCCESS':
            logger.warning(f"Failed to simplify question for sample {idx}: {resp_simp.get('error_message')}")
            failed_indices.append({"index": idx, "step": "simplify_q", "error": resp_simp})
            continue
            
        simple_q = resp_simp['text'].strip()
        
        # 2. Solve Simplified Question using Original Rationale
        prompt_solve = create_simplified_sample_solver_prompt(simple_q, original_exemplar_text, config)
        resp_solve = api_manager.generate_content(
            prompt_solve, model_name, temp, avalai_role="final_solver"
        )
        
        local_trace.append(create_trace_entry(
            "simplify_samples", "solve_simplified_sample",
            {"original_idx": idx, "simple_q": simple_q, "prompt": prompt_solve},
            resp_solve,
            {"model": model_name, "temp": temp}
        ))
        
        if resp_solve['status'] != 'SUCCESS':
            logger.warning(f"Failed to solve simplified question for sample {idx}: {resp_solve.get('error_message')}")
            failed_indices.append({"index": idx, "step": "solve_simple_q", "error": resp_solve})
            continue
            
        simple_solution = resp_solve['text'].strip()
        
        new_exemplar_text = f"Question: {simple_q}\nRationale and Answer: {simple_solution}"
        successful_simplifications.append(new_exemplar_text)
        print("      -> Success. New simplified exemplar created.")

    status = "SUCCESS" if successful_simplifications else "FAILURE"
    if successful_simplifications and failed_indices: status = "PARTIAL_SUCCESS"

    return {
        "status": status,
        "simplified_exemplars": successful_simplifications,
        "failed_indices": failed_indices,
        "trace": local_trace
    }

def adapt(
    target_query: str,
    retrieved_indices: List[int],
    exemplar_questions: List[str],
    exemplar_solutions: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info("Starting multi-stage adaptation step.")
    
    local_trace = []
    successful_texts = []
    failed_adaptations = []
    
    if isinstance(api_manager, GeminiAPIManager):
        model_name = config['GEMINI_MODEL_NAME_ADAPTATION']
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config['AVALAI_MODEL_NAME_ADAPTATION']
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config['OLLAMA_MODEL_NAME_ADAPTATION']
    else:
        raise TypeError(f"Unsupported API manager type for adaptation: {type(api_manager)}")
        
    temperature = config['DEFAULT_ADAPTATION_TEMPERATURE']

    for idx in retrieved_indices:
        original_question = exemplar_questions[idx]
        original_solution = exemplar_solutions[idx]
        current_text = EXEMPLAR_FORMAT.format(question=original_question, solution=original_solution)
        
        step_failed = False

        if config.get('APPLY_NORMALIZATION', False) and not step_failed:
            logger.info(f"Applying normalization to exemplar index {idx}.")
            print(f"    -> Normalizing exemplar {idx}...")
            prompt = create_normalization_prompt(current_text)
            
            print(f"      [API Context] Calling LLM for: Normalization (Exemplar #{idx})")
            response = api_manager.generate_content(
                prompt, model_name, temperature, avalai_role="adaptation"
            )
            
            local_trace.append(create_trace_entry(
                "adapt", "normalization",
                {"source_index": idx, "prompt": prompt},
                response,
                {"model": model_name, "temp": temperature}
            ))
            
            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                logger.warning(f"Normalization failed for exemplar {idx}: {response['error_message']}")
                failed_adaptations.append({"source_index": idx, "failed_at_step": "normalization", "error_info": response})
                step_failed = True

        if config.get('APPLY_TRANSFORMATION_1', False) and not step_failed:
            logger.info(f"Applying transformation 1 to exemplar index {idx}.")
            print(f"    -> Applying Transformation 1 to exemplar {idx}...")
            prompt = create_transformation_prompt(target_query, current_text, config, "PROMPT_TEMPLATE_TRANSFORMATION_1")
            print(f"      [API Context] Calling LLM for: Transformation 1 (Exemplar #{idx})")
            response = api_manager.generate_content(
                prompt, model_name, temperature, avalai_role="adaptation"
            )

            local_trace.append(create_trace_entry(
                "adapt", "transformation_1",
                {"source_index": idx, "prompt": prompt},
                response,
                {"model": model_name, "temp": temperature}
            ))

            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                logger.warning(f"Transformation 1 failed for exemplar {idx}: {response['error_message']}")
                failed_adaptations.append({"source_index": idx, "failed_at_step": "transformation_1", "error_info": response})
                step_failed = True
        
        if config.get('APPLY_TRANSFORMATION_2', False) and not step_failed:
            logger.info(f"Applying transformation 2 to exemplar index {idx}.")
            print(f"    -> Applying Transformation 2 to exemplar {idx}...")
            prompt = create_transformation_prompt(target_query, current_text, config, "PROMPT_TEMPLATE_TRANSFORMATION_2")
            print(f"      [API Context] Calling LLM for: Transformation 2 (Exemplar #{idx})")
            response = api_manager.generate_content(
                prompt, model_name, temperature, avalai_role="adaptation"
            )

            local_trace.append(create_trace_entry(
                "adapt", "transformation_2",
                {"source_index": idx, "prompt": prompt},
                response,
                {"model": model_name, "temp": temperature}
            ))

            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                logger.warning(f"Transformation 2 failed for exemplar {idx}: {response['error_message']}")
                failed_adaptations.append({"source_index": idx, "failed_at_step": "transformation_2", "error_info": response})
                step_failed = True

        if config.get('APPLY_TRANSFORMATION_3', False) and not step_failed:
            logger.info(f"Applying transformation 3 to exemplar index {idx}.")
            print(f"    -> Applying Transformation 3 to exemplar {idx}...")
            prompt = create_transformation_prompt(target_query, current_text, config, "PROMPT_TEMPLATE_TRANSFORMATION_3")
            print(f"      [API Context] Calling LLM for: Transformation 3 (Exemplar #{idx})")
            response = api_manager.generate_content(
                prompt, model_name, temperature, avalai_role="adaptation"
            )

            local_trace.append(create_trace_entry(
                "adapt", "transformation_3",
                {"source_index": idx, "prompt": prompt},
                response,
                {"model": model_name, "temp": temperature}
            ))

            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                logger.warning(f"Transformation 3 failed for exemplar {idx}: {response['error_message']}")
                failed_adaptations.append({"source_index": idx, "failed_at_step": "transformation_3", "error_info": response})
                step_failed = True
        
        if not step_failed:
            successful_texts.append(current_text)

    if not retrieved_indices:
        final_status = "SUCCESS"
    elif not successful_texts and failed_adaptations:
        final_status = "FAILURE"
    elif successful_texts and failed_adaptations:
        final_status = "PARTIAL_SUCCESS"
    else:
        final_status = "SUCCESS"

    return {
        "status": final_status,
        "adapted_texts": successful_texts,
        "failed_adaptations": failed_adaptations,
        "trace": local_trace
    }


def solve(
    target_query: str,
    final_exemplars: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info("Starting final solver step.")
    local_trace = []
    
    final_solver_prompt_name = config.get("PROMPT_TEMPLATE_FINAL_SOLVER")
    if final_solver_prompt_name == "duplicate_question_check_v1":
        logger.info("Running in 'Duplicate Question Check' mode.")
        
        retrieved_questions = []
        for exemplar_text in final_exemplars:
            match = re.search(r"Question:\s*(.*?)\s*Rationale and Answer:", exemplar_text, re.DOTALL)
            if match:
                retrieved_questions.append(match.group(1).strip())
        
        if not retrieved_questions:
            logger.warning("Duplicate check mode ran but no retrieved questions were found to check.")
            return {"status": "SUCCESS", "solution_attempts": ["no_retrieval"], "trace": local_trace}

        prompt = create_duplicate_check_prompt(target_query, retrieved_questions)
        
        if isinstance(api_manager, GeminiAPIManager):
            model_name = config['GEMINI_MODEL_NAME_ADAPTATION']
        elif isinstance(api_manager, AvalAIAPIManager):
            model_name = config['AVALAI_MODEL_NAME_ADAPTATION']
        elif isinstance(api_manager, OllamaAPIManager):
            model_name = config['OLLAMA_MODEL_NAME_ADAPTATION']
        else:
            raise TypeError(f"Unsupported API manager type for duplicate check: {type(api_manager)}")
            
        temperature = 0.0 
        
        print("    -> Checking for duplicate questions...")
        print("      [API Context] Calling LLM for: Duplicate Check")
        response = api_manager.generate_content(
            prompt, model_name, temperature, avalai_role="adaptation"
        )

        local_trace.append(create_trace_entry(
            "solve", "duplicate_check",
            {"prompt": prompt}, response, {"model": model_name, "temp": temperature}
        ))

        if response['status'] == 'SUCCESS':
            classification = response['text'].strip().lower()
            if "yes" in classification:
                result = "yes"
            elif "no" in classification:
                result = "no"
            else:
                result = "parsing_failed"
            return {"status": "SUCCESS", "solution_attempts": [result], "trace": local_trace}
        else:
            return {"status": "FAILURE", "solution_attempts": [{"status": "FAILURE", "error_info": response}], "trace": local_trace}

    prompt = create_final_reasoning_prompt(target_query, final_exemplars, config) if final_exemplars else create_final_reasoning_prompt_simple(target_query, config)
    logger.info(f"Using {'retrieval-augmented' if final_exemplars else 'simple'} prompt for the solver.")

    if is_prompt_construction_error(prompt):
        error_msg = f"Failed to create final reasoning prompt: {prompt}"
        logger.error(error_msg)
        return {"status": "FAILURE", "solution_attempts": [{"status": "FAILURE", "error_info": {"error_message": error_msg}}], "trace": local_trace}

    n_attempts = config.get("N_PASS_ATTEMPTS", 1)
    
    if isinstance(api_manager, GeminiAPIManager):
        model_name = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config['AVALAI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config['OLLAMA_MODEL_NAME_FINAL_SOLVER']
    else:
        raise TypeError(f"Unsupported API manager type for solver: {type(api_manager)}")
        
    temperature = config.get('DEFAULT_PASS_N_SOLVER_TEMPERATURE', 1.0)
    
    logger.info(f"Generating {n_attempts} solution attempts for Pass@{n_attempts}.")

    def run_attempt(i: int):
        logger.info(f"Generating attempt {i+1}/{n_attempts}.")
        print(f"    -> Generating solution attempt {i+1}/{n_attempts}...")
        print(f"      [API Context] Calling LLM for: Final Solution (Attempt #{i+1})")

        response = api_manager.generate_content(
            prompt, model_name, temperature, avalai_role="final_solver"
        )
        trace_entry = create_trace_entry(
            "solve", f"attempt_{i+1}",
            {"prompt": prompt}, response, {"model": model_name, "temp": temperature}
        )
        return response, trace_entry

    tasks = [lambda i=i: run_attempt(i) for i in range(n_attempts)]
    attempt_results = run_parallel_api_calls(tasks, config)

    solution_attempts: List[Union[str, Dict]] = []
    for response, trace_entry in attempt_results:
        local_trace.append(trace_entry)

        if response['status'] == 'SUCCESS':
            solution_attempts.append(response['text'])
        else:
            solution_attempts.append({
                "status": "FAILURE",
                "error_info": response
            })
            
    return {"status": "SUCCESS", "solution_attempts": solution_attempts, "trace": local_trace}

def solve_via_main_simplification(
    target_query: str,
    final_exemplars: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info("Starting Solver via Main Question Simplification.")
    local_trace = []
    
    if isinstance(api_manager, GeminiAPIManager):
        model_simp = config.get('GEMINI_MODEL_NAME_SIMPLIFICATION', config['GEMINI_MODEL_NAME_ADAPTATION'])
        model_solve = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, AvalAIAPIManager):
        model_simp = config.get('AVALAI_MODEL_NAME_SIMPLIFICATION', config['AVALAI_MODEL_NAME_ADAPTATION'])
        model_solve = config['AVALAI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, OllamaAPIManager):
        model_simp = config.get('OLLAMA_MODEL_NAME_SIMPLIFICATION', config['OLLAMA_MODEL_NAME_ADAPTATION'])
        model_solve = config['OLLAMA_MODEL_NAME_FINAL_SOLVER']
    else:
        raise TypeError("Unsupported API manager type.")

    temp_simp = config.get("DEFAULT_SIMPLIFICATION_TEMPERATURE", 0.3)
    temp_solve = config.get("DEFAULT_FINAL_SOLVER_TEMPERATURE", 1.0)
    
    print("    -> [Simplification] Generating Simplified Main Question...")
    prompt_simp = create_simplification_prompt(target_query, config)
    resp_simp = api_manager.generate_content(
        prompt_simp, model_simp, temp_simp, avalai_role="adaptation"
    )
    
    local_trace.append(create_trace_entry(
        "solve_via_simplification", "simplify_main",
        {"prompt": prompt_simp}, resp_simp, {"model": model_simp, "temp": temp_simp}
    ))

    if resp_simp['status'] != 'SUCCESS':
        logger.error(f"Failed to simplify main question: {resp_simp.get('error_message')}")
        return {"status": "FAILURE", "solution_attempts": [{"status": "FAILURE", "error_info": resp_simp, "phase": "simplify_main"}], "trace": local_trace}
    
    simple_main_q = resp_simp['text'].strip()
    print(f"       Simplified Q: '{simple_main_q[:50]}...'")

    print("    -> [Simplification] Solving Simplified Main Question...")
    
    if final_exemplars:
        prompt_solve_simple = create_final_reasoning_prompt(simple_main_q, final_exemplars, config)
    else:
        prompt_solve_simple = create_final_reasoning_prompt_simple(simple_main_q, config)
        
    resp_solve_simple = api_manager.generate_content(
        prompt_solve_simple, model_solve, temp_solve,
        avalai_role="final_solver",
    )
    
    local_trace.append(create_trace_entry(
        "solve_via_simplification", "solve_simple_q",
        {"simple_q": simple_main_q, "prompt": prompt_solve_simple}, resp_solve_simple, {"model": model_solve, "temp": temp_solve}
    ))

    if resp_solve_simple['status'] != 'SUCCESS':
        logger.error(f"Failed to solve simplified main question: {resp_solve_simple.get('error_message')}")
        return {"status": "FAILURE", "solution_attempts": [{"status": "FAILURE", "error_info": resp_solve_simple, "phase": "solve_simple"}], "trace": local_trace}
    
    simple_solution = resp_solve_simple['text']
    
    print("    -> [Simplification] Solving Original Main Question via Proxy...")
    prompt_proxy = create_main_from_simplified_proxy_prompt(target_query, simple_solution, config)
    
    n_attempts = config.get("N_PASS_ATTEMPTS", 1)
    solution_attempts = []
    
    for i in range(n_attempts):
        print(f"       Generating Attempt {i+1}/{n_attempts}...")
        resp_final = api_manager.generate_content(
            prompt_proxy, model_solve, temp_solve, avalai_role="final_solver"
        )
        
        local_trace.append(create_trace_entry(
            "solve_via_simplification", f"final_attempt_{i+1}",
            {"prompt": prompt_proxy}, resp_final, {"model": model_solve, "temp": temp_solve}
        ))
        
        if resp_final['status'] == 'SUCCESS':
            solution_attempts.append(resp_final['text'])
        else:
            solution_attempts.append({"status": "FAILURE", "error_info": resp_final})
            
    return {"status": "SUCCESS", "solution_attempts": solution_attempts, "trace": local_trace}

def reverse_transform_and_solve(
    target_query: str,
    retrieved_indices: List[int],
    exemplar_questions: List[str],
    exemplar_solutions: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Implements Reverse Transformation workflow:
    For each retrieved exemplar:
    1. Transform the main question to match the exemplar
    2. Solve the transformed question using the exemplar
    3. Collect all transformed solutions
    Then:
    4. Solve the original main question using all transformed solutions as analogical support
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Reverse Transformation step.")
    local_trace = []
    
    if isinstance(api_manager, GeminiAPIManager):
        model_transform = config['GEMINI_MODEL_NAME_ADAPTATION']
        model_solve = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, AvalAIAPIManager):
        model_transform = config['AVALAI_MODEL_NAME_ADAPTATION']
        model_solve = config['AVALAI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, OllamaAPIManager):
        model_transform = config['OLLAMA_MODEL_NAME_ADAPTATION']
        model_solve = config['OLLAMA_MODEL_NAME_FINAL_SOLVER']
    else:
        raise TypeError(f"Unsupported API manager type for reverse transformation: {type(api_manager)}")
    
    temp_transform = config.get('REVERSE_TRANSFORMATION_TEMPERATURE', 0.3)
    temp_solve = config.get('REVERSE_TRANSFORMATION_SOLVER_TEMPERATURE', 1.0)
    temp_final = config.get('DEFAULT_PASS_N_SOLVER_TEMPERATURE', 1.0)
    
    transformed_solutions = []
    failed_transformations = []
    
    print(f"    -> Starting Reverse Transformation for {len(retrieved_indices)} exemplars...")
    
    # Phase 1: For each retrieved exemplar, transform Q and solve Q_Transformed
    for idx_num, idx in enumerate(retrieved_indices):
        original_exemplar_q = exemplar_questions[idx]
        original_exemplar_sol = exemplar_solutions[idx]
        
        print(f"\n      [Reverse Transform {idx_num+1}/{len(retrieved_indices)}] Processing exemplar {idx}...")
        
        # Step 1: Transform main question to match this exemplar
        print(f"        -> Transforming main question to match exemplar {idx}...")
        prompt_transform = create_reverse_transformation_main_to_exemplar_prompt(
            target_query, original_exemplar_q, original_exemplar_sol, config
        )
        
        resp_transform = api_manager.generate_content(
            prompt_transform, model_transform, temp_transform,
            avalai_role="adaptation",
        )
        
        local_trace.append(create_trace_entry(
            "reverse_transform", f"transform_q_{idx}",
            {"exemplar_idx": idx, "prompt": prompt_transform},
            resp_transform,
            {"model": model_transform, "temp": temp_transform}
        ))
        
        if resp_transform['status'] != 'SUCCESS':
            logger.warning(f"Failed to transform main question for exemplar {idx}")
            failed_transformations.append({
                "exemplar_idx": idx,
                "phase": "transformation",
                "error_info": resp_transform
            })
            continue
        
        transformed_q = resp_transform['text']
        # Extract just the question part if the response includes "Transformed Main Question:"
        if "Transformed Main Question:" in transformed_q:
            transformed_q = transformed_q.split("Transformed Main Question:")[-1].strip()
        
        # Step 2: Solve the transformed question using the exemplar
        print(f"        -> Solving transformed question using exemplar {idx}...")
        prompt_solve_transformed = create_reverse_transformation_solve_transformed_prompt(
            transformed_q, original_exemplar_q, original_exemplar_sol, config
        )
        
        resp_solve_transformed = api_manager.generate_content(
            prompt_solve_transformed, model_solve, temp_solve,
            avalai_role="final_solver",
        )
        
        local_trace.append(create_trace_entry(
            "reverse_transform", f"solve_transformed_{idx}",
            {"exemplar_idx": idx, "transformed_q": transformed_q, "prompt": prompt_solve_transformed},
            resp_solve_transformed,
            {"model": model_solve, "temp": temp_solve}
        ))
        
        if resp_solve_transformed['status'] != 'SUCCESS':
            logger.warning(f"Failed to solve transformed question for exemplar {idx}")
            failed_transformations.append({
                "exemplar_idx": idx,
                "phase": "solving_transformed",
                "error_info": resp_solve_transformed
            })
            continue
        
        transformed_sol = resp_solve_transformed['text']
        transformed_solutions.append({
            "exemplar_idx": idx,
            "transformed_question": transformed_q,
            "transformed_solution": transformed_sol,
            "original_exemplar_question": original_exemplar_q,
            "original_exemplar_solution": original_exemplar_sol
        })
    
    if not transformed_solutions:
        logger.error("No successful reverse transformations completed.")
        return {
            "status": "FAILURE",
            "solution_attempts": [{"status": "FAILURE", "error_info": {"message": "No successful reverse transformations"}}],
            "trace": local_trace
        }
    
    # Phase 2: Solve the original main question using transformed solutions
    print(f"\n      [Final Solve] Solving original question using {len(transformed_solutions)} transformed solutions...")
    
    # Format transformed solutions for the final prompt
    transformed_solutions_text = []
    for i, trans_sol_info in enumerate(transformed_solutions):
        sol_text = f"Solution {i+1} (from exemplar {trans_sol_info['exemplar_idx']}):\nTransformed Question: {trans_sol_info['transformed_question']}\n\n{trans_sol_info['transformed_solution']}"
        transformed_solutions_text.append(sol_text)
    
    prompt_final = create_reverse_transformation_final_solve_prompt(
        target_query, transformed_solutions_text, config
    )
    
    n_attempts = config.get("N_PASS_ATTEMPTS", 1)
    solution_attempts = []
    
    for i in range(n_attempts):
        print(f"        -> Generating final solution attempt {i+1}/{n_attempts}...")
        resp_final = api_manager.generate_content(
            prompt_final, model_solve, temp_final, avalai_role="final_solver"
        )
        
        local_trace.append(create_trace_entry(
            "reverse_transform", f"final_attempt_{i+1}",
            {"prompt": prompt_final},
            resp_final,
            {"model": model_solve, "temp": temp_final}
        ))
        
        if resp_final['status'] == 'SUCCESS':
            solution_attempts.append(resp_final['text'])
        else:
            solution_attempts.append({
                "status": "FAILURE",
                "error_info": resp_final
            })
    
    return {
        "status": "SUCCESS",
        "solution_attempts": solution_attempts,
        "transformed_solutions_count": len(transformed_solutions),
        "failed_transformations": failed_transformations,
        "trace": local_trace
    }















#   Analogical Mirroring Logic (MIRROR_AS_EVALUATOR)

def optimize_demonstrations_via_mirroring(
    target_query: str,
    retrieved_indices: List[int],
    exemplar_data: Dict[str, Any],
    api_manager_solve: Any,
    api_manager_eval: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main entry point for MIRROR_AS_EVALUATOR.
    Orchestrates Phases 0-4: Baseline -> Hypothesis -> Verification -> ReRanking -> Filtering.
    """
    # Initialize trace accumulator for this step
    trace_accumulator = [] 
    
    try:
        # Phase 0: Initialization
        # 1. Apply Candidate Limit (Cost Saving)
        limit = config.get("MIRROR_ACTIVE_CANDIDATE_LIMIT", 5)
        active_candidates = retrieved_indices[:limit]
        
        # 2. Inject Zero-Shot Candidate (R0) if enabled
        if config.get("MIRROR_ENABLE_R0", True):
            # We use -1 to represent R0 (Zero-Shot)
            if -1 not in active_candidates:
                active_candidates.insert(0, -1)
        
        # 3. Calculate Baseline Difficulty (S_base)
        # We use the retrieved samples themselves as the Evaluation Set
        validation_set_indices = [idx for idx in retrieved_indices if idx != -1]
        
        baseline_scores = _calculate_baseline_difficulty(
            retrieved_indices=validation_set_indices,
            exemplar_data=exemplar_data,
            api_manager_solve=api_manager_solve,
            api_manager_eval=api_manager_eval,
            config=config,
            trace_accumulator=trace_accumulator
        )

        # Phase 1: Hypothesis Generation (Forward Pass) 
        hypotheses = _generate_hypotheses(
            target_query=target_query,
            candidate_indices=active_candidates,
            exemplar_data=exemplar_data,
            api_manager=api_manager_solve,
            config=config,
            trace_accumulator=trace_accumulator
        )
        
        # Phase 2: Mirror Evaluation (Backward Pass) 
        # Returns matrix: {candidate_idx: {validation_idx: score}}
        consistency_matrix = _evaluate_mirror_consistency(
            target_query=target_query,
            hypotheses=hypotheses,
            validation_indices=validation_set_indices,
            exemplar_data=exemplar_data,
            api_manager_solve=api_manager_solve,
            api_manager_eval=api_manager_eval,
            config=config,
            trace_accumulator=trace_accumulator
        )
        
        # Phase 3: Utility Scoring 
        candidate_scores = {} 
        candidate_contributions = {} 
        
        for cand_idx in active_candidates:
            total_utility = 0.0
            contribs = {}
            
            for val_idx in validation_set_indices:
                # Include self-test in utility calculation
                # if cand_idx == val_idx: continue
                
                mirror_acc = consistency_matrix.get(cand_idx, {}).get(val_idx, 0.0)
                base_acc = baseline_scores.get(val_idx, 0.0)
                
                if mirror_acc > base_acc:
                    delta = mirror_acc
                    total_utility += delta
                    contribs[val_idx] = delta
            
            candidate_scores[cand_idx] = total_utility
            candidate_contributions[cand_idx] = contribs

        # --- Phase 4: ReRanking and Filtering ---
        final_indices, base_indices, master_sorted_indices = _rank_and_filter_candidates(
            candidate_indices=active_candidates,
            candidate_scores=candidate_scores,
            candidate_contributions=candidate_contributions,
            config=config,
            trace_accumulator=trace_accumulator
        )
        
        return {
            "status": "SUCCESS",
            "master_sorted_indices": master_sorted_indices, # List 2 (For Track A Benchmark)
            "strategies": {
                "redundancy_filtering": final_indices, # List 4 (For Track B Validation)
                "base_filtering": base_indices         # List 3 (For Track B Validation)
            },
            "trace": trace_accumulator
        }

    except Exception as e:
        import traceback
        print(f"Mirroring failed: {e}")
        print(traceback.format_exc())
        # Fallback to original retrieval on error
        return {
            "status": "FAILED", 
            "error": str(e),
            "fallback_indices": retrieved_indices,
            "trace": trace_accumulator
        }

def apply_mirror_reranking(
    target_query: str,
    indices_to_rerank: List[int],
    exemplar_data: Dict[str, Any],
    api_manager_solve: Any,
    api_manager_eval: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Unified Mirror Re-Ranking Stage for Retrieved/Transformed Samples.
    
    Applies mirror-based re-ranking to optimize demonstration selection based on
    analogical consistency (True Consistency Score) rather than just similarity.
    
    This function:
    1. Generates Q_answered candidates for each demo (hypothesis generation)
    2. Evaluates mirror consistency by testing each demo against others
    3. Scores demos based on their utility (TCS > baseline similarity)
    4. Optionally filters based on thresholds
    5. Returns a re-ranked list optimized for analogical utility
    
    Args:
        target_query: The target problem to solve
        indices_to_rerank: List of indices to re-rank (from exemplar_data)
        exemplar_data: Dict containing 'questions' and 'solutions' lists
        api_manager: API manager instance for LLM calls
        config: Configuration dict with re-ranking parameters
    
    Returns:
        {
            "status": "SUCCESS" | "FAILED",
            "reranked_indices": [...],  # Re-ranked indices list (or original if failed)
            "ranking_scores": {...},    # Score for each index
            "trace": [...]              # Execution trace
        }
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting unified mirror re-ranking for {len(indices_to_rerank)} samples")
    
    print(f"\n{'='*80}")
    print("  [MIRROR RE-RANKING] Unified Analogical Consistency Re-Ranking")
    print(f"  Input: {len(indices_to_rerank)} samples | Target Query: {target_query[:60]}...")
    print(f"{'='*80}")
    
    trace_accumulator = []
    
    try:
        if not indices_to_rerank:
            logger.warning("No indices to re-rank, returning empty result")
            return {
                "status": "SUCCESS",
                "reranked_indices": [],
                "ranking_scores": {},
                "trace": trace_accumulator
            }
        
        # Extract configuration
        enable_r0 = config.get("MIRROR_RERANKING_ENABLE_R0", False)
        enable_filtering = config.get("MIRROR_RERANKING_ENABLE_FILTERING", True)
        enable_redundancy = config.get("MIRROR_RERANKING_ENABLE_REDUNDANCY_FILTER", True)
        evaluate_base = config.get("MIRROR_RERANKING_EVALUATE_BASE_FILTERING", False)
        active_limit = config.get("MIRROR_RERANKING_ACTIVE_LIMIT", None)
        
        # --- Phase 0: Prepare Active Candidates ---
        active_candidates = list(indices_to_rerank)
        
        # Apply limit if specified
        if active_limit is not None and len(active_candidates) > active_limit:
            active_candidates = active_candidates[:active_limit]
            logger.info(f"Limiting candidates to {active_limit}")
        
        # Inject R0 if enabled
        if enable_r0:
            if -1 not in active_candidates:
                active_candidates.insert(0, -1)
                logger.info("Injected zero-shot candidate (R0)")
        
        # Validation set = all non-R0 candidates
        validation_indices = [idx for idx in active_candidates if idx != -1]
        
        if not validation_indices:
            logger.warning("No valid validation indices after filtering")
            return {
                "status": "FAILED",
                "error": "No valid validation indices",
                "fallback_indices": indices_to_rerank,
                "trace": trace_accumulator
            }
        
        # --- Phase 0.5: Calculate Baseline Scores ---
        baseline_scores = _calculate_baseline_difficulty(
            retrieved_indices=validation_indices,
            exemplar_data=exemplar_data,
            api_manager_solve=api_manager_solve,
            api_manager_eval=api_manager_eval,
            config=config,
            trace_accumulator=trace_accumulator
        )
        
        # --- Phase 1: Hypothesis Generation (Q_answered Candidates) ---
        hypotheses = _generate_hypotheses(
            target_query=target_query,
            candidate_indices=active_candidates,
            exemplar_data=exemplar_data,
            api_manager=api_manager_solve,
            config=config,
            trace_accumulator=trace_accumulator
        )
        
        # --- Phase 2: Mirror Evaluation (True Consistency Score) ---
        consistency_matrix = _evaluate_mirror_consistency(
            target_query=target_query,
            hypotheses=hypotheses,
            validation_indices=validation_indices,
            exemplar_data=exemplar_data,
            api_manager_solve=api_manager_solve,
            api_manager_eval=api_manager_eval,
            config=config,
            trace_accumulator=trace_accumulator
        )
        
        # --- Phase 3: Utility Scoring ---
        candidate_scores = {}
        candidate_contributions = {}
        
        for cand_idx in active_candidates:
            total_utility = 0.0
            contribs = {}
            
            for val_idx in validation_indices:
                
                mirror_acc = consistency_matrix.get(cand_idx, {}).get(val_idx, 0.0)
                base_acc = baseline_scores.get(val_idx, 0.0)
                
                # Utility = amount by which this demo improves over baseline
                if mirror_acc > base_acc:
                    delta = mirror_acc - base_acc
                    total_utility += delta
                    contribs[val_idx] = delta
            
            candidate_scores[cand_idx] = total_utility
            candidate_contributions[cand_idx] = contribs
        
        logger.info("Candidate utility scores calculated")
        for cand_idx in active_candidates:
            logger.info(f"  Index {cand_idx}: utility={candidate_scores[cand_idx]:.3f}")
        
        # --- Phase 4: Ranking and Filtering ---
        final_indices, base_indices, master_sorted = _rank_and_filter_candidates(
            candidate_indices=active_candidates,
            candidate_scores=candidate_scores,
            candidate_contributions=candidate_contributions,
            config=config,
            trace_accumulator=trace_accumulator
        )
        
        # Choose which list to return based on filtering settings
        if evaluate_base or (enable_filtering and not enable_redundancy):
            # Return base-filtered list
            reranked = base_indices
        else:
            # Return redundancy-filtered list (most selective)
            reranked = final_indices
        
        # If all filtering is disabled, use master sorted list
        if not enable_filtering and not enable_redundancy:
            reranked = master_sorted
        
        logger.info(f"Re-ranking complete. Final order: {reranked}")
        
        return {
            "status": "SUCCESS",
            "reranked_indices": reranked,
            "ranking_scores": candidate_scores,
            "base_filtered_indices": base_indices,
            "redundancy_filtered_indices": final_indices,
            "master_sorted_indices": master_sorted,
            "trace": trace_accumulator
        }
    
    except Exception as e:
        import traceback
        logger.error(f"Mirror re-ranking failed: {e}", exc_info=True)
        print(f"[ERROR] Mirror re-ranking failed: {e}")
        print(traceback.format_exc())
        
        return {
            "status": "FAILED",
            "error": str(e),
            "fallback_indices": indices_to_rerank,
            "trace": trace_accumulator
        }










def _generate_reverse_validation_candidates(
    target_query: str,
    n_candidates: int,
    api_manager: Any,
    model_name: str,
    config: Dict[str, Any],
    trace_accumulator: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate direct candidates owned exclusively by reverse validation."""
    prompt = create_reverse_validation_candidate_prompt(target_query, config)
    temperature = config.get("REVERSE_VALIDATION_CANDIDATE_TEMPERATURE", 1.0)
    candidates = []
    failures = []

    for candidate_index in range(n_candidates):
        response = api_manager.generate_content(
            prompt, model_name, temperature, avalai_role="final_solver"
        )
        trace_accumulator.append(create_trace_entry(
            "reverse_validation",
            f"generate_candidate_direct_{candidate_index}",
            {"prompt": prompt},
            response,
            {"model": model_name, "temp": temperature},
        ))
        if response.get("status") == "SUCCESS":
            candidates.append(
                f"Question: {target_query}\n"
                f"Rationale and Answer: {response['text']}"
            )
        else:
            failures.append({
                "candidate_index": candidate_index,
                "error_info": response,
            })

    if candidates and failures:
        status = "PARTIAL_SUCCESS"
    elif candidates:
        status = "SUCCESS"
    else:
        status = "FAILURE"
    return {"status": status, "candidates": candidates, "failures": failures}


def solve_with_analogical_consistency(
    target_query: str,
    exemplar_data: Dict[str, Any],
    embedding_model: SentenceTransformer,
    api_manager_solve: Any,
    api_manager_eval: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Analogical Consistency Check for query: {target_query[:50]}...")
    print("\n" + "="*60)
    print("  [ANALOGICAL CONSISTENCY CHECK] Reverse Validation Mode")
    print("="*60)
    
    local_trace = []

    use_rag_generation = config.get("REVERSE_VALIDATION_USE_RAG_GENERATION", False)
    gen_k = config.get("REVERSE_VALIDATION_GENERATION_K", 3)
    enable_baseline_check = config.get("REVERSE_VALIDATION_ENABLE_BASELINE_CHECK", True) # <-- NEW MASTER SWITCH

    n_candidates = config.get("REVERSE_VALIDATION_CANDIDATES_N", 5)
    k_validators = config.get("REVERSE_VALIDATION_RETRIEVAL_K", 3)
    n_validation_attempts = config.get("REVERSE_VALIDATION_ATTEMPTS_N", 5)
    
    if isinstance(api_manager_solve, GeminiAPIManager): model_name = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager_solve, AvalAIAPIManager): model_name = config['AVALAI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager_solve, OllamaAPIManager): model_name = config['OLLAMA_MODEL_NAME_FINAL_SOLVER']
    else: raise TypeError(f"Unsupported API manager: {type(api_manager_solve)}")
    
    candidates = []
    validator_indices = []

    if use_rag_generation:
        # PHASE 0 & 1 (Unified Retrieval + RAG Generation)
        total_k_to_retrieve = gen_k + k_validators
        print(f"\n  [Phase 0] Unified Retrieval for {gen_k} Helpers + {k_validators} Validators...")
        
        retrieval_res = retrieve(
            target_query, embedding_model, 
            exemplar_data['questions'], exemplar_data['embeddings'], 
            top_k=total_k_to_retrieve, question_to_index_map=exemplar_data.get('question_to_index')
        )
        if retrieval_res.get('trace'):
            local_trace.extend(retrieval_res['trace'])
            
        if retrieval_res['status'] != 'SUCCESS' or not retrieval_res['retrieved_indices']:
            logger.error("Failed to retrieve samples for Unified RAG Generation.")
            return {"status": "FAILURE", "error": "Unified retrieval failed", "trace": local_trace}
            
        all_indices = retrieval_res['retrieved_indices']
        helper_indices = all_indices[:gen_k]
        

        validator_indices = all_indices 
        
        
        print(f"\n  [Phase 1] Generating Candidates using RAG Helpers ({len(helper_indices)} helpers)...")
        for h_idx in helper_indices:
            helper_q = exemplar_data['questions'][h_idx]
            helper_a = exemplar_data['solutions'][h_idx]
            helper_text = f"Question: {helper_q}\nRationale and Answer: {helper_a}"
            
            # Using the exact prompt structure as the evaluation phase
            prompt = create_reverse_validation_prompt(target_query, helper_text, config)
            temp = config.get("REVERSE_VALIDATION_SOLVER_TEMPERATURE", 1.0)
            
            resp = api_manager_solve.generate_content(
                prompt, model_name, temp, avalai_role="final_solver"
            )
            
            local_trace.append(create_trace_entry(
                "reverse_validation", f"generate_candidate_via_helper_{h_idx}",
                {"prompt": prompt}, resp, {"model": model_name, "temp": temp}
            ))

            if resp['status'] == 'SUCCESS':
                cand_text = f"Question: {target_query}\nRationale and Answer: {resp['text']}"
                candidates.append(cand_text)
            else:
                logger.warning(f"Failed to generate candidate with helper {h_idx}.")
                
        print(f"    -> Generated {len(candidates)} candidates.")

        # Optional: add zero-shot candidates (no retrieved helpers)
        if config.get("REVERSE_VALIDATION_ADD_ZEROSHOT_CANDIDATES", False):
            rz_n = config.get("REVERSE_VALIDATION_ZEROSHOT_CANDIDATES_N", 3)
            template_name = config.get("PROMPT_TEMPLATE_REVERSE_VALIDATION_ZERO_SHOT_SOLVER", "final_solver_simple_v1")
            template = PROMPT_TEMPLATES.get(template_name)
            print(f"\n  [Phase 1.5] Generating {rz_n} zero-shot candidates using template '{template_name}'...")
            for z_idx in range(rz_n):
                # support multiple possible placeholder names in templates
                prompt = None
                if template:
                    try:
                        prompt = template.format(main_question_text=target_query)
                    except Exception:
                        try:
                            prompt = template.format(target_query=target_query)
                        except Exception:
                            try:
                                prompt = template.format(question=target_query)
                            except Exception:
                                prompt = None

                if prompt is None:
                    # fallback to simplest zero-shot mirror prompt
                    prompt = create_mirror_hypothesis_zeroshot_prompt(target_query, config)

                temp_z = config.get("DEFAULT_FINAL_SOLVER_TEMPERATURE", 1.0)
                resp_z = api_manager_solve.generate_content(
                    prompt, model_name, temp_z, avalai_role="final_solver"
                )

                local_trace.append(create_trace_entry(
                    "reverse_validation", f"generate_candidate_zeroshot_{z_idx}",
                    {"prompt": prompt}, resp_z, {"model": model_name, "temp": temp_z}
                ))

                if resp_z['status'] == 'SUCCESS':
                    cand_text_z = f"Question: {target_query}\nRationale and Answer: {resp_z['text']}"
                    candidates.append(cand_text_z)
                else:
                    logger.warning(f"Failed to generate zero-shot candidate #{z_idx}.")

            print(f"    -> Added {len(candidates)} total candidates after zero-shot generation.")
        print(f"\n  [Phase 2] Using {len(validator_indices)} Pre-retrieved Validators...")

    else:
        print(f"\n  [Phase 1] Generating {n_candidates} Candidate Solutions...")

        candidates_result = _generate_reverse_validation_candidates(
            target_query=target_query,
            n_candidates=n_candidates,
            api_manager=api_manager_solve,
            model_name=model_name,
            config=config,
            trace_accumulator=local_trace,
        )

        if candidates_result['status'] == 'FAILURE':
            logger.error("Failed to generate any candidates.")
            return {"status": "FAILURE", "error": "Candidate generation failed", "trace": local_trace}

        candidates = candidates_result['candidates']
        print(f"    -> Generated {len(candidates)} candidates.")

        # Optional: add zero-shot candidates to the direct candidates
        if config.get("REVERSE_VALIDATION_ADD_ZEROSHOT_CANDIDATES", False):
            rz_n = config.get("REVERSE_VALIDATION_ZEROSHOT_CANDIDATES_N", 3)
            template_name = config.get("PROMPT_TEMPLATE_REVERSE_VALIDATION_ZERO_SHOT_SOLVER", "final_solver_simple_v1")
            template = PROMPT_TEMPLATES.get(template_name)
            print(f"\n  [Phase 1.5] Generating {rz_n} zero-shot candidates using template '{template_name}'...")
            for z_idx in range(rz_n):
                prompt = None
                if template:
                    try:
                        prompt = template.format(main_question_text=target_query)
                    except Exception:
                        try:
                            prompt = template.format(target_query=target_query)
                        except Exception:
                            try:
                                prompt = template.format(question=target_query)
                            except Exception:
                                prompt = None

                if prompt is None:
                    prompt = create_mirror_hypothesis_zeroshot_prompt(target_query, config)

                temp_z = config.get("DEFAULT_FINAL_SOLVER_TEMPERATURE", 1.0)
                resp_z = api_manager_solve.generate_content(
                    prompt, model_name, temp_z, avalai_role="final_solver"
                )

                local_trace.append(create_trace_entry(
                    "reverse_validation", f"generate_candidate_zeroshot_{z_idx}",
                    {"prompt": prompt}, resp_z, {"model": model_name, "temp": temp_z}
                ))

                if resp_z['status'] == 'SUCCESS':
                    cand_text_z = f"Question: {target_query}\nRationale and Answer: {resp_z['text']}"
                    candidates.append(cand_text_z)
                else:
                    logger.warning(f"Failed to generate zero-shot candidate #{z_idx}.")

            print(f"    -> Added {rz_n} zero-shot candidates. Total candidates: {len(candidates)}")

        print(f"\n  [Phase 2] Retrieving {k_validators} Validators (Ground Truths)...")
        
        retrieval_res = retrieve(
            target_query, embedding_model, 
            exemplar_data['questions'], exemplar_data['embeddings'], 
            top_k=k_validators, question_to_index_map=exemplar_data.get('question_to_index')
        )
        if retrieval_res.get('trace'):
            local_trace.extend(retrieval_res['trace'])
        
        if retrieval_res['status'] != 'SUCCESS' or not retrieval_res['retrieved_indices']:
            logger.error("Failed to retrieve validators.")
            return {"status": "FAILURE", "error": "Validator retrieval failed", "trace": local_trace}
            
        validator_indices = retrieval_res['retrieved_indices']
        print(f"    -> Retrieved {len(validator_indices)} validators.")

    # PHASE 2.5: FORMAT VALIDATORS & CALCULATE BASELINES
    if not candidates:
        return {"status": "FAILURE", "error": "No candidates were generated", "trace": local_trace}

    validators = []
    for idx in validator_indices:
        validators.append({
            "question": exemplar_data['questions'][idx],
            "ground_truth": exemplar_data['solutions'][idx]
        })

    baseline_scores = {}
    if enable_baseline_check:
        print(f"\n  [Phase 2.75] Calculating Zero-Shot Baseline for {len(validators)} Validators...")
        # Get template or default
        base_template = PROMPT_TEMPLATES.get(
            config.get("PROMPT_TEMPLATE_REVERSE_VALIDATION_BASELINE", "mirror_baseline_zero_shot_v1"),
            "Problem: {question}\nSolve this step-by-step.\nFinal Answer:"
        )
        
        for v_idx, val in enumerate(validators):
            prompt = base_template.format(question=val['question'])
            v_correct = 0
            
            for att in range(n_validation_attempts): # Reuse same N attempts for fair comparison
                resp = api_manager_solve.generate_content(
                    prompt, model_name, temperature=1.0,
                    avalai_role="final_solver",
                )
                
                local_trace.append(create_trace_entry(
                    "reverse_validation", f"baseline_validator_{v_idx}_attempt_{att}",
                    {"prompt": prompt}, resp, {"model": model_name, "temp": 1.0}
                ))

                if resp['status'] == 'SUCCESS':
                    eval_res = evaluate_single_answer_with_llm(resp['text'], val['ground_truth'], api_manager_eval, config)
                    if eval_res['status'] == 'SUCCESS' and eval_res['is_correct']:
                        v_correct += 1
                        
            baseline_scores[v_idx] = v_correct / n_validation_attempts
            print(f"      -> Validator #{v_idx} Baseline Score: {baseline_scores[v_idx]:.2f}")

    # PHASE 3: REVERSE VALIDATION LOOP WITH BASELINE THRESHOLDS
    print(f"\n  [Phase 3] Reverse Validation Loop ({len(candidates)} Candidates x {len(validators)} Validators x {n_validation_attempts} Attempts)...")
    
    candidate_stats = []
    
    for c_idx, cand_text in enumerate(candidates):
        
        total_utility = 0.0
        correct_attempts = 0 # Track total correct just for logging
        total_attempts = 0
        validator_details = []
        
        print(f"    -> Testing Candidate #{c_idx + 1}...")
        
        for v_idx, val in enumerate(validators):
            val_q = val['question']
            val_gt = val['ground_truth']
            
            prompt = create_reverse_validation_prompt(val_q, cand_text, config)
            temp = config.get("REVERSE_VALIDATION_SOLVER_TEMPERATURE", 1.0)
            
            v_correct = 0
            
            for att in range(n_validation_attempts):
                resp = api_manager_solve.generate_content(
                    prompt, model_name, temp, avalai_role="final_solver"
                )
                
                local_trace.append(create_trace_entry(
                    "reverse_validation", f"solve_candidate_{c_idx}_validator_{v_idx}_attempt_{att}",
                    {"prompt": prompt}, resp, {"model": model_name, "temp": temp}
                ))

                if resp['status'] == 'SUCCESS':
                    eval_res = evaluate_single_answer_with_llm(
                        resp['text'], val_gt, api_manager_eval, config
                    )
                    
                    if eval_res['status'] == 'SUCCESS' and eval_res['is_correct']:
                        v_correct += 1
                        
            cand_val_score = v_correct / n_validation_attempts
            total_attempts += n_validation_attempts
            correct_attempts += v_correct
            
            # --- THE NEW BASELINE COMPARISON LOGIC ---
            if enable_baseline_check:
                base_score = baseline_scores.get(v_idx, 0.0)
                if cand_val_score > base_score:
                    total_utility += cand_val_score  # Reward the candidate
                    status_str = f"Considered (Beat Base {base_score:.2f})"
                else:
                    total_utility += 0.0 # Ignored
                    status_str = f"Ignored (<= Base {base_score:.2f})"
            else:
                total_utility += cand_val_score
                status_str = "Considered (Check OFF)"
                
            validator_details.append({
                "validator_idx": v_idx, 
                "score": f"{v_correct}/{n_validation_attempts}",
                "status": status_str
            })
            
        # For logging simplicity, if check is OFF, we average the utility. 
        # If check is ON, we just use the raw utility sum as the consistency_score.
        consistency_score = total_utility / len(validators) if not enable_baseline_check and validators else total_utility
        
        print(f"       -> Utility Score: {consistency_score:.2f} (Total Correct: {correct_attempts}/{total_attempts})")
        
        candidate_stats.append({
            "candidate_id": c_idx,
            "candidate_text": cand_text,
            "consistency_score": consistency_score,
            "raw_score": f"{correct_attempts}/{total_attempts}",
            "validator_breakdown": validator_details
        })

    # PHASE 4: SELECTION & FALLBACK LOGIC
    if not candidate_stats:
        return {"status": "FAILURE", "error": "No stats generated", "trace": local_trace}
        
    candidate_stats.sort(key=lambda x: x['consistency_score'], reverse=True)
    best_candidate_stat = candidate_stats[0]
    
    # NEW FALLBACK LOGIC 
    if enable_baseline_check and best_candidate_stat['consistency_score'] <= 0.0:
        print("\n  [Selection] ALL candidates failed to beat the baseline on all validators.")
        print("  [Selection] Triggering FALLBACK to Candidate #1.")
        selected_text = candidates[0]
        selected_score = 0.0
    else:
        selected_text = best_candidate_stat['candidate_text']
        selected_score = best_candidate_stat['consistency_score']
        print(f"\n  [Selection] Selected Candidate #{best_candidate_stat['candidate_id'] + 1} with Utility Score {selected_score:.2f}")
    
    return {
        "status": "SUCCESS",
        "selected_candidate": selected_text,
        "selected_score": selected_score,
        "solution_attempts": [selected_text], 
        "consistency_stats": candidate_stats,
        "trace": local_trace
    }


def select_best_transformations(
    retrieved_indices: List[int],
    target_query: str,
    exemplar_data: Dict[str, Any],
    embedding_model: SentenceTransformer,
    api_manager_adapt: Any,
    api_manager_solve: Any,
    api_manager_eval: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    REFACTORED Best-of-Transformation: Centralized Candidate Pool Architecture
    
    Core Principle: Build ONE unified candidate pool per retrieved sample, run mirror
    evaluation ONCE over the entire pool, then use deterministic selection.
    
    Architecture (per retrieved sample):
    
    PHASE 1: CENTRALIZED POOL CONSTRUCTION
    - Build: [R_main, T_1, T_2, ..., T_N] (original + N transformations)
    - Index 0 = R_main (original)
    - Indices 1 to N = Transformations T_1 through T_N
    - Single pass: no redundant API calls
    
    PHASE 2: UNIFIED MIRROR SCORING (once per pool)
    - Score all N+1 candidates in single pass
    - Returns: Scores = [S_0, S_1, S_2, ..., S_N]
    
    PHASE 3: DETERMINISTIC SELECTION WITH TIE-BREAKING
    - Best = argmax_tiebreak(Scores)
    - Tie-breaking rule: (score DESC, index ASC)
    - On ties, favor lower indices (prefer R_main for safety)
    
    PHASE 4: STATE FAN-OUT
    - Selected context stored for downstream use
    - Solver runs with this single context
    
    Args:
        retrieved_indices: List of K retrieved exemplar indices
        target_query: Main problem query
        exemplar_data: Contains questions, solutions, embeddings
        api_manager_adapt: API manager for transformations
        api_manager_solve: API manager for solving
        api_manager_eval: API manager for evaluation
        config: Pipeline configuration
    
    Returns:
        {
            "status": "SUCCESS" | "FAILURE",
            "evaluation_contexts": {
                retrieved_idx: {
                    "scenario": "Best-of-Transformation (Centralized Pool)",
                    "pool_index": int,  # 0 = R_main, 1+ = transformations
                    "pool_size": int,
                    "selected_candidate": {...},
                    "selection_score": float,
                    "all_scores": [...],
                    "source": "original_retrieval" or "transformation"
                },
                ...
            },
            "telemetry": {
                "pool_construction_passes": K,
                "scoring_passes": K,
                "total_transformations": K*N,
                "selections": {
                    retrieved_idx: {
                        "pool_size": int,
                        "selected_index": int,
                        "selected_score": float,
                        "selected_source": str,
                        "all_scores": [...]
                    },
                    ...
                }
            },
            "trace": [...]
        }
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Best-of-Transformation with centralized pool for {len(retrieved_indices)} retrieved samples")
    
    print(f"\n{'='*80}")
    print("  [BEST-OF-TRANSFORMATION] Centralized Candidate Pool Architecture")
    print("  Phase 1: Pool Construction | Phase 2: Unified Scoring | Phase 3: Selection")
    print(f"{'='*80}")
    
    local_trace = []
    
    # Configuration
    n_transformations = config.get("BEST_OF_TRANSFORMATION_N_SAMPLES", 3)
    enable_mirror_eval = config.get("BEST_OF_TRANSFORMATION_ENABLE_MIRROR_EVAL", True)
    mirror_eval_attempts = config.get("BEST_OF_TRANSFORMATION_MIRROR_EVAL_ATTEMPTS", 3)
    
    # Determine model names
    if isinstance(api_manager_adapt, GeminiAPIManager):
        model_adapt = config['GEMINI_MODEL_NAME_ADAPTATION']
        model_solve = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager_adapt, AvalAIAPIManager):
        model_adapt = config['AVALAI_MODEL_NAME_ADAPTATION']
        model_solve = config['AVALAI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager_adapt, OllamaAPIManager):
        model_adapt = config['OLLAMA_MODEL_NAME_ADAPTATION']
        model_solve = config['OLLAMA_MODEL_NAME_FINAL_SOLVER']
    else:
        raise TypeError(f"Unsupported API manager: {type(api_manager_adapt)}")
    
    temp_transform = config.get("DEFAULT_ADAPTATION_TEMPERATURE", 0.0)
    temp_solve = config.get("BEST_OF_TRANSFORMATION_SOLVER_TEMPERATURE", 1.0)
    temp_eval = config.get("DEFAULT_EVALUATOR_TEMPERATURE", 0.0)
    
    evaluation_contexts = {}
    telemetry_selections = {}
    
    def argmax_tiebreak(scores: List[float]) -> int:
        """Deterministic argmax with tie-breaking favoring lower indices."""
        if not scores:
            return 0
        best_candidate = max(
            enumerate(scores),
            key=lambda pr: (pr[1], -pr[0])
        )
        return best_candidate[0]
    
    # PROCESS EACH RETRIEVED SAMPLE WITH CENTRALIZED POOL ARCHITECTURE
    for sample_idx, retrieved_idx in enumerate(retrieved_indices):
        original_q = exemplar_data['questions'][retrieved_idx]
        original_sol = exemplar_data['solutions'][retrieved_idx]
        original_combined = f"Question: {original_q}\nSolution: {original_sol}"
        
        print(f"\n[SAMPLE {sample_idx+1}/{len(retrieved_indices)}] Retrieved Index #{retrieved_idx}")
        print(f"  Original Q: {original_q[:60]}...")
        
        # PHASE 1: CENTRALIZED POOL CONSTRUCTION
        print("  [PHASE 1] Building centralized candidate pool...")
        
        pool = []
        
        # Index 0: Original retrieval (R_main)
        pool.append({
            "pool_index": 0,
            "text": original_combined,
            "is_original": True,
            "transformation_idx": None,
            "source": "original_retrieval"
        })
        
        # Indices 1+: Transformations (T_1, T_2, ..., T_N)
        for t_idx in range(n_transformations):
            prompt_transform = create_transformation_prompt(
                target_query=target_query,
                text_to_transform=original_combined,
                config=config,
                template_key_name=config.get("BEST_OF_TRANSFORMATION_TRANSFORMATION_TEMPLATE", "transformation_shallow-&-moderately-deep")
            )
            
            resp_transform = api_manager_adapt.generate_content(
                prompt_transform, model_adapt, temp_transform,
                avalai_role="adaptation",
            )
            
            local_trace.append(create_trace_entry(
                "best_of_transformation_centralized_pool",
                f"sample_{retrieved_idx}_transform_{t_idx}",
                {"prompt": prompt_transform}, resp_transform,
                {"model": model_adapt, "temp": temp_transform}
            ))
            
            if resp_transform['status'] == 'SUCCESS':
                pool.append({
                    "pool_index": t_idx + 1,
                    "text": resp_transform['text'],
                    "is_original": False,
                    "transformation_idx": t_idx,
                    "source": "transformation"
                })
            else:
                logger.warning(f"Transformation failed for sample {retrieved_idx}, transform {t_idx}")
                # Fallback: duplicate original
                pool.append({
                    "pool_index": t_idx + 1,
                    "text": f"[FALLBACK_T{t_idx}] {original_combined}",
                    "is_original": False,
                    "transformation_idx": t_idx,
                    "source": "transformation_fallback"
                })
        
        pool_size = len(pool)
        print(f"  Pool size: {pool_size} (1 original + {n_transformations} transformations)")
        
        # PHASE 2: UNIFIED MIRROR SCORING (ONCE PER POOL)
        print("  [PHASE 2] Running unified mirror scoring over entire pool...")
        
        scores = []
        candidate_texts = []
        
        for pool_idx, candidate_dict in enumerate(pool):
            # Generate solution for this candidate
            prompt_solve = create_best_of_transformation_solver_prompt(
                target_query,
                candidate_dict['text'],
                config
            )
            
            resp_solve = api_manager_solve.generate_content(
                prompt_solve, model_solve, temp_solve,
                avalai_role="final_solver",
            )
            
            local_trace.append(create_trace_entry(
                "best_of_transformation_centralized_pool",
                f"sample_{retrieved_idx}_pool_{pool_idx}_solve",
                {"prompt": prompt_solve}, resp_solve,
                {"model": model_solve, "temp": temp_solve}
            ))
            
            if resp_solve['status'] == 'FAILURE':
                logger.warning(f"Solve failed for sample {retrieved_idx}, pool index {pool_idx}")
                scores.append(0.0)
                candidate_texts.append("[SOLVE_FAILED]")
                continue
            
            solution_text = resp_solve['text']
            candidate_texts.append(solution_text)
            
            # Score this candidate via mirror evaluation
            score = 0.0
            
            if enable_mirror_eval:
                for eval_attempt in range(mirror_eval_attempts):
                    prompt_eval = create_reverse_validation_prompt(
                        solution_text,
                        original_combined,
                        config
                    )
                    
                    resp_eval = api_manager_eval.generate_content(
                        prompt_eval, model_solve, temp_eval,
                        avalai_role="evaluator",
                    )
                    
                    local_trace.append(create_trace_entry(
                        "best_of_transformation_centralized_pool",
                        f"sample_{retrieved_idx}_pool_{pool_idx}_eval_{eval_attempt}",
                        {"prompt": prompt_eval}, resp_eval,
                        {"model": model_solve, "temp": temp_eval}
                    ))
                    
                    if resp_eval['status'] == 'SUCCESS':
                        response_lower = resp_eval['text'].lower()
                        if any(word in response_lower for word in ['correct', 'yes', 'accurate', 'valid', 'true']):
                            score += 1.0
                
                score = score / mirror_eval_attempts
            else:
                score = 1.0  # Default if mirror eval disabled
            
            scores.append(score)
        
        print(f"  Scored all {len(pool)} candidates: {scores}")
        
        # PHASE 3: DETERMINISTIC SELECTION WITH TIE-BREAKING
        print("  [PHASE 3] Applying deterministic selection with tie-breaking...")
        
        best_idx = argmax_tiebreak(scores)
        best_candidate = pool[best_idx]
        best_score = scores[best_idx]
        best_source = "original_retrieval" if best_candidate['is_original'] else "transformation"
        
        print(f"  Selected: Index {best_idx} ({best_source}), Score: {best_score:.3f}")
        
        # BUILD EVALUATION CONTEXT
        evaluation_contexts[retrieved_idx] = {
            "scenario": "Best-of-Transformation (Centralized Pool)",
            "pool_index": best_idx,
            "pool_size": pool_size,
            "selected_candidate": {
                "text": best_candidate['text'],
                "source": best_source,
                "transformation_idx": best_candidate['transformation_idx']
            },
            "selection_score": best_score,
            "all_scores": scores,
            "candidate_texts": candidate_texts
        }
        
        telemetry_selections[retrieved_idx] = {
            "pool_size": pool_size,
            "selected_index": best_idx,
            "selected_score": best_score,
            "selected_source": best_source,
            "all_scores": scores
        }
    
    # BUILD OUTPUT
    print(f"\n{'='*80}")
    print("  [BEST-OF-TRANSFORMATION] Centralized Pool Processing Complete")
    print(f"  Total Retrieved Samples: {len(retrieved_indices)}")
    print(f"  Total Transformations Generated: {len(retrieved_indices) * n_transformations}")
    print(f"  Total Candidates Evaluated: {len(retrieved_indices) * (1 + n_transformations)}")
    print(f"{'='*80}\n")
    
    if not evaluation_contexts:
        return {
            "status": "FAILURE",
            "error": "No evaluation contexts created",
            "trace": local_trace
        }
    
    # Extract best candidates for backwards compatibility
    best_candidates = [
        {
            "retrieved_idx": retrieved_idx,
            "candidate_text": ctx["selected_candidate"]["text"],
            "candidate_answer": ctx["selected_candidate"]["text"],
            "source_transformation_idx": ctx["selected_candidate"]["transformation_idx"],
            "validation_score": ctx["selection_score"]
        }
        for retrieved_idx, ctx in evaluation_contexts.items()
    ]
    
    return {
        "status": "SUCCESS",
        "evaluation_contexts": evaluation_contexts,
        "best_candidates": best_candidates,
        "telemetry": {
            "pool_construction_passes": len(retrieved_indices),
            "scoring_passes": len(retrieved_indices),
            "total_transformations": len(retrieved_indices) * n_transformations,
            "selections": telemetry_selections
        },
        "trace": local_trace
    }
