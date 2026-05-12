# src/layer1_base_execution.py

"""
Layer 1: Base Execution Phase

This module implements the foundational data-gathering engine for the advanced
Analogical Mirroring system. Layer 1 is an execution-only, API-driven phase
designed to:

1. Perform all expensive LLM API operations exactly once
2. Capture complete execution state and intermediate results
3. Serialize everything into a persistent JSON cache file
4. Enable Layer 2's offline analytical experiments without API calls

The data generated here (retrieved samples, candidates, baselines, cross-evaluation
matrix, and ground-truth labels) is the complete foundation for Layer 2's
PTU calculations, reranking experiments, and optimization strategies.

Key Design Principles:
- CACHE-FIRST: Check for existing cache before making any API calls
- EXHAUSTIVE: Execute ALL phases even if intermediate steps seem to fail
- NON-ANALYTIC: No PTU math, ranking, or optimization—only execution and caching
- SERIALIZABLE: Every piece of data must be JSON-safe and storable
"""

import logging
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

from src.api_manager import GeminiAPIManager, AvalAIAPIManager, OllamaAPIManager
from src.utils import save_json, load_json, create_trace_entry
from src.pipeline_steps import (
    retrieve,
    _calculate_baseline_difficulty,
    _generate_hypotheses,
    _evaluate_mirror_consistency
)
from src.evaluation import evaluate_single_answer_with_llm


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def _get_cache_filename(
    target_query_index: int,
    top_k: int,
    n_candidates: int,
    dataset_name: str = "hard_questions"
) -> str:
    """
    Generates a deterministic cache filename based on configuration.
    Format: layer1_cache_idx{idx}_k{k}_n{n}_{dataset}.json
    """
    return f"layer1_cache_idx{target_query_index}_k{top_k}_n{n_candidates}_{dataset_name}.json"


def _get_cache_path(cache_dir: str, filename: str) -> str:
    """Returns the full path to a cache file."""
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, filename)


def _cache_exists(cache_path: str) -> bool:
    """Checks if a cache file exists and is valid."""
    if not os.path.exists(cache_path):
        return False
    try:
        data = load_json(cache_path)
        # Validate that it has all required sections
        required_keys = {
            "target_query_data",
            "retrieved_set",
            "candidate_set",
            "intrinsic_baselines",
            "cross_evaluation_matrix",
            "ground_truth_labels"
        }
        return all(key in data for key in required_keys)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Cache validation failed for {cache_path}: {e}")
        return False


def _load_cached_state(cache_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads a cached Layer 1 state from disk.
    Returns None if the cache is invalid or doesn't exist.
    """
    if not _cache_exists(cache_path):
        return None
    
    try:
        data = load_json(cache_path)
        logging.getLogger(__name__).info(f"Successfully loaded Layer 1 cache from {cache_path}")
        return data
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load Layer 1 cache from {cache_path}: {e}")
        return None


def _save_cached_state(cache_path: str, state: Dict[str, Any]) -> bool:
    """
    Saves the complete Layer 1 state to a JSON cache file.
    Returns True if successful, False otherwise.
    """
    try:
        save_json(cache_path, state)
        logging.getLogger(__name__).info(f"Successfully saved Layer 1 cache to {cache_path}")
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save Layer 1 cache to {cache_path}: {e}")
        return False


# ============================================================================
# STEP A: RETRIEVAL
# ============================================================================

def _execute_retrieval(
    target_query: str,
    embedding_model: SentenceTransformer,
    exemplar_questions: List[str],
    embedded_exemplars: np.ndarray,
    top_k: int,
    question_to_index_map: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    Executes Step A: Retrieval.
    
    Returns:
        {
            "status": "SUCCESS" | "FAILURE",
            "retrieved_indices": [list of indices],
            "retrieval_data": [list of dicts with question/solution/metadata]
        }
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[Layer 1 - Step A] Starting retrieval for top-{top_k} exemplars")
    
    try:
        # Call the existing retrieve function
        retrieval_result = retrieve(
            target_query=target_query,
            embedding_model=embedding_model,
            exemplar_questions=exemplar_questions,
            embedded_exemplars=embedded_exemplars,
            top_k=top_k,
            question_to_index_map=question_to_index_map
        )
        
        if retrieval_result["status"] != "SUCCESS":
            return {"status": "FAILURE", "error": "Retrieval failed"}
        
        retrieved_indices = retrieval_result.get("retrieved_indices", [])
        
        # Collect the actual exemplar data for storage
        from src.pipeline_steps import exemplar_questions as global_exemplars, exemplar_solutions
        retrieval_data = []
        
        # Use the exemplar data from the scope where available
        try:
            for idx in retrieved_indices:
                retrieval_data.append({
                    "corpus_index": int(idx),
                    "question": exemplar_questions[idx] if idx < len(exemplar_questions) else "",
                    "similarity_score": None  # Can be populated if needed from retrieval_result
                })
        except Exception as e:
            logger.warning(f"Could not fully populate retrieval data: {e}")
        
        return {
            "status": "SUCCESS",
            "retrieved_indices": retrieved_indices,
            "retrieval_data": retrieval_data,
            "trace": retrieval_result.get("trace", [])
        }
    
    except Exception as e:
        logger.error(f"Retrieval failed with exception: {e}", exc_info=True)
        return {"status": "FAILURE", "error": str(e)}


# ============================================================================
# STEP B: CANDIDATE GENERATION (1-Shot Constraint)
# ============================================================================

def _execute_candidate_generation(
    target_query: str,
    retrieved_indices: List[int],
    exemplar_data: Dict[str, Any],
    api_manager: Any,
    config: Dict[str, Any],
    trace_accumulator: List[Dict]
) -> Dict[str, Any]:
    """
    Executes Step B: Candidate Generation.
    
    Generates exactly one candidate per retrieved exemplar using strict 1-shot mapping.
    Each candidate H_i is generated using only exemplar R_i as the analogical reference.
    
    Returns:
        {
            "status": "SUCCESS" | "PARTIAL" | "FAILURE",
            "candidates": {
                <retrieved_idx>: {
                    "candidate_text": str,
                    "source_exemplar_idx": int,
                    "generation_status": "SUCCESS" | "FAILURE"
                }
            }
        }
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[Layer 1 - Step B] Generating {len(retrieved_indices)} candidates (1-shot constraint)")
    
    candidates = {}
    failed_count = 0
    
    try:
        # Generate candidates using the existing _generate_hypotheses function
        # We use the retrieved indices as the candidate_indices directly
        hypotheses = _generate_hypotheses(
            target_query=target_query,
            candidate_indices=retrieved_indices,
            exemplar_data=exemplar_data,
            api_manager=api_manager,
            config=config,
            trace_accumulator=trace_accumulator
        )
        
        # Convert hypotheses dict to our standardized format
        for exemplar_idx, hypothesis_text in hypotheses.items():
            if hypothesis_text:
                candidates[exemplar_idx] = {
                    "candidate_text": hypothesis_text,
                    "source_exemplar_idx": exemplar_idx,
                    "generation_status": "SUCCESS"
                }
            else:
                candidates[exemplar_idx] = {
                    "candidate_text": None,
                    "source_exemplar_idx": exemplar_idx,
                    "generation_status": "FAILURE"
                }
                failed_count += 1
        
        status = "SUCCESS" if failed_count == 0 else ("PARTIAL" if failed_count < len(retrieved_indices) else "FAILURE")
        
        return {
            "status": status,
            "candidates": candidates,
            "generated_count": len(hypotheses) - failed_count,
            "failed_count": failed_count
        }
    
    except Exception as e:
        logger.error(f"Candidate generation failed with exception: {e}", exc_info=True)
        return {"status": "FAILURE", "error": str(e), "candidates": {}}


# ============================================================================
# STEP C: BASELINE & EVALUATOR CONSTRUCTION
# ============================================================================

def _execute_baseline_calculation(
    retrieved_indices: List[int],
    exemplar_data: Dict[str, Any],
    api_manager: Any,
    config: Dict[str, Any],
    trace_accumulator: List[Dict]
) -> Dict[str, Any]:
    """
    Executes Step C: Intrinsic Baseline Calculation.
    
    Calculates the zero-shot solvability score (baseline difficulty) for each
    retrieved sample without any analogical help.
    
    Returns:
        {
            "status": "SUCCESS" | "PARTIAL" | "FAILURE",
            "intrinsic_baselines": {
                <retrieved_idx>: <baseline_score (float)>
            }
        }
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[Layer 1 - Step C] Calculating intrinsic baselines for {len(retrieved_indices)} samples")
    
    try:
        # Use the existing _calculate_baseline_difficulty function
        baselines = _calculate_baseline_difficulty(
            retrieved_indices=retrieved_indices,
            exemplar_data=exemplar_data,
            api_manager=api_manager,
            config=config,
            trace_accumulator=trace_accumulator
        )
        
        return {
            "status": "SUCCESS",
            "intrinsic_baselines": baselines
        }
    
    except Exception as e:
        logger.error(f"Baseline calculation failed with exception: {e}", exc_info=True)
        return {"status": "FAILURE", "error": str(e), "intrinsic_baselines": {}}


# ============================================================================
# STEP D: CROSS-EVALUATION MATRIX
# ============================================================================

def _execute_cross_evaluation(
    target_query: str,
    candidates: Dict[int, Dict[str, Any]],
    retrieved_indices: List[int],
    exemplar_data: Dict[str, Any],
    api_manager: Any,
    config: Dict[str, Any],
    trace_accumulator: List[Dict]
) -> Dict[str, Any]:
    """
    Executes Step D: Cross-Evaluation Matrix.
    
    Tests every generated candidate H_i against every evaluator (retrieved sample) R_j.
    Records raw binary success/failure results without calculating PTU deltas.
    
    Returns:
        {
            "status": "SUCCESS" | "PARTIAL" | "FAILURE",
            "cross_evaluation_matrix": {
                <candidate_idx>: {
                    <evaluator_idx>: <success_rate (float 0-1)>
                }
            },
            "evaluation_stats": {...}
        }
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[Layer 1 - Step D] Running cross-evaluation matrix ({len(candidates)} candidates x {len(retrieved_indices)} evaluators)")
    
    try:
        # Use the existing _evaluate_mirror_consistency function
        # This already computes the full cross-evaluation matrix
        consistency_matrix = _evaluate_mirror_consistency(
            target_query=target_query,
            hypotheses={idx: candidates[idx]["candidate_text"] for idx in candidates if candidates[idx]["generation_status"] == "SUCCESS"},
            validation_indices=retrieved_indices,
            exemplar_data=exemplar_data,
            api_manager=api_manager,
            config=config,
            trace_accumulator=trace_accumulator
        )
        
        return {
            "status": "SUCCESS",
            "cross_evaluation_matrix": consistency_matrix,
            "matrix_dimensions": {
                "candidates": len(candidates),
                "evaluators": len(retrieved_indices)
            }
        }
    
    except Exception as e:
        logger.error(f"Cross-evaluation failed with exception: {e}", exc_info=True)
        return {"status": "FAILURE", "error": str(e), "cross_evaluation_matrix": {}}


# ============================================================================
# STEP E: GROUND-TRUTH EVALUATION
# ============================================================================

def _execute_ground_truth_evaluation(
    target_query: str,
    ground_truth_answer: str,
    candidates: Dict[int, Dict[str, Any]],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes Step E: Ground-Truth Correctness Evaluation.
    
    Evaluates every candidate H_i against the known ground-truth solution
    for the main query. Records absolute correctness (True/False).
    
    Returns:
        {
            "status": "SUCCESS" | "PARTIAL" | "FAILURE",
            "ground_truth_labels": {
                <candidate_idx>: {
                    "is_correct": True | False | None,
                    "evaluation_status": "SUCCESS" | "API_FAILURE" | "PARSING_FAILED"
                }
            }
        }
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[Layer 1 - Step E] Evaluating {len(candidates)} candidates against ground truth")
    
    ground_truth_labels = {}
    successful_evals = 0
    failed_evals = 0
    
    try:
        for candidate_idx, candidate_info in candidates.items():
            candidate_text = candidate_info.get("candidate_text")
            
            if not candidate_text:
                ground_truth_labels[candidate_idx] = {
                    "is_correct": None,
                    "evaluation_status": "EMPTY_CANDIDATE"
                }
                failed_evals += 1
                continue
            
            # Call the existing evaluation function
            eval_result = evaluate_single_answer_with_llm(
                model_answer=candidate_text,
                ground_truth=ground_truth_answer,
                api_manager=api_manager,
                config=config
            )
            
            ground_truth_labels[candidate_idx] = {
                "is_correct": eval_result["is_correct"],
                "evaluation_status": eval_result["status"]
            }
            
            if eval_result["status"] == "SUCCESS":
                successful_evals += 1
            else:
                failed_evals += 1
        
        status = "SUCCESS" if failed_evals == 0 else ("PARTIAL" if successful_evals > 0 else "FAILURE")
        
        return {
            "status": status,
            "ground_truth_labels": ground_truth_labels,
            "successful_evals": successful_evals,
            "failed_evals": failed_evals
        }
    
    except Exception as e:
        logger.error(f"Ground-truth evaluation failed with exception: {e}", exc_info=True)
        return {"status": "FAILURE", "error": str(e), "ground_truth_labels": {}}


# ============================================================================
# MAIN LAYER 1 EXECUTION
# ============================================================================

def run_layer1_base_execution(
    target_query_index: int,
    target_query: str,
    ground_truth_answer: str,
    embedding_model: SentenceTransformer,
    exemplar_questions: List[str],
    exemplar_solutions: List[str],
    embedded_exemplars: np.ndarray,
    exemplar_data: Dict[str, Any],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main entry point for Layer 1: Base Execution Phase.
    
    Orchestrates the complete data-gathering pipeline:
    1. Check for cached state (Cache-First Rule)
    2. If cache miss: Execute Steps A-E
    3. Serialize complete state to JSON cache
    4. Return the cached/computed state
    
    Args:
        target_query_index: Index in the hard questions list
        target_query: The target query text
        ground_truth_answer: The known correct solution
        embedding_model: Sentence transformer for embeddings
        exemplar_questions: List of all exemplar questions
        exemplar_solutions: List of all exemplar solutions
        embedded_exemplars: Pre-computed embeddings
        exemplar_data: Structured exemplar data with questions/solutions
        api_manager: API manager instance
        config: Configuration dictionary
    
    Returns:
        The complete Layer 1 cached state (dict) with all phases populated
    """
    logger = logging.getLogger(__name__)
    
    # --- Initialization ---
    cache_dir = config.get("LAYER1_CACHE_DIR", "local_data/layer1_cache")
    top_k = config.get("TOP_N_CANDIDATES_RETRIEVAL", 5)
    n_candidates = config.get("LAYER1_N_CANDIDATES")
    if n_candidates is None:
        n_candidates = top_k  # Use top_k if LAYER1_N_CANDIDATES not explicitly set
    dataset_name = config.get("LAYER1_DATASET_NAME", "hard_questions")
    
    trace_accumulator = []
    
    print(f"\n{'='*80}")
    print(f"[LAYER 1 - BASE EXECUTION] Processing Query #{target_query_index}")
    print(f"Query: {target_query[:80]}...")
    print(f"{'='*80}")
    
    logger.info(f"[Layer 1] Starting base execution for Query #{target_query_index}")
    
    # --- CACHE-FIRST CHECK ---
    cache_filename = _get_cache_filename(target_query_index, top_k, n_candidates, dataset_name)
    cache_path = _get_cache_path(cache_dir, cache_filename)
    
    cached_state = _load_cached_state(cache_path)
    if cached_state:
        print(f"[LAYER 1 CACHE HIT] Loaded from {cache_path}")
        print(f"  Retrieved samples: {len(cached_state.get('retrieved_set', []))}")
        print(f"  Generated candidates: {len(cached_state.get('candidate_set', {}))}")
        logger.info(f"Cache hit for Layer 1 state at {cache_path}")
        return cached_state
    
    print(f"[LAYER 1 CACHE MISS] Will execute full pipeline. Cache will be saved to {cache_path}")
    
    # --- INITIALIZE RESULT STRUCTURE ---
    layer1_state = {
        "metadata": {
            "query_index": target_query_index,
            "cache_version": "1.0",
            "timestamp": time.time(),
            "config_snapshot": {
                "TOP_N_CANDIDATES_RETRIEVAL": config.get("TOP_N_CANDIDATES_RETRIEVAL"),
                "LAYER1_N_CANDIDATES": config.get("LAYER1_N_CANDIDATES"),
                "MIRROR_N_OPTIMIZATION": config.get("MIRROR_N_OPTIMIZATION"),
                "LAYER1_DATASET_NAME": dataset_name
            }
        },
        "target_query_data": {
            "query_text": target_query,
            "ground_truth_answer": ground_truth_answer,
            "query_index": target_query_index
        },
        "retrieved_set": [],
        "candidate_set": {},
        "intrinsic_baselines": {},
        "cross_evaluation_matrix": {},
        "ground_truth_labels": {},
        "execution_trace": trace_accumulator,
        "step_statuses": {
            "retrieval": "PENDING",
            "candidate_generation": "PENDING",
            "baseline_calculation": "PENDING",
            "cross_evaluation": "PENDING",
            "ground_truth_evaluation": "PENDING"
        }
    }
    
    # --- STEP A: RETRIEVAL ---
    print("\n[Step A] Retrieval...")
    start_time = time.time()
    
    # Create question_to_index_map for O(1) self-match detection
    question_to_index_map = {q: i for i, q in enumerate(exemplar_questions)}
    
    retrieval_result = _execute_retrieval(
        target_query=target_query,
        embedding_model=embedding_model,
        exemplar_questions=exemplar_questions,
        embedded_exemplars=embedded_exemplars,
        top_k=top_k,
        question_to_index_map=question_to_index_map
    )
    
    layer1_state["step_statuses"]["retrieval"] = retrieval_result["status"]
    
    if retrieval_result["status"] == "FAILURE":
        logger.error(f"Retrieval failed: {retrieval_result.get('error')}")
        layer1_state["step_statuses"]["retrieval"] = "FAILURE"
        return layer1_state
    
    retrieved_indices = retrieval_result.get("retrieved_indices", [])
    print(f"  ✓ Retrieved {len(retrieved_indices)} exemplars in {time.time() - start_time:.2f}s")
    
    # Store retrieval data
    layer1_state["retrieved_set"] = retrieval_result.get("retrieval_data", [])
    layer1_state["execution_trace"].extend(retrieval_result.get("trace", []))
    
    # --- STEP B: CANDIDATE GENERATION ---
    print("\n[Step B] Candidate Generation (1-shot)...")
    start_time = time.time()
    
    candidate_result = _execute_candidate_generation(
        target_query=target_query,
        retrieved_indices=retrieved_indices,
        exemplar_data=exemplar_data,
        api_manager=api_manager,
        config=config,
        trace_accumulator=trace_accumulator
    )
    
    layer1_state["step_statuses"]["candidate_generation"] = candidate_result["status"]
    
    if candidate_result["status"] == "FAILURE":
        logger.error(f"Candidate generation failed: {candidate_result.get('error')}")
        # Continue anyway to attempt other steps
    
    candidates = candidate_result.get("candidates", {})
    print(f"  ✓ Generated {candidate_result.get('generated_count', 0)} candidates in {time.time() - start_time:.2f}s")
    
    # Store candidates
    layer1_state["candidate_set"] = candidates
    
    # --- STEP C: BASELINE CALCULATION ---
    print("\n[Step C] Intrinsic Baseline Calculation...")
    start_time = time.time()
    
    baseline_result = _execute_baseline_calculation(
        retrieved_indices=retrieved_indices,
        exemplar_data=exemplar_data,
        api_manager=api_manager,
        config=config,
        trace_accumulator=trace_accumulator
    )
    
    layer1_state["step_statuses"]["baseline_calculation"] = baseline_result["status"]
    
    if baseline_result["status"] == "FAILURE":
        logger.error(f"Baseline calculation failed: {baseline_result.get('error')}")
    
    baselines = baseline_result.get("intrinsic_baselines", {})
    print(f"  ✓ Calculated baselines for {len(baselines)} samples in {time.time() - start_time:.2f}s")
    
    # Store baselines
    layer1_state["intrinsic_baselines"] = baselines
    
    # --- STEP D: CROSS-EVALUATION MATRIX ---
    print("\n[Step D] Cross-Evaluation Matrix...")
    start_time = time.time()
    
    cross_eval_result = _execute_cross_evaluation(
        target_query=target_query,
        candidates=candidates,
        retrieved_indices=retrieved_indices,
        exemplar_data=exemplar_data,
        api_manager=api_manager,
        config=config,
        trace_accumulator=trace_accumulator
    )
    
    layer1_state["step_statuses"]["cross_evaluation"] = cross_eval_result["status"]
    
    if cross_eval_result["status"] == "FAILURE":
        logger.error(f"Cross-evaluation failed: {cross_eval_result.get('error')}")
    
    cross_matrix = cross_eval_result.get("cross_evaluation_matrix", {})
    print(f"  ✓ Cross-evaluated matrix in {time.time() - start_time:.2f}s")
    
    # Store cross-evaluation matrix
    layer1_state["cross_evaluation_matrix"] = cross_matrix
    
    # --- STEP E: GROUND-TRUTH EVALUATION ---
    print("\n[Step E] Ground-Truth Correctness Evaluation...")
    start_time = time.time()
    
    gt_result = _execute_ground_truth_evaluation(
        target_query=target_query,
        ground_truth_answer=ground_truth_answer,
        candidates=candidates,
        api_manager=api_manager,
        config=config
    )
    
    layer1_state["step_statuses"]["ground_truth_evaluation"] = gt_result["status"]
    
    if gt_result["status"] == "FAILURE":
        logger.error(f"Ground-truth evaluation failed: {gt_result.get('error')}")
    
    gt_labels = gt_result.get("ground_truth_labels", {})
    print(f"  ✓ Evaluated {gt_result.get('successful_evals', 0)} candidates against ground truth in {time.time() - start_time:.2f}s")
    
    # Store ground-truth labels
    layer1_state["ground_truth_labels"] = gt_labels
    
    # --- CACHE & FINALIZE ---
    print(f"\n[Layer 1] All steps completed. Saving cache...")
    
    # Determine overall status
    all_statuses = layer1_state["step_statuses"].values()
    if all(s == "SUCCESS" for s in all_statuses):
        layer1_state["overall_status"] = "SUCCESS"
        print("  ✓ All steps completed successfully")
    elif any(s == "SUCCESS" for s in all_statuses):
        layer1_state["overall_status"] = "PARTIAL"
        print("  ⚠ Some steps failed; check step_statuses for details")
    else:
        layer1_state["overall_status"] = "FAILURE"
        print("  ✗ Critical failure; check step_statuses for details")
    
    # Save cache
    success = _save_cached_state(cache_path, layer1_state)
    if success:
        print(f"  ✓ Cache saved to {cache_path}")
    else:
        logger.warning(f"Failed to save cache to {cache_path}")
    
    print(f"{'='*80}\n")
    
    return layer1_state
