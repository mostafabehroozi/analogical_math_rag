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
    top_k: int,
    n_candidates: int,
    experiment_name: str = "default_experiment"
) -> str:
    """
    Generates a deterministic cache filename based on configuration.
    Format: layer1_cache_k{k}_n{n}_{experiment_name}.json
    
    All queries are stored in a single combined file per experiment.
    """
    # Sanitize experiment name for filename
    safe_experiment_name = experiment_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return f"layer1_cache_k{top_k}_n{n_candidates}_{safe_experiment_name}.json"


def _get_cache_path(cache_dir: str, filename: str) -> str:
    """Returns the full path to a cache file."""
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, filename)


def _cache_exists(cache_path: str) -> bool:
    """Checks if a combined cache file exists and has valid structure."""
    if not os.path.exists(cache_path):
        return False
    try:
        data = load_json(cache_path)
        # Validate that it has the required combined structure
        return isinstance(data, dict) and "metadata" in data and "queries" in data
    except Exception as e:
        logging.getLogger(__name__).warning(f"Cache validation failed for {cache_path}: {e}")
        return False


def _load_cached_state(cache_path: str, target_query_index: int) -> Optional[Dict[str, Any]]:
    """
    Loads a cached Layer 1 state from the combined cache file for a specific query.
    
    Args:
        cache_path: Path to the combined cache file
        target_query_index: The query index to extract from the combined file
    
    Returns:
        The Layer 1 state for the specific query, or None if not found/invalid.
    """
    if not _cache_exists(cache_path):
        return None
    
    try:
        combined_data = load_json(cache_path)
        query_key = str(target_query_index)
        
        if query_key not in combined_data.get("queries", {}):
            return None
        
        query_state = combined_data["queries"][query_key]
        logging.getLogger(__name__).info(
            f"Successfully loaded Layer 1 cache for query #{target_query_index} from {cache_path}"
        )
        return query_state
    except Exception as e:
        logging.getLogger(__name__).error(
            f"Failed to load Layer 1 cache for query #{target_query_index} from {cache_path}: {e}"
        )
        return None


def _save_cached_state(
    cache_path: str,
    target_query_index: int,
    state: Dict[str, Any],
    top_k: int,
    n_candidates: int,
    experiment_name: str
) -> bool:
    """
    Saves a Layer 1 state to the combined cache file for this query.
    Merges with existing data if the file already exists.
    Uses atomic writes (temp file + rename) to prevent corruption.
    
    Args:
        cache_path: Path to the combined cache file
        target_query_index: The query index to store
        state: The Layer 1 state for this query
        top_k: Top-k retrieval setting
        n_candidates: Number of candidates setting
        experiment_name: Name of the experiment
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        query_key = str(target_query_index)
        
        # Load existing combined file if it exists
        if os.path.exists(cache_path):
            combined_data = load_json(cache_path)
        else:
            # Create new combined structure
            combined_data = {
                "metadata": {
                    "created_at": time.time(),
                    "experiment_name": experiment_name,
                    "top_k": top_k,
                    "n_candidates": n_candidates,
                    "config_snapshot": {},
                    "completed_queries": [],
                    "queries_in_progress": []
                },
                "queries": {}
            }
        
        # Add/update this query's data
        combined_data["queries"][query_key] = state
        
        # Update metadata timestamp (latest update)
        combined_data["metadata"]["updated_at"] = time.time()
        combined_data["metadata"]["total_queries"] = len(combined_data["queries"])
        
        # Atomic write: write to temp file, then rename
        temp_path = cache_path + ".tmp"
        save_json(combined_data, temp_path)
        os.replace(temp_path, cache_path)  # Atomic rename operation
        
        logging.getLogger(__name__).info(
            f"Successfully saved Layer 1 cache for query #{target_query_index} to {cache_path} "
            f"(total queries: {len(combined_data['queries'])})"
        )
        return True
    except Exception as e:
        logging.getLogger(__name__).error(
            f"Failed to save Layer 1 cache for query #{target_query_index} to {cache_path}: {e}"
        )
        # Clean up temp file if it exists
        try:
            if os.path.exists(cache_path + ".tmp"):
                os.remove(cache_path + ".tmp")
        except:
            pass
        return False


def _save_step_checkpoint(
    cache_path: str,
    target_query_index: int,
    step_name: str,
    step_data: Dict[str, Any],
    top_k: int,
    n_candidates: int,
    experiment_name: str
) -> bool:
    """
    Saves a checkpoint for a single completed step atomically.
    This enables resuming from this exact point if a kernel crash occurs.
    
    Args:
        cache_path: Path to the combined cache file
        target_query_index: The query index
        step_name: Name of the step ("retrieval", "candidate_generation", etc.)
        step_data: Data to checkpoint for this step
        top_k: Top-k retrieval setting
        n_candidates: Number of candidates setting
        experiment_name: Name of the experiment
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        query_key = str(target_query_index)
        step_order = {
            "retrieval": 0,
            "candidate_generation": 1,
            "baseline_calculation": 2,
            "cross_evaluation": 3,
            "ground_truth_evaluation": 4
        }
        
        # Load existing combined file
        if os.path.exists(cache_path):
            combined_data = load_json(cache_path)
        else:
            combined_data = {
                "metadata": {
                    "created_at": time.time(),
                    "experiment_name": experiment_name,
                    "top_k": top_k,
                    "n_candidates": n_candidates,
                    "completed_queries": [],
                    "queries_in_progress": []
                },
                "queries": {}
            }
        
        # Initialize query if needed
        if query_key not in combined_data["queries"]:
            combined_data["queries"][query_key] = {
                "metadata": {
                    "query_index": target_query_index,
                    "cache_version": "3.0",
                    "last_checkpoint_timestamp": time.time(),
                    "last_completed_step": -1
                },
                "step_checkpoints": {},
                "step_statuses": {}
            }
        
        # Ensure metadata has the new fields
        if "last_completed_step" not in combined_data["queries"][query_key]["metadata"]:
            combined_data["queries"][query_key]["metadata"]["last_completed_step"] = -1
        if "step_checkpoints" not in combined_data["queries"][query_key]:
            combined_data["queries"][query_key]["step_checkpoints"] = {}
        if "step_statuses" not in combined_data["queries"][query_key]:
            combined_data["queries"][query_key]["step_statuses"] = {}
        
        # Save step checkpoint
        combined_data["queries"][query_key]["step_checkpoints"][step_name] = {
            "status": "SUCCESS",
            "data": step_data,
            "completed_at": time.time()
        }
        combined_data["queries"][query_key]["step_statuses"][step_name] = "SUCCESS"
        combined_data["queries"][query_key]["metadata"]["last_completed_step"] = step_order[step_name]
        combined_data["queries"][query_key]["metadata"]["last_checkpoint_timestamp"] = time.time()
        
        # Update global metadata
        completed_queries = combined_data["metadata"].get("completed_queries", [])
        queries_in_progress = combined_data["metadata"].get("queries_in_progress", [])
        
        # Remove from in-progress if adding to completed (happens when all 5 steps done)
        if target_query_index not in queries_in_progress:
            queries_in_progress.append(target_query_index)
        
        combined_data["metadata"]["queries_in_progress"] = queries_in_progress
        combined_data["metadata"]["updated_at"] = time.time()
        combined_data["metadata"]["total_queries"] = len(combined_data["queries"])
        
        # Atomic write: temp file + rename
        temp_path = cache_path + ".tmp"
        save_json(combined_data, temp_path)
        os.replace(temp_path, cache_path)
        
        logging.getLogger(__name__).debug(
            f"Step checkpoint saved: Query #{target_query_index}, Step {step_name} "
            f"(completed_step_index: {step_order[step_name]})"
        )
        return True
    except Exception as e:
        logging.getLogger(__name__).error(
            f"Failed to save step checkpoint for {step_name} (query #{target_query_index}): {e}"
        )
        try:
            if os.path.exists(cache_path + ".tmp"):
                os.remove(cache_path + ".tmp")
        except:
            pass
        return False


def _get_last_completed_step(cache_path: str, target_query_index: int) -> int:
    """
    Returns the index of the last completed step for a query.
    
    Returns:
        -1 if query not found or no steps completed
        0-4 corresponding to the last step completed (retrieval=0, candidate_gen=1, etc.)
    
    Step order: 0=retrieval, 1=candidate_generation, 2=baseline_calculation, 
                3=cross_evaluation, 4=ground_truth_evaluation
    """
    try:
        if not _cache_exists(cache_path):
            return -1
        
        combined_data = load_json(cache_path)
        query_key = str(target_query_index)
        
        if query_key not in combined_data.get("queries", {}):
            return -1
        
        query_data = combined_data["queries"][query_key]
        last_step = query_data.get("metadata", {}).get("last_completed_step", -1)
        return last_step
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"Could not get last completed step for query #{target_query_index}: {e}"
        )
        return -1


def _load_step_checkpoint(
    cache_path: str,
    target_query_index: int,
    step_name: str
) -> Optional[Dict[str, Any]]:
    """
    Loads previously saved checkpoint data for a specific step.
    
    Args:
        cache_path: Path to the combined cache file
        target_query_index: The query index
        step_name: Name of the step to load
    
    Returns:
        The checkpoint data dict, or None if not found/invalid.
    """
    try:
        if not _cache_exists(cache_path):
            return None
        
        combined_data = load_json(cache_path)
        query_key = str(target_query_index)
        
        if query_key not in combined_data.get("queries", {}):
            return None
        
        query_data = combined_data["queries"][query_key]
        checkpoints = query_data.get("step_checkpoints", {})
        
        if step_name not in checkpoints:
            return None
        
        checkpoint = checkpoints[step_name]
        return checkpoint.get("data")
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"Could not load checkpoint for {step_name} (query #{target_query_index}): {e}"
        )
        return None


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
        retrieved_scores = retrieval_result.get("retrieved_similarity_scores", [])
        
        # Collect the actual exemplar data for storage
        retrieval_data = []
        
        # Use the exemplar data from the function parameter
        try:
            for idx_position, idx in enumerate(retrieved_indices):
                similarity_score = None
                if idx_position < len(retrieved_scores):
                    similarity_score = float(retrieved_scores[idx_position])
                retrieval_data.append({
                    "corpus_index": int(idx),
                    "question": exemplar_questions[idx] if idx < len(exemplar_questions) else "",
                    "similarity_score": similarity_score
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
    config: Dict[str, Any],
    experiment_name: str = "default_experiment"
) -> Dict[str, Any]:
    """
    Main entry point for Layer 1: Base Execution Phase with RESUMABLE checkpointing.
    
    Orchestrates the complete data-gathering pipeline with step-level recovery:
    1. Check for cached state in combined file (Cache-First Rule)
    2. Detect if query is FULLY cached (all 5 steps done) → Return immediately
    3. Detect if query is PARTIALLY cached → Resume from last step
    4. If cache miss → Execute Steps A-E with checkpoint saves after each step
    5. Merge result into combined JSON cache file
    6. Return the cached/computed state
    
    ✅ NEW: Step-level checkpointing enables resumption after kernel crashes
    
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
        experiment_name: Name of the experiment (for combined cache filename)
    
    Returns:
        The complete Layer 1 cached state (dict) with all phases populated
    """
    logger = logging.getLogger(__name__)
    
    # --- Initialization ---
    cache_dir = config.get("LAYER1_CACHE_DIR", config.get("RESULTS_DIR", "local_data/outputs/results"))
    top_k = config.get("TOP_N_CANDIDATES_RETRIEVAL", 5)
    n_candidates = config.get("LAYER1_N_CANDIDATES")
    if n_candidates is None:
        n_candidates = top_k  # Use top_k if LAYER1_N_CANDIDATES not explicitly set
    
    trace_accumulator = []
    
    print(f"\n{'='*80}")
    print(f"[LAYER 1 - BASE EXECUTION] Processing Query #{target_query_index}")
    print(f"Query: {target_query[:80]}...")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*80}")
    
    logger.info(f"[Layer 1] Starting base execution for Query #{target_query_index} (Experiment: {experiment_name})")
    
    # --- CACHE-FIRST CHECK (from combined file) ---
    cache_filename = _get_cache_filename(top_k, n_candidates, experiment_name)
    cache_path = _get_cache_path(cache_dir, cache_filename)
    
    # Check for FULLY CACHED query
    cached_state = _load_cached_state(cache_path, target_query_index)
    if cached_state:
        # Verify that ALL steps are complete (last_completed_step == 4)
        last_step = cached_state.get("metadata", {}).get("last_completed_step", -1)
        if last_step == 4:  # All 5 steps complete (0-4)
            print(f"[LAYER 1 CACHE HIT - FULL] Loaded query #{target_query_index} from {cache_path}")
            print(f"  All 5 steps cached")
            print(f"  Retrieved samples: {len(cached_state.get('retrieved_set', []))}")
            print(f"  Generated candidates: {len(cached_state.get('candidate_set', {}))}")
            logger.info(f"Full cache hit for Layer 1 state (query #{target_query_index}) at {cache_path}")
            return cached_state
        else:
            # PARTIAL CACHE: Resume from last completed step
            print(f"[LAYER 1 CACHE HIT - PARTIAL] Found incomplete query #{target_query_index}")
            last_step_names = ["retrieval", "candidate_generation", "baseline_calculation", 
                               "cross_evaluation", "ground_truth_evaluation"]
            print(f"  Last completed step: {last_step_names[last_step + 1] if last_step >= 0 else 'none'}")
            print(f"  Will resume from step {last_step_names[last_step + 2] if last_step + 1 < 5 else 'complete'}")
    else:
        print(f"[LAYER 1 CACHE MISS] Query #{target_query_index} will execute full pipeline.")
        last_step = -1
        cached_state = None
    
    print(f"  Results will be merged into: {cache_path}")
    
    # --- GET LAST COMPLETED STEP (for resumption) ---
    last_completed_step = _get_last_completed_step(cache_path, target_query_index)
    
    # --- INITIALIZE RESULT STRUCTURE ---
    if cached_state:
        layer1_state = cached_state
    else:
        layer1_state = {
            "metadata": {
                "query_index": target_query_index,
                "cache_version": "3.0",  # ← BUMPED VERSION
                "timestamp": time.time(),
                "last_checkpoint_timestamp": time.time(),
                "last_completed_step": -1,
                "config_snapshot": {
                    "TOP_N_CANDIDATES_RETRIEVAL": config.get("TOP_N_CANDIDATES_RETRIEVAL"),
                    "LAYER1_N_CANDIDATES": config.get("LAYER1_N_CANDIDATES"),
                    "MIRROR_N_OPTIMIZATION": config.get("MIRROR_N_OPTIMIZATION")
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
            },
            "step_checkpoints": {}
        }
    
    # --- STEP A: RETRIEVAL ---
    if last_completed_step < 0:  # Only execute if not already done
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
        
        # ✅ SAVE CHECKPOINT IMMEDIATELY AFTER STEP A
        checkpoint_success = _save_step_checkpoint(
            cache_path=cache_path,
            target_query_index=target_query_index,
            step_name="retrieval",
            step_data={
                "retrieved_set": layer1_state["retrieved_set"],
                "retrieved_indices": retrieved_indices,
                "execution_trace": layer1_state["execution_trace"]
            },
            top_k=top_k,
            n_candidates=n_candidates,
            experiment_name=experiment_name
        )
        print(f"  ✓ Checkpoint saved (resumable from this point)" if checkpoint_success else "  ⚠ Checkpoint save failed")
    else:
        # ✅ RESUME: Load from checkpoint
        print("\n[Step A] RESUMING from checkpoint...")
        checkpoint_data = _load_step_checkpoint(cache_path, target_query_index, "retrieval")
        if checkpoint_data:
            layer1_state["retrieved_set"] = checkpoint_data.get("retrieved_set", [])
            retrieved_indices = checkpoint_data.get("retrieved_indices", [])
            layer1_state["execution_trace"] = checkpoint_data.get("execution_trace", [])
            print(f"  ✓ Loaded retrieval results from checkpoint ({len(retrieved_indices)} exemplars)")
        else:
            print(f"  ⚠ Could not load retrieval checkpoint, re-executing")
            # Fall back to executing
            question_to_index_map = {q: i for i, q in enumerate(exemplar_questions)}
            retrieval_result = _execute_retrieval(
                target_query=target_query,
                embedding_model=embedding_model,
                exemplar_questions=exemplar_questions,
                embedded_exemplars=embedded_exemplars,
                top_k=top_k,
                question_to_index_map=question_to_index_map
            )
            layer1_state["retrieved_set"] = retrieval_result.get("retrieval_data", [])
            retrieved_indices = retrieval_result.get("retrieved_indices", [])
    
    # --- STEP B: CANDIDATE GENERATION ---
    if last_completed_step < 1:  # Only execute if not already done
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
        
        # ✅ SAVE CHECKPOINT AFTER STEP B
        checkpoint_success = _save_step_checkpoint(
            cache_path=cache_path,
            target_query_index=target_query_index,
            step_name="candidate_generation",
            step_data={
                "candidate_set": candidates,
                "generated_count": candidate_result.get('generated_count', 0),
                "failed_count": candidate_result.get('failed_count', 0)
            },
            top_k=top_k,
            n_candidates=n_candidates,
            experiment_name=experiment_name
        )
        print(f"  ✓ Checkpoint saved" if checkpoint_success else "  ⚠ Checkpoint save failed")
    else:
        # ✅ RESUME: Load from checkpoint
        print("\n[Step B] RESUMING from checkpoint...")
        checkpoint_data = _load_step_checkpoint(cache_path, target_query_index, "candidate_generation")
        if checkpoint_data:
            candidates = checkpoint_data.get("candidate_set", {})
            print(f"  ✓ Loaded {len(candidates)} candidates from checkpoint")
        else:
            print(f"  ⚠ Could not load candidate checkpoint, re-executing")
            candidate_result = _execute_candidate_generation(
                target_query=target_query,
                retrieved_indices=retrieved_indices,
                exemplar_data=exemplar_data,
                api_manager=api_manager,
                config=config,
                trace_accumulator=trace_accumulator
            )
            candidates = candidate_result.get("candidates", {})
    
    # --- STEP C: BASELINE CALCULATION ---
    if last_completed_step < 2:  # Only execute if not already done
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
        
        # ✅ SAVE CHECKPOINT AFTER STEP C
        checkpoint_success = _save_step_checkpoint(
            cache_path=cache_path,
            target_query_index=target_query_index,
            step_name="baseline_calculation",
            step_data={
                "intrinsic_baselines": baselines
            },
            top_k=top_k,
            n_candidates=n_candidates,
            experiment_name=experiment_name
        )
        print(f"  ✓ Checkpoint saved" if checkpoint_success else "  ⚠ Checkpoint save failed")
    else:
        # ✅ RESUME: Load from checkpoint
        print("\n[Step C] RESUMING from checkpoint...")
        checkpoint_data = _load_step_checkpoint(cache_path, target_query_index, "baseline_calculation")
        if checkpoint_data:
            baselines = checkpoint_data.get("intrinsic_baselines", {})
            layer1_state["intrinsic_baselines"] = baselines
            print(f"  ✓ Loaded baselines for {len(baselines)} samples from checkpoint")
        else:
            print(f"  ⚠ Could not load baseline checkpoint, re-executing")
            baseline_result = _execute_baseline_calculation(
                retrieved_indices=retrieved_indices,
                exemplar_data=exemplar_data,
                api_manager=api_manager,
                config=config,
                trace_accumulator=trace_accumulator
            )
            baselines = baseline_result.get("intrinsic_baselines", {})
    
    # --- STEP D: CROSS-EVALUATION MATRIX ---
    if last_completed_step < 3:  # Only execute if not already done
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
        
        # ✅ SAVE CHECKPOINT AFTER STEP D
        checkpoint_success = _save_step_checkpoint(
            cache_path=cache_path,
            target_query_index=target_query_index,
            step_name="cross_evaluation",
            step_data={
                "cross_evaluation_matrix": cross_matrix,
                "matrix_dimensions": cross_eval_result.get("matrix_dimensions", {})
            },
            top_k=top_k,
            n_candidates=n_candidates,
            experiment_name=experiment_name
        )
        print(f"  ✓ Checkpoint saved" if checkpoint_success else "  ⚠ Checkpoint save failed")
    else:
        # ✅ RESUME: Load from checkpoint
        print("\n[Step D] RESUMING from checkpoint...")
        checkpoint_data = _load_step_checkpoint(cache_path, target_query_index, "cross_evaluation")
        if checkpoint_data:
            cross_matrix = checkpoint_data.get("cross_evaluation_matrix", {})
            layer1_state["cross_evaluation_matrix"] = cross_matrix
            dims = checkpoint_data.get("matrix_dimensions", {})
            print(f"  ✓ Loaded cross-evaluation matrix from checkpoint "
                  f"({dims.get('candidates', 0)} x {dims.get('evaluators', 0)})")
        else:
            print(f"  ⚠ Could not load cross-evaluation checkpoint, re-executing")
            cross_eval_result = _execute_cross_evaluation(
                target_query=target_query,
                candidates=candidates,
                retrieved_indices=retrieved_indices,
                exemplar_data=exemplar_data,
                api_manager=api_manager,
                config=config,
                trace_accumulator=trace_accumulator
            )
            cross_matrix = cross_eval_result.get("cross_evaluation_matrix", {})
    
    # --- STEP E: GROUND-TRUTH EVALUATION ---
    if last_completed_step < 4:  # Only execute if not already done
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
        
        # ✅ SAVE CHECKPOINT AFTER STEP E (FINAL)
        checkpoint_success = _save_step_checkpoint(
            cache_path=cache_path,
            target_query_index=target_query_index,
            step_name="ground_truth_evaluation",
            step_data={
                "ground_truth_labels": gt_labels,
                "successful_evals": gt_result.get('successful_evals', 0),
                "failed_evals": gt_result.get('failed_evals', 0)
            },
            top_k=top_k,
            n_candidates=n_candidates,
            experiment_name=experiment_name
        )
        print(f"  ✓ Checkpoint saved (query complete!)" if checkpoint_success else "  ⚠ Checkpoint save failed")
    else:
        # ✅ RESUME: Load from checkpoint
        print("\n[Step E] RESUMING from checkpoint...")
        checkpoint_data = _load_step_checkpoint(cache_path, target_query_index, "ground_truth_evaluation")
        if checkpoint_data:
            gt_labels = checkpoint_data.get("ground_truth_labels", {})
            layer1_state["ground_truth_labels"] = gt_labels
            successful = checkpoint_data.get('successful_evals', 0)
            print(f"  ✓ Loaded {successful} ground-truth evaluations from checkpoint")
        else:
            print(f"  ⚠ Could not load ground-truth checkpoint, re-executing")
            gt_result = _execute_ground_truth_evaluation(
                target_query=target_query,
                ground_truth_answer=ground_truth_answer,
                candidates=candidates,
                api_manager=api_manager,
                config=config
            )
            gt_labels = gt_result.get("ground_truth_labels", {})
    
    # --- CACHE & FINALIZE ---
    print(f"\n[Layer 1] All steps completed. Saving final state to combined cache...")
    
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
    
    # Save complete state to cache (merges with existing queries)
    success = _save_cached_state(
        cache_path=cache_path,
        target_query_index=target_query_index,
        state=layer1_state,
        top_k=top_k,
        n_candidates=n_candidates,
        experiment_name=experiment_name
    )
    
    if success:
        print(f"  ✓ Final state merged into {cache_path}")
        print(f"    (All queries for this experiment stored in one combined file)")
    else:
        logger.warning(f"Failed to save final cache to {cache_path}")
    
    print(f"{'='*80}\n")
    
    return layer1_state
