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


def _save_cached_states_batch(
    cache_path: str,
    states_by_query_index: Dict[int, Dict[str, Any]],
    top_k: int,
    n_candidates: int,
    experiment_name: str,
    batch_id: Optional[str] = None,
) -> bool:
    """Merge completed Layer-1 states and persist them with one atomic write.

    Worker threads must return their states to the batch coordinator; this helper
    is deliberately called only after every question in that batch is terminal.
    """
    if not states_by_query_index:
        return True
    try:
        combined_data = load_json(cache_path) if os.path.exists(cache_path) else None
        if not isinstance(combined_data, dict):
            combined_data = {
                "metadata": {
                    "created_at": time.time(), "experiment_name": experiment_name,
                    "top_k": top_k, "n_candidates": n_candidates,
                    "config_snapshot": {}, "completed_queries": [], "queries_in_progress": [],
                },
                "queries": {},
            }
        combined_data.setdefault("queries", {})
        metadata = combined_data.setdefault("metadata", {})
        completed = set(metadata.get("completed_queries", []))
        in_progress = set(metadata.get("queries_in_progress", []))
        for query_index, state in states_by_query_index.items():
            combined_data["queries"][str(query_index)] = state
            if isinstance(state, dict) and state.get("_needs_saving"):
                state.pop("_needs_saving", None)
            # Incomplete Layer-1 state stays resumable but is still included in
            # the coordinator's atomic batch artifact.
            statuses = state.get("step_statuses", {}) if isinstance(state, dict) else {}
            if statuses and all(value == "SUCCESS" for value in statuses.values()):
                completed.add(query_index)
                in_progress.discard(query_index)
            else:
                in_progress.add(query_index)
        metadata.update({
            "experiment_name": experiment_name,
            "top_k": top_k,
            "n_candidates": n_candidates,
            "updated_at": time.time(),
            "total_queries": len(combined_data["queries"]),
            "completed_queries": sorted(completed),
            "queries_in_progress": sorted(in_progress),
        })
        if batch_id:
            metadata["last_committed_batch_id"] = batch_id
        temp_path = cache_path + ".tmp"
        if not save_json(combined_data, temp_path):
            return False
        os.replace(temp_path, cache_path)
        logging.getLogger(__name__).info(
            "Saved %s Layer-1 states in committed batch %s to %s",
            len(states_by_query_index), batch_id or "<legacy>", cache_path,
        )
        return True
    except Exception as e:
        logging.getLogger(__name__).error("Failed batch Layer-1 cache save to %s: %s", cache_path, e, exc_info=True)
        return False
    finally:
        temp_path = cache_path + ".tmp"
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


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
    DISABLED FOR BATCH PROCESSING to prevent disk race conditions.
    The threads hold their state in memory, and the Batch Coordinator 
    saves everything atomically at the end of the batch.
    """
    # Simply return True to pretend it saved, allowing the pipeline to continue
    return True


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
    Generates exactly one candidate per retrieved exemplar (1-shot)
    AND optionally generates N Zero-Shot candidates.
    """
    logger = logging.getLogger(__name__)
    zs_n = config.get("LAYER1_ZERO_SHOT_CANDIDATES_N", 0)
    logger.info(f"[Layer 1 - Step B] Generating {len(retrieved_indices)} 1-shot candidates and {zs_n} zero-shot candidates")
    
    candidates = {}
    failed_count = 0
    
    try:
        # 1. Standard 1-Shot Generation
        hypotheses = _generate_hypotheses(
            target_query=target_query,
            candidate_indices=retrieved_indices,
            exemplar_data=exemplar_data,
            api_manager=api_manager,
            config=config,
            trace_accumulator=trace_accumulator
        )
        
        # USE ORIGINAL INTEGER KEYS FOR BACKWARD COMPATIBILITY
        for exemplar_idx, hypothesis_text in hypotheses.items():
            if hypothesis_text:
                candidates[exemplar_idx] = {
                    "candidate_id": exemplar_idx,
                    "candidate_text": hypothesis_text,
                    "source_exemplar_idx": exemplar_idx,
                    "generation_status": "SUCCESS"
                }
            else:
                candidates[exemplar_idx] = {
                    "candidate_id": exemplar_idx,
                    "candidate_text": None,
                    "source_exemplar_idx": exemplar_idx,
                    "generation_status": "FAILURE"
                }
                failed_count += 1
        
        # 2. NEW: Zero-Shot Candidate Generation
        if zs_n > 0:
            from src.prompts import PROMPT_TEMPLATES
            if isinstance(api_manager, GeminiAPIManager): model_name = config.get('GEMINI_MODEL_NAME_FINAL_SOLVER')
            elif isinstance(api_manager, AvalAIAPIManager): model_name = config.get('AVALAI_MODEL_NAME_FINAL_SOLVER')
            else: model_name = config.get('OLLAMA_MODEL_NAME_FINAL_SOLVER')
            
            tmpl_name = config.get("PROMPT_TEMPLATE_MIRROR_HYPOTHESIS_ZEROSHOT", "mirror_hypothesis_gen_zero_shot_v1")
            tmpl_zero = PROMPT_TEMPLATES.get(tmpl_name, "{target_query}")
            prompt = tmpl_zero.format(target_query=target_query)
            
            for i in range(zs_n):
                zs_id = f"zs_{i}"
                resp = api_manager.generate_content(prompt, model_name, temperature=0.7)
                
                trace_accumulator.append(create_trace_entry(
                    "layer1_candidate_gen", f"zero_shot_{i}",
                    {"prompt": prompt}, resp, {"model": model_name, "temp": 0.7}
                ))
                
                if resp['status'] == 'SUCCESS' and resp['text']:
                    candidates[zs_id] = {
                        "candidate_id": zs_id,
                        "candidate_text": resp['text'],
                        "source_exemplar_idx": -1, # Special flag for No Parent
                        "generation_status": "SUCCESS"
                    }
                else:
                    candidates[zs_id] = {
                        "candidate_id": zs_id,
                        "candidate_text": None,
                        "source_exemplar_idx": -1,
                        "generation_status": "FAILURE"
                    }
                    failed_count += 1
        
        total_expected = len(retrieved_indices) + zs_n
        status = "SUCCESS" if failed_count == 0 else ("PARTIAL" if failed_count < total_expected else "FAILURE")
        
        return {
            "status": status,
            "candidates": candidates,
            "generated_count": total_expected - failed_count,
            "failed_count": failed_count
        }
    
    except Exception as e:
        logger.error(f"Candidate generation failed: {e}", exc_info=True)
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
    experiment_name: str = "default_experiment",
    force_reexecution: bool = False
) -> Dict[str, Any]:
    """
    Main entry point for Layer 1: Base Execution Phase.
    MODIFIED: Now executes entirely in memory and returns the state without saving.
    Saving is handled synchronously by the Orchestrator.
    """
    logger = logging.getLogger(__name__)
    
    # --- Initialization ---
    cache_dir = config.get("LAYER1_CACHE_DIR", config.get("RESULTS_DIR", "local_data/outputs/results"))
    top_k = config.get("TOP_N_CANDIDATES_RETRIEVAL", 5)
    n_candidates = config.get("LAYER1_N_CANDIDATES")
    if n_candidates is None:
        n_candidates = top_k
    
    trace_accumulator = []
    
    print(f"\n{'='*80}")
    print(f"[LAYER 1 - BASE EXECUTION] Processing Query #{target_query_index}")
    print(f"Query: {target_query[:80]}...")
    print(f"Experiment: {experiment_name}")
    if force_reexecution:
        print(f"🔄 FORCE REEXECUTION MODE - Bypassing cache")
    print(f"{'='*80}")
    
    logger.info(f"[Layer 1] Starting base execution for Query #{target_query_index} (Experiment: {experiment_name})")
    
    # --- CACHE-FIRST CHECK ---
    cache_filename = _get_cache_filename(top_k, n_candidates, experiment_name)
    cache_path = _get_cache_path(cache_dir, cache_filename)
    
    cached_state = None
    if not force_reexecution:
        cached_state = _load_cached_state(cache_path, target_query_index)
        if cached_state:
            last_step = cached_state.get("metadata", {}).get("last_completed_step", -1)
            if last_step == 4:  # Full Cache Hit
                print(f"[LAYER 1 CACHE HIT - FULL] Loaded query #{target_query_index} from {cache_path}")
                logger.info(f"Full cache hit for Layer 1 state (query #{target_query_index})")
                return cached_state
            else:
                print(f"[LAYER 1 CACHE MISS / INCOMPLETE] Query #{target_query_index} will execute full pipeline in memory.")
                cached_state = None
    
    # --- INITIALIZE RESULT STRUCTURE ---
    layer1_state = {
        "metadata": {
            "query_index": target_query_index,
            "cache_version": "3.0",
            "timestamp": time.time(),
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
        }
    }
    
    # --- STEP A: RETRIEVAL ---
    print("\n[Step A] Retrieval...")
    start_time = time.time()
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
        return layer1_state
        
    retrieved_indices = retrieval_result.get("retrieved_indices", [])
    print(f"  ✓ Retrieved {len(retrieved_indices)} exemplars in {time.time() - start_time:.2f}s")
    layer1_state["retrieved_set"] = retrieval_result.get("retrieval_data", [])
    layer1_state["execution_trace"].extend(retrieval_result.get("trace", []))
    
    # --- STEP B: CANDIDATE GENERATION ---
    print("\n[Step B] Candidate Generation (1-shot)...")
    start_time = time.time()
    candidate_result = _execute_candidate_generation(
        target_query=target_query, retrieved_indices=retrieved_indices,
        exemplar_data=exemplar_data, api_manager=api_manager, config=config, trace_accumulator=trace_accumulator
    )
    layer1_state["step_statuses"]["candidate_generation"] = candidate_result["status"]
    candidates = candidate_result.get("candidates", {})
    print(f"  ✓ Generated {candidate_result.get('generated_count', 0)} candidates in {time.time() - start_time:.2f}s")
    layer1_state["candidate_set"] = candidates
    
    # --- STEP C: BASELINE CALCULATION ---
    print("\n[Step C] Intrinsic Baseline Calculation...")
    start_time = time.time()
    baseline_result = _execute_baseline_calculation(
        retrieved_indices=retrieved_indices, exemplar_data=exemplar_data,
        api_manager=api_manager, config=config, trace_accumulator=trace_accumulator
    )
    layer1_state["step_statuses"]["baseline_calculation"] = baseline_result["status"]
    baselines = baseline_result.get("intrinsic_baselines", {})
    print(f"  ✓ Calculated baselines for {len(baselines)} samples in {time.time() - start_time:.2f}s")
    layer1_state["intrinsic_baselines"] = baselines
    
    # --- STEP D: CROSS-EVALUATION MATRIX ---
    print("\n[Step D] Cross-Evaluation Matrix...")
    start_time = time.time()
    cross_eval_result = _execute_cross_evaluation(
        target_query=target_query, candidates=candidates, retrieved_indices=retrieved_indices,
        exemplar_data=exemplar_data, api_manager=api_manager, config=config, trace_accumulator=trace_accumulator
    )
    layer1_state["step_statuses"]["cross_evaluation"] = cross_eval_result["status"]
    layer1_state["cross_evaluation_matrix"] = cross_eval_result.get("cross_evaluation_matrix", {})
    print(f"  ✓ Cross-evaluated matrix in {time.time() - start_time:.2f}s")
    
    # --- STEP E: GROUND-TRUTH EVALUATION ---
    print("\n[Step E] Ground-Truth Correctness Evaluation...")
    start_time = time.time()
    gt_result = _execute_ground_truth_evaluation(
        target_query=target_query, ground_truth_answer=ground_truth_answer,
        candidates=candidates, api_manager=api_manager, config=config
    )
    layer1_state["step_statuses"]["ground_truth_evaluation"] = gt_result["status"]
    layer1_state["ground_truth_labels"] = gt_result.get("ground_truth_labels", {})
    print(f"  ✓ Evaluated {gt_result.get('successful_evals', 0)} candidates against ground truth in {time.time() - start_time:.2f}s")
    
# --- FINALIZE ---
    all_statuses = layer1_state["step_statuses"].values()
    if all(s == "SUCCESS" for s in all_statuses):
        layer1_state["overall_status"] = "SUCCESS"
        layer1_state["metadata"]["last_completed_step"] = 4
        print("  ✓ All steps completed successfully in memory.")
    elif any(s == "SUCCESS" for s in all_statuses):
        layer1_state["overall_status"] = "PARTIAL"
        print("  ⚠ Some steps failed; check step_statuses for details")
    else:
        layer1_state["overall_status"] = "FAILURE"
        print("  ✗ Critical failure; check step_statuses for details")
    
    # NEW: Always flag for saving if we executed the pipeline (Success or Failure)
    layer1_state["_needs_saving"] = True 
    
    print(f"{'='*80}\n")
    
    # Return the dictionary to the orchestrator WITHOUT saving it to disk here.
    return layer1_state
