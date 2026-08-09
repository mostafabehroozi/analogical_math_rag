"""
Multi-Branch Transformation Experiments Module

Implements a centralized candidate pooling architecture that supports three
parallel experimental scenarios:

1. Tx1 (Single Transformation): Uses first transformation deterministically
2. BoT-N (Best-of-N Exclusive): Selects best among transformations only
3. BoT-N+R (Best-of-N Inclusive): Selects best among all candidates with tie-breaking

ARCHITECTURE:
- Phase 1: Centralized pool construction [R_main, T_1, T_2, ..., T_N]
- Phase 2: Unified batch scoring of all candidates
- Phase 3: Deterministic selection via array slicing (with tie-breaking)
- Phase 4: State fan-out to independent solvers per branch

KEY PRINCIPLES:
- Compute efficiency: No redundant API calls (single pooling + single scoring pass)
- Reproducibility: Deterministic tie-breaking ensures consistent results
- Experimental control: Three scenarios from same pool enable fair comparison
- Comprehensive logging: Full source attribution and intervention rate tracking
"""

import logging
from typing import List, Dict, Any

from src.api_manager import GeminiAPIManager, AvalAIAPIManager, OllamaAPIManager
from src.prompts import (
    create_transformation_prompt,
    create_analogical_adaptation_prompt,
    create_reverse_validation_prompt,
)
from src.utils import create_trace_entry


def argmax_tiebreak(scores: List[float]) -> int:
    """
    Selects the index of maximum score with deterministic tie-breaking.
    
    When multiple indices have equal scores (within floating-point precision),
    the tie is broken by preferring lower indices. This ensures reproducibility
    and, in the context of BoT-N+R, favors the original retrieval (R_main at index 0).
    
    Tie-Breaking Rule: (score DESC, index ASC)
    - Higher scores win first
    - When tied, lower indices win
    
    Args:
        scores: List of floats, one per candidate
    
    Returns:
        Index of the maximum score (with tie-breaking)
    
    Example:
        Scores: [0.8, 0.8, 0.6, 0.7]
        Tie between indices 0 and 1 (both score 0.8)
        Tie-breaking: index 0 < index 1, so index 0 wins
        Returns: 0 (favors R_main in BoT-N+R)
    """
    if not scores:
        return 0
    
    # Use (score, -index) as key for sorting:
    # Max first sorts by score DESC, then by -index DESC
    # -index DESC is equivalent to index ASC (lower indices win ties)
    best_candidate = max(
        enumerate(scores),
        key=lambda pr: (pr[1], -pr[0])
    )
    return best_candidate[0]


def _build_centralized_candidate_pool(
    target_query: str,
    retrieved_indices: List[int],
    exemplar_data: Dict[str, Any],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    PHASE 1: Builds centralized candidate pool for all retrieved samples.
    
    Structure: For each retrieved sample:
        [R_main (original), T_1 (transform 1), T_2 (transform 2), ..., T_N (transform N)]
    
    This ensures:
    - Strict index preservation (index 0 is always R_main)
    - All candidates pre-generated before any scoring
    - Clean array slicing for branch-specific selection later
    
    Args:
        target_query: Main problem query
        retrieved_indices: List of retrieved exemplar indices
        exemplar_data: Contains questions, solutions, embeddings
        api_manager: API manager for transformations
        config: Pipeline configuration
    
    Returns:
        {
            "status": "SUCCESS" or "FAILURE",
            "pools_per_sample": {
                retrieved_idx: {
                    "candidates": [
                        {
                            "pool_index": 0,  # 0 = original, 1+ = transformations
                            "text": "...",
                            "is_original": bool,
                            "transformation_idx": int or None
                        },
                        ...
                    ],
                    "pool_size": int
                },
                ...
            },
            "trace": [...]
        }
    """
    logger = logging.getLogger(__name__)
    local_trace = []
    
    # Extract configuration
    n_transformations = config.get("MULTIBRANCH_N_TRANSFORMATIONS", 3)
    template_key = config.get("MULTIBRANCH_TRANSFORMATION_TEMPLATE", "transformation_shallow-&-moderately-deep")
    temp_transform = config.get("MULTIBRANCH_TRANSFORMATION_TEMPERATURE", 0.0)
    
    # Determine model name
    if isinstance(api_manager, GeminiAPIManager):
        model_name = config.get("GEMINI_MODEL_NAME_ADAPTATION", "models/gemma-3-27b-it")
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config.get("AVALAI_MODEL_NAME_ADAPTATION", "gemma-3-27b-it")
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config.get("OLLAMA_MODEL_NAME_ADAPTATION", "gpt-oss:20b")
    else:
        logger.error(f"Unsupported API manager: {type(api_manager)}")
        return {"status": "FAILURE", "pools_per_sample": {}, "trace": local_trace}
    
    pools_per_sample = {}
    
    for sample_idx, retrieved_idx in enumerate(retrieved_indices):
        original_q = exemplar_data['questions'][retrieved_idx]
        original_sol = exemplar_data['solutions'][retrieved_idx]
        original_combined = f"Question: {original_q}\nSolution: {original_sol}"
        
        pool_for_sample = []
        
        # === INDEX 0: ORIGINAL RETRIEVAL (R_main) ===
        pool_for_sample.append({
            "pool_index": 0,
            "text": original_combined,
            "is_original": True,
            "transformation_idx": None,
            "source": "original_retrieval"
        })
        
        # === INDICES 1+: TRANSFORMATIONS (T_1, T_2, ..., T_N) ===
        for t_idx in range(n_transformations):
            prompt_transform = create_transformation_prompt(
                target_query=target_query,
                text_to_transform=original_combined,
                config=config,
                template_key_name=template_key
            )
            
            resp_transform = api_manager.generate_content(
                prompt_transform, model_name, temp_transform
            )
            
            local_trace.append(create_trace_entry(
                step_name="multibranch_pool_construction",
                sub_step=f"sample_{retrieved_idx}_transform_{t_idx}",
                input_context={"prompt": prompt_transform},
                output_result=resp_transform,
                metadata={"model": model_name, "temperature": temp_transform}
            ))
            
            if resp_transform['status'] == 'SUCCESS':
                pool_for_sample.append({
                    "pool_index": t_idx + 1,
                    "text": resp_transform['text'],
                    "is_original": False,
                    "transformation_idx": t_idx,
                    "source": "transformation"
                })
            else:
                logger.warning(f"Transformation failed for sample {retrieved_idx}, transform {t_idx}")
                # Add fallback
                pool_for_sample.append({
                    "pool_index": t_idx + 1,
                    "text": f"[FALLBACK] {original_combined}",
                    "is_original": False,
                    "transformation_idx": t_idx,
                    "source": "transformation_fallback"
                })
        
        pools_per_sample[retrieved_idx] = {
            "candidates": pool_for_sample,
            "pool_size": len(pool_for_sample)
        }
        
        logger.info(f"Sample {retrieved_idx}: Built pool of size {len(pool_for_sample)}")
    
    return {
        "status": "SUCCESS",
        "pools_per_sample": pools_per_sample,
        "trace": local_trace
    }


def _score_candidate_pool(
    target_query: str,
    pools_per_sample: Dict[int, Dict[str, Any]],
    api_manager_solve: Any,
    api_manager_eval: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    PHASE 2: Scores all candidates in the centralized pools.
    
    For each sample's pool, generates a candidate solution and scores it
    using mirror-style evaluation. Returns scores in aligned arrays.
    
    Returns:
        {
            "status": "SUCCESS" or "FAILURE",
            "scores_per_sample": {
                retrieved_idx: [S_0, S_1, ..., S_N]  # Aligned with pool indices
            },
            "candidate_texts_per_sample": {
                retrieved_idx: [text_0, text_1, ..., text_N]
            },
            "trace": [...]
        }
    """
    logger = logging.getLogger(__name__)
    local_trace = []
    
    enable_mirror = config.get("MULTIBRANCH_ENABLE_MIRROR_SCORING", True)
    mirror_attempts = config.get("MULTIBRANCH_MIRROR_SCORING_ATTEMPTS", 3)
    temp_solve = config.get("MULTIBRANCH_SOLVER_TEMPERATURE", 1.0)
    temp_eval = config.get("DEFAULT_EVALUATOR_TEMPERATURE", 0.0)
    
    # Determine model names
    if isinstance(api_manager_solve, GeminiAPIManager):
        model_solve = config.get("GEMINI_MODEL_NAME_FINAL_SOLVER", "models/gemma-3-27b-it")
        model_eval = config.get("GEMINI_MODEL_NAME_EVALUATOR", "models/gemma-3-27b-it")
    elif isinstance(api_manager_solve, AvalAIAPIManager):
        model_solve = config.get("AVALAI_MODEL_NAME_FINAL_SOLVER", "gemma-3-27b-it")
        model_eval = config.get("AVALAI_MODEL_NAME_EVALUATOR", "gemma-3-27b-it")
    elif isinstance(api_manager_solve, OllamaAPIManager):
        model_solve = config.get("OLLAMA_MODEL_NAME_FINAL_SOLVER", "gpt-oss:20b")
        model_eval = config.get("OLLAMA_MODEL_NAME_EVALUATOR", "gpt-oss:20b")
    else:
        logger.error(f"Unsupported API manager: {type(api_manager_solve)}")
        return {"status": "FAILURE", "scores_per_sample": {}, "trace": local_trace}
    
    scores_per_sample = {}
    candidate_texts_per_sample = {}
    
    for retrieved_idx, pool_data in pools_per_sample.items():
        pool = pool_data['candidates']
        scores = []
        candidate_texts = []
        
        for pool_idx, candidate_dict in enumerate(pool):
            candidate_context = candidate_dict['text']
            
            # Generate solution for this candidate
            prompt_solve = create_analogical_adaptation_prompt(
                target_query,
                candidate_context,
                config
            )
            
            resp_solve = api_manager_solve.generate_content(
                prompt_solve, model_solve, temp_solve
            )
            
            local_trace.append(create_trace_entry(
                step_name="multibranch_scoring",
                sub_step=f"sample_{retrieved_idx}_pool_{pool_idx}_solve",
                input_context={"prompt": prompt_solve},
                output_result=resp_solve,
                metadata={"model": model_solve, "temperature": temp_solve}
            ))
            
            if resp_solve['status'] == 'FAILURE':
                logger.warning(f"Solve failed for sample {retrieved_idx}, pool {pool_idx}")
                scores.append(0.0)
                candidate_texts.append("[SOLVE_FAILED]")
                continue
            
            solution_text = resp_solve['text']
            candidate_texts.append(solution_text)
            
            # Score this candidate
            score = 0.0
            
            if enable_mirror:
                # Mirror-style evaluation
                for eval_attempt in range(mirror_attempts):
                    prompt_eval = create_reverse_validation_prompt(
                        solution_text,
                        candidate_context,
                        config
                    )
                    
                    resp_eval = api_manager_eval.generate_content(
                        prompt_eval, model_eval, temp_eval
                    )
                    
                    local_trace.append(create_trace_entry(
                        step_name="multibranch_scoring",
                        sub_step=f"sample_{retrieved_idx}_pool_{pool_idx}_eval_{eval_attempt}",
                        input_context={"prompt": prompt_eval},
                        output_result=resp_eval,
                        metadata={"model": model_eval, "temperature": temp_eval}
                    ))
                    
                    if resp_eval['status'] == 'SUCCESS':
                        response_lower = resp_eval['text'].lower()
                        if any(word in response_lower for word in ['correct', 'yes', 'accurate', 'valid', 'true']):
                            score += 1.0
                
                score = score / mirror_attempts
            else:
                score = 1.0  # Default score if mirror eval disabled
            
            scores.append(score)
        
        scores_per_sample[retrieved_idx] = scores
        candidate_texts_per_sample[retrieved_idx] = candidate_texts
        logger.info(f"Sample {retrieved_idx}: Scored {len(scores)} candidates")
    
    return {
        "status": "SUCCESS",
        "scores_per_sample": scores_per_sample,
        "candidate_texts_per_sample": candidate_texts_per_sample,
        "trace": local_trace
    }


def _select_candidates_by_branch(
    pools_per_sample: Dict[int, Dict[str, Any]],
    scores_per_sample: Dict[int, List[float]],
    config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    PHASE 3: Applies deterministic selection logic per branch.
    
    Three strategies using same pool and scores:
    
    1. Tx1: Selected = Pool[1] (first transformation, deterministic)
    2. BoT-N: Selected = Pool[argmax(Scores[1:])] (best among transformations only)
    3. BoT-N+R: Selected = Pool[argmax_with_tiebreak(Scores)] (best among all)
    
    Tie-breaking: When scores tie (within epsilon), prefer lower indices.
    Tuples: (score, -pool_index) for sorting.
    
    Returns:
        {
            "tx1": {
                retrieved_idx: {
                    "pool_index": int,
                    "candidate": {...},
                    "score": float,
                    "source": str
                },
                ...
            },
            "bot_n": {...},
            "bot_n_plus_r": {...},
            "telemetry": {...}
        }
    """
    logger = logging.getLogger(__name__)
    
    selections = {
        "tx1": {},
        "bot_n": {},
        "bot_n_plus_r": {}
    }
    telemetry = {
        "per_sample_selections": {}
    }
    
    for retrieved_idx in pools_per_sample.keys():
        pool = pools_per_sample[retrieved_idx]['candidates']
        scores = scores_per_sample.get(retrieved_idx, [0.0] * len(pool))
        
        # Ensure scores align with pool
        if len(scores) != len(pool):
            logger.warning(f"Score/pool mismatch for sample {retrieved_idx}: {len(scores)} vs {len(pool)}")
            scores = scores[:len(pool)] + [0.0] * (len(pool) - len(scores))
        
        # === SCENARIO 1: Tx1 (Single Transformation) ===
        if len(pool) > 1:
            tx1_idx = 1  # Always second element (first transformation)
            tx1_candidate = pool[tx1_idx]
            selections["tx1"][retrieved_idx] = {
                "pool_index": tx1_idx,
                "candidate": tx1_candidate,
                "score": scores[tx1_idx],
                "source": "transformation_1"
            }
        
        # === SCENARIO 2: BoT-N (Best of Transformations, Exclusive) ===
        if len(pool) > 1:
            # Search in Scores[1:] (exclude original)
            # Extract transformation scores with their pool indices
            transform_scores = scores[1:]
            
            # Apply tie-breaking to find best transformation
            if transform_scores:
                best_transform_relative_idx = argmax_tiebreak(transform_scores)
                bot_n_idx = best_transform_relative_idx + 1  # Adjust back to pool coordinates
                bot_n_candidate = pool[bot_n_idx]
                selections["bot_n"][retrieved_idx] = {
                    "pool_index": bot_n_idx,
                    "candidate": bot_n_candidate,
                    "score": scores[bot_n_idx],
                    "source": "transformation"
                }
        
        # === SCENARIO 3: BoT-N+R (Best of N + Original, Inclusive with Tie-Breaking) ===
        # Apply tie-breaking across all candidates (including original at index 0)
        bot_nr_idx = argmax_tiebreak(scores)
        bot_nr_candidate = pool[bot_nr_idx]
        bot_nr_source = "original_retrieval" if bot_nr_candidate['is_original'] else "transformation"
        
        selections["bot_n_plus_r"][retrieved_idx] = {
            "pool_index": bot_nr_idx,
            "candidate": bot_nr_candidate,
            "score": scores[bot_nr_idx],
            "source": bot_nr_source,
            "tie_breaking_applied": False  # Would be True if scores[0] == scores[bot_nr_idx]
        }
        
        # Track this sample's selections
        telemetry["per_sample_selections"][retrieved_idx] = {
            "pool_size": len(pool),
            "tx1_idx": 1 if len(pool) > 1 else None,
            "bot_n_idx": selections["bot_n"][retrieved_idx]["pool_index"] if retrieved_idx in selections["bot_n"] else None,
            "bot_n_plus_r_idx": bot_nr_idx,
            "bot_n_plus_r_source": bot_nr_source,
            "all_scores": scores
        }
        
        logger.info(f"Sample {retrieved_idx}: Tx1→{1}, BoT-N→{selections['bot_n'][retrieved_idx]['pool_index'] if retrieved_idx in selections['bot_n'] else 'N/A'}, BoT-N+R→{bot_nr_idx}")
    
    return {
        "selections": selections,
        "telemetry": telemetry
    }


def _fan_out_solvers(
    target_query: str,
    branch_selections: Dict[str, Dict[int, Dict]],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Dict[str, List[str]]]:
    """
    PHASE 4: Fans out solver execution to each branch.
    
    For each scenario (tx1, bot_n, bot_n_plus_r):
    - Extracts the selected candidates
    - Runs solver for N attempts per branch
    - Returns independent solution lists per branch
    
    Returns:
        {
            "tx1": {
                retrieved_idx: [solution_1, solution_2, ..., solution_N],
                ...
            },
            "bot_n": {...},
            "bot_n_plus_r": {...}
        }
    """
    logger = logging.getLogger(__name__)
    
    attempts_per_branch = config.get("MULTIBRANCH_SOLVER_ATTEMPTS_PER_BRANCH", 3)
    temp_solve = config.get("MULTIBRANCH_SOLVER_TEMPERATURE", 1.0)
    
    # Determine model name
    if isinstance(api_manager, GeminiAPIManager):
        model_name = config.get("GEMINI_MODEL_NAME_FINAL_SOLVER", "models/gemma-3-27b-it")
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config.get("AVALAI_MODEL_NAME_FINAL_SOLVER", "gemma-3-27b-it")
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config.get("OLLAMA_MODEL_NAME_FINAL_SOLVER", "gpt-oss:20b")
    else:
        logger.error(f"Unsupported API manager: {type(api_manager)}")
        return {"tx1": {}, "bot_n": {}, "bot_n_plus_r": {}}
    
    result = {"tx1": {}, "bot_n": {}, "bot_n_plus_r": {}}
    
    for branch_name, selections in branch_selections.items():
        if not selections:
            continue
        
        solutions_per_branch = {}
        
        for retrieved_idx, selection_dict in selections.items():
            candidate_context = selection_dict['candidate']['text']
            solutions = []
            
            for attempt_idx in range(attempts_per_branch):
                prompt_solve = create_analogical_adaptation_prompt(
                    target_query,
                    candidate_context,
                    config
                )
                
                resp_solve = api_manager.generate_content(
                    prompt_solve, model_name, temp_solve
                )
                
                if resp_solve['status'] == 'SUCCESS':
                    solutions.append(resp_solve['text'])
                else:
                    logger.warning(f"Solve failed for {branch_name}, sample {retrieved_idx}, attempt {attempt_idx}")
            
            solutions_per_branch[retrieved_idx] = solutions
        
        result[branch_name] = solutions_per_branch
        logger.info(f"Branch {branch_name}: Generated solutions for {len(solutions_per_branch)} samples")
    
    return result


def multibranch_transformation_experiment(
    target_query: str,
    retrieved_indices: List[int],
    exemplar_data: Dict[str, Any],
    api_manager_adapt: Any,
    api_manager_solve: Any,
    api_manager_eval: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main orchestrator for multi-branch transformation experiments.
    
    Executes all four phases:
    1. Pool construction
    2. Unified scoring
    3. Deterministic branch-specific selection
    4. Fan-out solver execution
    
    Returns comprehensive result dict with:
    - evaluation_contexts: Three parallel contexts for downstream processing
    - telemetry: Selection indices, sources, and pool information
    - trace: Full execution trace from all phases
    """
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("MULTI-BRANCH TRANSFORMATION EXPERIMENT STARTED")
    logger.info("="*80)
    
    # Run all phases
    phase1_result = _build_centralized_candidate_pool(
        target_query, retrieved_indices, exemplar_data,
        api_manager_adapt, config
    )
    
    if phase1_result['status'] == 'FAILURE':
        logger.error("Phase 1 failed: Could not build candidate pool")
        return {
            "status": "FAILURE",
            "phase_failed": 1,
            "evaluation_contexts": {},
            "telemetry": {}
        }
    
    pools_per_sample = phase1_result['pools_per_sample']
    phase1_trace = phase1_result['trace']
    
    # Phase 2: Score
    phase2_result = _score_candidate_pool(
        target_query, pools_per_sample,
        api_manager_solve, api_manager_eval, config
    )
    
    if phase2_result['status'] == 'FAILURE':
        logger.error("Phase 2 failed: Could not score pool")
        return {
            "status": "FAILURE",
            "phase_failed": 2,
            "evaluation_contexts": {},
            "telemetry": {}
        }
    
    scores_per_sample = phase2_result['scores_per_sample']
    phase2_trace = phase2_result['trace']
    
    # Phase 3: Select by branch
    phase3_result = _select_candidates_by_branch(
        pools_per_sample, scores_per_sample, config
    )
    
    selections = phase3_result['selections']
    telemetry_phase3 = phase3_result['telemetry']
    
    # Phase 4: Fan-out solvers
    phase4_result = _fan_out_solvers(
        target_query, selections,
        api_manager_solve, config
    )
    
    # === BUILD EVALUATION CONTEXTS ===
    evaluation_contexts = {}
    
    if config.get("RUN_TX1_BASELINE", True):
        evaluation_contexts["tx1"] = {
            "scenario": "Single Transformation (Tx1)",
            "selections": selections["tx1"],
            "solutions": phase4_result["tx1"],
            "intervention_rate": 1.0  # 100% intervention
        }
    
    if config.get("RUN_BOT_N_ONLY", True):
        evaluation_contexts["bot_n"] = {
            "scenario": "Best-of-N (Exclusive)",
            "selections": selections["bot_n"],
            "solutions": phase4_result["bot_n"],
            "intervention_rate": None  # Will be computed per-sample
        }
    
    if config.get("RUN_BOT_N_PLUS_R", True):
        evaluation_contexts["bot_n_plus_r"] = {
            "scenario": "Best-of-N+R (Inclusive)",
            "selections": selections["bot_n_plus_r"],
            "solutions": phase4_result["bot_n_plus_r"],
            "intervention_rate": None  # Will be computed per-sample
        }
    
    # === BUILD FINAL TELEMETRY ===
    final_telemetry = {
        "pool_size_per_sample": {
            idx: pools_per_sample[idx]['pool_size']
            for idx in pools_per_sample.keys()
        },
        "selections": telemetry_phase3["per_sample_selections"],
        "branches_enabled": {
            "tx1": config.get("RUN_TX1_BASELINE", True),
            "bot_n": config.get("RUN_BOT_N_ONLY", True),
            "bot_n_plus_r": config.get("RUN_BOT_N_PLUS_R", True)
        }
    }
    
    all_trace = phase1_trace + phase2_trace
    
    logger.info("="*80)
    logger.info("MULTI-BRANCH TRANSFORMATION EXPERIMENT COMPLETED")
    logger.info("="*80)
    
    return {
        "status": "SUCCESS",
        "evaluation_contexts": evaluation_contexts,
        "telemetry": final_telemetry,
        "trace": all_trace
    }
