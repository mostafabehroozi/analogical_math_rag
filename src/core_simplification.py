# src/core_simplification.py

import logging
import re
from typing import Dict, Any, List, Tuple
from src.prompts import (
    create_core_simp_zero_shot_prompt,
    create_core_simp_augmented_solver_prompt,
    create_final_reasoning_prompt_simple
)
from src.evaluation import evaluate_single_answer_with_llm
from src.utils import create_trace_entry
from src.api_manager import GeminiAPIManager, AvalAIAPIManager, OllamaAPIManager

logger = logging.getLogger(__name__)

def _parse_simplification_trace(llm_output: str) -> Dict[str, str]:
    """
    Parses the strict 4-part output from the zero-shot simplification prompt.
    Returns a dictionary containing the extracted parts.
    """
    parsed = {
        "topology_analysis": "",
        "trunk_breakdown": "",
        "simplification_methodology": "",
        "proxy_question": ""
    }
    
    # Simple regex parsing based on the expected headers
    try:
        # Extract Part 4 (The Proxy Question)
        part4_match = re.search(r"\*\*4\. The Proxy Question:\*\*(.*?)$", llm_output, re.DOTALL | re.IGNORECASE)
        if part4_match:
            parsed["proxy_question"] = part4_match.group(1).strip()
        else:
            # Fallback if markdown asterisks are missing
            part4_fallback = re.search(r"4\. The Proxy Question:(.*?)$", llm_output, re.DOTALL | re.IGNORECASE)
            if part4_fallback:
                parsed["proxy_question"] = part4_fallback.group(1).strip()
    except Exception as e:
        logger.warning(f"Failed to parse Proxy Question: {e}")
        
    return parsed

def _solve_and_evaluate(
    target_query: str,
    ground_truth: str,
    prompt: str,
    api_manager_solve: Any,
    api_manager_eval: Any,
    config: Dict[str, Any],
    n_attempts: int,
    temp: float,
    trace_name_prefix: str,
    local_trace: List[Dict],
    target_correct_to_beat: int = -1  # <--- NEW: Target threshold
) -> Tuple[float, List[str]]:
    """
    Helper function to generate N attempts and evaluate them on-the-fly.
    Returns the Pass@N accuracy score (0.0 to 1.0) and the list of generated text answers.
    """
    if isinstance(api_manager_solve, GeminiAPIManager):
        model_solve = config.get('GEMINI_MODEL_NAME_FINAL_SOLVER')
    elif isinstance(api_manager_solve, AvalAIAPIManager):
        model_solve = config.get('AVALAI_MODEL_NAME_FINAL_SOLVER')
    elif isinstance(api_manager_solve, OllamaAPIManager):
        model_solve = config.get('OLLAMA_MODEL_NAME_FINAL_SOLVER')
    else:
        raise TypeError(f"Unsupported API manager type for solving: {type(api_manager_solve)}")
    
    attempts = []
    correct_count = 0
    
    for i in range(n_attempts):
        resp = api_manager_solve.generate_content(prompt, model_solve, temp)
        
        local_trace.append(create_trace_entry(
            "core_simplification", f"{trace_name_prefix}_attempt_{i+1}",
            {"prompt": prompt}, resp, {"model": model_solve, "temp": temp}
        ))
        
        if resp['status'] == 'SUCCESS':
            ans_text = resp['text']
            attempts.append(ans_text)
            
            # On-the-fly evaluation
            eval_res = evaluate_single_answer_with_llm(ans_text, ground_truth, api_manager_eval, config)
            if eval_res.get('status') == 'SUCCESS' and eval_res.get('is_correct'):
                correct_count += 1
        else:
            attempts.append(f"[GENERATION FAILED: {resp.get('error_message')}]")
            
        # --- NEW FEATURE: EARLY STOP IF MATHEMATICALLY IMPOSSIBLE TO BEAT BASELINE ---
        if target_correct_to_beat >= 0:
            remaining_attempts = n_attempts - (i + 1)
            max_possible_correct = correct_count + remaining_attempts
            # If the maximum we can possibly get is less than or equal to what we need to beat...
            if max_possible_correct <= target_correct_to_beat:
                print(f"       => [EARLY STOP] Max possible score ({max_possible_correct}/{n_attempts}) cannot beat baseline ({target_correct_to_beat}/{n_attempts}). Halting loop.")
                break
        # ----------------------------------------------------------------------------
            
    if not attempts:
        return 0.0, attempts
        
    accuracy = correct_count / n_attempts
    return accuracy, attempts

def run_core_simplification_phase1(
    target_query: str,
    ground_truth: str,
    api_manager_solve: Any,
    api_manager_eval: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes the Phase 1 A/B testing pipeline for a single target question.
    """
    logger.info("Starting Core-Preserving Simplification Phase 1 pipeline.")
    local_trace = []
    
    n_attempts = config.get("CORE_SIMP_PHASE1_N_ATTEMPTS", 5)
    temp_gen = config.get("CORE_SIMP_TEMPERATURE_GEN", 0.3)
    temp_solve = config.get("CORE_SIMP_TEMPERATURE_SOLVE", 1.0)
    
    # Dynamically determine the generator (adaptation) and solver models
    if isinstance(api_manager_solve, GeminiAPIManager):
        model_gen = config.get('GEMINI_MODEL_NAME_ADAPTATION')
        model_solve = config.get('GEMINI_MODEL_NAME_FINAL_SOLVER')
    elif isinstance(api_manager_solve, AvalAIAPIManager):
        model_gen = config.get('AVALAI_MODEL_NAME_ADAPTATION')
        model_solve = config.get('AVALAI_MODEL_NAME_FINAL_SOLVER')
    elif isinstance(api_manager_solve, OllamaAPIManager):
        model_gen = config.get('OLLAMA_MODEL_NAME_ADAPTATION')
        model_solve = config.get('OLLAMA_MODEL_NAME_FINAL_SOLVER')
    else:
        raise TypeError(f"Unsupported API manager type: {type(api_manager_solve)}")
    
    print("\n" + "="*70)
    print("  [CORE SIMPLIFICATION: PHASE 1 (A/B TEST)]")
    print("="*70)

    print(f"  -> [Step A] Running Baseline Control (N={n_attempts})...")
    base_prompt = create_final_reasoning_prompt_simple(target_query, config)
    
    base_score, base_attempts = _solve_and_evaluate(
        target_query, ground_truth, base_prompt, 
        api_manager_solve, api_manager_eval, config, 
        n_attempts, temp_solve, "baseline_solve", local_trace
    )
    print(f"     => Base Consistency Score: {base_score:.2f} ({int(base_score * n_attempts)}/{n_attempts} correct)")

    # --- NEW FEATURE: EARLY EXIT FOR PERFECT BASELINE ---
    if base_score >= 1.0:
        print(f"     => [EARLY EXIT] Baseline is already perfect (1.0). Skipping simplification to save API costs.")
        return {
            "status": "SKIPPED_PERFECT_BASELINE", 
            "base_score": base_score,
            "original_question": target_query,
            "ground_truth": ground_truth,
            "trace": local_trace
        }

    # STEP B: Proxy Generation & Failsafe Check
    print(f"  -> [Step B] Generating Proxy via 4-Part Structural Prompt...")
    gen_prompt = create_core_simp_zero_shot_prompt(target_query)
    
    gen_resp = api_manager_solve.generate_content(gen_prompt, model_gen, temp_gen)
    
    local_trace.append(create_trace_entry(
        "core_simplification", "generate_proxy",
        {"prompt": gen_prompt}, gen_resp, {"model": model_gen, "temp": temp_gen}
    ))
    
    if gen_resp['status'] != 'SUCCESS':
        logger.error("Failed to generate proxy question.")
        return {"status": "FAILURE", "reason": "Proxy generation failed", "trace": local_trace}
        
    full_trace_text = gen_resp['text']
    parsed_parts = _parse_simplification_trace(full_trace_text)
    proxy_q = parsed_parts.get("proxy_question", "")
    methodology = parsed_parts.get("simplification_methodology", "").lower()
    
    if not proxy_q:
        logger.error("Failed to parse Proxy Question from output.")
        return {"status": "FAILURE", "reason": "Parsing failed", "trace": local_trace}

    # --- NEW FEATURE: ENHANCED EXPLICIT FAILSAFE CHECK ---
    # 1. Check if LLM explicitly declared it cannot simplify in the methodology
    if "no safe simplification" in methodology:
        print(f"     => [FAILSAFE TRIGGERED] Model explicitly stated no safe simplification is possible. Aborting.")
        return {"status": "SKIPPED_FAILSAFE", "trace": local_trace}
        
    # 2. String comparison fallback (original failsafe)
    def clean_text(t): return re.sub(r'\W+', '', t.lower())
    if clean_text(proxy_q) == clean_text(target_query):
        print(f"     => [FAILSAFE TRIGGERED] Model returned identical question. Aborting.")
        return {"status": "SKIPPED_FAILSAFE", "trace": local_trace}
    # ------------------------------------------------------
        
    print(f"     => Proxy generated successfully:\n        '{proxy_q[:80]}...'")

    # ---------------------------------------------------------
    # STEP C: Solve the Proxy
    # ---------------------------------------------------------
    print(f"  -> [Step C] Solving the Proxy Question...")
    proxy_solve_prompt = create_final_reasoning_prompt_simple(proxy_q, config)
    
    # We only need 1 good attempt for the proxy solution to use as context
    proxy_solve_resp = api_manager_solve.generate_content(proxy_solve_prompt, model_solve, temp_solve)
    
    local_trace.append(create_trace_entry(
        "core_simplification", "solve_proxy",
        {"prompt": proxy_solve_prompt}, proxy_solve_resp, {"model": model_solve, "temp": temp_solve}
    ))
    
    if proxy_solve_resp['status'] != 'SUCCESS':
        logger.error("Failed to solve proxy question.")
        return {"status": "FAILURE", "reason": "Proxy solve failed", "trace": local_trace}
        
    solved_proxy_rationale = proxy_solve_resp['text']
    solved_proxy_combined = f"Question: {proxy_q}\nRationale and Answer: {solved_proxy_rationale}"

    # STEP D: Augmented Main Solve Loop (A/B Test Comparison)
    print(f"  -> [Step D] Running Augmented Solve (N={n_attempts})...")
    aug_prompt = create_core_simp_augmented_solver_prompt(target_query, solved_proxy_combined)
    
    # Calculate exactly how many correct answers we need to beat the baseline
    base_correct_count = int(base_score * n_attempts)
    
    aug_score, aug_attempts = _solve_and_evaluate(
        target_query, ground_truth, aug_prompt, 
        api_manager_solve, api_manager_eval, config, 
        n_attempts, temp_solve, "augmented_solve", local_trace,
        target_correct_to_beat=base_correct_count # <--- NEW: Pass the target to trigger early stop
    )
    print(f"     => Augmented Consistency Score: {aug_score:.2f} ({int(aug_score * n_attempts)}/{n_attempts} correct)")

    # ---------------------------------------------------------
    # STEP E: Delta Filter & Package
    # ---------------------------------------------------------
    if aug_score > base_score:
        print(f"  => [SUCCESS] Simplification proved structurally sound! (Delta: +{aug_score - base_score:.2f})")
        status = "SUCCESS"
    else:
        print(f"  => [REJECTED] Simplification did not improve performance. (Delta: {aug_score - base_score:.2f})")
        status = "REJECTED_BY_FILTER"
        
    # We return the packaged data regardless, but marked with its status
    result_data = {
        "status": status,
        "base_score": base_score,
        "augmented_score": aug_score,
        "original_question": target_query,
        "ground_truth": ground_truth,
        "proxy_generation_full_trace": full_trace_text,
        "proxy_question": proxy_q,
        "proxy_solution": solved_proxy_rationale,
        "trace": local_trace
    }
    
    return result_data