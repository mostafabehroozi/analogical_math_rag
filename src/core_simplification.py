# src/core_simplification.py

import logging
import os
import re
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from src.prompts import (
    create_core_simp_zero_shot_prompt,
    create_core_simp_augmented_solver_prompt,
    create_final_reasoning_prompt_simple
)
from src.evaluation import evaluate_single_answer_with_llm
from src.benchmark_data import benchmark_name_for_target_index
from src.utils import create_trace_entry, load_json, save_json, save_json_atomic
from src.api_manager import GeminiAPIManager, AvalAIAPIManager, OllamaAPIManager
from src.batching import BatchCoordinator, QuestionWorkItem, QuestionResult
from src.distributed_execution import (
    apply_distributed_run_log_contract,
    distributed_enabled,
)
from src.hf_sync import periodic_sync_check, periodic_batch_sync_check
from src.parallel_utils import run_parallel_api_calls

logger = logging.getLogger(__name__)

_COMPLETED_PHASE1_STATUSES = {
    "SUCCESS",
    "REJECTED_BY_FILTER",
    "SKIPPED_FAILSAFE",
    "SKIPPED_PERFECT_BASELINE",
}


def _phase1_result_completed(result: Dict[str, Any]) -> bool:
    return str(result.get("status", "")).upper() in _COMPLETED_PHASE1_STATUSES

def _parse_simplification_trace(llm_output: str) -> Dict[str, str]:
    """
    Parses the streamlined output from the simplification prompt.
    Extracts the text following 'Simplified Question:'.
    """
    parsed = {"proxy_question": ""}
    
    # Extract everything after "Simplified Question:"
    match = re.search(r"Simplified Question:\s*(.*)", llm_output, re.DOTALL | re.IGNORECASE)
    
    if match:
        parsed["proxy_question"] = match.group(1).strip()
    else:
        # Fallback: If the LLM just outputs the question without the tag, grab it all
        parsed["proxy_question"] = llm_output.strip()
        
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
    
    attempts: List[str] = []
    correct_count = 0

    def run_attempt(i: int) -> Dict[str, Any]:
        """Run one independent generation/evaluation pair."""
        resp = api_manager_solve.generate_content(prompt, model_solve, temp)
        trace_entry = create_trace_entry(
            "core_simplification", f"{trace_name_prefix}_attempt_{i+1}",
            {"prompt": prompt}, resp, {"model": model_solve, "temp": temp}
        )

        if resp['status'] == 'SUCCESS':
            ans_text = resp['text']
            eval_res = evaluate_single_answer_with_llm(ans_text, ground_truth, api_manager_eval, config)
            is_correct = bool(eval_res.get('status') == 'SUCCESS' and eval_res.get('is_correct'))
        else:
            ans_text = f"[GENERATION FAILED: {resp.get('error_message')}]"
            is_correct = False

        return {
            "attempt_index": i,
            "answer": ans_text,
            "is_correct": is_correct,
            "trace_entry": trace_entry,
        }

    parallel_enabled = bool(config.get("QUESTION_PARALLEL_API_ENABLED", False))
    max_workers = max(1, int(config.get("QUESTION_PARALLEL_MAX_WORKERS", 3)))

    # With early stopping active, submit at most one worker-sized wave at a
    # time. This retains that optimization at wave boundaries while still
    # parallelizing API calls inside the question.
    wave_size = max_workers if parallel_enabled else 1
    for wave_start in range(0, n_attempts, wave_size):
        wave_indices = range(wave_start, min(wave_start + wave_size, n_attempts))
        tasks = [lambda attempt_idx=i: run_attempt(attempt_idx) for i in wave_indices]
        if parallel_enabled and len(tasks) > 1:
            wave_results = run_parallel_api_calls(tasks, config)
        else:
            wave_results = [task() for task in tasks]

        # Preserve attempt/trace order even when calls complete out of order.
        for attempt_result in wave_results:
            local_trace.append(attempt_result["trace_entry"])
            attempts.append(attempt_result["answer"])
            if attempt_result["is_correct"]:
                correct_count += 1

        # --- EARLY STOP IF MATHEMATICALLY IMPOSSIBLE TO BEAT BASELINE ---
        if target_correct_to_beat >= 0:
            remaining_attempts = n_attempts - len(attempts)
            max_possible_correct = correct_count + remaining_attempts
            if max_possible_correct <= target_correct_to_beat:
                print(f"       => [EARLY STOP] Max possible score ({max_possible_correct}/{n_attempts}) cannot beat baseline ({target_correct_to_beat}/{n_attempts}). Halting loop.")
                break

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
    
    base_score, _ = _solve_and_evaluate(
        target_query, ground_truth, base_prompt, 
        api_manager_solve, api_manager_eval, config, 
        n_attempts, temp_solve, "baseline_solve", local_trace
    )
    print(f"     => Base Consistency Score: {base_score:.2f} ({int(base_score * n_attempts)}/{n_attempts} correct)")

    # --- NEW FEATURE: EARLY EXIT FOR PERFECT BASELINE ---
    if base_score >= 1.0:
        print("     => [EARLY EXIT] Baseline is already perfect (1.0). Skipping simplification to save API costs.")
        return {
            "status": "SKIPPED_PERFECT_BASELINE", 
            "base_score": base_score,
            "original_question": target_query,
            "ground_truth": ground_truth,
            "trace": local_trace
        }

    # STEP B: Proxy Generation & Failsafe Check
    print("  -> [Step B] Generating Proxy Question...")
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
    
    if not proxy_q:
        logger.error("Failed to parse Proxy Question from output.")
        return {"status": "FAILURE", "reason": "Parsing failed", "trace": local_trace}

    # --- ENHANCED EXPLICIT FAILSAFE CHECK ---
    # We rely entirely on string comparison to see if the LLM changed the question
    def clean_text(t): return re.sub(r'\W+', '', t.lower())
    
    if clean_text(proxy_q) == clean_text(target_query):
        print("     => [FAILSAFE TRIGGERED] Model returned identical question. Aborting.")
        return {"status": "SKIPPED_FAILSAFE", "trace": local_trace}
    # ------------------------------------------------------
        
    print(f"     => Proxy generated successfully:\n        '{proxy_q[:80]}...'")

    # ---------------------------------------------------------
    # STEP C: Solve the Proxy
    # ---------------------------------------------------------
    print("  -> [Step C] Solving the Proxy Question...")
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
    
    aug_score, _ = _solve_and_evaluate(
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


def execute_core_simplification_phase1(
    hard_questions: List[str],
    hard_solutions: List[str],
    exemplar_data: Dict[str, Any],
    api_manager_solve: Any,
    api_manager_eval: Any,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build the Phase-1 donor dataset with resumable sequential or batch execution."""
    experiment_name = config.get("experiment_name", "core_simp_phase1")
    dataset_filename = config.get("CORE_SIMP_DATASET_NAME", "core_simp_dataset.json")
    dataset_path = os.path.join(config["RESULTS_DIR"], dataset_filename)
    log_file_path = os.path.join(config["RESULTS_DIR"], f"{experiment_name}_run_log.json")

    successful_samples = load_json(dataset_path) or []
    full_logs = load_json(log_file_path) or []
    completed_indices = {
        log.get("original_index")
        for log in full_logs
        if "original_index" in log and _phase1_result_completed(log)
    }

    def resolve_ground_truth(index: int) -> Any:
        if hard_solutions and index < len(hard_solutions):
            return hard_solutions[index]
        if "ground_truths" in exemplar_data and index < len(exemplar_data["ground_truths"]):
            return exemplar_data["ground_truths"][index]
        if (
            "solutions" in exemplar_data
            and len(exemplar_data["solutions"]) == len(hard_questions)
        ):
            return exemplar_data["solutions"][index]
        return None

    ground_truth_by_index = {
        index: resolve_ground_truth(index)
        for index in range(len(hard_questions))
        if index not in completed_indices
    }
    ground_truth_by_index = {
        index: ground_truth
        for index, ground_truth in ground_truth_by_index.items()
        if ground_truth
    }
    items = [
        QuestionWorkItem(index=index, question=hard_questions[index])
        for index in sorted(ground_truth_by_index)
    ]
    if distributed_enabled(config):
        owned_indices = set(config.get("_DISTRIBUTED_ALLOWED_QUERY_INDICES", []))
        items = [item for item in items if item.index in owned_indices]

    if not items:
        if distributed_enabled(config):
            if not os.path.exists(dataset_path) and not save_json_atomic(
                successful_samples, dataset_path
            ):
                raise RuntimeError(
                    "Failed to initialize core simplification Phase 1 donor shard"
                )
            if not os.path.exists(log_file_path) and not save_json_atomic(
                full_logs, log_file_path
            ):
                raise RuntimeError(
                    "Failed to initialize core simplification Phase 1 run-log shard"
                )
        logger.info(
            "All eligible queries for Phase 1 '%s' are already processed. Skipping.",
            experiment_name,
        )
        return full_logs

    def run_item(item: QuestionWorkItem) -> Dict[str, Any]:
        item_config = config.copy()
        item_config["_TARGET_BENCHMARK_FOR_QUERY"] = benchmark_name_for_target_index(
            config, item.index
        )
        result = run_core_simplification_phase1(
            target_query=item.question,
            ground_truth=ground_truth_by_index[item.index],
            api_manager_solve=api_manager_solve,
            api_manager_eval=api_manager_eval,
            config=item_config,
        )
        result["original_index"] = item.index
        if distributed_enabled(config):
            apply_distributed_run_log_contract(
                result,
                index=item.index,
                question=item.question,
                completed=_phase1_result_completed(result),
            )
        return result

    if config.get("BATCH_PROCESSING_ENABLED", False):
        # Index maps make retries idempotent if one checkpoint file was
        # committed before an interruption.
        unindexed_logs = [entry for entry in full_logs if "original_index" not in entry]
        logs_by_index = {
            entry["original_index"]: entry for entry in full_logs if "original_index" in entry
        }
        unindexed_samples = [
            entry for entry in successful_samples if "original_index" not in entry
        ]
        samples_by_index = {
            entry["original_index"]: entry
            for entry in successful_samples
            if "original_index" in entry
        }

        def ordered_logs() -> List[Dict[str, Any]]:
            return unindexed_logs + [logs_by_index[index] for index in sorted(logs_by_index)]

        def ordered_samples() -> List[Dict[str, Any]]:
            return unindexed_samples + [
                samples_by_index[index] for index in sorted(samples_by_index)
            ]

        def commit(
            results: List[QuestionResult], _batch_id: str, batch_number: int
        ) -> None:
            for question_result in results:
                value = question_result.value
                value.setdefault("original_index", question_result.item.index)
                value.setdefault("original_question", question_result.item.question)
                logs_by_index[question_result.item.index] = value
                if value.get("status") == "SUCCESS":
                    samples_by_index[question_result.item.index] = value

            # Save donors first. If the second write is interrupted, index-based
            # de-duplication prevents duplicate donors when the batch is retried.
            if not save_json_atomic(ordered_samples(), dataset_path):
                raise RuntimeError(
                    "Failed to commit core simplification Phase 1 donor dataset"
                )
            if not save_json_atomic(ordered_logs(), log_file_path):
                raise RuntimeError("Failed to commit core simplification Phase 1 run log")

            print(
                f"\n   [MILESTONE] Total verified donors collected so far: "
                f"{len(ordered_samples())}"
            )
            periodic_batch_sync_check(batch_number - 1, config)

        BatchCoordinator(config, experiment_name, "core_simp_phase1").run(
            items, run_item, commit
        )
        return ordered_logs()

    for loop_index, item in enumerate(
        tqdm(items, desc=f"Phase 1: {experiment_name}")
    ):
        result = run_item(item)
        full_logs.append(result)
        save_json(full_logs, log_file_path)

        if result.get("status") == "SUCCESS":
            successful_samples.append(result)
            save_json(successful_samples, dataset_path)
            print(
                f"\n   [MILESTONE] Total verified donors collected so far: "
                f"{len(successful_samples)}"
            )

        periodic_sync_check(loop_index, config)

    return full_logs
