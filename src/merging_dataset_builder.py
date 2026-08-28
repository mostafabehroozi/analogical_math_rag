# src/merging_dataset_builder.py

import os
import logging
from tqdm import tqdm
from typing import Dict, Any, List

from src.utils import save_json, save_json_atomic, load_json
from src.hf_sync import periodic_sync_check, periodic_batch_sync_check
from src.batching import BatchCoordinator, QuestionWorkItem, QuestionResult
from src.pipeline_steps import retrieve
from src.prompts import (
    create_final_reasoning_prompt,
    create_final_reasoning_prompt_simple,
    EXEMPLAR_FORMAT
)
from src.evaluation import evaluate_single_answer_with_llm
from src.api_manager import GeminiAPIManager, AvalAIAPIManager, OllamaAPIManager
from src.benchmark_data import benchmark_name_for_target_index
from src.distributed_execution import (
    apply_distributed_run_log_contract,
    distributed_enabled,
)
from src.parallel_utils import run_parallel_api_calls

logger = logging.getLogger(__name__)

_COMPLETED_STATUSES = {"SUCCESS", "REJECTED"}


def _result_completed(result: Dict[str, Any]) -> bool:
    return str(result.get("status", "")).upper() in _COMPLETED_STATUSES

def _get_model_name(api_manager: Any, config: Dict[str, Any]) -> str:
    """Helper to extract the correct solver model name based on the API manager."""
    if isinstance(api_manager, GeminiAPIManager):
        return config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, AvalAIAPIManager):
        return config['AVALAI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, OllamaAPIManager):
        return config['OLLAMA_MODEL_NAME_FINAL_SOLVER']
    else:
        raise TypeError(f"Unsupported API manager: {type(api_manager)}")

def process_single_question(
    target_query: str,
    ground_truth: str,
    exemplar_data: Dict[str, Any],
    embedding_model: Any,
    api_manager_solve: Any,
    api_manager_eval: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes Phase 1 and Phase 2 of the Merging dataset construction logic.
    """
    model_name = _get_model_name(api_manager_solve, config)
    
    # Load specific configs
    ccs_n = max(1, int(config.get("MERGING_DS_CCS_N", 5)))
    temperature = config.get("MERGING_DS_TEMPERATURE", 0.7)
    use_base_check = config.get("MERGING_DS_BASE_CHECK", True)

    result_log = {
        "status": "PENDING",
        "target_query": target_query,
        "few_shot_ccs": 0.0,
        "base_ccs": 0.0,
        "dataset_entry": None,
        "error_info": None
    }

    # =========================================================================
    # STEP 0: RETRIEVAL (Get R1 and R2)
    # =========================================================================
    retrieval_res = retrieve(
        target_query=target_query,
        embedding_model=embedding_model,
        exemplar_questions=exemplar_data['questions'],
        embedded_exemplars=exemplar_data['embeddings'],
        top_k=2, # We specifically need exactly 2 retrieved samples
        question_to_index_map=exemplar_data.get('question_to_index')
    )

    if retrieval_res['status'] != 'SUCCESS' or len(retrieval_res['retrieved_indices']) < 2:
        result_log['status'] = "FAILURE"
        result_log['error_info'] = "Failed to retrieve 2 exemplars."
        return result_log

    idx_r1, idx_r2 = retrieval_res['retrieved_indices'][:2]
    
    r1_text = EXEMPLAR_FORMAT.format(question=exemplar_data['questions'][idx_r1], solution=exemplar_data['solutions'][idx_r1])
    r2_text = EXEMPLAR_FORMAT.format(question=exemplar_data['questions'][idx_r2], solution=exemplar_data['solutions'][idx_r2])

    # =========================================================================
    # PHASE 1: TARGET/LABEL GENERATION (Q_target) & ACCEPTANCE CHECK
    # =========================================================================
    # 1A. Calculate Few-Shot CCS
    prompt_target = create_final_reasoning_prompt(target_query, [r1_text, r2_text], config)
    
    prompt_base = (
        create_final_reasoning_prompt_simple(target_query, config)
        if use_base_check
        else None
    )

    def run_ccs_attempt(prompt: str, group: str, attempt_index: int) -> Dict[str, Any]:
        """Run one independent generation/evaluation pair for a CCS group."""
        resp = api_manager_solve.generate_content(
            prompt, model_name, temperature, avalai_role="final_solver"
        )
        is_correct = False
        if resp.get('status') == 'SUCCESS':
            eval_res = evaluate_single_answer_with_llm(
                resp.get('text', ''), ground_truth, api_manager_eval, config
            )
            is_correct = bool(
                eval_res.get('status') == 'SUCCESS' and eval_res.get('is_correct')
            )
        return {
            "group": group,
            "attempt_index": attempt_index,
            "response": resp,
            "is_correct": is_correct,
        }

    # Few-shot and base CCS attempts are mutually independent. Running them in
    # one shared task list keeps QUESTION_PARALLEL_MAX_WORKERS as the cap for
    # the complete acceptance-check stage rather than multiplying worker pools.
    ccs_tasks = [
        lambda attempt_index=i: run_ccs_attempt(
            prompt_target, "few_shot", attempt_index
        )
        for i in range(ccs_n)
    ]
    if use_base_check:
        ccs_tasks.extend(
            lambda attempt_index=i: run_ccs_attempt(
                prompt_base, "base", attempt_index
            )
            for i in range(ccs_n)
        )

    ccs_results = run_parallel_api_calls(ccs_tasks, config)
    few_shot_results = [
        result for result in ccs_results if result["group"] == "few_shot"
    ]
    base_results = [result for result in ccs_results if result["group"] == "base"]
    few_shot_correct_answers = [
        result["response"].get("text", "")
        for result in few_shot_results
        if result["is_correct"]
    ]

    few_shot_ccs = len(few_shot_correct_answers) / ccs_n
    result_log['few_shot_ccs'] = few_shot_ccs

    # 1B. Calculate Base CCS (if required)
    base_ccs = 0.0
    if use_base_check:
        base_correct_count = sum(result["is_correct"] for result in base_results)
        base_ccs = base_correct_count / ccs_n
        result_log['base_ccs'] = base_ccs

    # 1C. Check Acceptance Criteria
    accepted = False
    if use_base_check:
        if few_shot_ccs > base_ccs:
            accepted = True
    else:
        if few_shot_ccs > 0:
            accepted = True

    if not accepted:
        result_log['status'] = "REJECTED"
        result_log['error_info'] = f"Failed criteria. Few-Shot CCS: {few_shot_ccs}, Base CCS: {base_ccs}"
        return result_log

    # Pick the first correct answer as our label (Q_target)
    label_target = few_shot_correct_answers[0]

    # =========================================================================
    # PHASE 2: INPUT CONTEXT GENERATION (Q_R1 and Q_R2)
    # =========================================================================
    # Q_R1 and Q_R2 have no dependency on each other, so use the same
    # in-question parallel runner (which remains sequential when disabled).
    prompt_r1 = create_final_reasoning_prompt(target_query, [r1_text], config)
    prompt_r2 = create_final_reasoning_prompt(target_query, [r2_text], config)
    resp_r1, resp_r2 = run_parallel_api_calls(
        [
            lambda: api_manager_solve.generate_content(
                prompt_r1, model_name, temperature, avalai_role="final_solver"
            ),
            lambda: api_manager_solve.generate_content(
                prompt_r2, model_name, temperature, avalai_role="final_solver"
            ),
        ],
        config,
    )

    if resp_r1['status'] != 'SUCCESS' or resp_r2['status'] != 'SUCCESS':
        result_log['status'] = "FAILURE"
        result_log['error_info'] = "API generation failed during Phase 2 (Input Generation)."
        return result_log

    # We do NOT check correctness. We take them exactly as generated.
    q_r1_solution = resp_r1['text']
    q_r2_solution = resp_r2['text']

    # =========================================================================
    # PHASE 3: DATASET ASSEMBLY
    # =========================================================================
    # We construct the simulated few-shot examples using the same main question,
    # but pairing it with the two different generated solutions.
    fake_exemplar_1 = EXEMPLAR_FORMAT.format(question=target_query, solution=q_r1_solution)
    fake_exemplar_2 = EXEMPLAR_FORMAT.format(question=target_query, solution=q_r2_solution)

    # Use the existing prompt creator to perfectly format the final fine-tuning input
    # This ensures it exactly matches the structure the LLM will see in production
    dataset_input_prompt = create_final_reasoning_prompt(target_query, [fake_exemplar_1, fake_exemplar_2], config)

    result_log['status'] = "SUCCESS"
    result_log['dataset_entry'] = {
        "input_prompt": dataset_input_prompt,
        "label": label_target
    }

    return result_log

def build_merging_dataset(
    hard_questions: List[str],
    hard_solutions: List[str],
    exemplar_data: Dict[str, Any],
    embedding_model: Any,
    api_managers: Dict[str, Any],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Main orchestrator for building the Merging fine-tuning dataset.
    Handles progress tracking, saving/loading state, and iterating over queries.
    """
    logger.info("Starting Merging Dataset Construction.")
    
    # Get the appropriate API managers based on config
    solver_mgr = api_managers.get(config.get("API_PROVIDER_SOLVER", "gemini"))
    eval_mgr = api_managers.get(config.get("API_PROVIDER_EVALUATOR", "gemini"))
    
    exp_name = config.get("experiment_name", "merging_dataset_construction")
    
    # Set up file paths
    results_dir = config.get("RESULTS_DIR")
    dataset_filename = config.get("MERGING_DS_OUTPUT_FILENAME", "merging_fine_tuning_dataset.json")
    
    dataset_path = os.path.join(results_dir, dataset_filename)
    log_file_path = os.path.join(results_dir, f"{exp_name}_run_log.json")
    
    # Load existing progress to resume safely
    successful_dataset = load_json(dataset_path) or []
    full_logs = load_json(log_file_path) or []
    
    completed_indices = {
        log.get("original_index")
        for log in full_logs
        if "original_index" in log and _result_completed(log)
    }
    
    queries_to_process = [(idx, q) for idx, q in enumerate(hard_questions) if idx not in completed_indices]
    if distributed_enabled(config):
        owned_indices = set(config.get("_DISTRIBUTED_ALLOWED_QUERY_INDICES", []))
        queries_to_process = [
            (index, query)
            for index, query in queries_to_process
            if index in owned_indices
        ]
    
    if not queries_to_process:
        if distributed_enabled(config):
            if not os.path.exists(dataset_path) and not save_json_atomic(
                successful_dataset, dataset_path
            ):
                raise RuntimeError("Failed to initialize merging dataset shard")
            if not os.path.exists(log_file_path) and not save_json_atomic(
                full_logs, log_file_path
            ):
                raise RuntimeError("Failed to initialize merging run-log shard")
        logger.info(f"All queries for '{exp_name}' are already processed. Returning existing results.")
        return full_logs

    if config.get("BATCH_PROCESSING_ENABLED", False):
        ground_truths = {}
        for original_idx, query in queries_to_process:
            gt = None
            if hard_solutions and original_idx < len(hard_solutions):
                gt = hard_solutions[original_idx]
            elif 'ground_truths' in exemplar_data and original_idx < len(exemplar_data['ground_truths']):
                gt = exemplar_data['ground_truths'][original_idx]
            elif 'solutions' in exemplar_data and len(exemplar_data['solutions']) == len(hard_questions):
                gt = exemplar_data['solutions'][original_idx]
            if gt:
                ground_truths[original_idx] = gt
            else:
                logger.warning("Skipping index %s: No ground truth available.", original_idx)

        unindexed_logs = [log for log in full_logs if "original_index" not in log]
        logs_by_index = {
            log["original_index"]: log
            for log in full_logs
            if "original_index" in log
        }
        unindexed_dataset = [
            entry
            for entry in successful_dataset
            if "original_index" not in entry.get("metadata", {})
        ]
        dataset_by_index = {
            entry["metadata"]["original_index"]: entry
            for entry in successful_dataset
            if "original_index" in entry.get("metadata", {})
        }

        def ordered_logs() -> List[Dict[str, Any]]:
            return unindexed_logs + [
                logs_by_index[index] for index in sorted(logs_by_index)
            ]

        def ordered_dataset() -> List[Dict[str, Any]]:
            return unindexed_dataset + [
                dataset_by_index[index] for index in sorted(dataset_by_index)
            ]

        def worker(item: QuestionWorkItem) -> Dict[str, Any]:
            item_config = config.copy()
            item_config["_TARGET_BENCHMARK_FOR_QUERY"] = benchmark_name_for_target_index(
                config, item.index
            )
            result = process_single_question(
                target_query=item.question, ground_truth=ground_truths[item.index], exemplar_data=exemplar_data,
                embedding_model=embedding_model, api_manager_solve=solver_mgr, api_manager_eval=eval_mgr, config=item_config,
            )
            result["original_index"] = item.index
            if distributed_enabled(config):
                apply_distributed_run_log_contract(
                    result,
                    index=item.index,
                    question=item.question,
                    completed=_result_completed(result),
                )
            return result
        def commit(results: List[QuestionResult], _batch_id: str, batch_number: int) -> None:
            for result in results:
                value = result.value
                value.setdefault("original_index", result.item.index)
                value.setdefault("target_query", result.item.question)
                logs_by_index[result.item.index] = value
                if value.get("status") == "SUCCESS" and value.get("dataset_entry"):
                    entry = value["dataset_entry"]
                    entry["metadata"] = {"original_index": result.item.index, "few_shot_ccs": value.get("few_shot_ccs"), "base_ccs": value.get("base_ccs")}
                    dataset_by_index[result.item.index] = entry

            # Save the dataset first. If the log commit is interrupted, the
            # index map de-duplicates successful entries when that batch retries.
            if not save_json_atomic(ordered_dataset(), dataset_path):
                raise RuntimeError("Failed to commit merging dataset")
            if not save_json_atomic(ordered_logs(), log_file_path):
                raise RuntimeError("Failed to commit merging run logs")
            periodic_batch_sync_check(batch_number - 1, config)
        BatchCoordinator(config, exp_name, "merging").run(
            [QuestionWorkItem(index=index, question=query) for index, query in queries_to_process if index in ground_truths], worker, commit
        )
        return ordered_logs()

    for loop_idx, (original_idx, query) in enumerate(tqdm(queries_to_process, desc=f"Merging DS: {exp_name}")):
        
        # Safely extract ground truth
        gt = None
        if hard_solutions and original_idx < len(hard_solutions):
            gt = hard_solutions[original_idx]
        elif 'ground_truths' in exemplar_data and original_idx < len(exemplar_data['ground_truths']):
            gt = exemplar_data['ground_truths'][original_idx]
        elif 'solutions' in exemplar_data and len(exemplar_data['solutions']) == len(hard_questions):
            gt = exemplar_data['solutions'][original_idx]
            
        if not gt:
            logger.warning(f"Skipping index {original_idx}: No ground truth available.")
            continue
            
        # Process the single query
        res = process_single_question(
            target_query=query,
            ground_truth=gt,
            exemplar_data=exemplar_data,
            embedding_model=embedding_model,
            api_manager_solve=solver_mgr,
            api_manager_eval=eval_mgr,
            config=config
        )
        
        # Add tracking metadata
        res['original_index'] = original_idx
        full_logs.append(res)
        
        # If accepted, add to the actual dataset file
        if res.get('status') == 'SUCCESS' and res.get('dataset_entry'):
            entry = res['dataset_entry']
            # Attach metadata to the final dataset entry for traceability
            entry['metadata'] = {
                "original_index": original_idx,
                "few_shot_ccs": res['few_shot_ccs'],
                "base_ccs": res['base_ccs']
            }
            successful_dataset.append(entry)
            
            # Save dataset progressively
            save_json(successful_dataset, dataset_path)
            print(f"\n   🌟 [ACCEPTED] Merging Dataset Size: {len(successful_dataset)}")
            
        # Save run logs progressively
        save_json(full_logs, log_file_path)
        periodic_sync_check(loop_idx, config)
        
    logger.info(f"Merging Dataset Construction Complete. Total valid dataset samples: {len(successful_dataset)}")
    return full_logs
