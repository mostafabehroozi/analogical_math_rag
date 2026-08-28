"""Construction of verified transformation-model training examples.

Each target is evaluated with three comparable conditions: zero-shot, the
unmodified top-1 retrieved exemplar, and each independently generated
transformation of that exemplar.  Only the best transformation is retained
when it strictly beats both baselines.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from src.api_manager import GeminiAPIManager, AvalAIAPIManager, OllamaAPIManager
from src.benchmark_data import benchmark_name_for_target_index
from src.batching import BatchCoordinator, QuestionResult, QuestionWorkItem
from src.distributed_execution import (
    apply_distributed_run_log_contract,
    distributed_enabled,
)
from src.evaluation import evaluate_single_answer_with_llm
from src.hf_sync import periodic_batch_sync_check, periodic_sync_check
from src.parallel_utils import run_parallel_api_calls
from src.pipeline_steps import retrieve
from src.prompts import (
    EXEMPLAR_FORMAT,
    create_final_reasoning_prompt,
    create_final_reasoning_prompt_simple,
    create_transformation_prompt,
)
from src.utils import load_json, save_json, save_json_atomic

logger = logging.getLogger(__name__)

_COMPLETED_STATUSES = {"SUCCESS", "REJECTED", "REJECTED_BY_FILTER"}


def _result_completed(result: Dict[str, Any]) -> bool:
    return str(result.get("status", "")).upper() in _COMPLETED_STATUSES


def _model_name(manager: Any, config: Dict[str, Any], role: str) -> str:
    """Resolve a model name while retaining support for lightweight test managers."""
    suffix = {
        "adaptation": "ADAPTATION",
        "solver": "FINAL_SOLVER",
        "evaluator": "EVALUATOR",
    }[role]
    if isinstance(manager, GeminiAPIManager):
        prefix = "GEMINI"
    elif isinstance(manager, AvalAIAPIManager):
        prefix = "AVALAI"
    elif isinstance(manager, OllamaAPIManager):
        prefix = "OLLAMA"
    else:
        prefix = str(config.get(f"API_PROVIDER_{role.upper()}", "gemini")).upper()
        if prefix == "SOLVER":
            prefix = "GEMINI"
    return config.get(f"{prefix}_MODEL_NAME_{suffix}", config.get("GEMINI_MODEL_NAME_FINAL_SOLVER"))


def _ccs(
    target_query: str,
    ground_truth: str,
    prompt: str,
    api_manager_solve: Any,
    api_manager_eval: Any,
    config: Dict[str, Any],
    trace_name: str,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Generate and evaluate a fixed number of independent CCS attempts."""
    del target_query  # Kept in the signature to make condition call sites explicit.
    n_attempts = max(1, int(config.get("TRANSFORMATION_DS_CCS_N", 5)))
    model_name = _model_name(api_manager_solve, config, "solver")
    temperature = config.get(
        "TRANSFORMATION_DS_SOLVER_TEMPERATURE",
        config.get("DEFAULT_FINAL_SOLVER_TEMPERATURE", 1.0),
    )

    def run_attempt(attempt_index: int) -> Dict[str, Any]:
        try:
            response = api_manager_solve.generate_content(prompt, model_name, temperature)
        except Exception as exc:  # Keep one failed attempt from losing the target.
            response = {"status": "FAILURE", "error_message": str(exc)}

        answer = response.get("text", "") if isinstance(response, dict) else ""
        is_correct = False
        evaluation = None
        if isinstance(response, dict) and response.get("status") == "SUCCESS" and answer:
            try:
                evaluation = evaluate_single_answer_with_llm(
                    answer, ground_truth, api_manager_eval, config
                )
                is_correct = bool(
                    evaluation.get("status") == "SUCCESS"
                    and evaluation.get("is_correct")
                )
            except Exception as exc:
                evaluation = {"status": "EVALUATION_EXCEPTION", "error_message": str(exc)}

        return {
            "attempt_index": attempt_index,
            "condition": trace_name,
            "response": response,
            "answer": answer,
            "is_correct": is_correct,
            "evaluation": evaluation,
        }

    tasks = [lambda index=i: run_attempt(index) for i in range(n_attempts)]
    attempts = run_parallel_api_calls(tasks, config)
    score = sum(bool(attempt.get("is_correct")) for attempt in attempts) / n_attempts
    return score, attempts


def _target_records(
    hard_questions: Optional[Sequence[str]],
    hard_solutions: Optional[Sequence[str]],
    exemplar_data: Dict[str, Any],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build stable target records for either supported target source."""
    source = str(config.get("TRANSFORMATION_DS_TARGET_SOURCE", "hard_questions")).lower()
    if source in {"exemplar", "exemplar_corpus", "corpus"}:
        questions = exemplar_data.get("questions", [])
        solutions = exemplar_data.get("solutions", [])
        source_name = "exemplar_corpus"
    else:
        questions = hard_questions or []
        solutions = hard_solutions or []
        source_name = "hard_questions"

    records: List[Dict[str, Any]] = []
    for index, question in enumerate(questions):
        ground_truth = solutions[index] if index < len(solutions) else None
        if not ground_truth and source_name == "hard_questions":
            if index < len(exemplar_data.get("ground_truths", [])):
                ground_truth = exemplar_data["ground_truths"][index]
            elif len(exemplar_data.get("solutions", [])) == len(questions):
                ground_truth = exemplar_data["solutions"][index]
        if question and ground_truth:
            records.append(
                {
                    "index": index,
                    "question": question,
                    "ground_truth": ground_truth,
                    "source": source_name,
                }
            )
    return records


def process_single_question(
    target: Dict[str, Any],
    exemplar_data: Dict[str, Any],
    embedding_model: Any,
    api_manager_adapt: Any,
    api_manager_solve: Any,
    api_manager_eval: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate one target and return its audit log plus optional dataset entry."""
    target_query = target["question"]
    ground_truth = target["ground_truth"]
    log: Dict[str, Any] = {
        "status": "PENDING",
        "original_index": target["index"],
        "target_source": target["source"],
        "target_query": target_query,
        "zero_shot_base_ccs": 0.0,
        "one_shot_base_ccs": 0.0,
        "candidate_evaluations": [],
        "selected_candidate_index": None,
        "dataset_entry": None,
        "error_info": None,
    }

    retrieval = retrieve(
        target_query=target_query,
        embedding_model=embedding_model,
        exemplar_questions=exemplar_data.get("questions", []),
        embedded_exemplars=exemplar_data.get("embeddings"),
        top_k=1,
        question_to_index_map=exemplar_data.get("question_to_index"),
    )
    retrieved_indices = retrieval.get("retrieved_indices", []) if retrieval else []
    if not retrieval or retrieval.get("status") != "SUCCESS" or not retrieved_indices:
        log.update({"status": "FAILURE", "error_info": "Top-1 retrieval failed."})
        return log

    retrieved_index = retrieved_indices[0]
    questions = exemplar_data.get("questions", [])
    solutions = exemplar_data.get("solutions", [])
    if retrieved_index >= len(questions) or retrieved_index >= len(solutions):
        log.update({"status": "FAILURE", "error_info": "Retrieved exemplar is incomplete."})
        return log

    retrieved_question = questions[retrieved_index]
    retrieved_solution = solutions[retrieved_index]
    original_sample = EXEMPLAR_FORMAT.format(
        question=retrieved_question, solution=retrieved_solution
    )
    similarity_scores = retrieval.get("retrieved_similarity_scores", [])
    retrieval_score = similarity_scores[0] if similarity_scores else None
    log["retrieved_exemplar"] = {
        "index": retrieved_index,
        "question": retrieved_question,
        "solution": retrieved_solution,
        "similarity": retrieval_score,
    }

    zero_prompt = create_final_reasoning_prompt_simple(target_query, config)
    one_shot_prompt = create_final_reasoning_prompt(target_query, [original_sample], config)
    zero_score, zero_attempts = _ccs(
        target_query, ground_truth, zero_prompt, api_manager_solve, api_manager_eval,
        config, "zero_shot",
    )
    one_score, one_attempts = _ccs(
        target_query, ground_truth, one_shot_prompt, api_manager_solve, api_manager_eval,
        config, "one_shot_base",
    )
    log["zero_shot_base_ccs"] = zero_score
    log["one_shot_base_ccs"] = one_score
    log["baseline_attempts"] = {
        "zero_shot": zero_attempts,
        "one_shot_base": one_attempts,
    }

    candidate_count = max(1, int(config.get("TRANSFORMATION_DS_N_CANDIDATES", 3)))
    transform_model = _model_name(api_manager_adapt, config, "adaptation")
    transform_temperature = config.get(
        "TRANSFORMATION_DS_TEMPERATURE",
        config.get("DEFAULT_ADAPTATION_TEMPERATURE", 0.0),
    )
    transform_tasks = []
    for candidate_index in range(candidate_count):
        transform_tasks.append(
            lambda index=candidate_index: _generate_transformation(
                index,
                target_query,
                original_sample,
                api_manager_adapt,
                transform_model,
                transform_temperature,
                config,
            )
        )
    generated_candidates = run_parallel_api_calls(transform_tasks, config)

    valid_candidates: List[Dict[str, Any]] = []
    for generated in generated_candidates:
        candidate_index = generated["candidate_index"]
        transformed_sample = generated.get("transformed_sample", "")
        candidate_log: Dict[str, Any] = {
            "candidate_index": candidate_index,
            "transformation_prompt": generated.get("transformation_prompt"),
            "transformation_response": generated.get("response"),
            "transformed_sample": transformed_sample,
            "transformed_aug_ccs": None,
            "solve_attempts": [],
            "status": "PENDING",
        }
        if not transformed_sample:
            candidate_log.update({"status": "FAILED", "error_info": "Empty transformation output."})
            log["candidate_evaluations"].append(candidate_log)
            continue

        transformed_prompt = create_final_reasoning_prompt(
            target_query, [transformed_sample], config
        )
        transformed_score, solve_attempts = _ccs(
            target_query, ground_truth, transformed_prompt,
            api_manager_solve, api_manager_eval, config,
            f"transformed_candidate_{candidate_index}",
        )
        candidate_log.update(
            {
                "status": "EVALUATED",
                "transformed_solver_prompt": transformed_prompt,
                "transformed_aug_ccs": transformed_score,
                "solve_attempts": solve_attempts,
            }
        )
        log["candidate_evaluations"].append(candidate_log)
        valid_candidates.append(
            {
                "candidate_index": candidate_index,
                "transformed_sample": transformed_sample,
                "transformation_prompt": generated["transformation_prompt"],
                "transformed_aug_ccs": transformed_score,
                "transformed_solver_prompt": transformed_prompt,
            }
        )

    if not valid_candidates:
        log.update({"status": "REJECTED", "error_info": "No usable transformation candidates."})
        return log

    best = max(
        valid_candidates,
        key=lambda candidate: (
            candidate["transformed_aug_ccs"],
            -candidate["candidate_index"],
        ),
    )
    log["selected_candidate_index"] = best["candidate_index"]
    log["selected_transformed_aug_ccs"] = best["transformed_aug_ccs"]

    if not (
        best["transformed_aug_ccs"] > zero_score
        and best["transformed_aug_ccs"] > one_score
    ):
        log["status"] = "REJECTED_BY_FILTER"
        log["error_info"] = "Best transformed CCS did not strictly beat both baselines."
        return log

    log["status"] = "SUCCESS"
    log["dataset_entry"] = {
        "input": best["transformation_prompt"],
        "output": best["transformed_sample"],
        "metadata": {
            "original_index": target["index"],
            "target_source": target["source"],
            "target_question": target_query,
            "ground_truth": ground_truth,
            "retrieved_index": retrieved_index,
            "retrieved_question": retrieved_question,
            "retrieved_solution": retrieved_solution,
            "retrieval_similarity": retrieval_score,
            "selected_candidate_index": best["candidate_index"],
            "zero_shot_base_ccs": zero_score,
            "one_shot_base_ccs": one_score,
            "transformed_aug_ccs": best["transformed_aug_ccs"],
            "ccs_n": max(1, int(config.get("TRANSFORMATION_DS_CCS_N", 5))),
            "transformation_template": config.get(
                "TRANSFORMATION_DS_PROMPT_TEMPLATE",
                "transformation_shallow-&-moderately-deep",
            ),
        },
    }
    return log


def _generate_transformation(
    candidate_index: int,
    target_query: str,
    original_sample: str,
    api_manager_adapt: Any,
    model_name: str,
    temperature: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = create_transformation_prompt(
        target_query,
        original_sample,
        config,
        "TRANSFORMATION_DS_PROMPT_TEMPLATE",
    )
    try:
        response = api_manager_adapt.generate_content(prompt, model_name, temperature)
    except Exception as exc:
        response = {"status": "FAILURE", "error_message": str(exc)}
    transformed_sample = ""
    if isinstance(response, dict) and response.get("status") == "SUCCESS":
        transformed_sample = str(response.get("text", "")).strip()
    return {
        "candidate_index": candidate_index,
        "transformation_prompt": prompt,
        "response": response,
        "transformed_sample": transformed_sample,
    }


def build_transformation_dataset(
    hard_questions: Optional[List[str]],
    hard_solutions: Optional[List[str]],
    exemplar_data: Dict[str, Any],
    embedding_model: Any,
    api_managers: Dict[str, Any],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build, persist, and resume the verified transformation dataset."""
    experiment_name = config.get("experiment_name", "transformation_dataset_construction")
    results_dir = config.get("RESULTS_DIR", ".")
    dataset_path = os.path.join(
        results_dir,
        config.get("TRANSFORMATION_DS_OUTPUT_FILENAME", "transformation_dataset.json"),
    )
    log_path = os.path.join(results_dir, f"{experiment_name}_run_log.json")

    dataset = load_json(dataset_path) or []
    logs = load_json(log_path) or []
    completed_indices = {
        entry.get("original_index")
        for entry in logs
        if "original_index" in entry and _result_completed(entry)
    }
    targets = _target_records(hard_questions, hard_solutions, exemplar_data, config)
    if not config.get("TRANSFORMATION_DS_DETERMINISTIC", True):
        rng = random.Random(config.get("TRANSFORMATION_DS_RANDOM_SEED"))
        rng.shuffle(targets)
    max_targets = config.get("TRANSFORMATION_DS_MAX_TARGETS")
    if max_targets is not None:
        targets = targets[: max(0, int(max_targets))]
    if distributed_enabled(config):
        owned_indices = set(config.get("_DISTRIBUTED_ALLOWED_QUERY_INDICES", []))
        targets = [target for target in targets if target["index"] in owned_indices]

    max_members = config.get("TRANSFORMATION_DS_MAX_MEMBERS")
    if max_members is not None and len(dataset) >= int(max_members):
        return logs
    items = [
        QuestionWorkItem(index=target["index"], question=target["question"])
        for target in targets
        if target["index"] not in completed_indices
    ]
    target_by_index = {target["index"]: target for target in targets}

    adapt_manager = api_managers.get(config.get("API_PROVIDER_ADAPTATION", "gemini"))
    solve_manager = api_managers.get(config.get("API_PROVIDER_SOLVER", "gemini"))
    eval_manager = api_managers.get(config.get("API_PROVIDER_EVALUATOR", "gemini"))

    def worker(item: QuestionWorkItem) -> Dict[str, Any]:
        item_config = config.copy()
        item_config["_TARGET_BENCHMARK_FOR_QUERY"] = benchmark_name_for_target_index(
            config, item.index
        )
        result = process_single_question(
            target_by_index[item.index], exemplar_data, embedding_model,
            adapt_manager, solve_manager, eval_manager, item_config,
        )
        if distributed_enabled(config):
            apply_distributed_run_log_contract(
                result,
                index=item.index,
                question=item.question,
                completed=_result_completed(result),
            )
        return result

    if distributed_enabled(config) and not items:
        if not os.path.exists(dataset_path) and not save_json_atomic(dataset, dataset_path):
            raise RuntimeError("Failed to initialize transformation dataset shard")
        if not os.path.exists(log_path) and not save_json_atomic(logs, log_path):
            raise RuntimeError("Failed to initialize transformation run-log shard")
        return logs

    if config.get("BATCH_PROCESSING_ENABLED", False) and items:
        logs_by_index = {
            entry["original_index"]: entry
            for entry in logs if "original_index" in entry
        }
        dataset_by_index = {
            entry["metadata"]["original_index"]: entry
            for entry in dataset
            if isinstance(entry.get("metadata"), dict)
            and "original_index" in entry["metadata"]
        }
        unindexed_logs = [entry for entry in logs if "original_index" not in entry]
        unindexed_dataset = [
            entry for entry in dataset
            if not isinstance(entry.get("metadata"), dict)
            or "original_index" not in entry["metadata"]
        ]

        def ordered_logs() -> List[Dict[str, Any]]:
            return unindexed_logs + [logs_by_index[index] for index in sorted(logs_by_index)]

        def ordered_dataset() -> List[Dict[str, Any]]:
            return unindexed_dataset + [dataset_by_index[index] for index in sorted(dataset_by_index)]

        def commit(results: List[QuestionResult], _batch_id: str, batch_number: int) -> None:
            for result in results:
                value = result.value
                value.setdefault("original_index", result.item.index)
                logs_by_index[result.item.index] = value
                if value.get("status") == "SUCCESS" and value.get("dataset_entry"):
                    accepted_count = len(unindexed_dataset) + len(dataset_by_index)
                    if max_members is None or accepted_count < int(max_members):
                        dataset_by_index[result.item.index] = value["dataset_entry"]
            if not save_json_atomic(ordered_dataset(), dataset_path):
                raise RuntimeError("Failed to commit transformation dataset")
            if not save_json_atomic(ordered_logs(), log_path):
                raise RuntimeError("Failed to commit transformation run log")
            periodic_batch_sync_check(batch_number - 1, config)

        BatchCoordinator(config, experiment_name, "transformation_dataset").run(
            items, worker, commit
        )
        return ordered_logs()

    for loop_index, item in enumerate(
        tqdm(items, desc=f"Transformation DS: {experiment_name}")
    ):
        result = worker(item)
        logs.append(result)
        save_json(logs, log_path)
        if result.get("status") == "SUCCESS" and result.get("dataset_entry"):
            dataset.append(result["dataset_entry"])
            save_json(dataset, dataset_path)
            if max_members is not None and len(dataset) >= int(max_members):
                periodic_sync_check(loop_index, config)
                break
        periodic_sync_check(loop_index, config)
    return logs
