"""
Error handling, reporting, and retry module for the Analogical Reasoning RAG project.

This module provides functions to:
- NEW: Retry generation pipelines that failed midway through.
- Retry failed LLM-based evaluation attempts with more granularity.
- Generate a comprehensive report of all API and processing errors that occurred
  during both generation and evaluation phases.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from tqdm import tqdm

from src.evaluation import evaluate_single_answer_with_llm
from src.orchestration import run_pipeline_for_single_query
from src.utils import load_json, save_json
from src.hf_sync import periodic_sync_check, sync_workspace_to_hub
from src.layer1_base_execution import _get_cache_filename, _get_cache_path


_LAYER1_REQUIRED_LAYER2_FIELDS = (
    "retrieved_set",
    "candidate_set",
    "intrinsic_baselines",
    "cross_evaluation_matrix",
    "ground_truth_labels",
)


def _coerce_query_index(value: Any) -> Optional[int]:
    """Return a hard-question index as int, or None when it cannot be trusted."""
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_log_query_index(log: Dict[str, Any]) -> Optional[int]:
    if not isinstance(log, dict):
        return None
    return _coerce_query_index(log.get("target_query_original_hard_list_idx"))


def _has_non_empty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple, set, str)):
        return len(value) > 0
    return True


def _is_incomplete_layer1_state(layer1_state: Any) -> bool:
    """
    True when a Layer-1 state is missing data that Layer 2 requires.

    A checkpoint skeleton may exist in the combined cache before the final
    state is saved. Layer 2 needs the final data fields, not just the query key.
    """
    if not isinstance(layer1_state, dict):
        return True

    step_statuses = layer1_state.get("step_statuses", {})
    steps_all_success = False
    if isinstance(step_statuses, dict) and step_statuses:
        if any(str(step_status).upper() != "SUCCESS" for step_status in step_statuses.values()):
            return True
        steps_all_success = True

    missing_layer2_fields = any(
        not _has_non_empty_value(layer1_state.get(field))
        for field in _LAYER1_REQUIRED_LAYER2_FIELDS
    )
    if missing_layer2_fields:
        return True

    status = str(layer1_state.get("overall_status", "")).upper()
    if status and status != "SUCCESS":
        return True

    # Legacy Layer-1 final states can contain complete data while metadata still
    # says last_completed_step=-1 because checkpoints update the file, not the
    # in-memory state later written by _save_cached_state().
    if status == "SUCCESS" or steps_all_success:
        return False

    last_completed_step = _coerce_query_index(
        layer1_state.get("metadata", {}).get("last_completed_step")
    )
    return last_completed_step is None or last_completed_step < 4


def _log_needs_generation_retry(log: Dict[str, Any]) -> bool:
    if not isinstance(log, dict):
        return False

    status = str(log.get("pipeline_status", "")).upper()
    if "FAILURE" in status or "PARTIAL" in status:
        return True

    if "layer1_base_execution_state" in log:
        return _is_incomplete_layer1_state(log["layer1_base_execution_state"])

    return False


def _validate_manual_retry_indices(
    manual_retry_query_indices: Optional[List[int]],
    hard_questions: List[str],
    logger: logging.Logger
) -> Set[int]:
    if not manual_retry_query_indices:
        return set()

    valid_indices = set()
    for raw_idx in manual_retry_query_indices:
        idx = _coerce_query_index(raw_idx)
        if idx is None:
            logger.warning(f"Skipping invalid manual retry index: {raw_idx!r}")
            continue
        if idx < 0 or idx >= len(hard_questions):
            logger.warning(
                f"Skipping out-of-range manual retry index #{idx}. "
                f"Expected 0 <= index < {len(hard_questions)}."
            )
            continue
        valid_indices.add(idx)

    return valid_indices


def _remove_index_from_metadata_list(
    metadata: Dict[str, Any],
    field_name: str,
    query_index: int
) -> bool:
    values = metadata.get(field_name)
    if not isinstance(values, list):
        return False

    filtered_values = [
        value for value in values
        if _coerce_query_index(value) != query_index
    ]
    if len(filtered_values) == len(values):
        return False

    metadata[field_name] = filtered_values
    return True


def _remove_layer1_cache_entry(
    cache_path: str,
    query_index: int,
    logger: logging.Logger
) -> bool:
    """
    Remove a query's stale Layer-1 state so retry starts from the root.

    This also cleans metadata lists that may otherwise mark the query complete
    or in-progress even after the query payload is deleted.
    """
    if not os.path.exists(cache_path):
        return False

    cache_data = load_json(cache_path)
    if not isinstance(cache_data, dict):
        logger.warning(f"Could not load Layer-1 cache for repair: {cache_path}")
        return False

    changed = False
    query_key = str(query_index)
    queries = cache_data.get("queries")
    if isinstance(queries, dict) and query_key in queries:
        del queries[query_key]
        changed = True

    metadata = cache_data.get("metadata")
    if isinstance(metadata, dict):
        changed = _remove_index_from_metadata_list(metadata, "completed_queries", query_index) or changed
        changed = _remove_index_from_metadata_list(metadata, "queries_in_progress", query_index) or changed
        if isinstance(queries, dict):
            metadata["total_queries"] = len(queries)
        metadata["updated_at"] = time.time()

    if changed:
        if save_json(cache_data, cache_path):
            logger.info(f"Removed stale Layer-1 cache entry for query #{query_index}: {cache_path}")
        else:
            logger.error(f"Failed to save repaired Layer-1 cache metadata: {cache_path}")

    return changed


def _collect_retry_targets_from_logs(
    logs: List[Dict[str, Any]],
    layer1_cache: Optional[Dict[str, Any]],
    layer1_enabled: bool,
    logger: logging.Logger,
    exp_name: str
) -> Set[int]:
    retry_targets = set()
    cache_queries = {}
    if isinstance(layer1_cache, dict):
        cache_queries = layer1_cache.get("queries", {}) or {}

    for log in logs:
        query_index = _get_log_query_index(log)
        if query_index is None:
            if _log_needs_generation_retry(log):
                logger.warning(
                    f"Skipping bad log without a usable query index in experiment '{exp_name}'."
                )
            continue

        if _log_needs_generation_retry(log):
            retry_targets.add(query_index)
            continue

        if layer1_enabled and str(query_index) in cache_queries:
            if _is_incomplete_layer1_state(cache_queries[str(query_index)]):
                logger.info(
                    f"Query #{query_index} in '{exp_name}' has an incomplete Layer-1 cache entry."
                )
                retry_targets.add(query_index)

    return retry_targets


def _upsert_run_log(
    logs: List[Dict[str, Any]],
    new_run_log: Dict[str, Any],
    query_index: int
) -> List[Dict[str, Any]]:
    """Replace the first log for query_index, remove duplicates, or append if missing."""
    updated_logs = []
    inserted = False

    for log in logs:
        if _get_log_query_index(log) == query_index:
            if not inserted:
                updated_logs.append(new_run_log)
                inserted = True
            continue
        updated_logs.append(log)

    if not inserted:
        updated_logs.append(new_run_log)

    return updated_logs


def _upsert_layer1_cache_state(
    cache_path: str,
    query_index: int,
    layer1_state: Dict[str, Any],
    top_k: int,
    n_candidates: int,
    experiment_name: str,
    logger: logging.Logger
) -> bool:
    """Persist the retried Layer-1 state into the combined cache file."""
    if not isinstance(layer1_state, dict):
        logger.warning(
            f"Cannot update Layer-1 cache for query #{query_index}: "
            "run log does not contain a dictionary Layer-1 state."
        )
        return False

    cache_data = load_json(cache_path) if os.path.exists(cache_path) else None
    if not isinstance(cache_data, dict):
        cache_data = {
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

    if not isinstance(cache_data.get("queries"), dict):
        cache_data["queries"] = {}

    target_query_data = layer1_state.get("target_query_data", {})
    layer1_state.setdefault("target_query_idx", query_index)
    if isinstance(target_query_data, dict):
        layer1_state.setdefault("target_query_text", target_query_data.get("query_text"))
        layer1_state.setdefault("ground_truth_answer", target_query_data.get("ground_truth_answer"))

    query_key = str(query_index)
    cache_data["queries"][query_key] = layer1_state

    metadata = cache_data.setdefault("metadata", {})
    metadata["experiment_name"] = experiment_name
    metadata["top_k"] = top_k
    metadata["n_candidates"] = n_candidates
    metadata["updated_at"] = time.time()
    metadata["total_queries"] = len(cache_data["queries"])

    completed_queries = metadata.get("completed_queries")
    if not isinstance(completed_queries, list):
        completed_queries = []

    queries_in_progress = metadata.get("queries_in_progress")
    if not isinstance(queries_in_progress, list):
        queries_in_progress = []

    completed_set = {
        idx for idx in (_coerce_query_index(value) for value in completed_queries)
        if idx is not None and idx != query_index
    }
    in_progress_set = {
        idx for idx in (_coerce_query_index(value) for value in queries_in_progress)
        if idx is not None and idx != query_index
    }

    if _is_incomplete_layer1_state(layer1_state):
        in_progress_set.add(query_index)
    else:
        completed_set.add(query_index)

    metadata["completed_queries"] = sorted(completed_set)
    metadata["queries_in_progress"] = sorted(in_progress_set)

    if save_json(cache_data, cache_path):
        logger.info(f"Upserted Layer-1 cache state for query #{query_index}: {cache_path}")
        return True

    logger.error(f"Failed to upsert Layer-1 cache state for query #{query_index}: {cache_path}")
    return False


def retry_failed_generation_pipelines(
    all_experiments_logs: Dict[str, List[Dict]],
    global_config: Dict[str, Any],
    hard_questions: List[str],
    embedding_model: Any,
    exemplar_data: Dict[str, Any],
    api_managers: Dict[str, Any],
    manual_retry_query_indices: Optional[List[int]] = None,
    hard_solutions: Optional[List[str]] = None
) -> Dict[str, List[Dict]]:
    """
    Retry failed generation logs and explicitly requested hard-question indices.

    Manual indices are hard-question indices, not positions in the loaded run-log
    list. They are retried even if absent from both the run log and Layer-1 cache.

    ``hard_solutions`` is optional for backward compatibility, but callers that
    load an external benchmark should pass it. This keeps retry ground truths
    aligned with the original run instead of falling back to exemplar solutions.
    """
    
    logger = logging.getLogger(__name__)
    logger.info("Starting SMART retry process for failed/partial generation pipelines.")
    manual_retry_targets = _validate_manual_retry_indices(
        manual_retry_query_indices,
        hard_questions,
        logger
    )
    
    for exp_name, original_logs in all_experiments_logs.items():
        log_file_path = os.path.join(global_config['RESULTS_DIR'], f"{exp_name}_run_log.json")
        disk_logs = load_json(log_file_path)
        if isinstance(disk_logs, list):
            logs_to_process = disk_logs
        elif isinstance(original_logs, list):
            logs_to_process = original_logs
        elif original_logs:
            logger.warning(
                f"Skipping generation retry for '{exp_name}' because its logs are not a list."
            )
            continue
        else:
            logs_to_process = []

        exp_config_overrides = logs_to_process[0].get("config_flags_used", {}) if logs_to_process else {}
        current_config = global_config.copy()
        current_config.update(exp_config_overrides)
        current_config["experiment_name"] = exp_name

        cache_dir = (
            current_config.get("LAYER1_CACHE_DIR")
            or current_config.get("RESULTS_DIR")
            or global_config["RESULTS_DIR"]
        )
        top_k = current_config.get("TOP_N_CANDIDATES_RETRIEVAL", 5)
        n_candidates = current_config.get("LAYER1_N_CANDIDATES") or top_k
        cache_filename = _get_cache_filename(top_k, n_candidates, exp_name)
        cache_path = _get_cache_path(cache_dir, cache_filename)

        l1_cache = load_json(cache_path) if os.path.exists(cache_path) else None
        retry_targets = _collect_retry_targets_from_logs(
            logs=logs_to_process,
            layer1_cache=l1_cache,
            layer1_enabled=bool(current_config.get("APPLY_LAYER1_BASE_EXECUTION", False)),
            logger=logger,
            exp_name=exp_name
        )
        retry_targets.update(manual_retry_targets)

        valid_retry_targets = set()
        for query_index in retry_targets:
            if 0 <= query_index < len(hard_questions):
                valid_retry_targets.add(query_index)
            else:
                logger.warning(
                    f"Skipping out-of-range retry target #{query_index} for '{exp_name}'."
                )

        if not valid_retry_targets:
            logger.info(f"No generation retry targets found for '{exp_name}'.")
            all_experiments_logs[exp_name] = logs_to_process
            continue

        logger.info(
            f"Retrying {len(valid_retry_targets)} generation pipeline(s) for '{exp_name}': "
            f"{sorted(valid_retry_targets)}"
        )

        for loop_idx, original_hard_list_idx in enumerate(
            tqdm(sorted(valid_retry_targets), desc=f"Retrying {exp_name}")
        ):
            target_query = hard_questions[original_hard_list_idx]

            _remove_layer1_cache_entry(cache_path, original_hard_list_idx, logger)

            # Force the pipeline to run all the way through, even if the experiment
            # originally deferred the solving step to phase 2.
            retry_config = current_config.copy()
            retry_config["DEFER_SOLVE_STEP"] = False
            retry_config["experiment_name"] = exp_name

            new_run_log = run_pipeline_for_single_query(
                hard_list_idx=original_hard_list_idx,
                target_query=target_query,
                config=retry_config,
                embedding_model=embedding_model,
                exemplar_data=exemplar_data,
                api_managers=api_managers,
                # UPDATE: Preserve benchmark-answer alignment during retries.
                hard_solutions=hard_solutions
            )

            if current_config.get("APPLY_LAYER1_BASE_EXECUTION", False):
                _upsert_layer1_cache_state(
                    cache_path=cache_path,
                    query_index=original_hard_list_idx,
                    layer1_state=new_run_log.get("layer1_base_execution_state"),
                    top_k=top_k,
                    n_candidates=n_candidates,
                    experiment_name=exp_name,
                    logger=logger
                )

            logs_to_process = _upsert_run_log(logs_to_process, new_run_log, original_hard_list_idx)
            all_experiments_logs[exp_name] = logs_to_process
            save_json(logs_to_process, log_file_path)
            periodic_sync_check(loop_idx, current_config)

        save_json(logs_to_process, log_file_path)
        all_experiments_logs[exp_name] = logs_to_process

    return all_experiments_logs
# --- UPGRADED: Evaluation Phase Retry Logic ---

def retry_failed_evaluations(
    all_experiments_logs: Dict[str, List[Dict]],
    ground_truths: List[str],
    api_managers: Dict[str, Any],
    config: Dict[str, Any]
) -> None:
    """
    Retries failed evaluation attempts for all experiments.

    This function now retries any evaluation that did not achieve a 'SUCCESS' status,
    including API errors, parsing failures, etc. It updates the detailed evaluation
    logs in place.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting the process to retry failed evaluation attempts.")
    results_dir = config['RESULTS_DIR']

    for exp_name, query_logs in all_experiments_logs.items():
        eval_file_path = os.path.join(results_dir, f"{exp_name}_detailed_eval.json")
        detailed_evaluations = load_json(eval_file_path)

        if not detailed_evaluations:
            logger.warning(f"No detailed evaluation log found for '{exp_name}'. Skipping retry.")
            continue

        logger.info(f"--- Retrying evaluations for Experiment: {exp_name} ---")
        
        # UPDATE: Honor the evaluator provider recorded by the original
        # experiment, with global config as a backward-compatible fallback.
        experiment_config = config.copy()
        if query_logs and isinstance(query_logs[0], dict):
            logged_config = query_logs[0].get("config_flags_used", {})
            if isinstance(logged_config, dict):
                experiment_config.update(logged_config)

        provider_for_eval = experiment_config.get('API_PROVIDER_EVALUATOR', 'gemini')
        manager_for_eval = api_managers[provider_for_eval]

        # Find all attempts across all queries that did not succeed
        retries_needed = []
        for eval_idx, log in enumerate(detailed_evaluations):
            for attempt_idx, status in enumerate(log.get("evaluation_status_list", [])):
                if status != "SUCCESS":
                    retries_needed.append((eval_idx, attempt_idx))

        if not retries_needed:
            logger.info(f"No failed evaluations found in '{exp_name}' to retry.")
            continue

        for loop_idx, (eval_idx, attempt_idx) in enumerate(tqdm(retries_needed, desc=f"Retrying evaluations for {exp_name}")):
            log = detailed_evaluations[eval_idx]
            hard_list_idx = log["hard_list_idx"]
            ground_truth = ground_truths[hard_list_idx]
            
            # The attempt can be a string (successful generation) or a dict (failed generation)
            attempt_text_or_dict = log["attempts"][attempt_idx]

            # Only retry if the generation was successful in the first place
            if isinstance(attempt_text_or_dict, str):
                print(f"    -> Retrying evaluation for query #{hard_list_idx}, attempt #{attempt_idx + 1}")
                
                # MODIFIED: Use the selected manager for evaluation
                eval_result = evaluate_single_answer_with_llm(
                    attempt_text_or_dict,
                    ground_truth,
                    manager_for_eval,
                    experiment_config,
                )
                
                # Update the log in place
                log["is_correct_list"][attempt_idx] = eval_result["is_correct"]
                log["evaluation_status_list"][attempt_idx] = eval_result["status"]
                log["evaluation_error_details"][attempt_idx] = eval_result["error_details"]
                print(f"       New status: {eval_result['status']}")

            # Persist changes after each retry
            save_json(detailed_evaluations, eval_file_path)
            periodic_sync_check(loop_idx, experiment_config)

        # Final save and sync
        save_json(detailed_evaluations, eval_file_path)
        sync_workspace_to_hub(experiment_config)

    logger.info("Finished retrying all failed evaluations across all experiments.")


# --- UPGRADed: Comprehensive Error Reporting ---

def generate_error_report(
    all_experiments_logs: Dict[str, List[Dict]],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Generates a detailed report of all errors from both generation and evaluation logs.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating a comprehensive error report.")
    error_records = []
    results_dir = config['RESULTS_DIR']

    for exp_name, query_logs in all_experiments_logs.items():
        # --- 1. Scan Generation Logs for Errors ---
        for log in query_logs:
            hard_idx = log['target_query_original_hard_list_idx']
            if "FAILURE" in log.get("pipeline_status", ""):
                # Find the first failing step
                for step_name, step_result in log.get("steps", {}).items():
                    if "FAILURE" in step_result.get("status", "") or "PARTIAL_SUCCESS" in step_result.get("status", ""):
                        # Handle failures within steps that process multiple items, like adapt
                        failures = step_result.get("failed_adaptations", []) + step_result.get("failed_merges", [])
                        if failures:
                            for failure in failures:
                                error_info = failure.get("error_info", {})
                                error_records.append({
                                    "phase": "generation", "experiment_name": exp_name, "query_hard_list_idx": hard_idx,
                                    "pipeline_step": step_name, "attempt_number": failure.get("source_index", "N/A"),
                                    "error_type": error_info.get("error_type"), "error_message": error_info.get("error_message")
                                })
                        # Handle failures in the final solve step
                        elif step_name == "solving":
                            for i, attempt in enumerate(step_result.get("solution_attempts", [])):
                                if isinstance(attempt, dict) and "FAILURE" in attempt.get("status", ""):
                                    error_info = attempt.get("error_info", {})
                                    error_records.append({
                                        "phase": "generation", "experiment_name": exp_name, "query_hard_list_idx": hard_idx,
                                        "pipeline_step": "solve", "attempt_number": i + 1,
                                        "error_type": error_info.get("error_type"), "error_message": error_info.get("error_message")
                                    })
                        break # Found the first major failing step for this log

        # --- 2. Scan Evaluation Logs for Errors ---
        eval_file_path = os.path.join(results_dir, f"{exp_name}_detailed_eval.json")
        detailed_evaluations = load_json(eval_file_path)
        if detailed_evaluations:
            for log in detailed_evaluations:
                hard_idx = log["hard_list_idx"]
                for i, status in enumerate(log.get("evaluation_status_list", [])):
                    if status != "SUCCESS":
                        error_details = log.get("evaluation_error_details", [])[i] or {}
                        error_records.append({
                            "phase": "evaluation", "experiment_name": exp_name, "query_hard_list_idx": hard_idx,
                            "pipeline_step": "evaluate_answer", "attempt_number": i + 1,
                            "error_type": status, "error_message": error_details.get("error_message", "N/A")
                        })

    if not error_records:
        logger.info("No errors were found in any experiment logs.")
        return pd.DataFrame()

    return pd.DataFrame(error_records)
