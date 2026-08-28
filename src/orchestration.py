# src/orchestration.py

"""
Orchestration module for the Analogical Reasoning RAG project.
This module chains together the individual steps from `pipeline_steps.py` to
run the full RAG pipeline.
It manages running experiments for multiple queries
and configurations, and handles the saving and loading of results for
pausing and resuming.
This version is updated to be API provider-agnostic and to handle detailed
error states from the pipeline steps.
It logs partial progress when failures
occur, preventing data loss and enabling targeted retries.
MODIFIED: This version now supports a deferred execution mode via the
`DEFER_SOLVE_STEP` config flag.
When enabled, it runs all intermediate steps
(retrieve and adapt) for all queries first, then runs the final solve
step for all queries in a second phase.
REWRITTEN: The `run_experiments` function now implements a global, cross-experiment
deferred execution.
If ANY experiment has `DEFER_SOLVE_STEP` set to True, the
entire run switches to a two-phase model:
1. Phase 1: All intermediate steps for ALL experiments are completed.
2. Phase 2: All final solving steps for ALL experiments are completed.
This optimizes API usage by batching all expensive 'solve' calls together.
PERFORMANCE FIX: The call to the `retrieve` function has been updated to pass
a pre-computed hash map, enabling O(1) self-match detection and resolving a
major performance bottleneck.
NEW FEATURE: Added `APPLY_FULL_PIPELINE_RETRY`. If True, the entire pipeline
(Retrieval -> Adaptation -> Solving) is re-run N times, rather than
just retrying the final Solver step N times.
NEW FEATURE: Added Pipeline Simplification.
- Workflow A: Simplification of Retrieved Samples (replaces standard adaptation).
- Workflow B: Simplification of Main Question (alters solving strategy).
NEW FEATURE: Multi-Strategy Branching (Mirroring).
- Supports branching the pipeline after Mirroring to evaluate "Base" vs "Redundancy"
  filtering strategies in parallel.
NEW FEATURE: High-Resolution Execution Tracing.
- Collects granular prompt/response logs from all steps and aggregates them
  into a master `execution_trace` list.
ARCHITECTURAL REFACTOR: "Acquire-Optimize-Fork".
- Implements a strict bifurcation between Benchmarking (Track A) and Validation (Track B)
  when MIRROR_AS_EVALUATOR is active.
"""

import logging
from tqdm import tqdm
import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from src.context_logger import tprint
from src.benchmark_data import benchmark_name_for_target_index


from src.pipeline_steps import (
    retrieve, adapt, solve,
    solve_with_analogical_consistency, 
    simplify_retrieved_samples,
    solve_via_main_simplification,
    optimize_demonstrations_via_mirroring,
    apply_mirror_reranking,
    reverse_transform_and_solve,
    select_best_transformations
)

from src.layer1_base_execution import (
    run_layer1_base_execution, 
    _save_cached_state, 
    _save_cached_states_batch,
    _get_cache_filename, 
    _get_cache_path
)
from src.layer2_integration import run_layer2_complete_pipeline

from src.utils import save_json, save_json_atomic, load_json
from src.hf_sync import (
    download_distributed_run_for_merge,
    ensure_distributed_manifest,
    initialize_workspace,
    periodic_sync_check,
    periodic_batch_sync_check,
    sync_distributed_merged_results,
    sync_workspace_to_hub,
)
from src.batching import BatchCoordinator, QuestionWorkItem, QuestionResult
from src.distributed_execution import (
    DistributedExecutionError,
    assigned_indices,
    build_run_manifest,
    configure_worker_paths,
    distributed_enabled,
    fingerprint_exemplar_data,
    indexed_output_artifacts,
    layer1_state_complete,
    merge_distributed_run,
    record_avalai_model_provenance,
    resolve_code_fingerprint,
    run_log_successful,
    start_worker_session,
    stop_new_batches_due,
    validate_worker_layer1_cache,
    validate_worker_run_logs,
    validate_experiment_name,
    validate_manifest_compatibility,
    write_or_validate_manifest,
    write_worker_status,
)

import builtins
def safe_thread_print(*args, **kwargs):
    # If a special print argument is used, fallback to normal Python print safely
    if 'file' in kwargs or kwargs.get('end', '\n') != '\n':
        builtins.print(*args, **kwargs)
        return
    sep = kwargs.get('sep', ' ')
    message = sep.join(str(a) for a in args)
    tprint(message, level="INFO")


def _announce_authorized_model_rotation(
    model_changes: Dict[str, Any],
    *,
    context: str,
) -> None:
    if not model_changes:
        return
    message = (
        f"AUTHORIZED MODEL ROTATION ({context}): the immutable run manifest is retained; "
        f"completed results may use the original models while new work uses active models. "
        f"Changes: {model_changes}"
    )
    logging.getLogger(__name__).warning(message)
    builtins.print(f"\n[WARNING] {message}\n", flush=True)


def _build_requested_distributed_manifest(
    global_config: Dict[str, Any],
    experiment_configs: List[Dict[str, Any]],
    hard_questions: List[str],
    hard_solutions: Optional[List[str]],
    exemplar_data: Dict[str, Any],
) -> Dict[str, Any]:
    return build_run_manifest(
        global_config,
        experiment_configs,
        hard_questions,
        hard_solutions,
        code_fingerprint=global_config.get("DISTRIBUTED_CODE_FINGERPRINT"),
        exemplar_fingerprint=fingerprint_exemplar_data(exemplar_data),
    )


def _auto_pin_local_legacy_code_fingerprint(
    global_config: Dict[str, Any],
    manifest_path: str,
    requested_manifest: Dict[str, Any],
) -> bool:
    """Pin a valid local pre-patch manifest only when model names are the sole drift."""
    if (
        not global_config.get("DISTRIBUTED_ALLOW_MODEL_ROTATION", False)
        or global_config.get("DISTRIBUTED_CODE_FINGERPRINT")
        or not os.path.exists(manifest_path)
    ):
        return False
    existing_manifest = load_json(manifest_path)
    if not isinstance(existing_manifest, dict):
        raise DistributedExecutionError(
            f"Malformed existing distributed manifest: {manifest_path}"
        )
    model_changes = validate_manifest_compatibility(
        existing_manifest,
        requested_manifest,
        allow_model_rotation=True,
        allow_legacy_code_fingerprint=True,
    )
    stored_code_fingerprint = existing_manifest.get("code_fingerprint")
    if stored_code_fingerprint == requested_manifest.get("code_fingerprint"):
        return False
    if not isinstance(stored_code_fingerprint, str) or not stored_code_fingerprint:
        raise DistributedExecutionError(
            "The existing distributed manifest has no usable code_fingerprint."
        )
    global_config["DISTRIBUTED_CODE_FINGERPRINT"] = stored_code_fingerprint
    global_config["_DISTRIBUTED_MODEL_ROTATION_CODE_FINGERPRINT_AUTO_PINNED"] = True
    logging.getLogger(__name__).warning(
        "Automatically pinned the pre-patch code fingerprint from %s after "
        "verifying that only approved model names changed: %s",
        manifest_path,
        model_changes,
    )
    return True


print = safe_thread_print


_REMOVED_FEATURE_FLAGS = {
    "APPLY_MULTIBRANCH_TRANSFORMATION": "Multi-Branch Transformation experiment",
    "APPLY_DATASET_CONSTRUCTION": "legacy validated two-shot dataset construction",
}


def _reject_removed_feature_flags(config: Dict[str, Any]) -> None:
    """Fail clearly when a configuration tries to enable a removed feature."""
    for flag_name, feature_name in _REMOVED_FEATURE_FLAGS.items():
        if config.get(flag_name):
            raise ValueError(
                f"{flag_name}=True requests the removed {feature_name}. "
                "Remove this stale activation from the configuration."
            )


_DISTRIBUTED_UNSUPPORTED_FLAGS = {
    "DEFER_SOLVE_STEP": "deferred solve mode",
}
_DISTRIBUTED_RUNTIME_PATH_KEYS = {
    "RESULTS_DIR", "LAYER1_CACHE_DIR", "LOGS_DIR", "OUTPUTS_DIR",
}


def _merge_experiment_config(
    global_config: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge one experiment while retaining coordinator-owned worker paths."""
    merged = global_config.copy()
    merged.update(overrides)
    if distributed_enabled(global_config):
        for key, value in global_config.items():
            if (
                key in _DISTRIBUTED_RUNTIME_PATH_KEYS
                or key.startswith("DISTRIBUTED_")
                or key.startswith("_DISTRIBUTED_")
            ):
                merged[key] = value
    return merged


def _validate_distributed_scope(
    global_config: Dict[str, Any],
    experiment_configs: List[Dict[str, Any]],
    hard_solutions: Optional[List[str]],
) -> None:
    """Validate shard-safe workflows and their immutable input/output contracts."""
    if not global_config.get("PERSIST_RESULTS_ONLINE", False):
        raise DistributedExecutionError(
            "Distributed execution requires PERSIST_RESULTS_ONLINE=True."
        )
    for setting in ("HF_SYNC_TOKEN", "HF_HUB_USERNAME", "HF_HUB_REPO_NAME"):
        value = global_config.get(setting)
        normalized = str(value or "").strip().lower()
        is_placeholder = (
            "your_hugging_face" in normalized
            or normalized in {"your-hf-username-here", "your-hf-repo-name-here"}
        )
        if not value or is_placeholder:
            raise DistributedExecutionError(
                f"{setting} must be supplied from Kaggle Secrets/environment before "
                "distributed execution starts."
            )
    experiment_names = set()
    published_result_filenames = set()
    for overrides in experiment_configs:
        for key in overrides:
            if key.startswith("DISTRIBUTED_") and overrides[key] != global_config.get(key):
                raise DistributedExecutionError(
                    f"{key} is worker/run identity and cannot be overridden per experiment."
                )
        effective = _merge_experiment_config(global_config, overrides)
        experiment_name = validate_experiment_name(effective.get("experiment_name"))
        if experiment_name in experiment_names:
            raise DistributedExecutionError(
                f"Duplicate distributed experiment_name: {experiment_name}"
            )
        experiment_names.add(experiment_name)
        special_families = []
        if effective.get("APPLY_TRANSFORMATION_DATASET_CONSTRUCTION", False):
            special_families.append("transformation dataset construction")
        if effective.get("APPLY_MERGING_DATASET_CONSTRUCTION", False):
            special_families.append("merging dataset construction")
        if effective.get("APPLY_CORE_SIMP_PHASE1", False) or effective.get(
            "APPLY_CORE_SIMP_PHASE2", False
        ):
            special_families.append("Core Simplification")
        if len(special_families) > 1:
            raise DistributedExecutionError(
                f"Distributed experiment {experiment_name!r} enables multiple special "
                f"workflow families: {', '.join(special_families)}. Split them into "
                "separate experiment configurations."
            )
        if effective.get("APPLY_CORE_SIMP_PHASE2", False) and not effective.get(
            "APPLY_CORE_SIMP_PHASE1", False
        ):
            raise DistributedExecutionError(
                "Distributed Core Simplification Phase 2 requires "
                "APPLY_CORE_SIMP_PHASE1=True in the same experiment. Worker notebooks "
                "build Phase 1 shards; the finalizer runs Phase 2 once from merged donors."
            )
        if effective.get("APPLY_TRANSFORMATION_DATASET_CONSTRUCTION", False):
            source = str(
                effective.get("TRANSFORMATION_DS_TARGET_SOURCE", "hard_questions")
            ).casefold()
            if source not in {"hard_questions", "hard", "targets"}:
                raise DistributedExecutionError(
                    "Distributed transformation dataset construction currently requires "
                    "TRANSFORMATION_DS_TARGET_SOURCE='hard_questions' so manifest indices "
                    "and question hashes remain aligned."
                )
            if effective.get("TRANSFORMATION_DS_MAX_TARGETS") is not None:
                raise DistributedExecutionError(
                    "Distributed transformation dataset construction requires "
                    "TRANSFORMATION_DS_MAX_TARGETS=None; cap the input question set before "
                    "starting the distributed run instead."
                )
            if effective.get("TRANSFORMATION_DS_MAX_MEMBERS") is not None:
                raise DistributedExecutionError(
                    "Distributed transformation dataset construction requires "
                    "TRANSFORMATION_DS_MAX_MEMBERS=None because a global accepted-member "
                    "cap cannot be enforced independently by workers."
                )
        if special_families and (
            hard_solutions is None
            or len(hard_solutions) == 0
            or any(solution is None or not str(solution).strip() for solution in hard_solutions)
        ):
            raise DistributedExecutionError(
                f"Distributed {special_families[0]} requires a non-empty ground-truth "
                "solution for every question."
            )
        if not effective.get("BATCH_PROCESSING_ENABLED", False):
            raise DistributedExecutionError(
                "Distributed execution requires BATCH_PROCESSING_ENABLED=True so each "
                "completed batch can be checkpointed synchronously."
            )
        for flag, workflow in _DISTRIBUTED_UNSUPPORTED_FLAGS.items():
            if effective.get(flag, False):
                raise DistributedExecutionError(
                    f"Distributed v1 intentionally does not support {workflow} ({flag}=True)."
                )
        indexed_artifacts = indexed_output_artifacts(effective)
        expected_result_filename_count = 1 + len(indexed_artifacts)
        result_filenames = {f"{experiment_name}_run_log.json"}
        result_filenames.update(artifact["filename"] for artifact in indexed_artifacts)
        if effective.get("APPLY_CORE_SIMP_PHASE2", False):
            phase2_filename = str(
                effective.get(
                    "CORE_SIMP_LAYER2_RESULTS_NAME",
                    "core_simp_layer2_results.json",
                )
                or ""
            ).strip()
            if (
                not phase2_filename
                or os.path.basename(phase2_filename) != phase2_filename
                or "/" in phase2_filename
                or "\\" in phase2_filename
                or not phase2_filename.casefold().endswith(".json")
            ):
                raise DistributedExecutionError(
                    "CORE_SIMP_LAYER2_RESULTS_NAME must be a direct .json filename "
                    "inside RESULTS_DIR for distributed finalization."
                )
            result_filenames.add(phase2_filename)
            expected_result_filename_count += 1
        collisions = result_filenames & published_result_filenames
        if collisions:
            raise DistributedExecutionError(
                "Distributed experiments must publish unique result filenames; "
                f"collisions: {sorted(collisions)}"
            )
        if len(result_filenames) != expected_result_filename_count:
            raise DistributedExecutionError(
                f"Distributed experiment {experiment_name!r} reuses its run-log filename "
                "for another output artifact."
            )
        published_result_filenames.update(result_filenames)
        if effective.get("APPLY_LAYER1_BASE_EXECUTION", False) and hard_solutions is None:
            raise DistributedExecutionError(
                "Distributed Layer 1 requires hard_solutions so ground-truth identity can be "
                "frozen in the manifest and checked during merge."
            )
        if effective.get("APPLY_LAYER1_BASE_EXECUTION", False) and any(
            solution is None or not str(solution).strip()
            for solution in ([] if hard_solutions is None else hard_solutions)
        ):
            raise DistributedExecutionError(
                "Distributed Layer 1 requires a non-empty ground-truth solution for every question."
            )


def _distributed_cache_for_experiment(
    current_config: Dict[str, Any],
    experiment_name: str,
) -> Optional[Dict[str, Any]]:
    if not current_config.get("APPLY_LAYER1_BASE_EXECUTION", False):
        return None
    top_k = current_config.get("TOP_N_CANDIDATES_RETRIEVAL", 5)
    n_candidates = current_config.get("LAYER1_ONE_SHOT_CANDIDATES_N")
    if n_candidates is None:
        n_candidates = current_config.get("LAYER1_N_CANDIDATES")
    if n_candidates is None:
        n_candidates = top_k
    filename = _get_cache_filename(
        top_k,
        n_candidates,
        experiment_name,
        current_config.get("LAYER1_ZERO_SHOT_CANDIDATES_N", 0),
    )
    cache_path = os.path.join(current_config["LAYER1_CACHE_DIR"], filename)
    if not os.path.exists(cache_path):
        return None
    cache = load_json(cache_path)
    if not isinstance(cache, dict):
        raise DistributedExecutionError(f"Malformed Layer-1 checkpoint: {cache_path}")
    return cache


def _distributed_progress_for_experiment(
    current_config: Dict[str, Any],
    experiment_name: str,
    run_logs: List[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    worker_id = int(current_config["DISTRIBUTED_WORKER_ID"])
    owned = assigned_indices(
        int(manifest["question_count"]),
        worker_id,
        int(manifest["worker_count"]),
    )
    validated_logs = validate_worker_run_logs(run_logs, manifest, worker_id)
    require_layer1 = bool(current_config.get("APPLY_LAYER1_BASE_EXECUTION", False))
    complete_cache_indices = set()
    if require_layer1:
        cache = _distributed_cache_for_experiment(current_config, experiment_name)
        if cache is not None:
            cache_queries = validate_worker_layer1_cache(cache, manifest, worker_id)
            complete_cache_indices = {
                int(index) for index, state in cache_queries.items()
                if layer1_state_complete(state)
            }
    successful = {
        index for index, log in validated_logs.items()
        if run_log_successful(log, require_layer1=require_layer1)
        and (not require_layer1 or index in complete_cache_indices)
    }
    return {
        "assigned": len(owned),
        "successful": len(successful),
        "remaining": len(owned) - len(successful),
        "checkpoint_entries": len(validated_logs),
        "successful_indices": sorted(successful),
    }


def _distributed_all_progress(
    global_config: Dict[str, Any],
    experiment_configs: List[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    progress: Dict[str, Any] = {}
    for overrides in experiment_configs:
        current_config = _merge_experiment_config(global_config, overrides)
        experiment_name = current_config.get("experiment_name", "unnamed_experiment")
        log_path = os.path.join(global_config["RESULTS_DIR"], f"{experiment_name}_run_log.json")
        run_logs = load_json(log_path) if os.path.exists(log_path) else []
        if not isinstance(run_logs, list):
            raise DistributedExecutionError(f"Malformed worker run log: {log_path}")
        progress[experiment_name] = _distributed_progress_for_experiment(
            current_config, experiment_name, run_logs, manifest
        )
    return progress


def _prepare_distributed_worker(
    global_config: Dict[str, Any],
    experiment_configs: List[Dict[str, Any]],
    hard_questions: List[str],
    hard_solutions: Optional[List[str]],
    exemplar_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate/create the immutable run and restore only this worker shard."""
    _validate_distributed_scope(global_config, experiment_configs, hard_solutions)
    paths = configure_worker_paths(global_config)
    allow_model_rotation = bool(
        global_config.get("DISTRIBUTED_ALLOW_MODEL_ROTATION", False)
    )
    if allow_model_rotation:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        global_config["_DISTRIBUTED_RUNTIME_CODE_FINGERPRINT"] = (
            resolve_code_fingerprint(project_root)
        )
    requested_manifest = _build_requested_distributed_manifest(
        global_config, experiment_configs, hard_questions, hard_solutions, exemplar_data
    )
    if _auto_pin_local_legacy_code_fingerprint(
        global_config, paths["manifest_path"], requested_manifest
    ):
        requested_manifest = _build_requested_distributed_manifest(
            global_config, experiment_configs, hard_questions, hard_solutions, exemplar_data
        )
    manifest_path = write_or_validate_manifest(
        paths["manifest_path"],
        requested_manifest,
        allow_model_rotation=allow_model_rotation,
    )

    # The results repo is created once by its owner.  Every notebook validates
    # or creates exactly the same write-once manifest before any provider call.
    ensure_distributed_manifest(global_config, manifest_path)
    if global_config.get("_DISTRIBUTED_MODEL_ROTATION_CODE_FINGERPRINT_AUTO_PINNED"):
        requested_manifest = _build_requested_distributed_manifest(
            global_config, experiment_configs, hard_questions, hard_solutions, exemplar_data
        )
    initialize_workspace(global_config)
    write_or_validate_manifest(
        paths["manifest_path"],
        requested_manifest,
        allow_model_rotation=allow_model_rotation,
    )
    manifest = load_json(paths["manifest_path"])
    if not isinstance(manifest, dict):
        raise DistributedExecutionError(
            f"Malformed authoritative distributed manifest: {paths['manifest_path']}"
        )
    model_changes = validate_manifest_compatibility(
        manifest,
        requested_manifest,
        allow_model_rotation=allow_model_rotation,
    )
    _announce_authorized_model_rotation(model_changes, context="worker resume")

    start_worker_session(global_config)
    global_config["_DISTRIBUTED_MANIFEST"] = manifest
    global_config["_DISTRIBUTED_ALLOWED_QUERY_INDICES"] = assigned_indices(
        len(hard_questions),
        int(global_config["DISTRIBUTED_WORKER_ID"]),
        int(global_config["DISTRIBUTED_WORKER_COUNT"]),
    )
    progress = _distributed_all_progress(global_config, experiment_configs, manifest)
    write_worker_status(
        global_config,
        "RUNNING",
        manifest=manifest,
        experiment_progress=progress,
        message="Worker session started or resumed; same worker id must not run concurrently.",
    )
    sync_workspace_to_hub(global_config)
    owned_indices = global_config["_DISTRIBUTED_ALLOWED_QUERY_INDICES"]
    owned_range = (
        f", indices {owned_indices[0]}..{owned_indices[-1]}"
        if owned_indices else ", empty chunk"
    )
    print(
        f"Distributed worker {global_config['DISTRIBUTED_WORKER_ID']}/"
        f"{global_config['DISTRIBUTED_WORKER_COUNT'] - 1} owns "
        f"{len(owned_indices)} global question(s){owned_range}."
    )
    return manifest


def _finalize_distributed_worker_status(
    global_config: Dict[str, Any],
    experiment_configs: List[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> str:
    progress = _distributed_all_progress(global_config, experiment_configs, manifest)
    complete = all(item["remaining"] == 0 for item in progress.values())
    if complete:
        state = "COMPLETE"
        message = "Every assigned question is durably complete for every experiment."
    elif stop_new_batches_due(global_config):
        state = "PAUSED_TIME_LIMIT"
        message = "Soft Kaggle session limit reached; restart this same worker id to resume."
    else:
        state = "INCOMPLETE"
        message = "Some assigned questions failed or remain incomplete; rerun this worker id."
    write_worker_status(
        global_config,
        state,
        manifest=manifest,
        experiment_progress=progress,
        message=message,
    )
    sync_workspace_to_hub(global_config)
    print(f"Distributed worker session finished with state={state}.")
    return state


def run_pipeline_for_single_query(
    hard_list_idx: int,
    target_query: str,
    config: Dict[str, Any],
    embedding_model: SentenceTransformer,
    exemplar_data: Dict[str, Any],
    api_managers: Dict[str, Any],
    run_mode: str = 'full',
    existing_log: Optional[Dict[str, Any]] = None,
    force_layer1_reexecution: bool = False,
    hard_solutions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Executes the RAG pipeline for a single query, supporting different execution modes
    and parallel mirror benchmarking.
    Args:
        ... (standard arguments) ...
        run_mode (str): Controls execution flow.
            - 'full': Runs the entire pipeline from start to finish.
            - 'intermediate': Runs only retrieve and adapt steps.
            - 'solve_only': Runs only the solve step, using pre-computed intermediate results.
        existing_log (Optional[Dict]): A pre-existing log from the intermediate phase,
                                       required for 'solve_only' mode.
        force_layer1_reexecution (bool): If True, forces Layer 1 to bypass cache and re-execute.
                                        Useful for surgical/manual retries where you want fresh Layer 1 data.
                                        Can also be set via CONFIG['FORCE_LAYER1_REEXECUTION'].
    """
    _reject_removed_feature_flags(config)
    # Isolate source-specific evaluation metadata per query.  This copy is
    # essential for batch workers: sharing a mutable current benchmark would
    # make mixed exact-answer and rationale evaluation race-dependent.
    config = config.copy()
    config["_TARGET_BENCHMARK_FOR_QUERY"] = benchmark_name_for_target_index(
        config, hard_list_idx
    )
    logger = logging.getLogger(__name__)
    
    # --- Log Initialization ---
    if run_mode == 'solve_only' and existing_log:
        run_log = existing_log
        # Reset status to ensure we correctly log the outcome of the solve step
        run_log['pipeline_status'] = "PENDING_SOLVE"
        # Ensure execution_trace exists if resuming old logs
        if "execution_trace" not in run_log:
            run_log["execution_trace"] = []
        print(f"\nResuming pipeline for Query #{hard_list_idx} (Solve Phase)...")
    else:
        # Standard initialization for 'full' or 'intermediate' modes
        print("\n" + "="*80)
        print(f"Processing Query #{hard_list_idx}: '{target_query[:100]}...'")
        print("="*80)
        logger.info(f"--- Starting pipeline for Query #{hard_list_idx}: '{target_query[:80]}...' ---")
        run_log = {
            "target_query_original_hard_list_idx": hard_list_idx,
            "target_query_text": target_query,
            "config_flags_used": {
                key: config.get(key) for key in [
                    # Core flags
                    "USE_RETRIEVAL", "APPLY_NORMALIZATION", "APPLY_TRANSFORMATION_1",
                    "APPLY_TRANSFORMATION_2", "APPLY_TRANSFORMATION_3",
                    "DEFER_SOLVE_STEP", "TOP_N_CANDIDATES_RETRIEVAL", "N_PASS_ATTEMPTS",
                    # UPDATE: Persist provider routing so a retry uses the same
                    # managers as the original experiment-level configuration.
                    "API_PROVIDER_ADAPTATION", "API_PROVIDER_SOLVER",
                    "API_PROVIDER_EVALUATOR", "API_PROVIDER_SIMPLIFICATION",
                    # Layer 1 Flags
                    "APPLY_LAYER1_BASE_EXECUTION", "LAYER1_ONLY_MODE", "LAYER1_CACHE_DIR",
                    "LAYER1_N_CANDIDATES", "LAYER1_ONE_SHOT_CANDIDATES_N", "LAYER1_DATASET_NAME",
                    # Reverse Validation Flags
                    "APPLY_REVERSE_VALIDATION", "REVERSE_VALIDATION_CANDIDATES_N",
                    "REVERSE_VALIDATION_RETRIEVAL_K", "REVERSE_VALIDATION_ATTEMPTS_N",
                    "REVERSE_VALIDATION_ENABLE_BASELINE_CHECK",
                    # Best-of-Transformation Flags
                    "APPLY_BEST_OF_TRANSFORMATION", "BEST_OF_TRANSFORMATION_N_SAMPLES",
                    "BEST_OF_TRANSFORMATION_TRANSFORMATION_TEMPLATE", "BEST_OF_TRANSFORMATION_ENABLE_MIRROR_EVAL",
                    # NEW: Unified Mirror Re-Ranking Flags
                    "APPLY_MIRROR_RERANKING", "MIRROR_RERANKING_APPLY_AFTER",
                    "MIRROR_RERANKING_N_OPTIMIZATION", "MIRROR_RERANKING_ENABLE_R0",
                    "MIRROR_RERANKING_ENABLE_FILTERING", "MIRROR_RERANKING_ENABLE_REDUNDANCY_FILTER",
                    "MIRROR_RERANKING_EVALUATE_BASE_FILTERING", "MIRROR_RERANKING_ACTIVE_LIMIT",
                    # Simplification Flags
                    "APPLY_SIMPLIFICATION", "SIMPLIFY_RETRIEVED_SAMPLES", "SIMPLIFY_MAIN_QUESTION",
                    # Mirroring Flags (Original)
                    "APPLY_MIRROR_AS_EVALUATOR", "MIRROR_EVALUATE_BASE_FILTERING",
                    "MIRROR_ENABLE_R0", "MIRROR_ACTIVE_CANDIDATE_LIMIT",
                    "MIRROR_ENABLE_REDUNDANCY_FILTER", "MIRROR_BENCHMARK_GROUPS",
                    "MIRROR_BENCHMARK_SAMPLES_N",
                    # Full Pipeline Retry Flag
                    "APPLY_FULL_PIPELINE_RETRY",
                    "EVAL_PARSE_BOXED_GROUND_TRUTH"
                ]
            },
            "pipeline_status": "PENDING",
            "execution_trace": [], 
            "steps": {},
            "steps_alternatives": {},
            "benchmarks": {} #Store Track A results
        }

    if distributed_enabled(config):
        record_avalai_model_provenance(run_log, config, run_mode)

    # API Manager Selection 
    def _get_api_manager(provider_name):
        manager = api_managers.get(provider_name)
        if not manager:
            raise ValueError(f"CRITICAL ERROR: API Manager '{provider_name}' is missing! Did the API cell crash or fail to initialize?")
        return manager

    provider_for_adapt = config.get('API_PROVIDER_ADAPTATION', 'gemini')
    manager_for_adapt = _get_api_manager(provider_for_adapt)
    
    provider_for_solve = config.get('API_PROVIDER_SOLVER', 'gemini')
    manager_for_solve = _get_api_manager(provider_for_solve)
    
    # Specific Manager for Evaluation
    provider_for_eval = config.get('API_PROVIDER_EVALUATOR', 'gemini')
    manager_for_eval = _get_api_manager(provider_for_eval)

    # Specific Manager for Simplification
    provider_for_simp = config.get('API_PROVIDER_SIMPLIFICATION', provider_for_adapt)
    manager_for_simp = _get_api_manager(provider_for_simp)

    # --- LAYER 1: BASE EXECUTION PHASE ---
    if config.get('APPLY_LAYER1_BASE_EXECUTION', False):
        print("\n[PRE-PROCESSING] LAYER 1: BASE EXECUTION PHASE ENABLED")
        print("  Mode: API-driven data generation with persistent combined caching")
        
        # Get experiment name for combined cache file
        experiment_name = config.get('experiment_name', 'default_experiment')
        
        # Determine if Layer 1 should be force re-executed (bypass cache)
        # Can be set via parameter OR via config flag
        do_force_layer1_reexecution = force_layer1_reexecution or config.get('FORCE_LAYER1_REEXECUTION', False)
        
        if do_force_layer1_reexecution:
            logger.info(f"🔄 Layer 1 force reexecution enabled for query #{hard_list_idx} (ignoring cache)")
        
        # Get ground truth for this query safely
        ground_truth = None

        # 1. Primary: Use explicitly provided test solutions
        if hard_solutions and hard_list_idx < len(hard_solutions):
            ground_truth = hard_solutions[hard_list_idx]
        # 2. Secondary: Check if injected into exemplar_data
        elif 'ground_truths' in exemplar_data and hard_list_idx < len(exemplar_data['ground_truths']):
            ground_truth = exemplar_data['ground_truths'][hard_list_idx]
        # 3. Fallback: Only allow fallback to training corpus IF it's a leave-one-out evaluation
        elif 'solutions' in exemplar_data:
            # SAFEGUARD: Are we actually testing the training corpus against itself?
            if config.get('hard_questions_length') == len(exemplar_data['solutions']):
                ground_truth = exemplar_data['solutions'][hard_list_idx]
            else:
                raise ValueError(
                    f"CRITICAL DATA MISALIGNMENT: Trying to evaluate test question #{hard_list_idx} "
                    f"using training corpus solution #{hard_list_idx}. Provide 'hard_solutions' to fix this."
                )
        
        if not ground_truth:
            logger.warning(f"Layer 1 enabled but no ground truth available for query #{hard_list_idx}. Skipping Layer 1.")
        else:
            # Execute Layer 1 with experiment_name for combined cache
            layer1_state = run_layer1_base_execution(
                target_query_index=hard_list_idx,
                target_query=target_query,
                ground_truth_answer=ground_truth,
                embedding_model=embedding_model,
                exemplar_questions=exemplar_data['questions'],
                exemplar_solutions=exemplar_data.get('solutions', []),
                embedded_exemplars=exemplar_data['embeddings'],
                exemplar_data=exemplar_data,
                api_manager_solve=manager_for_solve,  # CHANGED: Use Solver for generation
                api_manager_eval=manager_for_eval,    # ADDED: Use Evaluator for judgments
                config=config,
                experiment_name=experiment_name,  # Pass experiment name for combined cache
                force_reexecution=do_force_layer1_reexecution  # Force re-execution if retry
            )
            
            # Store Layer 1 state in run_log for reference
            run_log['layer1_base_execution_state'] = layer1_state
            
            # --- FIXED: Handle LAYER1_ONLY_MODE correctly regardless of success/failure ---
            if config.get('LAYER1_ONLY_MODE', False):
                overall_status = layer1_state.get('overall_status', 'FAILURE')
                
                print(f"\n[LAYER 1 COMPLETE] Layer 1 Only Mode is active. Status: {overall_status}. Pipeline execution halted.")
                
                # Assign the status from Layer 1 directly to the pipeline log
                run_log['pipeline_status'] = overall_status 
                run_log['execution_mode'] = 'layer1_only'
                
                logger.info(f"Layer 1 only mode: Query #{hard_list_idx} halted with status '{overall_status}'.")
                return run_log
            
            # Otherwise, continue with the main pipeline (if LAYER1_ONLY_MODE is False)
            logger.info(f"Layer 1 completed. Status: {layer1_state.get('overall_status')}. Proceeding with main pipeline.")

    # BEST-OF-TRANSFORMATION Pre-processing (Enhancement to best-of-N) 
    if config.get('APPLY_BEST_OF_TRANSFORMATION', False):
        print("\n[PRE-PROCESSING] BEST-OF-TRANSFORMATION ENABLED")
        
        # Step 1: Retrieve initial samples for transformation analysis
        if config.get('USE_RETRIEVAL', True):
            retrieval_result = retrieve(
                target_query, embedding_model,
                exemplar_data['questions'], exemplar_data['embeddings'],
                top_k=config.get('TOP_N_CANDIDATES_RETRIEVAL', 3),
                question_to_index_map=exemplar_data.get('question_to_index')
            )
            
            if 'trace' in retrieval_result:
                run_log['execution_trace'].extend(retrieval_result.pop('trace'))
            
            if retrieval_result['status'] == 'SUCCESS':
                retrieved_indices = retrieval_result['retrieved_indices']
                
                # Step 2: Apply best-of-transformation to each retrieved sample
                transform_result = select_best_transformations(
                    retrieved_indices=retrieved_indices,
                    target_query=target_query,
                    exemplar_data=exemplar_data,
                    embedding_model=embedding_model,
                    api_manager_adapt=manager_for_adapt,
                    api_manager_solve=manager_for_solve,
                    api_manager_eval=manager_for_eval,
                    config=config
                )
                
                if 'trace' in transform_result:
                    run_log['execution_trace'].extend(transform_result.pop('trace'))
                
                run_log['steps_alternatives']['best_of_transformation'] = transform_result
                
                if transform_result['status'] == 'SUCCESS':
                    # NEW ARCHITECTURE: Centralized Pool with Per-Sample Selection
                    evaluation_contexts = transform_result.get('evaluation_contexts', {})
                    best_candidates = transform_result.get('best_candidates', [])
                    telemetry = transform_result.get('telemetry', {})
                    
                    # Store for downstream reference
                    run_log['best_of_transformation_evaluation_contexts'] = evaluation_contexts
                    run_log['best_of_transformation_candidates'] = best_candidates
                    run_log['best_of_transformation_telemetry'] = telemetry
                    
                    # Extract candidate texts for optional use in further processing
                    candidate_texts = [c['candidate_text'] for c in best_candidates]
                    candidate_scores = [c['validation_score'] for c in best_candidates]
                    
                    run_log['best_transformations'] = candidate_texts  # For backwards compatibility
                    run_log['transformation_scores'] = candidate_scores  # For backwards compatibility
                    
                    # Log the centralized pool architecture 
                    logger.info(
                        f"Best-of-Transformation centralized pool architecture completed successfully. "
                        f"Selected {len(best_candidates)} best candidates from {len(evaluation_contexts)} retrieved samples"
                    )
                    
                    # Log pool statistics per sample
                    selections = telemetry.get('selections', {})
                    for retrieved_idx, selection_info in selections.items():
                        logger.info(
                            f"  Sample #{retrieved_idx}: "
                            f"Pool size={selection_info.get('pool_size', 'N/A')}, "
                            f"Selected index={selection_info.get('selected_index', 'N/A')}, "
                            f"Score={selection_info.get('selected_score', 'N/A'):.3f}, "
                            f"Source={selection_info.get('selected_source', 'N/A')}"
                        )
                    

                    # Integrated Mirror Re-Ranking Stage (After Transformation)
                    if config.get('APPLY_MIRROR_RERANKING', False):
                        print("\n[INTEGRATION] UNIFIED MIRROR RE-RANKING ENABLED")
                        
                        # Extract indices from best candidates for re-ranking
                        candidate_indices_for_reranking = [c['retrieved_idx'] for c in best_candidates]
                        
                        logger.info(
                            f"Applying mirror re-ranking to {len(candidate_indices_for_reranking)} "
                            f"best-of-transformation candidates"
                        )
                        
                        reranking_result = apply_mirror_reranking(
                            target_query=target_query,
                            indices_to_rerank=candidate_indices_for_reranking,
                            exemplar_data=exemplar_data,
                            api_manager_solve=manager_for_solve, # CHANGED
                            api_manager_eval=manager_for_eval,   # CHANGED
                            config=config
                        )
                        
                        if 'trace' in reranking_result:
                            run_log['execution_trace'].extend(reranking_result.pop('trace'))
                        
                        run_log['steps_alternatives']['mirror_reranking'] = reranking_result
                        
                        if reranking_result['status'] == 'SUCCESS':
                            reranked_indices = reranking_result['reranked_indices']
                            ranking_scores = reranking_result['ranking_scores']
                            
                            # Update the best_candidates ordering based on re-ranking
                            # Create a mapping from retrieved_idx to candidate for reordering
                            candidate_map = {c['retrieved_idx']: c for c in best_candidates}
                            
                            # Reorder best_candidates according to re-ranked indices
                            reranked_candidates = [
                                candidate_map[idx] for idx in reranked_indices 
                                if idx in candidate_map
                            ]
                            
                            run_log['best_of_transformation_candidates'] = reranked_candidates
                            run_log['mirror_reranking_scores'] = ranking_scores
                            
                            logger.info(
                                f"Mirror re-ranking completed successfully. "
                                f"New order: {reranked_indices}"
                            )
                            
                            # Update best_transformations for downstream compatibility
                            run_log['best_transformations'] = [c['candidate_text'] for c in reranked_candidates]
                        else:
                            logger.warning(
                                f"Mirror re-ranking failed: {reranking_result.get('error')}, "
                                f"proceeding with original candidate order"
                            )
                else:
                    logger.warning("Best-of-Transformation pre-processing failed, proceeding with standard best-of-N")
                    run_log['best_of_transformation_candidates'] = None
                    run_log['best_transformations'] = None
            else:
                logger.warning("Retrieval failed during best-of-transformation pre-processing")
                run_log['best_transformations'] = None
        else:
            logger.warning("USE_RETRIEVAL is False, cannot apply best-of-transformation")
            run_log['best_transformations'] = None

    # ALTERNATIVE: Direct Mirror Re-Ranking After Retrieval (without Best-of-Transformation) 
    if config.get('APPLY_MIRROR_RERANKING', False) and not config.get('APPLY_BEST_OF_TRANSFORMATION', False):
        if config.get('USE_RETRIEVAL', True):
            print("\n[INTEGRATION] UNIFIED MIRROR RE-RANKING ENABLED (Direct After Retrieval)")
            
            retrieval_result = retrieve(
                target_query, embedding_model,
                exemplar_data['questions'], exemplar_data['embeddings'],
                top_k=config.get('TOP_N_CANDIDATES_RETRIEVAL', 3),
                question_to_index_map=exemplar_data.get('question_to_index')
            )
            
            if 'trace' in retrieval_result:
                run_log['execution_trace'].extend(retrieval_result.pop('trace'))
            
            if retrieval_result['status'] == 'SUCCESS':
                retrieved_indices = retrieval_result['retrieved_indices']
                
                logger.info(
                    f"Applying mirror re-ranking to {len(retrieved_indices)} retrieved samples"
                )
                
                reranking_result = apply_mirror_reranking(
                    target_query=target_query,
                    indices_to_rerank=retrieved_indices,
                    exemplar_data=exemplar_data,
                    api_manager_solve=manager_for_solve,
                    api_manager_eval=manager_for_eval, 
                    config=config
                )
                
                if 'trace' in reranking_result:
                    run_log['execution_trace'].extend(reranking_result.pop('trace'))
                
                run_log['steps_alternatives']['mirror_reranking'] = reranking_result
                
                if reranking_result['status'] == 'SUCCESS':
                    # Store re-ranked indices for downstream use
                    reranked_indices = reranking_result['reranked_indices']
                    ranking_scores = reranking_result['ranking_scores']
                    
                    run_log['retrieved_indices'] = reranked_indices
                    run_log['mirror_reranking_scores'] = ranking_scores
                    
                    logger.info(
                        f"Mirror re-ranking completed. Re-ranked indices: {reranked_indices}"
                    )
                else:
                    logger.warning(
                        f"Mirror re-ranking failed: {reranking_result.get('error')}, "
                        f"proceeding with original retrieved order"
                    )
            else:
                logger.warning("Retrieval failed during mirror re-ranking")
        else:
            logger.warning("USE_RETRIEVAL is False, cannot apply mirror re-ranking directly")

    # MODE: Reverse Validation (Analogical Consistency) 
    if config.get('APPLY_REVERSE_VALIDATION', False):
        print("\n[MODE] ANALOGICAL CONSISTENCY (REVERSE VALIDATION) ACTIVATED")
        # This mode replaces the standard solve flow entirely.
        consistency_result = solve_with_analogical_consistency(
            target_query=target_query,
            exemplar_data=exemplar_data,
            embedding_model=embedding_model,
            api_manager_solve=manager_for_solve,
            api_manager_eval=manager_for_eval, 
            config=config
        )
        
        # Aggregation: Extract trace
        if 'trace' in consistency_result:
            run_log['execution_trace'].extend(consistency_result.pop('trace'))
        
        run_log['steps']['solving'] = consistency_result
        
        if consistency_result['status'] == 'SUCCESS':
            run_log['pipeline_status'] = "SUCCESS"
        else:
            run_log['pipeline_status'] = f"FAILURE: {consistency_result.get('error')}"
            
        return run_log

    # Standard Pipeline Execution (with Parallel Benchmarking & Mirroring)
    
    # SETUP FULL PIPELINE RETRY LOGIC 
    full_retry_mode = config.get('APPLY_FULL_PIPELINE_RETRY', False)
    
    if full_retry_mode and run_mode == 'full':
        if config.get("DEFER_SOLVE_STEP", False):
            logger.warning("Disabling DEFER_SOLVE_STEP because FULL_PIPELINE_RETRY is On.")
            config['DEFER_SOLVE_STEP'] = False
            
        n_pipeline_iterations = config.get("N_PASS_ATTEMPTS", 1)
        # In full retry mode, each pass is independent, so solve n=1 each time
        n_solver_attempts_per_pass = 1 
        logger.info(f"Full Pipeline Retry Enabled: Running complete pipeline {n_pipeline_iterations} times.")
    else:
        n_pipeline_iterations = 1
        n_solver_attempts_per_pass = config.get("N_PASS_ATTEMPTS", 1) 

    aggregated_solution_attempts = []
    iteration_details = []
    final_pipeline_status = "PENDING"

    # PIPELINE ITERATION LOOP
    for iteration_idx in range(n_pipeline_iterations):
        if full_retry_mode:
            print(f"\n[FULL PIPELINE ITERATION] {iteration_idx + 1}/{n_pipeline_iterations}")
            
        pipeline_halted = False
        
        # Local container for this iteration's logs
        # Structure: { "primary": {step: val}, "group_0": {step: val}, ... }
        iter_log_strategies = {}
        master_sorted_indices = []
        validation_strategies = {}


        # PHASE 1: ACQUIRE & OPTIMIZE (Retrieve + Mirror)
        if run_mode in ['full', 'intermediate']:
            # -- Step 1: Retrieve --
            retrieved_indices_initial = []
            
            if config.get('USE_RETRIEVAL', True):
                print("\n[STEP 1] RETRIEVE")
                # Fast retrieval with O(1) self-match hash map
                retrieval_result = retrieve(
                    target_query=target_query, embedding_model=embedding_model,
                    exemplar_questions=exemplar_data['questions'], embedded_exemplars=exemplar_data['embeddings'],
                    top_k=config['TOP_N_CANDIDATES_RETRIEVAL'],
                    question_to_index_map=exemplar_data.get('question_to_index')
                )
                
                # Aggregation: Extract trace
                if 'trace' in retrieval_result:
                    run_log['execution_trace'].extend(retrieval_result.pop('trace'))

                if retrieval_result['status'] == 'FAILURE':
                    final_pipeline_status = "FAILURE: Retrieval failed."
                    logger.error(final_pipeline_status)
                    print("  -> Retrieval FAILED. Halting pipeline for this query.")
                    pipeline_halted = True
                else:
                    retrieved_indices_initial = retrieval_result['retrieved_indices']
                    print(f"  -> Retrieved indices: {retrieved_indices_initial}")

            else:
                print("\n[STEP 1] RETRIEVE SKIPPED (USE_RETRIEVAL is False).")

            # Step 2: Optimization (Mirroring)
            # This creates List 2 (Master), List 3 (Base), List 4 (Redundancy)
            
            master_sorted_indices = []
            validation_strategies = {} # Will hold {'base_filtering': [...], 'redundancy_filtering': [...]}

            if not pipeline_halted:
                if config.get("APPLY_MIRROR_AS_EVALUATOR", False) and retrieved_indices_initial:
                    print("\n[STEP 2] MIRROR OPTIMIZATION ACTIVE")
                    
                    mirror_result = optimize_demonstrations_via_mirroring(
                        target_query=target_query,
                        retrieved_indices=retrieved_indices_initial,
                        exemplar_data=exemplar_data,
                        api_manager_solve=manager_for_solve, # CHANGED
                        api_manager_eval=manager_for_eval,   # ADDED
                        config=config
                    )

                    # Capture Trace
                    if 'trace' in mirror_result:
                        run_log['execution_trace'].extend(mirror_result.pop('trace'))
                    
                    run_log['steps']['mirror_optimization'] = mirror_result

                    if mirror_result['status'] == 'SUCCESS':
                        # EXTRACT THE MASTER LIST (List 2)
                        master_sorted_indices = mirror_result.get('master_sorted_indices', [])
                        # EXTRACT VALIDATION LISTS (List 3 & 4)
                        validation_strategies = mirror_result.get('strategies', {})
                        print(f"  -> Optimization Complete. Master List Count: {len(master_sorted_indices)}")
                    else:
                        logger.warning("Mirroring failed. Fallback to standard retrieval.")
                        master_sorted_indices = retrieved_indices_initial
                else:
                    # No Mirroring: Master list is just the retrieval result
                    master_sorted_indices = retrieved_indices_initial


        # PHASE 2: FORK (Track A & Track B)
        execution_tasks = []

        if not pipeline_halted and run_mode != 'solve_only':
            
            # TRACK A: BENCHMARKING (Ranking Robustness) 

            
            if config.get("APPLY_MIRROR_AS_EVALUATOR", False):
                benchmark_groups = config.get("MIRROR_BENCHMARK_GROUPS", [])
                benchmark_n = config.get("MIRROR_BENCHMARK_SAMPLES_N", 10)
                
                print(f"\n[TRACK A] BENCHMARKING (N={benchmark_n})")
                
                for group_tuple in benchmark_groups:
                    strat_name = "group_" + "_".join(map(str, group_tuple))
                    
                    # Strictly slice Master List
                    strat_indices = []
                    valid_group = True
                    for ptr in group_tuple:
                        if ptr < len(master_sorted_indices):
                            strat_indices.append(master_sorted_indices[ptr])
                        else:
                            valid_group = False
                            break
                    
                    if valid_group:
                        # Add to execution queue
                        execution_tasks.append({
                            "name": strat_name,
                            "indices": strat_indices,
                            "n_samples": benchmark_n,
                            "type": "benchmark"
                        })
                    else:
                        logger.warning(f"Skipping benchmark group {group_tuple}: Not enough items in Master List.")

            # TRACK B: VALIDATION (Feature Check)
            # Operates on 'validation_strategies' (List 3 & List 4)
            # Logic: Use whole list.
            
            if validation_strategies:
                print("\n[TRACK B] FEATURE VALIDATION (N=1)")
                
                if config.get("MIRROR_EVALUATE_BASE_FILTERING", False):
                    indices = validation_strategies.get("base_filtering", [])
                    if indices:
                        execution_tasks.append({
                            "name": "base_filtering",
                            "indices": indices,
                            "n_samples": 1,
                            "type": "validation"
                        })
                
                if config.get("MIRROR_ENABLE_REDUNDANCY_FILTER", True):
                    indices = validation_strategies.get("redundancy_filtering", [])
                    if indices:
                         # This is often the "Primary" output if redundancy is on
                        execution_tasks.append({
                            "name": "redundancy_filtering",
                            "indices": indices,
                            "n_samples": 1,
                            "type": "validation"
                        })
            
            # FALLBACK / DEFAULT 
            # If no mirroring and no benchmarking, we just run the master list as "primary"
            if not execution_tasks:
                 execution_tasks.append({
                    "name": "primary",
                    "indices": master_sorted_indices,
                    "n_samples": n_solver_attempts_per_pass,
                    "type": "primary"
                })



        # PHASE 3: EXECUTE TASKS (Adapt -> Solve)
        # If running in solve_only mode, we need to load the intermediate state differently
        if run_mode == 'solve_only':
             # In solve_only mode, we assume the log contains the necessary keys.
             # We reconstruct the task list based on available data in the existing log.
             if 'steps' in run_log and 'final_exemplars' in run_log['steps']:
                 execution_tasks.append({'name': 'primary', 'type': 'primary', 'indices': [], 'n_samples': n_solver_attempts_per_pass})
             
             if 'benchmarks' in run_log:
                 for name, benchmark_data in run_log['benchmarks'].items():
                     if not isinstance(benchmark_data, dict) or 'final_exemplars' not in benchmark_data:
                         continue
                     # Attempt to recover N from the stored log or config
                     execution_tasks.append({'name': name, 'type': 'benchmark', 'indices': [], 'n_samples': config.get("MIRROR_BENCHMARK_SAMPLES_N", 10)})
            
             if 'steps_alternatives' in run_log:
                 for name, alternative_data in run_log['steps_alternatives'].items():
                     if not isinstance(alternative_data, dict) or 'final_exemplars' not in alternative_data:
                         continue
                     execution_tasks.append({'name': name, 'type': 'validation', 'indices': [], 'n_samples': 1})

        for task in execution_tasks:
            strat_name = task['name']
            indices = task['indices']
            n_samples_for_task = task['n_samples']
            task_type = task['type']
            
            print(f"\n--- Executing Strategy: {strat_name.upper()} (Type: {task_type}, N={n_samples_for_task}) ---")
            
            local_step_log = {} 
            current_exemplars_for_step = [] 
            
            # ADAPTATION PHASE 
            if run_mode in ['full', 'intermediate']:
                if config.get('USE_RETRIEVAL'):
                    if indices == [-1]: # Zero-shot
                        current_exemplars_for_step = []
                        local_step_log['status'] = "ZERO_SHOT"
                    elif config.get('APPLY_SIMPLIFICATION', False) and config.get('SIMPLIFY_RETRIEVED_SAMPLES', False):
                        # Simplification Workflow
                        simp_result = simplify_retrieved_samples(
                            retrieved_indices=indices,
                            exemplar_questions=exemplar_data['questions'],
                            exemplar_solutions=exemplar_data['solutions'],
                            api_manager=manager_for_simp,
                            config=config
                        )
                        if 'trace' in simp_result:
                            run_log['execution_trace'].extend(simp_result.pop('trace'))
                        local_step_log['simplification_of_samples'] = simp_result
                        current_exemplars_for_step.extend(simp_result.get('simplified_exemplars', []))
                    else:
                        # Standard Adapt
                        adapt_result = adapt(
                            target_query=target_query, 
                            retrieved_indices=indices,
                            exemplar_questions=exemplar_data['questions'], 
                            exemplar_solutions=exemplar_data['solutions'],
                            api_manager=manager_for_adapt, 
                            config=config
                        )
                        if 'trace' in adapt_result:
                            run_log['execution_trace'].extend(adapt_result.pop('trace'))
                        local_step_log['adaptation'] = adapt_result
                        current_exemplars_for_step.extend(adapt_result.get('adapted_texts', []))
                        
                        # --- REVERSE TRANSFORMATION (Alternative/Complementary Branch) ---
                        if config.get('APPLY_REVERSE_TRANSFORMATION', False):
                            print(f"    -> Applying Reverse Transformation to indices {indices}...")
                            
                            rt_result = reverse_transform_and_solve(
                                target_query=target_query,
                                retrieved_indices=indices,
                                exemplar_questions=exemplar_data['questions'],
                                exemplar_solutions=exemplar_data['solutions'],
                                api_manager=manager_for_solve,  # Using solver model for final answer generation
                                config=config
                            )
                            
                            if 'trace' in rt_result:
                                run_log['execution_trace'].extend(rt_result.pop('trace'))
                            local_step_log['reverse_transformation'] = rt_result
                            
                            # If reverse transformation is successful, use its solutions as exemplars for final solve
                            if rt_result['status'] == 'SUCCESS':
                                # RT already includes solving, so we extract the solution attempts
                                # Format them as exemplars for potential further use
                                current_exemplars_for_step.extend([
                                    f"Question: Original Query\nRationale and Answer: {sol}"
                                    if isinstance(sol, str) else f"Question: Original Query\nRationale and Answer: {sol.get('error_info', {})}"
                                    for sol in rt_result.get('solution_attempts', [])
                                ])
                                
                # Store Intermediate
                iter_log_strategies[strat_name] = local_step_log
                iter_log_strategies[strat_name]['final_exemplars'] = current_exemplars_for_step

                if run_mode == 'intermediate':
                    if task_type == 'benchmark':
                        run_log['benchmarks'][strat_name] = local_step_log.copy()
                    elif task_type == 'validation':
                        run_log['steps_alternatives'][strat_name] = local_step_log.copy()

            #  SOLVE PHASE 
            if run_mode in ['full', 'solve_only']:
                # Recover exemplars if in solve_only mode
                if run_mode == 'solve_only':
                    if strat_name in iter_log_strategies:
                        current_exemplars_for_step = iter_log_strategies[strat_name]['final_exemplars']
                    elif strat_name == 'primary' and 'steps' in run_log:
                        current_exemplars_for_step = run_log['steps'].get('final_exemplars', [])
                    elif task_type == 'benchmark' and 'benchmarks' in run_log:
                         # Attempt to recover from benchmarks log if structured correctly
                         current_exemplars_for_step = run_log['benchmarks'].get(strat_name, {}).get('final_exemplars', [])
                    elif task_type == 'validation' and 'steps_alternatives' in run_log:
                         current_exemplars_for_step = run_log['steps_alternatives'].get(strat_name, {}).get('final_exemplars', [])

                current_solver_config = config.copy()
                current_solver_config['N_PASS_ATTEMPTS'] = n_samples_for_task
                
                print(f"  -> Solving {strat_name} (N={n_samples_for_task})...")
                
                # Check if reverse transformation already provided solutions
                if config.get('APPLY_REVERSE_TRANSFORMATION', False) and 'reverse_transformation' in local_step_log:
                    rt_log = local_step_log.get('reverse_transformation', {})
                    if rt_log.get('status') == 'SUCCESS':
                        # Reverse transformation already solved the problem, use its result directly
                        print("  -> Using solution from Reverse Transformation step (skipping regular solve)...")
                        solve_result = {
                            'status': 'SUCCESS',
                            'solution_attempts': rt_log.get('solution_attempts', []),
                            'trace': []
                        }
                    else:
                        # Reverse transformation failed, proceed with normal solve
                        # BRANCH: Main Question Simplification
                        if config.get('APPLY_SIMPLIFICATION', False) and config.get('SIMPLIFY_MAIN_QUESTION', False):
                             solve_result = solve_via_main_simplification(
                                target_query=target_query,
                                final_exemplars=current_exemplars_for_step,
                                api_manager=manager_for_simp,
                                config=config
                            )
                        else:
                            solve_result = solve(
                                target_query=target_query, 
                                final_exemplars=current_exemplars_for_step,
                                api_manager=manager_for_solve, 
                                config=current_solver_config
                            )
                else:
                    # Normal solve path (no reverse transformation)
                    # BRANCH: Main Question Simplification
                    if config.get('APPLY_SIMPLIFICATION', False) and config.get('SIMPLIFY_MAIN_QUESTION', False):
                         solve_result = solve_via_main_simplification(
                            target_query=target_query,
                            final_exemplars=current_exemplars_for_step,
                            api_manager=manager_for_simp,
                            config=config
                        )
                    else:
                        solve_result = solve(
                            target_query=target_query, 
                            final_exemplars=current_exemplars_for_step,
                            api_manager=manager_for_solve, 
                            config=current_solver_config
                        )
                
                if 'trace' in solve_result:
                    run_log['execution_trace'].extend(solve_result.pop('trace'))

                # STORAGE LOGIC 
                if task_type == 'benchmark':
                    if 'benchmarks' not in run_log: run_log['benchmarks'] = {}
                    # Preserve deferred intermediate data while adding solve results.
                    if run_mode == 'solve_only':
                        combined_data = run_log['benchmarks'].get(strat_name, {}).copy()
                    else:
                        combined_data = local_step_log.copy()
                    combined_data.update(solve_result)
                    run_log['benchmarks'][strat_name] = combined_data
                    
                elif task_type == 'validation':
                    if 'steps_alternatives' not in run_log: run_log['steps_alternatives'] = {}
                    if run_mode == 'solve_only':
                        combined_data = run_log['steps_alternatives'].get(strat_name, {}).copy()
                    else:
                        combined_data = {}
                    combined_data['solving'] = solve_result
                    # Also copy intermediate steps
                    if strat_name in iter_log_strategies:
                        combined_data.update(iter_log_strategies[strat_name])
                    run_log['steps_alternatives'][strat_name] = combined_data
                        
                elif task_type == 'primary':
                    # Store in main 'steps'
                    run_log['steps']['solving'] = solve_result
                    if solve_result['status'] == 'SUCCESS':
                        final_pipeline_status = "SUCCESS"
                    else:
                        final_pipeline_status = "FAILURE"
                    
                    if 'solution_attempts' in solve_result:
                        aggregated_solution_attempts.extend(solve_result['solution_attempts'])

        # Update logs for iteration
        if full_retry_mode:
            iteration_details.append({
                "iteration": iteration_idx,
                "steps": iter_log_strategies,
                "primary_solve": run_log['steps'].get('solving', {})
            })
            # Update main steps from primary
            if "primary" in iter_log_strategies:
                 run_log['steps'].update(iter_log_strategies["primary"])
        else:
            # Update main steps from primary
            if "primary" in iter_log_strategies:
                run_log['steps'].update(iter_log_strategies["primary"])
            elif "redundancy_filtering" in iter_log_strategies:
                run_log['steps'].update(iter_log_strategies["redundancy_filtering"])

    # Final Aggregation
    if run_mode == 'intermediate' and final_pipeline_status == 'PENDING':
        final_pipeline_status = 'INTERMEDIATE_COMPLETE'

    run_log['pipeline_status'] = final_pipeline_status
    if full_retry_mode:
        run_log['full_pipeline_iterations_data'] = iteration_details
    
    # Consolidate all solution attempts (strings) for the evaluator
    all_solution_texts = [attempt for attempt in aggregated_solution_attempts if isinstance(attempt, str)]
    run_log['llm_final_solution_attempts_texts'] = all_solution_texts
    
    if 'solving' in run_log['steps'] and aggregated_solution_attempts:
        run_log['steps']['solving']['solution_attempts'] = aggregated_solution_attempts

    logger.info(f"--- Pipeline finished for Query #{hard_list_idx} with status: {run_log['pipeline_status']} ---")
    return run_log


def _log_query_index(run_log: Dict[str, Any]) -> Optional[int]:
    value = run_log.get("target_query_original_hard_list_idx")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _run_pipeline_items_in_batches(
    items: List[QuestionWorkItem],
    existing_logs: List[Dict[str, Any]],
    log_file_path: str,
    experiment_name: str,
    phase_name: str,
    run_mode: str,
    current_config: Dict[str, Any],
    embedding_model: SentenceTransformer,
    exemplar_data: Dict[str, Any],
    api_managers: Dict[str, Any],
    hard_solutions: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Run a phase in fixed batches and commit logs/cache only at batch boundaries."""
    logs_by_index = {
        query_index: log for log in existing_logs
        if (query_index := _log_query_index(log)) is not None
    }
    unindexed_logs = [log for log in existing_logs if _log_query_index(log) is None]

    def worker(item: QuestionWorkItem) -> Dict[str, Any]:
        return run_pipeline_for_single_query(
            hard_list_idx=item.index,
            target_query=item.question,
            config=current_config,
            embedding_model=embedding_model,
            exemplar_data=exemplar_data,
            api_managers=api_managers,
            run_mode=run_mode,
            existing_log=item.existing_log,
            hard_solutions=hard_solutions,
        )

    def commit(results: List[QuestionResult], batch_id: str, batch_number: int) -> None:
        
        # --- NEW: Print a clean summary to the console! ---
        print(f"\n{'='*70}")
        print(f"[COMPLETED] BATCH {batch_number} [Phase: {phase_name.upper()}]")
        for result in results:
            q_idx = result.item.index
            
            # Safely get status without crashing
            if isinstance(result.value, dict):
                status_from_value = result.value.get("pipeline_status", "UNKNOWN")
            else:
                status_from_value = "UNKNOWN"
                
            status = result.terminal_status or status_from_value
            elapsed = round(result.elapsed_seconds, 2)
            
            # Add some nice color/icons based on status
            is_error = "FAIL" in status.upper() or "ERROR" in status.upper()
            icon = "[OK]" if "SUCCESS" in status else "[FAIL]" if is_error else "[PARTIAL]"
            
            print(f"  {icon} [Q#{q_idx}] Status: {status} ({elapsed}s)")
            
            # If the thread crashed, print the actual error message!
            if is_error and isinstance(result.value, dict) and "batch_error" in result.value:
                error_msg = result.value["batch_error"].get("message", "Unknown error")
                print(f"      ↳ ERROR DETAILS: {error_msg}")
                
        print(f"{'='*70}\n")

        # (The rest of your original commit logic stays exactly the same)
        for result in results:
            run_log = result.value
            run_log["batch_metadata"] = {
                "batch_id": batch_id,
                "phase": phase_name,
                "terminal_status": result.terminal_status,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
            }
            logs_by_index[result.item.index] = run_log

        committed_logs = [logs_by_index[index] for index in sorted(logs_by_index)] + unindexed_logs
        if not save_json_atomic(committed_logs, log_file_path):
            raise RuntimeError(f"Could not commit batch {batch_id} run log: {log_file_path}")

        layer1_states = {
            result.item.index: result.value["layer1_base_execution_state"]
            for result in results
            if isinstance(result.value.get("layer1_base_execution_state"), dict)
        }
        
        if layer1_states:
            top_k = current_config.get("TOP_N_CANDIDATES_RETRIEVAL", 5)
            n_candidates = current_config.get("LAYER1_ONE_SHOT_CANDIDATES_N") or current_config.get("LAYER1_N_CANDIDATES") or top_k
            cache_dir = current_config.get("LAYER1_CACHE_DIR", current_config.get("RESULTS_DIR"))
            cache_path = _get_cache_path(
                cache_dir,
                _get_cache_filename(
                    top_k,
                    n_candidates,
                    experiment_name,
                    current_config.get("LAYER1_ZERO_SHOT_CANDIDATES_N", 0),
                ),
            )
            if not _save_cached_states_batch(cache_path, layer1_states, top_k, n_candidates, experiment_name, batch_id):
                raise RuntimeError(f"Could not commit Layer-1 cache for batch {batch_id}")

        try:
            periodic_batch_sync_check(batch_number - 1, current_config)
        except Exception as sync_error:
            manifest = current_config.get("_DISTRIBUTED_MANIFEST")
            if distributed_enabled(current_config) and isinstance(manifest, dict):
                try:
                    progress = _distributed_progress_for_experiment(
                        current_config, experiment_name, committed_logs, manifest
                    )
                    write_worker_status(
                        current_config,
                        "SYNC_ERROR",
                        manifest=manifest,
                        experiment_progress={experiment_name: progress},
                        message=(
                            "Local batch commit succeeded, but its remote checkpoint failed; "
                            "no later batch was started."
                        ),
                    )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Could not write local distributed SYNC_ERROR status."
                    )
            raise sync_error

    BatchCoordinator(current_config, experiment_name, phase_name).run(items, worker, commit)
    return [logs_by_index[index] for index in sorted(logs_by_index)] + unindexed_logs


def run_experiments(
    experiment_configs: List[Dict[str, Any]],
    global_config: Dict[str, Any],
    hard_questions: List[str],
    embedding_model: SentenceTransformer,
    exemplar_data: Dict[str, Any],
    api_managers: Dict[str, Any],
    hard_solutions: Optional[List[str]] = None
) -> Dict[str, List[Dict]]:
    """
    Orchestrates running multiple experiments with different configurations.
    Supports both standard and cross-experiment deferred execution modes.
    """
    logger = logging.getLogger(__name__)
    all_results = {}
    distributed_manifest: Optional[Dict[str, Any]] = None
    distributed_experiment_configs = list(experiment_configs)

    for exp_overrides in experiment_configs:
        effective_config = _merge_experiment_config(global_config, exp_overrides)
        _reject_removed_feature_flags(effective_config)

    if distributed_enabled(global_config):
        if global_config.get("DISTRIBUTED_FINALIZER_MODE", False):
            raise DistributedExecutionError(
                "DISTRIBUTED_FINALIZER_MODE=True cannot call run_experiments; call "
                "finalize_distributed_experiments instead."
            )
        distributed_manifest = _prepare_distributed_worker(
            global_config,
            experiment_configs,
            hard_questions,
            hard_solutions,
            exemplar_data,
        )
    
    # ADDED THIS LINE TO MAKE THE SAFEGUARD WORK 
    global_config['hard_questions_length'] = len(hard_questions)

    # SPECIAL CASE: Transformation Dataset Construction
    transformation_ds_configs = [
        exp for exp in experiment_configs
        if _merge_experiment_config(global_config, exp).get(
            "APPLY_TRANSFORMATION_DATASET_CONSTRUCTION"
        )
    ]
    experiment_configs = [
        exp for exp in experiment_configs
        if not _merge_experiment_config(global_config, exp).get(
            "APPLY_TRANSFORMATION_DATASET_CONSTRUCTION"
        )
    ]

    if transformation_ds_configs:
        logger.info(
            "Found %s Transformation Dataset Construction config(s); running them now.",
            len(transformation_ds_configs),
        )
        from src.transformation_dataset_builder import build_transformation_dataset

        for exp_overrides in transformation_ds_configs:
            current_config = _merge_experiment_config(global_config, exp_overrides)
            exp_name = current_config.get(
                "experiment_name", "transformation_dataset_construction"
            )
            logger.info("--- Transformation Dataset Construction '%s' starting ---", exp_name)

            ds_results = build_transformation_dataset(
                hard_questions=hard_questions,
                hard_solutions=hard_solutions,
                exemplar_data=exemplar_data,
                embedding_model=embedding_model,
                api_managers=api_managers,
                config=current_config,
            )
            all_results[exp_name] = ds_results
            logger.info("--- Transformation Dataset Construction '%s' finished ---", exp_name)

    if not experiment_configs:
        if distributed_manifest is not None:
            _finalize_distributed_worker_status(
                global_config, distributed_experiment_configs, distributed_manifest
            )
        return all_results
        
    # SPECIAL CASE: Core-Preserving Simplification Phase 1 & Phase 2
    phase1_configs = [
        exp for exp in experiment_configs
        if _merge_experiment_config(global_config, exp).get("APPLY_CORE_SIMP_PHASE1")
    ]
    phase2_configs = [
        exp for exp in experiment_configs
        if _merge_experiment_config(global_config, exp).get("APPLY_CORE_SIMP_PHASE2")
    ]
    
    # Update normal_configs to exclude Phase 1 and Phase 2 configs from the standard pipeline
    experiment_configs = [
        exp for exp in experiment_configs
        if not (
            _merge_experiment_config(global_config, exp).get("APPLY_CORE_SIMP_PHASE1")
            or _merge_experiment_config(global_config, exp).get("APPLY_CORE_SIMP_PHASE2")
        )
    ]


    # EXECUTE PHASE 1: Build the Donor Dataset
    if phase1_configs:
        logger.info(f"Found {len(phase1_configs)} Core Simplification Phase 1 config(s); running them now.")
        from src.core_simplification import execute_core_simplification_phase1
        
        for exp_overrides in phase1_configs:
            current_config = _merge_experiment_config(global_config, exp_overrides)
            exp_name = current_config.get("experiment_name", "core_simp_phase1")
            
            logger.info(f"--- Core Simplification Phase 1 '{exp_name}' starting ---")
            
            solver_mgr = api_managers.get(current_config.get("API_PROVIDER_SOLVER", "gemini"))
            eval_mgr = api_managers.get(current_config.get("API_PROVIDER_EVALUATOR", "gemini"))
            
            full_logs = execute_core_simplification_phase1(
                hard_questions=hard_questions,
                hard_solutions=hard_solutions,
                exemplar_data=exemplar_data,
                api_manager_solve=solver_mgr,
                api_manager_eval=eval_mgr,
                config=current_config,
            )

            dataset_filename = current_config.get("CORE_SIMP_DATASET_NAME", "core_simp_dataset.json")
            dataset_path = os.path.join(current_config['RESULTS_DIR'], dataset_filename)
            successful_samples = load_json(dataset_path) or []

            all_results[exp_name] = full_logs
            logger.info(f"--- Phase 1 '{exp_name}' finished. Saved {len(successful_samples)} verified samples. ---")

    # EXECUTE PHASE 2: Analogical Evaluation (The 3 Branches)
    if phase2_configs and distributed_manifest is None:
        logger.info(f"Found {len(phase2_configs)} Core Simplification Phase 2 config(s); running them now.")
        from src.core_simplification_layer2 import execute_core_simplification_phase2
        
        for exp_overrides in phase2_configs:
            current_config = _merge_experiment_config(global_config, exp_overrides)
            exp_name = current_config.get("experiment_name", "core_simp_phase2")
            
            logger.info(f"--- Core Simplification Phase 2 '{exp_name}' starting ---")
            
            phase2_results = execute_core_simplification_phase2(
                hard_questions=hard_questions,
                hard_solutions=hard_solutions,
                exemplar_data=exemplar_data,
                embedding_model=embedding_model,
                api_managers=api_managers,
                config=current_config
            )
            
            all_results[exp_name] = phase2_results
            logger.info(f"--- Phase 2 '{exp_name}' finished. ---")
    elif phase2_configs:
        logger.info(
            "Core Simplification Phase 2 is deferred until the finalizer has "
            "strictly merged all Phase 1 donor shards."
        )

    # SPECIAL CASE: Merging Dataset Construction
    merging_ds_configs = [
        exp for exp in experiment_configs
        if _merge_experiment_config(global_config, exp).get(
            "APPLY_MERGING_DATASET_CONSTRUCTION"
        )
    ]
    # Update experiment_configs to exclude merging configs from the standard pipeline
    experiment_configs = [
        exp for exp in experiment_configs
        if not _merge_experiment_config(global_config, exp).get(
            "APPLY_MERGING_DATASET_CONSTRUCTION"
        )
    ]

    if merging_ds_configs:
        logger.info(f"Found {len(merging_ds_configs)} Merging Dataset Construction config(s); running them now.")
        # We will create this file in the next step!
        from src.merging_dataset_builder import build_merging_dataset
        
        for exp_overrides in merging_ds_configs:
            current_config = _merge_experiment_config(global_config, exp_overrides)
            exp_name = current_config.get("experiment_name", "merging_dataset_construction")
            
            logger.info(f"--- Merging Dataset Construction '{exp_name}' starting ---")
            
            # Pass everything to the new dataset builder
            ds_results = build_merging_dataset(
                hard_questions=hard_questions,
                hard_solutions=hard_solutions,
                exemplar_data=exemplar_data,
                embedding_model=embedding_model,
                api_managers=api_managers,
                config=current_config
            )
            
            all_results[exp_name] = ds_results
            logger.info(f"--- Merging Dataset Construction '{exp_name}' finished. ---")

    # If there are no normal experiments left to run, return early
    if not experiment_configs:
        if distributed_manifest is not None:
            _finalize_distributed_worker_status(
                global_config, distributed_experiment_configs, distributed_manifest
            )
        return all_results
        
        
    is_cross_experiment_defer_enabled = any(
        exp.get('DEFER_SOLVE_STEP', False) for exp in experiment_configs
    )

    if is_cross_experiment_defer_enabled:
        logger.info("Cross-experiment deferred mode is ENABLED. Running in two phases.")
        print("\n" + "#"*25 + " PHASE 1: EXECUTING INTERMEDIATE STEPS FOR ALL EXPERIMENTS " + "#"*25)
        
        # PHASE 1: Intermediate Steps for ALL experiments 
        for exp_overrides in experiment_configs:
            current_config = _merge_experiment_config(global_config, exp_overrides)
            exp_name = current_config.get("experiment_name", "unnamed_experiment")
            
            # Only run intermediate steps for experiments that are actually deferred
            if not current_config.get('DEFER_SOLVE_STEP', False):
                logger.warning(f"Experiment '{exp_name}' does not have DEFER_SOLVE_STEP enabled. It will be SKIPPED in this run.")
                continue

            logger.info(f"########## Starting Phase 1 (Intermediate) for Experiment: {exp_name} ##########")
            log_file_path = os.path.join(global_config['RESULTS_DIR'], f"{exp_name}_run_log.json")
            
            run_logs = load_json(log_file_path) or []
            completed_intermediate_indices = {log['target_query_original_hard_list_idx'] for log in run_logs if log.get('pipeline_status') == 'INTERMEDIATE_COMPLETE'}
            queries_to_process = [(idx, q) for idx, q in enumerate(hard_questions) if idx not in completed_intermediate_indices]

            if queries_to_process:
                if current_config.get("BATCH_PROCESSING_ENABLED", False):
                    run_logs = _run_pipeline_items_in_batches(
                        items=[QuestionWorkItem(index=index, question=question) for index, question in queries_to_process],
                        existing_logs=run_logs,
                        log_file_path=log_file_path,
                        experiment_name=exp_name,
                        phase_name="intermediate",
                        run_mode="intermediate",
                        current_config=current_config,
                        embedding_model=embedding_model,
                        exemplar_data=exemplar_data,
                        api_managers=api_managers,
                        hard_solutions=hard_solutions,
                    )
                else:
                    for loop_idx, (original_idx, query_text) in enumerate(tqdm(queries_to_process, desc=f"{exp_name} - Phase 1: Intermediate")):
                        intermediate_log = run_pipeline_for_single_query(
                            hard_list_idx=original_idx, target_query=query_text, config=current_config,
                            embedding_model=embedding_model, exemplar_data=exemplar_data, api_managers=api_managers,
                            run_mode='intermediate', hard_solutions=hard_solutions,
                        )
                        run_logs.append(intermediate_log)
                        save_json(run_logs, log_file_path)
                        if 'layer1_base_execution_state' in intermediate_log:
                            l1_state = intermediate_log['layer1_base_execution_state']
                            t_k = current_config.get("TOP_N_CANDIDATES_RETRIEVAL", 5)
                            n_cands = current_config.get("LAYER1_ONE_SHOT_CANDIDATES_N")
                            if n_cands is None: n_cands = current_config.get("LAYER1_N_CANDIDATES")
                            if n_cands is None: n_cands = t_k
                            c_dir = current_config.get("LAYER1_CACHE_DIR", current_config.get("RESULTS_DIR"))
                            c_file = _get_cache_filename(
                                t_k, n_cands, exp_name,
                                current_config.get("LAYER1_ZERO_SHOT_CANDIDATES_N", 0)
                            )
                            _save_cached_state(_get_cache_path(c_dir, c_file), original_idx, l1_state, t_k, n_cands, exp_name)
                        periodic_sync_check(loop_idx, current_config)
            else:
                logger.info(f"All intermediate steps for '{exp_name}' are already complete.")

        print("\n" + "#"*25 + " PHASE 1 COMPLETE " + "#"*25)
        print("\n" + "#"*25 + " PHASE 2: EXECUTING FINAL SOLVE STEPS FOR ALL EXPERIMENTS " + "#"*25)

        # --- PHASE 2: Final Solving Steps for ALL experiments ---
        for exp_overrides in experiment_configs:
            current_config = _merge_experiment_config(global_config, exp_overrides)
            exp_name = current_config.get("experiment_name", "unnamed_experiment")

            # Only run solve steps for experiments that are deferred
            if not current_config.get('DEFER_SOLVE_STEP', False):
                continue
            
            logger.info(f"########## Starting Phase 2 (Solving) for Experiment: {exp_name} ##########")
            log_file_path = os.path.join(global_config['RESULTS_DIR'], f"{exp_name}_run_log.json")

            intermediate_logs = load_json(log_file_path) or []
            logs_to_solve = [log for log in intermediate_logs if log.get('pipeline_status') == 'INTERMEDIATE_COMPLETE']
            
            if logs_to_solve:
                completed_logs_map = {log['target_query_original_hard_list_idx']: log for log in intermediate_logs if log.get('pipeline_status') != 'INTERMEDIATE_COMPLETE'}

                if current_config.get("BATCH_PROCESSING_ENABLED", False):
                    final_logs = _run_pipeline_items_in_batches(
                        items=[QuestionWorkItem(index=log['target_query_original_hard_list_idx'], question=log['target_query_text'], existing_log=log) for log in logs_to_solve],
                        existing_logs=list(completed_logs_map.values()),
                        log_file_path=log_file_path,
                        experiment_name=exp_name,
                        phase_name="solve",
                        run_mode="solve_only",
                        current_config=current_config,
                        embedding_model=embedding_model,
                        exemplar_data=exemplar_data,
                        api_managers=api_managers,
                        hard_solutions=hard_solutions,
                    )
                else:
                    for loop_idx, log_to_solve in enumerate(tqdm(logs_to_solve, desc=f"{exp_name} - Phase 2: Solving")):
                        original_idx = log_to_solve['target_query_original_hard_list_idx']
                        query_text = log_to_solve['target_query_text']
                        completed_log = run_pipeline_for_single_query(
                            hard_list_idx=original_idx, target_query=query_text, config=current_config,
                            embedding_model=embedding_model, exemplar_data=exemplar_data, api_managers=api_managers,
                            run_mode='solve_only', existing_log=log_to_solve, hard_solutions=hard_solutions,
                        )
                        completed_logs_map[original_idx] = completed_log
                        save_json(list(completed_logs_map.values()), log_file_path)
                        periodic_sync_check(loop_idx, current_config)
                    final_logs = list(completed_logs_map.values())
            else:
                 logger.info(f"All solve steps for '{exp_name}' are already complete.")
                 final_logs = intermediate_logs
            
            save_json(final_logs, log_file_path)
            all_results[exp_name] = final_logs
            logger.info(f"########## Finished Experiment: {exp_name} ##########")

            # --- AUTOMATIC LAYER 2 EXECUTION ---
            if (
                current_config.get('APPLY_LAYER2_ANALYSIS', False)
                and distributed_manifest is None
            ):
                print("\n" + "#"*25 + f" STARTING LAYER 2 ANALYSIS FOR: {exp_name} " + "#"*25)
                
                layer2_cfg = current_config.get('LAYER2_CONFIG', {})
                l2_out_dir = os.path.join(global_config['RESULTS_DIR'], f"{exp_name}_layer2_analytics")
                
                top_k_val = current_config.get('TOP_N_CANDIDATES_RETRIEVAL', 3)
                n_cand_val = current_config.get("LAYER1_ONE_SHOT_CANDIDATES_N")
                if n_cand_val is None: n_cand_val = current_config.get("LAYER1_N_CANDIDATES")
                if n_cand_val is None: n_cand_val = top_k_val
                
                solver_mgr = api_managers.get(current_config.get("API_PROVIDER_SOLVER", "gemini"))
                eval_mgr = api_managers.get(current_config.get("API_PROVIDER_EVALUATOR", "gemini"))
                
                run_layer2_complete_pipeline(
                    layer1_cache_dir=global_config.get('LAYER1_CACHE_DIR', global_config['RESULTS_DIR']),
                    layer2_output_dir=l2_out_dir,
                    experiment_name=exp_name,
                    api_manager_solve=solver_mgr,
                    api_manager_eval=eval_mgr,
                    layer2_config_dict=layer2_cfg,
                    top_k=top_k_val,
                    n_candidates=n_cand_val,
                    global_config=current_config,
                    exemplar_data=exemplar_data,       
                    hard_questions=hard_questions,      
                    hard_solutions=hard_solutions
                )
                
                print("#"*25 + f" LAYER 2 ANALYSIS COMPLETE FOR: {exp_name} " + "#"*25)
            # -----------------------------------

        print("\n" + "#"*25 + " PHASE 2 COMPLETE. ALL EXPERIMENTS FINISHED. " + "#"*25)

    else:
        # Original Mode: Run each experiment sequentially
        logger.info("Deferred mode is DISABLED. Running experiments sequentially.")
        for exp_overrides in experiment_configs:
            current_config = _merge_experiment_config(global_config, exp_overrides)
            exp_name = current_config.get("experiment_name", "unnamed_experiment")
            logger.info(f"########## Starting Experiment: {exp_name} ##########")
            log_file_path = os.path.join(global_config['RESULTS_DIR'], f"{exp_name}_run_log.json")
            
            # This logic handles both standard (non-deferred) and single-experiment deferred runs
            if not current_config.get('DEFER_SOLVE_STEP', False):
                # --- Standard Mode: Run query-by-query ---
                logger.info(f"Running '{exp_name}' in standard (query-by-query) mode.")
                run_logs = load_json(log_file_path) or []
                if not isinstance(run_logs, list):
                    raise DistributedExecutionError(f"Malformed run log: {log_file_path}")
                if distributed_manifest is not None:
                    progress = _distributed_progress_for_experiment(
                        current_config, exp_name, run_logs, distributed_manifest
                    )
                    completed_indices = set(progress["successful_indices"])
                    allowed_indices = set(global_config["_DISTRIBUTED_ALLOWED_QUERY_INDICES"])
                    queries_to_process = [
                        (idx, hard_questions[idx]) for idx in sorted(allowed_indices)
                        if idx not in completed_indices
                    ]
                    logger.info(
                        "Worker %s resume for '%s': assigned=%s successful=%s remaining=%s",
                        global_config["DISTRIBUTED_WORKER_ID"],
                        exp_name,
                        progress["assigned"],
                        progress["successful"],
                        progress["remaining"],
                    )
                else:
                    # Preserve the legacy single-notebook resume behavior.
                    completed_indices = {
                        log['target_query_original_hard_list_idx'] for log in run_logs
                    }
                    queries_to_process = [
                        (idx, q) for idx, q in enumerate(hard_questions)
                        if idx not in completed_indices
                    ]
                
                if not queries_to_process:
                    logger.info(f"All queries for '{exp_name}' are already processed. Skipping LLM execution.")
                elif current_config.get("BATCH_PROCESSING_ENABLED", False):
                    logger.info("Running '%s' in synchronized batch mode.", exp_name)
                    run_logs = _run_pipeline_items_in_batches(
                        items=[QuestionWorkItem(index=index, question=question) for index, question in queries_to_process],
                        existing_logs=run_logs,
                        log_file_path=log_file_path,
                        experiment_name=exp_name,
                        phase_name="full",
                        run_mode="full",
                        current_config=current_config,
                        embedding_model=embedding_model,
                        exemplar_data=exemplar_data,
                        api_managers=api_managers,
                        hard_solutions=hard_solutions,
                    )
                else:
                    for loop_idx, (original_idx, query_text) in enumerate(tqdm(queries_to_process, desc=f"Running {exp_name}")):
                        single_run_log = run_pipeline_for_single_query(
                            hard_list_idx=original_idx, target_query=query_text, config=current_config,
                            embedding_model=embedding_model, exemplar_data=exemplar_data, api_managers=api_managers,
                            run_mode='full',
                            hard_solutions=hard_solutions  # <--- ADDED FIXED PARAMETER
                        )
                        run_logs.append(single_run_log)
                        save_json(run_logs, log_file_path)
                        
                        # Save Layer 1 Cache Synchronously
                        if 'layer1_base_execution_state' in single_run_log:
                            l1_state = single_run_log['layer1_base_execution_state']
                            t_k = current_config.get("TOP_N_CANDIDATES_RETRIEVAL", 5)
                            n_cands = current_config.get("LAYER1_ONE_SHOT_CANDIDATES_N")
                            if n_cands is None: n_cands = current_config.get("LAYER1_N_CANDIDATES")
                            if n_cands is None: n_cands = t_k
                            c_dir = current_config.get("LAYER1_CACHE_DIR", current_config.get("RESULTS_DIR"))
                            c_file = _get_cache_filename(
                                t_k, n_cands, exp_name,
                                current_config.get("LAYER1_ZERO_SHOT_CANDIDATES_N", 0)
                            )
                            _save_cached_state(_get_cache_path(c_dir, c_file), original_idx, l1_state, t_k, n_cands, exp_name)
                        
                        periodic_sync_check(loop_idx, current_config)
                

                final_logs = run_logs

            else:
                # Single-Experiment Deferred Mode 
                logger.info(f"Running '{exp_name}' in single-experiment deferred solve mode.")
                print(f"\n--- {exp_name}: STARTING PHASE 1 of 2 (Intermediate Steps) ---")
                run_logs = load_json(log_file_path) or []
                completed_intermediate_indices = {log['target_query_original_hard_list_idx'] for log in run_logs if log.get('pipeline_status') == 'INTERMEDIATE_COMPLETE'}
                queries_to_process = [(idx, q) for idx, q in enumerate(hard_questions) if idx not in completed_intermediate_indices]

                if queries_to_process:
                    if current_config.get("BATCH_PROCESSING_ENABLED", False):
                        run_logs = _run_pipeline_items_in_batches(
                            items=[QuestionWorkItem(index=index, question=question) for index, question in queries_to_process],
                            existing_logs=run_logs,
                            log_file_path=log_file_path,
                            experiment_name=exp_name,
                            phase_name="intermediate",
                            run_mode="intermediate",
                            current_config=current_config,
                            embedding_model=embedding_model,
                            exemplar_data=exemplar_data,
                            api_managers=api_managers,
                            hard_solutions=hard_solutions,
                        )
                    else:
                        for loop_idx, (original_idx, query_text) in enumerate(tqdm(queries_to_process, desc=f"{exp_name} - Phase 1: Intermediate")):
                            intermediate_log = run_pipeline_for_single_query(
                                hard_list_idx=original_idx, target_query=query_text, config=current_config,
                                embedding_model=embedding_model, exemplar_data=exemplar_data, api_managers=api_managers,
                                run_mode='intermediate', hard_solutions=hard_solutions,
                            )
                            run_logs.append(intermediate_log)
                            save_json(run_logs, log_file_path)
                            if 'layer1_base_execution_state' in intermediate_log and intermediate_log['layer1_base_execution_state'].get('_needs_saving'):
                                l1_state = intermediate_log['layer1_base_execution_state']
                                t_k = current_config.get("TOP_N_CANDIDATES_RETRIEVAL", 5)
                                n_cands = current_config.get("LAYER1_ONE_SHOT_CANDIDATES_N")
                                if n_cands is None: n_cands = current_config.get("LAYER1_N_CANDIDATES")
                                if n_cands is None: n_cands = t_k
                                c_dir = current_config.get("LAYER1_CACHE_DIR", current_config.get("RESULTS_DIR"))
                                c_file = _get_cache_filename(
                                    t_k, n_cands, exp_name,
                                    current_config.get("LAYER1_ZERO_SHOT_CANDIDATES_N", 0)
                                )
                                _save_cached_state(_get_cache_path(c_dir, c_file), original_idx, l1_state, t_k, n_cands, exp_name)
                            periodic_sync_check(loop_idx, current_config)
                else:
                    logger.info(f"All intermediate steps for '{exp_name}' are already complete.")

                print(f"\n--- {exp_name}: STARTING PHASE 2 of 2 (Final Solving) ---")
                intermediate_logs = load_json(log_file_path) or run_logs
                logs_to_solve = [log for log in intermediate_logs if log.get('pipeline_status') == 'INTERMEDIATE_COMPLETE']
                
                if logs_to_solve:
                    completed_logs_map = {log['target_query_original_hard_list_idx']: log for log in intermediate_logs if log.get('pipeline_status') != 'INTERMEDIATE_COMPLETE'}
                    if current_config.get("BATCH_PROCESSING_ENABLED", False):
                        final_logs = _run_pipeline_items_in_batches(
                            items=[QuestionWorkItem(index=log['target_query_original_hard_list_idx'], question=log['target_query_text'], existing_log=log) for log in logs_to_solve],
                            existing_logs=list(completed_logs_map.values()),
                            log_file_path=log_file_path,
                            experiment_name=exp_name,
                            phase_name="solve",
                            run_mode="solve_only",
                            current_config=current_config,
                            embedding_model=embedding_model,
                            exemplar_data=exemplar_data,
                            api_managers=api_managers,
                            hard_solutions=hard_solutions,
                        )
                    else:
                        for loop_idx, log_to_solve in enumerate(tqdm(logs_to_solve, desc=f"{exp_name} - Phase 2: Solving")):
                            original_idx = log_to_solve['target_query_original_hard_list_idx']
                            query_text = log_to_solve['target_query_text']
                            completed_log = run_pipeline_for_single_query(
                                hard_list_idx=original_idx, target_query=query_text, config=current_config,
                                embedding_model=embedding_model, exemplar_data=exemplar_data, api_managers=api_managers,
                                run_mode='solve_only', existing_log=log_to_solve, hard_solutions=hard_solutions,
                            )
                            completed_logs_map[original_idx] = completed_log
                            save_json(list(completed_logs_map.values()), log_file_path)
                            periodic_sync_check(loop_idx, current_config)
                        final_logs = list(completed_logs_map.values())
                else:
                    logger.info(f"All solve steps for '{exp_name}' are already complete.")
                    final_logs = intermediate_logs
            
            save_json(final_logs, log_file_path)
            all_results[exp_name] = final_logs
            logger.info(f"########## Finished Experiment: {exp_name} ##########")

            if (
                current_config.get('APPLY_LAYER2_ANALYSIS', False)
                and distributed_manifest is None
            ):
                print("\n" + "#"*25 + f" STARTING LAYER 2 ANALYSIS FOR: {exp_name} " + "#"*25)
                
                layer2_cfg = current_config.get('LAYER2_CONFIG', {})
                l2_out_dir = os.path.join(global_config['RESULTS_DIR'], f"{exp_name}_layer2_analytics")
                
                top_k_val = current_config.get('TOP_N_CANDIDATES_RETRIEVAL', 3)
                n_cand_val = current_config.get("LAYER1_ONE_SHOT_CANDIDATES_N")
                if n_cand_val is None: n_cand_val = current_config.get("LAYER1_N_CANDIDATES")
                if n_cand_val is None: n_cand_val = top_k_val
                
                # Fetch the correct API managers to pass into Layer 2
                solver_mgr = api_managers.get(current_config.get("API_PROVIDER_SOLVER", "gemini"))
                eval_mgr = api_managers.get(current_config.get("API_PROVIDER_EVALUATOR", "gemini"))
                
                run_layer2_complete_pipeline(
                    layer1_cache_dir=global_config.get('LAYER1_CACHE_DIR', global_config['RESULTS_DIR']),
                    layer2_output_dir=l2_out_dir,
                    experiment_name=exp_name,
                    api_manager_solve=solver_mgr,       
                    api_manager_eval=eval_mgr,          
                    layer2_config_dict=layer2_cfg,
                    top_k=top_k_val,
                    n_candidates=n_cand_val,
                    global_config=current_config,
                    exemplar_data=exemplar_data,       
                    hard_questions=hard_questions,      
                    hard_solutions=hard_solutions
                )
                print("#"*25 + f" LAYER 2 ANALYSIS COMPLETE FOR: {exp_name} " + "#"*25)
            elif current_config.get('APPLY_LAYER2_ANALYSIS', False):
                logger.info(
                    "Layer 2 for '%s' is deferred until all worker shards pass strict merge.",
                    exp_name,
                )

        print("\n" + "#"*25 + " PHASE 2 COMPLETE. ALL EXPERIMENTS FINISHED. " + "#"*25)
        
    if distributed_manifest is not None:
        _finalize_distributed_worker_status(
            global_config, distributed_experiment_configs, distributed_manifest
        )
    return all_results


def finalize_distributed_experiments(
    experiment_configs: List[Dict[str, Any]],
    global_config: Dict[str, Any],
    hard_questions: List[str],
    exemplar_data: Dict[str, Any],
    api_managers: Dict[str, Any],
    hard_solutions: Optional[List[str]] = None,
    run_layer2: bool = True,
    embedding_model: Optional[SentenceTransformer] = None,
) -> Dict[str, Any]:
    """Download, strictly merge, publish, and optionally run Layer 2 once.

    Run this in one final notebook only after all worker notebooks report
    ``COMPLETE``.  Missing/offline/forgotten workers produce a clear error and
    no canonical merge marker is published.
    """
    if not distributed_enabled(global_config):
        raise DistributedExecutionError(
            "finalize_distributed_experiments requires DISTRIBUTED_EXECUTION_ENABLED=True."
        )
    _validate_distributed_scope(global_config, experiment_configs, hard_solutions)
    if run_layer2 and embedding_model is None and any(
        _merge_experiment_config(global_config, overrides).get(
            "APPLY_CORE_SIMP_PHASE2", False
        )
        for overrides in experiment_configs
    ):
        raise DistributedExecutionError(
            "Core Simplification Phase 2 finalization requires embedding_model."
        )
    paths = configure_worker_paths(global_config)
    run_root = download_distributed_run_for_merge(global_config)

    allow_model_rotation = bool(
        global_config.get("DISTRIBUTED_ALLOW_MODEL_ROTATION", False)
    )
    if allow_model_rotation:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        global_config["_DISTRIBUTED_RUNTIME_CODE_FINGERPRINT"] = (
            resolve_code_fingerprint(project_root)
        )
    requested_manifest = _build_requested_distributed_manifest(
        global_config, experiment_configs, hard_questions, hard_solutions, exemplar_data
    )
    manifest_path = os.path.join(run_root, "manifest.json")
    if _auto_pin_local_legacy_code_fingerprint(
        global_config, manifest_path, requested_manifest
    ):
        requested_manifest = _build_requested_distributed_manifest(
            global_config, experiment_configs, hard_questions, hard_solutions, exemplar_data
        )
    write_or_validate_manifest(
        manifest_path,
        requested_manifest,
        allow_model_rotation=allow_model_rotation,
    )
    expected_manifest = load_json(manifest_path)
    if not isinstance(expected_manifest, dict):
        raise DistributedExecutionError(
            f"Malformed authoritative distributed manifest: {manifest_path}"
        )
    model_changes = validate_manifest_compatibility(
        expected_manifest,
        requested_manifest,
        allow_model_rotation=allow_model_rotation,
    )
    _announce_authorized_model_rotation(model_changes, context="finalizer")

    merged_root = os.path.join(run_root, "merged")
    merge_result = merge_distributed_run(run_root, output_dir=merged_root)
    sync_distributed_merged_results(global_config, merged_root)
    print(
        f"Strict merge complete for run {expected_manifest['run_id']}: "
        f"{expected_manifest['question_count']} questions across "
        f"{expected_manifest['worker_count']} workers."
    )

    layer2_results: Dict[str, Any] = {}
    layer2_status: Dict[str, Any] = {}
    core_simplification_layer2_results: Dict[str, Any] = {}
    core_simplification_layer2_status: Dict[str, Any] = {}
    if run_layer2:
        start_worker_session(global_config)
        deadline_keys = (
            "DISTRIBUTED_EXECUTION_ENABLED",
            "_DISTRIBUTED_STOP_NEW_BATCH_MONOTONIC",
            "_DISTRIBUTED_API_DEADLINE_MONOTONIC",
        )
        for manager in api_managers.values():
            manager_config = getattr(manager, "config", None)
            if isinstance(manager_config, dict):
                manager_config.update({key: global_config[key] for key in deadline_keys})
        merged_results_dir = os.path.join(merged_root, "results")
        for overrides in experiment_configs:
            current_config = _merge_experiment_config(global_config, overrides)
            if not current_config.get("APPLY_LAYER2_ANALYSIS", False):
                continue
            current_config["RESULTS_DIR"] = merged_results_dir
            current_config["LAYER1_CACHE_DIR"] = merged_results_dir
            current_config["DISTRIBUTED_FINALIZER_MODE"] = True
            current_config["_DISTRIBUTED_MERGED_ROOT"] = merged_root
            current_config["_DISTRIBUTED_LAYER2_SYNC_ONLY"] = True
            experiment_name = current_config.get("experiment_name", "unnamed_experiment")
            top_k = current_config.get("TOP_N_CANDIDATES_RETRIEVAL", 3)
            n_candidates = current_config.get("LAYER1_ONE_SHOT_CANDIDATES_N")
            if n_candidates is None:
                n_candidates = current_config.get("LAYER1_N_CANDIDATES")
            if n_candidates is None:
                n_candidates = top_k
            output_dir = os.path.join(
                merged_results_dir, f"{experiment_name}_layer2_analytics"
            )
            solver_manager = api_managers.get(
                current_config.get("API_PROVIDER_SOLVER", "gemini")
            )
            evaluator_manager = api_managers.get(
                current_config.get("API_PROVIDER_EVALUATOR", "gemini")
            )
            layer2_result = run_layer2_complete_pipeline(
                layer1_cache_dir=merged_results_dir,
                layer2_output_dir=output_dir,
                experiment_name=experiment_name,
                api_manager_solve=solver_manager,
                api_manager_eval=evaluator_manager,
                layer2_config_dict=current_config.get("LAYER2_CONFIG", {}),
                top_k=top_k,
                n_candidates=n_candidates,
                global_config=current_config,
                exemplar_data=exemplar_data,
                hard_questions=hard_questions,
                hard_solutions=hard_solutions,
            )
            layer2_results[experiment_name] = layer2_result
            result_rows = layer2_result[0] if isinstance(layer2_result, tuple) else None
            completed_indices = set()
            for row in result_rows or []:
                value = (
                    row.get("target_query_idx")
                    if isinstance(row, dict)
                    else getattr(row, "target_query_idx", None)
                )
                try:
                    completed_indices.add(int(value))
                except (TypeError, ValueError):
                    continue
            missing_indices = sorted(set(range(len(hard_questions))) - completed_indices)
            if not missing_indices:
                state = "COMPLETE"
            elif stop_new_batches_due(global_config):
                state = "PAUSED_TIME_LIMIT"
            else:
                state = "INCOMPLETE"
            status_payload = {
                "state": state,
                "run_id": expected_manifest["run_id"],
                "manifest_sha256": expected_manifest["manifest_sha256"],
                "experiment_name": experiment_name,
                "expected_query_count": len(hard_questions),
                "completed_query_count": len(completed_indices),
                "missing_query_indices": missing_indices,
            }
            status_path = os.path.join(output_dir, "distributed_layer2_status.json")
            if not save_json_atomic(status_payload, status_path):
                raise DistributedExecutionError(
                    f"Could not save distributed Layer-2 status: {status_path}"
                )
            layer2_status[experiment_name] = status_payload

        for overrides in experiment_configs:
            current_config = _merge_experiment_config(global_config, overrides)
            if not current_config.get("APPLY_CORE_SIMP_PHASE2", False):
                continue
            current_config["RESULTS_DIR"] = merged_results_dir
            current_config["DISTRIBUTED_FINALIZER_MODE"] = True
            current_config["_DISTRIBUTED_MERGED_ROOT"] = merged_root
            current_config["_DISTRIBUTED_LAYER2_SYNC_ONLY"] = True
            experiment_name = current_config.get(
                "experiment_name", "core_simp_phase2"
            )
            from src.core_simplification_layer2 import (
                execute_core_simplification_phase2,
            )

            phase2_result = execute_core_simplification_phase2(
                hard_questions=hard_questions,
                hard_solutions=hard_solutions,
                exemplar_data=exemplar_data,
                embedding_model=embedding_model,
                api_managers=api_managers,
                config=current_config,
            )
            core_simplification_layer2_results[experiment_name] = phase2_result
            expected_indices = set(
                current_config.get("_CORE_SIMP_PHASE2_EXPECTED_TEST_INDICES", [])
            )
            completed_indices = {
                int(result["test_idx"])
                for result in phase2_result
                if isinstance(result, dict) and "test_idx" in result
            }
            missing_indices = sorted(expected_indices - completed_indices)
            if not missing_indices:
                state = "COMPLETE"
            elif stop_new_batches_due(global_config):
                state = "PAUSED_TIME_LIMIT"
            else:
                state = "INCOMPLETE"
            status_payload = {
                "state": state,
                "run_id": expected_manifest["run_id"],
                "manifest_sha256": expected_manifest["manifest_sha256"],
                "experiment_name": experiment_name,
                "expected_test_count": len(expected_indices),
                "completed_test_count": len(completed_indices & expected_indices),
                "missing_test_indices": missing_indices,
            }
            status_path = os.path.join(
                merged_results_dir,
                f"{experiment_name}_core_simplification_layer2_status.json",
            )
            if not save_json_atomic(status_payload, status_path):
                raise DistributedExecutionError(
                    f"Could not save Core Simplification Layer-2 status: {status_path}"
                )
            core_simplification_layer2_status[experiment_name] = status_payload

        if layer2_results or core_simplification_layer2_results:
            sync_distributed_merged_results(global_config, merged_root)

    return {
        "run_root": run_root,
        "merged_root": merged_root,
        "merge": merge_result,
        "layer2": layer2_results,
        "layer2_status": layer2_status,
        "core_simplification_layer2": core_simplification_layer2_results,
        "core_simplification_layer2_status": core_simplification_layer2_status,
        "worker_paths": paths,
    }
