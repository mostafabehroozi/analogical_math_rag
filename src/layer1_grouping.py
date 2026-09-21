"""Generate and evaluate few-shot candidates for every requested retrieval group.

This is deliberately separate from :mod:`src.layer1_base_execution`.  It does
not calculate evaluator baselines, candidate-conditioned CCS, or a cross-
evaluation matrix.  For a retrieval set of size K and configured group sizes
S, its generation cost is ``sum(comb(K, s) for s in S)`` solver calls plus the
same number of target-answer evaluations.
"""

from __future__ import annotations

import itertools
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.evaluation import evaluate_single_answer_with_llm
from src.parallel_utils import run_parallel_api_calls
from src.prompts import EXEMPLAR_FORMAT, PROMPT_TEMPLATES
from src.utils import create_trace_entry


def normalize_group_sizes(value: Any, retrieved_count: int) -> List[int]:
    """Validate, deduplicate, and sort configured group sizes."""
    if not isinstance(value, (list, tuple)):
        raise ValueError("LAYER1_GROUP_SIZES must be a list of integers.")
    if retrieved_count < 2:
        raise ValueError(
            "Layer-1 grouping requires at least two available retrieved samples."
        )

    normalized: List[int] = []
    for raw_size in value:
        if isinstance(raw_size, bool) or not isinstance(raw_size, (int, np.integer)):
            raise ValueError("Every LAYER1_GROUP_SIZES entry must be an integer.")
        size = int(raw_size)
        if size < 2 or size > retrieved_count:
            raise ValueError(
                "Every LAYER1_GROUP_SIZES entry must be between 2 and the "
                f"available retrieved-sample count ({retrieved_count}); got {size}."
            )
        if size not in normalized:
            normalized.append(size)
    if not normalized:
        raise ValueError("LAYER1_GROUP_SIZES cannot be empty.")
    return sorted(normalized)


def enumerate_retrieval_groups(
    retrieved_set: Sequence[Dict[str, Any]], group_sizes: Sequence[int]
) -> List[Tuple[int, ...]]:
    """Return every position-based combination in deterministic order."""
    positions = range(len(retrieved_set))
    return [
        combination
        for size in group_sizes
        for combination in itertools.combinations(positions, size)
    ]


def _question_to_index_map(exemplar_questions: Sequence[str]) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    for index, question in enumerate(exemplar_questions):
        mapping.setdefault(question, []).append(index)
    return mapping


def _retrieved_set_from_result(
    retrieval_result: Dict[str, Any], exemplar_questions: Sequence[str]
) -> List[Dict[str, Any]]:
    indices = retrieval_result.get("retrieved_indices", [])
    scores = retrieval_result.get("retrieved_similarity_scores", [])
    return [
        {
            "corpus_index": int(corpus_index),
            "question": exemplar_questions[corpus_index],
            "similarity_score": float(scores[position]),
            "retrieval_position": position,
        }
        for position, corpus_index in enumerate(indices)
    ]


def _validate_retrieved_set(
    retrieved_set: Sequence[Dict[str, Any]], exemplar_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    questions = exemplar_data.get("questions", [])
    solutions = exemplar_data.get("solutions", [])
    normalized: List[Dict[str, Any]] = []
    for position, item in enumerate(retrieved_set):
        corpus_index = int(item["corpus_index"])
        if corpus_index < 0 or corpus_index >= len(questions) or corpus_index >= len(solutions):
            raise ValueError(
                f"Retrieved corpus index {corpus_index} has no matching question/solution."
            )
        score = item.get("similarity_score")
        normalized.append(
            {
                "corpus_index": corpus_index,
                "question": str(item.get("question") or questions[corpus_index]),
                "similarity_score": float(score or 0.0),
                "retrieval_position": position,
            }
        )
    return normalized


def build_group_prompt(
    target_query: str,
    member_indices: Sequence[int],
    exemplar_data: Dict[str, Any],
    config: Dict[str, Any],
) -> str:
    """Build one few-shot prompt containing exactly the selected exemplars."""
    template_name = config.get(
        "LAYER1_GROUPING_PROMPT_TEMPLATE", "layer1_grouping_solver_v1"
    )
    template = PROMPT_TEMPLATES.get(template_name)
    if template is None:
        raise ValueError(f"Unknown LAYER1_GROUPING_PROMPT_TEMPLATE: {template_name!r}")

    questions = exemplar_data.get("questions", [])
    solutions = exemplar_data.get("solutions", [])
    examples = []
    for corpus_index in member_indices:
        if corpus_index < 0 or corpus_index >= len(questions) or corpus_index >= len(solutions):
            raise ValueError(
                f"Grouping member {corpus_index} has no matching question/solution."
            )
        examples.append(
            EXEMPLAR_FORMAT.format(
                question=questions[corpus_index], solution=solutions[corpus_index]
            )
        )
    return template.format(
        examples_block="\n\n".join(examples), main_question_text=target_query
    )


def _solver_model_name(config: Dict[str, Any]) -> str:
    provider = str(config.get("API_PROVIDER_SOLVER", "gemini")).strip().lower()
    key_by_provider = {
        "gemini": "GEMINI_MODEL_NAME_FINAL_SOLVER",
        "avalai": "AVALAI_MODEL_NAME_FINAL_SOLVER",
        "ollama": "OLLAMA_MODEL_NAME_FINAL_SOLVER",
    }
    if provider not in key_by_provider:
        raise ValueError(f"Unsupported solver provider for Layer-1 grouping: {provider!r}")
    return config[key_by_provider[provider]]


def _candidate_id(size: int, positions: Iterable[int]) -> str:
    return f"group_s{size}_p" + "-".join(str(position) for position in positions)


def run_layer1_grouping(
    target_query_index: int,
    target_query: str,
    ground_truth_answer: str,
    embedding_model: Any,
    exemplar_data: Dict[str, Any],
    api_manager_solve: Any,
    api_manager_eval: Any,
    config: Dict[str, Any],
    retrieved_set: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate all configured retrieval combinations and label their answers."""
    logger = logging.getLogger(__name__)
    trace: List[Dict[str, Any]] = []
    state: Dict[str, Any] = {
        "metadata": {
            "query_index": target_query_index,
            "schema_version": "1.0",
            "timestamp": time.time(),
            "config_snapshot": {
                "TOP_N_CANDIDATES_RETRIEVAL": config.get(
                    "TOP_N_CANDIDATES_RETRIEVAL", 5
                ),
                "LAYER1_GROUP_SIZES": list(config.get("LAYER1_GROUP_SIZES", [2])),
                "LAYER1_GROUPING_PROMPT_TEMPLATE": config.get(
                    "LAYER1_GROUPING_PROMPT_TEMPLATE", "layer1_grouping_solver_v1"
                ),
            },
        },
        "target_query_data": {
            "query_text": target_query,
            "ground_truth_answer": ground_truth_answer,
            "query_index": target_query_index,
        },
        "retrieved_set": [],
        "candidate_set": {},
        "ground_truth_labels": {},
        "execution_trace": trace,
        "step_statuses": {
            "retrieval": "PENDING",
            "group_generation": "PENDING",
            "ground_truth_evaluation": "PENDING",
        },
    }

    try:
        if retrieved_set is None:
            # Keep schema/combination helpers importable in lightweight offline
            # environments that do not load sentence-transformers.
            from src.pipeline_steps import retrieve

            questions = exemplar_data.get("questions", [])
            retrieval_result = retrieve(
                target_query,
                embedding_model,
                questions,
                exemplar_data["embeddings"],
                top_k=int(config.get("TOP_N_CANDIDATES_RETRIEVAL", 5)),
                question_to_index_map=exemplar_data.get("question_to_index")
                or _question_to_index_map(questions),
            )
            trace.extend(retrieval_result.get("trace", []))
            state["step_statuses"]["retrieval"] = retrieval_result.get(
                "status", "FAILURE"
            )
            if retrieval_result.get("status") != "SUCCESS":
                state["overall_status"] = "FAILURE"
                state["error"] = retrieval_result.get("error", "Retrieval failed.")
                return state
            active_retrieved_set = _retrieved_set_from_result(
                retrieval_result, questions
            )
        else:
            active_retrieved_set = _validate_retrieved_set(retrieved_set, exemplar_data)
            state["step_statuses"]["retrieval"] = "REUSED"

        state["retrieved_set"] = active_retrieved_set
        group_sizes = normalize_group_sizes(
            config.get("LAYER1_GROUP_SIZES", [2]), len(active_retrieved_set)
        )
        combinations = enumerate_retrieval_groups(active_retrieved_set, group_sizes)
        state["metadata"]["effective_group_sizes"] = group_sizes
        state["metadata"]["expected_candidate_count"] = len(combinations)

        model_name = _solver_model_name(config)
        temperature = float(config.get("LAYER1_GROUPING_TEMPERATURE", 0.0))

        def generation_task(positions: Tuple[int, ...]):
            members = [active_retrieved_set[position] for position in positions]
            member_indices = [member["corpus_index"] for member in members]
            prompt = build_group_prompt(
                target_query, member_indices, exemplar_data, config
            )
            response = api_manager_solve.generate_content(
                prompt, model_name, temperature=temperature, avalai_role="final_solver"
            )
            candidate_id = _candidate_id(len(positions), positions)
            trace_entry = create_trace_entry(
                "layer1_grouping",
                f"generate_{candidate_id}",
                {"prompt": prompt, "member_indices": member_indices},
                response,
                {"model": model_name, "temperature": temperature},
            )
            return candidate_id, positions, members, response, trace_entry

        tasks = [
            (lambda positions=positions: generation_task(positions))
            for positions in combinations
        ]
        generation_results = run_parallel_api_calls(tasks, config)
        successful_generation = 0
        for candidate_id, positions, members, response, trace_entry in generation_results:
            trace.append(trace_entry)
            success = response.get("status") == "SUCCESS" and bool(response.get("text"))
            successful_generation += int(success)
            scores = [float(member.get("similarity_score") or 0.0) for member in members]
            state["candidate_set"][candidate_id] = {
                "candidate_id": candidate_id,
                "candidate_text": response.get("text") if success else None,
                "source_exemplar_indices": [member["corpus_index"] for member in members],
                "source_retrieval_positions": list(positions),
                "group_size": len(positions),
                "similarity_scores": scores,
                "similarity_sum": float(sum(scores)),
                "generation_status": "SUCCESS" if success else "FAILURE",
            }

        expected = len(combinations)
        state["step_statuses"]["group_generation"] = (
            "SUCCESS"
            if successful_generation == expected
            else "PARTIAL"
            if successful_generation
            else "FAILURE"
        )

        def evaluation_task(candidate_id: str, candidate_text: str):
            result = evaluate_single_answer_with_llm(
                model_answer=candidate_text,
                ground_truth=ground_truth_answer,
                api_manager=api_manager_eval,
                config=config,
                target_index=target_query_index,
            )
            return candidate_id, result

        evaluation_tasks = []
        for candidate_id, candidate in state["candidate_set"].items():
            text = candidate.get("candidate_text")
            if text:
                evaluation_tasks.append(
                    lambda candidate_id=candidate_id, text=text: evaluation_task(
                        candidate_id, text
                    )
                )
            else:
                state["ground_truth_labels"][candidate_id] = {
                    "is_correct": None,
                    "evaluation_status": "EMPTY_CANDIDATE",
                }

        successful_evaluations = 0
        for candidate_id, result in run_parallel_api_calls(evaluation_tasks, config):
            state["ground_truth_labels"][candidate_id] = {
                "is_correct": result.get("is_correct"),
                "evaluation_status": result.get("status", "FAILURE"),
            }
            successful_evaluations += int(result.get("status") == "SUCCESS")

        state["step_statuses"]["ground_truth_evaluation"] = (
            "SUCCESS"
            if successful_evaluations == expected
            else "PARTIAL"
            if successful_evaluations
            else "FAILURE"
        )
        statuses = state["step_statuses"].values()
        state["overall_status"] = (
            "SUCCESS"
            if all(status in {"SUCCESS", "REUSED"} for status in statuses)
            else "PARTIAL"
            if successful_generation or successful_evaluations
            else "FAILURE"
        )
        return state
    except Exception as exc:
        logger.error("Layer-1 grouping failed: %s", exc, exc_info=True)
        state["overall_status"] = "FAILURE"
        state["error"] = str(exc)
        return state


__all__ = [
    "build_group_prompt",
    "enumerate_retrieval_groups",
    "normalize_group_sizes",
    "run_layer1_grouping",
]
