"""Load and normalize external math benchmarks for the experiment pipeline."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


# Benchmarks in this set provide a dedicated gold final-answer column. Their
# ground truths must be compared directly with evaluator_v2, without trying to
# parse a chain-of-thought or a ``\\boxed{...}`` line.
EXACT_FINAL_ANSWER_BENCHMARKS = frozenset(
    {"math500", "math_500", "math-500", "aime25", "aime26"}
)


HUGGINGFACE_BENCHMARK_SPECS: Dict[str, Dict[str, Any]] = {
    "math500": {
        "path": "HuggingFaceH4/MATH-500",
        "split": "test",
        "question_field": "problem",
        "answer_field": "answer",
        "answer_format": "exact",
    },
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "answer_format": "gsm8k",
    },
    "aime25": {
        "path": "MathArena/aime_2025",
        "split": "train",
        "question_field": "problem",
        "answer_field": "answer",
        "answer_format": "exact",
    },
    "aime26": {
        "path": "MathArena/aime_2026",
        "split": "train",
        "question_field": "problem",
        "answer_field": "answer",
        "answer_format": "exact",
    },
}

SUPPORTED_BENCHMARKS = ("numina_hard", *HUGGINGFACE_BENCHMARK_SPECS)


def normalize_benchmark_name(benchmark_name: Any) -> str:
    """Return the canonical configuration spelling used throughout the project."""
    return str(benchmark_name).strip().lower()


def uses_exact_final_answers(benchmark_name: Any) -> bool:
    """Whether the benchmark supplies gold final answers rather than solutions."""
    return normalize_benchmark_name(benchmark_name) in EXACT_FINAL_ANSWER_BENCHMARKS


def selected_benchmark_names(config: Mapping[str, Any]) -> List[str]:
    """Resolve the configured target selection to canonical benchmark names.

    ``TARGET_BENCHMARKS`` is deliberately opt-in: an empty list preserves the
    established scalar ``TARGET_BENCHMARK`` behavior.  A non-empty list is
    validated as one ordered, duplicate-free benchmark selection.
    """
    combined = config.get("TARGET_BENCHMARKS", [])
    if combined is None:
        combined = []
    if combined:
        if not isinstance(combined, list):
            raise ValueError("TARGET_BENCHMARKS must be a list of benchmark names.")
        raw_names: Sequence[Any] = combined
    elif isinstance(combined, list):
        raw_names = [config.get("TARGET_BENCHMARK", "numina_hard")]
    else:
        raise ValueError("TARGET_BENCHMARKS must be a list when configured.")

    names: List[str] = []
    for raw_name in raw_names:
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("Benchmark names must be non-empty strings.")
        name = normalize_benchmark_name(raw_name)
        if name not in SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"Unknown benchmark {raw_name!r}; expected one of: "
                f"{', '.join(SUPPORTED_BENCHMARKS)}."
            )
        if name in names:
            raise ValueError(f"TARGET_BENCHMARKS contains duplicate benchmark {name!r}.")
        names.append(name)
    return names


def benchmark_name_for_target_index(
    config: Mapping[str, Any], target_index: Optional[int] = None
) -> str:
    """Return the source benchmark for one combined-target position.

    The per-index list is internal pipeline metadata.  Its absence keeps old
    logs and scalar benchmark configurations fully backward compatible.
    """
    current_name = config.get("_TARGET_BENCHMARK_FOR_QUERY")
    if current_name is not None:
        return normalize_benchmark_name(current_name)
    indexed_names = config.get("TARGET_BENCHMARK_BY_INDEX")
    if target_index is not None and isinstance(indexed_names, Sequence) and not isinstance(indexed_names, str):
        if isinstance(target_index, bool) or not isinstance(target_index, int):
            raise ValueError("target_index must be an integer.")
        try:
            return normalize_benchmark_name(indexed_names[target_index])
        except IndexError as exc:
            raise IndexError(f"No benchmark metadata for target index {target_index}.") from exc
    return normalize_benchmark_name(config.get("TARGET_BENCHMARK", "numina_hard"))


def load_target_benchmarks(
    config: Mapping[str, Any],
    exemplar_data: Mapping[str, Any],
    *,
    load_json_fn: Callable[[str], Any],
    load_dataset_fn: Optional[Callable[..., Any]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Load one scalar benchmark or ordered benchmark blocks atomically.

    Returns questions, ground truths, and internal canonical benchmark metadata
    aligned one-to-one with the returned targets.  Combined mode intentionally
    ignores ``BENCHMARK_MAX_QUESTIONS``; scalar mode keeps its legacy cap.
    """
    names = selected_benchmark_names(config)
    combined_mode = bool(config.get("TARGET_BENCHMARKS"))
    blocks: List[Tuple[str, List[str], List[str]]] = []

    for name in names:
        if name in HUGGINGFACE_BENCHMARK_SPECS:
            questions, ground_truths = load_huggingface_benchmark(name, load_dataset_fn)
        else:  # ``selected_benchmark_names`` restricts this to numina_hard.
            hard_indices = load_json_fn(config.get("HARD_QUESTIONS_INDICES_PATH"))
            if not isinstance(hard_indices, list) or not hard_indices:
                raise ValueError("HARD_QUESTIONS_INDICES_PATH must contain a non-empty JSON list.")
            corpus_questions = exemplar_data.get("questions", [])
            corpus_solutions = exemplar_data.get("solutions", [])
            if len(corpus_questions) != len(corpus_solutions):
                raise ValueError("Numina exemplar questions and solutions must have matching lengths.")
            invalid_indices = [
                index for index in hard_indices
                if isinstance(index, bool) or not isinstance(index, int)
                or index < 0 or index >= len(corpus_questions)
            ]
            if invalid_indices:
                raise IndexError(f"Invalid hard-question indices: {invalid_indices[:10]}")
            if len(set(hard_indices)) != len(hard_indices):
                raise ValueError("HARD_QUESTIONS_INDICES_PATH must not contain duplicate indices.")
            questions = [str(corpus_questions[index]) for index in hard_indices]
            ground_truths = [str(corpus_solutions[index]) for index in hard_indices]

        if not questions or len(questions) != len(ground_truths):
            raise ValueError(f"Benchmark '{name}' produced invalid target data.")
        blocks.append((name, questions, ground_truths))

    questions = [question for _, block, _ in blocks for question in block]
    ground_truths = [answer for _, _, block in blocks for answer in block]
    target_names = [name for name, block, _ in blocks for _ in block]

    if not combined_mode:
        max_questions = config.get("BENCHMARK_MAX_QUESTIONS")
        if max_questions is not None:
            if isinstance(max_questions, bool) or not isinstance(max_questions, int) or max_questions <= 0:
                raise ValueError("BENCHMARK_MAX_QUESTIONS must be None or a positive integer.")
            questions = questions[:max_questions]
            ground_truths = ground_truths[:max_questions]
            target_names = target_names[:max_questions]

    if not questions:
        raise ValueError(f"Benchmark selection {names!r} produced no target questions.")
    return questions, ground_truths, target_names


def _format_gsm8k_answer(answer: Any) -> str:
    """Preserve GSM8K rationale while making its final answer easy to extract."""
    answer_text = str(answer)
    if "####" not in answer_text:
        return answer_text

    rationale, final_answer = answer_text.rsplit("####", 1)
    return f"{rationale.strip()}\n\nFinal Answer: \\boxed{{{final_answer.strip()}}}"


def _column_values(dataset: Any, field: str, benchmark_name: str) -> List[Any]:
    """Read one required dataset column and raise a useful schema error."""
    try:
        return list(dataset[field])
    except (KeyError, TypeError) as exc:
        available = getattr(dataset, "column_names", None)
        if available is None and isinstance(dataset, Mapping):
            available = list(dataset)
        raise ValueError(
            f"Benchmark '{benchmark_name}' is missing required field {field!r}; "
            f"available fields: {available}"
        ) from exc


def load_huggingface_benchmark(
    benchmark_name: Any,
    load_dataset_fn: Optional[Callable[..., Any]] = None,
) -> Tuple[List[str], List[str]]:
    """Download one supported external benchmark and normalize its two inputs.

    The returned question and ground-truth lists are the exact shape consumed by
    ``run_experiments``. Exact-answer datasets are converted to strings because
    MathArena stores AIME answers as ``int64`` while prompts require text.
    """
    canonical_name = normalize_benchmark_name(benchmark_name)
    if canonical_name not in HUGGINGFACE_BENCHMARK_SPECS:
        supported = ", ".join(HUGGINGFACE_BENCHMARK_SPECS)
        raise ValueError(
            f"Benchmark {canonical_name!r} is not an external Hugging Face benchmark; "
            f"expected one of: {supported}."
        )

    if load_dataset_fn is None:
        from datasets import load_dataset as load_dataset_fn

    spec = HUGGINGFACE_BENCHMARK_SPECS[canonical_name]
    load_args = [spec["path"]]
    if "name" in spec:
        load_args.append(spec["name"])
    dataset = load_dataset_fn(*load_args, split=spec["split"])

    raw_questions = _column_values(dataset, spec["question_field"], canonical_name)
    raw_answers = _column_values(dataset, spec["answer_field"], canonical_name)
    if len(raw_questions) != len(raw_answers):
        raise ValueError(
            f"Benchmark '{canonical_name}' question/answer lengths do not match: "
            f"{len(raw_questions)} != {len(raw_answers)}"
        )

    questions = [str(question).strip() for question in raw_questions]
    if spec["answer_format"] == "gsm8k":
        ground_truths = [_format_gsm8k_answer(answer) for answer in raw_answers]
    else:
        ground_truths = [str(answer).strip() for answer in raw_answers]

    if not questions or any(not question for question in questions):
        raise ValueError(f"Benchmark '{canonical_name}' contains an empty question.")
    if any(not answer for answer in ground_truths):
        raise ValueError(f"Benchmark '{canonical_name}' contains an empty ground-truth answer.")

    return questions, ground_truths
