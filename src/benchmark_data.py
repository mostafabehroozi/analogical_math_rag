"""Load and normalize external math benchmarks for the experiment pipeline."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple


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
