"""External benchmark selection and leakage filtering for fine-tuning notebooks."""

from typing import Any, Dict, Mapping, Optional, Sequence

from src.benchmark_data import HUGGINGFACE_BENCHMARK_SPECS, load_target_benchmarks
from src.merging_finetuning import build_evaluation_populations, normalize_question


def external_evaluation_benchmarks(names: Optional[Sequence[str]] = None):
    """Return an ordered external selection; Numina is reserved for construction."""
    selected = list(HUGGINGFACE_BENCHMARK_SPECS) if names is None else names
    if isinstance(selected, str) or not isinstance(selected, (list, tuple)) or not selected:
        raise ValueError("EVAL_BENCHMARKS must be a non-empty list of external benchmarks.")
    result = []
    for raw_name in selected:
        if not isinstance(raw_name, str):
            raise ValueError("Evaluation benchmark names must be strings.")
        name = raw_name.strip().lower()
        if name not in HUGGINGFACE_BENCHMARK_SPECS:
            raise ValueError(f"Unsupported external benchmark {raw_name!r}; numina_hard is excluded.")
        if name in result:
            raise ValueError(f"Duplicate evaluation benchmark {name!r}.")
        result.append(name)
    return result


def load_external_evaluation_benchmark(
    prepared: Mapping[str, Any], benchmark_name: str, config: Mapping[str, Any],
    *, load_dataset_fn=None,
):
    """Load one full benchmark, group duplicates, and exclude every construction split.

    Returns records, an exclusion/reference audit, and an isolated evaluator
    config. No Numina corpus or hard-question index file is needed to load targets.
    """
    name = external_evaluation_benchmarks([benchmark_name])[0]
    evaluator_config = dict(config)
    evaluator_config.update({
        "TARGET_BENCHMARK": name, "TARGET_BENCHMARKS": [],
        "BENCHMARK_MAX_QUESTIONS": None,
        "_TARGET_BENCHMARK_FOR_QUERY": name,
        "TARGET_BENCHMARK_BY_INDEX": [],
    })
    questions, ground_truths, _ = load_target_benchmarks(
        evaluator_config, {}, load_json_fn=lambda path: None,
        load_dataset_fn=load_dataset_fn,
    )
    audit: Dict[str, Any] = {}
    populations = build_evaluation_populations(
        prepared, questions, ground_truths, audit=audit, include_heldout=False,
    )
    records = populations["remaining_benchmark"]
    for record in records:
        record.update({
            "record_id": f"{name}:{record['benchmark_index']}",
            "question_id": normalize_question(record["question"]),
            "source_benchmark": name,
            "label_kind": "unlabeled", "status": "EXTERNAL_BENCHMARK",
        })
    audit.update({"benchmark": name, "excluded_splits": list(prepared["splits"])})
    return records, audit, evaluator_config
