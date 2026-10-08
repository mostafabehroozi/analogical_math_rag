"""Train and evaluate a question-only core-preserving simplifier from Phase 1 JSON."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from src.merging_evaluation_checkpoints import MergingEvaluationCheckpoint
from src.merging_finetuning import grouped_split, normalize_question, tokenize_splits
from src.prompts import (
    create_core_simp_augmented_solver_prompt,
    create_final_reasoning_prompt_simple,
)
from src.utils import save_json_atomic


SIMPLIFICATION_INSTRUCTION = (
    "Create an easier analogous math question that preserves the original question's "
    "core mathematical method and necessary constraints. Simplify peripheral "
    "difficulty only when it is safe. If no safe simplification is possible, "
    "repeat the original question exactly. Output only the resulting question."
)
_ACCEPTED = "SUCCESS"
_COPY_STATUSES = frozenset({"REJECTED_BY_FILTER", "SKIPPED_FAILSAFE"})
_PROMPT_MARKER = "\nOriginal Question:\n"
_PROMPT_END = "\n\nOUTPUT FORMAT (Strictly follow this format):"
SOLVER_PROMPT_TEMPLATE = "final_solver_simple_v3"
HELDOUT_POPULATION = "heldout"
TRAINING_SOURCE_FILE = "training_source.json"
# Greedy decoding repeats LENGTH_LIMIT, EMPTY and CONTEXT_OVERFLOW outputs exactly,
# so only crashed generations are worth repeating on a rerun.
RETRYABLE_GENERATION_STATUSES = frozenset({"GENERATION_FAILED"})


def simplification_prompt(question: str) -> str:
    return f"{SIMPLIFICATION_INSTRUCTION}\n\nOriginal question:\n{question}"


def solver_prompt(question: str) -> str:
    return create_final_reasoning_prompt_simple(
        question, {"PROMPT_TEMPLATE_FINAL_SOLVER_SIMPLE": SOLVER_PROMPT_TEMPLATE}
    )


def evaluation_prompts() -> Dict[str, str]:
    """Render every evaluation prompt around placeholders for the evaluation identity."""
    return {
        "simplifier": simplification_prompt("<QUESTION>"),
        "solver": solver_prompt("<QUESTION>"),
        "augmented_solver": create_core_simp_augmented_solver_prompt("<QUESTION>", "<SOLVED PROXY>"),
    }


def recover_original_question(row: Mapping[str, Any]) -> Optional[str]:
    """Use explicit fields or the exact Phase 1 generation-prompt boundary."""
    explicit = [
        row[key] for key in ("original_question", "target_query_text")
        if isinstance(row.get(key), str) and row[key].strip()
    ]
    if explicit:
        return explicit[0] if len(set(explicit)) == 1 else None
    found = []
    for entry in row.get("trace") or []:
        if not isinstance(entry, Mapping) or entry.get("sub_step") != "generate_proxy":
            continue
        prompt = (entry.get("input_context") or {}).get("prompt")
        if not isinstance(prompt, str) or prompt.count(_PROMPT_MARKER) != 1:
            continue
        tail = prompt.split(_PROMPT_MARKER, 1)[1]
        if tail.count(_PROMPT_END) != 1:
            continue
        question, remainder = tail.split(_PROMPT_END, 1)
        if question.strip() and "{original_question}" not in question and remainder:
            found.append(question)
    return found[0] if found and len(set(found)) == 1 else None


def _json_list(path: Path) -> list:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_generation_failure(row: Mapping[str, Any]) -> bool:
    """Reject records whose saved generation trace reports an API failure."""
    for entry in row.get("trace") or []:
        if not isinstance(entry, Mapping):
            continue
        step = str(entry.get("sub_step", ""))
        if step != "generate_proxy" and step != "solve_proxy" and "_attempt_" not in step:
            continue
        output = entry.get("output_result")
        if isinstance(output, Mapping) and output.get("status") != "SUCCESS":
            return True
    return False


def load_simplification_data(log_path: str | Path) -> Dict[str, Any]:
    """Build simplification and exact-copy labels from the Phase 1 run log."""
    log_path = Path(log_path)
    logs = _json_list(log_path)
    audit: list[dict] = []
    status_counts: Counter = Counter()

    records: list[dict] = []
    for position, row in enumerate(logs):
        if not isinstance(row, Mapping):
            audit.append({"source": "log", "row": position, "reason": "invalid_record"})
            continue
        status = str(row.get("status", "")).upper()
        status_counts[status or "MISSING"] += 1
        if status not in {_ACCEPTED, *_COPY_STATUSES}:
            audit.append({"source": "log", "row": position, "reason": f"excluded_status:{status}"})
            continue
        if status != _ACCEPTED and _has_generation_failure(row):
            audit.append({"source": "log", "row": position, "reason": "generation_failure_in_trace"})
            continue
        question = recover_original_question(row)
        if not question:
            audit.append({"source": "log", "row": position, "reason": "missing_original"})
            continue
        question_id = normalize_question(question)
        if status == _ACCEPTED:
            proxy = row.get("proxy_question")
            if not isinstance(proxy, str) or not proxy.strip():
                audit.append({"source": "log", "row": position, "reason": "missing_proxy"})
                continue
            target, label_kind = proxy, "simplify"
        else:
            target, label_kind = question, "copy"
        records.append({
            "record_id": f"log:{position}", "question": question, "question_id": question_id,
            "prompt": simplification_prompt(question), "label": target,
            "label_kind": label_kind, "status": status, "ground_truth": row.get("ground_truth"),
            "original_index": row.get("original_index"),
        })

    by_question: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_question[record["question_id"]].append(record)
    unique: list[dict] = []
    for group in by_question.values():
        successes = [item for item in group if item["status"] == _ACCEPTED]

        def improvement(item: dict) -> float:
            row = logs[int(item["record_id"].split(":")[1])]
            base, augmented = row.get("base_score"), row.get("augmented_score")
            if (isinstance(base, (int, float)) and not isinstance(base, bool)
                    and isinstance(augmented, (int, float)) and not isinstance(augmented, bool)
                    and math.isfinite(base) and math.isfinite(augmented)):
                return augmented - base
            return float("-inf")

        chosen = max(successes, key=improvement) if successes else group[0]
        unique.append(chosen)
        for item in group:
            if item is chosen:
                continue
            reason = "superseded_by_success" if successes and item["status"] != _ACCEPTED else "duplicate_question"
            audit.append({"source": "log", "row": int(item["record_id"].split(":")[1]),
                          "reason": reason})

    return {
        "records": unique,
        "audit": audit,
        "sources": [{"path": str(log_path.resolve()), "sha256": _sha256(log_path), "rows": len(logs)}],
        "status_counts": dict(status_counts),
    }


def prepare_simplification_data(
    log_path: str | Path, output_dir: str | Path,
    *, tokenizer: Optional[Any] = None, max_length: int = 4096, seed: int = 42,
) -> Dict[str, Any]:
    loaded = load_simplification_data(log_path)
    splits = grouped_split(loaded["records"], seed=seed)
    manifest = {
        "format_version": 1, "seed": seed, "split_ratios": [0.8, 0.1, 0.1],
        "instruction": SIMPLIFICATION_INSTRUCTION, "sources": loaded["sources"],
        "status_counts": loaded["status_counts"], "audit": loaded["audit"],
        "audit_counts": dict(Counter(item["reason"] for item in loaded["audit"])),
        "split_membership": {key: [r["record_id"] for r in rows] for key, rows in splits.items()},
        "split_label_counts": {
            key: dict(Counter(r["label_kind"] for r in rows)) for key, rows in splits.items()
        },
    }
    tokenized = None
    if tokenizer is not None:
        tokenized, report = tokenize_splits(splits, tokenizer, max_length)
        manifest["token_statistics"] = report
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not save_json_atomic(manifest, str(output / "simplification_data_manifest.json")):
        raise OSError("Could not save simplification data manifest")
    return {"splits": splits, "tokenized": tokenized, "manifest": manifest}


def training_identity(manifest: Mapping[str, Any], qlora_config: Any) -> Dict[str, Any]:
    """Describe an adapter's training data and settings without local paths.

    The result travels with the adapter, so a copy restored from the Hub in a
    later Kaggle session can be checked against the current notebook settings.
    """
    settings = asdict(qlora_config) if is_dataclass(qlora_config) else dict(qlora_config)
    identity = {
        "format_version": 1,
        "run_log_sha256": sorted(source["sha256"] for source in manifest["sources"]),
        "instruction": manifest["instruction"],
        "split_seed": manifest["seed"],
        "split_ratios": manifest["split_ratios"],
        "qlora": {key: value for key, value in settings.items() if key not in {"output_dir", "gpu_index"}},
    }
    return json.loads(json.dumps(identity))


def verify_adapter_training_source(adapter_dir: str | Path, expected: Mapping[str, Any]) -> None:
    """Refuse to evaluate an adapter trained on other data or settings."""
    path = Path(adapter_dir) / TRAINING_SOURCE_FILE
    remedy = ("Use a new HF_MODEL_REPO_ID (and WORK_DIR) for each training configuration, "
              "or set HF_REUSE_ADAPTER_IF_AVAILABLE=False and TRAIN=True to retrain.")
    if not path.is_file():
        raise ValueError(f"Adapter {adapter_dir} has no {TRAINING_SOURCE_FILE}, so its training "
                         f"data and settings cannot be verified. {remedy}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(dict(expected)))
    if saved != expected:
        changed = sorted(key for key in set(saved) | set(expected) if saved.get(key) != expected.get(key))
        raise ValueError(f"Adapter {adapter_dir} was trained with different {', '.join(changed)}. {remedy}")


def load_heldout_population(prepared: Mapping[str, Any], config: Mapping[str, Any]) -> tuple:
    """Return the labeled construction test split as its own evaluation population.

    These questions were never trained on, and their copy/simplify labels show
    whether the adapter changes a question exactly when it should. Judging
    routes to Numina, the construction source. SKIPPED_FAILSAFE rows carry no
    ground truth in Phase 1 logs, so they contribute behavior only.
    """
    records = []
    for record in prepared["splits"]["test"]:
        row = dict(record)
        row["benchmark_index"] = int(str(row["record_id"]).split(":", 1)[1])
        row["source_benchmark"] = HELDOUT_POPULATION
        records.append(row)
    evaluator_config = dict(config)
    evaluator_config.update({
        "TARGET_BENCHMARK": "numina_hard", "TARGET_BENCHMARKS": [],
        "BENCHMARK_MAX_QUESTIONS": None, "_TARGET_BENCHMARK_FOR_QUERY": "numina_hard",
        "TARGET_BENCHMARK_BY_INDEX": [],
    })
    audit = {
        "benchmark": HELDOUT_POPULATION, "source": "construction test split",
        "eligible_questions": len(records),
        "label_counts": dict(Counter(row["label_kind"] for row in records)),
        "with_ground_truth": sum(
            isinstance(row.get("ground_truth"), str) and bool(row["ground_truth"].strip())
            for row in records
        ),
    }
    return records, audit, evaluator_config


class SimplificationEvaluationCheckpoint(MergingEvaluationCheckpoint):
    """The merging notebook's durable per-question checkpoint, in its own Hub repo.

    A finished ``evaluate_question`` result is stored as the question's single
    ``phase_1_runs`` entry and must carry ``benchmark`` and ``benchmark_index``.
    """

    default_repo_name = "simplification-qwen3-4b-evaluation"
    label = "simplification"


def copy_metrics(record: Mapping[str, Any], generation: Mapping[str, Any]) -> Dict[str, Any]:
    text = generation.get("text") if generation.get("status") == "SUCCESS" else None
    question = record["question"]
    return {
        "exact_copy": text == question if text is not None else None,
        "normalized_copy": normalize_question(text) == normalize_question(question)
        if text is not None else None,
        "changed": text != question if text is not None else None,
    }


def evaluate_question(
    record: Mapping[str, Any], generator: Any, evaluator: Any, evaluator_config: Mapping[str, Any],
    *, seed: int = 42, simplifier_max_new_tokens: int = 512,
    solver_max_new_tokens: int = 1024,
    judge: Optional[Callable[[str, str], Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compare fixed-base solving with base/adapted simplification on one question.

    ``judge(answer, ground_truth)`` replaces the direct evaluator call, which lets
    the notebook reuse saved judgments after an interrupted session.
    """
    from src.evaluation import evaluate_single_answer_with_llm

    if judge is None:
        def judge(answer: str, ground_truth: str) -> Mapping[str, Any]:
            return evaluate_single_answer_with_llm(answer, ground_truth, evaluator, dict(evaluator_config))

    question = record["question"]
    result: Dict[str, Any] = {
        "record_id": record["record_id"], "question": question,
        "label_kind": record["label_kind"], "status": record["status"],
        "ground_truth": record.get("ground_truth"), "arms": {},
    }
    for field in ("source_benchmark", "benchmark_index", "benchmark_indices"):
        if field in record:
            result[field] = record[field]

    def generate(prompt: str, use_adapter: bool, max_new_tokens: int) -> Dict[str, Any]:
        return generator(
            prompt, use_adapter=use_adapter, seed=seed, temperature=0.0,
            top_p=1.0, max_new_tokens=max_new_tokens,
        )

    for arm, adapted in (("base", False), ("adapted", True)):
        proxy = generate(simplification_prompt(question), adapted, simplifier_max_new_tokens)
        result["arms"][arm] = {"simplification": proxy, **copy_metrics(record, proxy)}

    ground_truth = record.get("ground_truth")
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        result["solver_status"] = "NO_GROUND_TRUTH"
        return result

    direct = generate(solver_prompt(question), False, solver_max_new_tokens)
    direct_eval = judge(direct["text"], ground_truth) if direct.get("status") == "SUCCESS" else None
    result["direct"] = {"solution": direct, "evaluation": direct_eval}
    for arm in ("base", "adapted"):
        branch = result["arms"][arm]
        proxy = branch["simplification"]
        if proxy.get("status") != "SUCCESS":
            branch["solver_status"] = "SIMPLIFIER_FAILED"
            continue
        if branch["normalized_copy"]:
            branch["solver_status"] = "REUSED_DIRECT"
            branch["evaluation"] = direct_eval
            continue
        proxy_text = proxy["text"]
        proxy_solution = generate(solver_prompt(proxy_text), False, solver_max_new_tokens)
        branch["proxy_solution"] = proxy_solution
        if proxy_solution.get("status") != "SUCCESS":
            branch["solver_status"] = "PROXY_SOLVER_FAILED"
            continue
        augmented = generate(
            create_core_simp_augmented_solver_prompt(
                question, f"Question: {proxy_text}\nRationale and Answer: {proxy_solution['text']}"
            ), False, solver_max_new_tokens,
        )
        branch["original_solution"] = augmented
        if augmented.get("status") != "SUCCESS":
            branch["solver_status"] = "AUGMENTED_SOLVER_FAILED"
            continue
        branch["evaluation"] = judge(augmented["text"], ground_truth)
        branch["solver_status"] = "EVALUATED"
    evaluations = [direct_eval] + [
        result["arms"][arm].get("evaluation") for arm in ("base", "adapted")
    ]
    result["solver_status"] = "COMPLETE" if all(
        isinstance(value, Mapping) and value.get("status") == "SUCCESS"
        for value in evaluations
    ) else "PARTIAL"
    return result


def retryable_failures(case: Mapping[str, Any]) -> List[str]:
    """Name the failures a rerun should repeat: crashed generations and failed judgments.

    Deterministic generation outcomes stay final and are reported as unknown.
    An empty list means the question is complete.
    """
    branches = [("direct", case.get("direct") or {}, ("solution",))]
    branches += [
        (arm, (case.get("arms") or {}).get(arm) or {},
         ("simplification", "proxy_solution", "original_solution"))
        for arm in ("base", "adapted")
    ]
    failures = []
    for name, branch, keys in branches:
        for key in keys:
            output = branch.get(key)
            if isinstance(output, Mapping) and output.get("status") in RETRYABLE_GENERATION_STATUSES:
                failures.append(f"{name}.{key}: {output['status']}")
        judgment = branch.get("evaluation")
        if isinstance(judgment, Mapping) and judgment.get("status") != "SUCCESS":
            failures.append(f"{name}.evaluation: {judgment.get('status', 'UNKNOWN')}")
    return failures


def summarize_evaluation(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Report behavior by target class and paired solver accuracy on judged rows."""
    summary: Dict[str, Any] = {
        "questions": len(rows), "behavior": {}, "solver": {},
        "solver_status_counts": dict(Counter(r.get("solver_status", "UNKNOWN") for r in rows)),
    }
    kinds = ["copy", "simplify"]
    if any(row["label_kind"] == "unlabeled" for row in rows):
        kinds.append("unlabeled")
    for kind in kinds:
        subset = [r for r in rows if r["label_kind"] == kind]
        summary["behavior"][kind] = {"questions": len(subset)}
        for arm in ("base", "adapted"):
            values = [r["arms"][arm].get("exact_copy") for r in subset]
            valid = [value for value in values if value is not None]
            summary["behavior"][kind][arm] = {
                "generated": len(valid),
                "exact_copy_rate": sum(valid) / len(valid) if valid else None,
                "change_rate": 1 - sum(valid) / len(valid) if valid else None,
                "normalized_copy_rate": (
                    sum(r["arms"][arm]["normalized_copy"] for r in subset
                        if r["arms"][arm].get("normalized_copy") is not None) / len(valid)
                    if valid else None
                ),
            }
    paired = []
    for row in rows:
        results = [row.get("direct", {}).get("evaluation")]
        results += [row["arms"][arm].get("evaluation") for arm in ("base", "adapted")]
        if all(isinstance(value, Mapping) and value.get("status") == "SUCCESS"
               and value.get("is_correct") is not None for value in results):
            paired.append(tuple(bool(value["is_correct"]) for value in results))
    summary["solver"] = {
        "eligible_with_ground_truth": sum(bool(r.get("ground_truth")) for r in rows),
        "paired_judged": len(paired),
        "accuracy": {
            arm: sum(row[index] for row in paired) / len(paired) if paired else None
            for index, arm in enumerate(("direct", "base", "adapted"))
        },
        "paired_adapted_minus_base": (
            sum(row[2] - row[1] for row in paired) / len(paired) if paired else None
        ),
        "paired_adapted_minus_direct": (
            sum(row[2] - row[0] for row in paired) / len(paired) if paired else None
        ),
    }
    # Keep the original three-arm cohort metrics and also expose each arm's
    # coverage plus pairwise cohorts, so missing judgments cannot look wrong.
    from src.merging_evaluation_reporting import paired_arm_effect

    def evaluation(row, arm):
        return (row.get("direct", {}) if arm == "direct" else row["arms"][arm]).get("evaluation")

    summary["solver"]["arms"] = {}
    for arm in ("direct", "base", "adapted"):
        judgments = [evaluation(row, arm) for row in rows]
        known = [value["is_correct"] for value in judgments
                 if isinstance(value, Mapping) and value.get("status") == "SUCCESS"
                 and value.get("is_correct") is not None]
        summary["solver"]["arms"][arm] = {
            "questions": len(rows), "judged": len(known), "correct": sum(known),
            "unknown": len(rows) - len(known),
            "coverage": len(known) / len(rows) if rows else 0.0,
            "accuracy": sum(known) / len(known) if known else None,
            "judge_status_counts": dict(Counter(
                value.get("status", "UNKNOWN") if isinstance(value, Mapping) else "NOT_JUDGED"
                for value in judgments)),
        }
    summary["solver"]["pairwise"] = {}
    for baseline in ("base", "direct"):
        runs = []
        for index, row in enumerate(rows):
            for source, arm in ((baseline, "base"), ("adapted", "adapted")):
                value = evaluation(row, source)
                correct = value.get("is_correct") if isinstance(value, Mapping) and value.get("status") == "SUCCESS" else None
                runs.append({"benchmark_index": index, "arm": arm, "evaluation": {"root_correct": correct}})
        summary["solver"]["pairwise"][f"adapted_minus_{baseline}"] = paired_arm_effect(runs)
    summary["generation"] = {}
    for arm in ("direct", "base", "adapted"):
        outputs = []
        for row in rows:
            branch = row.get("direct", {}) if arm == "direct" else row["arms"][arm]
            keys = ("solution",) if arm == "direct" else ("simplification", "proxy_solution", "original_solution")
            outputs.extend(branch[key] for key in keys if isinstance(branch.get(key), Mapping))
        summary["generation"][arm] = {
            "recorded_calls": len(outputs),
            "status_counts": dict(Counter(value.get("status", "UNKNOWN") for value in outputs)),
            "input_tokens": sum(value.get("input_tokens", 0) or 0 for value in outputs),
            "output_tokens": sum(value.get("output_tokens", 0) or 0 for value in outputs),
            "elapsed_seconds": sum(value.get("elapsed_seconds", 0) or 0 for value in outputs),
            "resource_metadata_calls": sum(any(key in value for key in ("input_tokens", "output_tokens", "elapsed_seconds")) for value in outputs),
            "solver_status_counts": dict(Counter(row["arms"][arm].get("solver_status", "UNKNOWN") for row in rows)) if arm != "direct" else {},
        }
    return summary
