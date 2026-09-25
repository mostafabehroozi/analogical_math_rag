"""Train and evaluate a question-only core-preserving simplifier from Phase 1 JSON."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

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


def simplification_prompt(question: str) -> str:
    return f"{SIMPLIFICATION_INSTRUCTION}\n\nOriginal question:\n{question}"


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
) -> Dict[str, Any]:
    """Compare fixed-base solving with base/adapted simplification on one question."""
    from src.evaluation import evaluate_single_answer_with_llm

    question = record["question"]
    result: Dict[str, Any] = {
        "record_id": record["record_id"], "question": question,
        "label_kind": record["label_kind"], "status": record["status"],
        "ground_truth": record.get("ground_truth"), "arms": {},
    }

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

    direct = generate(
        create_final_reasoning_prompt_simple(
            question, {"PROMPT_TEMPLATE_FINAL_SOLVER_SIMPLE": "final_solver_simple_v3"}
        ), False, solver_max_new_tokens,
    )
    direct_eval = (
        evaluate_single_answer_with_llm(direct["text"], ground_truth, evaluator, dict(evaluator_config))
        if direct.get("status") == "SUCCESS" else None
    )
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
        proxy_solution = generate(
            create_final_reasoning_prompt_simple(
                proxy_text, {"PROMPT_TEMPLATE_FINAL_SOLVER_SIMPLE": "final_solver_simple_v3"}
            ), False, solver_max_new_tokens,
        )
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
        branch["evaluation"] = evaluate_single_answer_with_llm(
            augmented["text"], ground_truth, evaluator, dict(evaluator_config)
        )
        branch["solver_status"] = "EVALUATED"
    evaluations = [direct_eval] + [
        result["arms"][arm].get("evaluation") for arm in ("base", "adapted")
    ]
    result["solver_status"] = "COMPLETE" if all(
        isinstance(value, Mapping) and value.get("status") == "SUCCESS"
        for value in evaluations
    ) else "PARTIAL"
    return result


def summarize_evaluation(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Report behavior by target class and paired solver accuracy on judged rows."""
    summary: Dict[str, Any] = {
        "questions": len(rows), "behavior": {}, "solver": {},
        "solver_status_counts": dict(Counter(r.get("solver_status", "UNKNOWN") for r in rows)),
    }
    for kind in ("copy", "simplify"):
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
        if all(isinstance(value, Mapping) and value.get("status") == "SUCCESS" for value in results):
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
    return summary
