"""Replay logged LLM verdicts through the answer bank, with no API calls.

``replay_answer_judging`` reads run logs and Pass@K evaluation files judged
before the bank existed, measures how often the rules agree with the LLM's
recorded verdicts, estimates how many judge calls the bank would have made,
and writes every unambiguous verdict to ``answer_bank_replay.jsonl`` so the
next run starts with them::

    summary, disagreements = replay_answer_judging(all_experiments_logs, ground_truths, CONFIG)

Sources: Layer-1 ground-truth labels (``step_e``), grouping labels
(``grouping``), Pass@K attempts (``pass_at_k``), and the mirror runs of the
baseline (``step_c``) and cross-evaluation (``step_d``) steps.  The mirror steps
only log a rate per cell, so a cell's runs get verdicts only when the rate
allows a single assignment (runs with the same answer share a verdict; failed
solves count as incorrect).
"""

from __future__ import annotations

import itertools
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import pandas as pd

from src.answer_bank import REPLAY_FILENAME, bank_path, build_case, forget_answer_bank, rule_verdict
from src.utils import load_json

_BASELINE_STEP = re.compile(r"^baseline_idx_(\d+)_attempt_(\d+)$")
_MIRROR_STEP = re.compile(r"^cand_(.+)_vs_val_(\d+)_run_(\d+)$")
# Pass@K files written with the bank on name the method; only judge calls count.
_LLM_METHODS = {None, "llm", "llm_packed", "llm_joined", "audit"}

Item = Tuple[str, str, str, bool]  # (source, output text, ground truth, logged verdict)


def _lookup(mapping: Mapping[Any, Any], key: str) -> Any:
    """Read a JSON-loaded (string) or in-memory (int) key."""
    if key in mapping:
        return mapping[key]
    return mapping.get(int(key)) if key.isdigit() else None


def _labelled_candidates(state: Mapping[str, Any], ground_truth: str, source: str) -> Iterator[Item]:
    candidates = state.get("candidate_set") or {}
    for candidate_id, label in (state.get("ground_truth_labels") or {}).items():
        text = (candidates.get(candidate_id) or {}).get("candidate_text")
        if text and label.get("evaluation_status") == "SUCCESS" and isinstance(label.get("is_correct"), bool):
            yield source, text, ground_truth, label["is_correct"]


def _run_text(entry: Mapping[str, Any]) -> Optional[str]:
    output = entry.get("output_result") or {}
    return (output.get("text") or None) if output.get("status") == "SUCCESS" else None


def _mirror_cells(state: Mapping[str, Any]) -> Iterator[Tuple[str, str, float, List[Optional[str]]]]:
    """Yield (source, example solution, logged rate, run texts) per baseline and cross cell."""
    solutions: Dict[str, str] = {}
    baseline_runs: Dict[str, List[Optional[str]]] = defaultdict(list)
    mirror_runs: Dict[Tuple[str, str], List[Optional[str]]] = defaultdict(list)
    for entry in state.get("execution_trace") or []:
        sub_step = str(entry.get("sub_step", ""))
        baseline = _BASELINE_STEP.match(sub_step) if entry.get("step_name") == "mirror_phase_0" else None
        mirror = _MIRROR_STEP.match(sub_step) if entry.get("step_name") == "mirror_phase_2" else None
        if baseline:
            ground_truth = (entry.get("input_context") or {}).get("ground_truth")
            if isinstance(ground_truth, str):
                solutions[baseline.group(1)] = ground_truth
            baseline_runs[baseline.group(1)].append(_run_text(entry))
        elif mirror:
            mirror_runs[(mirror.group(1), mirror.group(2))].append(_run_text(entry))

    baselines = state.get("intrinsic_baselines") or {}
    for index, runs in baseline_runs.items():
        rate = _lookup(baselines, index)
        if index in solutions and isinstance(rate, (int, float)):
            yield "step_c", solutions[index], float(rate), runs
    matrix = state.get("cross_evaluation_matrix") or {}
    for (candidate_id, index), runs in mirror_runs.items():
        rate = _lookup(_lookup(matrix, candidate_id) or {}, index)
        if index in solutions and isinstance(rate, (int, float)):
            yield "step_d", solutions[index], float(rate), runs


def _resolve_cell(ground_truth: str, rate: float, runs: List[Optional[str]], n_runs: int) -> List[Tuple[str, bool]]:
    """Per-run verdicts of one cell, when the logged rate allows exactly one assignment."""
    correct = rate * n_runs
    if len(runs) != n_runs or abs(correct - round(correct)) > 1e-6:
        return []
    groups: Dict[str, List[str]] = defaultdict(list)
    for text in runs:
        if text:
            groups[build_case(text, ground_truth).key].append(text)
    sizes = [len(members) for members in groups.values()]
    assignments = [
        chosen for chosen in itertools.product((False, True), repeat=len(sizes))
        if sum(size for size, picked in zip(sizes, chosen) if picked) == round(correct)
    ]
    if len(assignments) != 1:
        return []
    return [(text, verdict) for members, verdict in zip(groups.values(), assignments[0]) for text in members]


def _logged_items(all_experiments_logs: Mapping[str, List[Dict]], ground_truths: List[str], config: Mapping[str, Any]) -> List[Item]:
    n_runs = int(config.get("MIRROR_N_OPTIMIZATION", 3))
    items: List[Item] = []
    for experiment_name, logs in all_experiments_logs.items():
        for log in logs or []:
            index = log.get("target_query_original_hard_list_idx")
            target = ground_truths[index] if isinstance(index, int) and 0 <= index < len(ground_truths) else None
            if target:
                for state_key, source in (("layer1_base_execution_state", "step_e"), ("layer1_grouping_state", "grouping")):
                    items.extend(_labelled_candidates(log.get(state_key) or {}, target, source))
            for source, solution, rate, runs in _mirror_cells(log.get("layer1_base_execution_state") or {}):
                items.extend((source, text, solution, verdict) for text, verdict in _resolve_cell(solution, rate, runs, n_runs))

        eval_path = os.path.join(config["RESULTS_DIR"], f"{experiment_name}_detailed_eval.json")
        for entry in load_json(eval_path) or []:
            index = entry.get("hard_list_idx")
            if not (isinstance(index, int) and 0 <= index < len(ground_truths)):
                continue
            attempts = entry.get("attempts") or []
            methods = entry.get("evaluation_method_list") or [None] * len(attempts)
            for text, status, verdict, method in zip(
                attempts, entry.get("evaluation_status_list") or [], entry.get("is_correct_list") or [], methods
            ):
                if isinstance(text, str) and status == "SUCCESS" and isinstance(verdict, bool) and method in _LLM_METHODS:
                    items.append(("pass_at_k", text, ground_truths[index], verdict))
    return items


def replay_answer_judging(
    all_experiments_logs: Mapping[str, List[Dict]],
    ground_truths: List[str],
    config: Dict[str, Any],
    *,
    write_bank: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary per source, rule-vs-LLM disagreements) and optionally seed the bank.

    Summary columns: ``items`` (logged LLM verdicts), ``answer_extracted``,
    ``rule_exact``/``rule_numeric`` (decided by rule), ``rule_agrees``/
    ``rule_disagrees`` (against the logged verdict), ``bank_reuse`` (an earlier
    verdict for the same answer would have been reused), ``llm_calls`` (still
    needed, before packing and audits) and ``saved_share``.
    """
    stats: Dict[str, Counter] = defaultdict(Counter)
    disagreements: List[Dict[str, Any]] = []
    seen = set()
    rows = []
    for source, text, ground_truth, verdict in _logged_items(all_experiments_logs, ground_truths, config):
        case = build_case(text, ground_truth)
        counter = stats[source]
        counter["items"] += 1
        counter["answer_extracted"] += case.answer is not None
        rule, method = rule_verdict(case)
        if rule is not None:
            counter[f"rule_{method}"] += 1
            counter["rule_agrees" if rule == verdict else "rule_disagrees"] += 1
            if rule != verdict:
                disagreements.append({
                    "source": source, "rule": method, "answer": case.raw_answer,
                    "ground_truth_answer": case.gt_answer, "rule_verdict": rule, "llm_verdict": verdict,
                })
        elif (case.scope, case.key) in seen:
            counter["bank_reuse"] += 1
        else:
            counter["llm_calls"] += 1
        seen.add((case.scope, case.key))
        keys = [case.text_key] + ([case.key] if case.answer is not None else [])
        rows.append({"scope": case.scope, "keys": keys, "verdict": verdict, "method": "replay",
                     "source": source, "answer": case.raw_answer})

    if write_bank and rows:
        path = os.path.join(os.path.dirname(bank_path(config)) or ".", REPLAY_FILENAME)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Rewritten whole on every replay, so replaying again never doubles a verdict.
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        forget_answer_bank(config)

    columns = ["items", "answer_extracted", "rule_exact", "rule_numeric", "rule_agrees",
               "rule_disagrees", "bank_reuse", "llm_calls"]
    summary = pd.DataFrame(
        [{"source": source, **{column: counter[column] for column in columns}} for source, counter in sorted(stats.items())],
        columns=["source", *columns],
    )
    if not summary.empty:
        summary.loc[len(summary)] = ["all", *summary[columns].sum().tolist()]
        summary["saved_share"] = (1 - summary["llm_calls"] / summary["items"]).round(3)
    return summary, pd.DataFrame(disagreements)
