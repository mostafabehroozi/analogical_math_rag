"""Reuse answer judgments so the LLM judge sees each answer only once.

Every comparison of a generated solution with a ground truth goes through
``evaluate_single_answer_with_llm``.  When ``ANSWER_BANK_ENABLED`` is on, that
function decides in this order:

1. The output's final answer is extracted (the last ``\\boxed{}`` or the line
   after the last ``Final Answer:`` header, whichever comes later) and
   normalized.
2. The bank answers when the LLM already judged the same normalized answer, or
   the same output text, against the same ground truth.
3. Rules answer next: equal to the ground-truth answer is correct; two plain
   numbers are compared by value; a number equal in value to an answer the bank
   already knows gets that answer's verdict.
4. Otherwise the LLM judges.  A request identical to one in flight waits for
   it, and requests that arrive together are packed into one prompt.

A bank is keyed by the ground-truth text.  The judge prompt only ever sees the
output and the ground truth, never the question, so a verdict holds for every
use of that ground truth: each retrieved example's solution in the mirror
steps (baselines, cross-evaluation, reverse validation) and each target
question's answer.

LLM verdicts are appended to ``answer_bank.jsonl`` in ``RESULTS_DIR`` (a direct
results file, so distributed workers sync it with their run logs).  Every
``answer_bank*.jsonl`` there is loaded on first use, including
``answer_bank_replay.jsonl`` written by ``replay_answer_judging``.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Dict, Hashable, Iterator, List, Optional, Sequence, Tuple

BANK_FILENAME = "answer_bank.jsonl"
REPLAY_FILENAME = "answer_bank_replay.jsonl"
MAX_ANSWER_CHARS = 120  # Longer "answers" are prose; only exact text is reused for them
TAIL_LINES = 15         # Output lines a packed prompt shows per item
TAIL_CHARS = 1500

_BOXED_START = re.compile(r"\\(?:boxed|fbox)\s*\{")
_FINAL_HEADER = re.compile(r"final\s+answer[\s*]*[:：]", re.IGNORECASE)
_ANSWER_PREFIX = re.compile(r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is\b|[:：])\s*[:：]?\s*", re.IGNORECASE)


# --- Answer extraction and normalization ---

def boxed_contents(text: str) -> List[str]:
    """Return every ``\\boxed{...}`` (or ``\\fbox{...}``) content, with balanced braces."""
    found = []
    for match in _BOXED_START.finditer(text):
        depth = 1
        for position in range(match.end(), len(text)):
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
                if depth == 0:
                    found.append(text[match.end():position])
                    break
    return found


def extract_final_answer(text: str) -> Optional[str]:
    """Return an output's final answer, or None when it cannot be found cleanly."""
    if not isinstance(text, str):
        return None
    boxed = list(_BOXED_START.finditer(text))
    headers = list(_FINAL_HEADER.finditer(text))
    if boxed and (not headers or boxed[-1].start() > headers[-1].start()):
        contents = boxed_contents(text[boxed[-1].start():])
        answer = contents[0].strip() if contents else ""
    elif headers:
        answer = ""
        for line in text[headers[-1].end():].splitlines():
            line = line.strip().strip("*").strip()
            if line:
                answer = _ANSWER_PREFIX.sub("", line).strip()
                break
    else:
        return None
    return answer if 0 < len(answer) <= MAX_ANSWER_CHARS else None


def ground_truth_answer(text: str) -> Optional[str]:
    """Return a ground truth's single final answer, as written.

    A bare answer (a benchmark's answer field) is returned as it is; a solution
    gives its ``\\boxed{}`` content.  A solution with several different boxed
    answers (a multi-part question) or none returns None, so outputs judged
    against it are only ever reused by their exact text.
    """
    boxed = [content.strip() for content in boxed_contents(text) if content.strip()]
    if boxed:
        return boxed[-1] if len({normalize_answer(content) for content in boxed}) == 1 else None
    stripped = text.strip()
    if stripped and "\n" not in stripped and len(stripped) <= MAX_ANSWER_CHARS:
        return stripped
    return None


def normalize_answer(answer: str) -> str:
    """Remove formatting that never changes an answer's meaning."""
    text = answer.strip()
    if _BOXED_START.match(text):
        inner = boxed_contents(text)
        text = inner[0].strip() if inner else text
    text = text.replace("\u2212", "-").rstrip(".").strip()
    text = text.strip("$").strip()
    if (text.startswith("\\(") and text.endswith("\\)")) or (text.startswith("\\[") and text.endswith("\\]")):
        text = text[2:-2]
    text = re.sub(r"\\(?:text|textbf|mathrm|mathbf|mbox)\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\displaystyle", "")
    text = re.sub(r"\\(?:left|right)(?![A-Za-z])", "", text)
    text = re.sub(r"\\[!,;: ]", "", text)
    text = re.sub(r"\^\s*\{?\\circ\}?", "", text).replace("°", "").replace("\\%", "%")
    text = re.sub(r"\s+", "", text).rstrip(".")
    text = re.sub(r"^[A-Za-z]=(?=[^=]*$)", "", text)  # "x=3" -> "3"; equations with two "=" stay
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        text = text.replace(",", "")
    if re.fullmatch(r"\([A-Z]\)", text):
        text = text[1]
    return text


_DECIMAL = re.compile(r"-?\d+(?:\.(\d+))?")
_SLASH = re.compile(r"(-?\d+)/(\d+)")
_FRAC = re.compile(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}")
_FRAC_SHORT = re.compile(r"(-?)\\frac(\d)(\d)")


def answer_value(answer: str) -> Optional[Tuple[Fraction, Optional[int]]]:
    """Exact value of a plain-number answer and its decimal places (None for exact forms)."""
    match = _DECIMAL.fullmatch(answer)
    if match:
        return Fraction(answer), (len(match.group(1)) if match.group(1) else None)
    match = _SLASH.fullmatch(answer)
    if match:
        sign, numerator, denominator = "", match.group(1), match.group(2)
    else:
        match = _FRAC.fullmatch(answer) or _FRAC_SHORT.fullmatch(answer)
        if not match:
            return None
        sign, numerator, denominator = match.groups()
    if int(denominator) == 0:
        return None
    value = Fraction(int(numerator), int(denominator))
    return (-value if sign else value), None


def numeric_verdict(answer: str, reference: str) -> Optional[bool]:
    """Compare two plain numbers: equal is True, clearly different is False.

    A decimal that matches the other value to its own precision (``0.33`` vs
    ``1/3``) is left to the judge, which decides whether rounding is accepted.
    """
    left, right = answer_value(answer), answer_value(reference)
    if left is None or right is None:
        return None
    if left[0] == right[0]:
        return True
    for _value, places in (left, right):
        if places is not None and abs(left[0] - right[0]) < Fraction(1, 10 ** places):
            return None
    return False


def same_answer(left: Optional[str], right: Optional[str]) -> bool:
    """Whether two extracted answers are the same after normalization."""
    if not left or not right:
        return False
    left, right = normalize_answer(left), normalize_answer(right)
    return left == right or numeric_verdict(left, right) is True


# --- One comparison ---

def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class JudgeCase:
    """What the bank, the rules and the judge need to know about one comparison."""

    scope: str                 # The ground truth's bank
    text_key: str              # The exact output text
    answer: Optional[str]      # Normalized final answer; None when it may not be reused
    raw_answer: Optional[str]  # Final answer as extracted
    gt_answer: Optional[str]   # The ground truth's single final answer, as written
    gt_norm: Optional[str]
    tail: str                  # End of the output, for packed prompts

    @property
    def key(self) -> str:
        return f"a:{self.answer}" if self.answer is not None else self.text_key


def build_case(output: str, ground_truth: str) -> JudgeCase:
    gt_answer = ground_truth_answer(ground_truth)
    raw_answer = extract_final_answer(output)
    answer = normalize_answer(raw_answer) if raw_answer and gt_answer else None
    tail = "\n".join(output.strip().splitlines()[-TAIL_LINES:])[-TAIL_CHARS:]
    return JudgeCase(
        scope=_digest(ground_truth.strip()),
        text_key="t:" + _digest(output),
        answer=answer or None,
        raw_answer=raw_answer,
        gt_answer=gt_answer,
        gt_norm=normalize_answer(gt_answer) if gt_answer else None,
        tail=tail,
    )


def rule_verdict(case: JudgeCase) -> Tuple[Optional[bool], Optional[str]]:
    """Verdict that needs no judge, with the rule that gave it."""
    if case.answer is None or case.gt_norm is None:
        return None, None
    if case.answer == case.gt_norm:
        return True, "exact"
    verdict = numeric_verdict(case.answer, case.gt_norm)
    return (verdict, "numeric") if verdict is not None else (None, None)


# --- The bank ---

class AnswerBank:
    """LLM verdicts per ground truth, kept in memory and appended to a JSONL file."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._lock = threading.Lock()
        self._verdicts: Dict[str, Dict[str, List[bool]]] = {}
        self._counts: Counter = Counter()
        self._hits: Counter = Counter()
        self._rechecked: set = set()
        if path:
            pattern = os.path.join(os.path.dirname(path) or ".", "answer_bank*.jsonl")
            for bank_file in sorted(glob.glob(pattern)):
                self._load(bank_file)

    def _load(self, bank_file: str) -> None:
        with open(bank_file, encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # A line cut off by an interrupted write
                if isinstance(row, dict) and isinstance(row.get("verdict"), bool):
                    for key in row.get("keys") or []:
                        self._add(str(row.get("scope")), str(key), row["verdict"])

    def _add(self, scope: str, key: str, verdict: bool) -> None:
        self._verdicts.setdefault(scope, {}).setdefault(key, []).append(verdict)

    def _get(self, scope: str, key: str) -> List[bool]:
        with self._lock:
            return list(self._verdicts.get(scope, {}).get(key, []))

    def lookup(self, scope: str, key: str) -> Optional[bool]:
        verdicts = self._get(scope, key)
        return verdicts[0] if verdicts and len(set(verdicts)) == 1 else None

    def conflicted(self, scope: str, key: str) -> bool:
        return len(set(self._get(scope, key))) > 1

    def value_verdict(self, scope: str, answer: str) -> Optional[bool]:
        """Verdict of banked answers with exactly this numeric value, if they agree."""
        value = answer_value(answer)
        if value is None:
            return None
        with self._lock:
            entries = [(key, set(verdicts)) for key, verdicts in self._verdicts.get(scope, {}).items()]
        found = set()
        for key, verdicts in entries:
            if key.startswith("a:") and len(verdicts) == 1:
                other = answer_value(key[2:])
                if other is not None and other[0] == value[0]:
                    found |= verdicts
        return found.pop() if len(found) == 1 else None

    def take_recheck(self, scope: str, key: str, threshold: int) -> bool:
        """True once per session when a banked answer has been reused ``threshold`` times
        but only one LLM verdict backs it."""
        with self._lock:
            self._hits[(scope, key)] += 1
            if self._hits[(scope, key)] < threshold or (scope, key) in self._rechecked:
                return False
            if len(self._verdicts.get(scope, {}).get(key, [])) >= 2:
                return False
            self._rechecked.add((scope, key))
            return True

    def record(self, scope: str, keys: Sequence[str], verdict: bool, **fields: Any) -> None:
        """Store one LLM verdict under every key it holds for."""
        row = {"scope": scope, "keys": list(keys), "verdict": bool(verdict), **fields, "time": round(time.time())}
        new_conflicts = []
        with self._lock:
            for key in keys:
                before = set(self._verdicts.get(scope, {}).get(key, []))
                self._add(scope, key, bool(verdict))
                if len(before) == 1 and bool(verdict) not in before:
                    new_conflicts.append(key)
            if self.path:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for key in new_conflicts:
            logging.getLogger(__name__).warning(
                "Answer bank conflict: the judge gave both verdicts for %s in bank %s; "
                "it goes to the LLM from now on.", key, scope,
            )

    def count(self, method: str) -> None:
        with self._lock:
            self._counts[method] += 1

    def counts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counts)


_BANKS: Dict[Optional[str], AnswerBank] = {}
_BANKS_LOCK = threading.Lock()


def bank_path(config: Dict[str, Any]) -> Optional[str]:
    """``ANSWER_BANK_PATH``, else ``RESULTS_DIR/answer_bank.jsonl``; None keeps the bank in memory."""
    path = config.get("ANSWER_BANK_PATH")
    if path:
        return str(path)
    results_dir = config.get("RESULTS_DIR")
    return os.path.join(str(results_dir), BANK_FILENAME) if results_dir else None


def get_answer_bank(config: Dict[str, Any]) -> AnswerBank:
    path = bank_path(config)
    cache_key = os.path.abspath(path) if path else None
    with _BANKS_LOCK:
        if cache_key not in _BANKS:
            _BANKS[cache_key] = AnswerBank(path)
        return _BANKS[cache_key]


def forget_answer_bank(config: Dict[str, Any]) -> None:
    """Drop the loaded bank so the next judgment reads its files again."""
    path = bank_path(config)
    with _BANKS_LOCK:
        _BANKS.pop(os.path.abspath(path) if path else None, None)


def answer_bank_summary(config: Dict[str, Any]) -> Dict[str, int]:
    """How this session's judgments were decided (``llm*`` methods made judge calls)."""
    return get_answer_bank(config).counts()


# --- Packed judge calls ---

_ITEM_HEADER = re.compile(r"^(?!\s*</)\W*item\s*(\d+)\W*$", re.IGNORECASE | re.MULTILINE)


def parse_packed_reply(text: str, count: int) -> List[Optional[Tuple[str, bool]]]:
    """Return ``(extracted answer, verdict)`` per item, None where the reply is unclear."""
    text = text or ""
    found: Dict[int, set] = {}
    headers = list(_ITEM_HEADER.finditer(text))
    for position, header in enumerate(headers):
        end = headers[position + 1].start() if position + 1 < len(headers) else len(text)
        block = text[header.end():end]
        extracted = re.search(r"Extracted Model Final Answer:\s*(.+)", block)
        verdict = re.search(r"Evaluation:\W*(true|false)", block, re.IGNORECASE)
        if extracted and verdict:
            item = (extracted.group(1).strip().strip("*").strip(), verdict.group(1).lower() == "true")
            found.setdefault(int(header.group(1)), set()).add(item)
    # An item answered twice with different content is not trusted.
    return [next(iter(found[n])) if len(found.get(n, ())) == 1 else None for n in range(1, count + 1)]


Outcome = Tuple[Dict[str, Any], bool]  # (evaluation result, whether it holds for identical requests)
_FAILED: Outcome = ({"is_correct": None, "status": "API_FAILURE", "error_details": None}, False)


class _Request:
    def __init__(self, case: JudgeCase):
        self.case = case
        self.packed: Optional[Outcome] = None
        self.packed_ready = threading.Event()
        self.outcome: Outcome = _FAILED
        self.done = threading.Event()


class _Batch:
    def __init__(self):
        self.requests: List[_Request] = []
        self.full = threading.Event()


class PackedJudge:
    """Share and pack the LLM calls that remain after the bank and the rules."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending: Dict[Tuple[str, str], _Request] = {}
        self._open: Dict[Hashable, _Batch] = {}
        self._active = 0

    @contextmanager
    def active(self) -> Iterator[None]:
        """Mark a thread as judging, so a batch waits for others only when there are any."""
        with self._lock:
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1

    def judge(
        self,
        case: JudgeCase,
        single: Callable[[JudgeCase], Outcome],
        pack: Callable[[List[JudgeCase]], List[Optional[Outcome]]],
        *,
        batch_key: Hashable,
        max_items: int,
        collect_seconds: float,
    ) -> Dict[str, Any]:
        """Return the verdict for ``case``.

        ``single(case)`` makes one ordinary judge call; ``pack(cases)`` makes one
        packed call and returns None for any item it could not decide, which is
        then judged on its own.  A request identical to one in flight waits for
        that one's verdict instead of calling again.
        """
        request_key = (case.scope, case.key)
        packable = max_items > 1 and case.answer is not None
        leader = False
        with self._lock:
            request = self._pending.get(request_key)
            owner = request is None
            if owner:
                request = _Request(case)
                self._pending[request_key] = request
                if packable:
                    batch = self._open.get(batch_key)
                    if batch is None:
                        batch, leader = _Batch(), True
                        self._open[batch_key] = batch
                    batch.requests.append(request)
                    if len(batch.requests) >= max_items:
                        self._open.pop(batch_key, None)
                        batch.full.set()
        if not owner:
            request.done.wait()
            result, shareable = request.outcome
            if shareable:
                return {**result, "judge_method": "llm_joined"}
            return single(case)[0]

        outcome = _FAILED
        try:
            if packable:
                if leader:
                    self._send(batch, batch_key, pack, collect_seconds)
                request.packed_ready.wait()
            outcome = request.packed or single(case)
        finally:
            with self._lock:
                self._pending.pop(request_key, None)
            request.outcome = outcome
            request.done.set()
        return outcome[0]

    def _send(self, batch: _Batch, batch_key: Hashable, pack: Callable, collect_seconds: float) -> None:
        try:
            with self._lock:
                others_judging = self._active > 1
            if others_judging:
                batch.full.wait(collect_seconds)
            with self._lock:
                if self._open.get(batch_key) is batch:
                    del self._open[batch_key]
                requests = list(batch.requests)
            if len(requests) > 1:
                for request, outcome in zip(requests, pack([request.case for request in requests])):
                    request.packed = outcome
        except Exception:
            logging.getLogger(__name__).warning(
                "Packed judge call failed; each answer is judged on its own.", exc_info=True,
            )
        finally:
            for request in batch.requests:
                request.packed_ready.set()


PACKED_JUDGE = PackedJudge()
