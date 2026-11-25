"""Console output of long runs.

Three pieces, set by config.py:

* Cell clearing (AUTO_CLEAR_CELL_OUTPUT): inside a Jupyter/Kaggle kernel,
  sys.stdout and sys.stderr are wrapped to count what is printed. Once the
  running cell holds AUTO_CLEAR_CELL_OUTPUT_MAX_LINES lines (or _MAX_CHARS
  characters), it is cleared at the next safe point: a finished question.
* printing.txt (PRINTING_LOG_ENABLED, by default on while clearing is on):
  every warning and error -- failed API attempts, final failure reports,
  failed questions -- is appended to RESULTS_DIR/printing.txt. Successful
  calls are never written there. RESULTS_DIR is synced to Hugging Face, so
  the file is uploaded with the results and restored on resume.
* API timing: the average time of a successful call per provider/model, over
  the last API_TIMING_WINDOW calls and over all calls, printed after every
  committed batch, after a clear, and every API_TIMING_PRINT_EVERY_CALLS
  successful calls when no batch is running.

This module imports nothing from the project, so tprint can use it.
"""

from __future__ import annotations

import os
import sys
import threading
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Tuple

PRINTING_LOG_NAME = "printing.txt"


class _Settings:
    clear_enabled = False
    max_lines = 3000
    max_chars = 1_500_000
    printing_log_path: Optional[str] = None
    printing_log_max_bytes = 100 * 1024 * 1024
    timing_window = 50
    timing_print_every = 100


_settings = _Settings()
_lock = threading.Lock()


# --------------------------------------------------------------------------
# Printed-output budget
# --------------------------------------------------------------------------
class _OutputBudget:
    def __init__(self) -> None:
        self.lines = 0
        self.chars = 0

    def add(self, text: Any) -> None:
        if not isinstance(text, str) or not text:
            return
        newlines = text.count("\n")
        if not newlines and "\r" in text:
            return  # A progress bar redrawn in place adds no line to the cell.
        # Unlocked on purpose: print() must stay cheap, and a lost count only
        # moves the next clear by a line.
        self.lines += newlines
        self.chars += len(text)

    def reset(self) -> None:
        self.lines = 0
        self.chars = 0


_budget = _OutputBudget()


class _CountingStream:
    """A stdout/stderr that counts what is written and is otherwise the stream itself.

    Attribute reads and writes go to the wrapped stream, so code that looks
    inside ipykernel's stream (guard_ipykernel_thread_output) still reaches it.
    """

    def __init__(self, stream: Any) -> None:
        object.__setattr__(self, "_counted_stream", stream)

    def write(self, text: Any) -> Any:
        _budget.add(text)
        return self._counted_stream.write(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._counted_stream, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._counted_stream, name, value)


def in_notebook_kernel() -> bool:
    shell = getattr(sys.modules.get("IPython"), "get_ipython", lambda: None)()
    return getattr(shell, "kernel", None) is not None


def _install_counter() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is not None and not isinstance(stream, _CountingStream):
            setattr(sys, name, _CountingStream(stream))


def printed_since_clear() -> Tuple[int, int]:
    return _budget.lines, _budget.chars


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
def printing_log_enabled(config: Mapping[str, Any]) -> bool:
    setting = config.get("PRINTING_LOG_ENABLED")
    if setting is None:  # Follows the cell clearing: what a clear removes stays on disk.
        return bool(config.get("AUTO_CLEAR_CELL_OUTPUT", True))
    return bool(setting)


def printing_log_path() -> Optional[str]:
    return _settings.printing_log_path


def configure(config: Mapping[str, Any]) -> None:
    """Read the settings; cheap, so every run loop calls it once per question."""
    _settings.clear_enabled = bool(config.get("AUTO_CLEAR_CELL_OUTPUT", True))
    _settings.max_lines = max(0, int(config.get("AUTO_CLEAR_CELL_OUTPUT_MAX_LINES", 3000) or 0))
    _settings.max_chars = max(0, int(config.get("AUTO_CLEAR_CELL_OUTPUT_MAX_CHARS", 1_500_000) or 0))
    _settings.printing_log_max_bytes = int(float(config.get("PRINTING_LOG_MAX_MB", 100)) * 1024 * 1024)
    _settings.timing_window = max(1, int(config.get("API_TIMING_WINDOW", 50)))
    _settings.timing_print_every = max(0, int(config.get("API_TIMING_PRINT_EVERY_CALLS", 100) or 0))
    if _settings.clear_enabled and in_notebook_kernel():
        _install_counter()

    path = None
    if printing_log_enabled(config) and config.get("RESULTS_DIR"):
        path = os.path.join(str(config["RESULTS_DIR"]), PRINTING_LOG_NAME)
    if path != _settings.printing_log_path:
        _settings.printing_log_path = path
        if path:
            _append(f"===== session started {datetime.now():%Y-%m-%d %H:%M:%S} "
                    f"(failures and warnings only; successful API calls are not written) =====\n")


# --------------------------------------------------------------------------
# printing.txt
# --------------------------------------------------------------------------
_capped_paths: set = set()


def _append(entry: str) -> None:
    path = _settings.printing_log_path
    if not path:
        return
    with _lock:
        try:
            size = os.path.getsize(path) if os.path.exists(path) else 0
            if size >= _settings.printing_log_max_bytes:
                if path not in _capped_paths:
                    _capped_paths.add(path)
                    print(f"⚠️  {PRINTING_LOG_NAME} reached PRINTING_LOG_MAX_MB; later failures go to "
                          f"batch_execution.log only.")
                return
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(entry)
        except OSError as error:
            # A full disk must not stop the run over a log line.
            sys.__stderr__.write(f"Could not write {path}: {error}\n")


def record_failure(text: str) -> None:
    """Append one failure or warning (any number of lines) to printing.txt."""
    if _settings.printing_log_path:
        _append(f"{datetime.now():%Y-%m-%d %H:%M:%S} {text.rstrip()}\n")


# --------------------------------------------------------------------------
# Clearing
# --------------------------------------------------------------------------
def _clear_due() -> bool:
    lines, chars = printed_since_clear()
    return ((_settings.max_lines and lines >= _settings.max_lines)
            or (_settings.max_chars and chars >= _settings.max_chars))


def maybe_clear_output(after_clear: Optional[Callable[[], None]] = None) -> bool:
    """Clear the running cell if its printed budget is spent; call from the main thread."""
    if not _settings.clear_enabled or not _clear_due() or not in_notebook_kernel():
        return False
    from IPython.display import clear_output

    lines, chars = printed_since_clear()
    clear_output(wait=True)  # Cleared when the next line arrives: the cell never flashes empty.
    _budget.reset()
    where = f"; warnings and failures are kept in {_settings.printing_log_path}" if _settings.printing_log_path else ""
    print(f"[OUTPUT CLEARED] after {lines:,} lines / {chars:,} characters{where}")
    print_api_timing_summary()
    if after_clear is not None:
        after_clear()
    return True


# --------------------------------------------------------------------------
# API timing
# --------------------------------------------------------------------------
class _ApiTimings:
    def __init__(self) -> None:
        self.recent: Dict[str, Deque[float]] = {}
        self.total: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0])
        self.failed: Dict[str, int] = defaultdict(int)
        self.successes = 0
        # For the green line above each failed call in quiet mode.
        self.successes_since_failure = 0
        self.failure_printed = False

    def record(self, label: str, seconds: float, succeeded: bool) -> bool:
        """Record one attempt; True when a periodic summary is due."""
        with _lock:
            if not succeeded:
                self.failed[label] += 1
                return False
            for key in (label, "ALL"):
                window = self.recent.get(key)
                if window is None or window.maxlen != _settings.timing_window:
                    window = self.recent[key] = deque(window or (), maxlen=_settings.timing_window)
                window.append(seconds)
                self.total[key][0] += seconds
                self.total[key][1] += 1
            self.successes += 1
            self.successes_since_failure += 1
            every = _settings.timing_print_every
            return bool(every) and self.successes % every == 0

    def summary(self) -> List[str]:
        with _lock:
            labels = sorted(key for key in self.total if key != "ALL")
            if not labels:
                return []
            rows = labels + (["ALL"] if len(labels) > 1 else [])
            width = max(len(row) for row in rows)
            lines = [f"[API TIMING] seconds per successful call: last {_settings.timing_window} | all calls"]
            for row in rows:
                window, (total, count) = self.recent[row], self.total[row]
                failed = sum(self.failed.values()) if row == "ALL" else self.failed.get(row, 0)
                lines.append(f"    {row:<{width}} : {sum(window) / len(window):6.2f}s | {total / count:6.2f}s "
                             f"({count:,} ok, {failed:,} failed attempts)")
            return lines


api_timings = _ApiTimings()
_batch_runs = 0


def record_api_call(provider: str, model: str, seconds: float, succeeded: bool) -> None:
    if api_timings.record(f"{provider} | {model}", seconds, succeeded) and not _batch_runs:
        print_api_timing_summary()


def take_successes_since_failure() -> Tuple[int, bool]:
    """Successful API calls since the previous printed failure, and restart that count.

    Returns ``(count, first)``; ``first`` is True when no failure was printed
    before in this process, so the count runs from the start of the run.
    """
    with _lock:
        count, first = api_timings.successes_since_failure, not api_timings.failure_printed
        api_timings.successes_since_failure = 0
        api_timings.failure_printed = True
        return count, first


def print_api_timing_summary() -> None:
    lines = api_timings.summary()
    if lines:
        print("\n".join(lines))


class batch_run:
    """Marks a running batch loop, which prints the timing after each batch itself."""

    def __enter__(self) -> "batch_run":
        global _batch_runs
        with _lock:
            _batch_runs += 1
        return self

    def __exit__(self, *_exc: Any) -> None:
        global _batch_runs
        with _lock:
            _batch_runs -= 1
