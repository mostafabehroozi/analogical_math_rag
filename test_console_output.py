"""Cell clearing, the failures-only printing.txt and the API timing summary."""

import io
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from src import api_manager, console_output, hf_sync
from src.batching import BatchCoordinator, QuestionWorkItem
from src.context_logger import tprint

NO_RETRY = {"ENABLE_API_RETRY": False, "GLOBAL_CONSECUTIVE_ERROR_LIMIT": 0,
            "GLOBAL_PERIODIC_PAUSE_INTERVAL_MINUTES": 0}


@pytest.fixture(autouse=True)
def fresh_console_state(monkeypatch):
    monkeypatch.setattr(console_output, "api_timings", console_output._ApiTimings())
    console_output.configure({"AUTO_CLEAR_CELL_OUTPUT": False})
    console_output._budget.reset()
    yield
    console_output.configure({"AUTO_CLEAR_CELL_OUTPUT": False})
    console_output._budget.reset()


def printing_txt(tmp_path: Path) -> str:
    path = tmp_path / "printing.txt"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_counting_stream_counts_lines_and_skips_progress_redraws():
    inner = io.StringIO()
    stream = console_output._CountingStream(inner)
    stream.write("one\ntwo\n")
    stream.write("\r 40%|████      |")  # tqdm redrawing its bar in place
    stream.write("three")
    assert console_output.printed_since_clear() == (2, len("one\ntwo\n") + len("three"))
    assert inner.getvalue().endswith("three")
    stream._thread_to_parent = {}  # ipykernel's guard sets this on the real stream
    assert inner._thread_to_parent == {}


def test_cell_is_cleared_once_its_line_budget_is_spent(monkeypatch, capsys):
    fake_display = types.ModuleType("IPython.display")
    cleared = []
    fake_display.clear_output = lambda wait=False: cleared.append(wait)
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display)
    monkeypatch.setattr(console_output, "in_notebook_kernel", lambda: True)
    monkeypatch.setattr(console_output, "_install_counter", lambda: None)
    console_output.configure({"AUTO_CLEAR_CELL_OUTPUT_MAX_LINES": 5, "AUTO_CLEAR_CELL_OUTPUT_MAX_CHARS": 0})
    console_output.api_timings.record("p | m", 2.0, True)
    after = []

    console_output._budget.add("line\n" * 4)
    assert not console_output.maybe_clear_output(lambda: after.append(1))
    console_output._budget.add("line\n")
    assert console_output.maybe_clear_output(lambda: after.append(1))

    assert cleared == [True] and after == [1]
    out = capsys.readouterr().out
    assert "[OUTPUT CLEARED] after 5 lines" in out and "[API TIMING]" in out
    assert console_output.printed_since_clear()[0] < 5  # Counting starts again.


def test_character_budget_also_clears(monkeypatch):
    monkeypatch.setitem(sys.modules, "IPython.display", types.SimpleNamespace(clear_output=lambda wait=False: None))
    monkeypatch.setattr(console_output, "in_notebook_kernel", lambda: True)
    monkeypatch.setattr(console_output, "_install_counter", lambda: None)
    console_output.configure({"AUTO_CLEAR_CELL_OUTPUT_MAX_LINES": 0, "AUTO_CLEAR_CELL_OUTPUT_MAX_CHARS": 100})
    console_output._budget.add("x" * 99 + "\n")
    assert console_output.maybe_clear_output()


def test_nothing_is_cleared_outside_a_notebook_or_when_off():
    console_output.configure({"AUTO_CLEAR_CELL_OUTPUT_MAX_LINES": 1})
    console_output._budget.add("a\nb\n")
    assert not console_output.maybe_clear_output()  # pytest is no Jupyter kernel
    console_output.configure({"AUTO_CLEAR_CELL_OUTPUT": False, "AUTO_CLEAR_CELL_OUTPUT_MAX_LINES": 1})
    assert not console_output.maybe_clear_output()


def test_printing_txt_holds_warnings_and_failures_only(tmp_path):
    console_output.configure({"RESULTS_DIR": str(tmp_path)})
    with patch("builtins.print"):
        tprint("all good", level="INFO")
        tprint("hidden", level="DEBUG")
        tprint("slow provider", level="WARNING")
        tprint("broken call", level="ERROR")
        api_manager.execute_with_retry(NO_RETRY, "kaggle", "m", "ok prompt", lambda: {"status": "SUCCESS", "text": "fine"})
        api_manager.execute_with_retry(NO_RETRY, "kaggle", "m", "the failing prompt",
                                       lambda: {"status": "ERROR", "error_type": "InternalServerError", "error_message": "boom"})
    text = printing_txt(tmp_path)
    assert "session started" in text
    assert "WARNING" in text and "slow provider" in text and "broken call" in text
    assert "all good" not in text and "hidden" not in text
    assert "ok prompt" not in text and "fine" not in text  # Successful calls are never written.
    assert "Final failure: kaggle | m" in text and "the failing prompt" in text and "boom" in text


def test_printing_txt_follows_the_clearing_switch(tmp_path):
    console_output.configure({"RESULTS_DIR": str(tmp_path), "AUTO_CLEAR_CELL_OUTPUT": False})
    assert console_output.printing_log_path() is None
    console_output.configure({"RESULTS_DIR": str(tmp_path), "AUTO_CLEAR_CELL_OUTPUT": False, "PRINTING_LOG_ENABLED": True})
    assert console_output.printing_log_path() == str(tmp_path / "printing.txt")
    console_output.configure({"RESULTS_DIR": str(tmp_path), "PRINTING_LOG_ENABLED": False})
    assert console_output.printing_log_path() is None
    console_output.configure({})  # No RESULTS_DIR: nothing is written to the working directory.
    assert console_output.printing_log_path() is None


def test_printing_txt_stops_at_its_size_limit(tmp_path, capsys):
    console_output.configure({"RESULTS_DIR": str(tmp_path), "PRINTING_LOG_MAX_MB": 0.0001})  # ~100 bytes
    for _ in range(20):
        console_output.record_failure("x" * 30)
    assert (tmp_path / "printing.txt").stat().st_size < 200
    assert capsys.readouterr().out.count("reached PRINTING_LOG_MAX_MB") == 1


def test_api_timing_averages_last_window_and_all_calls():
    console_output.configure({"API_TIMING_WINDOW": 2})
    for seconds in (1.0, 2.0, 6.0):
        console_output.api_timings.record("gemini | flash", seconds, True)
    console_output.api_timings.record("gemini | flash", 50.0, False)  # Failures are counted, not timed.
    console_output.api_timings.record("kaggle | m", 1.0, True)
    lines = console_output.api_timings.summary()
    assert "last 2 | all calls" in lines[0]
    flash = next(line for line in lines if "gemini | flash" in line)
    assert "4.00s |   3.00s (3 ok, 1 failed attempts)" in flash
    assert any(line.strip().startswith("ALL") for line in lines)


def test_timing_prints_every_n_calls_outside_batches(capsys):
    console_output.configure({"API_TIMING_PRINT_EVERY_CALLS": 3})
    for _ in range(3):
        console_output.record_api_call("p", "m", 1.0, True)
    assert capsys.readouterr().out.count("[API TIMING]") == 1
    with console_output.batch_run():  # A batch loop prints it after each batch instead.
        for _ in range(3):
            console_output.record_api_call("p", "m", 1.0, True)
    assert "[API TIMING]" not in capsys.readouterr().out


def test_batch_writes_failed_questions_and_prints_timing(tmp_path, capsys):
    config = {"BATCH_SIZE": 2, "BATCH_MAX_WORKERS": 2, "BATCH_WRITE_JOURNALS": False,
              "BATCH_FAILURE_POLICY": "continue", "RESULTS_DIR": str(tmp_path)}
    console_output.api_timings.record("p | m", 1.5, True)

    def worker(item):
        if item.index == 1:
            raise ValueError("worker broke")
        return {"pipeline_status": "SUCCESS"}

    BatchCoordinator(config, "exp", "phase").run(
        [QuestionWorkItem(0, "q0"), QuestionWorkItem(1, "q1")], worker, lambda *args: None)
    text = printing_txt(tmp_path)
    assert "FAILED QUESTION" in text and "Q#1 FAILED" in text and "ValueError: worker broke" in text
    assert "Q#0" not in text
    assert "[API TIMING]" in capsys.readouterr().out


def test_printing_txt_is_synced_by_distributed_workers():
    assert hf_sync._distributed_artifact_allowed(Path("results") / "printing.txt")
    assert not hf_sync._distributed_artifact_allowed(Path("results") / "notes.txt")
