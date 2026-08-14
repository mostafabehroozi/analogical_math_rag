"""Batch execution primitives shared by question-level pipeline runners."""

from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence
from tqdm import tqdm  
from src.context_logger import pipeline_context as api_call_context
from src.utils import save_json_atomic


@dataclass(frozen=True)
class QuestionWorkItem:
    index: int
    question: str
    existing_log: Optional[Dict[str, Any]] = None


@dataclass
class QuestionResult:
    item: QuestionWorkItem
    value: Dict[str, Any]
    terminal_status: str
    elapsed_seconds: float


def _terminal_status(result: Dict[str, Any]) -> str:
    status = str(result.get("pipeline_status", result.get("status", "SUCCESS")))
    return "FAILED" if "FAIL" in status.upper() or "ERROR" in status.upper() else "SUCCESS"


class BatchCoordinator:
    """Runs fixed, sequential batches while allowing parallel question execution."""

    def __init__(self, config: Dict[str, Any], experiment_name: str, phase_name: str) -> None:
        self.config = config
        self.experiment_name = experiment_name
        self.phase_name = phase_name
        self.batch_size = max(1, int(config.get("BATCH_SIZE", 5)))
        self.max_workers = max(1, int(config.get("BATCH_MAX_WORKERS", self.batch_size)))
        self.failure_policy = config.get("BATCH_FAILURE_POLICY", "halt_after_failed_batch")
        self.results_dir = config.get("RESULTS_DIR", ".")

    def _journal_path(self, batch_id: str) -> str:
        return os.path.join(self.results_dir, "batch_journals", self.experiment_name, self.phase_name, f"{batch_id}.json")

    def _write_journal(self, batch_id: str, state: str, items: Sequence[QuestionWorkItem], **extra: Any) -> None:
        if not self.config.get("BATCH_WRITE_JOURNALS", True):
            return
        payload = {
            "batch_id": batch_id,
            "experiment_name": self.experiment_name,
            "phase_name": self.phase_name,
            "state": state,
            "question_indices": [item.index for item in items],
            "updated_at": time.time(),
            **extra,
        }
        save_json_atomic(payload, self._journal_path(batch_id))

    def run(
        self,
        items: Sequence[QuestionWorkItem],
        worker: Callable[[QuestionWorkItem], Dict[str, Any]],
        commit: Callable[[List[QuestionResult], str, int], None],
    ) -> List[QuestionResult]:
        """Execute every question in a batch before one coordinator-owned commit."""
        all_results: List[QuestionResult] = []
        
        # Wrap the execution loop in a tqdm progress bar
        with tqdm(total=len(items), desc=f"{self.experiment_name} | {self.phase_name}", unit="q") as pbar:
            for batch_number, start in enumerate(range(0, len(items), self.batch_size), start=1):
                batch_items = list(items[start:start + self.batch_size])
                batch_id = f"{self.phase_name}-{batch_number:04d}-{batch_items[0].index}-{batch_items[-1].index}"
                print(f"\n[BATCH {batch_id}] started: {len(batch_items)} question(s), workers={min(self.max_workers, len(batch_items))}")
                self._write_journal(batch_id, "RUNNING", batch_items)

                results: List[QuestionResult] = []
                def execute(item: QuestionWorkItem) -> QuestionResult:
                    started = time.monotonic()
                    try:
                        with api_call_context(query_idx=item.index, batch_id=batch_id):
                            value = worker(item)
                        return QuestionResult(item, value, _terminal_status(value), time.monotonic() - started)
                    except Exception as exc:  # Keep sibling questions running after a worker failure.
                        value = {
                            "target_query_original_hard_list_idx": item.index,
                            "target_query_text": item.question,
                            "pipeline_status": "FAILED_WORKER_EXCEPTION",
                            "batch_error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                        }
                        return QuestionResult(item, value, "FAILED", time.monotonic() - started)

                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch_items)), thread_name_prefix="question") as executor:
                    futures: Dict[Future[QuestionResult], QuestionWorkItem] = {executor.submit(execute, item): item for item in batch_items}
                    for completed, future in enumerate(as_completed(futures), start=1):
                        result = future.result()
                        results.append(result)
                        print(f"[BATCH {batch_id} {completed}/{len(batch_items)}] Q#{result.item.index} {result.terminal_status} ({result.elapsed_seconds:.1f}s)")

                results.sort(key=lambda result: result.item.index)
                failed = [result for result in results if result.terminal_status != "SUCCESS"]
                self._write_journal(batch_id, "READY_TO_COMMIT", batch_items, terminal_statuses={str(result.item.index): result.terminal_status for result in results})
                
                commit(results, batch_id, batch_number)
                
                self._write_journal(batch_id, "COMMITTED", batch_items, terminal_statuses={str(result.item.index): result.terminal_status for result in results})
                print(f"[BATCH {batch_id}] committed: success={len(results) - len(failed)} failed={len(failed)}")
                all_results.extend(results)

                pbar.update(len(batch_items))

                if failed and self.failure_policy == "halt_after_failed_batch":
                    print(f"[BATCH {batch_id}] stopping before next batch due to BATCH_FAILURE_POLICY={self.failure_policy}")
                    break
                    
        return all_results
