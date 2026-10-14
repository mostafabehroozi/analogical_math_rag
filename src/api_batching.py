"""Coalesce concurrent API calls into provider batch submissions.

The pipeline makes blocking ``generate_content`` calls from question workers
(``BATCH_MAX_WORKERS``) and from in-question pools
(``QUESTION_PARALLEL_MAX_WORKERS``).  When ``API_BATCHING_ENABLED`` is on, a
batch-capable manager hands each call to an :class:`APIBatchDispatcher`
instead of sending it directly.  The dispatcher holds the calling thread,
collects calls that arrive within ``API_BATCH_COLLECT_SECONDS`` (up to
``API_BATCH_MAX_SIZE``), submits them as one provider batch and wakes every
caller with its own response.  Each caller still runs through
``execute_with_retry``: a failed item is retried individually and simply joins
a later batch, so retries, global pauses, session deadlines and logging keep
their single-call semantics.

Up to ``API_BATCH_MAX_CONCURRENT_BATCHES`` batches are in flight per provider.
While every slot is busy, new calls keep queueing, so the next batch is larger.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List, Optional

from src.context_logger import tprint


@dataclass(frozen=True)
class BatchSettings:
    max_size: int
    collect_seconds: float
    max_concurrent_batches: int
    poll_seconds: float
    timeout_seconds: float

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BatchSettings":
        settings = cls(
            max_size=int(config.get("API_BATCH_MAX_SIZE", 20)),
            collect_seconds=float(config.get("API_BATCH_COLLECT_SECONDS", 2.0)),
            max_concurrent_batches=int(config.get("API_BATCH_MAX_CONCURRENT_BATCHES", 2)),
            poll_seconds=float(config.get("API_BATCH_POLL_SECONDS", 10.0)),
            timeout_seconds=float(config.get("API_BATCH_TIMEOUT_SECONDS", 1800.0)),
        )
        if settings.max_size < 1 or settings.max_concurrent_batches < 1:
            raise ValueError("API_BATCH_MAX_SIZE and API_BATCH_MAX_CONCURRENT_BATCHES must be >= 1.")
        if settings.collect_seconds < 0 or settings.poll_seconds <= 0 or settings.timeout_seconds <= 0:
            raise ValueError(
                "API_BATCH_COLLECT_SECONDS must be >= 0; API_BATCH_POLL_SECONDS and "
                "API_BATCH_TIMEOUT_SECONDS must be > 0."
            )
        return settings


def batching_enabled(config: Dict[str, Any]) -> bool:
    return bool(config.get("API_BATCHING_ENABLED", False))


@dataclass
class BatchItem:
    """One caller's request.  ``payload`` is provider-specific and opaque here."""

    payload: Dict[str, Any]
    group_key: Hashable
    settings: BatchSettings
    enqueued_at: float = field(default_factory=time.monotonic)
    response: Optional[Dict[str, Any]] = None
    done: threading.Event = field(default_factory=threading.Event)


# ``submit_batch(items, settings)`` returns one APIResponse per item, in order.
SubmitBatch = Callable[[List[BatchItem], BatchSettings], List[Dict[str, Any]]]


class APIBatchDispatcher:
    """Thread-safe request coalescer shared by every binding of one manager."""

    def __init__(self, provider_name: str, submit_batch: SubmitBatch) -> None:
        self.provider_name = provider_name
        self._submit_batch = submit_batch
        self._condition = threading.Condition()
        self._pending: List[BatchItem] = []
        self._thread: Optional[threading.Thread] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._slots: Optional[threading.BoundedSemaphore] = None
        self._slot_count = 0
        self._batch_counter = 0

    def call(self, payload: Dict[str, Any], group_key: Hashable, settings: BatchSettings) -> Dict[str, Any]:
        """Queue one request and block until its batch has produced a response."""
        item = BatchItem(payload=payload, group_key=(group_key, settings), settings=settings)
        with self._condition:
            self._ensure_started(settings)
            self._pending.append(item)
            self._condition.notify_all()
        item.done.wait()
        return item.response or {
            "status": "ERROR", "text": None, "error_type": "BatchError",
            "error_message": "Batch finished without a response for this request.",
        }

    def _ensure_started(self, settings: BatchSettings) -> None:
        # Called with the condition held.  The slot count is fixed by the
        # first batch; experiments that need another value start a new kernel.
        if self._thread is not None:
            return
        self._slot_count = settings.max_concurrent_batches
        self._slots = threading.BoundedSemaphore(self._slot_count)
        self._executor = ThreadPoolExecutor(
            max_workers=self._slot_count, thread_name_prefix=f"{self.provider_name}-batch"
        )
        self._thread = threading.Thread(
            target=self._collect_loop, name=f"{self.provider_name}-batch-collector", daemon=True
        )
        self._thread.start()

    def _take_batch(self) -> List[BatchItem]:
        """Wait for the oldest request's window, then take its group (condition held)."""
        while not self._pending:
            self._condition.wait()
        while True:
            oldest = self._pending[0]
            group = [item for item in self._pending if item.group_key == oldest.group_key]
            remaining = oldest.enqueued_at + oldest.settings.collect_seconds - time.monotonic()
            if len(group) >= oldest.settings.max_size or remaining <= 0:
                batch = group[:oldest.settings.max_size]
                taken = {id(item) for item in batch}
                self._pending = [item for item in self._pending if id(item) not in taken]
                return batch
            self._condition.wait(timeout=remaining)

    def _collect_loop(self) -> None:
        while True:
            # Take a slot first: while every slot is busy, requests keep
            # accumulating and the next submission carries all of them.
            self._slots.acquire()
            with self._condition:
                batch = self._take_batch()
                self._batch_counter += 1
                batch_number = self._batch_counter
            self._executor.submit(self._run_batch, batch, batch_number)

    def _run_batch(self, batch: List[BatchItem], batch_number: int) -> None:
        settings = batch[0].settings
        started = time.monotonic()
        label = f"[API BATCH {self.provider_name} #{batch_number}]"
        tprint(f"{label} submitting {len(batch)} request(s).", level="INFO")
        try:
            responses = self._submit_batch(batch, settings)
            if len(responses) != len(batch):
                raise RuntimeError(f"provider returned {len(responses)} responses for {len(batch)} requests")
        except Exception as error:  # A batch-level failure fails each item; callers retry.
            responses = [{
                "status": "ERROR", "text": None, "error_type": "BatchError",
                "error_message": f"{type(error).__name__}: {error}", "error_details": repr(error),
            } for _ in batch]
        finally:
            self._slots.release()
        succeeded = sum(1 for response in responses if response.get("status") == "SUCCESS")
        tprint(
            f"{label} finished in {time.monotonic() - started:.1f}s: "
            f"{succeeded}/{len(batch)} succeeded.",
            level="INFO" if succeeded == len(batch) else "WARNING",
        )
        for item, response in zip(batch, responses):
            response.setdefault("request_meta", {})
            response["request_meta"]["api_batch"] = batch_number
            response["request_meta"]["api_batch_size"] = len(batch)
            item.response = response
            item.done.set()


__all__ = ["APIBatchDispatcher", "BatchItem", "BatchSettings", "batching_enabled"]
