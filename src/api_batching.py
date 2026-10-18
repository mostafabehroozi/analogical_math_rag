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
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional

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


_COLLECTIONS = (list, tuple, set, frozenset)
_announced_routes: set = set()
_announced_lock = threading.Lock()

# Set by the orchestrator in an experiment's merged config when the experiment's
# own API_BATCHING_ENABLED replaces CONFIG's, so logs and errors can say so.
BATCHING_SOURCE_KEY = "_API_BATCHING_ENABLED_SOURCE"


def batching_setting(config: Mapping[str, Any]) -> str:
    """``API_BATCHING_ENABLED=<value>``, naming the experiment that set it, if any."""
    text = f"API_BATCHING_ENABLED={config.get('API_BATCHING_ENABLED', False)!r}"
    source = config.get(BATCHING_SOURCE_KEY)
    return f"{text} ({source})" if source else text


def validate_batching_setting(value: Any, providers: Iterable[str]) -> None:
    """Raise ValueError for a switch value ``batching_enabled`` would misread.

    A misspelled provider name would otherwise turn batching off (or leave it
    on) without a word, and a malformed rule would fail every call.
    """
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, Mapping):
        names = list(value)
        for name, rule in value.items():
            if not (
                rule is None or isinstance(rule, (bool, str))
                or (isinstance(rule, _COLLECTIONS) and all(isinstance(item, str) for item in rule))
            ):
                raise ValueError(
                    f"API_BATCHING_ENABLED[{name!r}] must be True/False, a model ID or a list "
                    f"of model IDs, not {rule!r}."
                )
    elif isinstance(value, str):
        names = [value]
    elif isinstance(value, _COLLECTIONS):
        names = list(value)
    else:
        raise ValueError(
            "API_BATCHING_ENABLED must be True/False, a provider name, a list of provider "
            f"names or a dict such as {{'kaggle': True, 'openrouter': False}}, not {value!r}."
        )
    known = {str(provider).lower() for provider in providers}
    unknown = sorted(str(name) for name in names if str(name).strip().lower() not in known)
    if unknown:
        raise ValueError(
            f"API_BATCHING_ENABLED names unknown provider(s) {unknown}; "
            f"the providers are {sorted(known)}."
        )


def batching_enabled(
    config: Dict[str, Any], provider_name: str, model_name: Optional[str] = None,
    *, announce: bool = True,
) -> bool:
    """Whether this provider batches calls for this model.

    ``API_BATCHING_ENABLED`` may be:
      * a bool, for every model of every batch-capable provider;
      * a provider name or list of names, e.g. ``["kaggle"]``, for all their models;
      * a dict mapping a provider to a bool (all its models) or to a model ID /
        list of model IDs (only those), e.g.
        ``{"kaggle": True, "openrouter": ["openai/gpt-4o-mini"]}``.
    Anything not selected is sent directly.  While the switch is set, each
    (provider, model) route is logged once per setting, so a run shows what it
    batches and which setting decided it.
    """
    value = config.get("API_BATCHING_ENABLED", False)
    enabled = _batching_rule(value, provider_name, model_name)
    if announce and value is not False and value is not None:
        setting = batching_setting(config)
        route = (provider_name, model_name, enabled, setting)
        with _announced_lock:
            first_time = route not in _announced_routes
            _announced_routes.add(route)
        if first_time:
            tprint(
                f"[API BATCH {provider_name}] {model_name}: "
                f"{'batched' if enabled else 'sent directly'} ({setting}).",
                level="INFO",
            )
    return enabled


def _batching_rule(value: Any, provider_name: str, model_name: Optional[str]) -> bool:
    if isinstance(value, Mapping):
        rules = {str(key).strip().lower(): rule for key, rule in value.items()}
        rule = rules.get(provider_name)
        if rule is None or isinstance(rule, bool):
            return bool(rule)
        if isinstance(rule, str):
            rule = [rule]
        if isinstance(rule, _COLLECTIONS):
            return model_name in {str(item).strip() for item in rule}
        raise ValueError(
            f"API_BATCHING_ENABLED[{provider_name!r}] must be a bool, a model ID or a list of model IDs."
        )
    if isinstance(value, str):
        return value.strip().lower() == provider_name
    if isinstance(value, _COLLECTIONS):
        return provider_name in {str(item).strip().lower() for item in value}
    return bool(value)


def fallback_to_direct(config: Dict[str, Any]) -> bool:
    """Send a model directly once its provider refuses to batch it (full price)."""
    return bool(config.get("API_BATCH_FALLBACK_TO_DIRECT", False))


class BatchRefusedError(RuntimeError):
    """A provider refused to batch a model while API_BATCH_FALLBACK_TO_DIRECT was off."""


class _BatchRefusals:
    """Calls that failed because their provider will not batch their model.

    A refused call only fails its own response, so the pipeline would save the
    question as a FAILURE log, which the single-notebook resume counts as done.
    Every refused call is therefore recorded here, and the orchestrator calls
    :meth:`raise_if_refused` before it saves a question's log or commits a
    question batch.  Only calls after the check's :meth:`mark` count, so a
    refusal from an earlier run in the same kernel never stops a run whose
    calls are no longer refused.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._latest: Dict[str, int] = {}  # refusal message -> number of its latest call

    def record(self, message: str) -> None:
        """Record one call that failed with this refusal message."""
        with self._lock:
            self._count += 1
            self._latest[message] = self._count

    def mark(self) -> int:
        with self._lock:
            return self._count

    def raise_if_refused(self, since: int) -> None:
        """Raise :class:`BatchRefusedError` if a call was refused after ``since``."""
        with self._lock:
            messages = [message for message, number in self._latest.items() if number > since]
        if messages:
            raise BatchRefusedError("\n\n".join(messages))


batch_refusals = _BatchRefusals()


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
            try:
                self._executor.submit(self._run_batch, batch, batch_number)
            except Exception as error:  # e.g. the interpreter is shutting down
                self._slots.release()
                self._deliver(batch, batch_number, self._batch_error(batch, error))

    @staticmethod
    def _batch_error(batch: List[BatchItem], error: Exception) -> List[Dict[str, Any]]:
        return [{
            "status": "ERROR", "text": None, "error_type": "BatchError",
            "error_message": f"{type(error).__name__}: {error}", "error_details": repr(error),
        } for _ in batch]

    def _run_batch(self, batch: List[BatchItem], batch_number: int) -> None:
        settings = batch[0].settings
        started = time.monotonic()
        label = f"[API BATCH {self.provider_name} #{batch_number}]"
        tprint(f"{label} submitting {len(batch)} request(s).", level="INFO")
        try:
            responses = self._submit_batch(batch, settings)
            # A caller only wakes up with its own response dict, so anything
            # else would leave it waiting forever.
            if len(responses) != len(batch) or not all(isinstance(response, dict) for response in responses):
                raise RuntimeError(
                    f"provider returned {len(responses)} responses for {len(batch)} requests, "
                    "or a response that is not a dict"
                )
        except Exception as error:  # A batch-level failure fails each item; callers retry.
            responses = self._batch_error(batch, error)
        finally:
            self._slots.release()
        succeeded = sum(1 for response in responses if response.get("status") == "SUCCESS")
        tprint(
            f"{label} finished in {time.monotonic() - started:.1f}s: "
            f"{succeeded}/{len(batch)} succeeded.",
            level="INFO" if succeeded == len(batch) else "WARNING",
        )
        self._deliver(batch, batch_number, responses)

    @staticmethod
    def _deliver(batch: List[BatchItem], batch_number: int, responses: List[Dict[str, Any]]) -> None:
        for item, response in zip(batch, responses):
            meta = response.get("request_meta")
            if not isinstance(meta, dict):
                meta = response["request_meta"] = {}
            meta["api_batch"] = batch_number
            meta["api_batch_size"] = len(batch)
            item.response = response
            item.done.set()


__all__ = [
    "BATCHING_SOURCE_KEY", "APIBatchDispatcher", "BatchItem", "BatchRefusedError", "BatchSettings",
    "batch_refusals", "batching_enabled", "batching_setting", "fallback_to_direct",
    "validate_batching_setting",
]
