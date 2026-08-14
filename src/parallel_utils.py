# src/parallel_utils.py

"""Run independent API-call tasks with one bounded worker pool per step."""
import concurrent.futures
import contextvars
from typing import Any, Callable, Dict, List

# --- FIX: Import the custom thread-aware logger! ---
from src.context_logger import tprint

def run_parallel_api_calls(tasks: List[Callable[[], Any]], config: Dict[str, Any]) -> List[Any]:
    """Execute tasks sequentially or with one bounded, order-preserving pool."""
    if not tasks:
        return []

    if not config.get("QUESTION_PARALLEL_API_ENABLED", False):
        return [task() for task in tasks]

    max_workers = max(1, int(config.get("QUESTION_PARALLEL_MAX_WORKERS", 3)))
    worker_count = min(max_workers, len(tasks))
    all_results = [None] * len(tasks)

    tprint(
        f"    [PARALLEL] Starting {len(tasks)} API calls "
        f"(Max Workers: {worker_count})",
        level="INFO",
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="api-call",
    ) as executor:
        # ThreadPoolExecutor does not propagate contextvars automatically. Give
        # every task its own copy so question and batch metadata remain correct.
        future_to_index = {
            executor.submit(contextvars.copy_context().run, task): task_index
            for task_index, task in enumerate(tasks)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            task_index = future_to_index[future]
            try:
                all_results[task_index] = future.result()
            except Exception as exc:
                # Do not start queued work after a task has already made the
                # current step unusable. Running tasks finish during shutdown.
                for pending in future_to_index:
                    pending.cancel()
                raise RuntimeError(
                    f"Parallel task #{task_index} crashed unexpectedly: {exc}"
                ) from exc

    tprint(
        f"    [PARALLEL] Completed all {len(tasks)} API calls.",
        level="INFO",
    )
    return all_results
