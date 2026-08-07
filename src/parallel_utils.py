# src/parallel_utils.py

"""
Utility for running independent API calls in parallel batches within a single question's pipeline.
This respects the existing RPM limits and retry logic by simply wrapping the calls.
"""
import concurrent.futures
import contextvars
from typing import Any, Callable, Dict, List

# --- FIX: Import the custom thread-aware logger! ---
from src.context_logger import tprint

def run_parallel_api_calls(tasks: List[Callable[[], Any]], config: Dict[str, Any]) -> List[Any]:
    """
    Executes a list of API call tasks either sequentially or in parallel batches.
    """
    if not tasks:
        return []

    # 1. If the feature is disabled, run sequentially to preserve standard behavior
    if not config.get("QUESTION_PARALLEL_API_ENABLED", False):
        return [task() for task in tasks]

    # 2. Feature is enabled: run in batches
    max_workers = max(1, int(config.get("QUESTION_PARALLEL_MAX_WORKERS", 3)))
    all_results = [None] * len(tasks)
    
    total_batches = (len(tasks) + max_workers - 1) // max_workers
    
    # FIX: Use tprint so it attaches the [Q#] context tag and saves to the log file!
    tprint(f"    [PARALLEL] Starting {len(tasks)} API calls in {total_batches} batch(es) (Max Workers: {max_workers})", level="INFO")
    
    # Split tasks into chunks based on max_workers
    for batch_num, i in enumerate(range(0, len(tasks), max_workers), start=1):
        batch_tasks = tasks[i:i + max_workers]
        
        # Use ThreadPoolExecutor for the current batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_tasks)) as executor:
            
            # Submit tasks and map each future back to its original index
            # ThreadPoolExecutor does not propagate contextvars automatically.
            # Copy the current question/batch context into every API-call worker
            # so nested parallelism keeps accurate logs and API metadata.
            future_to_index = {
                executor.submit(contextvars.copy_context().run, task): batch_idx
                for batch_idx, task in enumerate(batch_tasks)
            }
            
            # Wait for ALL tasks in this batch to complete.
            for future in concurrent.futures.as_completed(future_to_index):
                original_batch_idx = future_to_index[future]
                original_global_idx = i + original_batch_idx
                
                try:
                    result = future.result()
                    all_results[original_global_idx] = result
                except Exception as exc:
                    # (This is the crash fix we did in the previous step)
                    raise RuntimeError(f"A parallel task crashed unexpectedly: {exc}") from exc
                    
        # FIX: Use tprint here too!
        tprint(f"    [PARALLEL] Batch {batch_num}/{total_batches} completed. All {len(batch_tasks)} calls returned.", level="INFO")
        
    return all_results
