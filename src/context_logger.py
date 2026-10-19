# src/context_logger.py

import logging
import contextvars
import contextlib
import os
import sys


# 1. CONTEXT VARIABLES (The "Nametags" for threads)
# contextvars ensures that when Thread A sets its name to "Q#5", 
# it doesn't accidentally overwrite Thread B which is named "Q#6".
ctx_query_idx = contextvars.ContextVar('ctx_query_idx', default="N/A")
ctx_batch_id = contextvars.ContextVar('ctx_batch_id', default="N/A")


# 2. CONTEXT MANAGER (How threads put their nametags on)
@contextlib.contextmanager
def pipeline_context(query_idx=None, batch_id=None, batch=None, **kwargs):
    """
    Injects thread-specific identity into the execution context.
    
    Professionally updated to handle API mismatches: accepts either a direct 
    'batch_id' or a 'batch' dictionary/object from the orchestrator.
    """
    # --- 1. Defensive parsing of the batch identifier ---
    resolved_batch_id = batch_id
    
    if batch is not None:
        if isinstance(batch, dict):
            # If batch is a dictionary, extract 'id' or 'batch_id'
            resolved_batch_id = batch.get("id", batch.get("batch_id", "unknown_batch"))
        elif hasattr(batch, "id"):
            # If batch is an object (like a dataclass), get its .id attribute
            resolved_batch_id = getattr(batch, "id")
        else:
            # Fallback: just cast whatever was passed to a string
            resolved_batch_id = str(batch)

    # --- 2. Set the context variables ---
    token_q = ctx_query_idx.set(query_idx)
    token_b = ctx_batch_id.set(resolved_batch_id)
    
    try:
        # Yield passes control back to the thread to do its work
        yield
    finally:
        # When the thread finishes, it takes its nametag off
        ctx_query_idx.reset(token_q)
        ctx_batch_id.reset(token_b)


# 3. CUSTOM LOG FORMATTER (For standard logger.info calls)
# This intercepts normal log messages and automatically adds the prefix.
class ContextFormatter(logging.Formatter):
    def format(self, record):
        q_idx = ctx_query_idx.get()
        b_id = ctx_batch_id.get()
        
        prefix = ""
        if q_idx != "N/A":
            prefix = f"[Batch {b_id} | Q#{q_idx}] "
            
        # Add the prefix to the original log message
        record.msg = f"{prefix}{record.msg}"
        return super().format(record)

def setup_context_logger(log_dir: str):
    """Sets up the root logger to save everything cleanly to a file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "batch_execution.log")
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Remove old handlers to prevent duplicate printing
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # File handler saves everything (DEBUG level and up) to the file
    fh = logging.FileHandler(log_file_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    formatter = ContextFormatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    return logger


# 4. THE CUSTOM PRINT FUNCTION (tprint)
# You will use this instead of print() in the future.
def tprint(message: str, level: str = "INFO"):
    """
    Thread-aware print function.
    - INFO: Prints to console AND writes to log.
    - WARNING/ERROR: Prints to console (with emoji) AND writes to log.
    - DEBUG: Writes to log ONLY (keeps console clean!).
    """
    q_idx = ctx_query_idx.get()
    b_id = ctx_batch_id.get()
    
    prefix = ""
    if q_idx != "N/A":
        prefix = f"[Batch {b_id} | Q#{q_idx}]"
        
    # Get the master logger we set up earlier
    logger = logging.getLogger()
    
    if level == "INFO":
        print(f"{prefix} {message}")
        logger.info(message)
        
    elif level == "WARNING":
        print(f"⚠️  {prefix} WARNING: {message}")
        logger.warning(message)
        
    elif level == "ERROR":
        print(f"🔴 {prefix} ERROR: {message}")
        logger.error(message)
        
    elif level == "DEBUG":
        # NOTICE: No print() statement here! 
        # This keeps the console from turning into "spaghetti", 
        # but the information is still saved safely in the log file.
        logger.debug(message)
        
    else:
        print(f"{prefix} {message}")
        logger.info(message)


# 5. IPYKERNEL THREAD-OUTPUT GUARD
# Kaggle runs ipykernel 6.29.5. On every write, its stdout/stderr walk
# `_thread_to_parent` (thread ident -> ident of the thread that created it)
# to send a thread's output to its oldest ancestor's cell. The OS reuses an
# ident as soon as its thread exits, but ipykernel prunes the map only during
# garbage collection, so thousands of short-lived pool threads eventually
# leave two stale entries that point at each other. Every thread whose chain
# reaches them then loops inside print() forever at 100% CPU: Kaggle batch
# full-0009-970-989 hung that way for 6 hours on 2026-10-19.
class _IgnoredThreadMap(dict):
    """A thread map that stays empty: ipykernel's writes to it are dropped."""

    def __setitem__(self, key, value):
        pass


def guard_ipykernel_thread_output() -> int:
    """Empty ipykernel's thread maps so that no print from a thread can hang.

    ipykernel 7.1 removed both maps. With them empty, a thread's output goes
    to the running cell, as it does there. A thread already caught in the
    loop re-reads the map on every step, so it is released as well.

    Returns how many streams were changed: 0 outside ipykernel 6.29-7.0, and
    0 once the guard is in place.
    """
    ipython = sys.modules.get("IPython")
    shell = getattr(ipython, "get_ipython", lambda: None)()
    kernel = getattr(shell, "kernel", None)
    streams = [sys.stdout, sys.stderr, getattr(kernel, "_stdout", None), getattr(kernel, "_stderr", None)]

    changed = 0
    for stream in streams:
        names = [name for name in ("_thread_to_parent", "_thread_to_parent_header")
                 if type(getattr(stream, name, None)) is dict]
        for name in names:
            setattr(stream, name, _IgnoredThreadMap())
        changed += bool(names)
    return changed


if guard_ipykernel_thread_output():
    print("[context_logger] Emptied ipykernel's thread maps: thread output goes to the running cell.")
