# src/utils.py

"""
Utility module for the Analogical Reasoning RAG project.

This file provides common helper functions used across the project, including:
- A centralized logging setup to write logs to both console and file.
- Standardized functions for reading and writing JSON and Pickle files with error handling.
- A data conversion helper to make NumPy objects JSON serializable.
- NEW: Helper functions for creating standardized execution trace entries.
"""

import logging
import os
import json
import pickle
import numpy as np
from datetime import datetime

# --- 1. Logging Setup ---

def setup_logger(logger_name: str, log_dir: str, level=logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger that writes to both a file and the console.

    The log file is named with the logger_name and a timestamp. It prevents
    adding duplicate handlers if called multiple times.

    Args:
        logger_name (str): The name for the logger.
        log_dir (str): The directory where the log file will be saved.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(logger_name)

    # Avoid adding duplicate handlers if the logger is already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Define the format for the log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create a file handler to write logs to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"{logger_name}_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a stream handler to print logs to the console
    # Handle Windows console encoding issues with a custom filter
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    # Add a filter to handle encoding issues by replacing problematic Unicode characters
    class UnicodeFilter(logging.Filter):
        def filter(self, record):
            # Replace common Unicode symbols with ASCII equivalents
            record.msg = str(record.msg).replace('\u2717', '[X]').replace('\u2713', '[OK]').replace('\u2728', '*')
            return True
    
    stream_handler.addFilter(UnicodeFilter())
    logger.addHandler(stream_handler)

    logger.info(f"Logger '{logger_name}' initialized. Logging to {log_file_path}")
    return logger


# 2. Data Type Conversion Utilities 

def convert_numpy_for_json(obj):
    """
    Custom JSON encoder function to handle common NumPy data types.
    To be used as the `default` argument in json.dump() or json.dumps().
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int_, np.intc, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    # Note: np.intp is machine dependent, often aliased to int64 or int32.
    # We include a catch-all for other integer types if needed, but the above covers specific numpy types.
    # Handling float types
    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    # If the object has an `item` method (like many NumPy scalars), use it
    if hasattr(obj, 'item'):
        return obj.item()
    # For other unhandled types, raise a TypeError to let json.dump know
    raise TypeError(f"<Object of type {obj.__class__.__name__} is not JSON serializable>")


# 3. File I/O Utilities (JSON, Pickle)

def save_json(data: dict or list, file_path: str, indent: int = 4) -> bool:
    """
    Saves a dictionary or list to a JSON file with robust error handling.
    Automatically handles NumPy data types.

    Args:
        data (dict or list): The Python object to save.
        file_path (str): The full path to the output file.
        indent (int): Indentation level for pretty-printing the JSON.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        # Ensure the directory exists before trying to write the file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=convert_numpy_for_json)
        return True
    except (TypeError, IOError) as e:
        # Log the error if a logger is available, otherwise print
        try:
            logging.getLogger(__name__).error(f"Failed to save JSON to {file_path}: {e}", exc_info=True)
        except Exception:
            print(f"ERROR: Failed to save JSON to {file_path}: {e}")
        return False


def save_json_atomic(data: dict or list, file_path: str, indent: int = 4) -> bool:
    """Atomically replace a JSON file after successfully writing its full content."""
    temp_path = file_path + ".tmp"
    try:
        if not save_json(data, temp_path, indent=indent):
            return False
        os.replace(temp_path, file_path)
        return True
    except OSError as e:
        logging.getLogger(__name__).error(f"Failed atomic JSON save to {file_path}: {e}", exc_info=True)
        return False
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

def load_json(file_path: str) -> dict or list or None:
    """
    Loads data from a JSON file with robust error handling.

    Args:
        file_path (str): The full path to the JSON file.

    Returns:
        The loaded data as a dict or list, or None if an error occurs.
    """
    if not os.path.exists(file_path):
        return None # Return None if file doesn't exist to allow for resume logic
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        try:
            logging.getLogger(__name__).error(f"Failed to load JSON from {file_path}: {e}", exc_info=True)
        except Exception:
            print(f"ERROR: Failed to load JSON from {file_path}: {e}")
        return None

def save_to_pickle(data, file_path: str) -> bool:
    """
    Saves a Python object to a Pickle file with error handling.

    Args:
        data: The Python object to save.
        file_path (str): The full path to the output file.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except (pickle.PicklingError, IOError) as e:
        try:
            logging.getLogger(__name__).error(f"Failed to save Pickle to {file_path}: {e}", exc_info=True)
        except Exception:
            print(f"ERROR: Failed to save Pickle to {file_path}: {e}")
        return False

def load_from_pickle(file_path: str):
    """
    Loads an object from a Pickle file with error handling.

    Args:
        file_path (str): The full path to the Pickle file.

    Returns:
        The loaded Python object, or None if an error occurs.
    """
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, IOError, EOFError) as e:
        try:
            logging.getLogger(__name__).error(f"Failed to load Pickle from {file_path}: {e}", exc_info=True)
        except Exception:
            print(f"ERROR: Failed to load Pickle from {file_path}: {e}")
        return None


def move_files_to_directories(files_to_move_map: dict) -> dict:
    """
    Moves files from their source paths to specified destination directories.

    Args:
        files_to_move_map (dict): A dictionary where:
            - Keys are strings representing destination directory paths.
            - Values are lists of strings, where each string is the
              full path to a source file that should be moved to the
              corresponding destination directory.

    Returns:
        dict: A dictionary containing two lists:
            - 'successful_moves': A list of tuples, each representing a
                                  successful move (source_path, destination_path).
            - 'failed_moves': A list of tuples, each representing a
                              failed move (source_path, destination_path_attempted, error_message).
    """
    successful_moves = []
    failed_moves = []

    for dest_dir, source_files_list in files_to_move_map.items():
        # 1. Validate and create the destination directory
        if not os.path.exists(dest_dir):
            try:
                os.makedirs(dest_dir, exist_ok=True) # exist_ok=True prevents error if dir already exists
                print(f"Created destination directory: '{dest_dir}'")
            except OSError as e:
                print(f"Error: Could not create directory '{dest_dir}'. Skipping files for this destination. Reason: {e}")
                for source_file in source_files_list:
                    failed_moves.append((source_file, os.path.join(dest_dir, os.path.basename(source_file)), f"Destination directory creation failed: {e}"))
                continue # Skip to the next destination directory

        elif not os.path.isdir(dest_dir):
            print(f"Error: Destination path '{dest_dir}' exists but is not a directory. Skipping files for this destination.")
            for source_file in source_files_list:
                failed_moves.append((source_file, os.path.join(dest_dir, os.path.basename(source_file)), f"Destination '{dest_dir}' is not a directory"))
            continue # Skip to the next destination directory

        # 2. Iterate through source files for the current destination
        for source_file_path in source_files_list:
            if not os.path.exists(source_file_path):
                print(f"Warning: Source file '{source_file_path}' not found. Skipping.")
                failed_moves.append((source_file_path, os.path.join(dest_dir, os.path.basename(source_file_path)), "Source file not found"))
                continue

            if not os.path.isfile(source_file_path):
                print(f"Warning: '{source_file_path}' is not a file. Skipping.")
                failed_moves.append((source_file_path, os.path.join(dest_dir, os.path.basename(source_file_path)), "Source is not a file"))
                continue

            # Construct the full destination path for the file
            # os.path.basename gets just the file name from the source path
            destination_file_path = os.path.join(dest_dir, os.path.basename(source_file_path))

            if source_file_path == destination_file_path:
                print(f"Info: Source and destination paths are the same for '{source_file_path}'. No action needed.")
                successful_moves.append((source_file_path, destination_file_path))
                continue

            try:
                # Using os.rename for moving
                # Note: os.rename might fail if moving across different file systems.
                #       shutil.move is generally more robust for such cases.
                os.rename(source_file_path, destination_file_path)
                print(f"Successfully moved: '{source_file_path}' -> '{destination_file_path}'")
                successful_moves.append((source_file_path, destination_file_path))
            except FileNotFoundError:
                print(f"Error: Source file '{source_file_path}' not found during move attempt.")
                failed_moves.append((source_file_path, destination_file_path, "Source file not found during move"))
            except PermissionError as e:
                print(f"Error: Permission denied for moving '{source_file_path}' to '{destination_file_path}'. Reason: {e}")
                failed_moves.append((source_file_path, destination_file_path, f"Permission denied: {e}"))
            except OSError as e:
                # This could catch errors like "Cannot overwrite existing directory"
                # or cross-device link errors
                print(f"Error: Failed to move '{source_file_path}' to '{destination_file_path}'. Reason: {e}")
                failed_moves.append((source_file_path, destination_file_path, f"OS Error during move: {e}"))
            except Exception as e:
                print(f"An unexpected error occurred while moving '{source_file_path}': {e}")
                failed_moves.append((source_file_path, destination_file_path, f"Unexpected error: {e}"))

    return {
        'successful_moves': successful_moves,
        'failed_moves': failed_moves
    }


#  4. Logging & Trace Utilities

def create_trace_entry(
    step_name: str,
    sub_step: str,
    input_context: dict,
    output_result: dict,
    api_call_meta: dict = None,
    error_info: dict = None
) -> dict:
    """
    Creates a standardized dictionary for the high-resolution execution log.

    This function helps maintain a consistent schema for capturing granular
    pipeline events (like individual API calls) without losing the broader
    pipeline structure.

    Args:
        step_name (str): The high-level phase (e.g., 'adapt', 'solve').
        sub_step (str): The specific action (e.g., 'normalization', 'attempt_1').
        input_context (dict): Data sent to the operation (e.g., prompts, source text).
        output_result (dict): Data received (e.g., raw LLM response).
        api_call_meta (dict, optional): Metadata like model name, temperature.
        error_info (dict, optional): If an error occurred, the error details.

    Returns:
        dict: A dictionary ready to be appended to the execution trace list.
    """
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "step_name": step_name,
        "sub_step": sub_step,
        "input_context": input_context,
        "api_call_meta": api_call_meta or {},
        "output_result": output_result,
        "error_info": error_info
    }


# --- 7. Local Resource Loading Utilities ---

def load_embedding_model(config: dict, logger=None):
    """
    Load a SentenceTransformer embedding model from either local disk or HuggingFace.
    
    Intelligently decides whether to load from a local path or download from HuggingFace
    based on the config settings and path existence.
    
    Args:
        config (dict): Configuration dictionary containing:
            - USE_LOCAL_MODEL (bool): Whether to prefer local loading
            - LOCAL_EMBEDDING_MODEL_PATH (str): Path to local model directory
            - EMBEDDING_MODEL_PATH (str): HuggingFace repo name or local path
            - EMBEDDING_MODEL_PATH_HF (str): HuggingFace repo name (backup)
        logger (logging.Logger, optional): Logger instance for logging messages.
    
    Returns:
        SentenceTransformer or None: The loaded model, or None if loading fails.
    """
    from sentence_transformers import SentenceTransformer
    
    try:
        use_local = config.get('USE_LOCAL_MODEL', False)
        local_path = config.get('LOCAL_EMBEDDING_MODEL_PATH')
        current_path = config.get('EMBEDDING_MODEL_PATH')
        hf_path = config.get('EMBEDDING_MODEL_PATH_HF')
        
        # Determine the actual path to use
        if use_local and local_path and os.path.exists(local_path):
            load_path = local_path
            source = "local disk"
        elif os.path.isdir(current_path):
            load_path = current_path
            source = "local disk"
        else:
            load_path = hf_path or current_path
            source = "HuggingFace"
        
        if logger:
            logger.info(f"Loading embedding model from {source}: {load_path}")
        
        model = SentenceTransformer(load_path)
        
        if logger:
            logger.info(f"[OK] Successfully loaded embedding model from {source}")
        
        return model
        
    except Exception as e:
        if logger:
            logger.critical(f"Failed to load embedding model. Error: {e}", exc_info=True)
        else:
            print(f"ERROR: Failed to load embedding model. Error: {e}")
        return None


def load_exemplar_corpus(config: dict, logger=None):
    """
    Load the exemplar corpus dataset from either local disk or HuggingFace.
    
    Intelligently decides whether to load from a local path or download from HuggingFace
    based on the config settings and path existence.
    
    Args:
        config (dict): Configuration dictionary containing:
            - USE_LOCAL_MODEL (bool): Whether to prefer local loading
            - LOCAL_EXEMPLAR_CORPUS_PATH (str): Path to local dataset directory
            - EXEMPLAR_CORPUS_NAME (str): HuggingFace repo name or local path
            - EXEMPLAR_CORPUS_NAME_HF (str): HuggingFace repo name (backup)
        logger (logging.Logger, optional): Logger instance for logging messages.
    
    Returns:
        datasets.Dataset or None: The loaded dataset, or None if loading fails.
    """
    from datasets import load_dataset, load_from_disk
    
    try:
        use_local = config.get('USE_LOCAL_MODEL', False)
        local_path = config.get('LOCAL_EXEMPLAR_CORPUS_PATH')
        current_path = config.get('EXEMPLAR_CORPUS_NAME')
        hf_path = config.get('EXEMPLAR_CORPUS_NAME_HF')
        
        # Determine the actual path to use
        if use_local and local_path and os.path.isdir(local_path):
            load_path = local_path
            source = "local disk"
            use_disk = True
        elif os.path.isdir(current_path):
            load_path = current_path
            source = "local disk"
            use_disk = True
        else:
            load_path = hf_path or current_path
            source = "HuggingFace"
            use_disk = False
        
        if logger:
            logger.info(f"Loading exemplar corpus from {source}: {load_path}")
        
        if use_disk:
            dataset = load_from_disk(load_path)
        else:
            dataset = load_dataset(load_path, split='train')
        
        if logger:
            logger.info(f"[OK] Successfully loaded exemplar corpus from {source}")
        
        return dataset
        
    except Exception as e:
        if logger:
            logger.critical(f"Failed to load exemplar corpus. Error: {e}", exc_info=True)
        else:
            print(f"ERROR: Failed to load exemplar corpus. Error: {e}")
        return None


# --- 8. NuminaMath Hard-Question Set Construction ---

NUMINA_MATH_SOURCES = (
    "aops_forum",
    "amc_aime",
    "cn_k12",
    "gsm8k",
    "math",
    "olympiads",
    "orca_math",
    "synthetic_amc",
    "synthetic_math",
)


def _allocate_percentage_counts(total, percentages):
    """Convert percentages to integer counts with the largest-remainder rule."""
    exact = {name: total * percentage / 100.0 for name, percentage in percentages.items()}
    counts = {name: int(value) for name, value in exact.items()}
    remaining = total - sum(counts.values())
    order = sorted(percentages, key=lambda name: (-(exact[name] - counts[name]), name))
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def _select_farthest_indices(indices, questions, count, embedding_model, rng, batch_size):
    """Greedy max-min selection using cosine distance from the selected set."""
    if count == len(indices):
        return list(indices)

    embeddings = np.asarray(
        embedding_model.encode(
            questions,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(indices):
        raise ValueError("The embedding model returned an unexpected embedding shape.")

    # Normalize here too, because custom embedding models may ignore the encode flag.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    first = int(rng.integers(len(indices)))
    selected_positions = [first]
    available = np.ones(len(indices), dtype=bool)
    available[first] = False
    nearest_similarity = embeddings @ embeddings[first]

    while len(selected_positions) < count:
        candidates = np.flatnonzero(available)
        candidate_scores = nearest_similarity[candidates]
        lowest_score = candidate_scores.min()
        ties = candidates[np.isclose(candidate_scores, lowest_score, rtol=1e-6, atol=1e-7)]
        next_position = int(rng.choice(ties))
        selected_positions.append(next_position)
        available[next_position] = False
        nearest_similarity = np.maximum(
            nearest_similarity,
            embeddings @ embeddings[next_position],
        )

    return [int(indices[position]) for position in selected_positions]


def create_numina_hard_question_set(
    dataset,
    total_questions,
    source_percentages=None,
    selection_strategy="random",
    embedding_model=None,
    output_dir=None,
    question_column="problem",
    source_column="source",
    random_seed=42,
    embedding_batch_size=64,
    candidate_pool_size=None,
    shuffle_output=True,
):
    """Select NuminaMath rows and save pipeline-compatible hard-question indices.

    ``source_percentages`` reserves the stated percentage for each named source.
    Any percentage left up to 100 is sampled from all unlisted sources as one
    integrated ``other`` pool. With an empty mapping, the full dataset is one pool.

    ``selection_strategy`` is either ``random`` or ``farthest``. The latter starts
    randomly inside every source pool and then greedily maximizes the minimum
    cosine distance from rows already chosen in that pool.

    The saved JSON is deliberately only ``list[int]``, matching the format read by
    ``HARD_QUESTIONS_INDICES_PATH`` in the main pipeline. A report containing the
    path, allocation, and selected indices is returned separately.
    """
    if isinstance(total_questions, bool) or not isinstance(total_questions, (int, np.integer)):
        raise TypeError("total_questions must be an integer.")
    total_questions = int(total_questions)
    if total_questions <= 0:
        raise ValueError("total_questions must be greater than zero.")
    if total_questions > len(dataset):
        raise ValueError(f"Requested {total_questions} rows from a dataset of {len(dataset)} rows.")

    strategy = str(selection_strategy).strip().lower()
    if strategy not in {"random", "farthest"}:
        raise ValueError("selection_strategy must be 'random' or 'farthest'.")
    if strategy == "farthest" and embedding_model is None:
        raise ValueError("embedding_model is required for selection_strategy='farthest'.")
    if not output_dir:
        raise ValueError("output_dir must be provided.")
    if candidate_pool_size is not None and (
        isinstance(candidate_pool_size, bool) or not isinstance(candidate_pool_size, (int, np.integer))
        or candidate_pool_size <= 0
    ):
        raise ValueError("candidate_pool_size must be a positive integer or None.")

    column_names = set(getattr(dataset, "column_names", []))
    if column_names and source_column not in column_names:
        raise KeyError(f"Dataset has no source column named '{source_column}'.")
    if strategy == "farthest" and column_names and question_column not in column_names:
        raise KeyError(f"Dataset has no question column named '{question_column}'.")

    raw_percentages = source_percentages or {}
    if not isinstance(raw_percentages, dict):
        raise TypeError("source_percentages must be a dictionary such as {'amc_aime': 10}.")
    percentages = {}
    for raw_name, raw_percentage in raw_percentages.items():
        name = str(raw_name).strip().lower()
        if name == "other":
            raise ValueError("Do not specify 'other'; it is automatically assigned the remainder.")
        if name not in NUMINA_MATH_SOURCES:
            raise ValueError(f"Unknown NuminaMath source '{name}'. Expected one of {NUMINA_MATH_SOURCES}.")
        if isinstance(raw_percentage, bool) or not isinstance(raw_percentage, (int, float, np.number)):
            raise TypeError(f"Percentage for '{name}' must be numeric.")
        percentage = float(raw_percentage)
        if not np.isfinite(percentage) or percentage < 0:
            raise ValueError(f"Percentage for '{name}' must be finite and non-negative.")
        percentages[name] = percentage

    specified_total = sum(percentages.values())
    if specified_total > 100.0 + 1e-9:
        raise ValueError(f"Source percentages sum to {specified_total:g}; they cannot exceed 100.")
    if percentages and specified_total < 100.0 - 1e-9:
        percentages["other"] = 100.0 - specified_total
    elif percentages:
        # Avoid floating-point residue while preserving an exact 100% allocation.
        percentages[next(iter(percentages))] += 100.0 - specified_total
    else:
        percentages = {"all": 100.0}

    source_values = [str(value).strip().lower() for value in dataset[source_column]]
    specified_sources = set(percentages) - {"other", "all"}
    pools = {}
    for block_name in percentages:
        if block_name == "all":
            pools[block_name] = list(range(len(dataset)))
        elif block_name == "other":
            pools[block_name] = [i for i, source in enumerate(source_values) if source not in specified_sources]
        else:
            pools[block_name] = [i for i, source in enumerate(source_values) if source == block_name]

    counts = _allocate_percentage_counts(total_questions, percentages)
    for block_name, count in counts.items():
        if count > len(pools[block_name]):
            raise ValueError(
                f"Block '{block_name}' needs {count} rows but only {len(pools[block_name])} are available."
            )

    rng = np.random.default_rng(random_seed)
    selected = []
    for block_name, count in counts.items():
        if count == 0:
            continue
        pool = pools[block_name]
        if candidate_pool_size is not None and len(pool) > max(count, candidate_pool_size):
            pool = rng.choice(pool, size=max(count, candidate_pool_size), replace=False).tolist()

        if strategy == "random":
            chosen = rng.choice(pool, size=count, replace=False).tolist()
        else:
            questions = [str(dataset[int(index)][question_column]) for index in pool]
            chosen = _select_farthest_indices(
                pool, questions, count, embedding_model, rng, embedding_batch_size
            )
        selected.extend(int(index) for index in chosen)

    if shuffle_output:
        rng.shuffle(selected)
    if len(selected) != total_questions or len(set(selected)) != total_questions:
        raise RuntimeError("Selection did not produce the requested number of unique indices.")

    source_slug = "-".join(
        f"{name}{percentage:g}" for name, percentage in percentages.items()
    ).replace(".", "p")
    seed_slug = "none" if random_seed is None else str(random_seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"hard_questions_n{total_questions}_{strategy}_src-{source_slug}"
        f"_seed{seed_slug}_{timestamp}.json"
    )
    output_path = os.path.abspath(os.path.join(os.fspath(output_dir), filename))
    if not save_json_atomic(selected, output_path, indent=2):
        raise OSError(f"Could not save hard-question indices to '{output_path}'.")

    selected_source_counts = {}
    for index in selected:
        source = source_values[index]
        selected_source_counts[source] = selected_source_counts.get(source, 0) + 1

    return {
        "output_path": output_path,
        "selected_indices": selected,
        "selection_strategy": strategy,
        "requested_percentages": percentages,
        "allocated_counts": counts,
        "available_counts": {name: len(pool) for name, pool in pools.items()},
        "selected_source_counts": dict(sorted(selected_source_counts.items())),
        "candidate_pool_size": candidate_pool_size,
        "random_seed": random_seed,
    }


def verify_local_resources(config: dict, logger=None) -> dict:
    """
    Verify that all required local resources exist and are accessible.
    
    Args:
        config (dict): Configuration dictionary with resource paths.
        logger (logging.Logger, optional): Logger instance for logging messages.
    
    Returns:
        dict: Verification report with keys like 'embedding_model', 'corpus', 'embeddings',
              'hard_questions_indices', each with 'exists' and 'path' fields.
    """
    resources = {
        'embedding_model': {
            'path': config.get('LOCAL_EMBEDDING_MODEL_PATH'),
            'exists': False
        },
        'corpus': {
            'path': config.get('LOCAL_EXEMPLAR_CORPUS_PATH'),
            'exists': False
        },
        'embeddings': {
            'path': config.get('EMBEDDED_EXEMPLAR_CORPUS_QUESTIONS_PATH'),
            'exists': False
        },
        'hard_questions_indices': {
            'path': config.get('HARD_QUESTIONS_INDICES_PATH'),
            'exists': False
        }
    }
    
    for resource_name, resource_info in resources.items():
        path = resource_info['path']
        if path and os.path.exists(path):
            resource_info['exists'] = True
            if logger:
                logger.info(f"[OK] {resource_name}: Available at {path}")
        else:
            if logger:
                logger.warning(f"[WARNING] {resource_name}: Not found at {path}")
    
    return resources
