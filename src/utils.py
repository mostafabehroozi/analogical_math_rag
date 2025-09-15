# src/utils.py

"""
Utility module for the Analogical Reasoning RAG project.

This file provides common helper functions used across the project, including:
- A centralized logging setup to write logs to both console and file.
- Standardized functions for reading and writing JSON and Pickle files with error handling.
- A data conversion helper to make NumPy objects JSON serializable.
"""

import logging
import os
import json
import pickle
import numpy as np
from datetime import datetime
import shutil

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
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info(f"Logger '{logger_name}' initialized. Logging to {log_file_path}")
    return logger


# --- 2. Data Type Conversion Utilities ---

def convert_numpy_for_json(obj):
    """
    Custom JSON encoder function to handle common NumPy data types.
    To be used as the `default` argument in json.dump() or json.dumps().
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    # If the object has an `item` method (like many NumPy scalars), use it
    if hasattr(obj, 'item'):
        return obj.item()
    # For other unhandled types, raise a TypeError to let json.dump know
    raise TypeError(f"<Object of type {obj.__class__.__name__} is not JSON serializable>")


# --- 3. File I/O Utilities (JSON, Pickle) ---

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




# While the request was primarily 'os', shutil.move is generally safer
# for moving files across different file systems.
# I will primarily use os.rename as requested, but add a note about shutil.

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


import time
import google.generativeai as genai

def check_api_keys(config: dict):
    """
    Checks the validity of all API keys in the config.

    Args:
        config (dict): The main configuration dictionary.
    """
    print("--- Checking API Keys ---")
    api_keys = config.get("GEMINI_API_KEYS", [])
    if not api_keys:
        print("No API keys found in the configuration.")
        return

    for i, key in enumerate(api_keys):
        key_for_log = f"...{key[-4:]}"
        print(f"Checking key #{i+1} ({key_for_log})...")
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content("hello")
            print(f"  -> SUCCESS: {response.text.strip()}")
        except Exception as e:
            print(f"  -> FAILURE: {e}")

        # Respect the global delay between calls
        if i < len(api_keys) - 1:
            delay = config.get("GLOBAL_API_CALL_DELAY_SECONDS", 0)
            print(f"  ...sleeping for {delay}s...")
            time.sleep(delay)
    print("--- API Key Check Complete ---\n")


def create_run_suffix_from_config(id_config: dict) -> str:
    """
    Creates a descriptive filename suffix from the hard question identification config.
    """
    num_samples = id_config.get("NUM_RANDOM_SAMPLES", "N/A")
    max_attempts = id_config.get("MAX_ATTEMPTS_PER_QUESTION", "N/A")
    
    # Check if a specific index file is used, which overrides random sampling
    if id_config.get("TARGET_INDICES_FILE_PATH"):
        # Extract a name from the file path to keep it concise
        indices_filename = os.path.basename(id_config["TARGET_INDICES_FILE_PATH"])
        indices_name = os.path.splitext(indices_filename)[0]
        sample_part = f"indices-{indices_name}"
    else:
        sample_part = f"samples-{num_samples}"
        
    attempt_part = f"attempts-{max_attempts}"
    
    return f"{sample_part}_{attempt_part}"