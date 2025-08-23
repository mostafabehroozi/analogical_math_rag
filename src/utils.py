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
