# src/wandb_sync.py

"""
Weights & Biases (W&B) Synchronization Module.

This file provides functions to manage the persistence of experiment results
and logs by synchronizing the local workspace with W&B Artifacts.

It runs in parallel with, but completely independently from, the Hugging Face sync.
Crucially, it uses a manual directory traversal (`os.walk`) to explicitly skip
heavy directories (like `embeddings/`) and files (like `.npy`), as W&B's native
`add_dir()` does not respect `.gitignore` files.

Functions:
- initialize_wandb_workspace: Downloads the latest W&B artifact on startup.
- sync_workspace_to_wandb: Uploads filtered local output directories to W&B.
- periodic_wandb_sync_check: A helper to trigger synchronization during long loops.
"""

import os
import logging

try:
    import wandb
except ImportError:
    wandb = None


def _get_artifact_name(config: dict) -> str:
    """
    Helper function to define a consistent Artifact name.
    We use a static name representing the entire workspace, or fall back to
    a cleaned version of the project name.
    """
    base_name = config.get("WANDB_PROJECT_NAME", "analogical-math-rag")
    # Sanitize name for W&B (only alphanumeric, dashes, underscores, dots allowed)
    clean_name = "".join(c if c.isalnum() or c in "-_." else "-" for c in base_name)
    return f"{clean_name}-workspace-outputs"


def initialize_wandb_workspace(config: dict):
    """
    Downloads the latest workspace files from W&B Artifacts to the local output directory.
    
    This populates the workspace with results from previous runs. It assumes
    `wandb.init()` has already been called in the notebook lifecycle.
    """
    logger = logging.getLogger(__name__)
    
    if not config.get("WANDB_PERSIST_ONLINE", False):
        logger.info("W&B online persistence is disabled. Skipping workspace initialization.")
        return

    if wandb is None:
        logger.error("W&B library is not installed. Cannot initialize W&B workspace.")
        return

    if wandb.run is None:
        logger.warning("No active W&B run found. Did you forget to call `wandb.init()`? Skipping download.")
        return

    local_outputs_dir = config["OUTPUTS_DIR"]
    artifact_name = _get_artifact_name(config)
    artifact_address = f"{artifact_name}:latest"

    logger.info(f"Initializing workspace from W&B Artifact: {artifact_address}")

    try:
        # Fetch the latest artifact version from the current project
        artifact = wandb.run.use_artifact(artifact_address)
        
        # Download its contents directly into our local outputs directory
        artifact.download(root=local_outputs_dir)
        logger.info(f"W&B Workspace synchronized. Files downloaded to {local_outputs_dir}.")

    except Exception as e:
        # This will naturally trigger on the very first run of a project when
        # 'latest' does not exist yet. We log it as info, not an error.
        logger.info(f"Could not download W&B artifact '{artifact_address}'. "
                    f"If this is the first run, this is expected. Details: {e}")


def sync_workspace_to_wandb(config: dict):
    """
    Uploads the local outputs directory to W&B Artifacts, strictly filtering out
    the `embeddings` folder and any `.npy` files.
    """
    logger = logging.getLogger(__name__)
    
    if not config.get("WANDB_PERSIST_ONLINE", False):
        return  # Silently exit if persistence is disabled

    if wandb is None:
        logger.error("W&B library is not installed. Cannot sync to W&B.")
        return

    if wandb.run is None:
        logger.warning("No active W&B run found. Cannot log artifact.")
        return

    local_outputs_dir = config["OUTPUTS_DIR"]
    artifact_name = _get_artifact_name(config)

    logger.info(f"Starting W&B sync of '{local_outputs_dir}' to Artifact '{artifact_name}'")

    try:
        # 1. Create a new Artifact object
        artifact = wandb.Artifact(name=artifact_name, type="workspace-results")

        # 2. Manually walk the directory to strictly enforce exclusion rules
        file_count = 0
        for root, dirs, files in os.walk(local_outputs_dir):
            
            # --- FOLDER EXCLUSION LOGIC ---
            # If 'embeddings' is in the list of subdirectories, remove it in-place.
            # This prevents os.walk from ever descending into that folder.
            if "embeddings" in dirs:
                dirs.remove("embeddings")
                
            # --- FILE EXCLUSION LOGIC ---
            for filename in files:
                if filename.endswith(".npy"):
                    continue  # Skip numpy arrays
                
                local_file_path = os.path.join(root, filename)
                
                # Calculate the relative path for W&B so folder structure is preserved.
                # W&B uses forward slashes (/) for its virtual file paths, so we
                # replace OS-specific separators (like \ on Windows) just in case.
                relative_path = os.path.relpath(local_file_path, local_outputs_dir)
                artifact_path = relative_path.replace(os.sep, '/')
                
                # 3. Add the file to the artifact payload
                artifact.add_file(local_file_path, name=artifact_path)
                file_count += 1

        # 4. Log the artifact to the current run to begin the upload
        wandb.run.log_artifact(artifact)
        logger.info(f"Successfully initiated W&B sync for {file_count} files to Artifact '{artifact_name}'.")

    except Exception as e:
        logger.error(f"Failed to sync workspace to Weights & Biases: {e}", exc_info=True)


def periodic_wandb_sync_check(loop_counter: int, config: dict):
    """
    Checks if a W&B sync is needed based on the counter and sync interval.
    """
    if not config.get("WANDB_PERSIST_ONLINE", False):
        return

    # Default to 10 if not set in config
    sync_interval = config.get("WANDB_SYNC_INTERVAL", 10)

    # Sync when the counter reaches the interval (e.g., at item 9 for 10)
    if (loop_counter + 1) % sync_interval == 0:
        print(f"\n--- Reached W&B sync interval at item #{loop_counter + 1}. Syncing results to Weights & Biases Artifacts. ---")
        sync_workspace_to_wandb(config)
        print("--- W&B Sync complete. ---\n")


# ============================================================================
# LOGGING AND METRICS FUNCTIONS
# ============================================================================

def log_experiment_metrics(config: dict, experiment_name: str, metrics: dict):
    """
    Logs experiment-level metrics to W&B.
    
    Args:
        config: Global configuration dictionary
        experiment_name: Name of the experiment
        metrics: Dictionary of metrics to log (e.g., pass@1, pass@3, accuracy)
    
    Example:
        log_experiment_metrics(CONFIG, "RAG_Only_Pass@3", {
            "pass@1": 0.45,
            "pass@3": 0.65,
            "accuracy": 0.65
        })
    """
    logger = logging.getLogger(__name__)
    
    if not config.get("WANDB_PERSIST_ONLINE", False):
        return
    
    if wandb is None or wandb.run is None:
        logger.warning("W&B is not initialized. Skipping metric logging.")
        return
    
    try:
        # Log metrics with experiment name as prefix
        prefixed_metrics = {
            f"{experiment_name}/{key}": value
            for key, value in metrics.items()
        }
        wandb.log(prefixed_metrics)
        logger.debug(f"Logged metrics for experiment '{experiment_name}': {prefixed_metrics}")
    except Exception as e:
        logger.warning(f"Failed to log metrics to W&B for experiment '{experiment_name}': {e}")


def log_sample_results(config: dict, experiment_name: str, query_idx: int, results: dict):
    """
    Logs detailed results for a single query/sample to W&B.
    
    Args:
        config: Global configuration dictionary
        experiment_name: Name of the experiment
        query_idx: Index of the query
        results: Dictionary with 'question', 'generated_answer', 'ground_truth', 'is_correct'
    """
    logger = logging.getLogger(__name__)
    
    if not config.get("WANDB_PERSIST_ONLINE", False):
        return
    
    if wandb is None or wandb.run is None:
        logger.warning("W&B is not initialized. Skipping sample result logging.")
        return
    
    try:
        log_data = {
            "experiment": experiment_name,
            "query_idx": query_idx,
            "is_correct": results.get("is_correct", False),
            "question": results.get("question", "")[:500],  # Truncate for readability
            "generated_answer": results.get("generated_answer", "")[:500],
            "ground_truth": results.get("ground_truth", "")[:500],
        }
        wandb.log(log_data)
        logger.debug(f"Logged sample result for query #{query_idx} in experiment '{experiment_name}'")
    except Exception as e:
        logger.warning(f"Failed to log sample result to W&B: {e}")


def log_error_batch(config: dict, errors_list: list):
    """
    Logs a batch of errors to W&B for tracking and analysis.
    
    Args:
        config: Global configuration dictionary
        errors_list: List of error dictionaries with 'query_idx', 'error_type', 'error_message'
    """
    logger = logging.getLogger(__name__)
    
    if not config.get("WANDB_PERSIST_ONLINE", False):
        return
    
    if wandb is None or wandb.run is None:
        logger.warning("W&B is not initialized. Skipping error logging.")
        return
    
    try:
        if not errors_list:
            return
        
        # Create a W&B Table for errors
        columns = ["query_idx", "error_type", "error_message", "timestamp"]
        error_table = wandb.Table(columns=columns)
        
        for error in errors_list:
            error_table.add_data(
                error.get("query_idx", "unknown"),
                error.get("error_type", "unknown"),
                error.get("error_message", "")[:500],  # Truncate long messages
                error.get("timestamp", "")
            )
        
        wandb.log({"error_batch": error_table})
        logger.info(f"Logged {len(errors_list)} errors to W&B")
    except Exception as e:
        logger.warning(f"Failed to log error batch to W&B: {e}")


def log_layer1_cache_stats(config: dict, cache_stats: dict):
    """
    Logs Layer 1 cache statistics to W&B.
    
    Args:
        config: Global configuration dictionary
        cache_stats: Dictionary with 'hit_count', 'miss_count', 'total_queries', 'cache_size_mb'
    """
    logger = logging.getLogger(__name__)
    
    if not config.get("WANDB_PERSIST_ONLINE", False):
        return
    
    if wandb is None or wandb.run is None:
        logger.warning("W&B is not initialized. Skipping cache stats logging.")
        return
    
    try:
        prefixed_stats = {f"layer1_cache/{key}": value for key, value in cache_stats.items()}
        wandb.log(prefixed_stats)
        logger.debug(f"Logged Layer 1 cache statistics: {prefixed_stats}")
    except Exception as e:
        logger.warning(f"Failed to log cache stats to W&B: {e}")


def log_checkpoint(config: dict, checkpoint_name: str, checkpoint_data: dict):
    """
    Saves checkpoint data and logs metadata to W&B.
    
    Args:
        config: Global configuration dictionary
        checkpoint_name: Name/identifier for the checkpoint
        checkpoint_data: Dictionary with checkpoint information
    """
    logger = logging.getLogger(__name__)
    
    if not config.get("WANDB_PERSIST_ONLINE", False):
        return
    
    if wandb is None or wandb.run is None:
        logger.warning("W&B is not initialized. Skipping checkpoint logging.")
        return
    
    try:
        wandb.log({
            "checkpoint": checkpoint_name,
            "checkpoint_timestamp": checkpoint_data.get("timestamp", ""),
            "checkpoint_summary": str(checkpoint_data)[:1000]  # Truncate summary
        })
        logger.info(f"Logged checkpoint '{checkpoint_name}' to W&B")
    except Exception as e:
        logger.warning(f"Failed to log checkpoint to W&B: {e}")
