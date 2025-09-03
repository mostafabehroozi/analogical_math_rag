# src/hf_sync.py

"""
Hugging Face Hub Synchronization Module (Token-Based).

This file provides functions to manage the persistence of experiment results
and logs by synchronizing the local workspace with a Hugging Face Hub dataset
repository.

This version is modified to pass the Hugging Face token directly from a
configuration dictionary for all API interactions, removing the need for a
separate login step or reliance on cached credentials.

Functions:
- initialize_workspace: Downloads the remote repo to the local machine on startup.
- sync_workspace_to_hub: Uploads local output directories to the remote repo.
- periodic_sync_check: A helper to trigger synchronization during long loops.
"""

import os
import logging
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

def _get_hf_config(config: dict) -> tuple:
    """Helper function to safely extract Hugging Face credentials from the config."""
    # This now correctly points to the token for synchronization.
    hf_token = config.get("HF_SYNC_TOKEN")
    hf_username = config.get("HF_HUB_USERNAME")
    repo_name = config.get("HF_HUB_REPO_NAME")
    return hf_token, hf_username, repo_name

def initialize_workspace(config: dict):
    """
    Downloads all files from the HF Hub repo to the local output directory.

    This populates the workspace with results from previous runs. It uses the
    HF token directly from the provided configuration dictionary.
    """
    logger = logging.getLogger(__name__)
    if not config.get("PERSIST_RESULTS_ONLINE"):
        logger.info("Online persistence is disabled. Skipping workspace initialization.")
        return

    hf_token, hf_username, repo_name = _get_hf_config(config)

    if not all([hf_token, hf_username, repo_name]):
        logger.warning("HF token, username, or repo name not found in config. Cannot initialize workspace.")
        return

    repo_id = f"{hf_username}/{repo_name}"
    local_outputs_dir = config["OUTPUTS_DIR"]

    logger.info(f"Initializing workspace from Hugging Face Hub repo: {repo_id}")

    # Check for specific revision settings from the config
    revision = None
    if config.get("HF_SYNC_REVISION_ENABLED", False):
        revision_id = config.get("HF_SYNC_REVISION_ID")
        if revision_id:
            revision = revision_id
            logger.info(f"Downloading specific revision: {revision}")
        else:
            logger.warning("HF_SYNC_REVISION_ENABLED is True, but HF_SYNC_REVISION_ID is not set. Downloading latest.")
    
    try:
        # 1. Instantiate the API client, passing the token directly.
        api = HfApi(token=hf_token)

        # 2. Ensure the repository exists, creating it if necessary.
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        logger.info(f"Repository {repo_id} exists or was created successfully.")

        # 3. Download the repository's contents, passing the token and revision.
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_outputs_dir,
            local_dir_use_symlinks=False, # Recommended for Kaggle/Docker
            resume_download=True,
            token=hf_token,               # Pass the token for authentication
            revision=revision             # Pass the specific version or None for latest
        )
        logger.info(f"Workspace synchronized. Files from {repo_id} are downloaded to {local_outputs_dir}.")

    except HfHubHTTPError as e:
        logger.error(f"HTTP Error initializing workspace from {repo_id}. Check your HF token permissions. Error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during workspace initialization: {e}", exc_info=True)

def sync_workspace_to_hub(config: dict):
    """
    Uploads the entire local outputs directory to the Hugging Face Hub repo.

    Uses the HF token directly from the provided configuration dictionary.
    """
    logger = logging.getLogger(__name__)
    if not config.get("PERSIST_RESULTS_ONLINE"):
        return # Silently exit if persistence is disabled.

    hf_token, hf_username, repo_name = _get_hf_config(config)

    if not all([hf_token, hf_username, repo_name]):
        logger.warning("HF token, username, or repo name not found in config. Cannot sync workspace.")
        return

    repo_id = f"{hf_username}/{repo_name}"
    local_outputs_dir = config["OUTPUTS_DIR"]

    logger.info(f"Starting synchronization of '{local_outputs_dir}' to HF Hub repo: {repo_id}")

    try:
        # 1. Instantiate the API client with the token.
        api = HfApi(token=hf_token)

        # 2. Upload the entire outputs folder.
        api.upload_folder(
            folder_path=local_outputs_dir,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Automated experiment results sync"
        )
        logger.info(f"Successfully synced '{local_outputs_dir}' to {repo_id}.")
    except Exception as e:
        logger.error(f"Failed to sync workspace to Hugging Face Hub: {e}", exc_info=True)

def periodic_sync_check(loop_counter: int, config: dict):
    """
    Checks if a sync is needed based on the counter and sync interval.
    """
    if not config.get("PERSIST_RESULTS_ONLINE"):
        return

    sync_interval = config.get("HF_SYNC_INTERVAL", 10)

    # Sync after the specified number of items (e.g., if interval is 1, sync after item 1, 2, etc.)
    # We check (loop_counter + 1) because loops are often 0-indexed.
    if (loop_counter + 1) % sync_interval == 0:
        print(f"\n--- Reached sync interval at item #{loop_counter + 1}. Syncing results to Hugging Face Hub. ---")
        sync_workspace_to_hub(config)
        print("--- Sync complete. ---\n")
