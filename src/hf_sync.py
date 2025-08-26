# src/hf_sync.py

"""
Hugging Face Hub Synchronization Module.

This file provides functions to manage the persistence of experiment results
and logs by synchronizing the local workspace with a Hugging Face Hub dataset repository.

Functions:
- initialize_workspace: Downloads the remote repo state to the local machine on startup.
- sync_workspace_to_hub: Uploads the local output directories to the remote repo.
- periodic_sync_check: A helper to trigger synchronization during long-running loops.
"""

import os
import logging
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

def initialize_workspace(config: dict):
    """
    Downloads all files from the HF Hub repo to the local output directory.
    This populates the workspace with the results from previous runs.
    """
    logger = logging.getLogger(__name__)
    if not config.get("PERSIST_RESULTS_ONLINE"):
        logger.info("Online persistence is disabled. Skipping workspace initialization from Hub.")
        return

    hf_username = config.get("HF_HUB_USERNAME")
    repo_name = config.get("HF_HUB_REPO_NAME")
    
    if not hf_username or "your-hf-username-here" in hf_username:
        logger.warning("HF_HUB_USERNAME is not set in config. Cannot initialize workspace.")
        return

    repo_id = f"{hf_username}/{repo_name}"
    local_outputs_dir = config["OUTPUTS_DIR"]
    
    logger.info(f"Initializing workspace from Hugging Face Hub repo: {repo_id}")
    
    try:
        # Ensure the repository exists, create it if it's the first run.
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        logger.info(f"Repository {repo_id} exists or was created successfully.")

        # Download the entire repository content to the local outputs directory.
        # This will overwrite local files if remote versions are newer.
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_outputs_dir,
            local_dir_use_symlinks=False, # Use direct copies in Kaggle
            resume_download=True,
        )
        logger.info(f"Workspace synchronized. All files from {repo_id} are downloaded to {local_outputs_dir}.")

    except HfHubHTTPError as e:
        # This can happen if the repo is private and the token is wrong.
        logger.error(f"HTTP Error initializing workspace from {repo_id}. Check your HF token and repo permissions. Error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during workspace initialization: {e}", exc_info=True)


def sync_workspace_to_hub(config: dict):
    """
    Uploads the entire local outputs directory to the Hugging Face Hub repo.
    """
    logger = logging.getLogger(__name__)
    if not config.get("PERSIST_RESULTS_ONLINE"):
        # This check prevents accidental uploads if the feature is off.
        return

    hf_username = config.get("HF_HUB_USERNAME")
    repo_name = config.get("HF_HUB_REPO_NAME")

    if not hf_username or "your-hf-username-here" in hf_username:
        logger.warning("HF_HUB_USERNAME is not set in config. Cannot sync workspace.")
        return

    repo_id = f"{hf_username}/{repo_name}"
    local_outputs_dir = config["OUTPUTS_DIR"]

    logger.info(f"Starting synchronization of local workspace to HF Hub repo: {repo_id}")

    try:
        api = HfApi()
        # Upload the entire 'outputs' folder. This will add new files, update existing ones,
        # and can even delete files in the repo if they are deleted locally (with allow_patterns).
        api.upload_folder(
            folder_path=local_outputs_dir,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Periodic experiment results sync"
        )
        logger.info(f"Successfully synced {local_outputs_dir} to {repo_id}.")
    except Exception as e:
        logger.error(f"Failed to sync workspace to Hugging Face Hub: {e}", exc_info=True)


def periodic_sync_check(counter: int, config: dict):
    """
    Checks if a sync is needed based on the counter and sync interval from the config.
    """
    if not config.get("PERSIST_RESULTS_ONLINE"):
        return

    sync_interval = config.get("HF_SYNC_INTERVAL", 10)
    # We check counter > 0 to avoid syncing on the very first item (index 0).
    if counter > 0 and (counter + 1) % sync_interval == 0:
        print(f"\n--- Reached sync interval at item #{counter + 1}. Syncing results to Hugging Face Hub. ---")
        sync_workspace_to_hub(config)
        print(f"--- Sync complete. ---\n")

