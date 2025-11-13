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
- sync_single_file_to_hub: Efficiently uploads a single file.
- sync_workspace_to_hub: Uploads local output directories to the remote repo.
- periodic_sync_check: A helper to trigger synchronization during long loops.
"""

import os
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
    if not config.get("PERSIST_RESULTS_ONLINE"):
        print("Info: Online persistence is disabled. Skipping workspace initialization.")
        return

    hf_token, hf_username, repo_name = _get_hf_config(config)

    if not all([hf_token, hf_username, repo_name]):
        print("Warning: HF token, username, or repo name not found in config. Cannot initialize workspace.")
        return

    repo_id = f"{hf_username}/{repo_name}"
    local_outputs_dir = config["OUTPUTS_DIR"]

    print(f"Info: Initializing workspace from Hugging Face Hub repo: {repo_id}")

    # Check for specific revision settings from the config
    revision = None
    if config.get("HF_SYNC_REVISION_ENABLED", False):
        revision_id = config.get("HF_SYNC_REVISION_ID")
        if revision_id:
            revision = revision_id
            print(f"Info: Downloading specific revision: {revision}")
        else:
            print("Warning: HF_SYNC_REVISION_ENABLED is True, but HF_SYNC_REVISION_ID is not set. Downloading latest.")
    
    try:
        # 1. Instantiate the API client, passing the token directly.
        api = HfApi(token=hf_token)

        # 2. Ensure the repository exists, creating it if necessary.
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"Info: Repository {repo_id} exists or was created successfully.")

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
        print(f"Info: Workspace synchronized. Files from {repo_id} are downloaded to {local_outputs_dir}.")

    except HfHubHTTPError as e:
        print(f"Error: HTTP Error initializing workspace from {repo_id}. Check your HF token permissions. Error: {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred during workspace initialization: {e}")

def sync_single_file_to_hub(config: dict, local_file_path: str):
    """
    Uploads a single local file to its corresponding path in the HF Hub repo.

    This is an efficient, targeted operation for in-loop checkpointing.
    """
    if not config.get("PERSIST_RESULTS_ONLINE"):
        return

    hf_token, hf_username, repo_name = _get_hf_config(config)
    if not all([hf_token, hf_username, repo_name]):
        # Silently return as a warning would have been printed by the main sync function
        return

    if not os.path.exists(local_file_path):
        print(f"Warning: Attempted to sync non-existent file: {local_file_path}")
        return

    repo_id = f"{hf_username}/{repo_name}"
    local_outputs_dir = config["OUTPUTS_DIR"]

    # Determine the destination path in the repository by making it relative
    # to the main outputs directory.
    # Example: /kaggle/working/outputs/exp_name/query_0.json -> exp_name/query_0.json
    path_in_repo = os.path.relpath(local_file_path, local_outputs_dir)

    print(f"  -> Syncing single file: '{path_in_repo}' to HF Hub...")

    try:
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Sync: {os.path.basename(path_in_repo)}"
        )
    except Exception as e:
        print(f"Error: Failed to sync single file '{local_file_path}' to Hub: {e}")


def sync_workspace_to_hub(config: dict):
    """
    Uploads the entire local outputs directory to the Hugging Face Hub repo.

    Uses the HF token directly from the provided configuration dictionary. This is
    a robust but potentially slow operation, best used at the end of a process.
    """
    if not config.get("PERSIST_RESULTS_ONLINE"):
        return # Silently exit if persistence is disabled.

    hf_token, hf_username, repo_name = _get_hf_config(config)

    if not all([hf_token, hf_username, repo_name]):
        print("Warning: HF token, username, or repo name not found in config. Cannot sync workspace.")
        return

    repo_id = f"{hf_username}/{repo_name}"
    local_outputs_dir = config["OUTPUTS_DIR"]

    print(f"Info: Starting synchronization of '{local_outputs_dir}' to HF Hub repo: {repo_id}")

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
        print(f"Info: Successfully synced '{local_outputs_dir}' to {repo_id}.")
    except Exception as e:
        print(f"Error: Failed to sync workspace to Hugging Face Hub: {e}")

def periodic_sync_check(loop_counter: int, config: dict, file_to_sync: str):
    """
    Checks if a sync is needed and syncs ONLY the specified file.
    """
    if not config.get("PERSIST_RESULTS_ONLINE"):
        return

    sync_interval = config.get("HF_SYNC_INTERVAL", 10)

    # Sync after the specified number of items (e.g., if interval is 1, sync after item 1, 2, etc.)
    # We check (loop_counter + 1) because loops are often 0-indexed.
    if (loop_counter + 1) % sync_interval == 0:
        print(f"\n--- Reached sync interval at item #{loop_counter + 1}. Syncing latest result file. ---")
        # Call the new, efficient single-file sync function
        # instead of the slow, full-directory sync.
        sync_single_file_to_hub(config, local_file_path=file_to_sync)
        print("--- Sync complete. ---\n")