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

import logging
import threading
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
    
    # 1. Check if persistence is enabled
    if not config.get("PERSIST_RESULTS_ONLINE"):
        logger.info("Online persistence is disabled. Skipping workspace initialization.")
        return

    # 2. Extract credentials safely
    hf_token, hf_username, repo_name = _get_hf_config(config)

    # Check for empty credentials or default placeholders
    if not all([hf_token, hf_username, repo_name]) or "YOUR_HUGGING_FACE" in str(hf_token):
        logger.warning("HF token, username, or repo name not found in config. Cannot initialize workspace.")
        print("\n⚠️ WARNING: Missing or default Hugging Face configuration. Skipping workspace initialization.")
        return

    repo_id = f"{hf_username}/{repo_name}"
    # Changed: Download to BASE_OUTPUT_DIR instead of OUTPUTS_DIR for consistency
    # This allows all subdirectories (outputs, results, logs) to be downloaded and available
    local_outputs_dir = config["BASE_OUTPUT_DIR"]

    logger.info(f"Initializing workspace from Hugging Face Hub repo: {repo_id}")

    # 3. Handle specific revision settings
    revision = None
    if config.get("HF_SYNC_REVISION_ENABLED", False):
        revision_id = config.get("HF_SYNC_REVISION_ID")
        if revision_id:
            revision = revision_id
            logger.info(f"Downloading specific revision: {revision}")
        else:
            logger.warning("HF_SYNC_REVISION_ENABLED is True, but HF_SYNC_REVISION_ID is not set. Downloading latest.")
    
    # 4. Attempt API interactions
    try:
        # Instantiate the API client
        api = HfApi(token=hf_token)

        # Ensure the repository exists, creating it if necessary
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        logger.info(f"Repository {repo_id} exists or was created successfully.")

        # Download the repository's contents
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_outputs_dir,
            token=hf_token,               # Pass the token for authentication
            revision=revision             # Pass the specific version or None for latest
        )

        logger.info(f"Workspace synchronized. Files from {repo_id} are downloaded to {local_outputs_dir}.")
        print(f"✅ Hugging Face workspace successfully initialized from {repo_id}")

    except HfHubHTTPError as e:
        # Clear, non-halting warning for HTTP/Authentication issues
        error_msg = f"HTTP Error initializing workspace from {repo_id}. Check your HF token permissions. Details: {str(e)}"
        print(f"\n⚠️ WARNING: HF Sync Failed (Download/Init HTTP Error) - {str(e)}")
        print("Continuing pipeline execution with local files...\n")
        logger.warning(error_msg)
        
    except Exception as e:
        # Clear, non-halting warning for any other unexpected errors
        error_msg = f"An unexpected error occurred during workspace initialization: {str(e)}"
        print(f"\n⚠️ WARNING: HF Sync Failed (Download/Init Unexpected Error) - {str(e)}")
        print("Continuing pipeline execution with local files...\n")
        logger.warning(error_msg)

def sync_workspace_to_hub(config: dict):
    """
    Uploads the entire local outputs directory to the Hugging Face Hub repo.

    Uses the HF token directly from the provided configuration dictionary.
    """
    logger = logging.getLogger(__name__)
    if not config.get("PERSIST_RESULTS_ONLINE"):
        return # Silently exit if persistence is disabled.

    hf_token, hf_username, repo_name = _get_hf_config(config)

    # Check for empty credentials or default placeholders
    if not all([hf_token, hf_username, repo_name]) or "YOUR_HUGGING_FACE" in str(hf_token):
        logger.warning("HF token, username, or repo name not found or contains placeholders. Cannot sync workspace.")
        print("\n⚠️ WARNING: Hugging Face sync configuration is incomplete or contains placeholder values. Sync skipped.")
        return

    repo_id = f"{hf_username}/{repo_name}"
    # Changed: Upload BASE_OUTPUT_DIR instead of OUTPUTS_DIR to include both /outputs/ and /results/ directories
    # This ensures Layer-2 analytics output (saved to RESULTS_DIR) gets synced to HF Hub
    local_outputs_dir = config["BASE_OUTPUT_DIR"]

    logger.info(f"Starting synchronization of '{local_outputs_dir}' to HF Hub repo: {repo_id}")

    try:
        # 1. Instantiate the API client with the token.
        api = HfApi(token=hf_token)

        # 2. Upload only the specific folders using an allowlist to prevent leaks
        api.upload_large_folder(
            folder_path=local_outputs_dir,
            repo_id=repo_id,
            repo_type="dataset",
            # ONLY upload files inside these specific directories
            allow_patterns=[
                "outputs/logs/**",     # Logs folder in Kaggle mode
                "outputs/results/**",  # Results folder in Kaggle mode
                "logs/**",             # Logs folder in Offline mode
                "results/**"           # Results folder in Offline mode
            ]
        )
        logger.info(f"Successfully synced '{local_outputs_dir}' to {repo_id}.")
        print(f"✅ Backup complete: Results synced to Hugging Face ({repo_id})")
        
    except Exception as e:
        # Clear, non-halting warning
        print(f"\n⚠️ WARNING: HF Sync Failed (Upload Error) - {str(e)}")
        print("Continuing pipeline execution. Results are still saved locally...\n")
        logger.warning(f"Failed to sync workspace to Hugging Face Hub: {str(e)}")

def periodic_sync_check(loop_counter: int, config: dict):
    """
    Checks if a sync is needed based on the counter and sync interval.
    """
    if not config.get("PERSIST_RESULTS_ONLINE"):
        return

    sync_interval = config.get("HF_SYNC_INTERVAL", 10)


    if (loop_counter + 1) % sync_interval == 0:
        print(f"\n--- Reached sync interval at item #{loop_counter + 1}. Syncing results to Hugging Face Hub. ---")
        sync_workspace_to_hub(config)
        print("--- Sync complete. ---\n")


_sync_lock = threading.Lock()
_active_sync_thread = None  

def _threaded_sync(config: dict):
    """Wrapper that only runs if no other sync is currently running."""
    # blocking=False means if the lock is already taken, it immediately gives up and returns
    if not _sync_lock.acquire(blocking=False):
        print("--- Background sync already in progress. Skipping this duplicate trigger. ---")
        return
    
    try:
        sync_workspace_to_hub(config)
    finally:
        # Always release the lock when finished, even if it crashes
        _sync_lock.release()

def periodic_batch_sync_check(batch_counter: int, config: dict):
    """Synchronize only after a committed batch, never from a worker thread."""
    global _active_sync_thread
    
    if not config.get("PERSIST_RESULTS_ONLINE"):
        return
    sync_interval = config.get("HF_SYNC_INTERVAL_BATCHES", config.get("HF_SYNC_INTERVAL", 10))
    if (batch_counter + 1) % max(1, int(sync_interval)) == 0:
        print(f"\n--- Reached committed batch #{batch_counter + 1}. Spawning background thread to sync to HF Hub. ---")
        
        _active_sync_thread = threading.Thread(target=_threaded_sync, args=(config,), daemon=False)
        _active_sync_thread.start()
        
        print("--- Background sync check requested. Pipeline continuing immediately... ---\n")

def wait_for_final_sync():
    """Call this at the end of your script to ensure uploads finish."""
    global _active_sync_thread
    if _active_sync_thread and _active_sync_thread.is_alive():
        print("\n⏳ Waiting for final Hugging Face background sync to finish before exiting...")
        _active_sync_thread.join()
        print("✅ Background sync complete!")
