# config.py

"""
Central configuration file for the Analogical Reasoning RAG project.

This file defines all parameters, file paths, model settings, and control flags
for the entire pipeline. By modifying this file, you can easily run different
experiments without changing the core logic of the source code.
"""

import os

# --- 1. Core Directory Structure ---
# Define the base directory for all outputs. This is the root for logs, results, etc.
# In Kaggle, this is typically '/kaggle/working/'.
BASE_OUTPUT_DIR = "/kaggle/working/"

# Define subdirectories for organized output.
DATA_DIR = os.path.join(BASE_OUTPUT_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_OUTPUT_DIR, "outputs")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
EMBEDDINGS_DIR = os.path.join(OUTPUTS_DIR, "embeddings")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")

# --- Main CONFIG Dictionary ---

CONFIG = {
    # --- 2. Logging & Control Settings ---
    "VERBOSE_LOGGING": True,  # Master switch for detailed print statements to the console.
    "BASE_OUTPUT_DIR": BASE_OUTPUT_DIR,
    "LOGS_DIR": LOGS_DIR, # Directory where detailed run logs will be saved.
    "OUTPUTS_DIR": OUTPUTS_DIR,
    "RESULTS_DIR": RESULTS_DIR,

    # --- 3. API & Model Settings ---
    "GEMINI_API_KEYS": [
        # Add your Gemini API keys here.
        # e.g., "AIzaSy...",
    ],
    # Per-model and global rate limiting settings.
    "GEMINI_MODEL_QUOTAS": {
        "gemini-1.5-flash": {"delay_seconds": 4, "rpd": 500},
        "gemini-2.5-flash": {"delay_seconds": 15, "rpd": 200},
        "gemini-2.5-pro": {"delay_seconds": 20, "rpd": 25},
    },
    "GLOBAL_API_CALL_DELAY_SECONDS": 5, # A minimum delay between any two API calls, regardless of the key.

    # Names of the models to be used for different pipeline stages.
    "GEMINI_MODEL_NAME_ADAPTATION": "gemini-2.5-flash",    # For transformation, summarization, merging.
    "GEMINI_MODEL_NAME_FINAL_SOLVER": "gemini-2.5-flash", # For generating the final solution.
    "GEMINI_MODEL_NAME_EVALUATOR": "gemini-2.5-flash",    # For LLM-based evaluation.

    # Default temperature settings for different LLM tasks.
    "DEFAULT_ADAPTATION_TEMPERATURE": 0.0,   # Low temp for deterministic tasks like reformatting.
    "DEFAULT_FINAL_SOLVER_TEMPERATURE": 1.0, # High temp for creative/diverse single-pass solutions.
    "DEFAULT_PASS_N_SOLVER_TEMPERATURE": 1.0,# High temp for generating diverse attempts in Pass@N.
    "DEFAULT_EVALUATOR_TEMPERATURE": 0.0,    # Low temp for deterministic, consistent evaluation.

    # --- 4. File Paths, Data & Embedding Settings ---
    "EMBEDDING_MODEL_PATH": 'math-similarity/Bert-MLM_arXiv-MP-class_zbMath',
    
    # MODIFIED: Input data source now points to a file with indices, not full questions.
    "HARD_QUESTIONS_INDICES_PATH": os.path.join(DATA_DIR, "hard_question_indices.json"),
    "EMBEDDINGS_DIR": EMBEDDINGS_DIR,
    
    # RAG exemplar corpus details.
    "EXEMPLAR_CORPUS_NAME": "AI-MO/NuminaMath-CoT",
    "EXEMPLAR_CORPUS_HF_TOKEN": None, # Your Hugging Face token if the dataset is private.

    # Paths for storing/loading the generated embeddings for the exemplar corpus.
    "EMBEDDED_EXEMPLAR_CORPUS_QUESTIONS_PATH": os.path.join(EMBEDDINGS_DIR, 'embedding_NuminaMath_with_Bert-MLM_arXiv-MP-class_zbMath.npy'),
    "EXEMPLAR_EMBEDDINGS_HF_REPO_ID": "mostafabehroozi/embedding_NuminaMath_with_Bert-MLM_arXiv-MP-class_zbMath",
    "EXEMPLAR_EMBEDDINGS_HF_FILENAME": "embeddings.npy",
    
    # Paths for saving experiment outputs.
    "ADVANCED_RAG_FULL_LOG_PATH": os.path.join(RESULTS_DIR, "advanced_rag_pipeline_full_log.json"),
    "ADVANCED_RAG_EVALUATION_RESULTS_PATH": os.path.join(RESULTS_DIR, "advanced_rag_evaluation_results.pkl"),

    # --- 5. Pipeline Step Control Flags & Parameters ---
    "USE_RETRIEVAL": True, # NEW: Master switch for the retrieval process.
    "PIPELINE_SEQUENCE": ["retrieve", "adapt", "merge", "solve"],
    "APPLY_STANDARDIZATION": False,
    "APPLY_TRANSFORMATION": False,
    "APPLY_MERGING": False,
    "TOP_N_CANDIDATES_RETRIEVAL": 1,
    "FINAL_K_SELECTION_ADAPTATION": 1,
    "TARGET_ADAPTED_SAMPLES_MERGING": 1,

    # --- 6. Pass@N & Evaluation Settings ---
    "N_PASS_ATTEMPTS": 3,
    "PASS_K_VALUES_TO_REPORT": [1, 2, 3, 4, 5],

    # --- 7. Prompt Template Selection ---
    "PROMPT_TEMPLATE_STANDARDIZATION": "standardization_v1",
    "PROMPT_TEMPLATE_TRANSFORMATION": "transformation_v1",
    "PROMPT_TEMPLATE_MERGING": "merging_v1",
    "PROMPT_TEMPLATE_FINAL_SOLVER": "final_solver_v1",
    "PROMPT_TEMPLATE_EVALUATOR": "evaluator_v1",
    # NEW: Prompt for when retrieval is turned off.
    "PROMPT_TEMPLATE_FINAL_SOLVER_SIMPLE": "final_solver_simple_v1",

    # --- 8. Hugging Face Hub Synchronization ---
    # Master switch to enable or disable the entire synchronization feature.
    "PERSIST_RESULTS_ONLINE": True,

    # This token is specifically for the synchronization process.
    # PASTE YOUR HUGGING FACE TOKEN HERE (e.g., "hf_...").
    "HF_SYNC_TOKEN": "YOUR_HUGGING_FACE_TOKEN_HERE",
    
    # Your Hugging Face username. The repo will be created under this account.
    # IMPORTANT: Change this to your actual username.
    "HF_HUB_USERNAME": "your-hf-username-here",
    
    # The name of the dataset repository on the Hub where results will be stored.
    "HF_HUB_REPO_NAME": "analogical-math-rag-results",
    
    # Set to True to download a specific version of the repository workspace.
    "HF_SYNC_REVISION_ENABLED": False,
    
    # The git revision (branch, tag, or commit hash) to download.
    # This is ONLY used if HF_SYNC_REVISION_ENABLED is set to True.
    # Examples: "main", "v1.0", "a1b2c3d4e5f6..."
    "HF_SYNC_REVISION_ID": "main",
    
    # How often to sync the local workspace to the Hub.
    # A sync will occur after this many queries are processed in a loop.
    "HF_SYNC_INTERVAL": 10,

} # This closes the main CONFIG dictionary


def setup_directories():
    """
    Creates the necessary directory structure defined in the configuration.
    This function should be called once at the beginning of a run.
    """
    print("--- Setting up project directories ---")
    for dir_path in [DATA_DIR, OUTPUTS_DIR, LOGS_DIR, EMBEDDINGS_DIR, RESULTS_DIR]:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Directory ensured: {dir_path}")
        except OSError as e:
            print(f"Error creating directory {dir_path}: {e}")
    print("--- Directory setup complete ---\n")
