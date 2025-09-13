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
    "VERBOSE_LOGGING": True,
    "BASE_OUTPUT_DIR": BASE_OUTPUT_DIR,
    "LOGS_DIR": LOGS_DIR,
    "OUTPUTS_DIR": OUTPUTS_DIR,
    "RESULTS_DIR": RESULTS_DIR,

    # --- 3. API & Model Settings ---
    "GEMINI_API_KEYS": [
        # Add your Gemini API keys here.
        # e.g., "AIzaSy...",
    ],
    "GEMINI_MODEL_QUOTAS": {
        "gemini-1.5-flash": {"delay_seconds": 4, "rpd": 500},
        "gemini-2.5-flash": {"delay_seconds": 15, "rpd": 200},
        "gemini-2.5-pro": {"delay_seconds": 20, "rpd": 25},
    },
    "GLOBAL_API_CALL_DELAY_SECONDS": 5,

    # Model names for different pipeline stages.
    "GEMINI_MODEL_NAME_ADAPTATION": "gemini-2.5-flash",
    "GEMINI_MODEL_NAME_FINAL_SOLVER": "gemini-2.5-flash",
    "GEMINI_MODEL_NAME_EVALUATOR": "gemini-2.5-flash",

    # Default temperature settings for different LLM tasks.
    "DEFAULT_ADAPTATION_TEMPERATURE": 0.0,
    "DEFAULT_FINAL_SOLVER_TEMPERATURE": 1.0,
    "DEFAULT_PASS_N_SOLVER_TEMPERATURE": 1.0,
    "DEFAULT_EVALUATOR_TEMPERATURE": 0.0,

    # --- 4. File Paths, Data & Embedding Settings ---
    "EMBEDDING_MODEL_PATH": 'math-similarity/Bert-MLM_arXiv-MP-class_zbMath',
    "HARD_QUESTIONS_INDICES_PATH": os.path.join(DATA_DIR, "hard_question_indices.json"),
    "EMBEDDINGS_DIR": EMBEDDINGS_DIR,
    
    # RAG exemplar corpus details.
    "EXEMPLAR_CORPUS_NAME": "AI-MO/NuminaMath-CoT",
    "EXEMPLAR_CORPUS_HF_TOKEN": None,

    # Paths for storing/loading embeddings for the exemplar corpus.
    "EMBEDDED_EXEMPLAR_CORPUS_QUESTIONS_PATH": os.path.join(EMBEDDINGS_DIR, 'embedding_NuminaMath_with_Bert-MLM_arXiv-MP-class_zbMath.npy'),
    "EXEMPLAR_EMBEDDINGS_HF_REPO_ID": "mostafabehroozi/embedding_NuminaMath_with_Bert-MLM_arXiv-MP-class_zbMath",
    "EXEMPLAR_EMBEDDINGS_HF_FILENAME": "embeddings.npy",
    
    # Paths for saving experiment outputs.
    "ADVANCED_RAG_FULL_LOG_PATH": os.path.join(RESULTS_DIR, "advanced_rag_pipeline_full_log.json"),
    "ADVANCED_RAG_EVALUATION_RESULTS_PATH": os.path.join(RESULTS_DIR, "advanced_rag_evaluation_results.pkl"),

    # --- 5. Pipeline Step Control Flags & Parameters ---
    "USE_RETRIEVAL": True,
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

    # --- NEW: Online Evaluation & Early Stopping ---
    # Master switch for real-time evaluation. If False, evaluation happens in a batch at the end.
    "ONLINE_EVALUATION_ENABLED": False,
    # If True (and online evaluation is enabled), stop generating attempts after the first correct answer.
    # WARNING: This is incompatible with Pass@K analysis and should be False for formal experiments.
    "STOP_ON_FIRST_SUCCESS": False,

    # --- 7. Prompt Template Selection ---
    "PROMPT_TEMPLATE_STANDARDIZATION": "standardization_v1",
    "PROMPT_TEMPLATE_TRANSFORMATION": "transformation_v1",
    "PROMPT_TEMPLATE_MERGING": "merging_v1",
    "PROMPT_TEMPLATE_FINAL_SOLVER": "final_solver_v1",
    "PROMPT_TEMPLATE_EVALUATOR": "evaluator_v1",
    "PROMPT_TEMPLATE_FINAL_SOLVER_SIMPLE": "final_solver_simple_v1",

    # --- 8. Hugging Face Hub Synchronization ---
    "PERSIST_RESULTS_ONLINE": True,
    "HF_SYNC_TOKEN": "YOUR_HUGGING_FACE_TOKEN_HERE",
    "HF_HUB_USERNAME": "your-hf-username-here",
    "HF_HUB_REPO_NAME": "analogical-math-rag-results",
    "HF_SYNC_REVISION_ENABLED": False,
    "HF_SYNC_REVISION_ID": "main",
    "HF_SYNC_INTERVAL": 10,

    # --- NEW: 9. Hard Question Identification Settings ---
    # This section contains parameters specifically for the `identify_hard_questions.ipynb` notebook.
    "HARD_QUESTION_IDENTIFICATION_CONFIG": {
        # Number of questions to randomly select from the dataset if no specific indices are provided.
        "NUM_RANDOM_SAMPLES": 100,
        # (Optional) Path to a JSON file containing a specific list of question indices to process.
        # If this is provided, NUM_RANDOM_SAMPLES will be ignored.
        "TARGET_INDICES_FILE_PATH": None, # e.g., os.path.join(DATA_DIR, "my_target_indices.json")
        # The number of attempts the model gets to solve a question before it is classified as "hard".
        "MAX_ATTEMPTS_PER_QUESTION": 5,
        # The file path where the final list of identified hard question indices will be saved.
        "HARD_QUESTIONS_OUTPUT_PATH": os.path.join(RESULTS_DIR, "identified_hard_question_indices.json")
    }
}

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