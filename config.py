# config.py

"""
Central configuration file for the Analogical Reasoning RAG project.

This file defines all parameters, file paths, model settings, and control flags
for the entire pipeline. By modifying this file, you can easily run different
experiments without changing the core logic of the source code.

This version has been updated to support multiple API providers (Gemini and AvalAI).
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
    "API_PROVIDER": "GEMINI",  # <-- MASTER SWITCH: "GEMINI" or "AVALAI"

    # --- 3.1. Gemini Provider Settings ---
    # (These settings are used only if API_PROVIDER is "GEMINI")
    "GEMINI_API_KEYS": [
        # Add your Gemini API keys here.
        # e.g., "AIzaSy...",
    ],
    "GEMINI_MODEL_QUOTAS": {
        "gemini-1.5-flash": {"delay_seconds": 4, "rpd": 500},
        # Add other Gemini models as needed
    },
    "GEMINI_MODEL_NAME_ADAPTATION": "gemini-1.5-flash",
    "GEMINI_MODEL_NAME_FINAL_SOLVER": "gemini-1.5-flash",
    "GEMINI_MODEL_NAME_EVALUATOR": "gemini-1.5-flash",

    # --- 3.2. AvalAI (OpenAI-Compatible) Provider Settings ---
    # (These settings are used only if API_PROVIDER is "AVALAI")
    "AVALAI_CONFIG": {
        "API_KEY": "YOUR_AVALAI_API_KEY_HERE",
        "BASE_URL": "https://api.avalai.ir/v1",
    },
    "AVALAI_MODEL_QUOTAS": {
        "openai.gpt-oss-20b-1:0": {"delay_seconds": 2, "rpd": 1000},
        # Add other AvalAI models as needed
    },
    "AVALAI_MODEL_NAME_ADAPTATION": "openai.gpt-oss-20b-1:0",
    "AVALAI_MODEL_NAME_FINAL_SOLVER": "openai.gpt-oss-20b-1:0",
    "AVALAI_MODEL_NAME_EVALUATOR": "openai.gpt-oss-20b-1:0",

    # NOTE: The notebooks will dynamically select the correct model names based on the
    # API_PROVIDER setting. The code will use generic keys like 'MODEL_NAME_ADAPTATION'
    # which will be populated at runtime.

    # --- 3.3. Global & Temperature Settings (Provider Agnostic) ---
    "GLOBAL_API_CALL_DELAY_SECONDS": 5,

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

    # --- 7. Online Evaluation & Early Stopping ---
    "ONLINE_EVALUATION_ENABLED": False,
    "STOP_ON_FIRST_SUCCESS": False,

    # --- 8. Prompt Template Selection ---
    "PROMPT_TEMPLATE_STANDARDIZATION": "standardization_v1",
    "PROMPT_TEMPLATE_TRANSFORMATION": "transformation_v1",
    "PROMPT_TEMPLATE_MERGING": "merging_v1",
    "PROMPT_TEMPLATE_FINAL_SOLVER": "final_solver_v1",
    "PROMPT_TEMPLATE_EVALUATOR": "evaluator_v1",
    "PROMPT_TEMPLATE_FINAL_SOLVER_SIMPLE": "final_solver_simple_v1",

    # --- 9. Hugging Face Hub Synchronization ---
    "PERSIST_RESULTS_ONLINE": True,
    "HF_SYNC_TOKEN": "YOUR_HUGGING_FACE_TOKEN_HERE",
    "HF_HUB_USERNAME": "your-hf-username-here",
    "HF_HUB_REPO_NAME": "analogical-math-rag-results",
    "HF_SYNC_REVISION_ENABLED": False,
    "HF_SYNC_REVISION_ID": "main",
    "HF_SYNC_INTERVAL": 10,

    # --- 10. Hard Question Identification Settings ---
    "HARD_QUESTION_IDENTIFICATION_CONFIG": {
        "RUN_NAME": "baseline_hard_question_run",
        "NUM_RANDOM_SAMPLES": 100,
        "TARGET_INDICES_FILE_PATH": None,
        "MAX_ATTEMPTS_PER_QUESTION": 5,
        "LOG_FILENAME_PREFIX": "hard_question_identification_log",
        "OUTPUT_FILENAME_PREFIX": "identified_hard_question_indices"
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