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

    # --- 3. API & Model Settings ---
    "GEMINI_API_KEYS": [
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
    
    # Input data source.
    "HARD_QUESTIONS_JSON_PATH": os.path.join(DATA_DIR, "hard_questions_passed@10_251.json"),
    
    # RAG exemplar corpus details.
    "EXEMPLAR_CORPUS_NAME": "AI-MO/NuminaMath-CoT",
    "EXEMPLAR_CORPUS_HF_TOKEN": None, # Your Hugging Face token if the dataset is private.

    # Paths for storing/loading the generated embeddings for the exemplar corpus.
    # The code will first check for a local file, then try to download from HF, then generate if missing.
    "EMBEDDED_EXEMPLAR_CORPUS_QUESTIONS_PATH": os.path.join(EMBEDDINGS_DIR, 'embedding_NuminaMath_with_Bert-MLM_arXiv-MP-class_zbMath.npy'),
    "EXEMPLAR_EMBEDDINGS_HF_REPO_ID": "mostafabehroozi/embedding_NuminaMath_with_Bert-MLM_arXiv-MP-class_zbMath",
    "EXEMPLAR_EMBEDDINGS_HF_FILENAME": "embeddings.npy",
    
    # Paths for saving experiment outputs.
    "ADVANCED_RAG_FULL_LOG_PATH": os.path.join(RESULTS_DIR, "advanced_rag_pipeline_full_log.json"),
    "ADVANCED_RAG_EVALUATION_RESULTS_PATH": os.path.join(RESULTS_DIR, "advanced_rag_evaluation_results.pkl"),

    # --- 5. Pipeline Step Control Flags & Parameters ---
    # This is the new modular control key. The orchestrator will run these steps in order.
    # Possible values: 'retrieve', 'adapt', 'merge', 'solve'
    "PIPELINE_SEQUENCE": ["retrieve", "adapt", "merge", "solve"],

    # Booleans to enable/disable specific adaptation sub-steps.
    "APPLY_TRANSFORMATION": False,
    "APPLY_SUMMARIZATION": False,
    "APPLY_MERGING": False,

    # Numeric parameters for pipeline stages.
    "TOP_N_CANDIDATES_RETRIEVAL": 1,   # How many exemplars to retrieve initially.
    "FINAL_K_SELECTION_ADAPTATION": 1, # How many of the retrieved to actually process in the 'adapt' step.
    "TARGET_ADAPTED_SAMPLES_MERGING": 1, # The target number of exemplars after the 'merge' step.

    # --- 6. Pass@N & Evaluation Settings ---
    "N_PASS_ATTEMPTS": 3, # How many solutions to generate for each question in Pass@N.
    "PASS_K_VALUES_TO_REPORT": [1, 2, 3, 4, 5], # Which Pass@K metrics to calculate (e.g., Pass@1, Pass@3).



    # --- 7. Prompt Template Selection ---
    # Keys should match the keys in src/prompts.py's PROMPT_TEMPLATES dictionary.
    # This allows easy A/B testing of different prompt versions.
    "PROMPT_TEMPLATE_TRANSFORMATION": "transformation_standardize_v1",
    "PROMPT_TEMPLATE_SUMMARIZATION": "summarization_v1",
    "PROMPT_TEMPLATE_MERGING": "merging_v1",
    "PROMPT_TEMPLATE_FINAL_SOLVER": "final_solver_v1",
    "PROMPT_TEMPLATE_EVALUATOR": "evaluator_v1",

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
