# config.py

"""
Central configuration file for the Analogical Reasoning RAG project.

This file defines all parameters, file paths, model settings, and control flags
for the entire pipeline. It now supports multiple API providers (Gemini and
OpenAI-compatible services like AvalAI).

By modifying this file, you can easily switch between API providers and run
different experiments without changing the core logic of the source code.
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
    "PRINT_API_CALL_DETAILS": True, # Master switch to print detailed API call info (prompt, response, errors) to the console.
    "PRINT_API_TIMING_CHECKPOINTS": True, # Prints the time elapsed between the start of consecutive API calls.
    "API_RESPONSE_TRUNCATION_LENGTH": 50, # Control truncation for successful API responses.
    "BASE_OUTPUT_DIR": BASE_OUTPUT_DIR,
    "LOGS_DIR": LOGS_DIR, # Directory where detailed run logs will be saved.
    "OUTPUTS_DIR": OUTPUTS_DIR,
    "RESULTS_DIR": RESULTS_DIR,

    # --- 3. API Provider Selection ---
    # Master switch to select the API provider.
    # Options: "gemini" or "avalai"
    # "API_PROVIDER": "gemini", # DEPRECATED: Provider is now set per-step.

    # MODIFIED: Define which API provider to use for each major stage of the pipeline.
    # Options: "gemini" or "avalai"
    "API_PROVIDER_ADAPTATION": "gemini",  # For normalization, transformations, merging
    "API_PROVIDER_SOLVER": "gemini",      # For generating the final solution
    "API_PROVIDER_EVALUATOR": "gemini",   # For LLM-based evaluation

    # --- 4. Gemini API Settings ---
    # These settings are used ONLY if API_PROVIDER is set to "gemini".
    "GEMINI_API_KEYS": [
        # Add your Gemini API keys here.
        # e.g., "AIzaSy...",
    ],
    # Per-model and global rate limiting settings.
    "GEMINI_MODEL_QUOTAS": {
        # MODIFIED: These model names can be updated to match the new format
        "models/gemma-3-27b-it": {"delay_seconds": 20, "rpd": 100},
    },
    "GLOBAL_API_CALL_DELAY_SECONDS": 5, # A minimum delay between any two API calls, regardless of the key.

    # MODIFIED: Names of the models to be used for different pipeline stages.
    # Using the new, fully-qualified model name format.
    "GEMINI_MODEL_NAME_ADAPTATION": "models/gemma-2-9b-it",    # For transformation, summarization, merging. A faster model is often sufficient.
    "GEMINI_MODEL_NAME_FINAL_SOLVER": "models/gemma-2-27b-it", # For generating the final solution. A more powerful model is better here.
    "GEMINI_MODEL_NAME_EVALUATOR": "models/gemma-2-9b-it",    # For LLM-based evaluation. A faster model is sufficient.

    # --- 5. AvalAI (OpenAI-Compatible) API Settings ---
    # These settings are used ONLY if API_PROVIDER is set to "avalai".
    "AVALAI_API_KEY": "YOUR_AVALAI_API_KEY_HERE",
    "AVALAI_BASE_URL": "https://api.avalai.ir/v1",

    # Simple rate limiting for AvalAI. Can be expanded if needed.
    "AVALAI_MODEL_QUOTAS": {
        "default": {"delay_seconds": 2} # A simple 2-second delay between calls.
    },

    # Model names for AvalAI. Replace with any supported model.
    "AVALAI_MODEL_NAME_ADAPTATION": "openai.gpt-oss-20b-1:0",
    "AVALAI_MODEL_NAME_FINAL_SOLVER": "openai.gpt-oss-20b-1:0",
    "AVALAI_MODEL_NAME_EVALUATOR": "openai.gpt-oss-20b-1:0",

    # --- 6. Generic LLM Generation Settings ---
    # These settings are provider-agnostic and will be used by whichever manager is active.
    
    # Temperature Settings
    "DEFAULT_ADAPTATION_TEMPERATURE": 0.0,   # Low temp for deterministic tasks like reformatting.
    "DEFAULT_FINAL_SOLVER_TEMPERATURE": 1.0, # High temp for creative/diverse single-pass solutions.
    "DEFAULT_PASS_N_SOLVER_TEMPERATURE": 1.0,# High temp for generating diverse attempts in Pass@N.
    "DEFAULT_EVALUATOR_TEMPERATURE": 0.0,    # Low temp for deterministic, consistent evaluation.

    # NEW: Max Output Tokens Settings
    # These will be used to construct the `GenerationConfig` for Gemini calls.
    "DEFAULT_ADAPTATION_MAX_TOKENS": 10000,   # Adaptation tasks (normalization, transformation) are usually short.
    "DEFAULT_FINAL_SOLVER_MAX_TOKENS": 10000, # Allow plenty of room for complex reasoning and step-by-step solutions.
    "DEFAULT_EVALUATOR_MAX_TOKENS": 10000,     # Evaluation produces a very short, structured response.

    # --- 7. File Paths, Data & Embedding Settings ---
    "EMBEDDING_MODEL_PATH": 'math-similarity/Bert-MLM_arXiv-MP-class_zbMath',
    
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

    # --- 8. Pipeline Step Control Flags & Parameters ---
    "USE_RETRIEVAL": True,
    "PIPELINE_SEQUENCE": ["retrieve", "adapt", "merge", "solve"],
    
    # MODIFIED: Granular adaptation steps
    "APPLY_NORMALIZATION": False,           # Renamed from APPLY_STANDARDIZATION
    "APPLY_TRANSFORMATION": False,          # DEPRECATED: Replaced by granular transformation flags.
    "APPLY_TRANSFORMATION_1": False,        # NEW: Controls the first transformation step.
    "APPLY_TRANSFORMATION_2": False,        # NEW: Controls the second transformation step.
    "APPLY_TRANSFORMATION_3": False,        # NEW: Controls the third transformation step.
    "APPLY_MERGING": False,
    
    "TOP_N_CANDIDATES_RETRIEVAL": 1,
    "FINAL_K_SELECTION_ADAPTATION": 1,
    "TARGET_ADAPTED_SAMPLES_MERGING": 1,

    # --- 9. Pass@N & Evaluation Settings ---
    "N_PASS_ATTEMPTS": 3,
    "PASS_K_VALUES_TO_REPORT": [1, 2, 3, 4, 5],

    # --- 10. Prompt Template Selection ---
    "PROMPT_TEMPLATE_NORMALIZATION": "standardization_v1",  # Renamed
    "PROMPT_TEMPLATE_STANDARDIZATION": "standardization_v1", # Kept for backward compatibility, but normalization is preferred.
    
    "PROMPT_TEMPLATE_TRANSFORMATION": "transformation_v1",  # DEPRECATED: Replaced by granular transformation flags.
    
    # NEW: Select a prompt for each transformation step independently
    "PROMPT_TEMPLATE_TRANSFORMATION_1": "transformation_shallow",
    "PROMPT_TEMPLATE_TRANSFORMATION_2": "transformation_shallow-&-moderately-deep",
    "PROMPT_TEMPLATE_TRANSFORMATION_3": "transformation_complete",
    
    "PROMPT_TEMPLATE_MERGING": "merging_v1",
    # MODIFIED: Default solver prompt is now v2.
    "PROMPT_TEMPLATE_FINAL_SOLVER": "final_solver_v2",
    "PROMPT_TEMPLATE_EVALUATOR": "evaluator_v1",
    "PROMPT_TEMPLATE_FINAL_SOLVER_SIMPLE": "final_solver_simple_v1",

    # --- 11. Hugging Face Hub Synchronization ---
    "PERSIST_RESULTS_ONLINE": True,
    "HF_SYNC_TOKEN": "YOUR_HUGGING_FACE_TOKEN_HERE",
    "HF_HUB_USERNAME": "your-hf-username-here",
    "HF_HUB_REPO_NAME": "analogical-math-rag-results",
    "HF_SYNC_REVISION_ENABLED": False,
    "HF_SYNC_REVISION_ID": "main",
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