#======================================================================
#   File: config.py
#======================================================================

import os

# --- 1. Core Directory Structure ---
BASE_OUTPUT_DIR = "/kaggle/working/"
DATA_DIR = os.path.join(BASE_OUTPUT_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_OUTPUT_DIR, "outputs")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
EMBEDDINGS_DIR = os.path.join(OUTPUTS_DIR, "embeddings")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")

# --- Main CONFIG Dictionary ---
CONFIG = {
    # --- Logging & Control ---
    "VERBOSE_LOGGING": True,
    "PRINT_API_CALL_DETAILS": True,
    "PRINT_API_TIMING_CHECKPOINTS": True,
    "API_RESPONSE_TRUNCATION_LENGTH": 50,
    "BASE_OUTPUT_DIR": BASE_OUTPUT_DIR,
    "LOGS_DIR": LOGS_DIR,
    "OUTPUTS_DIR": OUTPUTS_DIR,
    "RESULTS_DIR": RESULTS_DIR,

    # --- API Provider Selection ---
    "API_PROVIDER_ADAPTATION": "gemini",
    "API_PROVIDER_SOLVER": "gemini",
    "API_PROVIDER_EVALUATOR": "gemini",
    "API_PROVIDER_AUGMENTATION": "gemini",
    "API_PROVIDER_SIMPLIFICATION": "gemini",

    # --- Gemini API Settings ---
    "GEMINI_API_KEYS": [
        # Add your Gemini API keys here
    ],
    "GEMINI_MODEL_QUOTAS": {
        # UPDATED: Increased limits to accommodate Mirroring inner loops
        "models/gemma-3-27b-it": {"delay_seconds": 2, "rpd": 1000},
    },
    "GLOBAL_API_CALL_DELAY_SECONDS": 5,

    "GEMINI_MODEL_NAME_ADAPTATION": "models/gemma-3-27b-it",
    "GEMINI_MODEL_NAME_FINAL_SOLVER": "models/gemma-3-27b-it",
    "GEMINI_MODEL_NAME_EVALUATOR": "models/gemma-3-27b-it",
    "GEMINI_MODEL_NAME_AUGMENTATION": "models/gemma-3-27b-it",
    "GEMINI_MODEL_NAME_SIMPLIFICATION": "models/gemma-3-27b-it",

    # --- AvalAI (OpenAI-Compatible) API Settings ---
    "AVALAI_API_KEY": "YOUR_AVALAI_API_KEY_HERE",
    "AVALAI_BASE_URL": "https://api.avalai.ir/v1",
    "AVALAI_MODEL_QUOTAS": {
        "default": {"delay_seconds": 2}
    },
    "AVALAI_MODEL_NAME_ADAPTATION": "openai.gpt-oss-20b-1:0",
    "AVALAI_MODEL_NAME_FINAL_SOLVER": "openai.gpt-oss-20b-1:0",
    "AVALAI_MODEL_NAME_EVALUATOR": "openai.gpt-oss-20b-1:0",
    "AVALAI_MODEL_NAME_AUGMENTATION": "openai.gpt-oss-20b-1:0",
    "AVALAI_MODEL_NAME_SIMPLIFICATION": "openai.gpt-oss-20b-1:0",

    # --- Ollama (Local LLM) Settings ---
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "OLLAMA_MODEL_NAME_ADAPTATION": "gpt-oss:20b",
    "OLLAMA_MODEL_NAME_FINAL_SOLVER": "gpt-oss:20b",
    "OLLAMA_MODEL_NAME_EVALUATOR": "gpt-oss:20b",
    "OLLAMA_MODEL_NAME_AUGMENTATION": "llama3:8b",
    "OLLAMA_MODEL_NAME_SIMPLIFICATION": "llama3:8b",

    # --- Generic LLM Generation Settings ---
    "DEFAULT_ADAPTATION_TEMPERATURE": 0.0,
    "DEFAULT_ANALOGICAL_ADAPTATION_TEMPERATURE": 1.0,
    "DEFAULT_FINAL_SOLVER_TEMPERATURE": 1.0,
    "DEFAULT_PASS_N_SOLVER_TEMPERATURE": 1.0,
    "DEFAULT_EVALUATOR_TEMPERATURE": 0.0,
    "DEFAULT_AUGMENTATION_TEMPERATURE": 1.0,
    "DEFAULT_SIMPLIFICATION_TEMPERATURE": 0.3,

    "DEFAULT_ADAPTATION_MAX_TOKENS": 10000,
    "DEFAULT_FINAL_SOLVER_MAX_TOKENS": 10000,
    "DEFAULT_EVALUATOR_MAX_TOKENS": 10000,

    # --- File Paths & Data ---
    "EMBEDDING_MODEL_PATH": 'math-similarity/Bert-MLM_arXiv-MP-class_zbMath',
    "HARD_QUESTIONS_INDICES_PATH": os.path.join(DATA_DIR, "hard_question_indices.json"),
    "EMBEDDINGS_DIR": EMBEDDINGS_DIR,
    "EXEMPLAR_CORPUS_NAME": "AI-MO/NuminaMath-CoT",
    "EXEMPLAR_CORPUS_HF_TOKEN": None,
    "EMBEDDED_EXEMPLAR_CORPUS_QUESTIONS_PATH": os.path.join(EMBEDDINGS_DIR, 'embedding_NuminaMath_with_Bert-MLM_arXiv-MP-class_zbMath.npy'),
    "EXEMPLAR_EMBEDDINGS_HF_REPO_ID": "mostafabehroozi/embedding_NuminaMath_with_Bert-MLM_arXiv-MP-class_zbMath",
    "EXEMPLAR_EMBEDDINGS_HF_FILENAME": "embeddings.npy",
    
    "ADVANCED_RAG_FULL_LOG_PATH": os.path.join(RESULTS_DIR, "advanced_rag_pipeline_full_log.json"),
    "ADVANCED_RAG_EVALUATION_RESULTS_PATH": os.path.join(RESULTS_DIR, "advanced_rag_evaluation_results.pkl"),

    # --- Pipeline Control Flags ---
    "USE_RETRIEVAL": True,
    "PIPELINE_SEQUENCE": ["retrieve", "adapt", "merge", "solve"],
    "DEFER_SOLVE_STEP": False,
    "TOP_N_CANDIDATES_RETRIEVAL": 1,
    "FINAL_K_SELECTION_ADAPTATION": 1,
    "TARGET_ADAPTED_SAMPLES_MERGING": 1,

    # --- Adaptation Steps ---
    "APPLY_NORMALIZATION": False,
    "APPLY_TRANSFORMATION": False,
    "APPLY_TRANSFORMATION_1": False,
    "APPLY_TRANSFORMATION_2": False,
    "APPLY_TRANSFORMATION_3": False,
    "APPLY_MERGING": False,

    # --- Self-Sampling ---
    "APPLY_SELF_SAMPLING": False,
    "SELF_SAMPLING_N": 3,
    "SELF_SAMPLING_TEMPERATURE": 1.0,

    # --- Analogical Adaptation ---
    "APPLY_ANALOGICAL_ADAPTATION": False,
    "ANALOGICAL_ADAPTATION_SAMPLING_N": 3,

    # --- Augmentation & Selection ---
    "APPLY_SELF_SAMPLING_AUGMENTATION": False,
    "APPLY_ANALOGICAL_ADAPTATION_AUGMENTATION": False,
    "ANALOGICAL_USE_MAIN_QUERY_AS_AUGMENTATION": False,
    "SELECTIVE_AUGMENTATION_SAMPLING": False,
    "AUGMENTATION_SCHEDULE": None,
    "AUGMENT_K": 10,
    "AUGMENT_N": 3,
    "SELECTIVE_AUGMENTATION_SAMPLING_MODE": "auto",

    # --- Consistency Check (Pathway) ---
    "APPLY_CONSISTENCY_ANALOGICAL_CHECK": False,
    "CONSISTENCY_GENERATION_MODE": "distinct_augmentations",
    "CONSISTENCY_PATHWAYS_K": 3,
    "CONSISTENCY_LAYER_1_TEMPERATURE": 1.0,
    "CONSISTENCY_SAMPLES_PER_PATHWAY_N": 3,
    "CONSISTENCY_LAYER_2_TEMPERATURE": 1.0,
    "CONSISTENCY_VOTING_THRESHOLD": 0.6,

    # ==============================================================================
    # --- MIRROR Track A: Ranking Robustness Benchmark ---
    # ==============================================================================
    # Formerly "Group Consistency". In the new "Acquire-Optimize-Fork" architecture,
    # this defines Track A: The Statistical Benchmarking of the Master List (list_2).
    
    # If True, the pipeline executes the "Ranking Robustness" track.
    # It applies the slicing defined below to the Master Sorted List (list_2).
    "APPLY_GROUP_CONSISTENCY_SELECTION": True,

    # DEFINITION OF STRATEGIES (Slicing of list_2)
    # Each tuple represents specific indices from the Master List (list_2) to use.
    # Group 1 (0,): 1-Shot Benchmark -> Uses list_2[0]
    # Group 2 (0, 1): 2-Shot Benchmark -> Uses list_2[0] and list_2[1]
    # Group 3 (0, 1, 2): 3-Shot Benchmark -> Uses list_2[0], list_2[1], list_2[2]
    "GROUP_CONSISTENCY_CANDIDATES": [(0,), (0, 1), (0, 1, 2)],

    # N_gen (Evaluation Sampling):
    # Number of independent generations per slice to calculate Pass@K curves.
    # This is the "N" in the Benchmark Track (Track A).
    # Recommended: 10 <= N <= 40 for statistically significant benchmarking.
    "GROUP_CONSISTENCY_SAMPLES_N": 10,

    # These legacy scoring flags can be ignored or set to None as we are 
    # now doing independent Pass@K benchmarking, not internal voting.
    "CONSISTENCY_SCORING_METHOD": "benchmark_report", 
    "SEMANTIC_CONSISTENCY_WEIGHT": 0.0,

    # --- Hierarchical Augmentation ---
    "APPLY_HIERARCHICAL_AUGMENTATION": False,
    "HIERARCHICAL_AUGMENTATION_MODE": "decomposition", # "decomposition" or "simplification"
    "HIERARCHICAL_TREE_DEPTH": 2,
    "HIERARCHICAL_BRANCHING_FACTOR": 3,
    "HIERARCHICAL_LEAF_RETRIEVAL_ENABLED": True,
    "HIERARCHICAL_LEAF_RETRIEVAL_TOP_K": 3,
    "HIERARCHICAL_LEAF_RETRIEVAL_QUERY_MODE": "leaf", # "leaf" or "root"
    
    # --- NEW: Two-Step Augmentation ---
    "HIERARCHICAL_AUGMENTATION_TWO_STEP": False,

    # --- Simplification Mode ---
    "APPLY_SIMPLIFICATION": False,
    "SIMPLIFY_RETRIEVED_SAMPLES": False,
    "SIMPLIFY_MAIN_QUESTION": False,

    # --- Reverse Validation ---
    "APPLY_REVERSE_VALIDATION": False,
    "REVERSE_VALIDATION_CANDIDATES_N": 5,
    "REVERSE_VALIDATION_RETRIEVAL_K": 3,
    "REVERSE_VALIDATION_ATTEMPTS_N": 5,

    # --- Pass@N & Evaluation ---
    "N_PASS_ATTEMPTS": 3,
    "APPLY_FULL_PIPELINE_RETRY": False,
    "PASS_K_VALUES_TO_REPORT": [1, 2, 3, 4, 5],

    # --- Prompt Templates ---
    "PROMPT_TEMPLATE_NORMALIZATION": "standardization_v1",
    "PROMPT_TEMPLATE_STANDARDIZATION": "standardization_v1",
    
    "PROMPT_TEMPLATE_TRANSFORMATION": "transformation_v1",
    "PROMPT_TEMPLATE_TRANSFORMATION_1": "transformation_shallow",
    "PROMPT_TEMPLATE_TRANSFORMATION_2": "transformation_shallow-&-moderately-deep",
    "PROMPT_TEMPLATE_TRANSFORMATION_3": "transformation_complete",
    
    "PROMPT_TEMPLATE_MERGING": "merging_v1",
    "PROMPT_TEMPLATE_FINAL_SOLVER": "final_solver_v2",
    "PROMPT_TEMPLATE_FINAL_SOLVER_SIMPLE": "final_solver_simple_v1",
    "PROMPT_TEMPLATE_EVALUATOR": "evaluator_v1",

    "PROMPT_TEMPLATE_SELF_SAMPLING_GENERATOR": "self_sampling_generator",
    "PROMPT_TEMPLATE_SELF_SAMPLING_AUGMENTOR": "self_sampling_augmentor_v1",
    "PROMPT_TEMPLATE_ANALOGICAL_ADAPTATION": "analogical_adaptation_v1",
    "PROMPT_TEMPLATE_CONSISTENCY_SOLVER": "analogical_adaptation_v1",
    "PROMPT_TEMPLATE_REVERSE_VALIDATION_SOLVER": "analogical_adaptation_v1",

    "PROMPT_TEMPLATE_HIERARCHICAL_AUGMENTOR": "self_sampling_augmentor_decomposition_2",
    "PROMPT_TEMPLATE_HIERARCHICAL_PARENT_SOLVER": "hierarchical_parent_solver_v1",
    "PROMPT_TEMPLATE_HIERARCHICAL_LEAF_SOLVER": "final_solver_simple_v1",

    # --- NEW: Two-Step Augmentation Prompts ---
    "PROMPT_TEMPLATE_AUGMENTATION_STEP1_SOLVER": "final_solver_simple_v2",
    "PROMPT_TEMPLATE_AUGMENTATION_STEP2_GENERATOR": "self_sampling_augmentor_simplification_with_solution",

    "PROMPT_TEMPLATE_SIMPLIFICATION_GENERATOR": "simplification_generator_v1",
    "PROMPT_TEMPLATE_SIMPLIFIED_SAMPLE_SOLVER": "simplified_sample_solver_v1",
    "PROMPT_TEMPLATE_SIMPLIFIED_MAIN_PROXY_SOLVER": "main_from_simplified_proxy_v1",

    # --- Hugging Face Hub ---
    "PERSIST_RESULTS_ONLINE": True,
    "HF_SYNC_TOKEN": "YOUR_HUGGING_FACE_TOKEN_HERE",
    "HF_HUB_USERNAME": "your-hf-username-here",
    "HF_HUB_REPO_NAME": "analogical-math-rag-results",
    "HF_SYNC_REVISION_ENABLED": False,
    "HF_SYNC_REVISION_ID": "main",
    "HF_SYNC_INTERVAL": 10,

    # ==============================================================================
    # --- MIRROR_AS_EVALUATOR (Analogical Mirroring) ---
    # ==============================================================================
    # Master Switch: Enables the post-retrieval optimization loop.
    # This triggers Phase 1: Acquire & Optimize (Creating list_2).
    "APPLY_MIRROR_AS_EVALUATOR": True,

    # 2.1 Sampling Parameters (Optimization Phase)
    # N_mirror: Number of zero-shot attempts used to calculate consistency scores.
    # Higher values = more robust scoring but higher cost. (Recommended: 3-5)
    "MIRROR_N_OPTIMIZATION": 3,
    
    # 2.2 Feature Toggles (Optimization Phase)
    # enable_R0: If True, injects a virtual "Zero-Shot" candidate at Rank 0.
    "MIRROR_ENABLE_R0": True,
    
    # enable_filtering: Master switch for removing candidates (Score=0).
    "MIRROR_ENABLE_FILTERING": True,
    
    # enable_redundancy_filter: Sub-switch. If True, removes lower-ranked candidates 
    # that are "covered" by higher-ranked ones (Parsimony).
    "MIRROR_ENABLE_REDUNDANCY_FILTER": True,
    
    # active_candidate_limit: Limits the process to the top-K retrieved samples to save API costs.
    # Only these candidates will be scored.
    "MIRROR_ACTIVE_CANDIDATE_LIMIT": 5,

    # --- MIRROR Track B: Feature Validation ---
    # NEW: Multi-Strategy Branching Switch
    # If True, the pipeline executes the "Functional Validation" track (Track B).
    # It solves the explicit "Base Filtered" list (list_3) and "Redundancy Filtered" list (list_4)
    # using a single pass (N=1) to verify solvable context.
    "MIRROR_EVALUATE_BASE_FILTERING": True,

    # --- Mirroring Prompts (Keys pointing to templates in src/prompts.py) ---
    "PROMPT_TEMPLATE_MIRROR_BASELINE": "mirror_baseline_zero_shot_v1",
    "PROMPT_TEMPLATE_MIRROR_HYPOTHESIS": "mirror_hypothesis_gen_v1",
    "PROMPT_TEMPLATE_MIRROR_VERIFICATION": "mirror_verification_v1",
    "PROMPT_TEMPLATE_MIRROR_HYPOTHESIS_ZEROSHOT": "mirror_hypothesis_gen_zero_shot_v1",
}

def setup_directories():
    """Creates the necessary directory structure."""
    print("--- Setting up project directories ---")
    for dir_path in [DATA_DIR, OUTPUTS_DIR, LOGS_DIR, EMBEDDINGS_DIR, RESULTS_DIR]:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Directory ensured: {dir_path}")
        except OSError as e:
            print(f"Error creating directory {dir_path}: {e}")
    print("--- Directory setup complete ---\n")