import os

# --- 1. Core Directory Structure ---
# Determine the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DATA_BASE = os.path.join(PROJECT_ROOT, "local_data")

BASE_OUTPUT_DIR = os.path.join(LOCAL_DATA_BASE, "outputs")
DATA_DIR = os.path.join(LOCAL_DATA_BASE, "data")
OUTPUTS_DIR = os.path.join(BASE_OUTPUT_DIR, "outputs")
LOGS_DIR = os.path.join(BASE_OUTPUT_DIR, "logs")
EMBEDDINGS_DIR = os.path.join(BASE_OUTPUT_DIR, "embeddings")
RESULTS_DIR = os.path.join(BASE_OUTPUT_DIR, "results")

CONFIG = {
    # Execution Mode
    "OFFLINE_MODE": True,  # Set to False for Kaggle/online execution
    
    # Logging & Control
    "VERBOSE_LOGGING": True,
    "PRINT_API_CALL_DETAILS": True,
    "PRINT_API_TIMING_CHECKPOINTS": True,
    "ENABLE_API_RETRY": True,          
    "MAX_API_RETRIES": 200,              
    "API_RETRY_DELAY_SECONDS": 20.0,
    "RETRY_ALL_API_ERRORS": True,  # If True, retry all API error types; if False, only retry RETRYABLE_ERROR_TYPES
    "API_RESPONSE_TRUNCATION_LENGTH": 50,
    "BASE_OUTPUT_DIR": BASE_OUTPUT_DIR,
    "DATA_DIR": DATA_DIR,
    "LOGS_DIR": LOGS_DIR,
    "OUTPUTS_DIR": OUTPUTS_DIR,
    "RESULTS_DIR": RESULTS_DIR,
    "EMBEDDINGS_DIR": EMBEDDINGS_DIR,

    # API Provider Selection
    "API_PROVIDER_ADAPTATION": "gemini",
    "API_PROVIDER_SOLVER": "gemini",
    "API_PROVIDER_EVALUATOR": "gemini",
    "API_PROVIDER_AUGMENTATION": "gemini",
    "API_PROVIDER_SIMPLIFICATION": "gemini",

    # Gemini API Settings
    "GEMINI_API_KEYS": [
    ],
    # Per-model rate-limit settings. Each entry may be a dict for all keys,
    # or a list of quota dicts with optional api_key values for key-specific settings.
    "GEMINI_MODEL_QUOTAS": {
        "models/gemma-3-27b-it": [{"api_key": None, "delay_seconds": 2, "rpd": 1000}],
    },
    "GLOBAL_API_CALL_DELAY_SECONDS": 5,

    "GEMINI_MODEL_NAME_ADAPTATION": "models/gemma-3-27b-it",
    "GEMINI_MODEL_NAME_FINAL_SOLVER": "models/gemma-3-27b-it",
    "GEMINI_MODEL_NAME_EVALUATOR": "models/gemma-3-27b-it",
    "GEMINI_MODEL_NAME_AUGMENTATION": "models/gemma-3-27b-it",
    "GEMINI_MODEL_NAME_SIMPLIFICATION": "models/gemma-3-27b-it",

    # --- AvalAI (OpenAI-Compatible) API Settings ---
    "AVALAI_API_KEY": "YOUR_AVALAI_API_KEY_HERE",
    # "AVALAI_BASE_URL": "https://api.avalai.ir/v1",
    "AVALAI_BASE_URL": "https://api.avalapis.ir/v1",
    # Per-model rate-limit settings. Use a dict for all keys, or a list of quota dicts
    # with optional api_key for per-key customization.
    "AVALAI_MODEL_QUOTAS": {
        "default": {"delay_seconds": 2}
    },
    "AVALAI_MODEL_NAME_ADAPTATION": "openai.gpt-oss-20b-1:0",
    "AVALAI_MODEL_NAME_FINAL_SOLVER": "openai.gpt-oss-20b-1:0",
    "AVALAI_MODEL_NAME_EVALUATOR": "openai.gpt-oss-20b-1:0",
    "AVALAI_MODEL_NAME_AUGMENTATION": "openai.gpt-oss-20b-1:0",
    "AVALAI_MODEL_NAME_SIMPLIFICATION": "openai.gpt-oss-20b-1:0",

    # Ollama (Local LLM) Settings
    "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
    "OLLAMA_THINK_MODE" : "low" ,
    "OLLAMA_MODEL_NAME_ADAPTATION": "gpt-oss:20b",
    "OLLAMA_MODEL_NAME_FINAL_SOLVER": "gpt-oss:20b",
    "OLLAMA_MODEL_NAME_EVALUATOR": "gpt-oss:20b",
    "OLLAMA_MODEL_NAME_AUGMENTATION": "llama3:8b",
    "OLLAMA_MODEL_NAME_SIMPLIFICATION": "llama3:8b",

    # Generic LLM Generation Settings
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

    # File Paths & Data
    # === HuggingFace Paths (used when downloading) ===
    "EMBEDDING_MODEL_PATH_HF": 'math-similarity/Bert-MLM_arXiv-MP-class_zbMath',
    "EXEMPLAR_CORPUS_NAME_HF": "AI-MO/NuminaMath-CoT",
    "EXEMPLAR_CORPUS_HF_TOKEN": None,
    "EXEMPLAR_EMBEDDINGS_HF_REPO_ID": "mostafabehroozi/embedding_NuminaMath_with_Bert-MLM_arXiv-MP-class_zbMath",
    "EXEMPLAR_EMBEDDINGS_HF_FILENAME": "embeddings.npy",
    
    # === Local Paths (used when running offline) ===
    "EMBEDDING_MODEL_PATH": 'math-similarity/Bert-MLM_arXiv-MP-class_zbMath',  # Will be updated in notebook
    "EXEMPLAR_CORPUS_NAME": "AI-MO/NuminaMath-CoT",  # Will be updated in notebook
    "LOCAL_EMBEDDING_MODEL_PATH": None,  # Will be computed by rebuild_derived_paths()
    "LOCAL_EXEMPLAR_CORPUS_PATH": None,  # Will be computed by rebuild_derived_paths()
    "USE_LOCAL_MODEL": False,  # Set to True to load from disk instead of downloading
    
    "HARD_QUESTIONS_INDICES_PATH": None,  # Will be computed by rebuild_derived_paths()
    "EMBEDDINGS_DIR": EMBEDDINGS_DIR,
    "EMBEDDED_EXEMPLAR_CORPUS_QUESTIONS_PATH": None,  # Will be computed by rebuild_derived_paths()
    
    "ADVANCED_RAG_FULL_LOG_PATH": None,  # Will be computed by rebuild_derived_paths()
    "ADVANCED_RAG_EVALUATION_RESULTS_PATH": None,  # Will be computed by rebuild_derived_paths()

    # Pipeline Control Flags
    "USE_RETRIEVAL": True,
    "PIPELINE_SEQUENCE": ["retrieve", "adapt", "merge", "solve"],
    "DEFER_SOLVE_STEP": False,
    "TOP_N_CANDIDATES_RETRIEVAL": 1,
    "FINAL_K_SELECTION_ADAPTATION": 1,
    "TARGET_ADAPTED_SAMPLES_MERGING": 1,

    # --- Reverse Transformation Flags ---
    "APPLY_REVERSE_TRANSFORMATION": False,  
    "REVERSE_TRANSFORMATION_ORDER": "after_retrieve",  # "after_retrieve" or "after_adapt"
    "REVERSE_TRANSFORMATION_COMBINED_WITH_ADAPT": "sequential",  # "sequential", "integrated", or "replace_adapt"
    "REVERSE_TRANSFORMATION_USE_TRANSFORMED_R": True,  # Use transformed R when other steps are combined with reverse transformation
    "REVERSE_TRANSFORMATION_TEMPERATURE": 0.3,  # Temperature for transformation step
    "REVERSE_TRANSFORMATION_SOLVER_TEMPERATURE": 1.0,  # Temperature for solving transformed questions
    "REVERSE_TRANSFORMATION_FINAL_SOLVER_TEMPLATE": "reverse_transformation_final_solve",  # Template for final solve
    "REVERSE_TRANSFORMATION_SKIP_MERGE": True,  # Skip merge step when reverse transformation is used (since RT provides complete solution)

    # Adaptation Steps
    "APPLY_NORMALIZATION": False,
    "APPLY_TRANSFORMATION": False,
    "APPLY_TRANSFORMATION_1": False,
    "APPLY_TRANSFORMATION_2": False,
    "APPLY_TRANSFORMATION_3": False,
    "APPLY_MERGING": False,

    # Self-Sampling
    "APPLY_SELF_SAMPLING": False,
    "SELF_SAMPLING_N": 3,
    "SELF_SAMPLING_TEMPERATURE": 1.0,

    # Analogical Adaptation
    "APPLY_ANALOGICAL_ADAPTATION": False,
    "ANALOGICAL_ADAPTATION_SAMPLING_N": 3,

    # Augmentation & Selection
    "APPLY_SELF_SAMPLING_AUGMENTATION": False,
    "APPLY_ANALOGICAL_ADAPTATION_AUGMENTATION": False,
    "ANALOGICAL_USE_MAIN_QUERY_AS_AUGMENTATION": False,
    "SELECTIVE_AUGMENTATION_SAMPLING": False,
    "AUGMENTATION_SCHEDULE": None,
    "AUGMENT_K": 10,
    "AUGMENT_N": 3,
    "SELECTIVE_AUGMENTATION_SAMPLING_MODE": "auto",

    # Consistency Check (Pathway)
    "APPLY_CONSISTENCY_ANALOGICAL_CHECK": False,
    "CONSISTENCY_GENERATION_MODE": "distinct_augmentations",
    "CONSISTENCY_PATHWAYS_K": 3,
    "CONSISTENCY_LAYER_1_TEMPERATURE": 1.0,
    "CONSISTENCY_SAMPLES_PER_PATHWAY_N": 3,
    "CONSISTENCY_LAYER_2_TEMPERATURE": 1.0,
    "CONSISTENCY_VOTING_THRESHOLD": 0.6,

    "APPLY_GROUP_CONSISTENCY_SELECTION": False,

    "GROUP_CONSISTENCY_CANDIDATES": [(0,), (0, 1), (0, 1, 2)],

    "GROUP_CONSISTENCY_SAMPLES_N": 10,

    # These legacy scoring flags can be ignored or set to None as we are 
    # now doing independent Pass@K benchmarking, not internal voting.
    "CONSISTENCY_SCORING_METHOD": "benchmark_report", 
    "SEMANTIC_CONSISTENCY_WEIGHT": 0.0,

    # Hierarchical Augmentation 
    "APPLY_HIERARCHICAL_AUGMENTATION": False,
    "HIERARCHICAL_AUGMENTATION_MODE": "decomposition", # "decomposition" or "simplification"
    "HIERARCHICAL_TREE_DEPTH": 2,
    "HIERARCHICAL_BRANCHING_FACTOR": 3,
    "HIERARCHICAL_LEAF_RETRIEVAL_ENABLED": True,
    "HIERARCHICAL_LEAF_RETRIEVAL_TOP_K": 3,
    "HIERARCHICAL_LEAF_RETRIEVAL_QUERY_MODE": "leaf", # "leaf" or "root"
    
    # Two-Step Augmentation 
    "HIERARCHICAL_AUGMENTATION_TWO_STEP": False,

    # Simplification Mode 
    "APPLY_SIMPLIFICATION": False,
    "SIMPLIFY_RETRIEVED_SAMPLES": False,
    "SIMPLIFY_MAIN_QUESTION": False,

    # Reverse Validation
    "APPLY_REVERSE_VALIDATION": False,
    "REVERSE_VALIDATION_CANDIDATES_N": 5,
    "REVERSE_VALIDATION_RETRIEVAL_K": 3,
    "REVERSE_VALIDATION_ATTEMPTS_N": 5,
    "REVERSE_VALIDATION_ENABLE_BASELINE_CHECK": True,

    "REVERSE_VALIDATION_USE_RAG_GENERATION": True,  # Turns ON the new helper/analogical generation
    "REVERSE_VALIDATION_GENERATION_K": 3,           # How many past examples to fetch to help generate the candidates

    "REVERSE_VALIDATION_ADD_ZEROSHOT_CANDIDATES": False,
    "REVERSE_VALIDATION_ZEROSHOT_CANDIDATES_N": 3,

    "APPLY_BEST_OF_TRANSFORMATION": False,
    "BEST_OF_TRANSFORMATION_N_SAMPLES": 3,  # N: transformations per retrieved sample
    "BEST_OF_TRANSFORMATION_ATTEMPTS_PER_TRANSFORMATION": 1,  # M: candidate attempts per transformation
    "BEST_OF_TRANSFORMATION_TRANSFORMATION_TEMPLATE": "transformation_shallow-&-moderately-deep",
    "BEST_OF_TRANSFORMATION_ENABLE_MIRROR_EVAL": False,  # Use mirror evaluation to score candidates
    "BEST_OF_TRANSFORMATION_MIRROR_EVAL_ATTEMPTS": 3,  # Quick validation attempts per candidate

    
    "APPLY_MULTIBRANCH_TRANSFORMATION": False,  # Master control flag
    
    # Branch Control: Which scenarios to execute
    "RUN_TX1_BASELINE": False,          # Scenario 1: Single transformation baseline
    "RUN_BOT_N_ONLY": True,            # Scenario 2: Best-of-N (exclusive)
    "RUN_BOT_N_PLUS_R": True,          # Scenario 3: Best-of-N+R (inclusive)
    
    # Centralized Pool Configuration
    "MULTIBRANCH_N_TRANSFORMATIONS": 3,                    # N: number of transformations
    "MULTIBRANCH_TRANSFORMATION_TEMPLATE": "transformation_shallow-&-moderately-deep",
    "MULTIBRANCH_TRANSFORMATION_TEMPERATURE": 0.0,         # Temperature for transformation step
    
    # Unified Scoring Configuration
    "MULTIBRANCH_ENABLE_MIRROR_SCORING": True,            # Use mirror-style evaluation
    "MULTIBRANCH_MIRROR_SCORING_ATTEMPTS": 3,             # Attempts per candidate scoring
    
    # Deterministic Tie-Breaking
    "MULTIBRANCH_TIEBREAK_FAVOR_ORIGINAL": True,          # On ties, prefer R_main (safety principle)
    "MULTIBRANCH_TIEBREAK_EPSILON": 1e-6,                # Threshold for considering scores "tied"
    
    # Solver Configuration (per branch)
    "MULTIBRANCH_SOLVER_TEMPERATURE": 1.0,                # Temperature for final solving
    "MULTIBRANCH_SOLVER_ATTEMPTS_PER_BRANCH": 3,          # Pass@K attempts per branch
    
    # Pass@N & Evaluation
    "N_PASS_ATTEMPTS": 3,
    "APPLY_FULL_PIPELINE_RETRY": False,
    "PASS_K_VALUES_TO_REPORT": [1, 2, 3, 4, 5],


    
    "APPLY_LAYER1_BASE_EXECUTION": False,  # Enable/disable Layer 1 system
    "LAYER1_ONLY_MODE": False,             # If True, halt after Layer 1 (cache generation only)
    "LAYER1_CACHE_DIR": RESULTS_DIR,
    "LAYER1_N_CANDIDATES": None,           # Number of candidates (None = use TOP_N_CANDIDATES_RETRIEVAL)
    "LAYER1_DATASET_NAME": "hard_questions",  # Dataset name for cache filename organization
    
    "APPLY_LAYER2_ANALYSIS": False,
    "LAYER2_CONFIG": {
        "layer2_config_name": "utility_calibration_modes_test",
        "run_block_A_baseline": True,   
        "run_block_A": True,
        "run_block_B": False,
        "utility_calibration_modes": ["Marginal", "Absolute"],
        "evaluator_masking": ["Self", "Others", "All"],
        "experimental_blocks": ["Block_A", "Block_B", "Block_C"],
        "scoring_strategies": ["Candidate_Centric", "Evaluator_Centric"]
    },

    "APPLY_DATASET_CONSTRUCTION": False,

    "APPLY_DATASET_CONSTRUCTION": False,
    "DATASET_CONSTRUCTION_MAX_SEARCH": 1000,    # how many random queries to examine
    "DATASET_CONSTRUCTION_MAX_MEMBERS": 100,    # stop when this many valid entries are gathered
    "DATASET_CONSTRUCTION_RANDOM_SEED": None,   # seed for reproducibility (optional)
    "DATASET_CONSTRUCTION_PROMPT_TEMPLATE": None,  # overrides PROMPT_TEMPLATE_FINAL_SOLVER
    "DATASET_CONSTRUCTION_SOLVER_TEMPERATURE": 1.0,
    "DATASET_CONSTRUCTION_EVALUATOR_TEMPERATURE": 0.0,

    # Prompt Templates
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
    "PROMPT_TEMPLATE_REVERSE_VALIDATION_BASELINE": "mirror_baseline_zero_shot_v1",
    "PROMPT_TEMPLATE_REVERSE_VALIDATION_ZERO_SHOT_SOLVER": "final_solver_simple_v1",

    "PROMPT_TEMPLATE_HIERARCHICAL_AUGMENTOR": "self_sampling_augmentor_decomposition_2",
    "PROMPT_TEMPLATE_HIERARCHICAL_PARENT_SOLVER": "hierarchical_parent_solver_v1",
    "PROMPT_TEMPLATE_HIERARCHICAL_LEAF_SOLVER": "final_solver_simple_v1",

    # Two-Step Augmentation Prompts
    "PROMPT_TEMPLATE_AUGMENTATION_STEP1_SOLVER": "final_solver_simple_v2",
    "PROMPT_TEMPLATE_AUGMENTATION_STEP2_GENERATOR": "self_sampling_augmentor_simplification_with_solution",

    "PROMPT_TEMPLATE_SIMPLIFICATION_GENERATOR": "simplification_generator_v1",
    "PROMPT_TEMPLATE_SIMPLIFIED_SAMPLE_SOLVER": "simplified_sample_solver_v1",
    "PROMPT_TEMPLATE_SIMPLIFIED_MAIN_PROXY_SOLVER": "main_from_simplified_proxy_v1",

    # Reverse Transformation Prompts
    "PROMPT_TEMPLATE_REVERSE_TRANSFORMATION_MAIN_TO_EXEMPLAR": "reverse_transformation_main_to_exemplar",
    "PROMPT_TEMPLATE_REVERSE_TRANSFORMATION_SOLVE_TRANSFORMED": "reverse_transformation_solve_transformed",
    "PROMPT_TEMPLATE_REVERSE_TRANSFORMATION_FINAL_SOLVE": "reverse_transformation_final_solve",

    # Hugging Face Hub
    "PERSIST_RESULTS_ONLINE": True,
    "HF_SYNC_TOKEN": "YOUR_HUGGING_FACE_TOKEN_HERE",
    "HF_HUB_USERNAME": "your-hf-username-here",
    "HF_HUB_REPO_NAME": "analogical-math-rag-results",
    "HF_SYNC_REVISION_ENABLED": False,
    "HF_SYNC_REVISION_ID": "main",
    "HF_SYNC_INTERVAL": 10,

    
    "APPLY_MIRROR_RERANKING": False,  # Enable/disable the unified re-ranking stage
    "MIRROR_RERANKING_APPLY_AFTER": "transformation",  # "transformation", "retrieval", or "both"
    
    # Re-ranking Configuration Parameters (apply to the unified stage)
    "MIRROR_RERANKING_N_OPTIMIZATION": 3,  # Number of Q_answered candidates to generate per demo
    "MIRROR_RERANKING_ENABLE_R0": False,  # Include zero-shot candidate (R0) in re-ranking
    "MIRROR_RERANKING_ENABLE_FILTERING": True,  # Filter out demos with TCS <= baseline similarity
    "MIRROR_RERANKING_ENABLE_REDUNDANCY_FILTER": True,  # Remove redundant/similar demos
    "MIRROR_RERANKING_EVALUATE_BASE_FILTERING": False,  # Use "base" vs "redundancy" filtering comparison
    "MIRROR_RERANKING_ACTIVE_LIMIT": None,  # Optional: limit demos to re-rank (None = use all)

    "APPLY_MIRROR_AS_EVALUATOR": True,

    "MIRROR_N_OPTIMIZATION": 3,
    "MIRROR_ENABLE_R0": True,
    "MIRROR_ENABLE_FILTERING": True,
    "MIRROR_ENABLE_REDUNDANCY_FILTER": True,
    "MIRROR_ACTIVE_CANDIDATE_LIMIT": 10,
    "MIRROR_EVALUATE_BASE_FILTERING": True,
    "PROMPT_TEMPLATE_MIRROR_BASELINE": "mirror_baseline_zero_shot_v1",
    "PROMPT_TEMPLATE_MIRROR_HYPOTHESIS": "mirror_hypothesis_gen_v1",
    "PROMPT_TEMPLATE_MIRROR_VERIFICATION": "mirror_verification_v1",
    "PROMPT_TEMPLATE_MIRROR_HYPOTHESIS_ZEROSHOT": "mirror_hypothesis_gen_zero_shot_v1",
}

# Initialize derived paths on module load (before any functions are called)
def _initialize_derived_paths():
    """Internal function to initialize derived paths when module is loaded."""
    CONFIG["LOCAL_EMBEDDING_MODEL_PATH"] = os.path.join(CONFIG["DATA_DIR"], "Bert-MLM_arXiv-MP-class_zbMath")
    CONFIG["LOCAL_EXEMPLAR_CORPUS_PATH"] = os.path.join(CONFIG["DATA_DIR"], "NuminaMath-CoT")
    CONFIG["HARD_QUESTIONS_INDICES_PATH"] = os.path.join(CONFIG["DATA_DIR"], "hard_question_indices.json")
    CONFIG["EMBEDDED_EXEMPLAR_CORPUS_QUESTIONS_PATH"] = os.path.join(CONFIG["EMBEDDINGS_DIR"], 'embedding_NuminaMath_with_Bert-MLM_arXiv-MP-class_zbMath.npy')
    CONFIG["ADVANCED_RAG_FULL_LOG_PATH"] = os.path.join(CONFIG["RESULTS_DIR"], "advanced_rag_pipeline_full_log.json")
    CONFIG["ADVANCED_RAG_EVALUATION_RESULTS_PATH"] = os.path.join(CONFIG["RESULTS_DIR"], "advanced_rag_evaluation_results.pkl")
    CONFIG["LAYER1_CACHE_DIR"] = CONFIG["RESULTS_DIR"]

_initialize_derived_paths()


def rebuild_derived_paths():
    """
    Rebuilds all derived paths based on current CONFIG directory settings.
    
    Call this function after updating CONFIG directory values to ensure
    all downstream paths are computed with the new base directories.
    
    This is essential for supporting multiple execution modes (offline/online, local/Kaggle).
    """
    CONFIG["LOCAL_EMBEDDING_MODEL_PATH"] = os.path.join(CONFIG["DATA_DIR"], "Bert-MLM_arXiv-MP-class_zbMath")
    CONFIG["LOCAL_EXEMPLAR_CORPUS_PATH"] = os.path.join(CONFIG["DATA_DIR"], "NuminaMath-CoT")
    CONFIG["HARD_QUESTIONS_INDICES_PATH"] = os.path.join(CONFIG["DATA_DIR"], "hard_question_indices.json")
    CONFIG["EMBEDDED_EXEMPLAR_CORPUS_QUESTIONS_PATH"] = os.path.join(CONFIG["EMBEDDINGS_DIR"], 'embedding_NuminaMath_with_Bert-MLM_arXiv-MP-class_zbMath.npy')
    CONFIG["ADVANCED_RAG_FULL_LOG_PATH"] = os.path.join(CONFIG["RESULTS_DIR"], "advanced_rag_pipeline_full_log.json")
    CONFIG["ADVANCED_RAG_EVALUATION_RESULTS_PATH"] = os.path.join(CONFIG["RESULTS_DIR"], "advanced_rag_evaluation_results.pkl")
    CONFIG["LAYER1_CACHE_DIR"] = CONFIG["RESULTS_DIR"]


def setup_directories():
    """Creates the necessary directory structure using CONFIG values."""
    # Ensure all derived paths are up-to-date
    rebuild_derived_paths()
    
    print("--- Setting up project directories ---")
    dirs_to_create = [
        CONFIG.get("DATA_DIR", DATA_DIR),
        CONFIG.get("OUTPUTS_DIR", OUTPUTS_DIR),
        CONFIG.get("LOGS_DIR", LOGS_DIR),
        CONFIG.get("EMBEDDINGS_DIR", EMBEDDINGS_DIR),
        CONFIG.get("RESULTS_DIR", RESULTS_DIR),
        CONFIG.get("LAYER1_CACHE_DIR", RESULTS_DIR),  # Updated
    ]

    for dir_path in dirs_to_create:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Directory ensured: {dir_path}")
        except OSError as e:
            print(f"Error creating directory {dir_path}: {e}")

    print("--- Directory setup complete ---\n")


def setup_kaggle_mode(outputs_base_dir="/kaggle/working"):
    """
    Configure CONFIG for Kaggle/online execution.
    """
    CONFIG["OFFLINE_MODE"] = False
    CONFIG["BASE_OUTPUT_DIR"] = outputs_base_dir
    CONFIG["DATA_DIR"] = os.path.join(outputs_base_dir, "data")
    CONFIG["OUTPUTS_DIR"] = os.path.join(outputs_base_dir, "outputs")
    
    # FIX: Removed the extra "outputs" subfolder so it matches Hugging Face downloads
    CONFIG["LOGS_DIR"] = os.path.join(outputs_base_dir, "logs")
    CONFIG["EMBEDDINGS_DIR"] = os.path.join(outputs_base_dir, "embeddings")
    CONFIG["RESULTS_DIR"] = os.path.join(outputs_base_dir, "results")
    
    # Rebuild all derived paths with the new base directories
    rebuild_derived_paths()
    
    # Create the directory structure
    setup_directories()
    
    print("Kaggle mode configured. Directories:")
    print(f"  DATA_DIR: {CONFIG['DATA_DIR']}")
    print(f"  LOGS_DIR: {CONFIG['LOGS_DIR']}")
    print(f"  EMBEDDINGS_DIR: {CONFIG['EMBEDDINGS_DIR']}")
    print(f"  RESULTS_DIR: {CONFIG['RESULTS_DIR']}\n")


def setup_offline_mode():
    """
    Configure CONFIG for offline/local execution (default mode).
    
    This function resets all directory paths to the default local project structure.
    """
    CONFIG["OFFLINE_MODE"] = True
    CONFIG["BASE_OUTPUT_DIR"] = BASE_OUTPUT_DIR
    CONFIG["DATA_DIR"] = DATA_DIR
    CONFIG["OUTPUTS_DIR"] = OUTPUTS_DIR
    CONFIG["LOGS_DIR"] = LOGS_DIR
    CONFIG["EMBEDDINGS_DIR"] = EMBEDDINGS_DIR
    CONFIG["RESULTS_DIR"] = RESULTS_DIR
    
    # Rebuild all derived paths with the new base directories
    rebuild_derived_paths()
    
    # Create the directory structure
    setup_directories()
    
    print("Offline mode configured. Directories:")
    print(f"  DATA_DIR: {CONFIG['DATA_DIR']}")
    print(f"  LOGS_DIR: {CONFIG['LOGS_DIR']}")
    print(f"  EMBEDDINGS_DIR: {CONFIG['EMBEDDINGS_DIR']}")
    print(f"  RESULTS_DIR: {CONFIG['RESULTS_DIR']}\n")
