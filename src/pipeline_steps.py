# src/pipeline_steps.py

"""
Core pipeline steps for the Analogical Reasoning RAG project.

This module contains the primary functions that constitute the RAG pipeline,
broken down into modular, sequential steps:
1.  retrieve: Finds relevant exemplars from the corpus.
2.  adapt: Transforms and/or summarizes the retrieved exemplars.
3.  merge: Iteratively combines adapted exemplars into a more potent one.
4.  solve: Generates the final answer using the processed exemplars.

This version is updated to handle structured, detailed API error responses.
When an API call fails, the step captures the error information and continues
where possible, allowing the orchestrator to log partial results and enable
targeted retries.

This version also includes new, optional pipeline steps for self-sampling,
augmentation, analogical adaptation, and the NEW Analogical Consistency check.

PERFORMANCE FIX: The retrieve() function now uses O(1) hash map lookup instead
of O(n) linear search for self-match detection, reducing retrieval time from
~1300 seconds to <0.001 seconds per query.

UPGRADE: Now supports Recursive Analogical Chains (Tree-structured context).
UPGRADE: Now supports Group-Based Self-Consistency Selection.
UPGRADE: Now supports Hierarchical Augmentation with Backward Propagation.
"""

import logging
import re
import numpy as np
import uuid
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union, Tuple, Optional
import time
from collections import deque

# NEW: Explicitly import CONFIG for use throughout this module.
# The request was to *not* use a config flag for diagnostics, but CONFIG
# itself is essential for pipeline logic in many functions.
from config import CONFIG

# Import our custom modules
from src.prompts import (
    EXEMPLAR_FORMAT,
    create_normalization_prompt,
    create_transformation_prompt,
    create_merging_prompt,
    create_final_reasoning_prompt,
    create_final_reasoning_prompt_simple,
    create_duplicate_check_prompt,
    create_self_sampling_prompt,
    create_augmentation_prompt,
    create_analogical_adaptation_prompt,
    create_hierarchical_parent_solver_prompt # NEW import
)
# MODIFIED: Import manager classes for type checking
from src.api_manager import GeminiAPIManager, AvalAIAPIManager, OllamaAPIManager


def log_time_diagnostic(message: str, start_time: float, indent: int = 0) -> float:
    """Logs a diagnostic message with elapsed time and returns the current time."""
    end_time = time.time()
    elapsed = end_time - start_time
    indent_str = "  " * indent
    if elapsed > 0.001: # Only print if elapsed time is significant
        print(f"{indent_str}⏱️  DIAGNOSTIC: {message} took {elapsed:.4f} seconds.")
    return end_time

# --- Utility Function for Embedding Generation ---
def _generate_embeddings(
    texts: List[str],
    embedding_model: SentenceTransformer,
    batch_size: int = 32
) -> np.ndarray:
    """Helper function to generate sentence embeddings."""
    if not isinstance(embedding_model, SentenceTransformer) or not texts:
        return np.array([])
    try:
        return embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to generate embeddings: {e}", exc_info=True)
        return np.array([])


# --- 1. RETRIEVAL STEP ---
def retrieve(
    target_query: str,
    embedding_model: SentenceTransformer,
    exemplar_questions: List[str],
    embedded_exemplars: np.ndarray,
    top_k: int,
    question_to_index_map: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    Retrieves the top_k most relevant exemplars for a target query.
    
    Args:
        target_query (str): The query to find similar exemplars for.
        embedding_model (SentenceTransformer): The model used to generate embeddings.
        exemplar_questions (List[str]): List of all exemplar questions.
        embedded_exemplars (np.ndarray): Pre-computed embeddings for all exemplars.
        top_k (int): Number of most similar exemplars to retrieve.
        question_to_index_map (Optional[Dict[str, int]]): Pre-computed hash map 
            for O(1) self-match detection. If None, self-match check is skipped.
    
    Returns:
        Dict[str, Any]: Dictionary containing retrieval status and retrieved indices.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting retrieval for Top-{top_k} exemplars.")
    
    # Initialize internal timer for granular diagnostics
    current_diag_time = time.time() 
    indent_level = 3 # Consistent indentation for diagnostics within retrieve
    
    # --- DETAILED RETRIEVAL DIAGNOSTICS - ALWAYS ON ---
    print(f"{'  '*indent_level}--- STARTING DETAILED RETRIEVAL DIAGNOSTICS ---")
    print(f"{'  '*indent_level}Target Query (start): '{target_query[:100]}...'")
    print(f"{'  '*indent_level}Exemplar corpus size: {len(exemplar_questions)}")
    print(f"{'  '*indent_level}Embedded exemplars shape: {embedded_exemplars.shape}")
    print(f"{'  '*indent_level}Requested top_k: {top_k}")
    # --- END ALWAYS ON DIAGNOSTICS ---


    # Step 1: Generate query embedding
    query_embedding_start_time = time.time()
    query_embedding = _generate_embeddings([target_query], embedding_model)
    current_diag_time = log_time_diagnostic("Generate query embedding", query_embedding_start_time, indent=indent_level)
    print(f"{'  '*indent_level}Query embedding shape: {query_embedding.shape}")
    
    if query_embedding.size == 0:
        logger.error("Failed to generate embedding for the target query. Retrieval cannot proceed.")
        return {"status": "FAILURE", "retrieved_indices": [], "retrieved_exemplars": []}
    
    # Step 2: Calculate cosine similarity
    cosine_similarity_start_time = time.time()
    print(f"{'  '*indent_level}Starting cosine similarity calculation (query_embedding shape: {query_embedding.shape}, embedded_exemplars shape: {embedded_exemplars.shape})...")
    similarities = cosine_similarity(query_embedding, embedded_exemplars)[0]
    current_diag_time = log_time_diagnostic("Calculate cosine_similarity", cosine_similarity_start_time, indent=indent_level)
    print(f"{'  '*indent_level}Similarities array shape: {similarities.shape}")
    
    # Step 3: Handle potential self-match using O(1) hash map lookup
    self_match_start_time = time.time()

    if question_to_index_map is not None:
        # Use the pre-computed hash map for instant lookup
        query_index_in_corpus = question_to_index_map.get(target_query)
        
        if query_index_in_corpus is not None:
            # Self-match found - exclude it from results
            similarities[query_index_in_corpus] = -np.inf
            print(f"{'  '*indent_level}Self-match found at index {query_index_in_corpus}, set to -np.inf.")
        else:
            # Query not in corpus (expected for test queries)
            print(f"{'  '*indent_level}Target query not found in corpus (no self-match to remove).")
    else:
        # Fallback if hash map not provided (backward compatibility)
        print(f"{'  '*indent_level}Warning: question_to_index_map not provided. Skipping self-match check.")
        logger.warning("retrieve() called without question_to_index_map. Self-match detection skipped.")

    current_diag_time = log_time_diagnostic("Handle self-match (O(1) lookup)", self_match_start_time, indent=indent_level)


    # Step 4: Determine k_to_retrieve (ensure it's not more than available)
    k_retrieve_start_time = time.time()
    k_to_retrieve = min(top_k, len(similarities))
    current_diag_time = log_time_diagnostic("Determine k_to_retrieve", k_retrieve_start_time, indent=indent_level)
    print(f"{'  '*indent_level}Effective k_to_retrieve: {k_to_retrieve}")


    # --- START OF GRANULAR TIMING FOR TOP-K SELECTION LOGIC ---
    print(f"{'  '*indent_level}Starting granular timing for top-k selection...")
        
    # 4a. Perform partial sort using np.argpartition
    argpartition_start_time = time.time()
    print(f"{'  '*indent_level}  Calling np.argpartition on similarities array (shape: {similarities.shape}) for k={k_to_retrieve}...")
    
    # This gets indices of the k-th smallest element and elements smaller than it
    # We need the k largest, so we partition around the (N-k)-th smallest index.
    # This will put the top_k_indices in the *last* k positions of the returned array.
    partitioned_indices = np.argpartition(similarities, -k_to_retrieve)
    current_diag_time = log_time_diagnostic("np.argpartition (full array)", argpartition_start_time, indent=indent_level)
    print(f"{'  '*indent_level}  Resulting partitioned_indices shape: {partitioned_indices.shape}")
    # print(f"{'  '*indent_level}  Partitioned indices (first 10): {partitioned_indices[:10]}...") # Optional: if you want to see raw partitioned values


    # 4b. Slice the top K indices (these are still unsorted among themselves)
    slice_partitioned_start_time = time.time()
    print(f"{'  '*indent_level}  Slicing to get the top {k_to_retrieve} indices from partitioned_indices...")
    
    top_k_indices_unsorted = partitioned_indices[-k_to_retrieve:]
    current_diag_time = log_time_diagnostic("Slicing partitioned indices", slice_partitioned_start_time, indent=indent_level)
    print(f"{'  '*indent_level}  top_k_indices_unsorted shape: {top_k_indices_unsorted.shape}, Content (first 5): {top_k_indices_unsorted[:5]}...")

    # 4c. Sort the sliced top K indices based on their actual similarity values
    argsort_slice_start_time = time.time()
    print(f"{'  '*indent_level}  Calling np.argsort on the {k_to_retrieve} selected indices based on their similarities...")
    print(f"{'  '*indent_level}  Accessing similarities values for sorting (similarities[top_k_indices_unsorted])...")

    # Get the actual similarity values for the top_k_indices_unsorted
    relevant_similarities = similarities[top_k_indices_unsorted]
    
    # Sort these values to get the order, then apply that order to the indices
    sorted_order_in_slice = np.argsort(relevant_similarities)[::-1] # [::-1] for descending order
    
    top_k_indices = top_k_indices_unsorted[sorted_order_in_slice]

    current_diag_time = log_time_diagnostic("np.argsort & final sort (on small slice)", argsort_slice_start_time, indent=indent_level)
    print(f"{'  '*indent_level}  Final top_k_indices shape: {top_k_indices.shape}, Content: {top_k_indices.tolist()}")
    print(f"{'  '*indent_level}--- ENDING DETAILED RETRIEVAL DIAGNOSTICS ---")
    # --- END OF GRANULAR TIMING ---

    logger.info(f"Successfully retrieved indices: {top_k_indices.tolist()}")
    
    return {
        "status": "SUCCESS",
        "retrieved_indices": top_k_indices.tolist(),
    }


# --- 2. ADAPTATION STEP (REWRITTEN) ---
def adapt(
    target_query: str,
    retrieved_indices: List[int],
    exemplar_questions: List[str],
    exemplar_solutions: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Performs a multi-stage adaptation on retrieved exemplars:
    Normalization -> Transformation 1 -> Transformation 2 -> Transformation 3.
    Captures failures for individual exemplars without halting the entire step.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting multi-stage adaptation step.")
    
    successful_texts = []
    failed_adaptations = []
    
    # MODIFIED: Determine model name based on the type of the provided API manager
    if isinstance(api_manager, GeminiAPIManager):
        model_name = config['GEMINI_MODEL_NAME_ADAPTATION']
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config['AVALAI_MODEL_NAME_ADAPTATION']
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config['OLLAMA_MODEL_NAME_ADAPTATION']
    else:
        raise TypeError(f"Unsupported API manager type for adaptation: {type(api_manager)}")
        
    temperature = config['DEFAULT_ADAPTATION_TEMPERATURE']

    for idx in retrieved_indices:
        original_question = exemplar_questions[idx]
        original_solution = exemplar_solutions[idx]
        current_text = EXEMPLAR_FORMAT.format(question=original_question, solution=original_solution)
        
        step_failed = False

        # --- Step 1: Normalization (formerly Standardization) ---
        if config.get('APPLY_NORMALIZATION', False) and not step_failed:
            logger.info(f"Applying normalization to exemplar index {idx}.")
            print(f"    -> Normalizing exemplar {idx}...")
            prompt = create_normalization_prompt(current_text)
            
            print(f"      [API Context] Calling LLM for: Normalization (Exemplar #{idx})")
            response = api_manager.generate_content(prompt, model_name, temperature)
            
            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                logger.warning(f"Normalization failed for exemplar {idx}: {response['error_message']}")
                failed_adaptations.append({"source_index": idx, "failed_at_step": "normalization", "error_info": response})
                step_failed = True

        # --- Step 2: Transformation 1 ---
        if config.get('APPLY_TRANSFORMATION_1', False) and not step_failed:
            logger.info(f"Applying transformation 1 to exemplar index {idx}.")
            print(f"    -> Applying Transformation 1 to exemplar {idx}...")
            prompt = create_transformation_prompt(target_query, current_text, config, "PROMPT_TEMPLATE_TRANSFORMATION_1")
            print(f"      [API Context] Calling LLM for: Transformation 1 (Exemplar #{idx})")
            response = api_manager.generate_content(prompt, model_name, temperature)

            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                logger.warning(f"Transformation 1 failed for exemplar {idx}: {response['error_message']}")
                failed_adaptations.append({"source_index": idx, "failed_at_step": "transformation_1", "error_info": response})
                step_failed = True
        
        # --- Step 3: Transformation 2 ---
        if config.get('APPLY_TRANSFORMATION_2', False) and not step_failed:
            logger.info(f"Applying transformation 2 to exemplar index {idx}.")
            print(f"    -> Applying Transformation 2 to exemplar {idx}...")
            prompt = create_transformation_prompt(target_query, current_text, config, "PROMPT_TEMPLATE_TRANSFORMATION_2")
            print(f"      [API Context] Calling LLM for: Transformation 2 (Exemplar #{idx})")
            response = api_manager.generate_content(prompt, model_name, temperature)

            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                logger.warning(f"Transformation 2 failed for exemplar {idx}: {response['error_message']}")
                failed_adaptations.append({"source_index": idx, "failed_at_step": "transformation_2", "error_info": response})
                step_failed = True

        # --- Step 4: Transformation 3 ---
        if config.get('APPLY_TRANSFORMATION_3', False) and not step_failed:
            logger.info(f"Applying transformation 3 to exemplar index {idx}.")
            print(f"    -> Applying Transformation 3 to exemplar {idx}...")
            prompt = create_transformation_prompt(target_query, current_text, config, "PROMPT_TEMPLATE_TRANSFORMATION_3")
            print(f"      [API Context] Calling LLM for: Transformation 3 (Exemplar #{idx})")
            response = api_manager.generate_content(prompt, model_name, temperature)

            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                logger.warning(f"Transformation 3 failed for exemplar {idx}: {response['error_message']}")
                failed_adaptations.append({"source_index": idx, "failed_at_step": "transformation_3", "error_info": response})
                step_failed = True
        
        if not step_failed:
            successful_texts.append(current_text)

    # Determine final status based on outcomes
    if not retrieved_indices:
        final_status = "SUCCESS"
    elif not successful_texts and failed_adaptations:
        final_status = "FAILURE"
    elif successful_texts and failed_adaptations:
        final_status = "PARTIAL_SUCCESS"
    else:
        final_status = "SUCCESS"

    return {
        "status": final_status,
        "adapted_texts": successful_texts,
        "failed_adaptations": failed_adaptations
    }


# --- 3. MERGING STEP ---
def merge(
    target_query: str,
    adapted_texts: List[str],
    embedding_model: SentenceTransformer,
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Iteratively merges adapted exemplars. If a merge fails, the pair is discarded
    and the process continues.
    """
    logger = logging.getLogger(__name__)
    target_count = config.get('TARGET_ADAPTED_SAMPLES_MERGING', 1)

    if not config.get('APPLY_MERGING', False):
        logger.info("APPLY_MERGING is False. Skipping merge step.")
        return {"status": "SKIPPED", "merged_texts": adapted_texts[:target_count], "failed_merges": []}

    logger.info("Starting merging step.")
    current_texts = list(adapted_texts)
    failed_merges = []
    
    # MODIFIED: Determine model name based on the type of the provided API manager
    if isinstance(api_manager, GeminiAPIManager):
        model_name = config['GEMINI_MODEL_NAME_ADAPTATION']
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config['AVALAI_MODEL_NAME_ADAPTATION']
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config['OLLAMA_MODEL_NAME_ADAPTATION']
    else:
        raise TypeError(f"Unsupported API manager type for merging: {type(api_manager)}")
        
    temperature = config['DEFAULT_ADAPTATION_TEMPERATURE']
    
    iteration = 0
    while len(current_texts) > target_count and len(current_texts) >= 2:
        iteration += 1
        logger.info(f"Merge iteration {iteration}: Merging from {len(current_texts)} samples.")
        print(f"    -> Merging {len(current_texts)} samples down...")
        
        pair_to_merge = [current_texts.pop(0), current_texts.pop(0)]
        
        prompt = create_merging_prompt(target_query, pair_to_merge)
        if "Error:" in prompt:
            logger.error(f"Failed to create merging prompt: {prompt}")
            failed_merges.append({"pair_to_merge": pair_to_merge, "error_info": {"error_message": "Prompt creation failed."}})
            continue
            
        print(f"      [API Context] Calling LLM for: Merging (Iteration #{iteration})")
        response = api_manager.generate_content(prompt, model_name, temperature)
        
        if response['status'] == 'SUCCESS':
            current_texts.append(response['text'])
        else:
            logger.warning(f"Merging failed: {response['error_message']}. Discarding pair.")
            failed_merges.append({"pair_to_merge": pair_to_merge, "error_info": response})

    return {"status": "SUCCESS", "merged_texts": current_texts, "failed_merges": failed_merges}


# --- 4. SOLVER STEP ---
def solve(
    target_query: str,
    final_exemplars: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates final solution(s). If an attempt fails due to an API error,
    the error details are saved instead of a text solution for that attempt.
    
    MODIFIED: Can also be used to run a classification task like duplicate checking.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting final solver step.")
    
    # --- NEW: LOGIC FOR DUPLICATE QUESTION CHECKING ---
    final_solver_prompt_name = config.get("PROMPT_TEMPLATE_FINAL_SOLVER")
    if final_solver_prompt_name == "duplicate_question_check_v1":
        logger.info("Running in 'Duplicate Question Check' mode.")
        
        retrieved_questions = []
        for exemplar_text in final_exemplars:
            match = re.search(r"Question:\s*(.*?)\s*Rationale and Answer:", exemplar_text, re.DOTALL)
            if match:
                retrieved_questions.append(match.group(1).strip())
        
        if not retrieved_questions:
            logger.warning("Duplicate check mode ran but no retrieved questions were found to check.")
            return {"status": "SUCCESS", "solution_attempts": ["no_retrieval"]}

        prompt = create_duplicate_check_prompt(target_query, retrieved_questions)
        
        # MODIFIED: Determine model name based on manager type (uses adaptation model)
        if isinstance(api_manager, GeminiAPIManager):
            model_name = config['GEMINI_MODEL_NAME_ADAPTATION']
        elif isinstance(api_manager, AvalAIAPIManager):
            model_name = config['AVALAI_MODEL_NAME_ADAPTATION']
        elif isinstance(api_manager, OllamaAPIManager):
            model_name = config['OLLAMA_MODEL_NAME_ADAPTATION']
        else:
            raise TypeError(f"Unsupported API manager type for duplicate check: {type(api_manager)}")
            
        temperature = 0.0 # Low temp for deterministic classification
        
        print("    -> Checking for duplicate questions...")
        print("      [API Context] Calling LLM for: Duplicate Check")
        response = api_manager.generate_content(prompt, model_name, temperature)

        if response['status'] == 'SUCCESS':
            classification = response['text'].strip().lower()
            if "yes" in classification:
                result = "yes"
            elif "no" in classification:
                result = "no"
            else:
                result = "parsing_failed"
            return {"status": "SUCCESS", "solution_attempts": [result]}
        else:
            # If the API call fails, log the failure details
            return {"status": "FAILURE", "solution_attempts": [{"status": "FAILURE", "error_info": response}]}
    # --- END OF NEW LOGIC ---

    # --- Original Solver Logic ---
    prompt = create_final_reasoning_prompt(target_query, final_exemplars, config) if final_exemplars else create_final_reasoning_prompt_simple(target_query, config)
    logger.info(f"Using {'retrieval-augmented' if final_exemplars else 'simple'} prompt for the solver.")

    if "Error:" in prompt:
        error_msg = f"Failed to create final reasoning prompt: {prompt}"
        logger.error(error_msg)
        return {"status": "FAILURE", "solution_attempts": [{"status": "FAILURE", "error_info": {"error_message": error_msg}}]}

    n_attempts = config.get("N_PASS_ATTEMPTS", 1)
    
    # MODIFIED: Determine model name based on the type of the provided API manager
    if isinstance(api_manager, GeminiAPIManager):
        model_name = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config['AVALAI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config['OLLAMA_MODEL_NAME_FINAL_SOLVER']
    else:
        raise TypeError(f"Unsupported API manager type for solver: {type(api_manager)}")
        
    temperature = config.get('DEFAULT_PASS_N_SOLVER_TEMPERATURE', 1.0)
    
    solution_attempts: List[Union[str, Dict]] = []
    
    logger.info(f"Generating {n_attempts} solution attempts for Pass@{n_attempts}.")
    for i in range(n_attempts):
        logger.info(f"Generating attempt {i+1}/{n_attempts}.")
        print(f"    -> Generating solution attempt {i+1}/{n_attempts}...")
        
        print(f"      [API Context] Calling LLM for: Final Solution (Attempt #{i+1})")
        
        response = api_manager.generate_content(prompt, model_name, temperature)
        
        if response['status'] == 'SUCCESS':
            solution_attempts.append(response['text'])
        else:
            solution_attempts.append({
                "status": "FAILURE",
                "error_info": response
            })
            
    return {"status": "SUCCESS", "solution_attempts": solution_attempts}


# --- 5. NEW FEATURES: Self-Sampling, Augmentation, and Analogical Adaptation ---

def self_sample(
    target_query: str,
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates N synthetic exemplars by solving the target query N times.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting self-sampling step.")
    
    n_samples = config.get("SELF_SAMPLING_N", 3)
    temperature = config.get("SELF_SAMPLING_TEMPERATURE", 0.7)
    
    if isinstance(api_manager, GeminiAPIManager):
        model_name = config['GEMINI_MODEL_NAME_ADAPTATION']
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config['AVALAI_MODEL_NAME_ADAPTATION']
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config['OLLAMA_MODEL_NAME_ADAPTATION']
    else:
        raise TypeError(f"Unsupported API manager type for self-sampling: {type(api_manager)}")

    prompt = create_self_sampling_prompt(target_query, config)
    
    successful_texts = []
    failed_samples = []
    
    for i in range(n_samples):
        print(f"    -> Generating self-sample {i+1}/{n_samples} for query '{target_query[:50]}...'")
        response = api_manager.generate_content(prompt, model_name, temperature)
        
        if response['status'] == 'SUCCESS':
            # Format as standard exemplar
            formatted_text = f"Question: {target_query}\nRationale and Answer: {response['text']}"
            successful_texts.append(formatted_text)
        else:
            failed_samples.append({"sample_index": i, "error_info": response})
    
    if not successful_texts and failed_samples: status = "FAILURE"
    elif successful_texts and failed_samples: status = "PARTIAL_SUCCESS"
    else: status = "SUCCESS"
    
    return {"status": status, "self_sampled_texts": successful_texts, "failed_samples": failed_samples}

def parse_numbered_questions(text: str) -> List[str]:
    """Helper to parse a numbered list of questions from an LLM response."""
    questions = []
    # Regex to find lines starting with a number, period, and optional space
    matches = re.findall(r'^\s*\d+\.\s*(.*)', text, re.MULTILINE)
    for match in matches:
        questions.append(match.strip())
    return questions

def augment_question(
    target_query: str,
    n_augmentations: int,
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates N augmented versions of the target query.
    
    Supports a multi-call schedule via the `AUGMENTATION_SCHEDULE` config.
    If the schedule is defined (e.g., [2, 3]), it will make 2 API calls,
    each requesting 3 questions. Otherwise, it falls back to a single call
    for `n_augmentations` questions.
    """
    logger = logging.getLogger(__name__)

    # --- Determine Model and Temperature ---
    if isinstance(api_manager, GeminiAPIManager):
        model_name = config['GEMINI_MODEL_NAME_ADAPTATION']
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config['AVALAI_MODEL_NAME_ADAPTATION']
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config['OLLAMA_MODEL_NAME_ADAPTATION']
    else:
        raise TypeError(f"Unsupported API manager type for augmentation: {type(api_manager)}")
    
    temperature = config.get("DEFAULT_ADAPTATION_TEMPERATURE", 0.0)
    
    # --- Check for Augmentation Schedule ---
    schedule = config.get("AUGMENTATION_SCHEDULE")

    # --- Path 1: Scheduled, Multi-Call Augmentation ---
    if isinstance(schedule, list) and len(schedule) == 2:
        num_calls, questions_per_call = schedule
        logger.info(f"Using augmentation schedule: {num_calls} calls, {questions_per_call} questions per call.")
        
        all_augmented_questions = []
        failed_calls = []

        for i in range(num_calls):
            print(f"    -> Generating augmented questions (Call {i+1}/{num_calls})...")
            prompt = create_augmentation_prompt(target_query, questions_per_call, config)
            response = api_manager.generate_content(prompt, model_name, temperature)

            if response['status'] == 'SUCCESS':
                parsed_qs = parse_numbered_questions(response['text'])
                if len(parsed_qs) < questions_per_call:
                    logger.warning(f"Augmentation call {i+1} expected {questions_per_call} questions, but only parsed {len(parsed_qs)}.")
                all_augmented_questions.extend(parsed_qs)
            else:
                logger.error(f"Augmentation call {i+1}/{num_calls} failed: {response['error_message']}")
                failed_calls.append({"call_index": i + 1, "error_info": response})

        # Determine the final status based on the outcomes
        status = "SUCCESS"
        if failed_calls and not all_augmented_questions:
            status = "FAILURE"
        elif failed_calls:
            status = "PARTIAL_SUCCESS"
            
        return {
            "status": status, 
            "augmented_questions": all_augmented_questions, 
            "failed_calls": failed_calls
        }

    # --- Path 2: Fallback, Single-Call Augmentation ---
    else:
        logger.info(f"Generating {n_augmentations} augmented questions in a single call.")
        prompt = create_augmentation_prompt(target_query, n_augmentations, config)
        
        print(f"    -> Generating {n_augmentations} augmented questions...")
        response = api_manager.generate_content(prompt, model_name, temperature)
        
        if response['status'] != 'SUCCESS':
            return {"status": "FAILURE", "augmented_questions": [], "error_info": response}
        
        augmented_questions = parse_numbered_questions(response['text'])
        
        if len(augmented_questions) < n_augmentations:
            logger.warning(f"Augmentation expected {n_augmentations} questions, but only parsed {len(augmented_questions)}.")
        
        return {"status": "SUCCESS", "augmented_questions": augmented_questions, "error_info": None}

def _select_diverse_questions(questions: List[str], embeddings: np.ndarray, n: int) -> List[str]:
    """Selects n questions with the lowest average pairwise similarity."""
    if embeddings.shape[0] < n: return questions
    
    similarity_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_matrix, 0)
    avg_similarities = similarity_matrix.mean(axis=1)
    selected_indices = np.argsort(avg_similarities)[:n]
    return [questions[i] for i in selected_indices]

def _select_relevant_questions(aug_questions: List[str], aug_embeddings: np.ndarray, sample_embeddings: np.ndarray, n: int) -> List[str]:
    """Selects n augmented questions most relevant to retrieved samples."""
    if aug_embeddings.shape[0] < n: return aug_questions
    
    cross_similarity = cosine_similarity(aug_embeddings, sample_embeddings)
    max_similarities = cross_similarity.max(axis=1)
    selected_indices = np.argsort(max_similarities)[-n:][::-1] # Top N scores
    return [aug_questions[i] for i in selected_indices]

def select_augmented_questions(
    augmented_questions: List[str],
    config: Dict[str, Any],
    embedding_model: SentenceTransformer,
    retrieved_sample_texts: Optional[List[str]] = None
) -> List[str]:
    """
    Selects N best augmented questions from a larger pool based on config mode.
    """
    logger = logging.getLogger(__name__)
    target_n = config['AUGMENT_N']
    mode = config['SELECTIVE_AUGMENTATION_SAMPLING_MODE']

    if len(augmented_questions) <= target_n:
        return augmented_questions

    aug_embeddings = _generate_embeddings(augmented_questions, embedding_model)
    if aug_embeddings.size == 0:
        logger.error("Failed to generate embeddings for augmented questions. Cannot perform selection.")
        return augmented_questions[:target_n]
    
    if mode == "diversity" or (mode == "auto" and retrieved_sample_texts is None):
        logger.info(f"Selecting {target_n} most DIVERSE augmented questions.")
        return _select_diverse_questions(augmented_questions, aug_embeddings, target_n)
    
    elif mode == "relevance" or (mode == "auto" and retrieved_sample_texts is not None):
        logger.info(f"Selecting {target_n} most RELEVANT augmented questions.")
        sample_embeddings = _generate_embeddings(retrieved_sample_texts, embedding_model)
        if sample_embeddings.size == 0:
            logger.error("Failed to generate embeddings for samples. Falling back to diversity selection.")
            return _select_diverse_questions(augmented_questions, aug_embeddings, target_n)
        return _select_relevant_questions(augmented_questions, aug_embeddings, sample_embeddings, target_n)
    
    return augmented_questions[:target_n]

# --- RECURSIVE ANALOGICAL ADAPTATION LOGIC ---

def _count_processing_nodes(structure: Any) -> int:
    """
    Recursively counts the number of processing nodes in the group structure.
    A processing node is any list or tuple. Integers (leaves) are not processing nodes.
    
    Args:
        structure: The nested list/tuple structure from ANALOGICAL_GROUP_SETS.
        
    Returns:
        int: The total number of tuples/lists found.
    """
    count = 0
    if isinstance(structure, (list, tuple)):
        # If it's a container, it counts as 1 node (unless it is the top-level list container itself,
        # but since this function is called on elements OF the top list, it works correctly).
        # We treat the top-level iteration separately in analogical_adapt.
        # Here we are counting nodes inside a Group definition.
        count = 1 
        for item in structure:
            count += _count_processing_nodes(item)
            # We subtract the count for the item itself if it was a list/tuple because
            # the recursive call added 1 for it. Wait, no.
            # Example: ((1), 2)
            # Root is tuple: count = 1
            # Item 1: (1). Recursive call: count = 1 + recurse(1). recurse(1) is 0. So 1.
            # Item 2: 2. Recursive call: 0.
            # Total = 1 (root) + 1 (child tuple) + 0 = 2. Correct.
            
            # However, the top-level ANALOGICAL_GROUP_SETS is a list of groups.
            # We shouldn't count the container list itself if we iterate over it.
            # This helper is best designed to count nodes within a SINGLE group definition.
    
    return count

def _process_node_recursively(
    node: Any,
    aug_q_queue: deque,
    retrieved_texts_map: Dict[int, str],
    api_manager: Any,
    config: Dict[str, Any],
    depth: int = 0
) -> Union[str, None]:
    """
    Recursively processes a node in the analogical group structure.
    
    Args:
        node: An int (leaf), or tuple/list (processing node).
        aug_q_queue: Queue of pre-generated augmented questions.
        retrieved_texts_map: Map of 1-based indices to retrieved exemplar texts.
        api_manager: The API manager instance.
        config: Global config.
        depth: Current recursion depth for logging.
        
    Returns:
        str: The generated exemplar text (Question + Rationale), or None on failure.
    """
    indent = "  " * (depth + 2)
    
    # --- Base Case: Leaf Node (Integer) ---
    if isinstance(node, int):
        text = retrieved_texts_map.get(node)
        if not text:
            logging.getLogger(__name__).warning(f"{indent}Index {node} not found in retrieved map.")
            return None
        return text

    # --- Recursive Step: Processing Node (Tuple/List) ---
    elif isinstance(node, (list, tuple)):
        # 1. Process Children
        child_exemplars = []
        for child in node:
            child_result = _process_node_recursively(child, aug_q_queue, retrieved_texts_map, api_manager, config, depth + 1)
            if child_result:
                child_exemplars.append(child_result)
            else:
                # If a child fails, this node generally cannot proceed accurately.
                # However, we might continue with partial context. For strictness, let's log.
                logging.getLogger(__name__).warning(f"{indent}Child node {child} failed or returned None.")

        # 2. Get Augmented Question
        if not aug_q_queue:
            error_msg = "Augmented question queue exhausted! Check AUGMENT_K vs Structure complexity."
            logging.getLogger(__name__).error(error_msg)
            return None
        
        current_aug_q = aug_q_queue.popleft()
        print(f"{indent}-> Processing Node at depth {depth}. Context: {len(child_exemplars)} samples. solving AugQ: '{current_aug_q[:30]}...'")

        # 3. Setup Model & Temp
        if isinstance(api_manager, GeminiAPIManager): model_name = config['GEMINI_MODEL_NAME_ADAPTATION']
        elif isinstance(api_manager, AvalAIAPIManager): model_name = config['AVALAI_MODEL_NAME_ADAPTATION']
        elif isinstance(api_manager, OllamaAPIManager): model_name = config['OLLAMA_MODEL_NAME_ADAPTATION']
        else: return None
        
        # 4. Generate Content
        if not child_exemplars:
            # Empty Group -> Self-Solve Mode
            # We use the self-sampling prompt logic
            prompt = create_self_sampling_prompt(current_aug_q, config)
            temp = config.get("SELF_SAMPLING_TEMPERATURE", 0.7)
        else:
            # Occupied Group -> Analogical Adaptation Mode
            prompt = create_analogical_adaptation_prompt(current_aug_q, child_exemplars, config)
            temp = config.get("DEFAULT_ANALOGICAL_ADAPTATION_TEMPERATURE", 1.0)
            
        response = api_manager.generate_content(prompt, model_name, temp)
        
        if response['status'] == 'SUCCESS':
            # The prompts usually return just the solution/rationale part (depending on template).
            # We must wrap it to make it a valid exemplar for the parent node.
            # Note: create_self_sampling_prompt returns "Solution: ... Final Answer: ...".
            # We standardize to "Question: ... Rationale and Answer: ..."
            return f"Question: {current_aug_q}\nRationale and Answer: {response['text']}"
        else:
            logging.getLogger(__name__).warning(f"{indent}Generation failed for node at depth {depth}.")
            return None

    return None


def analogical_adapt(
    target_query: str,
    retrieved_indices: List[int],
    exemplar_data: Dict[str, Any],
    api_manager: Any,
    config: Dict[str, Any],
    embedding_model: SentenceTransformer,
    augmented_questions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Performs recursive analogical reasoning on groups of retrieved samples.
    
    Supports nested/recursive group structures defined in ANALOGICAL_GROUP_SETS.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting recursive analogical adaptation step.")
    
    group_sets = config.get("ANALOGICAL_GROUP_SETS", [])
    if not group_sets:
        logger.warning("ANALOGICAL_GROUP_SETS is empty. Skipping.")
        return {"status": "SKIPPED", "reason": "No groups defined."}

    # 1. Build Map for O(1) Retrieval Access
    # Indices in config are 1-based, so we map i+1 -> text
    retrieved_texts_map = {}
    for i, idx in enumerate(retrieved_indices):
        q = exemplar_data['questions'][idx]
        s = exemplar_data['solutions'][idx]
        retrieved_texts_map[i + 1] = EXEMPLAR_FORMAT.format(question=q, solution=s)

    # 2. Calculate Required Augmented Questions
    total_nodes_needed = 0
    for group in group_sets:
        # We count the nodes inside this group definition
        # If group is (1, 2), count is 1 (the tuple itself).
        # If group is ((1), 2), count is 2 (outer tuple, inner tuple).
        total_nodes_needed += _count_processing_nodes(group)
    
    logger.info(f"Structure requires {total_nodes_needed} augmented questions total.")

    # 3. Prepare Augmented Question Queue
    # If caller passed questions (e.g. from orchestration), use them if sufficient.
    # Otherwise, generate fresh ones here to ensure we have enough.
    if augmented_questions and len(augmented_questions) >= total_nodes_needed:
        final_aug_qs = augmented_questions[:total_nodes_needed]
    else:
        logger.info(f"Generating {total_nodes_needed} new augmented questions to satisfy structure demand.")
        # We use the augment_question helper
        aug_res = augment_question(target_query, total_nodes_needed, api_manager, config)
        if aug_res['status'] != 'SUCCESS' and not aug_res.get('augmented_questions'):
            return {"status": "FAILURE", "error_info": aug_res.get('error_info')}
        
        final_aug_qs = aug_res['augmented_questions']
        
        # Optional Selection Logic (only if we have excess)
        if config.get('SELECTIVE_AUGMENTATION_SAMPLING') and len(final_aug_qs) > total_nodes_needed:
             final_aug_qs = select_augmented_questions(final_aug_qs, config, embedding_model)
             # Ensure we still have enough after selection
             if len(final_aug_qs) < total_nodes_needed:
                 logger.warning("Selection reduced pool below required size. Using unselected pool.")
                 final_aug_qs = aug_res['augmented_questions'][:total_nodes_needed]

    if len(final_aug_qs) < total_nodes_needed:
        msg = f"Not enough augmented questions generated. Needed {total_nodes_needed}, got {len(final_aug_qs)}."
        logger.error(msg)
        return {"status": "FAILURE", "error_message": msg}

    aug_q_queue = deque(final_aug_qs)
    
    # 4. Process Each Group (Recursive Tree Traversal)
    successful_adaptations = []
    failed_adaptations = []
    
    n_sampling = config.get("ANALOGICAL_ADAPTATION_SAMPLING_N", 1)
    
    # Iterate over top-level groups
    for group_idx, group_structure in enumerate(group_sets):
        print(f"    -> Processing Top-Level Group #{group_idx + 1}: {group_structure}")
        
        # For top-level groups, we might want multiple attempts (sampling), 
        # OR just one pass if the tree determines the logic.
        # The config says "ANALOGICAL_ADAPTATION_SAMPLING_N". 
        # If we sample at the top level, we need to consume the queue multiple times?
        # That would require Nx augmented questions.
        # Simplified approach for Recursive Mode: 
        # We run the tree ONCE per group definition. n_sampling applies if we want to 
        # run the whole tree multiple times, but that complicates the queue.
        # Let's assume 1 pass per group definition for now to respect the queue budget.
        
        # To support N attempts properly in recursive mode, we'd need to copy the queue section
        # or generate N * needed questions. 
        # For now, we will run it once per group defined in the set.
        
        result_text = _process_node_recursively(
            node=group_structure, 
            aug_q_queue=aug_q_queue,
            retrieved_texts_map=retrieved_texts_map,
            api_manager=api_manager,
            config=config,
            depth=0
        )
        
        if result_text:
            successful_adaptations.append(result_text)
        else:
            failed_adaptations.append({
                "group_structure": str(group_structure),
                "error": "Recursive processing failed"
            })

    status = "SUCCESS" if successful_adaptations else "FAILURE"
    if successful_adaptations and failed_adaptations: status = "PARTIAL_SUCCESS"

    return {
        "status": status, 
        "analogically_adapted_texts": successful_adaptations, 
        "failed_adaptations": failed_adaptations
    }

# --- 6. NEW FEATURES: Analogical Consistency Generator ---

def generate_reasoning_pathways(
    target_query: str,
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates the 'First Layer' of reasoning pathways for Analogical Consistency.
    
    Supports two modes:
    1. 'distinct_augmentations': Generates K distinct augmented questions, solves each 1 time.
    2. 'single_augmentation_sampling': Generates 1 augmented question, solves it K times.
    
    Returns:
        Dict: Contains 'status' and a list of 'pathway_exemplars' (formatted text).
    """
    logger = logging.getLogger(__name__)
    mode = config.get("CONSISTENCY_GENERATION_MODE", "distinct_augmentations")
    k_pathways = config.get("CONSISTENCY_PATHWAYS_K", 3)
    
    # Determine model
    if isinstance(api_manager, GeminiAPIManager):
        model_name = config['GEMINI_MODEL_NAME_ADAPTATION']
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config['AVALAI_MODEL_NAME_ADAPTATION']
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config['OLLAMA_MODEL_NAME_ADAPTATION']
    else:
        raise TypeError(f"Unsupported API manager type: {type(api_manager)}")
        
    temp = config.get("CONSISTENCY_LAYER_1_TEMPERATURE", 0.7)
    
    pathways = []
    errors = []

    logger.info(f"Generating reasoning pathways in mode: {mode} (K={k_pathways})")

    if mode == "distinct_augmentations":
        # 1. Augment K times
        aug_res = augment_question(target_query, k_pathways, api_manager, config)
        if aug_res['status'] == 'FAILURE':
            return {"status": "FAILURE", "pathway_exemplars": [], "error_info": aug_res.get('error_info')}
        
        aug_qs = aug_res['augmented_questions']
        
        # Ensure we have enough
        if len(aug_qs) < k_pathways:
            logger.warning(f"Augmentation only returned {len(aug_qs)} questions, requested {k_pathways}.")
            
        # 2. Solve each augmented question ONCE
        for i, q in enumerate(aug_qs[:k_pathways]):
            print(f"    -> Solving Pathway {i+1} (Augmented Q): '{q[:50]}...'")
            prompt = create_self_sampling_prompt(q, config)
            resp = api_manager.generate_content(prompt, model_name, temp)
            
            if resp['status'] == 'SUCCESS':
                # Format into standard exemplar
                exemplar = f"Question: {q}\nRationale and Answer: {resp['text']}"
                pathways.append(exemplar)
            else:
                errors.append(resp)

    elif mode == "single_augmentation_sampling":
        # 1. Augment 1 time
        aug_res = augment_question(target_query, 1, api_manager, config)
        if aug_res['status'] != 'SUCCESS' or not aug_res['augmented_questions']:
             return {"status": "FAILURE", "pathway_exemplars": [], "error_info": aug_res.get('error_info')}
        
        q = aug_res['augmented_questions'][0]
        logger.info(f"Using single augmented question for sampling: '{q[:50]}...'")
        
        # 2. Solve the same question K times
        prompt = create_self_sampling_prompt(q, config)
        for i in range(k_pathways):
            print(f"    -> Solving Pathway Sample {i+1} for Single AugQ.")
            resp = api_manager.generate_content(prompt, model_name, temp)
            
            if resp['status'] == 'SUCCESS':
                exemplar = f"Question: {q}\nRationale and Answer: {resp['text']}"
                pathways.append(exemplar)
            else:
                errors.append(resp)

    else:
        return {"status": "FAILURE", "error_message": f"Unknown consistency mode: {mode}"}

    status = "SUCCESS"
    if not pathways: status = "FAILURE"
    elif errors: status = "PARTIAL_SUCCESS"

    return {"status": status, "pathway_exemplars": pathways, "errors": errors}


# --- 7. NEW FEATURES: Group-Based Self-Consistency Selection ---

def solve_with_group_consistency(
    target_query: str,
    available_exemplars: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes the Group-Based Self-Consistency strategy.
    
    1. Iterates through groups defined in config['GROUP_CONSISTENCY_CANDIDATES'].
    2. Constructs a prompt using the specific exemplars for that group.
    3. Solves the target query N times for EACH group to generate a sample set.
    4. Returns a detailed log of all attempts for analysis by the evaluator.
    
    Args:
        target_query: The main question to solve.
        available_exemplars: A list of all available adapted exemplars (0-indexed).
        api_manager: The API manager for the solver.
        config: The configuration dictionary.
        
    Returns:
        Dict: A structured log containing the attempts for each group.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Group-Based Self-Consistency Solving.")

    group_candidates = config.get("GROUP_CONSISTENCY_CANDIDATES", [])
    n_samples = config.get("GROUP_CONSISTENCY_SAMPLES_N", 5)
    
    # Determine model and temperature (Pass@N settings usually apply here for diversity)
    if isinstance(api_manager, GeminiAPIManager):
        model_name = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config['AVALAI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config['OLLAMA_MODEL_NAME_FINAL_SOLVER']
    else:
        raise TypeError(f"Unsupported API manager type for solver: {type(api_manager)}")
        
    temperature = config.get('DEFAULT_PASS_N_SOLVER_TEMPERATURE', 1.0) # High temp for diversity

    group_results = []

    for group_idx, indices_tuple in enumerate(group_candidates):
        print(f"\n    -> Processing Consistency Group #{group_idx} (Indices: {indices_tuple})...")
        
        # 1. Form the group context from available exemplars
        group_exemplars = []
        valid_group = True
        for idx in indices_tuple:
            if 0 <= idx < len(available_exemplars):
                group_exemplars.append(available_exemplars[idx])
            else:
                logger.warning(f"Group index {idx} out of bounds (Available: {len(available_exemplars)}). Skipping group.")
                valid_group = False
                break
        
        if not valid_group or not group_exemplars:
            continue

        # 2. Generate the prompt for this specific group
        prompt = create_final_reasoning_prompt(target_query, group_exemplars, config)
        
        # 3. Sampling Loop (N times)
        group_attempts = []
        for i in range(n_samples):
            print(f"        -> Generating sample {i+1}/{n_samples} for Group #{group_idx}...")
            response = api_manager.generate_content(prompt, model_name, temperature)
            
            if response['status'] == 'SUCCESS':
                group_attempts.append(response['text'])
            else:
                group_attempts.append({"status": "FAILURE", "error_info": response})

        # 4. Store Data
        group_results.append({
            "group_id": group_idx,
            "indices_used": indices_tuple,
            "attempts": group_attempts
        })

    return {
        "status": "SUCCESS" if group_results else "FAILURE",
        "group_consistency_results": group_results
    }


# --- 8. NEW FEATURES: Hierarchical Augmentation with Backward Propagation ---

class ReasoningNode:
    """Represents a node in the hierarchical augmentation tree."""
    def __init__(self, question: str, depth: int):
        self.id = str(uuid.uuid4())
        self.question = question
        self.depth = depth
        self.children: List['ReasoningNode'] = []
        self.retrieved_context: List[str] = [] # Exemplars found for this node
        self.solution: Optional[str] = None    # The solved answer/rationale
        self.solution_attempts: List[str] = [] # NEW: Stores list for Root node
        self.status: str = "PENDING"           # PENDING, SOLVED, FAILED

    def to_dict(self) -> Dict[str, Any]:
        """Recursive serialization for logging."""
        return {
            "id": self.id,
            "question": self.question,
            "depth": self.depth,
            "children": [child.to_dict() for child in self.children],
            "retrieved_context_count": len(self.retrieved_context),
            "solution_preview": (self.solution[:100] + "...") if self.solution else None,
            "solution_attempts_count": len(self.solution_attempts),
            "status": self.status
        }

def build_hierarchical_tree(
    current_question: str,
    current_depth: int,
    max_depth: int,
    branching_factor: int,
    api_manager: Any,
    config: Dict[str, Any]
) -> ReasoningNode:
    """
    Recursively builds the augmentation tree.
    """
    logger = logging.getLogger(__name__)
    node = ReasoningNode(current_question, current_depth)
    
    # Base Case: If we reached max depth, stop expanding.
    if current_depth >= max_depth:
        return node
    
    print(f"  -> [Tree Build] Expanding Node at Depth {current_depth} (Branching: {branching_factor})...")
    
    # Temporarily override the prompt template in config if needed for hierarchical augmentation
    # We use a shallow copy to not affect global config permanently if we were modifying it in place,
    # but since we pass config to augment_question, let's just ensure we use the right key.
    # augment_question uses PROMPT_TEMPLATE_SELF_SAMPLING_AUGMENTOR.
    # We can pass a modified config.
    local_config = config.copy()
    if config.get("PROMPT_TEMPLATE_HIERARCHICAL_AUGMENTOR"):
        local_config["PROMPT_TEMPLATE_SELF_SAMPLING_AUGMENTOR"] = config["PROMPT_TEMPLATE_HIERARCHICAL_AUGMENTOR"]
        
    aug_res = augment_question(current_question, branching_factor, api_manager, local_config)
    
    if aug_res['status'] != 'SUCCESS' and not aug_res.get('augmented_questions'):
        logger.warning(f"Failed to expand node at depth {current_depth}. Stopping this branch.")
        return node
        
    child_questions = aug_res['augmented_questions']
    
    # Recursively build children
    for child_q in child_questions:
        child_node = build_hierarchical_tree(child_q, current_depth + 1, max_depth, branching_factor, api_manager, config)
        node.children.append(child_node)
        
    return node

def _process_leaves(
    root: ReasoningNode,
    exemplar_data: Dict[str, Any],
    embedding_model: SentenceTransformer,
    api_manager_adapt: Any,
    api_manager_solve: Any,
    config: Dict[str, Any]
) -> None:
    """
    Traverses the tree to find leaves, optionally retrieves context, and solves them.
    """
    if not root.children:
        # This is a leaf
        print(f"    -> Processing Leaf Node (Depth {root.depth})...")
        
        # 1. Retrieval (if enabled)
        if config.get("HIERARCHICAL_LEAF_RETRIEVAL_ENABLED", True):
            top_k = config.get("HIERARCHICAL_LEAF_RETRIEVAL_TOP_K", 3)
            # Use standard retrieval
            ret_res = retrieve(
                root.question, embedding_model, 
                exemplar_data['questions'], exemplar_data['embeddings'], 
                top_k, exemplar_data.get('question_to_index')
            )
            
            if ret_res['status'] == 'SUCCESS':
                # Adapt retrieved samples
                # We use the standard adapt function but maybe with normalization only to save calls?
                # Using standard config settings for adapt
                adapt_res = adapt(
                    root.question, ret_res['retrieved_indices'], 
                    exemplar_data['questions'], exemplar_data['solutions'], 
                    api_manager_adapt, config
                )
                if adapt_res.get('adapted_texts'):
                    root.retrieved_context = adapt_res['adapted_texts']
                    print(f"      -> Leaf retrieved {len(root.retrieved_context)} samples.")
        
        # 2. Solve Leaf
        # If context exists, use RAG solver. If not, use simple solver (Self-Solve).
        # We reuse the `solve` function logic but applied to a single node.
        
        # Determine prompt template
        template_name = config.get("PROMPT_TEMPLATE_HIERARCHICAL_LEAF_SOLVER", "final_solver_simple_v1")
        
        # Construct prompt manually or use helpers
        if root.retrieved_context:
            # Use RAG prompt helper
            # We temporarily swap the config's solver template key to ensure the helper uses the one we want
            local_config = config.copy()
            local_config["PROMPT_TEMPLATE_FINAL_SOLVER"] = template_name 
            # Note: If template is simple, it ignores context. If complex, it uses it.
            # Assuming if context exists, user wants to use it.
            # If the user specified a simple template for leaves, context is ignored.
            prompt = create_final_reasoning_prompt(root.question, root.retrieved_context, local_config)
        else:
            # Use Simple prompt helper
            local_config = config.copy()
            local_config["PROMPT_TEMPLATE_FINAL_SOLVER_SIMPLE"] = template_name
            prompt = create_final_reasoning_prompt_simple(root.question, local_config)
            
        # Model Selection
        model_name = config.get("GEMINI_MODEL_NAME_FINAL_SOLVER") # Default
        if isinstance(api_manager_solve, AvalAIAPIManager): model_name = config.get("AVALAI_MODEL_NAME_FINAL_SOLVER")
        elif isinstance(api_manager_solve, OllamaAPIManager): model_name = config.get("OLLAMA_MODEL_NAME_FINAL_SOLVER")
        
        temp = config.get("DEFAULT_FINAL_SOLVER_TEMPERATURE", 1.0)
        
        resp = api_manager_solve.generate_content(prompt, model_name, temp)
        if resp['status'] == 'SUCCESS':
            root.solution = resp['text']
            root.status = "SOLVED"
        else:
            root.status = "FAILED"
            
    else:
        # Not a leaf, recurse
        for child in root.children:
            _process_leaves(child, exemplar_data, embedding_model, api_manager_adapt, api_manager_solve, config)

def propagate_solutions_upward(
    node: ReasoningNode,
    api_manager: Any,
    config: Dict[str, Any]
) -> None:
    """
    Post-order traversal to solve parents using children's solutions.
    """
    # 1. Process Children First
    for child in node.children:
        propagate_solutions_upward(child, api_manager, config)
    
    # 2. If node is already solved (it was a leaf), skip
    if node.status == "SOLVED":
        return

    # 3. Solve Parent
    # Gather valid children solutions
    child_data = []
    for child in node.children:
        if child.status == "SOLVED" and child.solution:
            child_data.append({"question": child.question, "solution": child.solution})
            
    if not child_data:
        logging.getLogger(__name__).warning(f"Node at depth {node.depth} has no solved children. Cannot propagate.")
        node.status = "FAILED_PROPAGATION"
        return
        
    print(f"  -> [Propagation] Solving Node at Depth {node.depth} using {len(child_data)} child solutions...")
    
    prompt = create_hierarchical_parent_solver_prompt(node.question, child_data, config)
    
    # Model Selection
    model_name = config.get("GEMINI_MODEL_NAME_FINAL_SOLVER") 
    if isinstance(api_manager, AvalAIAPIManager): model_name = config.get("AVALAI_MODEL_NAME_FINAL_SOLVER")
    elif isinstance(api_manager, OllamaAPIManager): model_name = config.get("OLLAMA_MODEL_NAME_FINAL_SOLVER")
    
    temp = config.get("DEFAULT_FINAL_SOLVER_TEMPERATURE", 1.0)
    
    if node.depth > 0:
        # Intermediate Node: Solve once
        resp = api_manager.generate_content(prompt, model_name, temp)
        if resp['status'] == 'SUCCESS':
            node.solution = resp['text']
            node.status = "SOLVED"
        else:
            node.status = "FAILED"
    else:
        # Root Node: Solve N times
        n_attempts = config.get("N_PASS_ATTEMPTS", 1)
        print(f"    -> Root Node detected. Solving {n_attempts} times (Pass@{n_attempts})...")
        
        success_count = 0
        for i in range(n_attempts):
            resp = api_manager.generate_content(prompt, model_name, temp)
            if resp['status'] == 'SUCCESS':
                node.solution_attempts.append(resp['text'])
                success_count += 1
                # Save the first success as the primary solution for backward compatibility
                if node.solution is None:
                    node.solution = resp['text']
        
        if success_count > 0:
            node.status = "SOLVED"
        else:
            node.status = "FAILED"

def solve_hierarchical_tree(
    target_query: str,
    exemplar_data: Dict[str, Any],
    embedding_model: SentenceTransformer,
    api_manager_adapt: Any,
    api_manager_solve: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Orchestrates the full Hierarchical Augmentation pipeline.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Hierarchical Augmentation Pipeline.")
    
    max_depth = config.get("HIERARCHICAL_TREE_DEPTH", 2)
    branching = config.get("HIERARCHICAL_BRANCHING_FACTOR", 3)
    
    # 1. Build Tree
    print("\n[HIERARCHICAL] Phase 1: Building Tree...")
    root = build_hierarchical_tree(target_query, 0, max_depth, branching, api_manager_adapt, config)
    
    # 2. Process Leaves
    print("\n[HIERARCHICAL] Phase 2: Processing Leaves...")
    _process_leaves(root, exemplar_data, embedding_model, api_manager_adapt, api_manager_solve, config)
    
    # 3. Propagate Upward
    print("\n[HIERARCHICAL] Phase 3: Backward Propagation...")
    propagate_solutions_upward(root, api_manager_solve, config)
    
    final_status = "SUCCESS" if root.status == "SOLVED" else "FAILURE"
    
    # Prepare list of attempts
    final_attempts = []
    if root.solution_attempts:
        final_attempts = root.solution_attempts
    elif root.solution:
        final_attempts = [root.solution]

    return {
        "status": final_status,
        "root_solution": root.solution,
        "root_solution_attempts": final_attempts,
        "tree_structure": root.to_dict()
    }