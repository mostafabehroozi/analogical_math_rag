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
augmentation, and analogical adaptation.
"""

import logging
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union, Tuple, Optional
import time

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
    top_k: int
) -> Dict[str, Any]:
    """Retrieves the top_k most relevant exemplars for a target query."""
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
    
    # Step 3: Handle potential self-match
    self_match_start_time = time.time()
    try:
        # --- NEW DIAGNOSTIC 1: Measure the .index() call ONLY ---
        index_search_start_time = time.time()
        
        # This is the line we expect to be extremely slow.
        query_index_in_corpus = exemplar_questions.index(target_query)
        
        # Immediately log the time taken for the search.
        log_time_diagnostic("  -> Sub-step: list.index() search", index_search_start_time, indent=indent_level)
        
        
        # --- NEW DIAGNOSTIC 2: Measure the remaining (fast) operations ---
        update_and_print_start_time = time.time()
        
        # These two lines should be nearly instantaneous.
        similarities[query_index_in_corpus] = -np.inf
        print(f"{'  '*indent_level}Self-match found at index {query_index_in_corpus}, set to -np.inf.")
        
        # Immediately log the time taken for the update and print.
        log_time_diagnostic("  -> Sub-step: Update similarities & print", update_and_print_start_time, indent=indent_level)

    except ValueError:
        print(f"{'  '*indent_level}Target query not found in corpus (no self-match to remove).")
        # Note: If this block is hit, the .index() call already finished its slow scan.
        # The diagnostic for the search will still have been logged if it was started.
        pass
        
    # --- MODIFIED: Rename the original diagnostic to clarify it's the total time ---
    current_diag_time = log_time_diagnostic("Handle self-match (Total)", self_match_start_time, indent=indent_level)


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

def _match_questions_to_groups(
    augmented_questions: List[str],
    retrieved_samples: List[str],
    group_sets: List[Tuple[int, ...]],
    embedding_model: SentenceTransformer
) -> List[Tuple[str, Tuple[int, ...]]]:
    """
    Directly pairs each augmented question with a group from the config list.
    This new logic allows for reusing identical group definitions.
    """
    logger = logging.getLogger(__name__)
    
    # Directly pair the first N augmented questions with the first N groups.
    # This uses a simple 1-to-1 mapping.
    num_pairs = min(len(augmented_questions), len(group_sets))
    matched_pairs = list(zip(augmented_questions[:num_pairs], group_sets[:num_pairs]))

    logger.info(f"Directly paired {len(matched_pairs)} augmented questions to sample groups.")
    return matched_pairs

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
    Performs analogical reasoning on groups of retrieved samples.
    
    MODIFIED: Now supports empty groups `()` which triggers a 'self-solve'
    action on an augmented question, using the self-sampling prompt.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting analogical adaptation step.")
    
    n_sampling = config.get("ANALOGICAL_ADAPTATION_SAMPLING_N", 1)
    group_sets = config.get("ANALOGICAL_GROUP_SETS", [])

    if not group_sets:
        logger.warning("ANALOGICAL_GROUP_SETS is empty. Skipping analogical adaptation.")
        return {"status": "SKIPPED", "reason": "No groups defined."}

    if isinstance(api_manager, GeminiAPIManager): model_name = config['GEMINI_MODEL_NAME_ADAPTATION']
    elif isinstance(api_manager, AvalAIAPIManager): model_name = config['AVALAI_MODEL_NAME_ADAPTATION']
    elif isinstance(api_manager, OllamaAPIManager): model_name = config['OLLAMA_MODEL_NAME_ADAPTATION']
    else: raise TypeError(f"Unsupported API manager type for analogical adaptation: {type(api_manager)}")
        
    # Get temperatures for both modes
    analogical_temp = config.get("DEFAULT_ANALOGICAL_ADAPTATION_TEMPERATURE", 1.0)
    self_solve_temp = config.get("SELF_SAMPLING_TEMPERATURE", 0.7)

    # Build full text for retrieved samples
    retrieved_samples_texts = [
        EXEMPLAR_FORMAT.format(question=exemplar_data['questions'][i], solution=exemplar_data['solutions'][i])
        for i in retrieved_indices
    ]
    
    question_group_pairs = []
    if config.get("APPLY_ANALOGICAL_ADAPTATION_AUGMENTATION") and augmented_questions:
        question_group_pairs = _match_questions_to_groups(augmented_questions, retrieved_samples_texts, group_sets, embedding_model)
    else:
        question_group_pairs = [(target_query, group) for group in group_sets]

    successful_adaptations = []
    failed_adaptations = []

    for question_for_prompt, group_indices in question_group_pairs:
        # --- START OF MODIFICATION ---

        # Case 1: The group is EMPTY -> Perform self-solve
        if not group_indices:
            print(f"    -> Performing SELF-SOLVE for empty group using question: '{question_for_prompt[:50]}...'")
            prompt = create_self_sampling_prompt(question_for_prompt, config)
            # Use the self-sampling temperature for this mode
            response = api_manager.generate_content(prompt, model_name, self_solve_temp)
            
            if response['status'] == 'SUCCESS':
                # The self-sampling prompt only returns the solution. We must format it
                # into a full exemplar to match the output of the analogical path.
                formatted_text = f"Question: {question_for_prompt}\nRationale and Answer: {response['text']}"
                successful_adaptations.append(formatted_text)
            else:
                failed_adaptations.append({
                    "group_indices": "() - Self-Solve",
                    "attempt": 0, # Self-solve is a single attempt per group
                    "question_used": question_for_prompt,
                    "error_info": response
                })
            continue # Move to the next group

        # Case 2: The group is NOT EMPTY -> Perform standard analogical adaptation
        else:
            # Check if group indices are valid for the retrieved samples
            if any(i > len(retrieved_samples_texts) for i in group_indices):
                logger.warning(f"Group {group_indices} has an index out of bounds for the {len(retrieved_samples_texts)} retrieved samples. Skipping.")
                continue
            
            sample_group_texts = [retrieved_samples_texts[i-1] for i in group_indices]
            
            for attempt in range(n_sampling):
                print(f"    -> Analogical adaptation for group {group_indices}, attempt {attempt+1}/{n_sampling}...")
                prompt = create_analogical_adaptation_prompt(question_for_prompt, sample_group_texts, config)
                # Use the analogical adaptation temperature for this mode
                response = api_manager.generate_content(prompt, model_name, analogical_temp)
                
                if response['status'] == 'SUCCESS':
                    successful_adaptations.append(response['text'])
                else:
                    failed_adaptations.append({
                        "group_indices": group_indices,
                        "attempt": attempt,
                        "question_used": question_for_prompt,
                        "error_info": response
                    })
        # --- END OF MODIFICATION ---

    if not successful_adaptations and failed_adaptations: status = "FAILURE"
    elif successful_adaptations and failed_adaptations: status = "PARTIAL_SUCCESS"
    else: status = "SUCCESS"

    return {"status": status, "analogically_adapted_texts": successful_adaptations, "failed_adaptations": failed_adaptations}
