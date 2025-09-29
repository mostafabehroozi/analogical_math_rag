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
"""

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union

# Import our custom modules
from src.prompts import (
    EXEMPLAR_FORMAT,
    create_standardization_prompt,
    create_transformation_prompt,
    create_merging_prompt,
    create_final_reasoning_prompt,
    create_final_reasoning_prompt_simple
)

# --- Utility Function for Embedding Generation (No changes needed) ---
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


# --- 1. RETRIEVAL STEP (No changes needed, does not use an LLM API) ---
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
    
    query_embedding = _generate_embeddings([target_query], embedding_model)
    if query_embedding.size == 0:
        logger.error("Failed to generate embedding for the target query. Retrieval cannot proceed.")
        return {"status": "FAILURE", "retrieved_indices": [], "retrieved_exemplars": []}
    
    similarities = cosine_similarity(query_embedding, embedded_exemplars)[0]
    
    try:
        query_index_in_corpus = exemplar_questions.index(target_query)
        similarities[query_index_in_corpus] = -np.inf
    except ValueError:
        pass

    k_to_retrieve = min(top_k, len(similarities))
    top_k_indices = np.argpartition(similarities, -k_to_retrieve)[-k_to_retrieve:]
    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])][::-1]

    logger.info(f"Successfully retrieved indices: {top_k_indices.tolist()}")
    
    return {
        "status": "SUCCESS",
        "retrieved_indices": top_k_indices.tolist(),
    }


# --- 2. ADAPTATION STEP (MODIFIED for error handling) ---
def adapt(
    target_query: str,
    retrieved_indices: List[int],
    exemplar_questions: List[str],
    exemplar_solutions: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Performs standardization and transformation on retrieved exemplars.
    Captures failures for individual exemplars without halting the entire step.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting adaptation step.")
    
    successful_texts = []
    failed_adaptations = []
    
    provider = config.get("API_PROVIDER", "gemini").lower()
    model_name = config[f'{"AVALAI" if provider == "avalai" else "GEMINI"}_MODEL_NAME_ADAPTATION']
    temperature = config['DEFAULT_ADAPTATION_TEMPERATURE']
    apply_standardize = config.get('APPLY_STANDARDIZATION', False)
    apply_transform = config.get('APPLY_TRANSFORMATION', False)

    for idx in retrieved_indices:
        original_question = exemplar_questions[idx]
        original_solution = exemplar_solutions[idx]
        current_text = EXEMPLAR_FORMAT.format(question=original_question, solution=original_solution)
        
        step_failed = False

        if apply_standardize and not step_failed:
            logger.info(f"Applying standardization to exemplar index {idx}.")
            print(f"    -> Standardizing exemplar {idx}...")
            prompt = create_standardization_prompt(current_text)
            
            print(f"      [API Context] Calling LLM for: Standardization (Exemplar #{idx})")
            response = api_manager.generate_content(prompt, model_name, temperature)
            
            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                logger.warning(f"Standardization failed for exemplar {idx}: {response['error_message']}")
                failed_adaptations.append({"source_index": idx, "failed_at_step": "standardization", "error_info": response})
                step_failed = True

        if apply_transform and not step_failed:
            logger.info(f"Applying transformation to exemplar index {idx}.")
            print(f"    -> Transforming exemplar {idx}...")
            prompt = create_transformation_prompt(target_query, current_text)
            
            print(f"      [API Context] Calling LLM for: Transformation (Exemplar #{idx})")
            response = api_manager.generate_content(prompt, model_name, temperature)

            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                logger.warning(f"Transformation failed for exemplar {idx}: {response['error_message']}")
                failed_adaptations.append({"source_index": idx, "failed_at_step": "transformation", "error_info": response})
                step_failed = True
        
        if not step_failed:
            successful_texts.append(current_text)

    # Determine overall status of the adaptation step
    if not retrieved_indices:
        final_status = "SUCCESS" # Nothing to do is a success
    elif not successful_texts and failed_adaptations:
        final_status = "FAILURE" # All attempts failed
    elif successful_texts and failed_adaptations:
        final_status = "PARTIAL_SUCCESS" # Some succeeded, some failed
    else:
        final_status = "SUCCESS" # All succeeded

    return {
        "status": final_status,
        "adapted_texts": successful_texts,
        "failed_adaptations": failed_adaptations
    }


# --- 3. MERGING STEP (MODIFIED for error handling) ---
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
    
    provider = config.get("API_PROVIDER", "gemini").lower()
    model_name = config[f'{"AVALAI" if provider == "avalai" else "GEMINI"}_MODEL_NAME_ADAPTATION']
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


# --- 4. SOLVER STEP (MODIFIED for error handling and config passing) ---
def solve(
    target_query: str,
    final_exemplars: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates final solution(s). If an attempt fails due to an API error,
    the error details are saved instead of a text solution for that attempt.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting final solver step.")
    
    # MODIFIED: Pass the 'config' dictionary to the simple prompt creator.
    # This allows it to use the prompt template specified in the experiment config.
    prompt = create_final_reasoning_prompt(target_query, final_exemplars) if final_exemplars else create_final_reasoning_prompt_simple(target_query, config)
    logger.info(f"Using {'retrieval-augmented' if final_exemplars else 'simple'} prompt for the solver.")

    if "Error:" in prompt:
        error_msg = f"Failed to create final reasoning prompt: {prompt}"
        logger.error(error_msg)
        return {"status": "FAILURE", "solution_attempts": [{"status": "FAILURE", "error_info": {"error_message": error_msg}}]}

    n_attempts = config.get("N_PASS_ATTEMPTS", 1)
    
    provider = config.get("API_PROVIDER", "gemini").lower()
    model_name = config[f'{"AVALAI" if provider == "avalai" else "GEMINI"}_MODEL_NAME_FINAL_SOLVER']
    temperature = config.get('DEFAULT_PASS_N_SOLVER_TEMPERATURE', 1.0)
    
    solution_attempts: List[Union[str, Dict]] = []
    
    logger.info(f"Generating {n_attempts} solution attempts for Pass@{n_attempts}.")
    for i in range(n_attempts):
        logger.info(f"Generating attempt {i+1}/{n_attempts}.")
        print(f"    -> Generating solution attempt {i+1}/{n_attempts}...")
        
        print(f"      [API Context] Calling LLM for: Final Solution (Attempt #{i+1})")