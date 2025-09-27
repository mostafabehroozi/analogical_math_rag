# src/pipeline_steps.py

"""
Core pipeline steps for the Analogical Reasoning RAG project.

This module contains the primary functions that constitute the RAG pipeline,
broken down into modular, sequential steps:
1.  retrieve: Finds relevant exemplars from the corpus.
2.  adapt: Transforms and/or summarizes the retrieved exemplars.
3.  merge: Iteratively combines adapted exemplars into a more potent one.
4.  solve: Generates the final answer using the processed exemplars.

Each function is designed to be called in sequence by the orchestration module
and is now API provider-agnostic.
"""

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

# Import our custom modules
# No specific API manager is imported; the object is passed into the functions.
from src.prompts import (
    EXEMPLAR_FORMAT,
    create_standardization_prompt,
    create_transformation_prompt,
    create_merging_prompt,
    create_final_reasoning_prompt,
    create_final_reasoning_prompt_simple
)

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


# --- 2. ADAPTATION STEP ---

def adapt(
    target_query: str,
    retrieved_indices: List[int],
    exemplar_questions: List[str],
    exemplar_solutions: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Performs standardization and transformation on a list of retrieved exemplars."""
    logger = logging.getLogger(__name__)
    logger.info("Starting adaptation step.")
    adapted_texts = []
    
    provider = config.get("API_PROVIDER", "gemini").lower()
    if provider == "avalai":
        model_name = config['AVALAI_MODEL_NAME_ADAPTATION']
    else:
        model_name = config['GEMINI_MODEL_NAME_ADAPTATION']
    
    temperature = config['DEFAULT_ADAPTATION_TEMPERATURE']
    apply_standardize = config.get('APPLY_STANDARDIZATION', False)
    apply_transform = config.get('APPLY_TRANSFORMATION', False)

    for idx in retrieved_indices:
        original_question = exemplar_questions[idx]
        original_solution = exemplar_solutions[idx]
        current_text = EXEMPLAR_FORMAT.format(question=original_question, solution=original_solution)
        
        if apply_standardize:
            logger.info(f"Applying standardization to exemplar index {idx}.")
            print(f"    -> Standardizing exemplar {idx}...")
            prompt = create_standardization_prompt(current_text)
            
            # NEW: Add contextual print before the API call
            print(f"      [API Context] Calling LLM for: Standardization (Exemplar #{idx})")
            response = api_manager.generate_content(prompt, model_name, temperature)
            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                logger.warning(f"Standardization failed for exemplar {idx}: {response['error_message']}. Using original text.")

        if apply_transform:
            logger.info(f"Applying transformation to exemplar index {idx}.")
            print(f"    -> Transforming exemplar {idx}...")
            prompt = create_transformation_prompt(target_query, current_text)
            
            # NEW: Add contextual print before the API call
            print(f"      [API Context] Calling LLM for: Transformation (Exemplar #{idx})")
            response = api_manager.generate_content(prompt, model_name, temperature)
            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                logger.warning(f"Transformation failed for exemplar {idx}: {response['error_message']}. Using text from previous step.")
        
        adapted_texts.append(current_text)

    return {"status": "SUCCESS", "adapted_texts": adapted_texts}


# --- 3. MERGING STEP ---

def merge(
    target_query: str,
    adapted_texts: List[str],
    embedding_model: SentenceTransformer,
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Iteratively merges a list of adapted exemplars down to a target count."""
    logger = logging.getLogger(__name__)
    if not config.get('APPLY_MERGING', False):
        logger.info("APPLY_MERGING is False. Skipping merge step.")
        target_count = config.get('TARGET_ADAPTED_SAMPLES_MERGING', 1)
        return {"status": "SKIPPED", "merged_texts": adapted_texts[:target_count]}

    logger.info("Starting merging step.")
    current_texts = list(adapted_texts)
    target_count = config.get('TARGET_ADAPTED_SAMPLES_MERGING', 1)
    
    provider = config.get("API_PROVIDER", "gemini").lower()
    if provider == "avalai":
        model_name = config['AVALAI_MODEL_NAME_ADAPTATION']
    else:
        model_name = config['GEMINI_MODEL_NAME_ADAPTATION']
        
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
            break
            
        # NEW: Add contextual print before the API call
        print(f"      [API Context] Calling LLM for: Merging (Iteration #{iteration})")
        response = api_manager.generate_content(prompt, model_name, temperature)
        if response['status'] == 'SUCCESS':
            current_texts.append(response['text'])
        else:
            logger.warning(f"Merging failed: {response['error_message']}. Discarding pair.")

    return {"status": "SUCCESS", "merged_texts": current_texts}


# --- 4. SOLVER STEP ---

def solve(
    target_query: str,
    final_exemplars: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Generates the final solution(s) for the target query using the processed exemplars."""
    logger = logging.getLogger(__name__)
    logger.info("Starting final solver step.")
    
    if final_exemplars:
        prompt = create_final_reasoning_prompt(target_query, final_exemplars)
        logger.info("Using retrieval-augmented prompt for the solver.")
    else:
        prompt = create_final_reasoning_prompt_simple(target_query)
        logger.info("Using simple prompt for the solver (no retrieval).")

    if "Error:" in prompt:
        logger.error(f"Failed to create final reasoning prompt: {prompt}")
        return {"status": "FAILURE", "solution_attempts": [prompt]}

    n_attempts = config.get("N_PASS_ATTEMPTS", 1)
    
    provider = config.get("API_PROVIDER", "gemini").lower()
    if provider == "avalai":
        model_name = config['AVALAI_MODEL_NAME_FINAL_SOLVER']
    else:
        model_name = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
        
    temperature = config.get('DEFAULT_PASS_N_SOLVER_TEMPERATURE', 1.0)
    
    solution_attempts = []
    logger.info(f"Generating {n_attempts} solution attempts for Pass@{n_attempts}.")
    for i in range(n_attempts):
        logger.info(f"Generating attempt {i+1}/{n_attempts}.")
        print(f"    -> Generating solution attempt {i+1}/{n_attempts}...")
        
        # NEW: Add contextual print before the API call
        print(f"      [API Context] Calling LLM for: Final Solution (Attempt #{i+1})")
        response = api_manager.generate_content(prompt, model_name, temperature)
        
        if response['status'] == 'SUCCESS':
            solution_attempts.append(response['text'])
        else:
            error_str = f"Error on attempt {i+1}: {response['error_message']}"
            solution_attempts.append(error_str)
            logger.warning(error_str)

    return {"status": "SUCCESS", "solution_attempts": solution_attempts}