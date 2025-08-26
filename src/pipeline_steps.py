# src/pipeline_steps.py

"""
Core pipeline steps for the Analogical Reasoning RAG project.

This module contains the primary functions that constitute the RAG pipeline,
broken down into modular, sequential steps:
1.  retrieve: Finds relevant exemplars from the corpus.
2.  adapt: Transforms and/or summarizes the retrieved exemplars.
3.  merge: Iteratively combines adapted exemplars into a more potent one.
4.  solve: Generates the final answer using the processed exemplars.

Each function is designed to be called in sequence by the orchestration module.
"""

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

# Import our custom modules
from src.api_manager import GeminiAPIManager
from src.prompts import (
    EXEMPLAR_FORMAT,
    create_standardization_prompt,
    create_transformation_prompt,
    create_merging_prompt,
    create_final_reasoning_prompt
)

# --- Utility Function for Embedding Generation ---
# Moved from the original notebook to be a helper here.
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
            show_progress_bar=False,  # Assuming this runs non-interactively
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
    """
    Retrieves the top_k most relevant exemplars for a target query.

    Args:
        target_query (str): The new question to find exemplars for.
        embedding_model (SentenceTransformer): The model to create the query embedding.
        exemplar_questions (List[str]): The list of all questions in the corpus.
        embedded_exemplars (np.ndarray): The pre-computed embeddings of all corpus questions.
        top_k (int): The number of exemplars to retrieve.

    Returns:
        A dictionary containing the retrieval results, including indices and raw text.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting retrieval for Top-{top_k} exemplars.")
    
    query_embedding = _generate_embeddings([target_query], embedding_model)
    if query_embedding.size == 0:
        logger.error("Failed to generate embedding for the target query. Retrieval cannot proceed.")
        return {"status": "FAILURE", "retrieved_indices": [], "retrieved_exemplars": []}
    
    similarities = cosine_similarity(query_embedding, embedded_exemplars)[0]
    
    # Exclude the query itself if it exists in the corpus
    try:
        query_index_in_corpus = exemplar_questions.index(target_query)
        similarities[query_index_in_corpus] = -np.inf
    except ValueError:
        pass # Query is not in the corpus

    # Get the indices of the top_k most similar exemplars
    # Using argpartition is more efficient than a full sort for large k
    k_to_retrieve = min(top_k, len(similarities))
    top_k_indices = np.argpartition(similarities, -k_to_retrieve)[-k_to_retrieve:]
    # Sort these k indices by their similarity scores
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
    gemini_manager: GeminiAPIManager,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Performs standardization and transformation on a list of retrieved exemplars.
    ...
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting adaptation step.")
    adapted_texts = []
    
    model_name = config['GEMINI_MODEL_NAME_ADAPTATION']
    temperature = config['DEFAULT_ADAPTATION_TEMPERATURE']
    apply_standardize = config.get('APPLY_STANDARDIZATION', False)
    apply_transform = config.get('APPLY_TRANSFORMATION', False)

    for idx in retrieved_indices:
        original_question = exemplar_questions[idx]
        original_solution = exemplar_solutions[idx]
        
        # MODIFIED: Create the initial text using the standardized format
        current_text = EXEMPLAR_FORMAT.format(question=original_question, solution=original_solution)
        
        # Step 2a: Standardization
        if apply_standardize:
            logger.info(f"Applying standardization to exemplar index {idx}.")
            print(f"    -> Standardizing exemplar {idx}...")
            
            # MODIFIED: Call the updated prompt creation function
            prompt = create_standardization_prompt(current_text)
            
            response = gemini_manager.generate_content(prompt, model_name, temperature)
            if response['status'] == 'SUCCESS':
                current_text = response['text']
                logger.info("Standardization successful.")
                print(f"       Standardized text (start): '{current_text[:120]}...'")
            else:
                logger.warning(f"Standardization failed for exemplar {idx}: {response['error_message']}. Using original text.")

        # Step 2b: Transformation
        if apply_transform:
            logger.info(f"Applying transformation to exemplar index {idx}.")
            print(f"    -> Transforming exemplar {idx}...")
            
            # MODIFIED: Call the updated prompt creation function
            prompt = create_transformation_prompt(target_query, current_text)
            
            response = gemini_manager.generate_content(prompt, model_name, temperature)
            if response['status'] == 'SUCCESS':
                current_text = response['text']
                logger.info("Transformation successful.")
                print(f"       Transformed text (start): '{current_text[:120]}...'")
            else:
                logger.warning(f"Transformation failed for exemplar {idx}: {response['error_message']}. Using text from previous step.")
        
        adapted_texts.append(current_text)

    return {"status": "SUCCESS", "adapted_texts": adapted_texts}


# --- 3. MERGING STEP ---

def merge(
    target_query: str,
    adapted_texts: List[str],
    embedding_model: SentenceTransformer,
    gemini_manager: GeminiAPIManager,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Iteratively merges a list of adapted exemplars down to a target count.

    Args:
        target_query (str): The main question.
        adapted_texts (List[str]): The list of texts from the 'adapt' step.
        embedding_model (SentenceTransformer): The embedding model.
        gemini_manager (GeminiAPIManager): The API manager instance.
        config (Dict): The main configuration dictionary.

    Returns:
        A dictionary containing the final list of merged texts.
    """
    logger = logging.getLogger(__name__)
    if not config.get('APPLY_MERGING', False):
        logger.info("APPLY_MERGING is False. Skipping merge step.")
        # Return the top N samples as specified by the config
        target_count = config.get('TARGET_ADAPTED_SAMPLES_MERGING', 1)
        return {"status": "SKIPPED", "merged_texts": adapted_texts[:target_count]}

    logger.info("Starting merging step.")
    current_texts = list(adapted_texts)
    target_count = config.get('TARGET_ADAPTED_SAMPLES_MERGING', 1)
    
    model_name = config['GEMINI_MODEL_NAME_ADAPTATION']
    temperature = config['DEFAULT_ADAPTATION_TEMPERATURE']
    
    # TODO: The selection logic for which pair to merge can be re-implemented here.
    # For simplicity in this refactoring step, we'll merge the first two items repeatedly.
    # A more advanced implementation would use the embedding-based selection from the notebook.
    
    iteration = 0
    while len(current_texts) > target_count and len(current_texts) >= 2:
        iteration += 1
        logger.info(f"Merge iteration {iteration}: Merging from {len(current_texts)} samples.")
        
        # Simple strategy: merge the first two samples.
        pair_to_merge = [current_texts.pop(0), current_texts.pop(0)]
        
        # --- ADDED FOR MONITORING ---
        print(f"    -> Merging pair in iteration {iteration}:")
        print(f"       Sample 1 (start): '{pair_to_merge[0][:100]}...'")
        print(f"       Sample 2 (start): '{pair_to_merge[1][:100]}...'")
        
        prompt = create_merging_prompt(target_query, pair_to_merge)
        if "Error:" in prompt:
            logger.error(f"Failed to create merging prompt: {prompt}")
            break # Stop merging if there's a prompt error.
            
        response = gemini_manager.generate_content(prompt, model_name, temperature)
        if response['status'] == 'SUCCESS':
            # --- ADDED FOR MONITORING ---
            print(f"       Merged text (start): '{response['text'][:120]}...'")
            current_texts.append(response['text']) # Add the new merged sample to the pool
            logger.info("Merging successful.")
        else:
            logger.warning(f"Merging failed: {response['error_message']}. Discarding pair.")
            # If a merge fails, we could add the originals back, but for now we discard.

    return {"status": "SUCCESS", "merged_texts": current_texts}


# --- 4. SOLVER STEP ---

def solve(
    target_query: str,
    final_exemplars: List[str],
    gemini_manager: GeminiAPIManager,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates the final solution(s) for the target query using the processed exemplars.

    Args:
        target_query (str): The main question to solve.
        final_exemplars (List[str]): The final list of exemplars after adaptation/merging.
        gemini_manager (GeminiAPIManager): The API manager instance.
        config (Dict): The main configuration dictionary.

    Returns:
        A dictionary containing a list of all generated solution attempts (for Pass@N).
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting final solver step.")
    
    prompt = create_final_reasoning_prompt(target_query, final_exemplars)
    if "Error:" in prompt:
        logger.error(f"Failed to create final reasoning prompt: {prompt}")
        return {"status": "FAILURE", "solution_attempts": [prompt]}

    n_attempts = config.get("N_PASS_ATTEMPTS", 1)
    model_name = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    temperature = config.get('DEFAULT_PASS_N_SOLVER_TEMPERATURE', 1.0)
    
    solution_attempts = []
    logger.info(f"Generating {n_attempts} solution attempts for Pass@{n_attempts}.")
    for i in range(n_attempts):
        logger.info(f"Generating attempt {i+1}/{n_attempts}.")
        # --- ADDED FOR MONITORING ---
        print(f"    -> Generating solution attempt {i+1}/{n_attempts}...")
        response = gemini_manager.generate_content(prompt, model_name, temperature)
        
        if response['status'] == 'SUCCESS':
            solution_attempts.append(response['text'])
            # --- ADDED FOR MONITORING ---
            print(f"       Attempt {i+1} successful.")
        else:
            # Append a formatted error message to maintain the list size for Pass@K analysis
            error_str = f"Error on attempt {i+1}: {response['error_message']}"
            solution_attempts.append(error_str)
            logger.warning(error_str)

    return {"status": "SUCCESS", "solution_attempts": solution_attempts}