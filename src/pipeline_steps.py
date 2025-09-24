# src/pipeline_steps.py

"""
Core pipeline steps for the Analogical Reasoning RAG project.

This module contains the primary functions that constitute the RAG pipeline,
broken down into modular, sequential steps:
1.  retrieve: Finds relevant exemplars from the corpus.
2.  adapt: Transforms and/or summarizes the retrieved exemplars.
3.  merge: Iteratively combines adapted exemplars into a more potent one.
4.  solve: Generates the final answer using the processed exemplars, with
           new capabilities for online evaluation and early stopping.

This version is updated to be provider-agnostic, working with any API manager
that follows the project's standard interface (e.g., GeminiAPIManager, AvalAIManager).
"""

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

# Import our custom modules
from src.prompts import (
    EXEMPLAR_FORMAT,
    create_standardization_prompt,
    create_transformation_prompt,
    create_merging_prompt,
    create_final_reasoning_prompt,
    create_final_reasoning_prompt_simple
)
from src.evaluation import evaluate_single_answer_with_llm

# --- Utility Function for Embedding Generation (Unchanged) ---
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


# --- 1. RETRIEVAL STEP (Unchanged) ---

def retrieve(
    target_query: str,
    embedding_model: SentenceTransformer,
    exemplar_questions: List[str],
    embedded_exemplars: np.ndarray,
    top_k: int
) -> Dict[str, Any]:
    """
    Retrieves the top_k most relevant exemplars for a target query.
    """
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
        pass # Query is not in the corpus

    k_to_retrieve = min(top_k, len(similarities))
    top_k_indices = np.argpartition(similarities, -k_to_retrieve)[-k_to_retrieve:]
    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])][::-1]

    logger.info(f"Successfully retrieved indices: {top_k_indices.tolist()}")
    
    return {
        "status": "SUCCESS",
        "retrieved_indices": top_k_indices.tolist(),
    }


# --- 2. ADAPTATION STEP (MODIFIED for Provider Agnosticism) ---

def adapt(
    target_query: str,
    retrieved_indices: List[int],
    exemplar_questions: List[str],
    exemplar_solutions: List[str],
    api_manager: Any,  # MODIFIED: Generic API manager
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Performs standardization and transformation on a list of retrieved exemplars.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting adaptation step.")
    adapted_texts = []
    
    # MODIFIED: Use generic model name key set by the notebook's factory logic.
    model_name = config.get('MODEL_NAME_ADAPTATION')
    temperature = config['DEFAULT_ADAPTATION_TEMPERATURE']
    apply_standardize = config.get('APPLY_STANDARDIZATION', False)
    apply_transform = config.get('APPLY_TRANSFORMATION', False)

    for idx in retrieved_indices:
        original_question = exemplar_questions[idx]
        original_solution = exemplar_solutions[idx]
        
        current_text = EXEMPLAR_FORMAT.format(question=original_question, solution=original_solution)
        
        if apply_standardize:
            prompt = create_standardization_prompt(current_text)
            # MODIFIED: Call the generic api_manager
            response = api_manager.generate_content(prompt, model_name, temperature)
            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                print(f"--- ERROR during ADAPTATION (Standardization) ---")
                print(f"    - Error: {response['error_message']}")
                print(f"-------------------------------------------------")

        if apply_transform:
            prompt = create_transformation_prompt(target_query, current_text)
            # MODIFIED: Call the generic api_manager
            response = api_manager.generate_content(prompt, model_name, temperature)
            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                print(f"--- ERROR during ADAPTATION (Transformation) ---")
                print(f"    - Error: {response['error_message']}")
                print(f"------------------------------------------------")
        
        adapted_texts.append(current_text)

    return {"status": "SUCCESS", "adapted_texts": adapted_texts}


# --- 3. MERGING STEP (MODIFIED for Provider Agnosticism) ---

def merge(
    target_query: str,
    adapted_texts: List[str],
    embedding_model: SentenceTransformer,
    api_manager: Any,  # MODIFIED: Generic API manager
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Iteratively merges a list of adapted exemplars down to a target count.
    """
    logger = logging.getLogger(__name__)
    if not config.get('APPLY_MERGING', False):
        logger.info("APPLY_MERGING is False. Skipping merge step.")
        target_count = config.get('TARGET_ADAPTED_SAMPLES_MERGING', 1)
        return {"status": "SKIPPED", "merged_texts": adapted_texts[:target_count]}

    logger.info("Starting merging step.")
    current_texts = list(adapted_texts)
    target_count = config.get('TARGET_ADAPTED_SAMPLES_MERGING', 1)
    
    # MODIFIED: Use generic model name key. Merging re-uses the adaptation model.
    model_name = config.get('MODEL_NAME_ADAPTATION')
    temperature = config['DEFAULT_ADAPTATION_TEMPERATURE']
    
    while len(current_texts) > target_count and len(current_texts) >= 2:
        pair_to_merge = [current_texts.pop(0), current_texts.pop(0)]
        prompt = create_merging_prompt(target_query, pair_to_merge)
        
        # MODIFIED: Call the generic api_manager
        response = api_manager.generate_content(prompt, model_name, temperature)
        if response['status'] == 'SUCCESS':
            current_texts.append(response['text'])
        else:
            print(f"--- ERROR during MERGING ---")
            print(f"    - Error: {response['error_message']}")
            print(f"----------------------------")
            logger.warning(f"Merging failed: {response['error_message']}. Discarding pair.")

    return {"status": "SUCCESS", "merged_texts": current_texts}


# --- 4. SOLVER STEP (MODIFIED for Provider Agnosticism) ---

def solve(
    target_query: str,
    final_exemplars: List[str],
    ground_truth: str,
    api_manager: Any,  # MODIFIED: Generic API manager
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates solution(s) for the target query, with optional online evaluation.
    This function is provider-agnostic.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting final solver step.")
    
    prompt = (
        create_final_reasoning_prompt(target_query, final_exemplars)
        if final_exemplars
        else create_final_reasoning_prompt_simple(target_query)
    )

    if "Error:" in prompt:
        return {"status": "FAILURE", "solution_attempts": [prompt], "evaluation_results": []}

    n_attempts = config.get("N_PASS_ATTEMPTS", 1)
    if 'MAX_ATTEMPTS_PER_QUESTION' in config.get('HARD_QUESTION_IDENTIFICATION_CONFIG', {}):
        n_attempts = config['HARD_QUESTION_IDENTIFICATION_CONFIG']['MAX_ATTEMPTS_PER_QUESTION']

    # MODIFIED: Use generic model name key
    model_name = config.get('MODEL_NAME_FINAL_SOLVER')
    temperature = config.get('DEFAULT_PASS_N_SOLVER_TEMPERATURE', 1.0)
    
    solution_attempts = []
    evaluation_results = []
    
    online_eval_enabled = config.get("ONLINE_EVALUATION_ENABLED", False)
    stop_on_success = config.get("STOP_ON_FIRST_SUCCESS", False)

    logger.info(f"Generating up to {n_attempts} solution attempts. Online evaluation: {online_eval_enabled}.")

    for i in range(n_attempts):
        print(f"    -> Generating solution attempt {i+1}/{n_attempts}...")
        # MODIFIED: Call the generic api_manager
        response = api_manager.generate_content(prompt, model_name, temperature)
        
        if response['status'] != 'SUCCESS':
            error_str = f"Error on attempt {i+1}: {response['error_message']}"
            solution_attempts.append(error_str)
            logger.error(f"API generation failed for query. Halting. Error: {error_str}")
            return {
                "status": "UN-EVALUABLE",
                "reason": "API error during solution generation.",
                "solution_attempts": solution_attempts,
                "evaluation_results": evaluation_results
            }

        solution_text = response['text']
        solution_attempts.append(solution_text)
        print(f"       Attempt {i+1} successful.")

        if online_eval_enabled:
            print(f"       -> Performing online evaluation for attempt {i+1}...")
            # MODIFIED: Pass the generic api_manager to the evaluation function
            is_correct, eval_status = evaluate_single_answer_with_llm(
                solution_text, ground_truth, api_manager, config
            )
            evaluation_results.append({"is_correct": is_correct, "status": eval_status})

            if eval_status != "SUCCESS":
                logger.error(f"Online evaluation failed with status '{eval_status}'. Halting for this query.")
                return {
                    "status": "UN-EVALUABLE",
                    "reason": f"Evaluation failed with status: {eval_status}",
                    "solution_attempts": solution_attempts,
                    "evaluation_results": evaluation_results
                }
            
            print(f"          Evaluation result: {'Correct' if is_correct else 'Incorrect'}")

            if stop_on_success and is_correct:
                logger.info(f"Correct answer found on attempt {i+1}. Stopping further attempts as per config.")
                break
        else:
            evaluation_results.append({"is_correct": None, "status": "NOT_EVALUATED"})

    return {
        "status": "SUCCESS",
        "solution_attempts": solution_attempts,
        "evaluation_results": evaluation_results
    }