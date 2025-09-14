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
    create_final_reasoning_prompt,
    create_final_reasoning_prompt_simple
)
# --- NEW: Import for online evaluation capability ---
from src.evaluation import evaluate_single_answer_with_llm

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


# --- 2. ADAPTATION STEP (MODIFIED) ---

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
        
        current_text = EXEMPLAR_FORMAT.format(question=original_question, solution=original_solution)
        
        if apply_standardize:
            prompt = create_standardization_prompt(current_text)
            response = gemini_manager.generate_content(prompt, model_name, temperature)
            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                # --- NEW: Enhanced Error Printing ---
                print(f"--- ERROR during ADAPTATION (Standardization) ---")
                print(f"    - Error: {response['error_message']}")
                print(f"-------------------------------------------------")


        if apply_transform:
            prompt = create_transformation_prompt(target_query, current_text)
            response = gemini_manager.generate_content(prompt, model_name, temperature)
            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                # --- NEW: Enhanced Error Printing ---
                print(f"--- ERROR during ADAPTATION (Transformation) ---")
                print(f"    - Error: {response['error_message']}")
                print(f"------------------------------------------------")
        
        adapted_texts.append(current_text)

    return {"status": "SUCCESS", "adapted_texts": adapted_texts}


# --- 3. MERGING STEP (MODIFIED) ---

def merge(
    target_query: str,
    adapted_texts: List[str],
    embedding_model: SentenceTransformer,
    gemini_manager: GeminiAPIManager,
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
    model_name = config['GEMINI_MODEL_NAME_ADAPTATION']
    temperature = config['DEFAULT_ADAPTATION_TEMPERATURE']
    
    while len(current_texts) > target_count and len(current_texts) >= 2:
        pair_to_merge = [current_texts.pop(0), current_texts.pop(0)]
        prompt = create_merging_prompt(target_query, pair_to_merge)
        
        response = gemini_manager.generate_content(prompt, model_name, temperature)
        if response['status'] == 'SUCCESS':
            current_texts.append(response['text'])
        else:
            # --- NEW: Enhanced Error Printing ---
            print(f"--- ERROR during MERGING ---")
            print(f"    - Error: {response['error_message']}")
            print(f"----------------------------")
            logger.warning(f"Merging failed: {response['error_message']}. Discarding pair.")

    return {"status": "SUCCESS", "merged_texts": current_texts}


# --- 4. SOLVER STEP (REWRITTEN) ---

def solve(
    target_query: str,
    final_exemplars: List[str],
    ground_truth: str,
    gemini_manager: GeminiAPIManager,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates solution(s) for the target query, with optional online evaluation.

    If online evaluation is enabled, it evaluates each attempt immediately. If an
    error occurs during generation or evaluation, it halts and marks the question
    as 'UN-EVALUABLE'. It also supports stopping after the first correct answer.

    Args:
        target_query (str): The main question to solve.
        final_exemplars (List[str]): Final exemplars after adaptation/merging.
        ground_truth (str): The ground truth solution for online evaluation.
        gemini_manager (GeminiAPIManager): The API manager instance.
        config (Dict): The main configuration dictionary.

    Returns:
        A dictionary containing the overall status, a list of solution attempts,
        and a corresponding list of evaluation results if applicable.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting final solver step.")
    
    # Select the prompt based on whether exemplars exist
    prompt = (
        create_final_reasoning_prompt(target_query, final_exemplars)
        if final_exemplars
        else create_final_reasoning_prompt_simple(target_query)
    )

    if "Error:" in prompt:
        return {"status": "FAILURE", "solution_attempts": [prompt], "evaluation_results": []}

    # Determine how many attempts to make based on config
    n_attempts = config.get("N_PASS_ATTEMPTS", 1)
    # For hard question identification, a different key might be used
    if 'MAX_ATTEMPTS_PER_QUESTION' in config.get('HARD_QUESTION_IDENTIFICATION_CONFIG', {}):
        n_attempts = config['HARD_QUESTION_IDENTIFICATION_CONFIG']['MAX_ATTEMPTS_PER_QUESTION']

    model_name = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    temperature = config.get('DEFAULT_PASS_N_SOLVER_TEMPERATURE', 1.0)
    
    solution_attempts = []
    evaluation_results = []
    
    online_eval_enabled = config.get("ONLINE_EVALUATION_ENABLED", False)
    stop_on_success = config.get("STOP_ON_FIRST_SUCCESS", False)

    logger.info(f"Generating up to {n_attempts} solution attempts. Online evaluation: {online_eval_enabled}.")

    for i in range(n_attempts):
        print(f"    -> Generating solution attempt {i+1}/{n_attempts}...")
        response = gemini_manager.generate_content(prompt, model_name, temperature)
        
        if response['status'] != 'SUCCESS':
            # API error during generation. Halt all processing for this question.
            # The error is already printed by api_manager, so we just log and return.
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

        # --- Online Evaluation Logic ---
        if online_eval_enabled:
            print(f"       -> Performing online evaluation for attempt {i+1}...")
            is_correct, eval_status = evaluate_single_answer_with_llm(
                solution_text, ground_truth, gemini_manager, config
            )
            evaluation_results.append({"is_correct": is_correct, "status": eval_status})

            if eval_status != "SUCCESS":
                # Evaluation itself failed. Halt all processing for this question.
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
                break # Exit the loop early
        else:
            # If online eval is off, append a placeholder.
            evaluation_results.append({"is_correct": None, "status": "NOT_EVALUATED"})

    return {
        "status": "SUCCESS",
        "solution_attempts": solution_attempts,
        "evaluation_results": evaluation_results
    }