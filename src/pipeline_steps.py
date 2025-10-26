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
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union

# Import our custom modules
from src.prompts import (
    EXEMPLAR_FORMAT,
    create_normalization_prompt,
    create_transformation_prompt,
    create_merging_prompt,
    create_final_reasoning_prompt,
    create_final_reasoning_prompt_simple,
    create_duplicate_check_prompt
)
# MODIFIED: Import manager classes for type checking
from src.api_manager import GeminiAPIManager, AvalAIAPIManager, OllamaAPIManager


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