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
    create_duplicate_check_prompt,
    create_self_sampling_generation_prompt,
    create_self_sampling_augmentation_prompt,
    create_analogical_adaptation_prompt
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


# --- NEW: ANALOGICAL ADAPTATION STEP ---
def analogical_adaptation(
    target_query: str,
    adapted_texts: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Performs an intermediate analogical reasoning step to generate new, synthetic exemplars.
    Groups adapted samples according to the config, has an LLM solve the main query using
    each group, and collects the N generated solutions to pass to the next step.
    """
    logger = logging.getLogger(__name__)

    # 1. Guard Clause: Skip if the feature is disabled in the config.
    if not config.get('APPLY_ANALOGICAL_ADAPTATION', False):
        logger.info("APPLY_ANALOGICAL_ADAPTATION is False. Skipping this step.")
        # Pass the input texts directly to the next step without modification.
        return {"status": "SKIPPED", "newly_generated_exemplars": adapted_texts, "failed_generations": []}

    logger.info("Starting analogical adaptation step.")
    
    # 2. Get parameters from config.
    grouping_config = config.get("ANALOGICAL_ADAPTATION_GROUPING", [])
    samples_per_group = config.get("ANALOGICAL_ADAPTATION_SAMPLES_PER_GROUP", 1)
    
    # Use the powerful solver model and high temperature for this creative step.
    if isinstance(api_manager, GeminiAPIManager):
        model_name = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config['AVALAI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config['OLLAMA_MODEL_NAME_FINAL_SOLVER']
    else:
        raise TypeError(f"Unsupported API manager type for analogical adaptation: {type(api_manager)}")
        
    temperature = config.get('DEFAULT_PASS_N_SOLVER_TEMPERATURE', 1.0)

    all_new_exemplars = []
    failed_generations = []

    # 3. Process each group defined in the configuration.
    for i, group_indices in enumerate(grouping_config):
        group_num = i + 1
        print(f"    -> Processing Analogical Group #{group_num} with indices {group_indices}...")
        
        # 4. Select the samples for the current group.
        # Convert 1-based config indices to 0-based list indices.
        current_group_samples = []
        for idx in group_indices:
            if 0 < idx <= len(adapted_texts):
                current_group_samples.append(adapted_texts[idx - 1])
            else:
                logger.warning(f"Index {idx} in group {group_num} is out of bounds for adapted_texts list (size: {len(adapted_texts)}). Skipping this index.")
        
        if not current_group_samples:
            logger.warning(f"Group {group_num} is empty after index validation. Skipping generation for this group.")
            continue

        # 5. Generate N samples for this group.
        for j in range(samples_per_group):
            sample_num = j + 1
            print(f"      -> Generating synthetic sample {sample_num}/{samples_per_group} for group #{group_num}...")
            
            # Use the new, dedicated prompt for this intermediate task.
            prompt = create_analogical_adaptation_prompt(target_query, current_group_samples, config)
            
            print(f"        [API Context] Calling LLM for: Analogical Adaptation (Group #{group_num}, Sample #{sample_num})")
            response = api_manager.generate_content(prompt, model_name, temperature)

            if response['status'] == 'SUCCESS':
                # Format the successful generation into a standard exemplar format.
                formatted_exemplar = EXEMPLAR_FORMAT.format(
                    question=target_query, 
                    solution=response['text']
                )
                all_new_exemplars.append(formatted_exemplar)
            else:
                logger.warning(f"Analogical adaptation failed for group {group_num}, sample {sample_num}: {response['error_message']}")
                failed_generations.append({"group_num": group_num, "sample_num": sample_num, "error_info": response})

    # 6. Determine final status and return structured output.
    if not all_new_exemplars and failed_generations:
        final_status = "FAILURE"
    elif all_new_exemplars and failed_generations:
        final_status = "PARTIAL_SUCCESS"
    else:
        final_status = "SUCCESS"
        
    return {
        "status": final_status,
        "newly_generated_exemplars": all_new_exemplars,
        "failed_generations": failed_generations
    }


# <<< --- START OF NEW CODE --- >>>
# --- 2b. SELF-SAMPLING STEP ---
def generate_synthetic_samples(
    target_query: str,
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates N synthetic exemplars. Supports two modes:
    1. Standard: Solves the same target query N times.
    2. Augmented: Generates N distinct questions first, then solves each one.
    """
    logger = logging.getLogger(__name__)
    n_samples = config.get("N_SELF_SAMPLES", 3)
    
    # We use the powerful solver model and high temperature for diverse outputs
    if isinstance(api_manager, GeminiAPIManager):
        model_name = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config['AVALAI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config['OLLAMA_MODEL_NAME_FINAL_SOLVER']
    else:
        raise TypeError(f"Unsupported API manager type for self-sampling: {type(api_manager)}")
        
    temperature = config.get('DEFAULT_PASS_N_SOLVER_TEMPERATURE', 1.0)
    
    synthetic_samples = []
    failed_generations = []
    
    # --- AUGMENTED MODE ---
    if config.get('APPLY_SELF_SAMPLING_AUGMENTATION', False):
        logger.info("Starting AUGMENTED self-sampling to generate distinct exemplars.")
        print("\n[STEP 1a] AUGMENT: Generating distinct question variations...")
        
        # Step A: Generate N distinct questions
        aug_prompt = create_self_sampling_augmentation_prompt(target_query, n_samples, config)
        print("  [API Context] Calling LLM for: Question Augmentation")
        aug_response = api_manager.generate_content(aug_prompt, model_name, temperature)

        if aug_response['status'] != 'SUCCESS':
            logger.error(f"Question augmentation failed: {aug_response['error_message']}")
            return {"status": "FAILURE", "synthetic_samples": [], "failed_generations": [{"attempt_number": "N/A", "failed_at_step": "augmentation", "error_info": aug_response}]}
        
        # Parse the augmented questions from the response
        # Try to find lines that start with a number and a dot
        augmented_questions = re.findall(r'^\d+\.\s*(.*)', aug_response['text'], re.MULTILINE)
        if not augmented_questions:
             logger.warning("Could not parse augmented questions from LLM response using regex. Falling back to splitting by newline.")
             # Fallback: Split by newline and filter empty lines if regex fails
             augmented_questions = [q.strip() for q in aug_response['text'].strip().split('\n') if q.strip()]

        # Ensure we don't process more than requested if the LLM over-generated
        augmented_questions = augmented_questions[:n_samples]
        print(f"  -> Generated {len(augmented_questions)} augmented questions. Now solving each one...")

        # Step B: Solve each augmented question
        for i, aug_question in enumerate(augmented_questions):
            print(f"    -> [STEP 1b] SOLVE: Solving augmented sample {i+1}/{len(augmented_questions)}...")
            
            solve_prompt = create_self_sampling_generation_prompt(aug_question, config)
            print(f"      [API Context] Calling LLM for: Solving Augmented Question (Attempt #{i+1})")
            
            solve_response = api_manager.generate_content(solve_prompt, model_name, temperature)

            if solve_response['status'] == 'SUCCESS':
                # Format using the AUGMENTED question, not the original target query
                formatted_exemplar = EXEMPLAR_FORMAT.format(question=aug_question, solution=solve_response['text'])
                synthetic_samples.append(formatted_exemplar)
            else:
                logger.warning(f"Augmented sample solution failed for attempt {i+1}: {solve_response['error_message']}")
                failed_generations.append({"attempt_number": i+1, "failed_at_step": "solving_augmented", "error_info": solve_response})

    # --- STANDARD MODE ---
    else:
        logger.info("Starting STANDARD self-sampling to generate synthetic exemplars.")
        prompt = create_self_sampling_generation_prompt(target_query, config)
        for i in range(n_samples):
            print(f"    -> Generating synthetic sample {i+1}/{n_samples}...")
            print(f"      [API Context] Calling LLM for: Synthetic Sample Generation (Attempt #{i+1})")
            
            response = api_manager.generate_content(prompt, model_name, temperature)

            if response['status'] == 'SUCCESS':
                # Format using the ORIGINAL target query
                formatted_exemplar = EXEMPLAR_FORMAT.format(question=target_query, solution=response['text'])
                synthetic_samples.append(formatted_exemplar)
            else:
                logger.warning(f"Synthetic sample generation failed for attempt {i+1}: {response['error_message']}")
                failed_generations.append({"attempt_number": i+1, "error_info": response})
    
    # Determine the final status
    if not synthetic_samples and failed_generations:
        final_status = "FAILURE"
    elif synthetic_samples and failed_generations:
        final_status = "PARTIAL_SUCCESS"
    else:
        final_status = "SUCCESS"

    return {
        "status": final_status,
        "synthetic_samples": synthetic_samples,
        "failed_generations": failed_generations
    }
# <<< --- END OF NEW CODE --- >>>


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