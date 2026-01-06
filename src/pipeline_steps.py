#======================================================================
#   File: src/pipeline_steps.py
#======================================================================

import logging
import re
import numpy as np
import uuid
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union, Tuple, Optional
import time
from collections import deque

from config import CONFIG
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
    create_hierarchical_parent_solver_prompt,
    create_reverse_validation_prompt,
    create_simplification_prompt,
    create_simplified_sample_solver_prompt,
    create_main_from_simplified_proxy_prompt,
    create_augmentation_with_solution_prompt 
)
from src.utils import save_json, load_json, create_trace_entry
from src.hf_sync import periodic_sync_check
from src.api_manager import GeminiAPIManager, AvalAIAPIManager, OllamaAPIManager
from src.evaluation import evaluate_single_answer_with_llm

def log_time_diagnostic(message: str, start_time: float, indent: int = 0) -> float:
    end_time = time.time()
    elapsed = end_time - start_time
    indent_str = "  " * indent
    if elapsed > 0.001: 
        print(f"{indent_str}⏱️  DIAGNOSTIC: {message} took {elapsed:.4f} seconds.")
    return end_time

def _generate_embeddings(
    texts: List[str],
    embedding_model: SentenceTransformer,
    batch_size: int = 32
) -> np.ndarray:
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

def retrieve(
    target_query: str,
    embedding_model: SentenceTransformer,
    exemplar_questions: List[str],
    embedded_exemplars: np.ndarray,
    top_k: int,
    question_to_index_map: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info(f"Starting retrieval for Top-{top_k} exemplars.")
    
    local_trace = []
    
    current_diag_time = time.time() 
    indent_level = 3
    
    print(f"{'  '*indent_level}--- STARTING DETAILED RETRIEVAL DIAGNOSTICS ---")
    print(f"{'  '*indent_level}Target Query (start): '{target_query[:100]}...'")
    print(f"{'  '*indent_level}Exemplar corpus size: {len(exemplar_questions)}")
    print(f"{'  '*indent_level}Embedded exemplars shape: {embedded_exemplars.shape}")
    print(f"{'  '*indent_level}Requested top_k: {top_k}")

    query_embedding_start_time = time.time()
    query_embedding = _generate_embeddings([target_query], embedding_model)
    current_diag_time = log_time_diagnostic("Generate query embedding", query_embedding_start_time, indent=indent_level)
    print(f"{'  '*indent_level}Query embedding shape: {query_embedding.shape}")
    
    if query_embedding.size == 0:
        logger.error("Failed to generate embedding for the target query. Retrieval cannot proceed.")
        local_trace.append(create_trace_entry("retrieve", "embedding_generation", {"target": target_query}, {"error": "Failed to generate embedding"}, error_info={"msg": "Empty embedding"}))
        return {"status": "FAILURE", "retrieved_indices": [], "retrieved_exemplars": [], "trace": local_trace}
    
    cosine_similarity_start_time = time.time()
    print(f"{'  '*indent_level}Starting cosine similarity calculation (query_embedding shape: {query_embedding.shape}, embedded_exemplars shape: {embedded_exemplars.shape})...")
    similarities = cosine_similarity(query_embedding, embedded_exemplars)[0]
    current_diag_time = log_time_diagnostic("Calculate cosine_similarity", cosine_similarity_start_time, indent=indent_level)
    print(f"{'  '*indent_level}Similarities array shape: {similarities.shape}")
    
    self_match_start_time = time.time()

    if question_to_index_map is not None:
        query_index_in_corpus = question_to_index_map.get(target_query)
        if query_index_in_corpus is not None:
            similarities[query_index_in_corpus] = -np.inf
            print(f"{'  '*indent_level}Self-match found at index {query_index_in_corpus}, set to -np.inf.")
        else:
            print(f"{'  '*indent_level}Target query not found in corpus (no self-match to remove).")
    else:
        print(f"{'  '*indent_level}Warning: question_to_index_map not provided. Skipping self-match check.")
        logger.warning("retrieve() called without question_to_index_map. Self-match detection skipped.")

    current_diag_time = log_time_diagnostic("Handle self-match (O(1) lookup)", self_match_start_time, indent=indent_level)

    k_retrieve_start_time = time.time()
    k_to_retrieve = min(top_k, len(similarities))
    current_diag_time = log_time_diagnostic("Determine k_to_retrieve", k_retrieve_start_time, indent=indent_level)
    print(f"{'  '*indent_level}Effective k_to_retrieve: {k_to_retrieve}")

    print(f"{'  '*indent_level}Starting granular timing for top-k selection...")
        
    argpartition_start_time = time.time()
    print(f"{'  '*indent_level}  Calling np.argpartition on similarities array (shape: {similarities.shape}) for k={k_to_retrieve}...")
    
    partitioned_indices = np.argpartition(similarities, -k_to_retrieve)
    current_diag_time = log_time_diagnostic("np.argpartition (full array)", argpartition_start_time, indent=indent_level)
    print(f"{'  '*indent_level}  Resulting partitioned_indices shape: {partitioned_indices.shape}")

    slice_partitioned_start_time = time.time()
    print(f"{'  '*indent_level}  Slicing to get the top {k_to_retrieve} indices from partitioned_indices...")
    
    top_k_indices_unsorted = partitioned_indices[-k_to_retrieve:]
    current_diag_time = log_time_diagnostic("Slicing partitioned indices", slice_partitioned_start_time, indent=indent_level)
    print(f"{'  '*indent_level}  top_k_indices_unsorted shape: {top_k_indices_unsorted.shape}, Content (first 5): {top_k_indices_unsorted[:5]}...")

    argsort_slice_start_time = time.time()
    print(f"{'  '*indent_level}  Calling np.argsort on the {k_to_retrieve} selected indices based on their similarities...")
    print(f"{'  '*indent_level}  Accessing similarities values for sorting (similarities[top_k_indices_unsorted])...")

    relevant_similarities = similarities[top_k_indices_unsorted]
    
    sorted_order_in_slice = np.argsort(relevant_similarities)[::-1] 
    
    top_k_indices = top_k_indices_unsorted[sorted_order_in_slice]

    current_diag_time = log_time_diagnostic("np.argsort & final sort (on small slice)", argsort_slice_start_time, indent=indent_level)
    print(f"{'  '*indent_level}  Final top_k_indices shape: {top_k_indices.shape}, Content: {top_k_indices.tolist()}")
    print(f"{'  '*indent_level}--- ENDING DETAILED RETRIEVAL DIAGNOSTICS ---")

    logger.info(f"Successfully retrieved indices: {top_k_indices.tolist()}")

    # Log the retrieval result as a trace event
    local_trace.append(create_trace_entry(
        step_name="retrieve",
        sub_step="cosine_similarity_search",
        input_context={"target_query": target_query, "top_k": top_k},
        output_result={"retrieved_indices": top_k_indices.tolist()}
    ))
    
    return {
        "status": "SUCCESS",
        "retrieved_indices": top_k_indices.tolist(),
        "trace": local_trace
    }

def simplify_retrieved_samples(
    retrieved_indices: List[int],
    exemplar_questions: List[str],
    exemplar_solutions: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info("Starting Simplification of Retrieved Samples.")
    
    local_trace = []
    successful_simplifications = []
    failed_indices = []

    if isinstance(api_manager, GeminiAPIManager):
        model_name = config.get('GEMINI_MODEL_NAME_SIMPLIFICATION', config['GEMINI_MODEL_NAME_ADAPTATION'])
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config.get('AVALAI_MODEL_NAME_SIMPLIFICATION', config['AVALAI_MODEL_NAME_ADAPTATION'])
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config.get('OLLAMA_MODEL_NAME_SIMPLIFICATION', config['OLLAMA_MODEL_NAME_ADAPTATION'])
    else:
        raise TypeError(f"Unsupported API manager type: {type(api_manager)}")
        
    temp = config.get("DEFAULT_SIMPLIFICATION_TEMPERATURE", 0.3)

    for idx in retrieved_indices:
        original_q = exemplar_questions[idx]
        original_s = exemplar_solutions[idx]
        original_exemplar_text = EXEMPLAR_FORMAT.format(question=original_q, solution=original_s)
        
        print(f"    -> Processing Sample #{idx} for Simplification...")

        # 1. Simplify Question
        prompt_simp = create_simplification_prompt(original_q, config)
        resp_simp = api_manager.generate_content(prompt_simp, model_name, temp)
        
        local_trace.append(create_trace_entry(
            "simplify_samples", "simplify_question", 
            {"original_idx": idx, "original_q": original_q, "prompt": prompt_simp},
            resp_simp, 
            {"model": model_name, "temp": temp}
        ))
        
        if resp_simp['status'] != 'SUCCESS':
            logger.warning(f"Failed to simplify question for sample {idx}: {resp_simp.get('error_message')}")
            failed_indices.append({"index": idx, "step": "simplify_q", "error": resp_simp})
            continue
            
        simple_q = resp_simp['text'].strip()
        
        # 2. Solve Simplified Question using Original Rationale
        prompt_solve = create_simplified_sample_solver_prompt(simple_q, original_exemplar_text, config)
        resp_solve = api_manager.generate_content(prompt_solve, model_name, temp)
        
        local_trace.append(create_trace_entry(
            "simplify_samples", "solve_simplified_sample",
            {"original_idx": idx, "simple_q": simple_q, "prompt": prompt_solve},
            resp_solve,
            {"model": model_name, "temp": temp}
        ))
        
        if resp_solve['status'] != 'SUCCESS':
            logger.warning(f"Failed to solve simplified question for sample {idx}: {resp_solve.get('error_message')}")
            failed_indices.append({"index": idx, "step": "solve_simple_q", "error": resp_solve})
            continue
            
        simple_solution = resp_solve['text'].strip()
        
        new_exemplar_text = f"Question: {simple_q}\nRationale and Answer: {simple_solution}"
        successful_simplifications.append(new_exemplar_text)
        print(f"      -> Success. New simplified exemplar created.")

    status = "SUCCESS" if successful_simplifications else "FAILURE"
    if successful_simplifications and failed_indices: status = "PARTIAL_SUCCESS"

    return {
        "status": status,
        "simplified_exemplars": successful_simplifications,
        "failed_indices": failed_indices,
        "trace": local_trace
    }

def adapt(
    target_query: str,
    retrieved_indices: List[int],
    exemplar_questions: List[str],
    exemplar_solutions: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info("Starting multi-stage adaptation step.")
    
    local_trace = []
    successful_texts = []
    failed_adaptations = []
    
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

        if config.get('APPLY_NORMALIZATION', False) and not step_failed:
            logger.info(f"Applying normalization to exemplar index {idx}.")
            print(f"    -> Normalizing exemplar {idx}...")
            prompt = create_normalization_prompt(current_text)
            
            print(f"      [API Context] Calling LLM for: Normalization (Exemplar #{idx})")
            response = api_manager.generate_content(prompt, model_name, temperature)
            
            local_trace.append(create_trace_entry(
                "adapt", "normalization",
                {"source_index": idx, "prompt": prompt},
                response,
                {"model": model_name, "temp": temperature}
            ))
            
            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                logger.warning(f"Normalization failed for exemplar {idx}: {response['error_message']}")
                failed_adaptations.append({"source_index": idx, "failed_at_step": "normalization", "error_info": response})
                step_failed = True

        if config.get('APPLY_TRANSFORMATION_1', False) and not step_failed:
            logger.info(f"Applying transformation 1 to exemplar index {idx}.")
            print(f"    -> Applying Transformation 1 to exemplar {idx}...")
            prompt = create_transformation_prompt(target_query, current_text, config, "PROMPT_TEMPLATE_TRANSFORMATION_1")
            print(f"      [API Context] Calling LLM for: Transformation 1 (Exemplar #{idx})")
            response = api_manager.generate_content(prompt, model_name, temperature)

            local_trace.append(create_trace_entry(
                "adapt", "transformation_1",
                {"source_index": idx, "prompt": prompt},
                response,
                {"model": model_name, "temp": temperature}
            ))

            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                logger.warning(f"Transformation 1 failed for exemplar {idx}: {response['error_message']}")
                failed_adaptations.append({"source_index": idx, "failed_at_step": "transformation_1", "error_info": response})
                step_failed = True
        
        if config.get('APPLY_TRANSFORMATION_2', False) and not step_failed:
            logger.info(f"Applying transformation 2 to exemplar index {idx}.")
            print(f"    -> Applying Transformation 2 to exemplar {idx}...")
            prompt = create_transformation_prompt(target_query, current_text, config, "PROMPT_TEMPLATE_TRANSFORMATION_2")
            print(f"      [API Context] Calling LLM for: Transformation 2 (Exemplar #{idx})")
            response = api_manager.generate_content(prompt, model_name, temperature)

            local_trace.append(create_trace_entry(
                "adapt", "transformation_2",
                {"source_index": idx, "prompt": prompt},
                response,
                {"model": model_name, "temp": temperature}
            ))

            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                logger.warning(f"Transformation 2 failed for exemplar {idx}: {response['error_message']}")
                failed_adaptations.append({"source_index": idx, "failed_at_step": "transformation_2", "error_info": response})
                step_failed = True

        if config.get('APPLY_TRANSFORMATION_3', False) and not step_failed:
            logger.info(f"Applying transformation 3 to exemplar index {idx}.")
            print(f"    -> Applying Transformation 3 to exemplar {idx}...")
            prompt = create_transformation_prompt(target_query, current_text, config, "PROMPT_TEMPLATE_TRANSFORMATION_3")
            print(f"      [API Context] Calling LLM for: Transformation 3 (Exemplar #{idx})")
            response = api_manager.generate_content(prompt, model_name, temperature)

            local_trace.append(create_trace_entry(
                "adapt", "transformation_3",
                {"source_index": idx, "prompt": prompt},
                response,
                {"model": model_name, "temp": temperature}
            ))

            if response['status'] == 'SUCCESS':
                current_text = response['text']
            else:
                logger.warning(f"Transformation 3 failed for exemplar {idx}: {response['error_message']}")
                failed_adaptations.append({"source_index": idx, "failed_at_step": "transformation_3", "error_info": response})
                step_failed = True
        
        if not step_failed:
            successful_texts.append(current_text)

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
        "failed_adaptations": failed_adaptations,
        "trace": local_trace
    }

def merge(
    target_query: str,
    adapted_texts: List[str],
    embedding_model: SentenceTransformer,
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    target_count = config.get('TARGET_ADAPTED_SAMPLES_MERGING', 1)

    if not config.get('APPLY_MERGING', False):
        logger.info("APPLY_MERGING is False. Skipping merge step.")
        return {"status": "SKIPPED", "merged_texts": adapted_texts[:target_count], "failed_merges": [], "trace": []}

    logger.info("Starting merging step.")
    local_trace = []
    current_texts = list(adapted_texts)
    failed_merges = []
    
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
        
        local_trace.append(create_trace_entry(
            "merge", f"iteration_{iteration}",
            {"input_pair_count": 2, "prompt": prompt},
            response,
            {"model": model_name, "temp": temperature}
        ))
        
        if response['status'] == 'SUCCESS':
            current_texts.append(response['text'])
        else:
            logger.warning(f"Merging failed: {response['error_message']}. Discarding pair.")
            failed_merges.append({"pair_to_merge": pair_to_merge, "error_info": response})

    return {"status": "SUCCESS", "merged_texts": current_texts, "failed_merges": failed_merges, "trace": local_trace}

def solve(
    target_query: str,
    final_exemplars: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info("Starting final solver step.")
    local_trace = []
    
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
            return {"status": "SUCCESS", "solution_attempts": ["no_retrieval"], "trace": local_trace}

        prompt = create_duplicate_check_prompt(target_query, retrieved_questions)
        
        if isinstance(api_manager, GeminiAPIManager):
            model_name = config['GEMINI_MODEL_NAME_ADAPTATION']
        elif isinstance(api_manager, AvalAIAPIManager):
            model_name = config['AVALAI_MODEL_NAME_ADAPTATION']
        elif isinstance(api_manager, OllamaAPIManager):
            model_name = config['OLLAMA_MODEL_NAME_ADAPTATION']
        else:
            raise TypeError(f"Unsupported API manager type for duplicate check: {type(api_manager)}")
            
        temperature = 0.0 
        
        print("    -> Checking for duplicate questions...")
        print("      [API Context] Calling LLM for: Duplicate Check")
        response = api_manager.generate_content(prompt, model_name, temperature)

        local_trace.append(create_trace_entry(
            "solve", "duplicate_check",
            {"prompt": prompt}, response, {"model": model_name, "temp": temperature}
        ))

        if response['status'] == 'SUCCESS':
            classification = response['text'].strip().lower()
            if "yes" in classification:
                result = "yes"
            elif "no" in classification:
                result = "no"
            else:
                result = "parsing_failed"
            return {"status": "SUCCESS", "solution_attempts": [result], "trace": local_trace}
        else:
            return {"status": "FAILURE", "solution_attempts": [{"status": "FAILURE", "error_info": response}], "trace": local_trace}

    prompt = create_final_reasoning_prompt(target_query, final_exemplars, config) if final_exemplars else create_final_reasoning_prompt_simple(target_query, config)
    logger.info(f"Using {'retrieval-augmented' if final_exemplars else 'simple'} prompt for the solver.")

    if "Error:" in prompt:
        error_msg = f"Failed to create final reasoning prompt: {prompt}"
        logger.error(error_msg)
        return {"status": "FAILURE", "solution_attempts": [{"status": "FAILURE", "error_info": {"error_message": error_msg}}], "trace": local_trace}

    n_attempts = config.get("N_PASS_ATTEMPTS", 1)
    
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
        
        local_trace.append(create_trace_entry(
            "solve", f"attempt_{i+1}",
            {"prompt": prompt}, response, {"model": model_name, "temp": temperature}
        ))

        if response['status'] == 'SUCCESS':
            solution_attempts.append(response['text'])
        else:
            solution_attempts.append({
                "status": "FAILURE",
                "error_info": response
            })
            
    return {"status": "SUCCESS", "solution_attempts": solution_attempts, "trace": local_trace}

def solve_via_main_simplification(
    target_query: str,
    final_exemplars: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info("Starting Solver via Main Question Simplification.")
    local_trace = []
    
    if isinstance(api_manager, GeminiAPIManager):
        model_simp = config.get('GEMINI_MODEL_NAME_SIMPLIFICATION', config['GEMINI_MODEL_NAME_ADAPTATION'])
        model_solve = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, AvalAIAPIManager):
        model_simp = config.get('AVALAI_MODEL_NAME_SIMPLIFICATION', config['AVALAI_MODEL_NAME_ADAPTATION'])
        model_solve = config['AVALAI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, OllamaAPIManager):
        model_simp = config.get('OLLAMA_MODEL_NAME_SIMPLIFICATION', config['OLLAMA_MODEL_NAME_ADAPTATION'])
        model_solve = config['OLLAMA_MODEL_NAME_FINAL_SOLVER']
    else:
        raise TypeError(f"Unsupported API manager type.")

    temp_simp = config.get("DEFAULT_SIMPLIFICATION_TEMPERATURE", 0.3)
    temp_solve = config.get("DEFAULT_FINAL_SOLVER_TEMPERATURE", 1.0)
    
    print("    -> [Simplification] Generating Simplified Main Question...")
    prompt_simp = create_simplification_prompt(target_query, config)
    resp_simp = api_manager.generate_content(prompt_simp, model_simp, temp_simp)
    
    local_trace.append(create_trace_entry(
        "solve_via_simplification", "simplify_main",
        {"prompt": prompt_simp}, resp_simp, {"model": model_simp, "temp": temp_simp}
    ))

    if resp_simp['status'] != 'SUCCESS':
        logger.error(f"Failed to simplify main question: {resp_simp.get('error_message')}")
        return {"status": "FAILURE", "solution_attempts": [{"status": "FAILURE", "error_info": resp_simp, "phase": "simplify_main"}], "trace": local_trace}
    
    simple_main_q = resp_simp['text'].strip()
    print(f"       Simplified Q: '{simple_main_q[:50]}...'")

    print("    -> [Simplification] Solving Simplified Main Question...")
    
    if final_exemplars:
        prompt_solve_simple = create_final_reasoning_prompt(simple_main_q, final_exemplars, config)
    else:
        prompt_solve_simple = create_final_reasoning_prompt_simple(simple_main_q, config)
        
    resp_solve_simple = api_manager.generate_content(prompt_solve_simple, model_solve, temp_solve)
    
    local_trace.append(create_trace_entry(
        "solve_via_simplification", "solve_simple_q",
        {"simple_q": simple_main_q, "prompt": prompt_solve_simple}, resp_solve_simple, {"model": model_solve, "temp": temp_solve}
    ))

    if resp_solve_simple['status'] != 'SUCCESS':
        logger.error(f"Failed to solve simplified main question: {resp_solve_simple.get('error_message')}")
        return {"status": "FAILURE", "solution_attempts": [{"status": "FAILURE", "error_info": resp_solve_simple, "phase": "solve_simple"}], "trace": local_trace}
    
    simple_solution = resp_solve_simple['text']
    
    print("    -> [Simplification] Solving Original Main Question via Proxy...")
    prompt_proxy = create_main_from_simplified_proxy_prompt(target_query, simple_solution, config)
    
    n_attempts = config.get("N_PASS_ATTEMPTS", 1)
    solution_attempts = []
    
    for i in range(n_attempts):
        print(f"       Generating Attempt {i+1}/{n_attempts}...")
        resp_final = api_manager.generate_content(prompt_proxy, model_solve, temp_solve)
        
        local_trace.append(create_trace_entry(
            "solve_via_simplification", f"final_attempt_{i+1}",
            {"prompt": prompt_proxy}, resp_final, {"model": model_solve, "temp": temp_solve}
        ))
        
        if resp_final['status'] == 'SUCCESS':
            solution_attempts.append(resp_final['text'])
        else:
            solution_attempts.append({"status": "FAILURE", "error_info": resp_final})
            
    return {"status": "SUCCESS", "solution_attempts": solution_attempts, "trace": local_trace}

def self_sample(
    target_query: str,
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info("Starting self-sampling step.")
    local_trace = []
    
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
        
        local_trace.append(create_trace_entry(
            "self_sample", f"sample_{i+1}",
            {"prompt": prompt}, response, {"model": model_name, "temp": temperature}
        ))
        
        if response['status'] == 'SUCCESS':
            formatted_text = f"Question: {target_query}\nRationale and Answer: {response['text']}"
            successful_texts.append(formatted_text)
        else:
            failed_samples.append({"sample_index": i, "error_info": response})
    
    if not successful_texts and failed_samples: status = "FAILURE"
    elif successful_texts and failed_samples: status = "PARTIAL_SUCCESS"
    else: status = "SUCCESS"
    
    return {"status": status, "self_sampled_texts": successful_texts, "failed_samples": failed_samples, "trace": local_trace}

def parse_numbered_questions(text: str) -> List[str]:
    questions = []
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
    logger = logging.getLogger(__name__)
    local_trace = []

    if isinstance(api_manager, GeminiAPIManager):
        model_name = config.get('GEMINI_MODEL_NAME_AUGMENTATION', config['GEMINI_MODEL_NAME_ADAPTATION'])
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config.get('AVALAI_MODEL_NAME_AUGMENTATION', config['AVALAI_MODEL_NAME_ADAPTATION'])
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config.get('OLLAMA_MODEL_NAME_AUGMENTATION', config['OLLAMA_MODEL_NAME_ADAPTATION'])
    else:
        raise TypeError(f"Unsupported API manager type for augmentation: {type(api_manager)}")
    
    temperature = config.get("DEFAULT_AUGMENTATION_TEMPERATURE", 0.7)
    aug_mode = config.get("HIERARCHICAL_AUGMENTATION_MODE", "decomposition")
    schedule = config.get("AUGMENTATION_SCHEDULE")

    # --- NEW: Two-Step Augmentation Mode (Solve -> Simplify) ---
    if config.get("HIERARCHICAL_AUGMENTATION_TWO_STEP", False):
        logger.info("Running Two-Step Augmentation (Solve -> Simplify).")
        
        # Step 1: Solve Base Question
        step1_template = config.get("PROMPT_TEMPLATE_AUGMENTATION_STEP1_SOLVER", "final_solver_simple_v2")
        step1_config = config.copy()
        step1_config["PROMPT_TEMPLATE_FINAL_SOLVER_SIMPLE"] = step1_template
        
        prompt_s1 = create_final_reasoning_prompt_simple(target_query, step1_config)
        print(f"    -> [Augment Step 1] Solving base question...")
        resp_s1 = api_manager.generate_content(prompt_s1, model_name, temperature)
        
        local_trace.append(create_trace_entry(
            "augment", "step1_solve_base",
            {"prompt": prompt_s1}, resp_s1, {"model": model_name, "temp": temperature}
        ))

        if resp_s1['status'] != 'SUCCESS':
            return {"status": "FAILURE", "error_info": resp_s1, "trace": local_trace} 
        
        solution_text = resp_s1['text']

        # Step 2: Augment using Context
        prompt_s2 = create_augmentation_with_solution_prompt(target_query, solution_text, n_augmentations, config)
        print(f"    -> [Augment Step 2] Generating simplified question using solution context...")
        response = api_manager.generate_content(prompt_s2, model_name, temperature)
        
        local_trace.append(create_trace_entry(
            "augment", "step2_generate_augmented",
            {"prompt": prompt_s2}, response, {"model": model_name, "temp": temperature}
        ))

        if response['status'] != 'SUCCESS':
            return {"status": "FAILURE", "augmented_questions": [], "error_info": response, "trace": local_trace}
        
        if aug_mode == "simplification":
            augmented_questions = [response['text'].strip()]
        else:
            augmented_questions = parse_numbered_questions(response['text'])
        
        return {"status": "SUCCESS", "augmented_questions": augmented_questions, "error_info": None, "trace": local_trace}

    # --- Existing Schedule-based Logic ---
    if isinstance(schedule, list) and len(schedule) == 2:
        num_calls, questions_per_call = schedule
        logger.info(f"Using augmentation schedule: {num_calls} calls, {questions_per_call} questions per call.")
        
        all_augmented_questions = []
        failed_calls = []

        for i in range(num_calls):
            print(f"    -> Generating augmented questions (Call {i+1}/{num_calls})...")
            prompt = create_augmentation_prompt(target_query, questions_per_call, config)
            response = api_manager.generate_content(prompt, model_name, temperature)
            
            local_trace.append(create_trace_entry(
                "augment", f"schedule_call_{i+1}",
                {"prompt": prompt}, response, {"model": model_name, "temp": temperature}
            ))

            if response['status'] == 'SUCCESS':
                if aug_mode == "simplification":
                    parsed_qs = [response['text'].strip()]
                else:
                    parsed_qs = parse_numbered_questions(response['text'])

                if len(parsed_qs) < questions_per_call and aug_mode == "decomposition":
                    logger.warning(f"Augmentation call {i+1} expected {questions_per_call} questions, but only parsed {len(parsed_qs)}.")
                all_augmented_questions.extend(parsed_qs)
            else:
                logger.error(f"Augmentation call {i+1}/{num_calls} failed: {response['error_message']}")
                failed_calls.append({"call_index": i + 1, "error_info": response})

        status = "SUCCESS"
        if failed_calls and not all_augmented_questions:
            status = "FAILURE"
        elif failed_calls:
            status = "PARTIAL_SUCCESS"
            
        return {
            "status": status, 
            "augmented_questions": all_augmented_questions, 
            "failed_calls": failed_calls,
            "trace": local_trace
        }

    # --- Existing Single-Call Logic ---
    else:
        logger.info(f"Generating {n_augmentations} augmented questions in a single call.")
        prompt = create_augmentation_prompt(target_query, n_augmentations, config)
        
        print(f"    -> Generating {n_augmentations} augmented questions...")
        response = api_manager.generate_content(prompt, model_name, temperature)
        
        local_trace.append(create_trace_entry(
            "augment", "single_call",
            {"prompt": prompt}, response, {"model": model_name, "temp": temperature}
        ))
        
        if response['status'] != 'SUCCESS':
            return {"status": "FAILURE", "augmented_questions": [], "error_info": response, "trace": local_trace}
        
        if aug_mode == "simplification":
            augmented_questions = [response['text'].strip()]
        else:
            augmented_questions = parse_numbered_questions(response['text'])
        
        if len(augmented_questions) < n_augmentations and aug_mode == "decomposition":
            logger.warning(f"Augmentation expected {n_augmentations} questions, but only parsed {len(augmented_questions)}.")
        
        return {"status": "SUCCESS", "augmented_questions": augmented_questions, "error_info": None, "trace": local_trace}

def _select_diverse_questions(questions: List[str], embeddings: np.ndarray, n: int) -> List[str]:
    if embeddings.shape[0] < n: return questions
    
    similarity_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_matrix, 0)
    avg_similarities = similarity_matrix.mean(axis=1)
    selected_indices = np.argsort(avg_similarities)[:n]
    return [questions[i] for i in selected_indices]

def _select_relevant_questions(aug_questions: List[str], aug_embeddings: np.ndarray, sample_embeddings: np.ndarray, n: int) -> List[str]:
    if aug_embeddings.shape[0] < n: return aug_questions
    
    cross_similarity = cosine_similarity(aug_embeddings, sample_embeddings)
    max_similarities = cross_similarity.max(axis=1)
    selected_indices = np.argsort(max_similarities)[-n:][::-1] 
    return [aug_questions[i] for i in selected_indices]

def select_augmented_questions(
    augmented_questions: List[str],
    config: Dict[str, Any],
    embedding_model: SentenceTransformer,
    retrieved_sample_texts: Optional[List[str]] = None
) -> List[str]:
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

def _count_processing_nodes(structure: Any) -> int:
    count = 0
    if isinstance(structure, (list, tuple)):
        count = 1 
        for item in structure:
            count += _count_processing_nodes(item)
    return count

def _process_node_recursively(
    node: Any,
    aug_q_queue: deque,
    retrieved_texts_map: Dict[int, str],
    api_manager: Any,
    config: Dict[str, Any],
    trace_accumulator: List[Dict], # Passed down to capture recursion
    depth: int = 0
) -> Union[str, None]:
    indent = "  " * (depth + 2)
    
    if isinstance(node, int):
        text = retrieved_texts_map.get(node)
        if not text:
            logging.getLogger(__name__).warning(f"{indent}Index {node} not found in retrieved map.")
            return None
        return text

    elif isinstance(node, (list, tuple)):
        child_exemplars = []
        for child in node:
            child_result = _process_node_recursively(child, aug_q_queue, retrieved_texts_map, api_manager, config, trace_accumulator, depth + 1)
            if child_result:
                child_exemplars.append(child_result)
            else:
                logging.getLogger(__name__).warning(f"{indent}Child node {child} failed or returned None.")

        if not aug_q_queue:
            error_msg = "Augmented question queue exhausted! Check AUGMENT_K vs Structure complexity."
            logging.getLogger(__name__).error(error_msg)
            return None
        
        current_aug_q = aug_q_queue.popleft()
        print(f"{indent}-> Processing Node at depth {depth}. Context: {len(child_exemplars)} samples. solving AugQ: '{current_aug_q[:30]}...'")

        if isinstance(api_manager, GeminiAPIManager): model_name = config['GEMINI_MODEL_NAME_ADAPTATION']
        elif isinstance(api_manager, AvalAIAPIManager): model_name = config['AVALAI_MODEL_NAME_ADAPTATION']
        elif isinstance(api_manager, OllamaAPIManager): model_name = config['OLLAMA_MODEL_NAME_ADAPTATION']
        else: return None
        
        if not child_exemplars:
            prompt = create_self_sampling_prompt(current_aug_q, config)
            temp = config.get("SELF_SAMPLING_TEMPERATURE", 0.7)
        else:
            prompt = create_analogical_adaptation_prompt(current_aug_q, child_exemplars, config)
            temp = config.get("DEFAULT_ANALOGICAL_ADAPTATION_TEMPERATURE", 1.0)
            
        response = api_manager.generate_content(prompt, model_name, temp)
        
        # Append to the accumulator instead of a local list, effectively flattening the trace
        trace_accumulator.append(create_trace_entry(
            "analogical_adapt_recursion", f"depth_{depth}",
            {"aug_q": current_aug_q, "child_count": len(child_exemplars), "prompt": prompt},
            response, {"model": model_name, "temp": temp}
        ))
        
        if response['status'] == 'SUCCESS':
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
    api_manager_augment: Any, 
    config: Dict[str, Any],
    embedding_model: SentenceTransformer,
    augmented_questions: Optional[List[str]] = None
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info("Starting recursive analogical adaptation step.")
    local_trace = []
    
    group_sets = config.get("ANALOGICAL_GROUP_SETS", [])
    if not group_sets:
        logger.warning("ANALOGICAL_GROUP_SETS is empty. Skipping.")
        return {"status": "SKIPPED", "reason": "No groups defined.", "trace": local_trace}

    retrieved_texts_map = {}
    for i, idx in enumerate(retrieved_indices):
        q = exemplar_data['questions'][idx]
        s = exemplar_data['solutions'][idx]
        retrieved_texts_map[i + 1] = EXEMPLAR_FORMAT.format(question=q, solution=s)

    total_nodes_needed = 0
    for group in group_sets:
        total_nodes_needed += _count_processing_nodes(group)
    
    logger.info(f"Structure requires {total_nodes_needed} augmented questions total.")

    if config.get("ANALOGICAL_USE_MAIN_QUERY_AS_AUGMENTATION", False):
        logger.info("Identity Augmentation Mode ENABLED. Injecting Main Question into all nodes.")
        final_aug_qs = [target_query] * total_nodes_needed
    else:
        if augmented_questions and len(augmented_questions) >= total_nodes_needed:
            final_aug_qs = augmented_questions[:total_nodes_needed]
        else:
            logger.info(f"Generating {total_nodes_needed} new augmented questions to satisfy structure demand.")
            aug_res = augment_question(target_query, total_nodes_needed, api_manager_augment, config)
            # Capture trace from augmentation call
            if aug_res.get('trace'):
                local_trace.extend(aug_res['trace'])
                
            if aug_res['status'] != 'SUCCESS' and not aug_res.get('augmented_questions'):
                return {"status": "FAILURE", "error_info": aug_res.get('error_info'), "trace": local_trace}
            
            final_aug_qs = aug_res['augmented_questions']
            
            if config.get('SELECTIVE_AUGMENTATION_SAMPLING') and len(final_aug_qs) > total_nodes_needed:
                 final_aug_qs = select_augmented_questions(final_aug_qs, config, embedding_model)
                 if len(final_aug_qs) < total_nodes_needed:
                     logger.warning("Selection reduced pool below required size. Using unselected pool.")
                     final_aug_qs = aug_res['augmented_questions'][:total_nodes_needed]

    if len(final_aug_qs) < total_nodes_needed:
        msg = f"Not enough augmented questions generated. Needed {total_nodes_needed}, got {len(final_aug_qs)}."
        logger.error(msg)
        return {"status": "FAILURE", "error_message": msg, "trace": local_trace}

    aug_q_queue = deque(final_aug_qs)
    
    successful_adaptations = []
    failed_adaptations = []
    
    for group_idx, group_structure in enumerate(group_sets):
        print(f"    -> Processing Top-Level Group #{group_idx + 1}: {group_structure}")
        
        result_text = _process_node_recursively(
            node=group_structure, 
            aug_q_queue=aug_q_queue,
            retrieved_texts_map=retrieved_texts_map,
            api_manager=api_manager, 
            config=config,
            trace_accumulator=local_trace, # Pass the local trace to gather recursive events
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
        "failed_adaptations": failed_adaptations,
        "trace": local_trace
    }

def generate_reasoning_pathways(
    target_query: str,
    api_manager: Any, 
    config: Dict[str, Any]
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    local_trace = []
    mode = config.get("CONSISTENCY_GENERATION_MODE", "distinct_augmentations")
    k_pathways = config.get("CONSISTENCY_PATHWAYS_K", 3)
    
    if isinstance(api_manager, GeminiAPIManager):
        model_name = config.get('GEMINI_MODEL_NAME_AUGMENTATION', config['GEMINI_MODEL_NAME_ADAPTATION'])
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config.get('AVALAI_MODEL_NAME_AUGMENTATION', config['AVALAI_MODEL_NAME_ADAPTATION'])
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config.get('OLLAMA_MODEL_NAME_AUGMENTATION', config['OLLAMA_MODEL_NAME_ADAPTATION'])
    else:
        raise TypeError(f"Unsupported API manager type: {type(api_manager)}")
        
    temp = config.get("CONSISTENCY_LAYER_1_TEMPERATURE", 0.7)
    
    pathways = []
    errors = []

    logger.info(f"Generating reasoning pathways in mode: {mode} (K={k_pathways})")

    if mode == "distinct_augmentations":
        aug_res = augment_question(target_query, k_pathways, api_manager, config)
        if aug_res.get('trace'):
            local_trace.extend(aug_res['trace'])

        if aug_res['status'] == 'FAILURE':
            return {"status": "FAILURE", "pathway_exemplars": [], "error_info": aug_res.get('error_info'), "trace": local_trace}
        
        aug_qs = aug_res['augmented_questions']
        
        if len(aug_qs) < k_pathways:
            logger.warning(f"Augmentation only returned {len(aug_qs)} questions, requested {k_pathways}.")
            
        for i, q in enumerate(aug_qs[:k_pathways]):
            print(f"    -> Solving Pathway {i+1} (Augmented Q): '{q[:50]}...'")
            prompt = create_self_sampling_prompt(q, config)
            resp = api_manager.generate_content(prompt, model_name, temp)
            
            local_trace.append(create_trace_entry(
                "pathways", f"solve_pathway_{i+1}",
                {"pathway_q": q, "prompt": prompt}, resp, {"model": model_name, "temp": temp}
            ))
            
            if resp['status'] == 'SUCCESS':
                exemplar = f"Question: {q}\nRationale and Answer: {resp['text']}"
                pathways.append(exemplar)
            else:
                errors.append(resp)

    elif mode == "single_augmentation_sampling":
        aug_res = augment_question(target_query, 1, api_manager, config)
        if aug_res.get('trace'):
            local_trace.extend(aug_res['trace'])
            
        if aug_res['status'] != 'SUCCESS' or not aug_res['augmented_questions']:
             return {"status": "FAILURE", "pathway_exemplars": [], "error_info": aug_res.get('error_info'), "trace": local_trace}
        
        q = aug_res['augmented_questions'][0]
        logger.info(f"Using single augmented question for sampling: '{q[:50]}...'")
        
        prompt = create_self_sampling_prompt(q, config)
        for i in range(k_pathways):
            print(f"    -> Solving Pathway Sample {i+1} for Single AugQ.")
            resp = api_manager.generate_content(prompt, model_name, temp)
            
            local_trace.append(create_trace_entry(
                "pathways", f"solve_sample_{i+1}",
                {"base_aug_q": q, "prompt": prompt}, resp, {"model": model_name, "temp": temp}
            ))

            if resp['status'] == 'SUCCESS':
                exemplar = f"Question: {q}\nRationale and Answer: {resp['text']}"
                pathways.append(exemplar)
            else:
                errors.append(resp)

    else:
        return {"status": "FAILURE", "error_message": f"Unknown consistency mode: {mode}", "trace": local_trace}

    status = "SUCCESS"
    if not pathways: status = "FAILURE"
    elif errors: status = "PARTIAL_SUCCESS"

    return {"status": status, "pathway_exemplars": pathways, "errors": errors, "trace": local_trace}

def solve_with_group_consistency(
    target_query: str,
    available_exemplars: List[str],
    api_manager: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info("Starting Group-Based Self-Consistency Solving.")
    local_trace = []

    group_candidates = config.get("GROUP_CONSISTENCY_CANDIDATES", [])
    n_samples = config.get("GROUP_CONSISTENCY_SAMPLES_N", 5)
    
    if isinstance(api_manager, GeminiAPIManager):
        model_name = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, AvalAIAPIManager):
        model_name = config['AVALAI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager, OllamaAPIManager):
        model_name = config['OLLAMA_MODEL_NAME_FINAL_SOLVER']
    else:
        raise TypeError(f"Unsupported API manager type for solver: {type(api_manager)}")
        
    temperature = config.get('DEFAULT_PASS_N_SOLVER_TEMPERATURE', 1.0) 

    group_results = []

    for group_idx, indices_tuple in enumerate(group_candidates):
        print(f"\n    -> Processing Consistency Group #{group_idx} (Indices: {indices_tuple})...")
        
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

        prompt = create_final_reasoning_prompt(target_query, group_exemplars, config)
        
        group_attempts = []
        for i in range(n_samples):
            print(f"        -> Generating sample {i+1}/{n_samples} for Group #{group_idx}...")
            response = api_manager.generate_content(prompt, model_name, temperature)
            
            local_trace.append(create_trace_entry(
                "group_consistency", f"group_{group_idx}_sample_{i+1}",
                {"indices": indices_tuple, "prompt": prompt}, response, {"model": model_name, "temp": temperature}
            ))

            if response['status'] == 'SUCCESS':
                group_attempts.append(response['text'])
            else:
                group_attempts.append({"status": "FAILURE", "error_info": response})

        group_results.append({
            "group_id": group_idx,
            "indices_used": indices_tuple,
            "attempts": group_attempts
        })

    return {
        "status": "SUCCESS" if group_results else "FAILURE",
        "group_consistency_results": group_results,
        "trace": local_trace
    }

class ReasoningNode:
    def __init__(self, question: str, depth: int):
        self.id = str(uuid.uuid4())
        self.question = question
        self.depth = depth
        self.children: List['ReasoningNode'] = []
        self.retrieved_context: List[str] = [] 
        self.solution: Optional[str] = None    
        self.solution_attempts: List[str] = [] 
        self.status: str = "PENDING"           

    def to_dict(self) -> Dict[str, Any]:
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
    api_manager_augment: Any, 
    config: Dict[str, Any],
    trace_accumulator: List[Dict]
) -> ReasoningNode:
    logger = logging.getLogger(__name__)
    node = ReasoningNode(current_question, current_depth)
    
    if current_depth >= max_depth:
        return node
    
    print(f"  -> [Tree Build] Expanding Node at Depth {current_depth} (Branching: {branching_factor})...")
    
    local_config = config.copy()
    if config.get("PROMPT_TEMPLATE_HIERARCHICAL_AUGMENTOR"):
        local_config["PROMPT_TEMPLATE_SELF_SAMPLING_AUGMENTOR"] = config["PROMPT_TEMPLATE_HIERARCHICAL_AUGMENTOR"]
        
    aug_res = augment_question(current_question, branching_factor, api_manager_augment, local_config)
    
    # Capture the trace from the augmentation call
    if aug_res.get('trace'):
        trace_accumulator.extend(aug_res['trace'])
    
    if aug_res['status'] != 'SUCCESS' and not aug_res.get('augmented_questions'):
        logger.warning(f"Failed to expand node at depth {current_depth}. Stopping this branch.")
        return node
        
    child_questions = aug_res['augmented_questions']
    
    for child_q in child_questions:
        child_node = build_hierarchical_tree(child_q, current_depth + 1, max_depth, branching_factor, api_manager_augment, config, trace_accumulator)
        node.children.append(child_node)
        
    return node

def _process_leaves(
    root: ReasoningNode,
    target_query: str, 
    exemplar_data: Dict[str, Any],
    embedding_model: SentenceTransformer,
    api_manager_adapt: Any,
    api_manager_solve: Any,
    config: Dict[str, Any],
    trace_accumulator: List[Dict]
) -> None:
    if not root.children:
        print(f"    -> Processing Leaf Node (Depth {root.depth})...")
        
        if config.get("HIERARCHICAL_LEAF_RETRIEVAL_ENABLED", True):
            top_k = config.get("HIERARCHICAL_LEAF_RETRIEVAL_TOP_K", 3)
            query_mode = config.get("HIERARCHICAL_LEAF_RETRIEVAL_QUERY_MODE", "leaf")
            
            if query_mode == "root":
                search_query = target_query
                print(f"      [Retrieval] Mode: ROOT (Using Main Question: '{search_query[:50]}...')")
            else:
                search_query = root.question
                print(f"      [Retrieval] Mode: LEAF (Using Simplified Question: '{search_query[:50]}...')")
            
            ret_res = retrieve(
                search_query, 
                embedding_model, 
                exemplar_data['questions'], exemplar_data['embeddings'], 
                top_k, exemplar_data.get('question_to_index')
            )
            # Capture retrieval trace
            if ret_res.get('trace'):
                trace_accumulator.extend(ret_res['trace'])
            
            if ret_res['status'] == 'SUCCESS':
                adapt_res = adapt(
                    root.question, ret_res['retrieved_indices'], 
                    exemplar_data['questions'], exemplar_data['solutions'], 
                    api_manager_adapt, config
                )
                # Capture adaptation trace
                if adapt_res.get('trace'):
                    trace_accumulator.extend(adapt_res['trace'])
                
                if adapt_res.get('adapted_texts'):
                    root.retrieved_context = adapt_res['adapted_texts']
                    print(f"      -> Leaf retrieved {len(root.retrieved_context)} samples.")
        
        template_name = config.get("PROMPT_TEMPLATE_HIERARCHICAL_LEAF_SOLVER", "final_solver_simple_v1")
        
        if root.retrieved_context:
            local_config = config.copy()
            local_config["PROMPT_TEMPLATE_FINAL_SOLVER"] = template_name 
            prompt = create_final_reasoning_prompt(root.question, root.retrieved_context, local_config)
        else:
            local_config = config.copy()
            local_config["PROMPT_TEMPLATE_FINAL_SOLVER_SIMPLE"] = template_name
            prompt = create_final_reasoning_prompt_simple(root.question, local_config)
            
        model_name = config.get("GEMINI_MODEL_NAME_FINAL_SOLVER") 
        if isinstance(api_manager_solve, AvalAIAPIManager): model_name = config.get("AVALAI_MODEL_NAME_FINAL_SOLVER")
        elif isinstance(api_manager_solve, OllamaAPIManager): model_name = config.get("OLLAMA_MODEL_NAME_FINAL_SOLVER")
        
        temp = config.get("DEFAULT_FINAL_SOLVER_TEMPERATURE", 1.0)
        
        resp = api_manager_solve.generate_content(prompt, model_name, temp)
        
        trace_accumulator.append(create_trace_entry(
            "hierarchical_tree", "solve_leaf",
            {"question": root.question, "prompt": prompt}, resp, {"model": model_name, "temp": temp}
        ))
        
        if resp['status'] == 'SUCCESS':
            root.solution = resp['text']
            root.status = "SOLVED"
        else:
            root.status = "FAILED"
            
    else:
        for child in root.children:
            _process_leaves(child, target_query, exemplar_data, embedding_model, api_manager_adapt, api_manager_solve, config, trace_accumulator)

def propagate_solutions_upward(
    node: ReasoningNode,
    api_manager: Any,
    config: Dict[str, Any],
    trace_accumulator: List[Dict]
) -> None:
    for child in node.children:
        propagate_solutions_upward(child, api_manager, config, trace_accumulator)
    
    if node.status == "SOLVED":
        return

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
    
    model_name = config.get("GEMINI_MODEL_NAME_FINAL_SOLVER") 
    if isinstance(api_manager, AvalAIAPIManager): model_name = config.get("AVALAI_MODEL_NAME_FINAL_SOLVER")
    elif isinstance(api_manager, OllamaAPIManager): model_name = config.get("OLLAMA_MODEL_NAME_FINAL_SOLVER")
    
    temp = config.get("DEFAULT_FINAL_SOLVER_TEMPERATURE", 1.0)
    
    if node.depth > 0:
        resp = api_manager.generate_content(prompt, model_name, temp)
        
        trace_accumulator.append(create_trace_entry(
            "hierarchical_tree", "propagate_solve_node",
            {"depth": node.depth, "prompt": prompt}, resp, {"model": model_name, "temp": temp}
        ))

        if resp['status'] == 'SUCCESS':
            node.solution = resp['text']
            node.status = "SOLVED"
        else:
            node.status = "FAILED"
    else:
        n_attempts = config.get("N_PASS_ATTEMPTS", 1)
        print(f"    -> Root Node detected. Solving {n_attempts} times (Pass@{n_attempts})...")
        
        success_count = 0
        for i in range(n_attempts):
            resp = api_manager.generate_content(prompt, model_name, temp)
            
            trace_accumulator.append(create_trace_entry(
                "hierarchical_tree", f"solve_root_attempt_{i+1}",
                {"prompt": prompt}, resp, {"model": model_name, "temp": temp}
            ))

            if resp['status'] == 'SUCCESS':
                node.solution_attempts.append(resp['text'])
                success_count += 1
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
    api_manager_augment: Any, 
    config: Dict[str, Any]
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info("Starting Hierarchical Augmentation Pipeline.")
    local_trace = []
    
    max_depth = config.get("HIERARCHICAL_TREE_DEPTH", 2)
    branching = config.get("HIERARCHICAL_BRANCHING_FACTOR", 3)
    
    print("\n[HIERARCHICAL] Phase 1: Building Tree...")
    root = build_hierarchical_tree(target_query, 0, max_depth, branching, api_manager_augment, config, local_trace)
    
    print("\n[HIERARCHICAL] Phase 2: Processing Leaves...")
    
    _process_leaves(root, target_query, exemplar_data, embedding_model, api_manager_adapt, api_manager_solve, config, local_trace)
    
    print("\n[HIERARCHICAL] Phase 3: Backward Propagation...")
    propagate_solutions_upward(root, api_manager_solve, config, local_trace)
    
    final_status = "SUCCESS" if root.status == "SOLVED" else "FAILURE"
    
    final_attempts = []
    if root.solution_attempts:
        final_attempts = root.solution_attempts
    elif root.solution:
        final_attempts = [root.solution]

    return {
        "status": final_status,
        "root_solution": root.solution,
        "root_solution_attempts": final_attempts,
        "tree_structure": root.to_dict(),
        "trace": local_trace
    }

def solve_with_analogical_consistency(
    target_query: str,
    exemplar_data: Dict[str, Any],
    embedding_model: SentenceTransformer,
    api_manager_solve: Any,
    api_manager_eval: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Analogical Consistency Check for query: {target_query[:50]}...")
    print("\n" + "="*60)
    print("  [ANALOGICAL CONSISTENCY CHECK] Reverse Validation Mode")
    print("="*60)
    
    local_trace = []

    n_candidates = config.get("REVERSE_VALIDATION_CANDIDATES_N", 5)
    k_validators = config.get("REVERSE_VALIDATION_RETRIEVAL_K", 3)
    n_validation_attempts = config.get("REVERSE_VALIDATION_ATTEMPTS_N", 5)
    
    if isinstance(api_manager_solve, GeminiAPIManager): model_name = config['GEMINI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager_solve, AvalAIAPIManager): model_name = config['AVALAI_MODEL_NAME_FINAL_SOLVER']
    elif isinstance(api_manager_solve, OllamaAPIManager): model_name = config['OLLAMA_MODEL_NAME_FINAL_SOLVER']
    else: raise TypeError(f"Unsupported API manager: {type(api_manager_solve)}")
    
    print(f"\n  [Phase 1] Generating {n_candidates} Candidate Solutions...")
    
    candidate_config = config.copy()
    candidate_config["SELF_SAMPLING_N"] = n_candidates
    
    candidates_result = self_sample(target_query, api_manager_solve, candidate_config)
    
    if candidates_result.get('trace'):
        local_trace.extend(candidates_result['trace'])
    
    if candidates_result['status'] == 'FAILURE':
        logger.error("Failed to generate any candidates.")
        return {"status": "FAILURE", "error": "Candidate generation failed", "trace": local_trace}
        
    candidates = candidates_result['self_sampled_texts'] 
    print(f"    -> Generated {len(candidates)} candidates.")

    print(f"\n  [Phase 2] Retrieving {k_validators} Validators (Ground Truths)...")
    
    retrieval_res = retrieve(
        target_query, embedding_model, 
        exemplar_data['questions'], exemplar_data['embeddings'], 
        top_k=k_validators, question_to_index_map=exemplar_data.get('question_to_index')
    )
    if retrieval_res.get('trace'):
        local_trace.extend(retrieval_res['trace'])
    
    if retrieval_res['status'] != 'SUCCESS' or not retrieval_res['retrieved_indices']:
        logger.error("Failed to retrieve validators.")
        return {"status": "FAILURE", "error": "Validator retrieval failed", "trace": local_trace}
        
    validator_indices = retrieval_res['retrieved_indices']
    validators = []
    for idx in validator_indices:
        validators.append({
            "question": exemplar_data['questions'][idx],
            "ground_truth": exemplar_data['solutions'][idx]
        })
    print(f"    -> Retrieved {len(validators)} validators.")

    print(f"\n  [Phase 3] Reverse Validation Loop ({len(candidates)} Candidates x {len(validators)} Validators x {n_validation_attempts} Attempts)...")
    
    candidate_stats = []
    
    for c_idx, cand_text in enumerate(candidates):
        
        total_attempts = 0
        correct_attempts = 0
        validator_details = []
        
        print(f"    -> Testing Candidate #{c_idx + 1}...")
        
        for v_idx, val in enumerate(validators):
            val_q = val['question']
            val_gt = val['ground_truth']
            
            prompt = create_reverse_validation_prompt(val_q, cand_text, config)
            temp = config.get("DEFAULT_ANALOGICAL_ADAPTATION_TEMPERATURE", 1.0) 
            
            v_correct = 0
            
            for att in range(n_validation_attempts):
                resp = api_manager_solve.generate_content(prompt, model_name, temp)
                
                local_trace.append(create_trace_entry(
                    "reverse_validation", f"solve_candidate_{c_idx}_validator_{v_idx}_attempt_{att}",
                    {"prompt": prompt}, resp, {"model": model_name, "temp": temp}
                ))

                if resp['status'] == 'SUCCESS':
                    eval_res = evaluate_single_answer_with_llm(
                        resp['text'], val_gt, api_manager_eval, config
                    )
                    
                    # Log the evaluator's call if available (assuming evaluate returns standard result, 
                    # but we can't easily capture the evaluator's internal trace here without modifying evaluate_single_answer.
                    # For now, we rely on the evaluate function being largely independent or modify it separately if needed.
                    # But since this file only imports it, we'll just skip evaluating trace for now unless requested.)
                    
                    if eval_res['status'] == 'SUCCESS' and eval_res['is_correct']:
                        v_correct += 1
                        
            
            total_attempts += n_validation_attempts
            correct_attempts += v_correct
            validator_details.append({"validator_idx": v_idx, "score": f"{v_correct}/{n_validation_attempts}"})
            
        consistency_score = correct_attempts / total_attempts if total_attempts > 0 else 0
        print(f"       -> Score: {consistency_score:.2f} ({correct_attempts}/{total_attempts})")
        
        candidate_stats.append({
            "candidate_id": c_idx,
            "candidate_text": cand_text,
            "consistency_score": consistency_score,
            "raw_score": f"{correct_attempts}/{total_attempts}",
            "validator_breakdown": validator_details
        })

    if not candidate_stats:
        return {"status": "FAILURE", "error": "No stats generated", "trace": local_trace}
        
    candidate_stats.sort(key=lambda x: x['consistency_score'], reverse=True)
    
    best_candidate = candidate_stats[0]
    print(f"\n  [Selection] Selected Candidate #{best_candidate['candidate_id'] + 1} with Score {best_candidate['consistency_score']:.2f}")
    
    return {
        "status": "SUCCESS",
        "selected_candidate": best_candidate['candidate_text'],
        "selected_score": best_candidate['consistency_score'],
        "solution_attempts": [best_candidate['candidate_text']], 
        "consistency_stats": candidate_stats,
        "trace": local_trace
    }