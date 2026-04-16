# src/dataset_builder.py
"""
Dataset construction utilities for the Analogical Math RAG project.

The purpose of this module is to implement the "new feature" described by the
user: iterate over the existing exemplar corpus, choose a random query Q, find
its two most similar samples (A and B), solve Q in a two‑shot fashion using A
and B, verify that the answer is correct with an LLM evaluator, and if so
record the material and a structured input/output pair for downstream training.

The generated JSON contains two sections:
* ``materials`` – the raw examples selected (indices, questions & answers for
  Q, A and B).
* ``dataset`` – the actual training instances where ``input`` is a list of two
  solved versions of the query (one solved using A, the other solved using B)
  and ``output`` is a short string label indicating which examples were used
  and that the two‑shot call succeeded.  We also include the full two‑shot
  reasoning as ``two_shot_solution`` so that it can be inspected later.

The configuration keys used by this module are documented in ``config.py``.

The construction function will write its results to
``CONFIG['OUTPUTS_DIR']/constructed_dataset.json`` and returns the same
structure so callers (e.g. notebook cells) can inspect it directly.

The routine is intentionally simple; it re‑uses existing helpers from the
pipeline and evaluation modules instead of re‑implementing embedding or
evaluation logic.
"""

import os
import random
import logging
from typing import Dict, Any, List

from sentence_transformers import SentenceTransformer

from src.pipeline_steps import retrieve
from src.prompts import create_final_reasoning_prompt
from src.utils import save_json
from src.evaluation import evaluate_single_answer_with_llm
from src.hf_sync import periodic_sync_check

# Import API manager classes solely for model selection logic
from src.api_manager import GeminiAPIManager, AvalAIAPIManager, OllamaAPIManager


def construct_two_shot_dataset(
    exemplar_data: Dict[str, Any],
    embedding_model: SentenceTransformer,
    api_manager_solver: Any,
    api_manager_eval: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Builds a labelled dataset using the two‑shot procedure described above.

    Args:
        exemplar_data: dictionary with keys ``questions`` (list of str),
            ``solutions`` (list of str), ``embeddings`` (np.ndarray), and
            optionally ``question_to_index`` for fast self‑match filtering.
        embedding_model: a ``SentenceTransformer`` used by ``retrieve``.
        api_manager_solver: manager used for solving prompts (Gemini/AvalAI/Ollama).
        api_manager_eval: manager used for evaluation calls.
        config: global configuration dictionary (``CONFIG`` plus any overrides).

    Returns:
        A dictionary with two top‑level keys: ``materials`` and ``dataset``.
        Each is a list of entries as described in the module docstring.
    """
    logger = logging.getLogger(__name__)

    # read parameters from config, with sensible fallbacks
    max_search = config.get("DATASET_CONSTRUCTION_MAX_SEARCH", 1000)
    max_members = config.get("DATASET_CONSTRUCTION_MAX_MEMBERS", 100)
    seed = config.get("DATASET_CONSTRUCTION_RANDOM_SEED")
    if seed is not None:
        random.seed(seed)

    # determine solver model name & temperature based on manager type
    if isinstance(api_manager_solver, GeminiAPIManager):
        solver_model = config.get("GEMINI_MODEL_NAME_FINAL_SOLVER")
    elif isinstance(api_manager_solver, AvalAIAPIManager):
        solver_model = config.get("AVALAI_MODEL_NAME_FINAL_SOLVER")
    elif isinstance(api_manager_solver, OllamaAPIManager):
        solver_model = config.get("OLLAMA_MODEL_NAME_FINAL_SOLVER")
    else:
        solver_model = config.get("GEMINI_MODEL_NAME_FINAL_SOLVER")
    solver_temp = config.get("DATASET_CONSTRUCTION_SOLVER_TEMPERATURE", config.get("DEFAULT_FINAL_SOLVER_TEMPERATURE"))

    # evaluator selections
    if isinstance(api_manager_eval, GeminiAPIManager):
        eval_model = config.get("GEMINI_MODEL_NAME_EVALUATOR")
    elif isinstance(api_manager_eval, AvalAIAPIManager):
        eval_model = config.get("AVALAI_MODEL_NAME_EVALUATOR")
    elif isinstance(api_manager_eval, OllamaAPIManager):
        eval_model = config.get("OLLAMA_MODEL_NAME_EVALUATOR")
    else:
        eval_model = config.get("GEMINI_MODEL_NAME_EVALUATOR")
    eval_temp = config.get("DATASET_CONSTRUCTION_EVALUATOR_TEMPERATURE", config.get("DEFAULT_EVALUATOR_TEMPERATURE"))

    questions: List[str] = exemplar_data.get("questions", [])
    solutions: List[str] = exemplar_data.get("solutions", [])
    embeddings = exemplar_data.get("embeddings")

    if not questions or embeddings is None:
        logger.error("Exemplar data is missing questions or embeddings; cannot build dataset.")
        return {"materials": [], "dataset": []}

    results = {"materials": [], "dataset": []}

    total_questions = len(questions)
    search_count = 0
    dataset_count = 0

    logger.info(f"Starting two-shot dataset construction (max_search={max_search}, max_members={max_members})")

    while search_count < max_search and dataset_count < max_members:
        search_count += 1

        q_idx = random.randrange(total_questions)
        q_text = questions[q_idx]
        q_ans = solutions[q_idx]

        # retrieve two nearest neighbors (self-match removed by retrieve())
        retrieval = retrieve(
            target_query=q_text,
            embedding_model=embedding_model,
            exemplar_questions=questions,
            embedded_exemplars=embeddings,
            top_k=3,
            question_to_index_map=exemplar_data.get("question_to_index")
        )

        if retrieval.get("status") != "SUCCESS":
            logger.warning(f"Retrieval failed for Q#{q_idx}; skipping (status {retrieval.get('status')}).")
            continue
        neigh = retrieval.get("retrieved_indices", [])
        if len(neigh) < 2:
            # corpus too small or something went wrong
            continue

        a_idx, b_idx = neigh[0], neigh[1]
        a_text, a_ans = questions[a_idx], solutions[a_idx]
        b_text, b_ans = questions[b_idx], solutions[b_idx]

        # optionally override the prompt template for this run
        original_template = config.get("PROMPT_TEMPLATE_FINAL_SOLVER")
        if config.get("DATASET_CONSTRUCTION_PROMPT_TEMPLATE"):
            config["PROMPT_TEMPLATE_FINAL_SOLVER"] = config["DATASET_CONSTRUCTION_PROMPT_TEMPLATE"]

        # 2-shot solve
        two_examples = [f"{a_text}\nSolution: {a_ans}", f"{b_text}\nSolution: {b_ans}"]
        prompt_2shot = create_final_reasoning_prompt(q_text, two_examples, config)
        resp_2shot = api_manager_solver.generate_content(prompt_2shot, solver_model, solver_temp)

        # restore the original template
        if config.get("DATASET_CONSTRUCTION_PROMPT_TEMPLATE"):
            config["PROMPT_TEMPLATE_FINAL_SOLVER"] = original_template

        if resp_2shot.get("status") != "SUCCESS":
            continue
        two_shot_text = resp_2shot.get("text", "")

        # evaluate correctness
        eval_result = evaluate_single_answer_with_llm(two_shot_text, q_ans, api_manager_eval, config)
        if not eval_result.get("is_correct"):
            # only accept items that passed the check
            continue

        # one-shot solves for input
        # reapply template override if necessary
        if config.get("DATASET_CONSTRUCTION_PROMPT_TEMPLATE"):
            config["PROMPT_TEMPLATE_FINAL_SOLVER"] = config["DATASET_CONSTRUCTION_PROMPT_TEMPLATE"]
        prompt_as = create_final_reasoning_prompt(q_text, [f"{a_text}\nSolution: {a_ans}"], config)
        prompt_bs = create_final_reasoning_prompt(q_text, [f"{b_text}\nSolution: {b_ans}"], config)
        if config.get("DATASET_CONSTRUCTION_PROMPT_TEMPLATE"):
            config["PROMPT_TEMPLATE_FINAL_SOLVER"] = original_template

        resp_a = api_manager_solver.generate_content(prompt_as, solver_model, solver_temp)
        resp_b = api_manager_solver.generate_content(prompt_bs, solver_model, solver_temp)

        a_solved = resp_a.get("text") if resp_a.get("status") == "SUCCESS" else resp_a
        b_solved = resp_b.get("text") if resp_b.get("status") == "SUCCESS" else resp_b

        label = f"{a_idx}_{b_idx}__Solved_True"

        results["materials"].append({
            "q_idx": q_idx,
            "question": q_text,
            "solution": q_ans,
            "a_idx": a_idx,
            "a_question": a_text,
            "a_solution": a_ans,
            "b_idx": b_idx,
            "b_question": b_text,
            "b_solution": b_ans,
        })

        results["dataset"].append({
            "input": [
                {"Q": q_text, "A_solved": a_solved},
                {"Q": q_text, "B_solved": b_solved}
            ],
            "output": label,
            "two_shot_solution": two_shot_text
        })

        dataset_count += 1

        # periodically sync to hub if enabled
        periodic_sync_check(search_count, config)

    # save results
    output_path = os.path.join(config["OUTPUTS_DIR"], "constructed_dataset.json")
    save_json(results, output_path)
    logger.info(f"Dataset construction finished (searched {search_count}, created {dataset_count} entries).\n"
                f"Saved output to {output_path}")

    return results
