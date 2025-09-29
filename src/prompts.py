# src/prompts.py

"""
Prompt engineering module for the Analogical Reasoning RAG project.

This file contains all prompt templates and functions for creating the specific
prompts used in the pipeline. It uses a registry pattern to allow for easy
management and selection of different prompt versions via the main configuration.
"""

from typing import List, Dict

# --- NEW: Standard format for a question-solution pair ---
# This will be used to create a consistent text block for exemplars.
EXEMPLAR_FORMAT = "Question: {question}\nRationale and Answer: {solution}"

# --- 1. Prompt Template Registry ---
PROMPT_TEMPLATES: Dict[str, str] = {
    "standardization_v1": """You are a helpful math assistant.

Your task is to rewrite the following solved math example into a **clear, well-structured, and standardized format** that is easier for a language model to learn from.

Make the question and its reasoning more readable, formal, and useful for solving similar problems — **without changing the logic, reasoning steps, or final answer**.

---

**Guidelines**:

1. Keep the original reasoning process and final answer exactly the same — only improve the writing and formatting.
2. Use correct grammar, clean math language, and consistent formatting.
3. Write the reasoning step by step in a formal, easy-to-follow style.
4. Clarify any unclear or missing reasoning, and highlight helpful patterns.
5. Format the question, rationale, and final answer like a standard math textbook.
6. Do **not** add extra comments or explanations outside the rewritten version.

---

**Original Example (Input)**:
{original_example}

---

**Output Format (Strictly follow this format)**:

Question: [Your rewritten question]

Rationale: [Your rewritten reasoning here, written clearly and step by step]

Final Answer: [Your clean and direct final answer]
""",

    "transformation_v1": """You are provided with a **Main Question** and a **Sample to Transform**.
Your task is to transform the **Sample's Rationale** into a version that is more aligned with the **Main Question**, while ensuring it remains accurate to its own question.

**Main Question:**
{target_query}

**Sample to Transform:**
{text_to_transform}

**Instructions for Transforming the Sample's Rationale:**
1. Analyze the **Sample's Rationale** in the context of its own question to understand its core reasoning.
2. Rewrite the rationale, prioritizing elements most transferable to solving the **Main Question**.
3. Use clear, straightforward language.
4. Do not alter the core logic to solve the **Main Question**, nor modify the **Sample's Question** or its **Final Answer** (as presented in the 'Sample to Transform').
5. Ensure the transformed rationale clearly conveys the reasoning flow.

**Output Format (Strictly follow this format):**
Question: [Original Question from the 'Sample to Transform']
Rationale and Answer: [Transformed Rationale, followed by the Original Answer from the 'Sample to Transform']
""",

    "merging_v1": """You are provided with a main question and two adapted samples, each consisting of a question and its rationale plus answer. Your task is to merge these samples into a single, more potent sample. Combine their rationales into a cohesive and concise rationale that is highly relevant to solving the main question. The merged sample must retain the same format and preserve critical reasoning.

**Main Question:**
{target_query}

**Adapted Sample 1:**
{sample_1}

**Adapted Sample 2:**
{sample_2}

**Instructions:**
- Analyze both samples' rationales in the context of the main question.
- Combine the rationales, integrating complementary reasoning steps, facts, and knowledge.
- Ensure the merged rationale is concise, clear, and directly relevant to the main question.
- Preserve all critical patterns and logical steps useful for the main question.
- Resolve inconsistencies or redundancies, selecting the most accurate and relevant information.
- Create a new question for the merged sample that reflects the combined focus and aligns with the main question's domain.
- Use an answer from one sample if identical; if different, select the most consistent or combine logically. Ensure the final answer part is preserved.
- Ensure the output format matches the input samples format.

**Output Format (Strictly follow this format):**
Question: [New Merged Question]
Rationale and Answer: [Merged Rationale and Answer]
""",

    "final_solver_v1": """**Objective:**
Your task is to solve the **Main Question** by generating a clear, step-by-step **Rationale** and the **Final Answer**.

**Your Method & Constraints:**
1.  **Use Provided Inputs Only:** Base your solution *exclusively* on the given **Main Question** and the **Adapted Samples**. Do **not** use external information or prior knowledge beyond basic arithmetic/logic.
2.  **Reason from Adapted Samples:** Analyze the **Adapted Samples** for logic, methods, and key facts. Intelligently generalize and apply these to the **Main Question**.
3.  **Construct Your Solution:** Develop a logical, step-by-step **Rationale** for the **Main Question**. This rationale should implicitly show how the Adapted Samples informed your thinking, **without explicitly mentioning them**. Perform calculations accurately. Clearly state the **Final Answer**.

**Required Output Format (Strictly Adhere):**
Rationale:
[Your step-by-step rationale for the Main Question]

Final Answer:
[Your final answer to the Main Question]

---
**Inputs:**
**Main Question:**
{main_question_text}
{adapted_samples_block}
---
**Your Solution:**
""",
    
    # NEW: Simple prompt for solving without retrieval.
    "final_solver_simple_v1": """**Objective:**
Your task is to solve the **Main Question** by generating a clear, step-by-step **Rationale** and the **Final Answer**.

**Your Method & Constraints:**
1.  **Construct Your Solution:** Develop a logical, step-by-step **Rationale** for the **Main Question**.
2.  Perform calculations accurately and show your work.
3.  Clearly state the **Final Answer** at the end.

**Required Output Format (Strictly Adhere):**
Rationale:
[Your step-by-step rationale for the Main Question]

Final Answer:
[Your final answer to the Main Question]

---
**Inputs:**
**Main Question:**
{main_question_text}
---
**Your Solution:**
""",

    "evaluator_v1": """Your task is to evaluate if the final answer in 'Model Output' is equivalent to the final answer in 'Ground Truth'.
Both 'Model Output' and 'Ground Truth' may contain intermediate steps (Chain-of-Thought) leading to a final answer.

Follow these two steps precisely:

Step 1: Extract Final Answers
- From 'Model Output', extract only the final numerical or definitive answer. Try to isolate the number or simple expression.
- From 'Ground Truth', extract only the final numerical or definitive answer. Try to isolate the number or simple expression.
- Present these extracted answers clearly. If you cannot confidently extract an answer, state "Extraction Failed" for that part.

Step 2: Evaluate Equivalence
- Compare the 'Extracted Model Answer' with the 'Extracted Ground Truth Answer'.
- If either extraction failed, the evaluation must be 'false'.
- Consider common mathematical equivalences (e.g., "2+2" vs "4", "sqrt(9)" vs "3", "1/2" vs "0.5", "1,000" vs "1000", "$5" vs "5").
- Respond ONLY with the single word 'true' or 'false' for this evaluation part.

Output Format (Strictly follow this format):
Extracted Model Answer: [Your extracted answer from Model Output]
Extracted Ground Truth Answer: [Your extracted answer from Ground Truth]
Evaluation: [true OR false]

---
Model Output:
{model_answer}
---
Ground Truth:
{ground_truth}
---
Begin Output:
""" ,


    "solveablelity_filter" : """You are a filtering assistant for building a math benchmark.

Task: Decide if the given math question can be solved by an isolated LLM with only the text of the question as input.

Rules:

*  Answer yes if the question:

  * Contains all needed information in plain text (numbers, words, symbols).
  * Does not require any external data.

*  Answer no if the question:

  * Refers to images, diagrams, graphs, tables, audio, or any non-textual input.
  * Refers to external resources (links, websites, files).
  * Requires missing context not present in the text itself.

Important:

* Ignore the question’s difficulty, grammar, or factual accuracy.
* Do not explain. Do not output anything except one word.

Output format:

* Only one lowercase word: yes or no.



Question:
{Insert Question Here}

Your output:"""
}


# --- 2. Prompt Creation Functions (MODIFIED) ---

def create_standardization_prompt(original_example: str) -> str:
    """Creates a prompt for the 'standardization' step."""
    template = PROMPT_TEMPLATES["standardization_v1"]
    return template.format(
        original_example=original_example
    )

def create_transformation_prompt(target_query: str, text_to_transform: str) -> str:
    """Creates a prompt for the 'transformation' step."""
    template = PROMPT_TEMPLATES["transformation_v1"]
    return template.format(
        target_query=target_query,
        text_to_transform=text_to_transform
    )

def create_merging_prompt(target_query: str, samples_to_merge: List[str]) -> str:
    """Creates a prompt for the 'merging' step."""
    if len(samples_to_merge) != 2:
        return "Error: create_merging_prompt requires exactly two samples."
    
    template = PROMPT_TEMPLATES["merging_v1"]
    return template.format(
        target_query=target_query,
        sample_1=samples_to_merge[0],
        sample_2=samples_to_merge[1]
    )

def create_final_reasoning_prompt(main_question_text: str, final_adapted_samples: List[str]) -> str:
    """Creates the final prompt for the solver LLM, including adapted samples."""
    if not final_adapted_samples:
        return "Error: At least one adapted sample is required for the final reasoning prompt."

    samples_block = ""
    for i, sample_text in enumerate(final_adapted_samples):
        samples_block += f"\n**Adapted Sample {i+1}:**\n{sample_text}\n"
    
    template = PROMPT_TEMPLATES["final_solver_v1"]
    return template.format(
        main_question_text=main_question_text,
        adapted_samples_block=samples_block.strip()
    )

# NEW: Function for the simple solver prompt
def create_final_reasoning_prompt_simple(main_question_text: str) -> str:
    """Creates the final prompt for the solver LLM without any adapted samples."""
    template_name = "final_solver_simple_v1"
    template = PROMPT_TEMPLATES[template_name]
    return template.format(
        main_question_text=main_question_text
    )

def create_evaluation_prompt(model_answer: str, ground_truth: str) -> str:
    """Creates the prompt for the evaluator LLM."""
    template = PROMPT_TEMPLATES["evaluator_v1"]
    return template.format(
        model_answer=model_answer,
        ground_truth=ground_truth
    )