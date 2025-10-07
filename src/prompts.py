# src/prompts.py

"""
Prompt engineering module for the Analogical Reasoning RAG project.

This file centralizes all prompt templates and the functions that construct
the final prompts used in the pipeline. It employs a registry pattern
(the `PROMPT_TEMPLATES` dictionary) to allow for easy management, versioning,
and selection of different prompt variations via the main configuration file.

This rewritten version includes hardened instructions in key templates to ensure
LLMs adhere strictly to the expected output format, reducing parsing errors
in downstream pipeline steps.
"""

from typing import List, Dict, Any

# Define the standard text format for a question-solution pair.
# This ensures consistency when constructing and parsing exemplars.
EXEMPLAR_FORMAT = "Question: {question}\nRationale and Answer: {solution}"

# --- 1. Prompt Template Registry ---
# A dictionary holding all versioned prompt templates for the project.
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
**Output Format (Strictly follow this format, including the exact headers 'Question:', 'Rationale:', and 'Final Answer:')**:

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

**Output Format (Strictly follow this format, including the exact headers 'Question:' and 'Rationale and Answer:')**:
Question: [Original Question from the 'Sample to Transform']
Rationale and Answer: [Transformed Rationale, followed by the Original Answer from the 'Sample to Transform']
""",

    "merging_v1": """You are provided with a main question and two samples, each consisting of a question and its rationale plus answer. Your task is to merge these samples into a single, more potent sample. Combine their rationales into a cohesive and concise rationale that is highly relevant to solving the main question. The merged sample must retain the same format and preserve critical reasoning.

**Main Question:**
{target_query}

**Sample 1:**
{sample_1}

**Sample 2:**
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

    "merging_2" : """You are an expert in analogical reasoning for mathematical problem-solving.  
Your task is to merge two solved math problems into a single new, synthesized problem-solution pair that will serve as a strong exemplar for solving a target question.  

These two input samples are called Parent Sample A and Parent Sample B, because they will be merged to form one new "child" example.   
The child example should combine the most relevant and valuable reasoning patterns from its parents in a coherent, context-aware manner.


<Your Objective> 
Create a new merged example that:
1. Retains the core reasoning structures and mathematical logic from both Parent A and Parent B.
2. Selectively integrates only the parts most relevant to the Target Question. 
3. Produces a new, logically consistent and useful exemplar that the LLM can learn from for analogical reasoning.
</Your Objective> 


</Core Guidelines> 
1. Principled Construction from Parent Materials   
- Use the reasoning chains, strategies, and key steps from both parents.  
- Avoid inventing completely new mathematical methods.  
- Preserve the internal logic and structure of the parent materials.

2. Guided by the Target Question   
- The merge must be performed in the context of the Target Question.   
- Keep and emphasize the parts of each parent that are most helpful for solving the target question.   
- If one parent is much more relevant, prioritize it — the merge does not need to be symmetrical.

3. Coherence and Controlled Generation  
- You may add minor connective text to make the final reasoning smooth and clear.  
- The merged result must be mathematically correct, coherent, and self-contained.

4. Superficial Re-contextualization 
- Adapt the merged sample’s phrasing and structure so it stylistically resembles the Target Question.  
- Do not alter the core mathematics — only adjust presentation and framing.

5. Relevance-Driven Asymmetry 
- It is acceptable if the final merged sample resembles one parent more than the other.  
- Discard irrelevant parts. The goal is maximum usefulness, not balance. 
</Core Guidelines> 


</Input Materials>  
Parent Sample A: 
{sample_1}

Parent Sample B:
{sample_2}

Target Question: 
{target_query}
</Input Materials> 


<Output Instructions> 
You must output ONLY in the following format.   
Do not include any explanations, comments, or text outside this format.  

Output Format (Strictly follow this format):  
Question: [New Merged Question]
Rationale and Answer: [Merged Rationale and Answer]
</Output Instructions> 
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
    
    # --- NEW: final_solver_v2 Template ---
    "final_solver_v2": """You are an expert in analogical reasoning, highly skilled at identifying and extracting patterns, reasoning pathways, problem-solving strategies, and conceptual frameworks from similar solved examples. Your primary task is to solve the main question by drawing meaningful analogies from the provided solved examples.



<Instructions>
Carefully analyze each example: pinpoint common reasoning steps, patterns (including structural similarities, logical sequences, mathematical transformations, conceptual mappings, or recurring problem-solving techniques), and effective strategies that led to the final answers. Focus on extracting only the most useful and relevant elements from these examples as supportive guides—treat them as verified, correct rationales to inform your approach, but not as strict templates that must be replicated exactly. Instead, adapt them flexibly to fit the unique aspects of the main question, even when surface details differ, while prioritizing your own independent reasoning to develop a robust solution.
</Instructions>

<Solved Examples>
{examples_block}
</Solved Examples>

<Main Question to Solve>
{main_question_text}
</Main Question to Solve>



<Your Answer/Output Format>
Rationale:
[Your step-by-step rationale for the Main Question]

Final Answer:
[Your final answer to the Main Question]
</Your  Answer/Output Format>
""",

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
""",

    "duplicate_question_check_v1": """You are a text comparison assistant. Your task is to determine if the 'Main Question' is identical to ANY of the questions in the 'Retrieved Questions' list.

**Rules:**
1.  Compare the text of the 'Main Question' against each 'Retrieved Question' verbatim.
2.  Ignore differences in whitespace, capitalization, or minor punctuation unless they change the meaning.
3.  If you find an exact match, your job is done.
4.  Your entire output must be a single word: **yes** or **no**. Do not provide any explanation.

---
**Main Question:**
{main_question_text}
---
**Retrieved Questions:**
{retrieved_questions_block}
---
**Is there an exact match? (yes/no):**
""",
}


# --- 2. Prompt Creation Functions ---
# These functions abstract the process of selecting and formatting a template.

def create_standardization_prompt(original_example: str) -> str:
    """Creates a prompt for the 'standardization' pipeline step."""
    template = PROMPT_TEMPLATES["standardization_v1"]
    return template.format(original_example=original_example)

def create_transformation_prompt(target_query: str, text_to_transform: str) -> str:
    """Creates a prompt for the 'transformation' pipeline step."""
    template = PROMPT_TEMPLATES["transformation_v1"]
    return template.format(target_query=target_query, text_to_transform=text_to_transform)

def create_merging_prompt(target_query: str, samples_to_merge: List[str]) -> str:
    """Creates a prompt for the 'merging' pipeline step."""
    if len(samples_to_merge) != 2:
        # This guard clause prevents errors if the merging logic provides the wrong number of samples.
        return "Error: create_merging_prompt requires exactly two samples."
    
    template = PROMPT_TEMPLATES["merging_v1"]
    return template.format(target_query=target_query, sample_1=samples_to_merge[0], sample_2=samples_to_merge[1])

def create_final_reasoning_prompt(main_question_text: str, final_examples: List[str], config: Dict[str, Any]) -> str:
    """
    Creates the final prompt for the solver, including processed examples (RAG).
    This function now dynamically selects the template and formats the examples
    based on the template's requirements.
    """
    if not final_examples:
        return "Error: At least one example is required for the RAG-based final reasoning prompt."

    template_name = config.get("PROMPT_TEMPLATE_FINAL_SOLVER", "final_solver_v2")
    template = PROMPT_TEMPLATES[template_name]
    
    # --- MODIFIED: Dynamic block creation based on template ---
    if template_name == "final_solver_v2":
        # Format for the new v2 template with XML-style tags
        examples_block = ""
        for i, sample_text in enumerate(final_examples):
            examples_block += f"<Example {i+1}>\n{sample_text}\n</Example {i+1}>\n\n"
        return template.format(main_question_text=main_question_text, examples_block=examples_block.strip())
    
    elif template_name == "final_solver_v1":
        # Original formatting for the v1 template
        samples_block = ""
        for i, sample_text in enumerate(final_examples):
            samples_block += f"\n**Adapted Sample {i+1}:**\n{sample_text}\n"
        return template.format(main_question_text=main_question_text, adapted_samples_block=samples_block.strip())
    
    else:
        # Fallback or error for unknown templates
        return f"Error: Unknown final solver template '{template_name}' specified in config."


def create_final_reasoning_prompt_simple(main_question_text: str, config: Dict[str, Any]) -> str:
    """Creates the final prompt for the solver without any adapted samples (No RAG)."""
    template_name = config.get("PROMPT_TEMPLATE_FINAL_SOLVER_SIMPLE", "final_solver_simple_v1")
    template = PROMPT_TEMPLATES[template_name]
    return template.format(main_question_text=main_question_text)

def create_evaluation_prompt(model_answer: str, ground_truth: str, config: Dict[str, Any]) -> str:
    """Creates the prompt for the evaluator LLM."""
    template_name = config.get("PROMPT_TEMPLATE_EVALUATOR", "evaluator_v1")
    template = PROMPT_TEMPLATES[template_name]
    return template.format(model_answer=model_answer, ground_truth=ground_truth)

def create_duplicate_check_prompt(main_question_text: str, retrieved_questions: List[str]) -> str:
    """Creates the prompt for the special duplicate question check task."""
    template = PROMPT_TEMPLATES["duplicate_question_check_v1"]
    
    # Format the list of retrieved questions into a numbered block for the LLM.
    retrieved_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(retrieved_questions))
        
    return template.format(main_question_text=main_question_text, retrieved_questions_block=retrieved_block.strip())