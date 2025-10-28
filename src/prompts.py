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
    
    "transformation_shallow" : """<Objective>
Your task is to transform the given Sample (which includes a question and its step-by-step rationale) into a new version that becomes more analogous and relevant to the Target Question.
The transformation should be directed toward the Target Question, meaning every change you make should help the transformed sample better reflect, match, or resonate with the Target Question’s theme, context, or style — while strictly preserving the sample’s original reasoning path and final answer.
</Objective>

<Transformation Guidelines>
Target-Directed Adaptation:
- Adapt the sample's theme, context, and entities to mirror those in the Target Question.
- Think of this as reframing the sample so it feels like it belongs to the same world or problem type as the Target Question.
- Crucially, any changes made to the question (e.g., numbers, objects, context) must be consistently and accurately reflected throughout the transformed rationale.


Preserve Core Reasoning:
- The underlying logical pathway and mathematical operations must remain identical.
- Do not alter the sequence of steps, the problem-solving strategy, or how one calculation leads to the next. The core method of solving must be perfectly preserved.

Restrict Transformations to the Surface Level:
- Your transformations must be limited to the surface and contextual layers of the problem.
- This includes changing nouns (entities, objects), numbers and quantities (while ensuring the logic and final answer are preserved), and the overall setting or story.
- Do not change the fundamental problem structure or the reasoning schema. The goal is to change the "story" of the problem, not the "logic" of the solution.

Maintain Naturalness, Clarity, and Safety:
- The transformed question and rationale must remain natural, realistic, and logically coherent.
- Avoid any unnatural, illogical, or meaningless transformations (e.g., “a cat eats an apple”).
- If a transformation cannot be made safely or meaningfully, keep the sample as close to the original as possible rather than forcing changes.
- Always prioritize clarity, realism, and logical consistency over aggressive transformation.

Keep the Final Answer Unchanged:
- The final numerical or categorical answer at the end of the rationale must not be changed. It should remain exactly as it was in the original sample.

</Transformation Guidelines>
</Example Transformation>

<Example Input>
Target Question: A laboratory has 35 beakers. A new experiment requires 5 beakers per station. If the lab manager sets up 4 stations, how many beakers are left over?

Sample to Transform:
Question: A baker has 50 cookies. He decides to package them into boxes, with each box holding 6 cookies. If he sells 7 boxes, how many cookies does he have left?

Rationale:
To find the remaining cookies, we first need to calculate how many cookies were sold.
The baker sold 7 boxes, and each box contains 6 cookies.
Total cookies sold = 7 boxes * 6 cookies/box = 42 cookies.
The baker started with 50 cookies.
Remaining cookies = Initial amount - Amount sold = 50 - 42 = 8 cookies.

Final Answer: 8
</Example Input>

</Example Output>
Question: A scientist starts with 28 test tubes for an analysis. She arranges them into racks, with each rack holding 4 test tubes. If she uses 5 full racks for her experiment, how many test tubes are left unused?

Rationale:
To find the remaining test tubes, we first need to calculate how many test tubes were used.
The scientist used 5 racks, and each rack contains 4 test tubes.
Total test tubes used = 5 racks * 4 test tubes/rack = 20 test tubes.
The scientist started with 28 test tubes.
Remaining test tubes = Initial amount - Amount used = 28 - 20 = 8 test tubes.

Final Answer: 8
</Example Output>
</Example Transformation>

<Task>
<Input>
Target Question:
{target_query}

Sample to Transform:
{text_to_transform}
</Input>

<Output>
- Do not include any explanations, comments, or text outside this format.

Output Format (Strictly follow this format):
Question: [New Merged Question]
Rationale and Answer: [Merged Rationale and Answer]
</Output>
</Task>
""",
    "transformation_complete":"""<Objective>
Your task is to transform the given Sample (which includes a question and its step-by-step rationale) into a new version that becomes more analogous and relevant to the Target Question.
The transformation should be holistic, aiming to align the sample with the Target Question on multiple levels—from surface features like entities and context to deeper conceptual and structural similarities. The goal is to reframe the sample to make its reasoning pattern as clear and applicable as possible for solving the Target Question, while strictly preserving the sample's original reasoning process and final answer.
</Objective>

<Transformation Guidelines>
Target-Directed Adaptation:
- Transform the sample in a way that brings it conceptually, thematically, and structurally closer to the Target Question.
- Think of this as reframing the sample so it feels like it belongs to the same problem family or domain as the Target Question.
- Any changes made to the question (e.g., numbers, objects, context) must be consistently and accurately reflected throughout the transformed rationale.

Depth-Aware Adaptation:
- Transformations can occur at any depth. Shallow changes (e.g., swapping entities, adjusting context) are often safer and should be prioritized when they effectively create a strong analogy.
- Moderately deep changes (e.g., altering the scenario to match the target's domain) are also encouraged.
- Deeper structural or conceptual transformations are permissible but should only be performed if they are meaningful, non-disruptive, and significantly improve the analogical link without corrupting the core logic.
- Gently favor shallow-to-mid-level adaptations, as they are less likely to introduce errors, but do not avoid deeper changes if the opportunity for a safe and powerful transformation exists.

Preserve Core Reasoning:
- The fundamental logical steps, mathematical operations, and the overall reasoning strategy of the sample must remain completely intact.
- The method of solving the problem is the core pattern to be preserved; do not alter how the solution is derived.

Maintain Naturalness, Clarity, and Safety:
- The transformed question and rationale must be natural, realistic, and logically coherent.
- Avoid any nonsensical or forced transformations. If a meaningful adaptation is not possible at a certain depth, it is better to keep that part of the sample closer to the original.
- Always prioritize logical consistency and clarity over aggressive or risky transformations.

Keep the Final Answer Unchanged:
- The final numerical or categorical answer at the end of the rationale must not be changed. It must remain exactly as it was in the original sample.

</Transformation Guidelines>
</Example Transformation>

<Example Input>
Target Question: A spaceship has 120 units of fuel. Its main engine consumes 15 units per hour, and its auxiliary systems consume 5 units per hour. How many hours can the spaceship operate before running out of fuel?

Sample to Transform:

Question: A charity has a fund of $5,000. It spends $400 per month on rent and $100 per month on utilities. How many months can the charity operate before the fund is depleted?

Rationale:
To find out how long the fund will last, we first need to calculate the total monthly expenses.
The total monthly expense is the sum of rent and utilities.
Total expenses per month = $400 (rent) + $100 (utilities) = $500.
The total fund is $5,000.
Number of months the fund will last = Total fund / Total monthly expenses = $5,000 / $500 = 10 months.

Final Answer: 10
</Example Input>

</Example Output>
Question: A research station has a 5000-liter water tank. The main water purifier processes 400 liters per day, and a secondary filtration system uses an additional 100 liters per day. For how many days will the water supply last?

Rationale:
To find out how long the water supply will last, we first need to calculate the total daily water consumption.
The total daily consumption is the sum of water for the purifier and the filtration system.
Total consumption per day = 400 liters (purifier) + 100 liters (filtration) = 500 liters.
The total water supply is 5000 liters.
Number of days the supply will last = Total supply / Total daily consumption = 5000 / 500 = 10 days.

Final Answer: 10
</Example Output>
</Example Transformation>

<Task>
<Input>
Target Question:
{target_query}

Sample to Transform:
{text_to_transform}
</Input>

<Output>
- Do not include any explanations, comments, or text outside this format.

Output Format (Strictly follow this format):
Question: [New Merged Question]
Rationale and Answer: [Merged Rationale and Answer]
</Output>
</Task>
""" ,
    
    "transformation_shallow-&-moderately-deep" : """<Objective>   
Your task is to transform the given Sample (which includes a question and its step-by-step rationale) into a new version that becomes more analogous and relevant to the Target Question.
The transformation should be directed toward the Target Question, meaning every change you make should help the transformed sample better reflect, match, or resonate with the Target Question’s area, structure, or style — while still preserving the sample’s original reasoning path and final answer.
</Objective>

<Transformation Guidelines>  
Target-Directed Adaptation:  
- Transform the sample in a way that brings it conceptually and structurally closer to the Target Question.  
- Think of this as reframing the sample so it feels like it belongs to the same world or problem type as the Target Question.  
- Crucially, any changes made to the question (e.g., numbers, objects, context) must be consistently and accurately reflected throughout the transformed rationale.  

Preserve Core Reasoning:
- Keep the logical relations, steps, and reasoning strategy intact. 
- Do not change the essential operations or the fundamental reasoning pattern that lead to the solution. The method of solving should be identical. 

Avoid Deep or Complex Alterations:
- Do not transform very deep or complex internal structures (like full reasoning chains or logic patterns). 
- Instead, you may adjust surface-level or moderately deep aspects — such as the domain, quantities, entities, or context in both the question and the rationale. 

Maintain Naturalness, Clarity, and Safety:
- The transformed question and rationale must remain natural, realistic, and logically coherent. 
- Avoid any unnatural, illogical, or meaningless transformations (e.g., “a cat eats an apple”). 
- If a transformation cannot be made safely or meaningfully, keep the sample as close to the original as possible rather than forcing changes. 
- Always prioritize clarity, realism, and logical consistency over aggressive transformation. 

Keep the Final Answer Unchanged:
- The final numerical or categorical answer at the end of the rationale must not be changed. It should remain exactly as it was in the original sample.

</Transformation Guidelines>
</Example Transformation>

<Example Input>
Target Question: A laboratory has 35 beakers. A new experiment requires 5 beakers per station. If the lab manager sets up 4 stations, how many beakers are left over?

Sample to Transform:

Question: A baker has 50 cookies. He decides to package them into boxes, with each box holding 6 cookies. If he sells 7 boxes, how many cookies does he have left?

Rationale:
To find the remaining cookies, we first need to calculate how many cookies were sold.
The baker sold 7 boxes, and each box contains 6 cookies.
Total cookies sold = 7 boxes * 6 cookies/box = 42 cookies.
The baker started with 50 cookies.
Remaining cookies = Initial amount - Amount sold = 50 - 42 = 8 cookies.

Final Answer: 8
</Example Input>

</Example Output>
Question: A scientist starts with 28 test tubes for an analysis. She arranges them into racks, with each rack holding 4 test tubes. If she uses 5 full racks for her experiment, how many test tubes are left unused?

Rationale:
To find the remaining test tubes, we first need to calculate how many test tubes were used.
The scientist used 5 racks, and each rack contains 4 test tubes.
Total test tubes used = 5 racks * 4 test tubes/rack = 20 test tubes.
The scientist started with 28 test tubes.
Remaining test tubes = Initial amount - Amount used = 28 - 20 = 8 test tubes.

Final Answer: 8
</Example Output>
</Example Transformation>

<Task>  
<Input>  
Target Question:  
{target_query}  

Sample to Transform:
{text_to_transform}
</Input>

<Output>  
- Do not include any explanations, comments, or text outside this format.  

Output Format (Strictly follow this format):
Question: [New Merged Question]
Rationale and Answer: [Merged Rationale and Answer]
</Output>
</Task>
""" ,

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

    "merging_v2" : """You are an expert in analogical reasoning for mathematical problem-solving.  
Your task is to merge two solved math problems into a single new, synthesized problem-solution pair that will serve as a strong exemplar for solving a target question.  

These two input samples are called Parent Sample A and Parent Sample B, because they will be merged to form one new "child" example.   
The child example should combine the most relevant and valuable reasoning patterns from its parents in a coherent, context-aware manner.


<Your Objective> 
Create a new merged example that:
1. Retains the core reasoning structures and mathematical logic from both Parent A and Parent B.
2. Selectively integrates only the parts most relevant to the Target Question. 
3. Produces a new, logically consistent and useful exemplar that the LLM can learn from for analogical reasoning.
</Your Objective> 


<Core Guidelines> 
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

    # NEW: Template for the Self-Sampling generation step
    "self_sampling_generator_v1": """**Objective:**
Your task is to solve the **Main Question** by generating a clear, step-by-step **Rationale** and the **Final Answer**.

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

    "self_sampling_generator_v2":"""Objective:
Your task is to solve the Main Question by providing a formal, step-by-step solution and a final answer. The solution should be presented in an academic, textbook-style format.

Style Guidelines:

Avoid conversational language: Do not use phrases like "Let's start by...", "Now, we will...", or any chatbot-like pleasantries.

Be direct and concise: Focus on showing the mathematical steps, formulas, and calculations directly.

Formal Tone: The entire output should be objective and formal, as if written in a mathematics textbook.

Required Output Format (Strictly Adhere):
Solution:
[Your step-by-step solution, presenting the mathematical derivation directly.]

Final Answer:
[Your final answer to the Main Question.]

Inputs:
Main Question:
{main_question_text}

Your Solution:
"""

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

def create_normalization_prompt(original_example: str) -> str:
    """Creates a prompt for the 'normalization' pipeline step."""
    template = PROMPT_TEMPLATES["standardization_v1"]
    return template.format(original_example=original_example)

# Alias for backward compatibility
create_standardization_prompt = create_normalization_prompt

def create_transformation_prompt(target_query: str, text_to_transform: str, config: Dict[str, Any], template_key_name: str) -> str:
    """
    Creates a prompt for a 'transformation' pipeline step using a dynamically specified template key.
    
    Args:
        target_query (str): The main question the transformation is being guided by.
        text_to_transform (str): The exemplar text to be transformed.
        config (Dict[str, Any]): The main configuration dictionary.
        template_key_name (str): The key in the config that holds the name of the prompt template to use
                                 (e.g., "PROMPT_TEMPLATE_TRANSFORMATION_1").
    """
    # Dynamically get the template name from the config using the provided key.
    template_name = config.get(template_key_name, "transformation_v1") # Fallback to a default.
    
    if template_name not in PROMPT_TEMPLATES:
        return f"Error: Prompt template '{template_name}' specified by key '{template_key_name}' not found in registry."
        
    template = PROMPT_TEMPLATES[template_name]
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

# NEW: Function to create the prompt for generating synthetic samples
def create_self_sampling_generation_prompt(main_question_text: str, config: Dict[str, Any]) -> str:
    """Creates the prompt for generating a single synthetic sample."""
    template_name = config.get("PROMPT_TEMPLATE_SELF_SAMPLING_GENERATOR", "self_sampling_generator_v1")
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
