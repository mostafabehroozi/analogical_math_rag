from typing import List, Dict, Any, Optional

from src.benchmark_data import benchmark_name_for_target_index, uses_exact_final_answers

EXEMPLAR_FORMAT = "Question: {question}\nRationale and Answer: {solution}"

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
The transformation should be directed toward the Target Question, meaning every change you make should help the transformed sample better reflect, match, or resonate with the Target Question's theme, context, or style — while strictly preserving the sample's original reasoning path and final answer.
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
- Avoid any unnatural, illogical, or meaningless transformations (e.g., "a cat eats an apple").
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
The transformation should be directed toward the Target Question, meaning every change you make should help the transformed sample better reflect, match, or resonate with the Target Question's area, structure, or style — while still preserving the sample's original reasoning path and final answer.
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
- Avoid any unnatural, illogical, or meaningless transformations (e.g., "a cat eats an apple"). 
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

    "final_solver_v2": """You are an expert in analogical reasoning, highly skilled at identifying and extracting patterns, reasoning pathways, problem-solving strategies, and conceptual frameworks from similar solved examples. Your primary task is to solve the main question by drawing meaningful analogies from the provided solved examples.

<Instructions>
Carefully analyze each example: pinpoint common reasoning steps, patterns (including structural similarities, logical sequences, mathematical transformations, conceptual mappings, or recurring problem-solving techniques), and effective strategies that led to the final answers.
Focus on extracting only the most useful and relevant elements from these examples as supportive guides—treat them as verified, correct rationales to inform your approach, but not as strict templates that must be replicated exactly. Instead, adapt them flexibly to fit the unique aspects of the main question, even when surface details differ, while prioritizing your own independent reasoning to develop a robust solution.
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

    "final_solver_v3": """You are an expert mathematician. Use the provided True Solved Example to solve the new problem.

<True Solved Example>
{examples_block}
</True Solved Example>

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
    "final_solver_v4": """You are an expert in analogical reasoning, highly skilled at identifying and extracting patterns, reasoning pathways, problem-solving strategies, and conceptual frameworks from similar solved examples. Your primary task is to solve the main question by drawing meaningful analogies from the provided solved examples.

<Instructions>
Analyze the solved examples to identify key reasoning patterns, structures, and strategies that lead to their answers. Use these insights as guidance—rather than strict templates—and adapt them to the main question while applying your own reasoning to reach a solution.
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
    "final_solver_simple_v2": """Solve the Main Question using standard mathematical methods.

<Instructions>
1. Analyze the question to identify the necessary logical steps.
2. Provide the solution as a concise, technical derivation.
3. Focus strictly on the mathematical steps and logic, avoiding conversational language, teaching explanations, or filler words.
</Instructions>

<Main Question to Solve>
{main_question_text}
</Main Question to Solve>


<Your Answer/Output Format>
Rationale:
[Formal step-by-step derivation]

Final Answer:
[Your final answer]
</Your Answer/Output Format>
""",

    "final_solver_simple_v3": """Solve the Main Question using standard mathematical methods. Focus strictly on the mathematical steps and logic, avoiding conversational language, teaching explanations, or filler words.

<Main Question to Solve>
{main_question_text}
</Main Question to Solve>


<Your Answer/Output Format>
Rationale:
[Formal step-by-step derivation]

Final Answer:
[Your final answer]
</Your Answer/Output Format>
""",


"evaluator_v1": """Your task is to evaluate if the final answer in 'Model Output' is equivalent to the final answer in 'Ground Truth'.
Both 'Model Output' and 'Ground Truth' may contain intermediate steps (Chain-of-Thought) leading to a final answer.

Follow these two steps precisely:

Step 1: Extract Final Answers
Extract only the final numerical or definitive answer expression from both 'Model Output' and 'Ground Truth', writing "Extraction Failed" if you cannot confidently extract an answer.

Step 2: Evaluate Equivalence
- Compare the 'Extracted Model Answer' with the 'Extracted Ground Truth Answer'.
- If either extraction failed, the evaluation must be 'false'.
- Consider common mathematical equivalences (e.g., "2+2" vs "4", "sqrt(9)" vs "3", "1/2" vs "0.5", "1,000" vs "1000", "$5" vs "5").
- Respond ONLY with the single word 'true' or 'false' for the evaluation.

Output Format (Strictly follow this format):
Extracted Model Answer: [Your extracted answer from Model Output]
Extracted Ground Truth Answer: [Your extracted answer from Ground Truth]
Evaluation: [true OR false]

Model Output:
{model_answer}

Ground Truth:
{ground_truth}

""",
"evaluator_v2": """Your task is to evaluate if the final answer in 'Model Output' is equivalent to the 'Ground Truth'.
The 'Model Output' may contain intermediate steps (Chain-of-Thought), but the 'Ground Truth' provides the exact final answer.

Follow these two steps precisely:

Step 1: Extract Final Answer
Extract only the final numerical or definitive answer expression from the 'Model Output', writing "Extraction Failed" if you cannot confidently extract an answer.

Step 2: Evaluate Equivalence
- Compare your 'Extracted Model Final Answer' directly with the 'Ground Truth'.
- If extraction failed, the evaluation must be 'false'.
- Consider common mathematical equivalences (e.g., "1/2" vs "0.5", "1,000" vs "1000", "$5" vs "5").
- Respond ONLY with the single word 'true' or 'false' for the evaluation.

Output Format (Strictly follow this format):
Extracted Model Final Answer: [Your extracted Final answer from Model Output]
Evaluation: [true OR false]

Model Output:
{model_answer}

Ground Truth:
{ground_truth}

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

    "reverse_validation_candidate_generator_v1": """Objective:
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
""",





    



    "analogical_adaptation_v1": """You are an expert in analogical reasoning for mathematical problem-solving.

Your task is to solve the Main Question by using analogical reasoning based on the provided group of Solved Sample Problems.

<Instructions>
1. Carefully analyze each solved sample problem to understand its reasoning pattern, problem-solving strategy, and logical structure.
2. Identify the core reasoning principles and mathematical techniques that can be transferred to the Main Question.
3. Apply these analogical insights to solve the Main Question, adapting the reasoning patterns to fit the specific context and requirements of the Main Question.
4. Present your solution in a clear, step-by-step format that shows how you applied analogical reasoning.
5. Do NOT simply copy the solutions from the samples—adapt and apply their reasoning patterns intelligently.
</Instructions>

<Main Question>
{main_question_text}
</Main Question>

<Solved Sample Problems>
{samples_block}
</Solved Sample Problems>

<Output Format (Strictly follow this format)>
Question: [The Main Question]
Rationale and Answer: [Your step-by-step solution using analogical reasoning from the samples, followed by the final answer]
</Output Format>
""",


    "analogical_adaptation_v2": """You are an expert in analogical reasoning for mathematical problem-solving.

<Instructions>
1. Carefully analyze each solved sample problem to understand its reasoning pattern, problem-solving strategy, and logical structure.
2. Identify the core reasoning principles and mathematical techniques that can be transferred to the Main Question.
3. Apply these analogical insights to solve the Main Question, adapting the reasoning patterns to fit the specific context and requirements of the Main Question.
4. Present your solution in a clear, step-by-step format that shows how you applied analogical reasoning.
5. Do NOT simply copy the solutions from the samples—adapt and apply their reasoning patterns intelligently.
</Instructions>

<Solved Examples>
{samples_block}
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
    
    "analogical_refinement":"""You are an expert mathematical problem solver.

<Instructions>
1. Review the <Solved Examples> to see how similar problems might be approached.
2. **Use judgment:** Only use the logic from the examples if it actually works for the Main Question.
3. **Refine and Correct:** If the examples use a pattern that doesn't fit or is incorrect for the Main Question, discard it and use the correct mathematical method instead.
4. Solve the Main Question. Your priority is getting the correct answer, not mimicking the examples.
</Instructions>

<Solved Examples>
{samples_block}
</Solved Examples>

<Main Question to Solve>
{main_question_text}
</Main Question to Solve>

<Your Answer/Output Format>
Rationale:
[Explain your steps. Briefly state which parts of the examples helped and which parts you had to change or ignore to solve this specific question correctly.]

Final Answer:
[Your final answer to the Main Question]
</Your Answer/Output Format>
""",
    



    "reverse_validation_v1": """You are an expert in analogical reasoning, highly skilled at identifying and extracting patterns, reasoning pathways, problem-solving strategies, and conceptual frameworks from similar solved examples. Your primary task is to solve the main question by drawing meaningful analogies from the provided solved variations.

<Instructions>
Carefully analyze each variation: pinpoint common reasoning steps, patterns (including structural similarities, logical sequences, mathematical transformations, conceptual mappings, or recurring problem-solving techniques), and effective strategies that led to the final answers. Focus on extracting only the most useful and relevant elements from these variations as supportive guides. Adapt them flexibly to fit the unique aspects of the main question, even when surface details differ, while prioritizing your own independent reasoning to develop a robust solution.
</Instructions>

<Solved Variations>
{candidate_exemplar}
</Solved Variations>

<Main Question to Solve>
{validator_question}
</Main Question to Solve>

<Your Answer/Output Format>
Rationale:
[Your step-by-step rationale for the Main Question]

Final Answer:
[Your final answer to the Main Question]
</Your  Answer/Output Format>
""",

    "simplification_generator_v1": """Your task is to create a simplified version of the Main Question that retains the core logical problem but simplifies the shallow or non-core elements.
<Instructions>
1. Replace large numbers or complex values with small, single-digit integers to make the math easier.
2. Rewrite complex scenarios or wordy descriptions into a plain, direct statement to remove distractions.
3. Reduce the number of steps or constraints slightly, as long as the underlying relationship remains the same.
4. CRITICAL: If the question is already simple or cannot be simplified without breaking the core logic, or if the simplification results in a drastic change rather than a shallow adjustment, do NOT change it; just repeat the Main Question exactly.
</Instructions>
<Main Question>
{text_to_simplify}
</Main Question>
<Output Format>
Simplified Question:
[Your simplified version]
</Output Format>
""",

    "simplified_sample_solver_v1": """You are an expert mathematician. Your task is to solve a 'Simplified Question' by applying the logic found in an 'Original Solved Example'.

<Original Solved Example>
{original_exemplar}
</Original Solved Example>

<Simplified Question>
{simplified_question}
</Simplified Question>

<Instructions>
1. Analyze the Original Solved Example to understand the underlying logic and problem-solving method.
2. Apply that SAME logic to solve the Simplified Question.
3. Output the solution in the standard format.
</Instructions>

<Output Format (Strictly follow this format)>
Rationale:
[Your step-by-step rationale for the Simplified Question]

Final Answer:
[Your final answer]
</Output Format>
""",

    "main_from_simplified_proxy_v1": """You are an expert mathematician. You have a 'Complex Main Question' and a solution to a 'Simplified Version' of that question.
Use the logic from the Simplified Solution to solve the Complex Main Question.

<Simplified Version Solution>
{simplified_solution}
</Simplified Version Solution>

<Complex Main Question>
{original_main_question}
</Complex Main Question>

<Instructions>
1. Read the Simplified Version Solution to understand the method used.
2. Apply that same method to the Complex Main Question (handling the extra complexity or larger numbers).
3. Output the solution in the standard format.
</Instructions>

<Output Format (Strictly follow this format)>
Rationale:
[Your step-by-step rationale for the Complex Main Question]

Final Answer:
[Your final answer]
</Output Format>
""",




    "mirror_baseline_zero_shot_v1": """Objective:
Your task is to solve the Main Question by generating a clear, step-by-step Rationale and the Final Answer.

Your Method & Constraints:
1.  Construct Your Solution: Develop a logical, step-by-step Rationale for the Main Question.
2.  Perform calculations accurately and show your work.
3.  Clearly state the Final Answer at the end.

Required Output Format (Strictly Adhere):
Rationale:
[Your step-by-step rationale for the Main Question]

Final Answer:
[Your final answer to the Main Question]


Inputs:
Main Question:
{question}

Your Solution:
""",


    "mirror_hypothesis_gen_v1": """You are an expert in analogical reasoning, highly skilled at identifying and extracting patterns, reasoning pathways, problem-solving strategies, and conceptual frameworks from similar solved examples. Your primary task is to solve the main question by drawing meaningful analogies from the provided solved examples.

<Instructions>
Analyze the solved examples to identify key reasoning patterns, structures, and strategies that lead to their answers. Use these insights as guidance—rather than strict templates—and adapt them to the main question while applying your own reasoning to reach a solution.
</Instructions>

<Solved Examples>
question:
{exemplar_question}

Solution:
{exemplar_solution}
</Solved Examples>

<Main Question to Solve>
{target_query}
</Main Question to Solve>

<Your Answer/Output Format>
Rationale:
[Your step-by-step rationale for the Main Question]

Final Answer:
[Your final answer to the Main Question]
</Your  Answer/Output Format>
""",

    "mirror_hypothesis_gen_zero_shot_v1": """You are an expert mathematician. Solve the following problem step-by-step.

<Question to Solve>
{target_query}
</Question to Solve>

<Your Answer/Output Format>
Rationale:
[Formal step-by-step derivation]

Final Answer:
[Your final answer] 
</Your Answer/Output Format>
""",

    "mirror_verification_v1": """You are an expert in analogical reasoning, highly skilled at identifying and extracting patterns, reasoning pathways, problem-solving strategies, and conceptual frameworks from similar solved examples. Your primary task is to solve the main question by drawing meaningful analogies from the provided solved examples.

<Instructions>
Analyze the solved examples to identify key reasoning patterns, structures, and strategies that lead to their answers. Use these insights as guidance—rather than strict templates—and adapt them to the main question while applying your own reasoning to reach a solution.
</Instructions>

<Solved Examples>
question:
{hypothesis_question}

Solution:
{hypothesis_solution}
</Solved Examples>

<Main Question to Solve>
{validation_question}
</Main Question to Solve>

<Your Answer/Output Format>
Rationale:
[Your step-by-step rationale for the Main Question]

Final Answer:
[Your final answer to the Main Question]
</Your  Answer/Output Format>
""",

    "reverse_transformation_main_to_exemplar": """<Objective>
Your task is to transform the given Main Question into a version that becomes more analogous and relevant to the Retrieved Exemplar.
The transformation should be directed toward the Retrieved Exemplar, meaning every change you make should help the transformed main question better reflect, match, or resonate with the Retrieved Exemplar's theme, context, or style — while strictly preserving the main question's underlying logical structure and ensuring the transformed question remains solvable.
</Objective>

<Transformation Guidelines>
Target-Directed Adaptation:
- Adapt the main question's theme, context, and entities to mirror those in the Retrieved Exemplar.
- Think of this as reframing the main question so it feels like it belongs to the same world or problem type as the Retrieved Exemplar.
- Crucially, any changes made to the question (e.g., numbers, objects, context) must be consistently and accurately reflected if they affect the logical relationships.

Preserve Core Logic:
- The underlying logical pathway and problem-solving structure must remain identical or analogous.
- Do not alter the sequence of logical steps or the fundamental problem type.
- The core reasoning required to solve must be preserved.

Maintain Naturalness, Clarity, and Safety:
- The transformed main question must remain natural, realistic, and logically coherent.
- Avoid any unnatural, illogical, or meaningless transformations.
- If a transformation cannot be made safely or meaningfully, keep the main question as close to the original as possible.
- Always prioritize clarity, realism, and logical consistency.

</Transformation Guidelines>

<Input>
Original Main Question:
{main_question}

Retrieved Exemplar (Question and Solution):
{exemplar_text}
</Input>

<Output Format (Strictly follow this format)>
Transformed Main Question:
[Your rewritten main question, closely aligned with the Retrieved Exemplar's theme and context]
</Output Format>
""",


    "reverse_transformation_solve_transformed": """You are an expert in analogical reasoning, highly skilled at identifying and extracting patterns, reasoning pathways, problem-solving strategies, and conceptual frameworks from similar solved examples. Your primary task is to solve the main question by drawing meaningful analogies from the provided solved examples.

<Instructions>
Carefully analyze each example: pinpoint common reasoning steps, patterns (including structural similarities, logical sequences, mathematical transformations, conceptual mappings, or recurring problem-solving techniques), and effective strategies that led to the final answers. Focus on extracting only the most useful and relevant elements from these examples as supportive guides—treat them as verified, correct rationales to inform your approach, but not as strict templates that must be replicated exactly. Instead, adapt them flexibly to fit the unique aspects of the main question, even when surface details differ, while prioritizing your own independent reasoning to develop a robust solution.
</Instructions>

<Solved Examples>
{exemplar_text}
</Solved Examples>

<Main Question to Solve>
{transformed_question}
</Main Question to Solve>

<Your Answer/Output Format>
Rationale:
[Your step-by-step rationale for the Main Question]

Final Answer:
[Your final answer to the Main Question]
</Your  Answer/Output Format>
""",



    "reverse_transformation_final_solve": """You are an expert in analogical reasoning, highly skilled at identifying and extracting patterns, reasoning pathways, problem-solving strategies, and conceptual frameworks from similar solved examples. Your primary task is to solve the main question by drawing meaningful analogies from the provided solved examples.

<Instructions>
Carefully analyze each example: pinpoint common reasoning steps, patterns (including structural similarities, logical sequences, mathematical transformations, conceptual mappings, or recurring problem-solving techniques), and effective strategies that led to the final answers. Focus on extracting only the most useful and relevant elements from these examples as supportive guides—treat them as verified, correct rationales to inform your approach, but not as strict templates that must be replicated exactly. Instead, adapt them flexibly to fit the unique aspects of the main question, even when surface details differ, while prioritizing your own independent reasoning to develop a robust solution.
</Instructions>

<Solved Examples>
{transformed_solutions}
</Solved Examples>

<Main Question to Solve>
{original_question}
</Main Question to Solve>

<Your Answer/Output Format>
Rationale:
[Your step-by-step rationale for the Main Question]

Final Answer:
[Your final answer to the Main Question]
</Your  Answer/Output Format>
""",

    "reverse_transformation_shallow-&-moderately-deep" : """<Objective>   
Your task is to transform the given Main Question into a new version that becomes more analogous and relevant to the Retrieved Exemplar.
The transformation should be directed toward the Retrieved Exemplar, meaning every change you make should help the transformed main question better reflect, match, or resonate with the Retrieved Exemplar's area, structure, or style — while still preserving the main question's original logical structure, solvability, and required reasoning path.
</Objective>

<Transformation Guidelines>  
Target-Directed Adaptation:  
- Transform the Main Question in a way that brings it conceptually and structurally closer to the Retrieved Exemplar.  
- Think of this as reframing the Main Question so it feels like it belongs to the same world or problem type as the Retrieved Exemplar.  
- Crucially, any changes made to the question (e.g., numbers, objects, context) must be consistently and accurately applied so that the mathematical and logical relationships remain perfectly intact.  

Preserve Core Logic:
- Keep the logical relations and the fundamental problem type intact. 
- Do not change the essential operations or the fundamental reasoning pattern that the Main Question demands. The steps required to solve it should remain identical. 

Avoid Deep or Complex Alterations:
- Do not transform very deep or complex internal structures (like core logic patterns or the type of problem). 
- Instead, you may adjust surface-level or moderately deep aspects — such as the domain, entities, or contextual narrative in the Main Question to match the Exemplar. 

Maintain Naturalness, Clarity, and Safety:
- The transformed Main Question must remain natural, realistic, and logically coherent. 
- Avoid any unnatural, illogical, or meaningless transformations (e.g., "a cat eats an apple"). 
- If a transformation cannot be made safely or meaningfully, keep the Main Question as close to the original as possible rather than forcing changes. 
- Always prioritize clarity, realism, and logical consistency over aggressive transformation. 

</Transformation Guidelines>

<Example Transformation>
<Example Input>
Retrieved Exemplar (Target Style):
Question: A baker has 50 cookies. He decides to package them into boxes, with each box holding 6 cookies. If he sells 7 boxes, how many cookies does he have left?
Rationale:
To find the remaining cookies, we first need to calculate how many cookies were sold.
The baker sold 7 boxes, and each box contains 6 cookies.
Total cookies sold = 7 boxes * 6 cookies/box = 42 cookies.
The baker started with 50 cookies.
Remaining cookies = Initial amount - Amount sold = 50 - 42 = 8 cookies.
Final Answer: 8

Original Main Question (To Transform): 
A laboratory has 35 beakers. A new experiment requires 5 beakers per station. If the lab manager sets up 4 stations, how many beakers are left over?
</Example Input>

<Example Output>
Transformed Main Question:
A pastry chef baked 35 cupcakes. She decides to arrange them onto display trays, with each tray holding 5 cupcakes. If she sets up 4 display trays, how many cupcakes are left over?
</Example Output>
</Example Transformation>

<Task>  
<Input>  
Original Main Question:  
{main_question}  

Retrieved Exemplar (Question and Solution):
{exemplar_text}
</Input>

<Output>  
- Do not include any explanations, comments, or text outside this format.  

Output Format (Strictly follow this format):
Transformed Main Question:
[Your rewritten main question, closely aligned with the Retrieved Exemplar's theme and context]
</Output>
</Task>
""",

    "reverse_transformation_shallow" : """<Objective>
Your task is to transform the given Sample (which includes a question and its step-by-step rationale) into a new version that becomes more analogous and relevant to the Target Question.
The transformation should be directed toward the Target Question, meaning every change you make should help the transformed sample better reflect, match, or resonate with the Target Question's theme, context, or style — while strictly preserving the sample's original reasoning path and final answer.
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
- Avoid any unnatural, illogical, or meaningless transformations (e.g., "a cat eats an apple").
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
{main_question}

Sample to Transform:
{exemplar_text}
</Input>

<Output>
- Do not include any explanations, comments, or text outside this format.

Output Format (Strictly follow this format):
Question: [New Merged Question]
Rationale and Answer: [Merged Rationale and Answer]
</Output>
</Task>
""",
    
    "core_simplification_zero_shot_v1": """You are an expert mathematical AI. Your task is to perform "Core-Preserving Simplification" on a complex math competition problem.

Your goal is to create a simpler "Proxy Question" that acts as an easy analogical stepping-stone. The Proxy Question MUST share the EXACT SAME logical structure, Chain of Thought, and theorems as the original problem, but be computationally and cognitively much easier.

To do this, use the following framework:

STEP 1: Identify the "Core Logical Engine" (The Untouchable Logic)
Identify the fundamental math concept required (e.g., Stars and Bars, Vieta's formulas, modular inverses). You MUST NOT alter the problem so much that this core theorem is bypassed.

STEP 2: Apply Vectors of Simplification
Analyze the problem and safely simplify its peripheral complexities. Below is a diverse toolkit of simplification vectors. 
IMPORTANT: The examples provided below are basic "toy" examples used merely to illustrate the concepts. The actual input problem you receive will be significantly more complex, convoluted, and multi-layered. You must look past the surface complexity, deeply analyze the specific problem, and apply the *underlying principles* of these vectors. Use them as strong inspiration, but feel free to combine them, adapt them, or invent other contextual simplifications required to dismantle your specific problem, provided they preserve the Core Engine.

Study the Valid vs. Invalid examples to understand the exact boundaries of what preserves versus what destroys logic:

[Arithmetic]
- Magnitude Downscaling: Reduce giant constants to friendly numbers.
  Valid: "Find 2024^2024 mod 5" -> "Find 4^4 mod 5" (Modular exponentiation preserved).
  Invalid: "Find the sum of the digits of 2024" -> "Find the sum of the digits of 1" (Trivializes the problem).
- Fraction Flattening: Convert messy rationals to simple whole numbers.
  Valid: "Solve (17/13)x + 5/13 = 22/13" -> "Solve 2x + 1 = 5" (Algebraic isolation preserved).
  Invalid: "Find the fractional part of 17/5" -> "Find the fractional part of 3" (Answer becomes trivially 0; logic destroyed).
- Radical Taming: Replace ugly irrationals with perfect squares or small surds.
  Valid: "Solve sqrt(199 - x) = 13" -> "Solve sqrt(25 - x) = 4" (Squaring/isolation logic preserved).
  Invalid: "Rationalize 1 / (sqrt(3) + sqrt(2))" -> "Rationalize 1 / (3 + 2)" (Conjugate multiplication step destroyed).
- Prime Substitution: Swap giant primes for small primes.
  Valid: "Highest power of 997 dividing 10000!" -> "Highest power of 5 dividing 20!" (Legendre's formula preserved).
  Invalid: "Prove prime p=997 cannot be a^2 - b^2" -> "Prove p=8 cannot be a^2 - b^2" (8 is not prime; breaks the premise).

[Algebra]
- Degree Lowering: Shrink massive exponents to the lowest non-trivial degree.
  Valid: "Find roots of x^100 - 1 = 0" -> "Find roots of x^4 - 1 = 0" (Roots of unity concept preserved).
  Invalid: "Coefficient of x^50 in (1+x)^100" -> "Coefficient of x^1 in (1+x)^2" (Trivializes binomial expansion).
- Coefficient Smoothing: Replace un-factorable coefficients with easy ones.
  Valid: "Minimum of 143x^2 - 286x + 5" -> "Minimum of 2x^2 - 4x + 1" (Vertex formula x=-b/2a preserved).
  Invalid: "Factor 143x^2 + 15x + 2" -> "Factor x^2 + 3x + 2" (Removes the need for the AC method/grouping).
- Variable Reduction: Reduce symmetric multi-variable systems.
  Valid: "System x+y+z=6, xy+yz+zx=11, xyz=6" -> "System x+y=3, xy=2" (Vieta's logic preserved on a lower degree).
  Invalid: "Volume of 3D ellipsoid x^2/a^2 + y^2/b^2 + z^2/c^2 = 1" -> "Area of 1D line x^2/a^2 = 1" (Destroys 3D calculus).
- Sequence Truncation: Shorten long sums/products to a few terms.
  Valid: "Telescoping sum of 1/(n(n+1)) to 1000 terms" -> "...to 4 terms" (Telescoping cancellation still happens).
  Invalid: "Find the limit as n -> infinity" -> "Find the 3rd term" (Calculus limit concept completely removed).

[Combinatorics]
- Population Scaling: Shrink massive sets to small, countable sets.
  Valid: "Seat 100 people at a circular table" -> "Seat 5 people..." ((n-1)! logic preserved).
  Invalid: "Seat 2 people at a circular table" (Outputs (2-1)!=1, bypassing combinatorial thought).
- State-Space Shrinking: Reduce dimensions of grids or graphs.
  Valid: "Paths on 10x10 grid avoiding (5,5)" -> "Paths on 4x4 grid avoiding (2,2)" (Inclusion-exclusion preserved).
  Invalid: "Paths on a 1x1 grid" (No room for an obstacle; destroys the logic).
- Event Iteration Reduction: Lower the number of repeated probability events.
  Valid: "Expected heads in 100 coin flips" -> "Expected heads in 4 flips" (Linearity of expectation preserved).
  Invalid: "Expected heads in 1 coin flip" (Bypasses the summation/expected value of multiple events).
- Die/Spinner Downgrading: Reduce the complexity of random generators.
  Valid: "Roll five 20-sided dice, sum to 50" -> "Roll three 4-sided dice, sum to 8" (Generating functions preserved).
  Invalid: "Roll one 4-sided die, chance it equals 2" (Removes convolution/summation logic entirely).

[Geometry]
- Coordinate Normalization: Shift huge/negative coordinates near the origin.
  Valid: "Distance from (1005, -2048) to (1008, -2044)" -> "Distance from (0,0) to (3,4)" (Distance formula preserved).
  Invalid: "Area of polygon with vertices (10,10), (10,-10)..." -> "Area of polygon with vertex (0,0)" (A single point isn't a polygon).
- Angle Standardizing: Replace obscure angles with standard unit-circle angles.
  Valid: "sin(13)cos(17) + cos(13)sin(17)" -> "sin(15)cos(15) + cos(15)sin(15)" (Angle addition formula preserved).
  Invalid: "Period of sin(17x)" -> "Period of sin(x)" (Removes the 2pi/b calculation entirely).
- Polygon Side Reduction: Shrink N-gons to simpler shapes.
  Valid: "Sum of interior angles of 100-gon" -> "...of a hexagon" ((n-2)*180 formula preserved).
  Invalid: "Number of diagonals in 100-gon" -> "...in a triangle" (Triangles have 0 diagonals; trivializes the formula).
- Shape Regularization: Simplify arbitrary shapes if the theorem is universal.
  Valid: "Arbitrary convex quadrilateral with orthogonal diagonals, find area" -> "Kite with orthogonal diagonals..." (Area=1/2*d1*d2 holds).
  Invalid: "Prove Euler line passes through orthocenter of scalene triangle" -> "...of equilateral triangle" (Centers share the same point; line vanishes).

[Number Theory]
- Modulo Minimization: Lower giant modulo bases.
  Valid: "Last 3 digits of 7^999 (i.e., mod 1000)" -> "Last digit of 7^99 (i.e., mod 10)" (Euler's totient preserved).
  Invalid: "Find x mod 1000" -> "Find x mod 1" (Mod 1 is always 0; modular arithmetic destroyed).
- Base/Radix Lowering: Convert arithmetic in large bases to smaller bases.
  Valid: "Convert A4F base 16 to base 10" -> "Convert 210 base 3 to base 10" (Base expansion logic preserved).
  Invalid: "Trailing zeros of 100! in base 16" -> "...in base 10" (Base 16 requires checking powers of 2, a different logical step than base 10).
- Diophantine Shrinking: Simplify constants in linear Diophantine equations.
  Valid: "Integer solutions to 1001x + 2003y = 5" -> "Integer solutions to 7x + 11y = 5" (Extended Euclidean Algorithm preserved).
  Invalid: "Integer solutions to 1001x + 2003y = 5" -> "1x + 1y = 5" (Trivializes the algorithm).

[Formatting]
- Goal Un-nesting: Remove arbitrary final formatting arithmetic.
  Valid: "Find radius r, then calculate floor(100*pi*r^2 - 17)" -> "Find the area pi*r^2" (Keeps core geometry, removes formatting noise).
  Invalid: "Find roots r1, r2, then calculate (r1^2 + r2^2)" -> "Find roots r1, r2" (Calculating sum of squares requires a specific algebraic manipulation that is lost).


STEP 3: The Mathematical Safety Check
Ensure your selected simplifications do not break math rules:
- Parity/Divisibility: If a number MUST be even for the problem to work, your smaller number must be even.
- Geometry Constraints: Do not violate the Triangle Inequality.
- Probabilities: Numerators must not exceed denominators.
- Keywords: NEVER remove words defining constraints (e.g., "distinct", "integer", "consecutive").

STEP 4: The Failsafe
If the question relies entirely on specific numbers (e.g., factoring a specific prime), or if any simplification would destroy the Core Logical Engine, DO NOT CHANGE IT. Output the exact original question.

INPUT
Original Question:
{original_question}

OUTPUT FORMAT (Strictly follow this format):
Simplified Question: [Your simplified proxy question here, or the exact original question if no safe simplification is possible]
""",


    "core_simplification_few_shot_v1": """You are an expert mathematical AI. Your task is to perform "Core-Preserving Simplification" on a complex math competition problem.

Your goal is to create a simpler "Proxy Question" that acts as an easy analogical stepping-stone. The Proxy Question MUST share the EXACT SAME logical structure, Chain of Thought, and theorems as the original problem, but be computationally and cognitively much easier.

To do this, use the following framework:

STEP 1: Identify the "Core Logical Engine" (The Untouchable Logic)
Identify the fundamental math concept required (e.g., Stars and Bars, Vieta's formulas, modular inverses). You MUST NOT alter the problem so much that this core theorem is bypassed.

STEP 2: Apply Vectors of Simplification
Analyze the problem and safely simplify its peripheral complexities. Below is a diverse toolkit of simplification vectors. 
IMPORTANT: The examples provided below are basic "toy" examples used merely to illustrate the concepts. The actual input problem you receive will be significantly more complex, convoluted, and multi-layered. You must look past the surface complexity, deeply analyze the specific problem, and apply the *underlying principles* of these vectors. Use them as strong inspiration, but feel free to combine them, adapt them, or invent other contextual simplifications required to dismantle your specific problem, provided they preserve the Core Engine.

Study the Valid vs. Invalid examples to understand the exact boundaries of what preserves versus what destroys logic:

[Arithmetic]
- Magnitude Downscaling: Reduce giant constants to friendly numbers.
  Valid: "Find 2024^2024 mod 5" -> "Find 4^4 mod 5" (Modular exponentiation preserved).
  Invalid: "Find the sum of the digits of 2024" -> "Find the sum of the digits of 1" (Trivializes the problem).
- Fraction Flattening: Convert messy rationals to simple whole numbers.
  Valid: "Solve (17/13)x + 5/13 = 22/13" -> "Solve 2x + 1 = 5" (Algebraic isolation preserved).
  Invalid: "Find the fractional part of 17/5" -> "Find the fractional part of 3" (Answer becomes trivially 0; logic destroyed).
- Radical Taming: Replace ugly irrationals with perfect squares or small surds.
  Valid: "Solve sqrt(199 - x) = 13" -> "Solve sqrt(25 - x) = 4" (Squaring/isolation logic preserved).
  Invalid: "Rationalize 1 / (sqrt(3) + sqrt(2))" -> "Rationalize 1 / (3 + 2)" (Conjugate multiplication step destroyed).
- Prime Substitution: Swap giant primes for small primes.
  Valid: "Highest power of 997 dividing 10000!" -> "Highest power of 5 dividing 20!" (Legendre's formula preserved).
  Invalid: "Prove prime p=997 cannot be a^2 - b^2" -> "Prove p=8 cannot be a^2 - b^2" (8 is not prime; breaks the premise).

[Algebra]
- Degree Lowering: Shrink massive exponents to the lowest non-trivial degree.
  Valid: "Find roots of x^100 - 1 = 0" -> "Find roots of x^4 - 1 = 0" (Roots of unity concept preserved).
  Invalid: "Coefficient of x^50 in (1+x)^100" -> "Coefficient of x^1 in (1+x)^2" (Trivializes binomial expansion).
- Coefficient Smoothing: Replace un-factorable coefficients with easy ones.
  Valid: "Minimum of 143x^2 - 286x + 5" -> "Minimum of 2x^2 - 4x + 1" (Vertex formula x=-b/2a preserved).
  Invalid: "Factor 143x^2 + 15x + 2" -> "Factor x^2 + 3x + 2" (Removes the need for the AC method/grouping).
- Variable Reduction: Reduce symmetric multi-variable systems.
  Valid: "System x+y+z=6, xy+yz+zx=11, xyz=6" -> "System x+y=3, xy=2" (Vieta's logic preserved on a lower degree).
  Invalid: "Volume of 3D ellipsoid x^2/a^2 + y^2/b^2 + z^2/c^2 = 1" -> "Area of 1D line x^2/a^2 = 1" (Destroys 3D calculus).
- Sequence Truncation: Shorten long sums/products to a few terms.
  Valid: "Telescoping sum of 1/(n(n+1)) to 1000 terms" -> "...to 4 terms" (Telescoping cancellation still happens).
  Invalid: "Find the limit as n -> infinity" -> "Find the 3rd term" (Calculus limit concept completely removed).

[Combinatorics]
- Population Scaling: Shrink massive sets to small, countable sets.
  Valid: "Seat 100 people at a circular table" -> "Seat 5 people..." ((n-1)! logic preserved).
  Invalid: "Seat 2 people at a circular table" (Outputs (2-1)!=1, bypassing combinatorial thought).
- State-Space Shrinking: Reduce dimensions of grids or graphs.
  Valid: "Paths on 10x10 grid avoiding (5,5)" -> "Paths on 4x4 grid avoiding (2,2)" (Inclusion-exclusion preserved).
  Invalid: "Paths on a 1x1 grid" (No room for an obstacle; destroys the logic).
- Event Iteration Reduction: Lower the number of repeated probability events.
  Valid: "Expected heads in 100 coin flips" -> "Expected heads in 4 flips" (Linearity of expectation preserved).
  Invalid: "Expected heads in 1 coin flip" (Bypasses the summation/expected value of multiple events).
- Die/Spinner Downgrading: Reduce the complexity of random generators.
  Valid: "Roll five 20-sided dice, sum to 50" -> "Roll three 4-sided dice, sum to 8" (Generating functions preserved).
  Invalid: "Roll one 4-sided die, chance it equals 2" (Removes convolution/summation logic entirely).

[Geometry]
- Coordinate Normalization: Shift huge/negative coordinates near the origin.
  Valid: "Distance from (1005, -2048) to (1008, -2044)" -> "Distance from (0,0) to (3,4)" (Distance formula preserved).
  Invalid: "Area of polygon with vertices (10,10), (10,-10)..." -> "Area of polygon with vertex (0,0)" (A single point isn't a polygon).
- Angle Standardizing: Replace obscure angles with standard unit-circle angles.
  Valid: "sin(13)cos(17) + cos(13)sin(17)" -> "sin(15)cos(15) + cos(15)sin(15)" (Angle addition formula preserved).
  Invalid: "Period of sin(17x)" -> "Period of sin(x)" (Removes the 2pi/b calculation entirely).
- Polygon Side Reduction: Shrink N-gons to simpler shapes.
  Valid: "Sum of interior angles of 100-gon" -> "...of a hexagon" ((n-2)*180 formula preserved).
  Invalid: "Number of diagonals in 100-gon" -> "...in a triangle" (Triangles have 0 diagonals; trivializes the formula).
- Shape Regularization: Simplify arbitrary shapes if the theorem is universal.
  Valid: "Arbitrary convex quadrilateral with orthogonal diagonals, find area" -> "Kite with orthogonal diagonals..." (Area=1/2*d1*d2 holds).
  Invalid: "Prove Euler line passes through orthocenter of scalene triangle" -> "...of equilateral triangle" (Centers share the same point; line vanishes).

[Number Theory]
- Modulo Minimization: Lower giant modulo bases.
  Valid: "Last 3 digits of 7^999 (i.e., mod 1000)" -> "Last digit of 7^99 (i.e., mod 10)" (Euler's totient preserved).
  Invalid: "Find x mod 1000" -> "Find x mod 1" (Mod 1 is always 0; modular arithmetic destroyed).
- Base/Radix Lowering: Convert arithmetic in large bases to smaller bases.
  Valid: "Convert A4F base 16 to base 10" -> "Convert 210 base 3 to base 10" (Base expansion logic preserved).
  Invalid: "Trailing zeros of 100! in base 16" -> "...in base 10" (Base 16 requires checking powers of 2, a different logical step than base 10).
- Diophantine Shrinking: Simplify constants in linear Diophantine equations.
  Valid: "Integer solutions to 1001x + 2003y = 5" -> "Integer solutions to 7x + 11y = 5" (Extended Euclidean Algorithm preserved).
  Invalid: "Integer solutions to 1001x + 2003y = 5" -> "1x + 1y = 5" (Trivializes the algorithm).

[Formatting]
- Goal Un-nesting: Remove arbitrary final formatting arithmetic.
  Valid: "Find radius r, then calculate floor(100*pi*r^2 - 17)" -> "Find the area pi*r^2" (Keeps core geometry, removes formatting noise).
  Invalid: "Find roots r1, r2, then calculate (r1^2 + r2^2)" -> "Find roots r1, r2" (Calculating sum of squares requires a specific algebraic manipulation that is lost).


STEP 3: The Mathematical Safety Check
Ensure your selected simplifications do not break math rules:
- Parity/Divisibility: If a number MUST be even for the problem to work, your smaller number must be even.
- Geometry Constraints: Do not violate the Triangle Inequality.
- Probabilities: Numerators must not exceed denominators.
- Keywords: NEVER remove words defining constraints (e.g., "distinct", "integer", "consecutive").

STEP 4: The Failsafe
If the question relies entirely on specific numbers (e.g., factoring a specific prime), or if any simplification would destroy the Core Logical Engine, DO NOT CHANGE IT. Output the exact original question.

STEP 5: Learn from the Donor Demonstration (Few-Shot Analogical Reasoning)
Below is a verified example (the "Donor Demonstration") showing how a structurally similar problem was successfully simplified. 
CRITICAL INSTRUCTION: Use this example to learn *how* to apply the principles and analogical reasoning. However, DO NOT blindly copy its exact transformations. You must independently analyze the new Original Question, use professional mathematical reasoning to recognize which specific simplifications are safe and appropriate for its unique context, and ensure you do not violate the Valid vs. Invalid boundaries defined in STEP 2. Rely on deep structural understanding, not superficial mimicry.

DEMONSTRATION:
{donor_demonstration}

INPUT
Original Question:
{original_question}

OUTPUT FORMAT (Strictly follow this format):
Simplified Question: [Your simplified proxy question here, or the exact original question if no safe simplification is possible]
""",


    "core_simp_augmented_solver_v1": """You are an expert mathematical and logical reasoning system. Solve the Main Question by directly applying the reasoning trajectory of the provided Solved Example. This example is highly similar and shares the exact same underlying structural logic as your target question.

To help you find the correct Chain of Thought and optimal first step, you have been provided with an Analogous Stepping-Stone. This stepping-stone is a simplified version of the exact same logical problem, alongside its correct solution. 
Study how the core logic (the "Trunk") was solved in the stepping-stone, and map that exact same logical strategy to the complex numbers and specific details of the Main Problem.

<Analogous Stepping-Stone>
{solved_proxy_question}
</Analogous Stepping-Stone>

<Main Problem to Solve>
{main_question}
</Main Problem to Solve>

<Your Answer/Output Format>
Rationale:
[Step-by-step derivation for the Main Problem, using the logical pathway demonstrated in the stepping-stone]

Final Answer:
[Your final answer]
</Your Answer/Output Format>""",

    "core_simp_augmented_solver_v2": """You are an expert math solver. 

You must solve the Main Question by copying the exact logical steps used in the Solved Example. 
The Solved Example is a simpler version of the exact same problem. Do not invent a new method. Read the Solved Example, find the step-by-step logic it used, and apply that exact same logic to the numbers in the Main Question.

<Solved Example>
{solved_proxy_question}
</Solved Example>

<Main Problem to Solve>
{main_question}
</Main Problem to Solve>

<Your Answer/Output Format>
Rationale:
[Write your step-by-step derivation here, strictly copying the method from the Solved Example]

Final Answer:
[Your final answer]
</Your Answer/Output Format>""",


    "core_simplification_few_shot_medium": """You are an expert mathematical AI. Your task is to perform "Core-Preserving Simplification" on a complex math competition problem.

Your goal is to create a simpler "Proxy Question" that acts as an easy analogical stepping-stone. The Proxy Question MUST share the EXACT SAME logical structure, Chain of Thought, and theorems as the original problem, but be computationally and cognitively much easier.

To do this, use the following framework:

STEP 1: Identify the "Core Logical Engine" (The Untouchable Logic)
Identify the fundamental math concept required (e.g., Stars and Bars, Vieta's formulas, modular inverses). You MUST NOT alter the problem so much that this core theorem is bypassed.

STEP 2: Apply Vectors of Simplification
Analyze the problem and safely simplify its peripheral complexities. Below is a condensed toolkit of simplification vectors. 
IMPORTANT: The examples below are basic "toy" examples to illustrate the concepts. You must look past the surface complexity of your actual problem, deeply analyze it, and apply the *underlying principles* of these vectors.

Study the Valid vs. Invalid examples to understand the exact boundaries of what preserves versus what destroys logic:

[Arithmetic & Algebra]
- Downscaling & Smoothing: Reduce giant constants, roots, fractions, and polynomial degrees without bypassing necessary operations.
  Valid: "2024^2024 mod 5" -> "4^4 mod 5" (Modular exponentiation preserved).
  Invalid: "...mod 5" -> "Sum of digits of 1" (Trivializes problem).
  Valid: "Roots of x^100 - 1 = 0" -> "Roots of x^4 - 1 = 0" (Roots of unity preserved).
  Invalid: "Coeff of x^50 in (1+x)^100" -> "Coeff of x^1 in (1+x)^2" (Trivializes binomial expansion).
- Variable & Sequence Reduction: Shorten long sums, products, or symmetric systems.
  Valid: "Telescoping sum to 1000 terms" -> "...to 4 terms" (Cancellation logic remains).
  Invalid: "Limit as n -> infinity" -> "Find the 3rd term" (Destroys calculus limit concept).

[Combinatorics]
- State-Space & Population Shrinking: Reduce grid sizes, set sizes, or probability iterations.
  Valid: "Paths on 10x10 grid avoiding (5,5)" -> "4x4 grid avoiding (2,2)" (Inclusion-exclusion kept).
  Invalid: "...on a 1x1 grid" (No room for obstacle; logic destroyed).
  Valid: "Seat 100 people at round table" -> "Seat 5 people" ((n-1)! formula kept).
  Invalid: "Seat 2 people..." (Yields (2-1)!=1; bypasses combinatorial thought).

[Geometry]
- Normalization & Shape Reduction: Shift coordinates to origin, shrink N-gons, or use standard angles.
  Valid: "Distance from (1005, -2048) to..." -> "Distance from (0,0) to (3,4)" (Distance formula kept).
  Invalid: "Area of polygon with vertices..." -> "...with vertex (0,0)" (A point is not a polygon).
  Valid: "Interior angles of 100-gon" -> "...of a hexagon".
  Invalid: "Diagonals in 100-gon" -> "...in a triangle" (Triangles have 0 diagonals; destroys formula).

[Number Theory & Formatting]
- Modulo/Base Lowering & Un-nesting: Shrink modulos/bases and remove arbitrary final formatting math.
  Valid: "Last 3 digits of 7^999" -> "Last digit of 7^99" (Euler's totient preserved).
  Invalid: "Find x mod 1000" -> "Find x mod 1" (Always 0; destroys modular arithmetic).
  Valid: "Find r, then calc floor(100*pi*r^2 - 17)" -> "Find area pi*r^2" (Removes formatting noise).
  Invalid: "Find roots r1, r2, then calc r1^2 + r2^2" -> "Find roots r1, r2" (Loses algebraic manipulation).

STEP 3: The Mathematical Safety Check
Ensure your selected simplifications do not break math rules:
- Parity/Divisibility: If a number MUST be even for the problem to work, your smaller number must be even.
- Geometry Constraints: Do not violate the Triangle Inequality.
- Probabilities: Numerators must not exceed denominators.
- Keywords: NEVER remove words defining constraints (e.g., "distinct", "integer", "consecutive").

STEP 4: The Failsafe
If the question relies entirely on specific numbers (e.g., factoring a specific prime), or if any simplification would destroy the Core Logical Engine, DO NOT CHANGE IT. Output the exact original question.

STEP 5: Learn from the Donor Demonstration (Few-Shot Analogical Reasoning)
Below is a verified example (the "Donor Demonstration") showing how a structurally similar problem was successfully simplified. 
CRITICAL INSTRUCTION: Use this example to learn *how* to apply the principles and analogical reasoning. However, DO NOT blindly copy its exact transformations. You must independently analyze the new Original Question, use professional mathematical reasoning to recognize which specific simplifications are safe and appropriate for its unique context, and ensure you do not violate the Valid vs. Invalid boundaries defined in STEP 2. Rely on deep structural understanding, not superficial mimicry.

DEMONSTRATION:
{donor_demonstration}

INPUT
Original Question:
{original_question}

OUTPUT FORMAT (Strictly follow this format):
Simplified Question: [Your simplified proxy question here, or the exact original question if no safe simplification is possible]
""",

    "core_simplification_few_shot_small": """You are an expert mathematical AI. Your task is to perform "Core-Preserving Simplification" on a complex math competition problem.

Your goal is to create a simpler "Proxy Question" that acts as an easy analogical stepping-stone. The Proxy Question MUST share the EXACT SAME logical structure, Chain of Thought, and theorems as the original problem, but be computationally and cognitively much easier.

To do this, use the following framework:

STEP 1: Identify the "Core Logical Engine" (The Untouchable Logic)
Identify the fundamental math concept required (e.g., Stars and Bars, Vieta's formulas, modular inverses). You MUST NOT alter the problem so much that this core theorem is bypassed.

STEP 2: Safely Simplify Peripheral Complexities
Analyze the problem and safely simplify its non-core elements. You may downscale giant constants, reduce polynomial degrees, shrink grid/population sizes, or simplify geometric shapes. 
IMPORTANT: You must preserve the mathematical steps required to solve the problem. Do not trivialize the problem (e.g., reducing a complex modulo question to "mod 1", which destroys the logic, or shrinking a polygon to a single point). The simplified parameters must still force the solver to use the exact same underlying theorem.

STEP 3: The Mathematical Safety Check
Ensure your selected simplifications do not break math rules:
- Parity/Divisibility: If a number MUST be even for the problem to work, your smaller number must be even.
- Geometry Constraints: Do not violate the Triangle Inequality.
- Probabilities: Numerators must not exceed denominators.
- Keywords: NEVER remove words defining constraints (e.g., "distinct", "integer", "consecutive").

STEP 4: The Failsafe
If the question relies entirely on specific numbers (e.g., factoring a specific prime), or if any simplification would destroy the Core Logical Engine, DO NOT CHANGE IT. Output the exact original question.

STEP 5: Learn from the Donor Demonstration (Few-Shot Analogical Reasoning)
Below is a verified example (the "Donor Demonstration") showing how a structurally similar problem was successfully simplified. 
CRITICAL INSTRUCTION: Use this example to learn *how* to apply the principles and analogical reasoning. However, DO NOT blindly copy its exact transformations. You must independently analyze the new Original Question, use professional mathematical reasoning to recognize which specific simplifications are safe and appropriate for its unique context, and ensure you do not trivialize the underlying logic. Rely on deep structural understanding, not superficial mimicry.

DEMONSTRATION:
{donor_demonstration}

INPUT
Original Question:
{original_question}

OUTPUT FORMAT (Strictly follow this format):
Simplified Question: [Your simplified proxy question here, or the exact original question if no safe simplification is possible]
""",

    "core_simplification_few_shot_nano": """You are a helpful mathematical AI. Your task is to simplify a complex math problem into an easier "Proxy Question".

To do this, you must keep the EXACT same mathematical logic, theorems, and critical keywords as the original problem, but scale down the complexity. Do not make the problem so simple that it breaks the underlying math or solves itself. If the problem cannot be simplified safely without destroying the core logic, just output the exact original question.

HOW TO DO IT:
Below is a verified example. Look very carefully at how the Original Question in the example was simplified. Notice exactly what parts were changed to be easier, and what parts were left completely untouched to preserve the core math. The example shows you exactly what a safe, effective simplification looks like for this specific type of problem. Deeply understand the example, and then apply that exact same style of simplification to your new problem.

EXAMPLE DEMONSTRATION:
{donor_demonstration}

YOUR TURN:
Original Question:
{original_question}

OUTPUT FORMAT (Strictly follow this format):
Simplified Question: [Your simplified proxy question here, or the exact original question if no safe simplification is possible]
"""




}




def create_reverse_transformation_main_to_exemplar_prompt(main_question: str, exemplar_question: str, exemplar_solution: str, config: Dict[str, Any]) -> str:
    """Creates a prompt to transform the main question to match the retrieved exemplar."""
    template = PROMPT_TEMPLATES["reverse_transformation_main_to_exemplar"]
    exemplar_text = EXEMPLAR_FORMAT.format(question=exemplar_question, solution=exemplar_solution)
    return template.format(main_question=main_question, exemplar_text=exemplar_text)

def create_reverse_transformation_solve_transformed_prompt(transformed_question: str, exemplar_question: str, exemplar_solution: str, config: Dict[str, Any]) -> str:
    """Creates a prompt to solve the transformed main question using the exemplar."""
    template = PROMPT_TEMPLATES["reverse_transformation_solve_transformed"]
    exemplar_text = EXEMPLAR_FORMAT.format(question=exemplar_question, solution=exemplar_solution)
    return template.format(transformed_question=transformed_question, exemplar_text=exemplar_text)

def create_reverse_transformation_final_solve_prompt(original_question: str, transformed_solutions: List[str], config: Dict[str, Any]) -> str:
    """Creates a prompt to solve the original question using transformed solutions."""
    template = PROMPT_TEMPLATES["reverse_transformation_final_solve"]
    transformed_solutions_text = "\n\n".join([f"Transformed Solution {i+1}:\n{sol}" for i, sol in enumerate(transformed_solutions)])
    return template.format(original_question=original_question, transformed_solutions=transformed_solutions_text)

def create_normalization_prompt(original_example: str) -> str:
    template = PROMPT_TEMPLATES["standardization_v1"]
    return template.format(original_example=original_example)

create_standardization_prompt = create_normalization_prompt

def create_transformation_prompt(target_query: str, text_to_transform: str, config: Dict[str, Any], template_key_name: str) -> str:
    template_name = config.get(template_key_name, "transformation_v1")
    if template_name not in PROMPT_TEMPLATES:
        return f"Error: Prompt template '{template_name}' specified by key '{template_key_name}' not found in registry."
    template = PROMPT_TEMPLATES[template_name]
    return template.format(target_query=target_query, text_to_transform=text_to_transform)



def create_final_reasoning_prompt(main_question_text: str, final_examples: List[str], config: Dict[str, Any]) -> str:
    if not final_examples:
        return "Error: At least one example is required for the RAG-based final reasoning prompt."

    template_name = config.get("PROMPT_TEMPLATE_FINAL_SOLVER", "final_solver_v2")
    template = PROMPT_TEMPLATES[template_name]
    
    if template_name in ["final_solver_v2", "final_solver_v3", "final_solver_v4", "analogical_adaptation_v2"]:
        examples_block = ""
        for i, sample_text in enumerate(final_examples):
            examples_block += f"<Example {i+1}>\n{sample_text}\n</Example {i+1}>\n\n"
        return template.format(main_question_text=main_question_text, examples_block=examples_block.strip())
    
    elif template_name == "final_solver_v1":
        samples_block = ""
        for i, sample_text in enumerate(final_examples):
            samples_block += f"\n**Adapted Sample {i+1}:**\n{sample_text}\n"
        return template.format(main_question_text=main_question_text, adapted_samples_block=samples_block.strip())
    
    else:
        return f"Error: Unknown final solver template '{template_name}' specified in config."


def create_final_reasoning_prompt_simple(main_question_text: str, config: Dict[str, Any]) -> str:
    template_name = config.get("PROMPT_TEMPLATE_FINAL_SOLVER_SIMPLE", "final_solver_simple_v1")
    template = PROMPT_TEMPLATES[template_name]
    return template.format(main_question_text=main_question_text)


def is_prompt_construction_error(prompt: str) -> bool:
    """Return whether a prompt builder returned its explicit error sentinel."""
    return prompt.startswith("Error:")

def create_evaluation_prompt(model_answer: str, ground_truth: str, config: Dict[str, Any], template_name: Optional[str] = None) -> str:
    if not template_name:
        target_benchmark = benchmark_name_for_target_index(config)
        if uses_exact_final_answers(target_benchmark):
            template_name = "evaluator_v2"
        else:
            template_name = config.get("PROMPT_TEMPLATE_EVALUATOR", "evaluator_v1")
    template = PROMPT_TEMPLATES[template_name]
    return template.format(model_answer=model_answer, ground_truth=ground_truth)

def create_duplicate_check_prompt(main_question_text: str, retrieved_questions: List[str]) -> str:
    template = PROMPT_TEMPLATES["duplicate_question_check_v1"]
    retrieved_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(retrieved_questions))
    return template.format(main_question_text=main_question_text, retrieved_questions_block=retrieved_block.strip())


def _create_single_example_analogical_prompt(
    main_question: str,
    candidate_text: str,
    template_name: str,
) -> str:
    """Format one retained-feature exemplar with a shared analogical template."""
    if template_name not in PROMPT_TEMPLATES:
        return f"Error: Prompt template '{template_name}' not found in registry."
    sample_block = f"<Sample 1>\n{candidate_text}\n</Sample 1>"
    return PROMPT_TEMPLATES[template_name].format(
        main_question_text=main_question,
        samples_block=sample_block,
        examples_block=sample_block,
    )


def create_best_of_transformation_solver_prompt(
    main_question: str,
    candidate_text: str,
    config: Dict[str, Any],
) -> str:
    template_name = config.get(
        "PROMPT_TEMPLATE_BEST_OF_TRANSFORMATION_SOLVER",
        "analogical_adaptation_v1",
    )
    return _create_single_example_analogical_prompt(
        main_question, candidate_text, template_name
    )


def create_reverse_validation_candidate_prompt(
    main_question: str,
    config: Dict[str, Any],
) -> str:
    template_name = config.get(
        "PROMPT_TEMPLATE_REVERSE_VALIDATION_CANDIDATE_GENERATOR",
        "reverse_validation_candidate_generator_v1",
    )
    if template_name not in PROMPT_TEMPLATES:
        return f"Error: Prompt template '{template_name}' not found in registry."
    return PROMPT_TEMPLATES[template_name].format(main_question_text=main_question)








def create_reverse_validation_prompt(validator_question: str, candidate_text: str, config: Dict[str, Any]) -> str:
    template_name = config.get("PROMPT_TEMPLATE_REVERSE_VALIDATION_SOLVER", "reverse_validation_v1")
    if template_name == "analogical_adaptation_v1" or template_name == "analogical_adaptation_v2":
        return _create_single_example_analogical_prompt(
            validator_question, candidate_text, template_name
        )
    if template_name not in PROMPT_TEMPLATES:
        return f"Error: Prompt template '{template_name}' not found in registry."
    template = PROMPT_TEMPLATES[template_name]
    return template.format(validator_question=validator_question, candidate_exemplar=candidate_text)


def create_simplification_prompt(text_to_simplify: str, config: Dict[str, Any]) -> str:
    template_name = config.get("PROMPT_TEMPLATE_SIMPLIFICATION_GENERATOR", "simplification_generator_v1")
    if template_name not in PROMPT_TEMPLATES:
        return f"Error: Prompt template '{template_name}' not found in registry."
    template = PROMPT_TEMPLATES[template_name]
    return template.format(text_to_simplify=text_to_simplify)

def create_simplified_sample_solver_prompt(simplified_question: str, original_exemplar: str, config: Dict[str, Any]) -> str:
    template_name = config.get("PROMPT_TEMPLATE_SIMPLIFIED_SAMPLE_SOLVER", "simplified_sample_solver_v1")
    if template_name not in PROMPT_TEMPLATES:
        return f"Error: Prompt template '{template_name}' not found in registry."
    template = PROMPT_TEMPLATES[template_name]
    return template.format(simplified_question=simplified_question, original_exemplar=original_exemplar)

def create_main_from_simplified_proxy_prompt(original_main_question: str, simplified_solution: str, config: Dict[str, Any]) -> str:
    template_name = config.get("PROMPT_TEMPLATE_SIMPLIFIED_MAIN_PROXY_SOLVER", "main_from_simplified_proxy_v1")
    if template_name not in PROMPT_TEMPLATES:
        return f"Error: Prompt template '{template_name}' not found in registry."
    template = PROMPT_TEMPLATES[template_name]
    return template.format(original_main_question=original_main_question, simplified_solution=simplified_solution)



def create_mirror_hypothesis_zeroshot_prompt(target_query: str, config: Dict[str, Any]) -> str:
    """
    Creates a prompt to solve the Target Query zero-shot (for the R0 candidate).
    """
    template_name = config.get("PROMPT_TEMPLATE_MIRROR_HYPOTHESIS_ZEROSHOT", "mirror_hypothesis_gen_zero_shot_v1")
    if template_name not in PROMPT_TEMPLATES:
        return f"Error: Template {template_name} not found."
    template = PROMPT_TEMPLATES[template_name]
    return template.format(target_query=target_query)

def create_core_simp_zero_shot_prompt(original_question: str) -> str:
    template = PROMPT_TEMPLATES["core_simplification_zero_shot_v1"]
    return template.format(original_question=original_question)

def create_core_simp_few_shot_short_prompt(original_question: str, donor_demonstration: str, config: dict = None) -> str:
    """
    Creates a concise, short-instruction few-shot prompt for core-preserving simplification (Branch D).
    """
    config = config or {}
    
    # Fetch the template name from config, fallback to the medium template key
    template_name = config.get("PROMPT_TEMPLATE_CORE_SIMP_BRANCH_D", "core_simplification_few_shot_medium")
    
    # Safely get the template from the dictionary
    template = PROMPT_TEMPLATES.get(template_name)
    
    # Absolute fallback just in case the dictionary lookup fails
    if not template:
        template = PROMPT_TEMPLATES.get("core_simplification_few_shot_medium")

    # The keys here MUST match {original_question} and {donor_demonstration} in the string
    return template.format(
        original_question=original_question,
        donor_demonstration=donor_demonstration
    )

def create_core_simp_few_shot_prompt(original_question: str, donor_demonstration: str, config: dict = None) -> str:
    """
    Creates a detailed few-shot prompt for core-preserving simplification (Branch C).
    """
    config = config or {}
    
    # Fetch the template name from config, fallback to the v1 template key
    template_name = config.get("PROMPT_TEMPLATE_CORE_SIMP_BRANCH_C", "core_simplification_few_shot_v1")
    
    # Safely get the template
    template = PROMPT_TEMPLATES.get(template_name)
    
    # Absolute fallback just in case
    if not template:
        template = PROMPT_TEMPLATES.get("core_simplification_few_shot_v1")
    
    return template.format(
        original_question=original_question,
        donor_demonstration=donor_demonstration
    )

def create_core_simp_augmented_solver_prompt(main_question: str, solved_proxy_question: str) -> str:
    template = PROMPT_TEMPLATES["core_simp_augmented_solver_v1"]
    return template.format(
        main_question=main_question,
        solved_proxy_question=solved_proxy_question
    )
