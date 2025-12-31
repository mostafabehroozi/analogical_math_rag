#======================================================================
#   File: src/prompts.py
#======================================================================

from typing import List, Dict, Any

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
- Adapt the merged sample's phrasing and structure so it stylistically resembles the Target Question.  
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

    "final_solver_v3": """You are an expert solver. Solve the Main Question, using the Solved Examples as inspiration for your methodology.
    
<Instructions>
Look for patterns in the Solved Examples that might help solve the Main Question. Adopt these strategies where they fit, but feel free to deviate from them if the new question requires a different logic. Treat the examples as suggestions, not strict rules.
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

"final_solver_v4": """Use the Solved Examples as a blueprint to solve the Main Question.
<Instructions>
- Study the examples to understand the logical method used to reach the solution.
- Extract the key reasoning pattern.
- Apply this pattern to the Main Question.
- Provide the solution as a concise, technical derivation. Focus strictly on the mathematical steps and logic, avoiding conversational language.
</Instructions>

<Solved Examples>
{examples_block}
</Solved Examples>

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

    "self_sampling_generator": """Objective:
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

    "self_sampling_augmentor_v1": """You are an expert math problem creator. Your task is to generate {n_samples} new, distinct math problems that are conceptually similar to the provided 'Base Question'.

<Instructions>
1. Read the 'Base Question' to understand its core mathematical concept, structure, and context.
2. Create {n_samples} new questions s that are relevant to the base question. Your new questions should be distinct from each other and from the base question (e.g., involving different numbers and names).
3. Crucially, you must NOT provide any rationale or answer. Your output should consist ONLY of the generated question statements.
4. Ensure the generated questions are unique from each other and from the base question.
5. Present your output as a numbered list, as shown in the example.
</Instructions>

<Example>
<Base Question>
An airline serves a dinner to all the passengers on an airplane. They get their choice of steak or fish. Three steak meals and three fish meals are set aside for the six-member crew. If the meals are distributed to the crew members randomly, what is the probability that both pilots get the fish?
</Base Question>

<Your Output (for n_samples = 2)>
1. In a bag, there are 5 red balls and 3 blue balls. If two balls are drawn at random without replacement, what is the probability that both balls are red?
2. A box contains 10 red marbles and 5 blue marbles. If three marbles are drawn at random without replacement, what is the probability that all three marbles are red?
</Your Output>
</Example>

<Task>
<Base Question>
{main_question_text}
</Base Question>
</Task>
""",


    "self_sampling_augmentor_decomposition":"""You are an expert mathematical reasoning assistant. Your task is to decompose a complex 'Base Question' into {n_samples} simpler, distinct sub-problems that isolate specific mathematical concepts or logical steps found in the base question.

<Instructions>
1. Analyze the 'Base Question' to identify its underlying distinct mathematical components (e.g., the counting method, the probability rule, the geometric property, or the arithmetic relationship).
2. Create {n_samples} new math questions.
3. CRITICAL CONSTRAINTS for the new questions:
    - Simpler: They must be somewhat easier than the Base Question.
    - Decomposed Perspectives: Each generated question should focus on a *different* mechanism or aspect of the Base Question. (For example, if the base question involves probability with combinations, Problem 1 might strictly test calculating combinations, while Problem 2 tests the specific probability logic with smaller numbers).
    - Distinct: They must employ entirely different scenarios, contexts, or objects compared to the Base Question, ensuring the new problems differ significantly in setting (not just changing the numbers).
4. Do NOT provide any rationale, steps, or solutions.
5. Output ONLY the numbered list of question statements.
</Instructions>

<Example>
<Base Question>
An airline serves a dinner to all the passengers on an airplane. They get their choice of steak or fish. Three steak meals and three fish meals are set aside for the six-member crew. If the meals are distributed to the crew members randomly, what is the probability that both pilots get the fish?
</Base Question>

<Your Output (for n_samples = 2)>
1. How many distinct ways can you arrange 3 red balls and 3 blue balls in a row of 6 slots?
2. There are 6 seats numbered 1 to 6. If we choose 2 seats at random to be the "Captain's seats", what is the probability that we chose specifically seat #1 and seat #2?
</Your Output>
</Example>

<Task>
<Base Question>
{main_question_text}
</Base Question>
</Task>
""",

    "self_sampling_augmentor_decomposition_2":"""You are an expert mathematical reasoning assistant. Your task is to decompose a complex 'Base Question' into {n_samples} simpler, distinct sub-problems that isolate specific mathematical concepts or logical steps found in the base question.

<Instructions>
1. Analyze the 'Base Question' to identify its underlying distinct mathematical components (e.g., the counting method, the probability rule, the geometric property, or the arithmetic relationship).
2. Create {n_samples} new math questions.
3. CRITICAL CONSTRAINTS for the new questions:
    - Simpler: They must be somewhat easier than the Base Question.
    - Decomposed Perspectives: Each generated question should focus on a *different* mechanism or aspect of the Base Question. (For example, if the base question involves probability with combinations, Problem 1 might strictly test calculating combinations, while Problem 2 tests the specific probability logic with smaller numbers).
    - Distinct: They must employ entirely different scenarios, contexts, or objects compared to the Base Question, ensuring the new problems differ significantly in setting (not just changing the numbers).
4. Do NOT provide any rationale, steps, or solutions.
5. Output ONLY the numbered list of question statements.
</Instructions>

<Example>
<Base Question>
An airline serves a dinner to all the passengers on an airplane. They get their choice of steak or fish. Three steak meals and three fish meals are set aside for the six-member crew. If the meals are distributed to the crew members randomly, what is the probability that both pilots get the fish?
</Base Question>

<Your Output (for n_samples = 2)>
1. A flight attendant has 3 steak meals and 3 fish meals on a cart. In how many distinct ways can these 6 meals be distributed to 6 specific crew members?
2. A crew of 6 people includes exactly 2 pilots. If we randomly select a group of 3 crew members to receive the fish meals, what is the probability that the group chosen includes both pilots?
</Your Output>
</Example>

<Task>
<Base Question>
{main_question_text}
</Base Question>
</Task>
""",
    "self_sampling_augmentor_decomposition_3":"""You are an expert mathematical reasoning assistant. Your task is to decompose a complex 'Base Question' into {n_samples} simpler, distinct sub-problems that isolate specific mathematical concepts or logical steps found in the base question.

<Instructions>
1. Analyze the 'Base Question' to identify its underlying distinct mathematical components (e.g., the counting method, the probability rule, the geometric property, or the arithmetic relationship).
2. Create {n_samples} new math questions.
3. CRITICAL CONSTRAINTS for the new questions:
    - Simpler: They must be moderately simpler than the Base Question and conceptually isolated to reduce overall difficulty.
    - Decomposed Perspectives: Each generated question should focus on a *different* mechanism or aspect of the Base Question. (For example, if the base question involves probability with combinations, Problem 1 might strictly test calculating combinations, while Problem 2 tests the specific probability logic with smaller numbers).
    - Distinct: They must employ entirely different scenarios, contexts, or objects compared to the Base Question, ensuring the new problems differ significantly in setting (not just changing the numbers).
4. Do NOT provide any rationale, steps, or solutions.
5. Output ONLY the numbered list of question statements.
</Instructions>

<Example>
<Base Question>
An airline serves a dinner to all the passengers on an airplane. They get their choice of steak or fish. Three steak meals and three fish meals are set aside for the six-member crew. If the meals are distributed to the crew members randomly, what is the probability that both pilots get the fish?
</Base Question>

<Your Output (for n_samples = 2)>
1. A flight attendant has 3 steak meals and 3 fish meals on a cart. In how many distinct ways can these 6 meals be distributed to 6 specific crew members?
2. A crew of 6 people includes exactly 2 pilots. If we randomly select a group of 3 crew members to receive the fish meals, what is the probability that the group chosen includes both pilots?
</Your Output>
</Example>

<Task>
<Base Question>
{main_question_text}
</Base Question>
</Task>
""",
    "self_sampling_augmentor_decomposition_4":"""You are an expert mathematical reasoning assistant. Your task is to decompose a complex 'Base Question' into {n_samples} simpler, distinct sub-problems that isolate specific mathematical concepts or logical steps found in the base question.
<Instructions>
1. Analyze the 'Base Question' to identify its underlying distinct mathematical components (e.g., the counting method, the probability rule, the geometric property, or the arithmetic relationship).
2. Create {n_samples} new math questions.
3. CRITICAL CONSTRAINTS for the new questions:
    - Moderately Simpler & Reduced Complexity: The new questions must be distinctly easier than the Base Question. Reduce the cognitive load by lowering numerical magnitudes or removing multi-step logic layers, ensuring the core concept remains but the execution is more straightforward.
    - Decomposed Perspectives: Each generated question should focus on a *different* mechanism or aspect of the Base Question. (For example, if the base question involves probability with combinations, Problem 1 might strictly test calculating combinations, while Problem 2 tests the specific probability logic with smaller numbers).
    - Distinct Contexts: They must employ entirely different scenarios, contexts, or objects compared to the Base Question, ensuring the new problems differ significantly in setting (not just changing the numbers).
4. Do NOT provide any rationale, steps, or solutions.
5. Output ONLY the numbered list of question statements.
</Instructions>

<Example>
<Base Question>
An airline serves a dinner to all the passengers on an airplane. They get their choice of steak or fish. Three steak meals and three fish meals are set aside for the six-member crew. If the meals are distributed to the crew members randomly, what is the probability that both pilots get the fish?
</Base Question>
<Your Output (for n_samples = 2)>
1. A flight attendant has 3 steak meals and 3 fish meals on a cart. In how many distinct ways can these 6 meals be distributed to 6 specific crew members?
2. A crew of 6 people includes exactly 2 pilots. If we randomly select a group of 3 crew members to receive the fish meals, what is the probability that the group chosen includes both pilots?
</Your Output>
</Example>

<Task>
<Base Question>
{main_question_text}
</Base Question>
</Task>
""",
    "self_sampling_augmentor_decomposition_complex":"""You are an expert Logical Decomposition Specialist and Mathematical Architect. Your objective is not merely to simplify a problem, but to deconstruct a complex 'Base Question' into {n_samples} distinct, orthogonal sub-problems using the method of **Variable Isolation via Zero-State Application**.

<Core_Philosophy>
You must view the Base Question as a system composed of a **Trunk** and multiple **Aspects**.
1.  **The Trunk (The Skeleton):** The fundamental reasoning pathway, physical law, or algorithmic structure that defines the problem type. This MUST exist in every sub-question. You effectively "freeze" the core logic.
2.  **The Aspects (The Variables):** The specific constraints, complexities, or forces acting upon the Trunk (e.g., friction, tax rates, conditional probabilities).
3.  **Orthogonality:** Each sub-question must focus on exactly ONE active Aspect while the others are dormant.
4.  **The Zero-State Rule:** You do not *delete* the other aspects; you set them to their "Identity Value" or "Ideal State" so they no longer contribute to the difficulty but the Trunk remains valid.
</Core_Philosophy>

<Instructions>
1.  **Analyze the Base Question:**
    *   Identify the **Trunk**. (e.g., "Newton's Second Law" or "Compound Interest Formula").
    *   List all distinct **Aspects** (complexities) present in the prompt.
    *   Determine the **Zero-State** for each aspect (e.g., Friction → 0, Efficiency → 100%, Delay → 0 seconds).

2.  **Generate {n_samples} Sub-Questions:**
    *   For each sub-question, select **ONE** target Aspect to highlight.
    *   Apply the **Zero-State Rule** to all *other* non-target aspects.
    *   **Preserve the Trunk:** Ensure the core solving method remains identical to the Base Question, even if the values are simplified.

3.  **Strict Constraints:**
    *   **No Logic Deletion:** Do not remove the Trunk. If the Base Question is about calculating speed, every sub-question must still be about calculating speed/motion, not just "counting objects."
    *   **Independence:** The specific challenge in Question 1 must not overlap with the specific challenge in Question 2. They must be orthogonal.
    *   **Contextual Divergence (CRITICAL):** To prevent analogical hallucination, the sub-questions must NOT use the same setting or objects as the Base Question. If the Base Question is about "Cars," the sub-questions must be about "Blocks," "Particles," or "Widgets." Change the *Skin*, keep the *Math*.

4.  **Verification:**
    *   Ensure that if a student solves all {n_samples} questions, they have practiced every mechanism required to solve the Base Question.
</Instructions>

<Zero_State_Protocol>
When deactivating an Aspect, use these canonical values:
*   **Additive constraints** (e.g., wind speed, fees): Set to **0**.
*   **Multiplicative constraints** (e.g., coefficients, efficiency): Set to **1** (or 100%).
*   **Logic gates** (e.g., "if it is raining"): Set to the **Simplest Valid Truth Value** (e.g., "It is sunny/dry").
*   *Warning:* Do not set values that cause division by zero or logical paradoxes.
</Zero_State_Protocol>

<One_Shot_Example>
<Base_Question>
A 1000kg car drives up a 30-degree incline. The coefficient of kinetic friction is 0.1, and air resistance acts against the car with a force of 200N. Calculate the total engine force required to maintain a constant velocity.
</Base_Question>

<Reasoning_Trace>
*   **Trunk:** Forces Logic (Sum of opposing forces = Engine Force).
*   **Aspects:** 1. Gravity (Incline), 2. Friction, 3. Air Resistance.
*   **Zero-States:** Incline → 0 degrees (Flat), Friction → $\mu=0$ (Smooth), Air Resistance → 0N (Vacuum).
*   **Context Strategy:** Change "Car/Road" to "Block/Ramp", "Crate/Floor", "Sled/Track".
</Reasoning_Trace>

<Your_Output (for n_samples = 3)>
1. A heavy stone block of mass 1000kg is being pushed up a smooth, frictionless ramp angled at 30 degrees. Assuming a vacuum (no air resistance), calculate the pushing force required to keep the block moving at a constant velocity.
2. A large wooden crate (1000kg) slides across a flat, horizontal warehouse floor (0-degree incline). The floor is rough with a kinetic friction coefficient of 0.1, and we assume air resistance is negligible. Calculate the horizontal force needed to maintain constant velocity.
3. A 1000kg test sled moves along a flat, smooth horizontal track (no friction). However, a parachute attached to the back creates a drag force of 200N. Calculate the propulsion force required to maintain constant velocity.
</Your_Output>
</One_Shot_Example>

<Task>
<Base_Question>
{main_question_text}
</Base_Question>
</Task>

<Output_Requirement>
Output ONLY the numbered list of question statements. Do not include your reasoning trace or target labels in the final output.
</Output_Requirement>
""",

    "self_sampling_augmentor_decomposition_complex_2":"""You are an expert Logical Decomposition Specialist and Mathematical Architect. Your objective is to deconstruct a complex 'Base Question' into {n_samples} distinct, orthogonal sub-problems using the method of **Variable Isolation via Zero-State Application**.

<Core_Philosophy>
You must view the Base Question as a system composed of a **Trunk** and multiple **Aspects**.
1.  **The Trunk (The Skeleton):** The fundamental reasoning pathway or formula. This MUST exist in every sub-question.
2.  **The Aspects (The Variables):** The specific constraints acting upon the Trunk.
3.  **Orthogonality:** Each sub-question must focus on exactly ONE active Aspect while the others are dormant (Zero-State).
4.  **The Zero-State Rule:** Do not delete aspects; set them to their "Identity Value" or "Ideal State" so the Trunk remains executable but the aspect is trivialized.
</Core_Philosophy>

<Instructions>
1.  **Analyze the Base Question:** Identify the Trunk (logic), the Aspects (complexities), and the Zero-State for each aspect.
2.  **Generate {n_samples} Sub-Questions:**
    *   Select **ONE** target Aspect to highlight per question.
    *   Apply the **Zero-State Rule** to all *other* non-target aspects.
    *   **Preserve the Trunk:** Ensure the core solving method remains identical.
3.  **Strict Constraints:**
    *   **No Logic Deletion:** Do not remove the Trunk. The underlying equation/logic must remain isomorphic.
    *   **Contextual Divergence (CRITICAL):** To prevent analogical hallucination, the sub-questions must NOT use the same setting or objects as the Base Question. If the Base Question is about "Cars," the sub-questions must be about "Blocks," "Particles," or "Widgets." Change the *Skin*, keep the *Math*.
    *   **Independence:** The specific challenge in Question 1 must not overlap with Question 2.

<Zero_State_Protocol>
*   **Additive constraints** (e.g., wind, fees): Set to **0**.
*   **Multiplicative constraints** (e.g., efficiency): Set to **1** (or 100%).
*   **Logic gates**: Set to the **Simplest Valid Truth Value**.
</Zero_State_Protocol>

<One_Shot_Example>
<Base_Question>
A 1000kg car drives up a 30-degree incline. The coefficient of kinetic friction is 0.1, and air resistance acts against the car with a force of 200N. Calculate the total engine force required to maintain a constant velocity.
</Base_Question>

<Reasoning_Trace>
*   **Trunk:** Forces Logic (Sum of opposing forces = Forward Force).
*   **Aspects:** Gravity (Incline), Friction, Drag.
*   **Context Strategy:** Base is "Car/Road". Sub-questions must use different scenarios like "Block/Ramp" or "Sled/Ice".
</Reasoning_Trace>

<Your_Output (for n_samples = 3)>
1. A heavy stone block of mass 1000kg is being pushed up a smooth, frictionless ramp angled at 30 degrees. Assuming a vacuum (no air resistance), what is the pushing force required to keep the block moving at a constant velocity?
2. A large wooden crate (1000kg) slides across a flat, horizontal warehouse floor. The floor is rough with a kinetic friction coefficient of 0.1. Ignoring air resistance, calculate the horizontal force needed to maintain constant velocity.
3. A 1000kg test sled moves along a flat, frictionless magnetic track. A parachute attached to the back creates a drag force of 200N. Calculate the propulsion force required to maintain constant velocity.
</Your_Output>
</One_Shot_Example>

<Task>
<Base_Question>
{main_question_text}
</Base_Question>
</Task>

<Output_Requirement>
Output ONLY the numbered list of question statements. Do not include your reasoning trace.
</Output_Requirement>
""",
    
    "self_sampling_augmentor_simplification":"""You are an expert mathematical simplification assistant. Your task is to take a 'Base Question' and produce a ONE-STEP SIMPLIFIED version of it. This simplified version will be used as a stepping stone to solve the original problem.

<Instructions>
1. Analyze the 'Base Question' to identify the "Core Logic" (the main rule or formula needed) and the "Complexity Layers" (large numbers, extra arithmetic steps, or difficult variables).
2. Create EXACTLY ONE new math question.
3. CRITICAL CONSTRAINTS for the new question:
    - Lite Simplification: Make the question ONLY slightly easier. Do not make it trivial.
    - Prune One Leaf: Remove exactly one layer of complexity. For example:
        *   If the numbers are large, make them smaller integers.
        *   If there is a pre-calculation required (e.g., "radius is 2+3"), change it to the direct value ("radius is 5").
        *   If there are many variables, remove one variable.
    - Preserve the Trunk: The main mathematical logic required to solve the simplified question MUST remain the same as the Base Question.
    - Standalone: The new question must be a complete sentence and solvable.
4. Do NOT provide any rationale, steps, or solutions.
5. Output ONLY the simplified question statement.
</Instructions>

<Example>
<Base Question>
Find the area of a circle inscribed in a square that has a diagonal length of 8.
</Base Question>

<Your Output>
Find the area of a circle inscribed in a square that has a side length of 4.
</Your Output>
(Note: The simplification removed the step of calculating the side from the diagonal, but kept the core logic of the circle-square relationship.)
</Example>

<Task>
<Base Question>
{main_question_text}
</Base Question>
</Task>
""",

    "self_sampling_augmentor_simplification_shallow":"""You are an expert mathematical simplification assistant specializing in Shallow Simplification. Your task is to take a 'Base Question' and produce a version that is computationally trivial but logically identical.

<Instructions>
1. Analyze the 'Base Question' to identify "Surface Noise." This includes complex numbers (decimals, fractions, large integers, irrationals like $\pi$), messy units, or overly wordy variable descriptions.
2. Create EXACTLY ONE new math question.
3. CRITICAL CONSTRAINTS for the new question:
    - Numerical Smoothing: Replace ALL difficult values with "Toy Integers" (e.g., replace 4.87 with 5, replace 137 with 10, replace $2\sqrt 3$ with 4).
    - Syntactic Cleanup: If the question uses complex units or wordy names for variables, standardize them (e.g., change "nautical miles" to "meters", change "the number of apples John holds" to "x").
    - Logic Lock: Do NOT remove any reasoning steps, formulas, or intermediate derivations. The path to the solution must remain exactly the same, only the arithmetic should become effortless.
    - Standalone: The new question must be a complete sentence and solvable.
4. Do NOT provide any rationale, steps, or solutions.
5. Output ONLY the simplified question statement.
</Instructions>

<Example>
<Base Question>
Calculate the kinetic energy of an object with a mass of 4.5kg moving at a velocity of $12.2 m/s$.
</Base Question>

<Your Output>
Calculate the kinetic energy of an object with a mass of 2kg moving at a velocity of $4 m/s$.
</Your Output>
(Note: The decimal values were replaced with simple integers to make the calculation trivial, but the physics formula required remains exactly the same.)
</Example>

<Task>
<Base Question>
{main_question_text}
</Base Question>
</Task>
""",
    "self_sampling_augmentor_simplification_simple_shallow":"""Your task is to create a simplified version of the Main Question that retains the core logical problem but simplifies the shallow or non-core elements.
<Instructions>
1. Replace large numbers or complex values with small, single-digit integers to make the math easier.
2. Rewrite complex scenarios or wordy descriptions into a plain, direct statement to remove distractions.
3. Reduce the number of steps or constraints slightly, as long as the underlying relationship remains the same.
4. CRITICAL: If the question is already simple or cannot be simplified without breaking the core logic, or if the simplification results in a drastic change rather than a shallow adjustment, do NOT change it; just repeat the Main Question exactly.
</Instructions>
<Main Question>
{main_question_text}
</Main Question>
<Output Format>
Simplified Question:
[Your simplified version]
</Output Format>
""",

    "self_sampling_augmentor_simplification_safe":"""You are a Logic Preservation Engine. Your goal is to analyze the <Main Question> to determine if it can be "Safely Simplified" for an Analogical Reasoning task.

<Definition of Safe Simplification>
A "Safe Simplification" creates an easier version of the problem to solve as a reference, BUT it must adhere to these strict rules:
1. PRESERVE LOGIC: The mathematical formulas, logical relationships, and required solution steps must remain identical to the original.
2. REDUCE NOISE: You may replace large numbers with small integers (e.g., 5,392 -> 3), or remove narrative fluff (names, backstory).
3. DO NO HARM: If changing a number or removing a sentence alters the fundamental question type or logic, it is UNSAFE.
</Definition of Safe Simplification>

<Evaluation Protocol>
Before generating an output, analyze the question for these conditions:
- Condition A (Simplifiable): The question uses large numbers or distracting text that are NOT essential to the logic.
- Condition B (Unsafe/Core Only): The question is already short, abstract, or every number/constraint is structurally vital (e.g., specific constants, core puzzle constraints).

<Instructions>
- IF Condition A is met: Rewrite the question using small numbers and plain language. Keep the core logic exactly the same.
- IF Condition B is met (or if you are unsure): Do NOT simplify. Output the <Main Question> exactly as it appears word-for-word.
- WARNING: Better to output the original complex question than a simplified version with broken logic.
</Instructions>

<Main Question>
{main_question_text}
</Main Question>

<Output Format>
Simplified Question:
[Your Output Here]
</Output Format>
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
    
    "hierarchical_parent_solver_v1": """You are an expert mathematical problem solver.
You are tasked with solving a **Main Question**.
To assist you, we have broken this problem down into several related sub-problems or variations, and provided their solutions below.

<Instructions>
1. Analyze the 'Solved Variations'. Identify the underlying mathematical principles, formulas, or logic used to solve them.
2. Apply these principles to the 'Main Question'. The Main Question is the parent problem of these variations, so the logic should be directly applicable or composable.
3. Provide a clear, step-by-step rationale for the Main Question.
4. State the Final Answer clearly.
</Instructions>

<Solved Examples>
{child_solutions_block}
</Solved Examples>

<Main Question to Solve>
{main_question_text}
</Main Question to Solve>

<Output Format>
Rationale:
[Step-by-step derivation]

Final Answer:
[The final result]
</Output Format>
""",
    "hierarchical_parent_solver_v2": """You are an expert in analogical reasoning, highly skilled at identifying and extracting patterns, reasoning pathways, problem-solving strategies, and conceptual frameworks from similar solved examples. Your primary task is to solve the main question by drawing meaningful analogies from the provided solved variations.

<Instructions>
Carefully analyze each variation: pinpoint common reasoning steps, patterns (including structural similarities, logical sequences, mathematical transformations, conceptual mappings, or recurring problem-solving techniques), and effective strategies that led to the final answers. Focus on extracting only the most useful and relevant elements from these variations as supportive guides. Adapt them flexibly to fit the unique aspects of the main question, even when surface details differ, while prioritizing your own independent reasoning to develop a robust solution.
</Instructions>

<Solved Variations>
{child_solutions_block}
</Solved Variations>

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

    "hierarchical_parent_solver_v3": """Use the Solved Examples as a blueprint to solve the Main Question.
<Instructions>
- Study the examples to understand the logical method used to reach the solution.
- Extract the key reasoning pattern.
- Apply this pattern to the Main Question.
- Provide the solution as a concise, technical derivation. Focus strictly on the mathematical steps and logic, avoiding conversational language.
</Instructions>

<Solved Variations>
{child_solutions_block}
</Solved Variations>

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
    "hierarchical_parent_solver_v4": """Use the Solved Examples as a blueprint to solve the Main Question.
<Instructions>
- Study the examples to understand the logical method used to reach the solution.
- Extract the key reasoning pattern.
- Apply this pattern to the Main Question.
</Instructions>

<Solved Variations>
{child_solutions_block}
</Solved Variations>

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
    "hierarchical_parent_solver_v5": """Use the method from the Simplified Reference to solve the Main Question.

<Instructions>
1. The Simplified Reference solves the same problem with easier numbers/logic.
2. Follow the exact same sequence of steps used in the Reference, but apply them to the Main Question's values.
3. Perform the calculations carefully. The logic is identical, but the arithmetic is more complex.
</Instructions>

<Simplified Reference>
{child_solutions_block}
</Simplified Reference>

<Main Question to Solve>
{main_question_text}
</Main Question to Solve>

<Your Answer/Output Format>
Rationale:
[Your step-by-step rationale for the Main Question]

Final Answer:
[Your final answer]
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

    "self_sampling_augmentor_simplification_with_solution": """You are a Logic Preservation Engine. Your goal is to analyze the <Main Question> and its <Reference Solution> to determine if it can be "Safely Simplified" for an Analogical Reasoning task.

<Definition of Safe Simplification>
A "Safe Simplification" creates an easier version of the problem to solve as a reference, BUT it must adhere to these strict rules:
1. PRESERVE LOGIC: The mathematical formulas, logical relationships, and required solution steps must remain identical to the original.
2. REDUCE NOISE: You may replace large numbers with small integers (e.g., 5,392 -> 3), or remove narrative fluff (names, backstory).
3. DO NO HARM: If changing a number or removing a sentence alters the fundamental question type or logic, it is UNSAFE.
</Definition of Safe Simplification>

<Evaluation Protocol>
Before generating an output, analyze the question and solution for these conditions:
- Condition A (Simplifiable): The question uses large numbers or distracting text that are NOT essential to the logic (as confirmed by the Reference Solution).
- Condition B (Unsafe/Core Only): The question is already short, abstract, or every number/constraint is structurally vital (e.g., specific constants, core puzzle constraints).

<Instructions>
- IF Condition A is met: Rewrite the question using small numbers and plain language. Keep the core logic exactly the same.
- IF Condition B is met (or if you are unsure): Do NOT simplify. Output the <Main Question> exactly as it appears word-for-word.
- WARNING: Better to output the original complex question than a simplified version with broken logic.
</Instructions>

<Main Question>
{main_question_text}
</Main Question>

<Reference Solution>
{generated_solution}
</Reference Solution>

<Output Format>
Simplified Question:
[Your Output Here]
</Output Format>
""",

    "self_sampling_augmentor_simplification_shallow_with_solution": """You are an expert mathematical simplification assistant specializing in Shallow Simplification. Your task is to take a 'Base Question' and produce a version that is computationally trivial but logically identical, aided by the provided 'Reference Solution'.

<Instructions>
1. Analyze the 'Base Question' and 'Reference Solution' to identify "Surface Noise." This includes complex numbers (decimals, fractions, large integers, irrationals like $\pi$), messy units, or overly wordy variable descriptions visible in the calculation steps.
2. Create EXACTLY ONE new math question.
3. CRITICAL CONSTRAINTS for the new question:
    - Numerical Smoothing: Replace ALL difficult values with "Toy Integers" (e.g., replace 4.87 with 5, replace 137 with 10, replace $2\sqrt 3$ with 4).
    - Syntactic Cleanup: If the question uses complex units or wordy names for variables, standardize them (e.g., change "nautical miles" to "meters", change "the number of apples John holds" to "x").
    - Logic Lock: Do NOT remove any reasoning steps, formulas, or intermediate derivations found in the Reference Solution. The path to the solution must remain exactly the same, only the arithmetic should become effortless.
    - Standalone: The new question must be a complete sentence and solvable.
4. Do NOT provide any rationale, steps, or solutions.
5. Output ONLY the simplified question statement.
</Instructions>

<Example>
<Base Question>
Calculate the kinetic energy of an object with a mass of 4.5kg moving at a velocity of $12.2 m/s$.
</Base Question>

<Your Output>
Calculate the kinetic energy of an object with a mass of 2kg moving at a velocity of $4 m/s$.
</Your Output>
(Note: The decimal values were replaced with simple integers to make the calculation trivial, but the physics formula required remains exactly the same.)
</Example>

<Task>
<Base Question>
{main_question_text}
</Base Question>

<Reference Solution>
{generated_solution}
</Reference Solution>
</Task>
""",
    "self_sampling_augmentor_simplification_simple_shallow_with_solution": """Your task is to create a simplified version of the Main Question that retains the core logical problem but simplifies the shallow or non-core elements, using the provided Reference Solution as a guide.
<Instructions>
1. Replace large numbers or complex values with small, single-digit integers to make the math easier.
2. Rewrite complex scenarios or wordy descriptions into a plain, direct statement to remove distractions.
3. Reduce the number of steps or constraints slightly, as long as the underlying relationship (as shown in the solution) remains the same.
4. CRITICAL: If the question is already simple or cannot be simplified without breaking the core logic, or if the simplification results in a drastic change rather than a shallow adjustment, do NOT change it; just repeat the Main Question exactly.
</Instructions>
<Main Question>
{main_question_text}
</Main Question>
<Reference Solution>
{generated_solution}
</Reference Solution>
<Output Format>
Simplified Question:
[Your simplified version]
</Output Format>
""",

}

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


def create_merging_prompt(target_query: str, samples_to_merge: List[str]) -> str:
    if len(samples_to_merge) != 2:
        return "Error: create_merging_prompt requires exactly two samples."
    template = PROMPT_TEMPLATES["merging_v1"]
    return template.format(target_query=target_query, sample_1=samples_to_merge[0], sample_2=samples_to_merge[1])

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

def create_evaluation_prompt(model_answer: str, ground_truth: str, config: Dict[str, Any]) -> str:
    template_name = config.get("PROMPT_TEMPLATE_EVALUATOR", "evaluator_v1")
    template = PROMPT_TEMPLATES[template_name]
    return template.format(model_answer=model_answer, ground_truth=ground_truth)

def create_duplicate_check_prompt(main_question_text: str, retrieved_questions: List[str]) -> str:
    template = PROMPT_TEMPLATES["duplicate_question_check_v1"]
    retrieved_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(retrieved_questions))
    return template.format(main_question_text=main_question_text, retrieved_questions_block=retrieved_block.strip())


def create_self_sampling_prompt(main_question: str, config: Dict[str, Any]) -> str:
    template_name = config.get("PROMPT_TEMPLATE_SELF_SAMPLING_GENERATOR", "self_sampling_generator")
    if template_name not in PROMPT_TEMPLATES:
        return f"Error: Prompt template '{template_name}' not found in registry."
    template = PROMPT_TEMPLATES[template_name]
    return template.format(main_question_text=main_question)


def create_augmentation_prompt(main_question: str, n_samples: int, config: Dict[str, Any]) -> str:
    template_name = config.get("PROMPT_TEMPLATE_SELF_SAMPLING_AUGMENTOR", "self_sampling_augmentor_v1")
    if template_name not in PROMPT_TEMPLATES:
        return f"Error: Prompt template '{template_name}' not found in registry."
    template = PROMPT_TEMPLATES[template_name]
    return template.format(main_question_text=main_question, n_samples=n_samples)


def create_analogical_adaptation_prompt(main_question: str, sample_group: List[str], config: Dict[str, Any]) -> str:
    template_name = config.get("PROMPT_TEMPLATE_ANALOGICAL_ADAPTATION", "analogical_adaptation_v1")
    if template_name not in PROMPT_TEMPLATES:
        return f"Error: Prompt template '{template_name}' not found in registry."
    template = PROMPT_TEMPLATES[template_name]
    samples_block = "\n\n".join([f"<Sample {i+1}>\n{s}\n</Sample {i+1}>" for i, s in enumerate(sample_group)])
    return template.format(
        main_question_text=main_question, 
        samples_block=samples_block,
        examples_block=samples_block
    )

def create_hierarchical_parent_solver_prompt(main_question: str, child_nodes_data: List[Dict[str, str]], config: Dict[str, Any]) -> str:
    template_name = config.get("PROMPT_TEMPLATE_HIERARCHICAL_PARENT_SOLVER", "hierarchical_parent_solver_v1")
    if template_name not in PROMPT_TEMPLATES:
        return f"Error: Prompt template '{template_name}' not found in registry."
    template = PROMPT_TEMPLATES[template_name]
    child_block = ""
    for i, child in enumerate(child_nodes_data):
        child_block += f"<Variation {i+1}>\nQuestion: {child.get('question', '')}\nSolution: {child.get('solution', '')}\n</Variation {i+1}>\n\n"
    return template.format(main_question_text=main_question, child_solutions_block=child_block.strip())

def create_reverse_validation_prompt(validator_question: str, candidate_text: str, config: Dict[str, Any]) -> str:
    template_name = config.get("PROMPT_TEMPLATE_REVERSE_VALIDATION_SOLVER", "reverse_validation_v1")
    if template_name == "analogical_adaptation_v1" or template_name == "analogical_adaptation_v2":
        if template_name not in PROMPT_TEMPLATES:
            return f"Error: Template {template_name} not found."
        samples_block = f"<Sample>\n{candidate_text}\n</Sample>"
        template = PROMPT_TEMPLATES[template_name]
        return template.format(
            main_question_text=validator_question,
            samples_block=samples_block,
            examples_block=samples_block
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

def create_augmentation_with_solution_prompt(main_question: str, generated_solution: str, n_samples: int, config: Dict[str, Any]) -> str:
    template_name = config.get("PROMPT_TEMPLATE_AUGMENTATION_STEP2_GENERATOR", "self_sampling_augmentor_simplification_with_solution")
    if template_name not in PROMPT_TEMPLATES:
        return f"Error: Template {template_name} not found."
    template = PROMPT_TEMPLATES[template_name]
    return template.format(main_question_text=main_question, generated_solution=generated_solution, n_samples=n_samples)