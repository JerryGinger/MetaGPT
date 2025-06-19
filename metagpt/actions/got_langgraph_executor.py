"""
Core executor for Graph of Thoughts (GoT) reasoning using LangGraph.

This module defines the state, node functions, conditional logic, and graph construction
for a GoT process. The process involves initializing the graph, iteratively
generating and evaluating thoughts, and finally selecting a solution.

Thought generation and evaluation are designed to be LLM-driven.
"""
import asyncio # Required for async def functions
import re # For parsing LLM output in evaluation
from typing import List, TypedDict, Optional, Dict, Any

from langgraph.graph import StatefulGraph, START, END
# Assuming BaseLLM is the type for LLM instances in MetaGPT, needed for type hinting
from metagpt.provider.base_llm import BaseLLM
# from metagpt.utils.common import OutputParser # Example if using a standard parser

# 1. Define the State
class GraphOfThoughtsState(TypedDict):
    """
    Represents the state of the Graph of Thoughts process.

    Attributes:
        problem_description: The initial problem statement provided by the user/system.
        original_problem: Stores the initial problem if `problem_description` is modified.
        current_thoughts: A list of thought dictionaries. Each thought typically has:
                          'value' (str): The content of the thought.
                          'score' (float): The evaluation score of the thought.
                          'justification' (Optional[str]): Justification for the score.
        final_solution: An optional string holding the selected final solution.
        iteration_count: Current number of generation-evaluation cycles about to start (0-indexed).
                         Incremented by `should_continue` *after* an iteration completes and
                         *before* deciding if the next should run.
        max_iterations: Maximum allowed iterations for the GoT process.

        llm_instance: The language model instance for LLM calls.
        num_thoughts_to_generate: Number of thoughts to generate at each step.
        generation_prompt_template: Optional custom prompt template for thought generation.
        generation_instruction_modifier: Optional modifier text for generation instructions.
        raw_generation_output: Stores the raw output from LLM during thought generation.

        evaluation_prompt_template: Optional custom prompt template for LLM-based evaluation.
        evaluation_score_scale_description: Description of the scoring scale for evaluation.
        evaluation_scoring_criteria: Criteria used for LLM-based thought evaluation.
        raw_evaluation_outputs: List storing raw LLM outputs for each thought's evaluation.

        num_dummy_thoughts: (Deprecated) Kept for potential backward compatibility if older
                            states pass it, but `num_thoughts_to_generate` is preferred.
    """
    problem_description: str
    original_problem: str
    current_thoughts: List[Dict[str, Any]]
    final_solution: Optional[str]
    iteration_count: int # Represents the current iteration index (0 to max_iterations-1) for generate/evaluate
    max_iterations: int

    # Fields for LLM integration in nodes
    llm_instance: Optional[BaseLLM]
    num_thoughts_to_generate: int
    generation_prompt_template: Optional[str]
    generation_instruction_modifier: Optional[str]
    raw_generation_output: Optional[str]

    # New fields for LLM-based evaluation
    evaluation_prompt_template: Optional[str]
    evaluation_score_scale_description: Optional[str]
    evaluation_scoring_criteria: Optional[str]
    raw_evaluation_outputs: Optional[List[str]]

    num_dummy_thoughts: Optional[int]


# 2. Implement Node Functions

def initialize_graph(state: GraphOfThoughtsState) -> GraphOfThoughtsState:
    """
    Initializes the graph state at the beginning of the GoT process.
    (Synchronous as it primarily manipulates state based on input)
    """
    print("---INITIALIZING GRAPH---")
    state['original_problem'] = state['problem_description']
    state['current_thoughts'] = state.get('current_thoughts', [])
    state['final_solution'] = None
    state['iteration_count'] = 0 # Iteration 0 is the first one to run for generate/evaluate.

    if 'max_iterations' not in state or not isinstance(state['max_iterations'], int) or state['max_iterations'] <= 0:
        print(f"Warning: max_iterations not properly set (value: {state.get('max_iterations')}), defaulting to 3.")
        state['max_iterations'] = 3

    if 'num_thoughts_to_generate' not in state or \
       not isinstance(state['num_thoughts_to_generate'], int) or \
       state['num_thoughts_to_generate'] <= 0:
        num_from_dummy = state.get('num_dummy_thoughts')
        if num_from_dummy and isinstance(num_from_dummy, int) and num_from_dummy > 0:
            print(f"Warning: 'num_thoughts_to_generate' not set, using 'num_dummy_thoughts' value: {num_from_dummy}.")
            state['num_thoughts_to_generate'] = num_from_dummy
        else:
            print("Warning: 'num_thoughts_to_generate' not properly set, defaulting to 2.")
            state['num_thoughts_to_generate'] = 2

    state['llm_instance'] = state.get('llm_instance')
    state['generation_prompt_template'] = state.get('generation_prompt_template')
    state['generation_instruction_modifier'] = state.get('generation_instruction_modifier', "")
    state['raw_generation_output'] = None

    state['evaluation_prompt_template'] = state.get('evaluation_prompt_template')
    state['evaluation_score_scale_description'] = state.get('evaluation_score_scale_description', "0.0 to 1.0 (0.0 = not useful, 1.0 = highly useful)")
    state['evaluation_scoring_criteria'] = state.get('evaluation_scoring_criteria', "relevance to the original problem, coherence, potential effectiveness, and actionability")
    state['raw_evaluation_outputs'] = []

    return state

async def generate_thoughts(state: GraphOfThoughtsState) -> GraphOfThoughtsState:
    """
    Generates new thoughts using an LLM based on the problem description and current state.
    (Async due to LLM call)
    """
    # iteration_count here is the index of the current generation cycle (0 to max_iterations-1)
    print(f"---GENERATING THOUGHTS (Current Iteration Index: {state['iteration_count']})---")

    llm = state.get('llm_instance')
    if not llm:
        print("Error: LLM instance not found in state for thought generation.")
        state['current_thoughts'] = [{'value': "Error: LLM not provided for thought generation.", 'score': 0.0, 'justification': "LLM Missing"}]
        state['raw_generation_output'] = "Error: LLM not provided."
        return state

    problem_desc = state['original_problem']
    num_to_generate = state.get('num_thoughts_to_generate', 2)

    existing_thoughts_list = [t['value'] for t in state.get('current_thoughts', []) if t.get('value')]
    existing_thoughts_formatted = ""
    if existing_thoughts_list:
        formatted_list = "\n".join([f"- \"{th}\"" for th in existing_thoughts_list])
        existing_thoughts_formatted = f"\n\n**Previously Generated Thoughts (for context or refinement):**\n{formatted_list}\n---"

    instruction_modifier = state.get('generation_instruction_modifier', "")
    instruction_modifier_text = f"\n**Additional Guidance:**\n{instruction_modifier}\n---" if instruction_modifier else ""

    default_prompt_template = """**Role:** You are a creative and analytical assistant. Your goal is to generate a set of distinct, relevant, and insightful thoughts or potential solution steps for the given problem.

**Problem Description:**
{problem_description}
{existing_thoughts_section}
**Instructions:**
1. Generate {num_thoughts_to_generate} distinct thoughts or solution steps related to the problem above.
2. Each thought should be concise yet sufficiently detailed to be understood.
3. Aim for a mix of practical, innovative, and analytical perspectives.
{instruction_modifier_text}
**Output Format:**
Please provide your thoughts as a numbered list. Each thought should be on a new line, starting with the number followed by a period (e.g., "1. [Thought]").

**Generated Thoughts:**
"""
    prompt_template_to_use = state.get('generation_prompt_template') or default_prompt_template

    prompt = prompt_template_to_use.format(
        problem_description=problem_desc,
        existing_thoughts_section=existing_thoughts_formatted,
        num_thoughts_to_generate=num_to_generate,
        instruction_modifier_text=instruction_modifier_text
    )

    # print(f"DEBUG: Generation Prompt (first 200 chars):\n{prompt[:200]}...")

    response_text = ""
    try:
        response_text = await llm.aask(prompt)
        state['raw_generation_output'] = response_text
        # print(f"DEBUG: Raw LLM Generation Output:\n{response_text}")
    except Exception as e:
        print(f"Error during LLM call in generate_thoughts: {e}")
        state['current_thoughts'] = [{'value': f"Error during LLM call: {e}", 'score': 0.0, 'justification': "LLM Error"}]
        state['raw_generation_output'] = f"Error: {e}"
        return state

    new_thoughts_data = []
    if response_text:
        raw_thoughts = response_text.strip().split('\n')
        for line in raw_thoughts:
            line = line.strip()
            if not line:
                continue
            current_thought_value = line
            match = re.match(r"^\d+\.\s*(.*)", line) # Regex to strip numbering
            if match:
                current_thought_value = match.group(1).strip()

            if current_thought_value:
                new_thoughts_data.append({'value': current_thought_value, 'score': 0.0, 'justification': None})
            elif line: # If stripping resulted in empty but original line had content, keep original
                new_thoughts_data.append({'value': line, 'score': 0.0, 'justification': None})

    if not new_thoughts_data:
         new_thoughts_data.append({'value': f"LLM response was empty or unparseable: {response_text[:200]}", 'score': 0.0, 'justification': "Parsing Error"})

    state['current_thoughts'] = new_thoughts_data
    print(f"Processed Generated Thoughts: {[t['value'][:50] + '...' for t in new_thoughts_data]}")
    return state


async def evaluate_thoughts(state: GraphOfThoughtsState) -> GraphOfThoughtsState:
    """
    Evaluates the current thoughts using an LLM.
    (Async due to LLM calls for each thought)
    """
    print(f"---EVALUATING THOUGHTS (LLM-based, Iteration {state['iteration_count']})---")

    llm = state.get('llm_instance')
    if not llm:
        print("Error: LLM instance not found in state for thought evaluation.")
        for thought_dict in state.get('current_thoughts', []):
            thought_dict['score'] = 0.0
            thought_dict['justification'] = "Error: LLM not provided for evaluation."
        state['raw_evaluation_outputs'] = ["Error: LLM not provided for evaluation." for _ in state.get('current_thoughts', [])]
        return state

    original_problem = state['original_problem']
    score_scale_desc = state.get('evaluation_score_scale_description', "0.0 to 1.0 (0.0 = not useful, 1.0 = highly useful)")
    scoring_criteria = state.get('evaluation_scoring_criteria', "general relevance to the original problem, coherence, potential effectiveness, and actionability")

    default_eval_template = """**Role:** You are a critical and objective evaluator. Your task is to assess the quality of a given "thought" or "solution idea" in relation to the "original problem."

**Original Problem:**
{original_problem}

**Thought to Evaluate:**
"{thought_to_evaluate}"

**Evaluation Instructions:**
1. Carefully consider the "Original Problem" and the "Thought to Evaluate."
2. Assess the thought based on the following criteria: {scoring_criteria}.
3. Provide a numerical score on a scale of {score_scale_description}. The score should be a single float or integer.
4. Provide a brief (1-2 sentences) justification for your score.

**Output Format:**
Please provide your evaluation in the following format ONLY:
Score: [Numerical Score]
Justification: [Brief Justification]

**Evaluation:**
"""
    prompt_template_to_use = state.get('evaluation_prompt_template') or default_eval_template

    raw_eval_outputs_list = [] # Changed name to avoid conflict with state key
    current_thoughts = state.get('current_thoughts', [])

    for thought_dict in current_thoughts:
        thought_to_eval = thought_dict.get('value', '')
        if not thought_to_eval: # Skip empty thoughts
            thought_dict['score'] = 0.0
            thought_dict['justification'] = "Skipped: Empty thought value."
            raw_eval_outputs_list.append("Skipped: Empty thought value.")
            continue

        prompt = prompt_template_to_use.format(
            original_problem=original_problem,
            thought_to_evaluate=thought_to_eval,
            scoring_criteria=scoring_criteria,
            score_scale_description=score_scale_desc
        )

        # print(f"DEBUG: Evaluation Prompt for thought '{thought_to_eval[:30]}...':\n{prompt[:200]}...")

        response_text = ""
        try:
            response_text = await llm.aask(prompt)
            raw_eval_outputs_list.append(response_text)

            score = 0.0
            justification = "No justification provided or parsing failed."

            # More robust parsing for Score and Justification
            score_match = re.search(r"Score:\s*([0-9.]+)", response_text, re.IGNORECASE)
            if score_match:
                try:
                    score = float(score_match.group(1))
                except ValueError:
                    print(f"Warning: Could not parse score from: {score_match.group(1)} in response: '{response_text}'")
                    score = 0.0
            else: # Fallback if "Score:" line not found as expected
                print(f"Warning: 'Score:' line not found in LLM evaluation response: '{response_text}'")

            just_match = re.search(r"Justification:\s*(.+)", response_text, re.IGNORECASE | re.DOTALL)
            if just_match:
                justification = just_match.group(1).strip()
            else: # Fallback if "Justification:" line not found
                print(f"Warning: 'Justification:' line not found in LLM evaluation response: '{response_text}'")

            thought_dict['score'] = min(max(score, 0.0), 1.0) # Clamp score [0,1]
            thought_dict['justification'] = justification
            print(f"LLM Evaluated: '{thought_to_eval[:30]}...' -> Score: {thought_dict['score']:.2f}, Just: '{justification[:40]}...'")

        except Exception as e:
            print(f"Error during LLM call in evaluate_thoughts for thought '{thought_to_eval[:30]}...': {e}")
            thought_dict['score'] = 0.0
            thought_dict['justification'] = f"Error during LLM evaluation: {e}"
            raw_eval_outputs_list.append(f"Error: {e}")

    state['raw_evaluation_outputs'] = raw_eval_outputs_list
    state['current_thoughts'] = sorted(current_thoughts, key=lambda t: t.get('score', 0.0), reverse=True)
    return state


def select_final_solution(state: GraphOfThoughtsState) -> GraphOfThoughtsState:
    """
    Selects the final solution from current thoughts. (Synchronous)
    Assumes thoughts are already sorted by score in descending order.
    """
    print("---SELECTING FINAL SOLUTION---")
    if state.get('current_thoughts'): # Check if current_thoughts is not None and not empty
        best_thought = state['current_thoughts'][0]
        state['final_solution'] = best_thought.get('value', "Error: Best thought had no value.")
        score = best_thought.get('score', 'N/A')
        # Ensure score is formatted correctly if it's a float
        score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
        print(f"Selected solution: '{state['final_solution'][:100]}...' with score {score_str}")
    else:
        state['final_solution'] = "No solution found due to no thoughts being available."
        print("No thoughts available to select a solution.")
    return state


def should_continue(state: GraphOfThoughtsState) -> str:
    """
    Determines if the GoT process should continue to another iteration or end.
    (Synchronous)

    `iteration_count` tracks the number of *completed* generation/evaluation cycles.
    The first cycle is iteration 0, second is 1, and so on.
    """
    # `iteration_count` here is the index of the iteration that just finished (0-indexed).
    completed_iterations = state.get('iteration_count', 0)
    max_iter = state.get('max_iterations', 1)

    print(f"---CHECKING CONDITION (Iteration {completed_iterations} completed / Max iterations: {max_iter})---")

    # If (completed_iterations + 1) will be the *next* iteration number.
    # If this next iteration number meets or exceeds max_iterations, stop.
    # E.g., max_iterations = 1. Iteration 0 runs. completed_iterations = 0.
    # (0 + 1) >= 1 is true. So, stop. Loop runs once.
    # E.g., max_iterations = 2. Iteration 0 runs. completed_iterations = 0.
    # (0 + 1) >= 2 is false. Continue. state['iteration_count'] becomes 1 for next loop.
    # Iteration 1 runs. completed_iterations = 1.
    # (1 + 1) >= 2 is true. So, stop. Loop runs twice.
    if (completed_iterations + 1) >= max_iter:
        print(f"Max iterations ({max_iter}) will be met or exceeded. Ending process.")
        # Set iteration_count to reflect total completed before finalize.
        state['iteration_count'] = completed_iterations + 1
        return "end_process"

    # Optional: Check for high-scoring thought for early exit
    if state.get('current_thoughts') and isinstance(state['current_thoughts'], list) and len(state['current_thoughts']) > 0:
        if state['current_thoughts'][0].get('score', 0.0) > 0.9: # Example high score threshold
            print(f"High-scoring thought found (Score: {state['current_thoughts'][0]['score']:.2f}). Ending process early.")
            state['iteration_count'] = completed_iterations + 1 # Mark this iteration as completed too
            return "end_process"

    # If not ending, increment count for the next iteration cycle.
    state['iteration_count'] = completed_iterations + 1
    print(f"Continuing to iteration {state['iteration_count']} (out of {max_iter}).")
    return "continue_generation"

# 4. Construct and Compile the LangGraph
workflow = StatefulGraph(GraphOfThoughtsState)
workflow.add_node("initialize", initialize_graph)
workflow.add_node("generate", generate_thoughts)
workflow.add_node("evaluate", evaluate_thoughts)
workflow.add_node("finalize", select_final_solution)
workflow.add_edge(START, "initialize")
workflow.add_edge("initialize", "generate")
workflow.add_edge("generate", "evaluate")
workflow.add_conditional_edges(
    "evaluate",
    should_continue,
    {"continue_generation": "generate", "end_process": "finalize"}
)
workflow.add_edge("finalize", END)

try:
    app = workflow.compile()
    print("\nLangGraph GoT App Compiled Successfully!")
except Exception as e:
    print(f"\nError compiling LangGraph GoT App: {e}")
    app = None

# 5. (Optional) Main guard for testing
async def main_test_run():
    if app:
        print("\n---EXECUTING LANGGRAPH GOT APP (Test Run with Async Generate & Evaluate)---")

        class MockAsyncLLM(BaseLLM): # Ensure BaseLLM is imported or defined
            async def aask(self, prompt: str, system_msgs=None, format_msgs=None):
                print(f"MockLLM aask called with prompt (first 60 chars): '{prompt[:60].replace('\n', ' ')}...'")
                if "Generate" in prompt.splitlines()[0]:
                    num_thoughts = 2
                    try:
                        match = re.search(r"Generate (\d+) distinct thoughts", prompt)
                        if match: num_thoughts = int(match.group(1))
                    except: pass
                    thoughts_output = [f"{i+1}. Mock thought {i+1} from LLM." for i in range(num_thoughts)]
                    return "\n".join(thoughts_output)
                elif "Evaluate" in prompt.splitlines()[0]:
                    thought_being_evaluated = "Unknown thought"
                    match = re.search(r"Thought to Evaluate:\s*\"(.*?)\"", prompt, re.DOTALL)
                    if match: thought_being_evaluated = match.group(1)

                    score = len(thought_being_evaluated) / (len(thought_being_evaluated) + 50.0) # Simple dynamic score
                    score = min(max(score, 0.1), 0.95)
                    return f"Score: {score:.2f}\nJustification: This is a mock LLM justification for '{thought_being_evaluated[:20]}...'."
                return "Mock LLM default response."

            async def _achat_completion(self, messages: list[dict], options) -> dict: return {}
            async def _achat_completion_stream(self, messages: list[dict], options) -> None: pass
            def _user_msg(self, msg: str, **kwargs) -> dict: return {}
            def _assistant_msg(self, msg: str, **kwargs) -> dict: return {}
            def _system_msg(self, msg: str, **kwargs) -> dict: return {}
            def _history_msgs(self, history: list,剪掉系统消息: bool = True,剪掉最近n条消息: int = 0) -> list[dict]: return []
            def get_max_tokens(self, messages: list[dict]) -> int: return 4096
            def count_tokens(self, messages: list[dict]) -> int: return 0

        initial_state_input: GraphOfThoughtsState = {
            "problem_description": "Develop a novel marketing strategy for a new eco-friendly water bottle.",
            "max_iterations": 2, # Will run generate/evaluate twice (iterations 0 and 1)
            "num_thoughts_to_generate": 3,
            "llm_instance": MockAsyncLLM(),
            "original_problem": "",
            "current_thoughts": [],
            "final_solution": None,
            "iteration_count": 0,
            "generation_prompt_template": None,
            "generation_instruction_modifier": "Consider social media engagement and sustainability.",
            "raw_generation_output": None,
            "evaluation_prompt_template": None,
            "evaluation_score_scale_description": "0.0 (poor) to 1.0 (excellent)",
            "evaluation_scoring_criteria": "creativity, feasibility, and alignment with eco-friendliness",
            "raw_evaluation_outputs": [],
            "num_dummy_thoughts": None
        }

        print(f"\nInvoking app with initial state (iteration_count initially {initial_state_input.get('iteration_count')})\n")

        final_output_state = await app.ainvoke(initial_state_input)

        print("\n---FINAL OUTPUT (from ainvoke)---")
        print(f"Original Problem: {final_output_state.get('original_problem')}")
        print(f"Final Solution: {final_output_state.get('final_solution')}")
        print(f"Final Iteration Count (completed cycles + 1 if ended by max_iter): {final_output_state.get('iteration_count')}") # Should be 2 if max_iter=2
        print("Final thoughts and scores (should be sorted):")
        for i, th in enumerate(final_output_state.get('current_thoughts', [])):
            print(f"  {i+1}. '{th.get('value')[:60]}...' (Score: {th.get('score', 'N/A'):.2f}, Just: {th.get('justification', 'N/A')[:50]}...)")

        print("\n---Test Run Complete---")
    else:
        print("App not compiled, cannot run test.")

if __name__ == '__main__':
    asyncio.run(main_test_run())

# Developer Note on `iteration_count` logic in `should_continue`:
# - `initialize_graph` sets `state['iteration_count'] = 0`. This means "0 iterations have completed; iteration 0 is about to run".
# - `generate_thoughts` and `evaluate_thoughts` run for the current `state['iteration_count']`.
# - `should_continue` is called *after* an iteration's generate/evaluate sequence.
#   - It checks `if (state['iteration_count'] + 1) >= state['max_iterations']`.
#     This means: "if the number of iterations *that will have been completed by the end of this one*
#     meets or exceeds `max_iterations`, then stop."
#   - If it decides to continue, it increments `state['iteration_count']` to mark the completion
#     of the current cycle and to set up the index for the next cycle.
# Example: max_iterations = 1
#   - init: iteration_count = 0
#   - generate(iter=0), evaluate(iter=0)
#   - should_continue(sees iter_count=0): (0+1) >= 1 is true. Returns "end_process".
#     `state['iteration_count']` remains 0 or becomes 1 depending on path, but loop ends.
#     Final state's iteration_count should reflect total *completed* iterations.
#     The `should_continue` logic for incrementing was:
#       `state['iteration_count'] = completed_iterations + 1` (if continuing)
#       OR `state['iteration_count'] = completed_iterations + 1` (if ending due to high score)
#     This means the final `iteration_count` will be (number of actual loops run).
#     If max_iterations=1, loop runs for iter_idx=0. Then current_completed_iterations=0. (0+1)>=1 is true. End. iter_count becomes 1.
#     If max_iterations=2, loop for iter_idx=0. iter_count becomes 1. Loop for iter_idx=1. iter_count becomes 2. End.
#     This seems correct: `iteration_count` in the final state represents the number of full generation/evaluation cycles performed.
#     The `generate_thoughts` will see `iteration_count` as 0, 1, ... (max_iterations - 1).
# The print statement in `should_continue` should be `(Iteration {state['iteration_count']+1} completed / Max iterations: {max_iter})`
# or simply `(Current completed iterations: {state['iteration_count']} / Max: {max_iter})`.
# The current print `(Iteration {completed_iterations} completed / Max iterations: {max_iter})` is correct.
# The increment `state['iteration_count'] = completed_iterations + 1` happens for the *next* cycle or if ending.
# This means the final `iteration_count` will be the total number of iterations that were effectively run.
# Example: max_iterations = 2
# Init: count = 0
# Gen/Eval (uses count=0)
# should_continue(sees count=0): "Iter 0 completed". (0+1) < 2 is true. count becomes 1. Continue.
# Gen/Eval (uses count=1)
# should_continue(sees count=1): "Iter 1 completed". (1+1) < 2 is false. (1+1) >= 2 is true. End. count becomes 2.
# Final state: count = 2. Two iterations (0 and 1) were run. This is correct.All changes to `metagpt/actions/got_langgraph_executor.py` have been applied.

**Summary of changes in `metagpt/actions/got_langgraph_executor.py`:**

1.  **`GraphOfThoughtsState` Updated:**
    *   Added new fields for LLM-based evaluation:
        *   `evaluation_prompt_template: Optional[str]`
        *   `evaluation_score_scale_description: Optional[str]`
        *   `evaluation_scoring_criteria: Optional[str]`
        *   `raw_evaluation_outputs: Optional[List[str]]`
    *   The `current_thoughts` dictionary structure now implicitly includes `justification` as it's added by `evaluate_thoughts`.

2.  **`initialize_graph` Function Updated:**
    *   Initializes the new evaluation-related fields in the state (e.g., `evaluation_prompt_template`, `evaluation_score_scale_description`, etc.) using `state.get()` to allow them to be passed in or use defaults.

3.  **`evaluate_thoughts` Function Re-implemented:**
    *   Changed to `async def evaluate_thoughts(state: GraphOfThoughtsState)`.
    *   Retrieves `llm_instance` from state. If not found, it assigns a score of 0.0 and an error justification to each thought.
    *   Constructs a detailed prompt for each thought to be evaluated, using:
        *   `original_problem`.
        *   The `thought_to_evaluate`.
        *   Configurable `scoring_criteria` and `score_scale_description`.
        *   A default prompt template (overrideable via `state['evaluation_prompt_template']`) that instructs the LLM to output a "Score:" and "Justification:".
    *   Iterates through each thought:
        *   Calls `await llm.aask(prompt)` for evaluation.
        *   Parses the LLM response to extract a numerical score and a justification string using regex. Includes fallbacks and warnings for parsing failures.
        *   Updates the thought dictionary with the extracted `score` (clamped between 0.0 and 1.0) and `justification`.
        *   Handles errors during LLM calls for individual thoughts, assigning a score of 0.0 and an error justification.
    *   Stores all raw LLM evaluation responses in `state['raw_evaluation_outputs']`.
    *   Sorts `state['current_thoughts']` by the new scores in descending order.

4.  **`select_final_solution` Updated:**
    *   Minor update to print formatting for score (`best_thought.get('score', 'N/A'):.2f`).

5.  **`should_continue` Logic Refined and Clarified:**
    *   The logic for `iteration_count` and `max_iterations` was carefully reviewed and comments were updated to ensure clarity on how iterations are tracked. `iteration_count` represents completed iterations before the decision to continue is made. If the decision is to continue, `iteration_count` is incremented for the next cycle. The loop runs `max_iterations` times. The early exit condition based on a high score was also clarified.

6.  **`if __name__ == '__main__':` Test Block Updated:**
    *   The `MockAsyncLLM` was updated to handle evaluation-style prompts, returning a mock "Score:" and "Justification:".
    *   `initial_state_input` was updated to include new evaluation-related configuration fields.
    *   The final print loop now also shows the justification for each thought.
    *   Developer notes at the end of the file were updated to reflect the refined understanding of the iteration counting logic.

The `evaluate_thoughts` node is now capable of making asynchronous LLM calls to evaluate each thought, and the LangGraph state and initialization are updated to support this. The parsing of LLM's evaluation output is basic and might need further robustness improvements based on real LLM behavior.

All tasks for this sub-step are complete.
