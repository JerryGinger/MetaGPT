"""
MetaGPT Action for performing Graph of Thoughts (GoT) reasoning.

This module defines `GoTReasoningAction`, a MetaGPT Action that leverages a
LangGraph-based executor (`metagpt.actions.got_langgraph_executor.app`)
to implement a Graph of Thoughts reasoning process. The action is responsible for
initializing the GoT state with the problem description, LLM instance, and various
operational configurations, then invoking the asynchronous LangGraph application.
It processes the final state from the graph to return a solution.

The behavior of the GoT process, including the number of iterations, thoughts generated
per step, and prompt templates, can be customized through parameters passed to the
`run` method or via the Action's `self.config` object (typically populated from a
Role's YAML configuration).
"""
import asyncio
from typing import Dict, Any, Optional, List

from metagpt.actions.action import Action
from metagpt.actions.got_langgraph_executor import app as got_langgraph_app
from metagpt.actions.got_langgraph_executor import GraphOfThoughtsState
from metagpt.schema import Message
from metagpt.provider.base_llm import BaseLLM
from metagpt.logs import logger
from metagpt.config2 import Config # For self.config type hinting & default

class GoTReasoningAction(Action):
    """
    A MetaGPT Action that orchestrates a Graph of Thoughts (GoT) reasoning process
    using a pre-compiled LangGraph application (`got_langgraph_app`).

    This action initializes the GoT state, including the problem description,
    the LLM instance to be used by graph nodes, and various configurations for
    thought generation and evaluation. It then invokes the asynchronous LangGraph
    executor (`app.ainvoke`) and returns the final solution derived from the GoT process.

    Attributes:
        name (str): The name of the action, defaults to "GoTReasoningAction".
        llm (Optional[BaseLLM]): The LLM instance associated with this action. This is
                                 expected to be populated by the MetaGPT framework when
                                 the Action is initialized (e.g., within a Role).
        config (Optional[Config]): Action-specific configuration object, typically
                                   populated from a Role's YAML configuration. This allows
                                   customization of the GoT process.

    Expected `self.config` keys for GoT customization (can also be overridden by `run` parameters):
    - `got_max_iterations` (int): Max iterations for the GoT loop (default: 3).
    - `got_num_thoughts_to_generate` (int): Number of thoughts to generate per step (default: 3).
    - `got_generation_prompt_template` (str, optional): Custom prompt template for thought generation.
      Uses placeholders like `{problem_description}`, `{existing_thoughts_section}`, etc.
    - `got_generation_instruction_modifier` (str, optional): Additional instructions for the generation prompt.
    - `got_evaluation_prompt_template` (str, optional): Custom prompt template for thought evaluation.
      Uses placeholders like `{original_problem}`, `{thought_to_evaluate}`, etc.
    - `got_evaluation_score_scale_description` (str, optional): Description of the score scale for evaluation.
    - `got_evaluation_scoring_criteria` (str, optional): Specific criteria for evaluation.

    Note: `got_generation_llm_config` and `got_evaluation_llm_config` are placeholders
    for future fine-grained LLM control per node type, but currently the single `self.llm`
    instance is used for all LLM calls within the graph.
    """
    name: str = "GoTReasoningAction"
    # self.llm and self.config are populated by the base Action class or Role context

    async def run(self,
                  problem_description: str,
                  max_iterations: Optional[int] = None,
                  num_thoughts_to_generate: Optional[int] = None,
                  initial_thoughts: Optional[List[Dict[str, Any]]] = None,
                  generation_prompt_template: Optional[str] = None,
                  generation_instruction_modifier: Optional[str] = None,
                  evaluation_prompt_template: Optional[str] = None,
                  evaluation_score_scale_description: Optional[str] = None,
                  evaluation_scoring_criteria: Optional[str] = None,
                  # generation_llm_config and evaluation_llm_config are placeholders
                  # for potential future use if nodes need distinct LLM settings beyond the main llm_instance.
                  # Currently, the main self.llm is passed and used by all nodes.
                  generation_llm_config: Optional[Dict[str, Any]] = None, # Not actively used by executor nodes yet
                  evaluation_llm_config: Optional[Dict[str, Any]] = None   # Not actively used by executor nodes yet
                 ) -> str:
        """
        Runs the Graph of Thoughts reasoning process using the compiled LangGraph executor.

        This method prepares the initial state for the GoT process, including LLM instance
        and configurations, then invokes the asynchronous LangGraph application (`app.ainvoke`).
        Parameters passed directly to this method override any corresponding values
        set in `self.config` (which are typically loaded from a Role's YAML configuration).

        Args:
            problem_description: The problem statement to be solved.
            max_iterations: Max generation-evaluation cycles. Overrides `self.config.got_max_iterations`.
            num_thoughts_to_generate: Number of thoughts per step. Overrides `self.config.got_num_thoughts_to_generate`.
            initial_thoughts: Optional list of initial thoughts to seed the process.
            generation_prompt_template: Custom prompt for generation. Overrides `self.config.got_generation_prompt_template`.
            generation_instruction_modifier: Modifier for generation instructions. Overrides `self.config.got_generation_instruction_modifier`.
            evaluation_prompt_template: Custom prompt for evaluation. Overrides `self.config.got_evaluation_prompt_template`.
            evaluation_score_scale_description: Score scale desc. Overrides `self.config.got_evaluation_score_scale_description`.
            evaluation_scoring_criteria: Criteria for evaluation. Overrides `self.config.got_evaluation_scoring_criteria`.
            generation_llm_config: Specific LLM config for generation (currently placeholder).
            evaluation_llm_config: Specific LLM config for evaluation (currently placeholder).

        Returns:
            A string representing the final solution from the GoT process,
            or an error message if it fails or no solution is determined.
        """
        if not self.llm:
            logger.error(f"{self.name}: LLM not configured for this action. Ensure an LLM is assigned to the action or its role.")
            return "Error: LLM not configured for GoTReasoningAction."

        # Use a Config object for self.config if it's not already one (e.g. if it's a dict from direct init)
        # In normal MetaGPT flow, self.config is usually a ConfigManager instance.
        cfg = self.config
        if not isinstance(cfg, Config) and isinstance(cfg, dict):
            cfg = Config(config_dict=cfg) # Wrap dict in Config for consistent .get
        elif not cfg: # If self.config is None
            cfg = Config.default()


        # Determine effective configuration values: direct params > self.config > hardcoded defaults
        effective_max_iterations = max_iterations if max_iterations is not None else cfg.get('got_max_iterations', 3)
        effective_num_thoughts = num_thoughts_to_generate if num_thoughts_to_generate is not None else cfg.get('got_num_thoughts_to_generate', 3)

        # Prepare the initial state payload for the LangGraph GoT executor.
        # All fields defined in GraphOfThoughtsState must be present.
        initial_state_payload: GraphOfThoughtsState = {
            "problem_description": problem_description,
            "original_problem": problem_description,
            "current_thoughts": initial_thoughts if initial_thoughts is not None else [],
            "final_solution": None,
            "iteration_count": 0,
            "max_iterations": effective_max_iterations,

            "llm_instance": self.llm, # Pass the action's configured LLM instance

            "num_thoughts_to_generate": effective_num_thoughts,
            "generation_prompt_template": generation_prompt_template or cfg.get('got_generation_prompt_template'),
            "generation_instruction_modifier": generation_instruction_modifier or cfg.get('got_generation_instruction_modifier'),

            "evaluation_prompt_template": evaluation_prompt_template or cfg.get('got_evaluation_prompt_template'),
            "evaluation_score_scale_description": evaluation_score_scale_description or cfg.get('got_evaluation_score_scale_description'),
            "evaluation_scoring_criteria": evaluation_scoring_criteria or cfg.get('got_evaluation_scoring_criteria'),

            # raw_outputs are initialized by the graph nodes themselves or can be None initially
            "raw_generation_output": None,
            "raw_evaluation_outputs": [], # Initialize as empty list

            # num_dummy_thoughts is deprecated but part of GraphOfThoughtsState for type completeness.
            # It's not actively used if num_thoughts_to_generate is set.
            "num_dummy_thoughts": 0
        }

        logger.info(f"{self.name}: Invoking GoT LangGraph with effective state: "
                    f"problem='{problem_description[:50]}...', max_iter={effective_max_iterations}, "
                    f"num_thoughts_gen={effective_num_thoughts}")

        final_langgraph_state: Optional[GraphOfThoughtsState] = None

        try:
            # Use ainvoke as graph nodes (generate_thoughts, evaluate_thoughts) are async.
            final_langgraph_state = await got_langgraph_app.ainvoke(initial_state_payload)
        except Exception as e:
            logger.error(f"{self.name}: Error during LangGraph execution for problem '{problem_description[:50]}...': {e}", exc_info=True)
            return f"Error in GoT reasoning process: {str(e)}"

        if final_langgraph_state and final_langgraph_state.get('final_solution'):
            logger.info(f"{self.name}: GoT process completed. Solution found: '{str(final_langgraph_state['final_solution'])[:100]}...'")
            return str(final_langgraph_state['final_solution']) # Ensure it's a string
        else:
            logger.warning(f"{self.name}: GoT process completed for '{problem_description[:50]}...', "
                           f"but no final solution was determined. Final state: {final_langgraph_state}")
            return "GoT process completed, but no final solution was explicitly set."

async def main_test_action_directly():
    """
    Example function to demonstrate running GoTReasoningAction directly.
    This requires mocking or providing a real LLM and config.
    """
    from unittest.mock import MagicMock
    try:
        # Try to use a real LLM if API keys are configured
        llm_instance = LLM()
        logger.info("Using real LLM for test run of GoTReasoningAction.")
    except Exception as e:
        logger.warning(f"Failed to initialize real LLM (error: {e}), using MagicMock for LLM in test run.")
        llm_instance = MagicMock(spec=BaseLLM)
        async def mock_aask(prompt_text: str, system_msgs=None, format_msgs=None):
            logger.debug(f"Mock LLM received prompt: {prompt_text[:100]}...")
            if "Generate" in prompt_text: return "1. Mocked thought 1 from main_test\n2. Mocked thought 2 (solution) from main_test"
            if "Evaluate" in prompt_text: return "Score: 0.88\nJustification: Mocked good thought from main_test"
            return "Default mock response from main_test"
        llm_instance.aask = mock_aask
        llm_instance.config = MagicMock()

    action = GoTReasoningAction()
    action.llm = llm_instance

    # Simulate self.config as it would be in a Role context
    action_config_data = {
        "got_max_iterations": 2, # Default for this test config
        "got_num_thoughts_to_generate": 2,
        "got_generation_instruction_modifier": "Keep suggestions brief and actionable.",
        "got_evaluation_scoring_criteria": "clarity, actionability, and novelty within a budget"
    }
    action.config = Config(config_dict=action_config_data) # Use actual Config object

    problem1 = "Explain the concept of 'market saturation' to a new intern."
    logger.info(f"Test 1: Running GoTReasoningAction for: '{problem1}' using action.config defaults where not overridden by run params.")
    solution1 = await action.run(
        problem_description=problem1
        # max_iterations and num_thoughts_to_generate will use action.config values from above
    )
    print(f"\nProblem: {problem1}\nSolution: {solution1}\n" + "-"*30)

    problem2 = "Propose three innovative features for a smart home assistant for elderly users."
    logger.info(f"Test 2: Running GoTReasoningAction for: '{problem2}' with direct parameter overrides.")
    solution2 = await action.run(
        problem_description=problem2,
        max_iterations=1,
        num_thoughts_to_generate=3,
        generation_instruction_modifier="Focus on ease of use and safety features.",
        evaluation_scoring_criteria="ease of use, safety benefit, and innovation"
    )
    print(f"\nProblem: {problem2}\nSolution: {solution2}\n" + "-"*30)

if __name__ == '__main__':
    asyncio.run(main_test_action_directly())
