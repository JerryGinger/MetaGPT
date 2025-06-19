import pytest
import asyncio
from unittest.mock import patch, MagicMock, ANY

from metagpt.actions.got_reasoning_action import GoTReasoningAction
from metagpt.actions.got_langgraph_executor import GraphOfThoughtsState
from metagpt.provider.base_llm import BaseLLM # For mocking self.llm
from metagpt.config2 import Config # For mocking self.config

# Mock LLM to be used by the Action instance
@pytest.fixture
def mock_llm_for_action():
    llm = MagicMock(spec=BaseLLM)
    # If GoTReasoningAction or its callees try to access llm.config, mock that too
    llm.config = MagicMock()
    return llm

# Mock Config to be used by the Action instance
@pytest.fixture
def mock_config_for_action():
    config = MagicMock(spec=Config)
    # Setup default .get behaviors for all expected keys
    config.get.side_effect = lambda key, default=None: {
        'got_max_iterations': 3,
        'got_num_thoughts_to_generate': 2,
        'got_generation_prompt_template': None,
        'got_generation_instruction_modifier': None,
        'got_evaluation_prompt_template': None,
        'got_evaluation_score_scale_description': "0.0 to 1.0",
        'got_evaluation_scoring_criteria': "relevance, coherence",
        # 'got_generation_llm_config': {}, # Not used yet
        # 'got_evaluation_llm_config': {}, # Not used yet
    }.get(key, default)
    return config

@pytest.mark.asyncio
@patch('metagpt.actions.got_reasoning_action.got_langgraph_app')
async def test_got_reasoning_action_run_success(mock_got_app_ainvoke, mock_llm_for_action, mock_config_for_action):
    action = GoTReasoningAction()
    action.llm = mock_llm_for_action # Assign the mocked LLM
    action.config = mock_config_for_action # Assign the mocked Config

    problem_desc = "Test problem description"
    # Use different values than config defaults to test override
    max_iter_override = 2
    num_thoughts_override = 4
    gen_prompt_override = "Override gen prompt: {problem_description}"
    eval_criteria_override = "Override eval criteria"


    mock_final_state = GraphOfThoughtsState(
        problem_description=problem_desc, # This would be set by graph
        original_problem=problem_desc,
        current_thoughts=[{'value': "Mock solution thought", 'score': 0.9, 'justification': "Good"}],
        final_solution="Mocked final solution from LangGraph",
        iteration_count=max_iter_override, # Should reflect completion
        max_iterations=max_iter_override,
        num_thoughts_to_generate=num_thoughts_override,
        llm_instance=mock_llm_for_action,
        generation_prompt_template=gen_prompt_override,
        evaluation_scoring_criteria=eval_criteria_override,
        # Fill other state fields as expected by GraphOfThoughtsState type hint
        generation_instruction_modifier=None,
        raw_generation_output="Raw gen output",
        evaluation_prompt_template=None,
        evaluation_score_scale_description="0.0 to 1.0", # from config default
        raw_evaluation_outputs=[],
        num_dummy_thoughts=0
    )
    # mock_got_app_ainvoke is already a MagicMock due to @patch
    mock_got_app_ainvoke.ainvoke = AsyncMock(return_value=mock_final_state)


    solution = await action.run(
        problem_description=problem_desc,
        max_iterations=max_iter_override,
        num_thoughts_to_generate=num_thoughts_override,
        generation_prompt_template=gen_prompt_override,
        evaluation_scoring_criteria=eval_criteria_override
    )

    assert solution == "Mocked final solution from LangGraph"

    expected_initial_state_payload = {
        "problem_description": problem_desc,
        "original_problem": problem_desc,
        "current_thoughts": [],
        "final_solution": None,
        "iteration_count": 0,
        "max_iterations": max_iter_override,
        "llm_instance": mock_llm_for_action,
        "num_thoughts_to_generate": num_thoughts_override,
        "generation_prompt_template": gen_prompt_override,
        "generation_instruction_modifier": mock_config_for_action.get('got_generation_instruction_modifier'), # from config
        "evaluation_prompt_template": mock_config_for_action.get('got_evaluation_prompt_template'), # from config
        "evaluation_score_scale_description": mock_config_for_action.get('got_evaluation_score_scale_description'), # from config
        "evaluation_scoring_criteria": eval_criteria_override,
        "raw_generation_output": None,
        "raw_evaluation_outputs": [],
        "num_dummy_thoughts": 0
    }
    mock_got_app_ainvoke.ainvoke.assert_called_once_with(expected_initial_state_payload)

@pytest.mark.asyncio
@patch('metagpt.actions.got_reasoning_action.got_langgraph_app')
async def test_got_reasoning_action_run_no_solution(mock_got_app_ainvoke, mock_llm_for_action, mock_config_for_action):
    action = GoTReasoningAction()
    action.llm = mock_llm_for_action
    action.config = mock_config_for_action
    problem_desc = "Another test problem"

    mock_final_state_no_solution = GraphOfThoughtsState(
        problem_description=problem_desc, original_problem=problem_desc, current_thoughts=[],
        final_solution=None, # Key: no solution
        iteration_count=1, max_iterations=1, num_thoughts_to_generate=1,
        llm_instance=mock_llm_for_action,
        # Fill other fields minimally for type correctness
        generation_prompt_template=None, generation_instruction_modifier=None, raw_generation_output=None,
        evaluation_prompt_template=None, evaluation_score_scale_description="0.0 to 1.0",
        evaluation_scoring_criteria="criteria", raw_evaluation_outputs=[], num_dummy_thoughts=0
    )
    mock_got_app_ainvoke.ainvoke = AsyncMock(return_value=mock_final_state_no_solution)

    solution = await action.run(problem_description=problem_desc, max_iterations=1, num_thoughts_to_generate=1)

    assert "GoT process completed, but no final solution was explicitly set." in solution
    mock_got_app_ainvoke.ainvoke.assert_called_once()

@pytest.mark.asyncio
@patch('metagpt.actions.got_reasoning_action.got_langgraph_app')
async def test_got_reasoning_action_run_langgraph_exception(mock_got_app_ainvoke, mock_llm_for_action, mock_config_for_action):
    action = GoTReasoningAction()
    action.llm = mock_llm_for_action
    action.config = mock_config_for_action
    problem_desc = "Problem causing exception"

    mock_got_app_ainvoke.ainvoke = AsyncMock(side_effect=RuntimeError("LangGraph internal error"))

    solution = await action.run(problem_description=problem_desc)

    assert "Error in GoT reasoning process: LangGraph internal error" in solution
    mock_got_app_ainvoke.ainvoke.assert_called_once()

@pytest.mark.asyncio
async def test_got_reasoning_action_no_llm_configured():
    action = GoTReasoningAction()
    action.llm = None # Simulate LLM not being configured
    action.config = MagicMock(spec=Config) # Add a mock config

    solution = await action.run(problem_description="Test problem")
    assert "Error: LLM not configured" in solution


@pytest.mark.asyncio
@patch('metagpt.actions.got_reasoning_action.logger') # To check log messages
async def test_got_reasoning_action_main_example_runs(mock_logger):
    # This tests the if __name__ == '__main__' block's helper function main_test_action_directly
    # It will use the actual LangGraph app with its dummy/mocked LLM logic from the executor.

    # Temporarily patch sys.argv or other conditions if main_test_action_directly depends on them
    # For this test, we call it directly.
    from metagpt.actions.got_reasoning_action import main_test_action_directly

    # Patch the LLM within the main_test_action_directly scope if it tries to init a real one
    with patch('metagpt.actions.got_reasoning_action.LLM') as MockLLMInMain:
        mock_llm_instance_for_main = MagicMock(spec=BaseLLM)
        async def main_mock_aask(prompt_text: str, system_msgs=None, format_msgs=None):
            if "Generate" in prompt_text: return "1. Main test thought 1"
            if "Evaluate" in prompt_text: return "Score: 0.7\nJustification: Main test justification"
            return "Main default mock"
        mock_llm_instance_for_main.aask = main_mock_aask
        mock_llm_instance_for_main.config = MagicMock()
        MockLLMInMain.return_value = mock_llm_instance_for_main

        await main_test_action_directly()

    # Check if logger was used for info messages (e.g., "Running GoTReasoningAction for...")
    # This depends on what main_test_action_directly logs.
    # Example: mock_logger.info.assert_any_call(ANY)
    assert mock_logger.info.call_count > 0
    # More specific checks can be added if main_test_action_directly's output is captured or logged consistently.


def test_action_name_default():
    action = GoTReasoningAction()
    assert action.name == "GoTReasoningAction"

def test_action_custom_name():
    action = GoTReasoningAction(name="CustomGoTAction")
    assert action.name == "CustomGoTAction"

# Test that config default is used if a run param is None
@pytest.mark.asyncio
@patch('metagpt.actions.got_reasoning_action.got_langgraph_app')
async def test_got_reasoning_action_config_override_logic(mock_got_app_ainvoke, mock_llm_for_action, mock_config_for_action):
    action = GoTReasoningAction()
    action.llm = mock_llm_for_action

    # Set specific values in mock_config_for_action that differ from run defaults
    mock_config_for_action.get.side_effect = lambda key, default: {
        'got_max_iterations': 5, # Config default
        'got_num_thoughts_to_generate': 6, # Config default
        'got_generation_instruction_modifier': "Config modifier",
        # other keys...
    }.get(key, default)
    action.config = mock_config_for_action

    problem_desc = "Test config override"

    # Case 1: Pass None for some params, should use config values
    await action.run(
        problem_description=problem_desc,
        max_iterations=None, # Should use config's 5
        num_thoughts_to_generate=None, # Should use config's 6
        generation_instruction_modifier=None # Should use config's "Config modifier"
    )

    called_args, _ = mock_got_app_ainvoke.ainvoke.call_args
    payload1 = called_args[0]
    assert payload1['max_iterations'] == 5
    assert payload1['num_thoughts_to_generate'] == 6
    assert payload1['generation_instruction_modifier'] == "Config modifier"

    # Case 2: Pass specific values for params, should override config values
    await action.run(
        problem_description=problem_desc,
        max_iterations=10,
        num_thoughts_to_generate=11,
        generation_instruction_modifier="Direct run modifier"
    )
    called_args_2, _ = mock_got_app_ainvoke.ainvoke.call_args
    payload2 = called_args_2[0]
    assert payload2['max_iterations'] == 10
    assert payload2['num_thoughts_to_generate'] == 11
    assert payload2['generation_instruction_modifier'] == "Direct run modifier"
