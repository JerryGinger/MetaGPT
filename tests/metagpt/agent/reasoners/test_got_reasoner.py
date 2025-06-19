import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from metagpt.agent.reasoners.got_reasoner import GraphOfThoughtsReasoner
from metagpt.agent.reasoners.got_prompter import GoTPrompter
from metagpt.agent.reasoners.got_parser import GoTParser

# Mock GoT components if they are not easily testable or have external dependencies
# For this test, we'll mock the main GoT classes.
@patch('metagpt.agent.reasoners.got_reasoner.OpenAILanguageModel')
@patch('metagpt.agent.reasoners.got_reasoner.Controller')
@patch('metagpt.agent.reasoners.got_reasoner.GraphOfOperations')
@patch('metagpt.agent.reasoners.got_reasoner.Generate')
@patch('metagpt.agent.reasoners.got_reasoner.Score')
def test_got_reasoner_init(MockScore, MockGenerate, MockGOP, MockController, MockOpenAILM):
    mock_llm_config = {"api_key": "test_key", "model": "gpt-test"}
    mock_agent_config = {"max_thoughts": 7, "max_depth": 4, "reasoning_method": "GoT"}

    # Mock the _create_dummy_lm_config to avoid file system operations
    with patch.object(GraphOfThoughtsReasoner, '_create_dummy_lm_config', return_value="dummy_path/config.json"):
        reasoner = GraphOfThoughtsReasoner(llm_config=mock_llm_config, agent_config=mock_agent_config)

    MockOpenAILM.assert_called_once_with(config_path="dummy_path/config.json")

    # Check if Prompter and Parser are instantiated
    assert isinstance(reasoner.prompter, GoTPrompter)
    assert isinstance(reasoner.parser, GoTParser)

    # Check if GOP is created and operations are appended
    MockGOP.assert_called_once()
    assert MockGenerate.call_count > 0 # Generate op should be added
    assert MockScore.call_count > 0  # Score op should be added

    # Check if Controller is instantiated with the right components
    MockController.assert_called_once()
    controller_args, _ = MockController.call_args
    assert controller_args[0].lm == MockOpenAILM.return_value # lm
    assert controller_args[0].gop == MockGOP.return_value    # gop
    assert controller_args[0].prompter == reasoner.prompter  # prompter
    assert controller_args[0].parser == reasoner.parser      # parser
    assert controller_args[0].max_thoughts == 7
    assert controller_args[0].max_depth == 4


@patch('metagpt.agent.reasoners.got_reasoner.OpenAILanguageModel')
@patch('metagpt.agent.reasoners.got_reasoner.Controller')
@patch('metagpt.agent.reasoners.got_reasoner.GraphOfOperations')
def test_got_reasoner_setup(MockGOP, MockController, MockOpenAILM):
    mock_llm_config = {"api_key": "test_key"}
    mock_agent_config = {}

    with patch.object(GraphOfThoughtsReasoner, '_create_dummy_lm_config', return_value="dummy_path/config.json"):
        reasoner = GraphOfThoughtsReasoner(llm_config=mock_llm_config, agent_config=mock_agent_config)

    problem_desc = "Solve for x: 2x + 3 = 7"
    initial_ctx = {"variable_type": "integer"}
    reasoner.setup(problem_description=problem_desc, initial_context=initial_ctx)

    assert reasoner.problem_description == problem_desc
    assert reasoner.initial_context == initial_ctx

    # Check if the controller's initial_state is updated (or prepared for run)
    expected_current_thought = f"Problem: {problem_desc}. Context: {initial_ctx}"
    assert reasoner.controller.initial_state["original_problem"] == problem_desc
    assert reasoner.controller.initial_state["current"] == expected_current_thought


@patch('metagpt.agent.reasoners.got_reasoner.OpenAILanguageModel')
@patch('metagpt.agent.reasoners.got_reasoner.Controller')
@patch('metagpt.agent.reasoners.got_reasoner.GraphOfOperations')
@pytest.mark.asyncio
async def test_execute_reasoning_step(MockGOP, MockController, MockOpenAILM):
    mock_llm_config = {"api_key": "test_key"}
    mock_agent_config = {}

    with patch.object(GraphOfThoughtsReasoner, '_create_dummy_lm_config', return_value="dummy_path/config.json"):
        reasoner = GraphOfThoughtsReasoner(llm_config=mock_llm_config, agent_config=mock_agent_config)

    # Setup the reasoner
    problem_desc = "Test problem"
    reasoner.setup(problem_description=problem_desc, initial_context={})

    # Mock controller.run()
    mock_controller_instance = MockController.return_value
    mock_controller_instance.run = AsyncMock() # For async version if controller.run is async
    # If controller.run is synchronous, use: mock_controller_instance.run = Mock()

    # Mock controller.thoughts to simulate results after run
    mock_controller_instance.thoughts = [Mock(value="Thought 1", score=0.8)]


    result = await reasoner.execute_reasoning_step()

    mock_controller_instance.run.assert_called_once()
    # The initial_thoughts passed to run should be based on the setup
    expected_initial_thought_for_run = [reasoner.controller.initial_state.get("current", problem_desc)]
    mock_controller_instance.run.assert_called_with(initial_thoughts=expected_initial_thought_for_run)

    assert result["status"] == "success"
    assert "GoT step executed" in result["message"]
    assert result["thoughts_count"] == 1


@patch('metagpt.agent.reasoners.got_reasoner.OpenAILanguageModel')
@patch('metagpt.agent.reasoners.got_reasoner.Controller')
@patch('metagpt.agent.reasoners.got_reasoner.GraphOfOperations')
@pytest.mark.asyncio
async def test_get_current_solution(MockGOP, MockController, MockOpenAILM):
    mock_llm_config = {"api_key": "test_key"}
    mock_agent_config = {}

    with patch.object(GraphOfThoughtsReasoner, '_create_dummy_lm_config', return_value="dummy_path/config.json"):
        reasoner = GraphOfThoughtsReasoner(llm_config=mock_llm_config, agent_config=mock_agent_config)

    mock_controller_instance = MockController.return_value

    # Case 1: No thoughts
    mock_controller_instance.thoughts = []
    solution = await reasoner.get_current_solution()
    assert solution is None

    # Case 2: Thoughts with scores
    thought1 = MagicMock()
    thought1.value = "Solution A"
    thought1.score = 0.9
    thought2 = MagicMock()
    thought2.value = "Solution B"
    thought2.score = 0.7
    mock_controller_instance.thoughts = [thought2, thought1] # Unsorted

    solution = await reasoner.get_current_solution()
    assert solution == "Solution A" # Should pick the one with higher score

    # Case 3: Thoughts without scores (should pick the last one as fallback)
    thought3 = MagicMock(spec=['value']) # Only has 'value'
    thought3.value = "Solution C"
    thought4 = MagicMock(spec=['value'])
    thought4.value = "Solution D"
    # Remove score attribute for this test or ensure it's not present
    del thought3.score
    del thought4.score

    mock_controller_instance.thoughts = [thought3, thought4]
    solution = await reasoner.get_current_solution()
    assert solution == "Solution D"

    # Case 4: Thoughts are just strings (if that's a possible state, though less likely with GoT objects)
    # This part of get_current_solution might be less relevant if thoughts are always objects.
    # For now, the code tries to access .value, so raw strings would cause AttributeError handled by the try-except.
    # If thoughts can be raw strings:
    # mock_controller_instance.thoughts = ["Raw Sol E", "Raw Sol F"]
    # solution = await reasoner.get_current_solution()
    # assert solution == "Raw Sol F"


def test_is_finished(mock_llm_config, mock_agent_config):
    with patch.object(GraphOfThoughtsReasoner, '_create_dummy_lm_config', return_value="dummy_path/config.json"):
        with patch('metagpt.agent.reasoners.got_reasoner.OpenAILanguageModel'), \
             patch('metagpt.agent.reasoners.got_reasoner.Controller'), \
             patch('metagpt.agent.reasoners.got_reasoner.GraphOfOperations'):
            reasoner = GraphOfThoughtsReasoner(llm_config=mock_llm_config, agent_config=mock_agent_config)

    # Default behavior is False
    assert not reasoner.is_finished()

@pytest.fixture
def mock_llm_config():
    return {"api_key": "test_key", "model": "gpt-test"}

@pytest.fixture
def mock_agent_config():
    return {"max_thoughts": 5, "max_depth": 3}


# Test for _simple_scoring_function
def test_simple_scoring_function():
    score1 = GraphOfThoughtsReasoner._simple_scoring_function("This is a short thought.")
    assert 0.0 <= score1 <= 1.0

    score2 = GraphOfThoughtsReasoner._simple_scoring_function("This is a much longer thought, which should theoretically get a higher base score before clamping.")
    assert 0.0 <= score2 <= 1.0

    score_solution = GraphOfThoughtsReasoner._simple_scoring_function("This thought contains the magic word: solution.")
    assert 0.0 <= score_solution <= 1.0
    if "solution" in "This thought contains the magic word: solution.": # check logic
        assert score_solution > GraphOfThoughtsReasoner._simple_scoring_function("This thought is similar length but no keyword.")

    score_invalid = GraphOfThoughtsReasoner._simple_scoring_function(123) # Invalid type
    assert score_invalid == 0.0

    # Test clamping
    long_string = "solution " * 100 # very long string with "solution"
    score_clamped = GraphOfThoughtsReasoner._simple_scoring_function(long_string)
    assert score_clamped == 1.0

# Test for _create_dummy_lm_config
# This test involves file system operations, which can be tricky in some CI environments.
# It's good to have, but can be marked to skip if needed.
@patch.dict(os.environ, {"OPENAI_API_KEY": "env_api_key"}, clear=True)
def test_create_dummy_lm_config_with_env_key(tmp_path):
    llm_config = {"model": "gpt-test-dummy"}
    # Temporarily change working directory to tmp_path to ensure config is created there
    # and easily cleaned up.
    original_cwd = os.getcwd()
    # Create a unique subdirectory within tmp_path for this test run
    test_run_config_dir = tmp_path / "test_config_run"
    os.makedirs(test_run_config_dir, exist_ok=True)
    os.chdir(test_run_config_dir)

    try:
        config_path_output = GraphOfThoughtsReasoner._create_dummy_lm_config(
            MagicMock(llm_config=llm_config) # Mock self, pass only llm_config
        )

        assert config_path_output is not None
        assert os.path.exists(config_path_output)

        import json
        with open(config_path_output, 'r') as f:
            config_data = json.load(f)

        assert config_data["api_key"] == "env_api_key"
        assert config_data["model_name"] == "gpt-test-dummy"
    finally:
        os.chdir(original_cwd) # Restore original CWD

def test_create_dummy_lm_config_with_direct_key(tmp_path):
    llm_config = {"api_key": "direct_api_key", "model": "gpt-test-direct"}
    original_cwd = os.getcwd()
    test_run_config_dir = tmp_path / "test_config_run_direct"
    os.makedirs(test_run_config_dir, exist_ok=True)
    os.chdir(test_run_config_dir)
    try:
        config_path_output = GraphOfThoughtsReasoner._create_dummy_lm_config(
            MagicMock(llm_config=llm_config)
        )
        assert config_path_output is not None
        assert os.path.exists(config_path_output)
        import json
        with open(config_path_output, 'r') as f:
            config_data = json.load(f)
        assert config_data["api_key"] == "direct_api_key"
    finally:
        os.chdir(original_cwd)

def test_create_dummy_lm_config_no_key(tmp_path):
    # Ensure no relevant env var is set for this test
    with patch.dict(os.environ, {}, clear=True):
        llm_config = {"model": "gpt-test-no-key"}
        original_cwd = os.getcwd()
        test_run_config_dir = tmp_path / "test_config_run_no_key"
        os.makedirs(test_run_config_dir, exist_ok=True)
        os.chdir(test_run_config_dir)
        try:
            config_path_output = GraphOfThoughtsReasoner._create_dummy_lm_config(
                MagicMock(llm_config=llm_config)
            )
            assert config_path_output is None # Should fail to create if no key
        finally:
            os.chdir(original_cwd)
