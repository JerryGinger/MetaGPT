import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from metagpt.agent.reasoners.simple_cot_reasoner import SimpleCotReasoner
from metagpt.provider.base_llm import BaseLLM
from metagpt.schema import AIMessage # Assuming AIMessage can be used for LLM interaction result

# Mock LLM for testing
class MockLLM(BaseLLM):
    def __init__(self, config=None):
        super().__init__(config)
        self.resp = "LLM response"

    async def aask(self, prompt: str, system_msgs=None, format_msgs=None):
        return self.resp

    # Add other abstract methods if necessary for instantiation, though not used in these tests
    async def _achat_completion(self, messages: list[dict], options) -> dict: # pragma: no cover
        return {}

    async def _achat_completion_stream(self, messages: list[dict], options) -> None: # pragma: no cover
        pass

    def _user_msg(self, msg: str, **kwargs) -> dict: # pragma: no cover
        return {}

    def _assistant_msg(self, msg: str, **kwargs) -> dict: # pragma: no cover
        return {}

    def _system_msg(self, msg: str, **kwargs) -> dict: # pragma: no cover
        return {}

    def _history_msgs(self, history: list[BaseLLM._MessageType],剪掉系统消息: bool = True,剪掉最近n条消息: int = 0) -> list[dict]:
        return [] # pragma: no cover

    def get_max_tokens(self, messages: list[dict]) -> int:
        return 4096 # pragma: no cover

    def count_tokens(self, messages: list[dict]) -> int:
        return 0 # pragma: no cover


@pytest.fixture
def mock_llm_instance(): # Renamed to avoid conflict if a class MockLLM is also used as a type hint
    return MockLLM()

@pytest.fixture
def cot_reasoner(mock_llm_instance): # Updated fixture name
    return SimpleCotReasoner(llm=mock_llm_instance)

def test_simple_cot_reasoner_init(mock_llm_instance): # Updated fixture name
    reasoner = SimpleCotReasoner(llm=mock_llm_instance)
    assert reasoner.llm is mock_llm_instance
    assert reasoner.problem_description == ""
    assert reasoner.solution == ""
    assert not reasoner.finished

def test_simple_cot_reasoner_init_no_llm():
    with pytest.raises(ValueError, match="SimpleCotReasoner requires an LLM instance."):
        SimpleCotReasoner(llm=None)

def test_setup(cot_reasoner):
    problem = "What is 2+2?"
    context = {"type": "math"}
    cot_reasoner.setup(problem, context)
    assert cot_reasoner.problem_description == problem
    assert cot_reasoner.initial_context == context
    assert cot_reasoner.solution == ""
    assert not cot_reasoner.finished

@pytest.mark.asyncio
async def test_execute_reasoning_step_no_setup(cot_reasoner):
    # Ensure problem_description is empty to simulate no setup
    cot_reasoner.problem_description = ""
    result = await cot_reasoner.execute_reasoning_step()
    assert result["status"] == "error"
    assert "Problem not set up" in result["message"]

@pytest.mark.asyncio
async def test_execute_reasoning_step_success(cot_reasoner, mock_llm_instance): # Updated fixture name
    problem = "What is the capital of France?"
    context = {"hint": "It's a famous city."}
    cot_reasoner.setup(problem, context)

    expected_llm_response = "Paris is the capital of France."
    mock_llm_instance.resp = expected_llm_response # Set the mock response for the instance

    # Mock the aask method on the instance of MockLLM
    mock_llm_instance.aask = AsyncMock(return_value=expected_llm_response)

    result = await cot_reasoner.execute_reasoning_step()

    assert result["status"] == "success"
    assert result["solution"] == expected_llm_response
    assert cot_reasoner.solution == expected_llm_response
    assert cot_reasoner.finished

    expected_prompt = f"Problem: {problem}\n\nLet's think step by step to arrive at a solution.\n\nInitial Context:\n"
    for key, value in context.items():
        expected_prompt += f"- {key}: {value}\n"
    expected_prompt += "\nNow, let's think step by step:"

    mock_llm_instance.aask.assert_called_once_with(expected_prompt)


@pytest.mark.asyncio
async def test_execute_reasoning_step_already_finished(cot_reasoner):
    problem = "Test problem"
    cot_reasoner.setup(problem)
    # Manually set finished to True for this test case
    cot_reasoner.finished = True
    cot_reasoner.solution = "Old solution"

    result = await cot_reasoner.execute_reasoning_step()
    assert result["status"] == "already_finished"
    assert result["solution"] == "Old solution"

@pytest.mark.asyncio
async def test_get_current_solution_not_finished(cot_reasoner):
    cot_reasoner.setup("New problem")
    solution = await cot_reasoner.get_current_solution()
    assert solution is None

@pytest.mark.asyncio
async def test_get_current_solution_finished(cot_reasoner, mock_llm_instance): # Updated fixture name
    problem = "What is 2+2?"
    cot_reasoner.setup(problem)

    # Simulate successful execution
    expected_response = "The answer is 4."
    mock_llm_instance.aask = AsyncMock(return_value=expected_response)
    await cot_reasoner.execute_reasoning_step()

    solution = await cot_reasoner.get_current_solution()
    assert solution == expected_response

def test_is_finished(cot_reasoner):
    assert not cot_reasoner.is_finished()
    cot_reasoner.finished = True # Manually set for testing
    assert cot_reasoner.is_finished()
