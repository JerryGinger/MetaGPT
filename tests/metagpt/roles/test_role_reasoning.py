import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from metagpt.roles.role import Role, RoleContext
from metagpt.schema import Message, AIMessage, ActionOutput
from metagpt.actions import Action
from metagpt.provider.base_llm import BaseLLM
from metagpt.config2 import Config
from metagpt.context import Context # Assuming Context is needed for Role

# Mock LLM for Role
class MockRoleLLM(BaseLLM):
    async def aask(self, prompt: str, system_msgs=None, format_msgs=None):
        # Used by Role's _think method to choose next state
        if "Now choose one of the following stages" in prompt:
            return "0" # Choose the first action/state
        return "LLM response for action"

    async def _achat_completion(self, messages: list[dict], options) -> dict: return {} # pragma: no cover
    async def _achat_completion_stream(self, messages: list[dict], options) -> None: pass # pragma: no cover
    def _user_msg(self, msg: str, **kwargs) -> dict: return {} # pragma: no cover
    def _assistant_msg(self, msg: str, **kwargs) -> dict: return {} # pragma: no cover
    def _system_msg(self, msg: str, **kwargs) -> dict: return {} # pragma: no cover
    def _history_msgs(self, history: list,剪掉系统消息: bool = True,剪掉最近n条消息: int = 0) -> list[dict]: return []
    def get_max_tokens(self, messages: list[dict]) -> int: return 4096
    def count_tokens(self, messages: list[dict]) -> int: return 0


# Mock Action for Role
class MockRoleAction(Action):
    name: str = "MockRoleAction"
    def __init__(self, name="MockRoleAction", context=None, llm=None):
        super().__init__(name, context, llm)
        # Define an action_outcls_map if your _act method tries to parse into specific types
        self.action_outcls_map = {self.name: ActionOutput}


    async def run(self, history: list[Message] = None) -> ActionOutput:
        return ActionOutput(content="Default action response", instruct_content=None)

@pytest.fixture
def mock_role_llm():
    return MockRoleLLM()

@pytest.fixture
def mock_context(mock_role_llm):
    # Create a basic Config instance, assuming it can be instantiated like this
    # or use a more specific mock if Config has complex dependencies.
    # We need to ensure llm_config and agent_config parts are accessible.

    # The Role's _process_role_extra expects self.llm to have a 'config' attribute
    # which in turn has api_key, model etc.
    mock_llm_config_obj = MagicMock()
    mock_llm_config_obj.api_key = "test_api_key"
    mock_llm_config_obj.model = "test_model"
    mock_llm_config_obj.base_url = "test_base_url"
    mock_llm_config_obj.temperature = 0.7
    mock_llm_config_obj.max_token = 1500
    mock_role_llm.config = mock_llm_config_obj # Attach the config object to the llm mock

    # The Role's _process_role_extra also expects self.config to be a Config object
    # where it can call self.config.get("reasoning_method", "none")
    # Let's create a mock for self.config (which is normally a Config instance from ConfigManager)
    # For this test, we'll mock the `get` method directly.
    role_specific_config = MagicMock(spec=Config)

    # Create a Context instance and assign the mocked config to it
    # Role's self.config comes from self.context.config
    ctx = Context()
    ctx.config = role_specific_config
    ctx.llm = mock_role_llm # Assign the llm to context as Role might use self.context.llm

    return ctx


@pytest.fixture
def generic_role(mock_role_llm, mock_context):
    # The Role needs a context which has a config object
    # The config object (mock_context.config) needs a `get` method
    # The Role also needs an llm, which is provided by mock_role_llm
    # Assign the llm to the context if Role expects self.context.llm
    mock_context.llm = mock_role_llm

    role = Role(name="TestRole", profile="Tester", goal="Test reasoning", context=mock_context)
    role.set_actions([MockRoleAction]) # Give it a default action
    role.llm = mock_role_llm # Explicitly set role's llm
    return role


@pytest.mark.asyncio
@patch('metagpt.agent.reasoners.got_reasoner.GraphOfThoughtsReasoner')
async def test_role_with_got_reasoner(MockGoTReasoner, generic_role: Role, mock_context):
    # Configure context to use GoT
    mock_context.config.get = Mock(side_effect=lambda key, default: {"reasoning_method": "GoT", "max_reasoning_steps": 1}.get(key, default))

    # Re-run _process_role_extra to apply new config (or re-initialize role for cleaner test)
    generic_role._process_role_extra() # This will initialize the reasoner based on new config

    assert MockGoTReasoner.called # Check if GoT Reasoner was attempted to be initialized
    mock_got_instance = MockGoTReasoner.return_value
    mock_got_instance.setup = Mock()
    mock_got_instance.execute_reasoning_step = AsyncMock(return_value={"status": "success", "message": "GoT step done"})
    mock_got_instance.get_current_solution = AsyncMock(return_value="GoT solution")
    mock_got_instance.is_finished = Mock(return_value=True) # Finishes in one step

    generic_role.rc.reasoner = mock_got_instance # Ensure the role uses our mock instance

    # Simulate an action to be performed
    generic_role.set_todo(generic_role.actions[0])

    # Provide a message for the role to react to, so _observe doesn't return immediately
    await generic_role.run(with_message="Test message for GoT") # run calls _observe, then _react -> _think -> _act

    mock_got_instance.setup.assert_called_once()
    mock_got_instance.execute_reasoning_step.assert_called_once()
    mock_got_instance.get_current_solution.assert_called_once()

    # Check the message generated by _act (which should be based on GoT solution)
    # The actual message is published, check memory for it.
    final_message = generic_role.rc.memory.get()[-1]
    assert final_message.content == "GoT solution"
    assert final_message.metadata.get("reasoner_used") == "GraphOfThoughtsReasoner"


@pytest.mark.asyncio
@patch('metagpt.agent.reasoners.simple_cot_reasoner.SimpleCotReasoner')
async def test_role_with_cot_reasoner(MockCoTReasoner, generic_role: Role, mock_context):
    mock_context.config.get = Mock(side_effect=lambda key, default: {"reasoning_method": "CoT", "max_reasoning_steps": 1}.get(key, default))
    generic_role._process_role_extra()

    assert MockCoTReasoner.called
    mock_cot_instance = MockCoTReasoner.return_value
    mock_cot_instance.setup = Mock()
    mock_cot_instance.execute_reasoning_step = AsyncMock(return_value={"status": "success", "solution": "CoT solution"})
    mock_cot_instance.get_current_solution = AsyncMock(return_value="CoT solution")
    mock_cot_instance.is_finished = Mock(return_value=True)
    generic_role.rc.reasoner = mock_cot_instance

    generic_role.set_todo(generic_role.actions[0])
    await generic_role.run(with_message="Test message for CoT")

    mock_cot_instance.setup.assert_called_once()
    mock_cot_instance.execute_reasoning_step.assert_called_once()
    mock_cot_instance.get_current_solution.assert_called_once()

    final_message = generic_role.rc.memory.get()[-1]
    assert final_message.content == "CoT solution"
    assert final_message.metadata.get("reasoner_used") == "SimpleCotReasoner"

@pytest.mark.asyncio
async def test_role_with_no_reasoner(generic_role: Role, mock_context):
    mock_context.config.get = Mock(side_effect=lambda key, default: {"reasoning_method": "none"}.get(key, default))
    generic_role._process_role_extra() # This will set reasoner to None

    assert generic_role.rc.reasoner is None

    # Mock the default action's run method to check if it's called
    default_action_instance = generic_role.actions[0]
    default_action_instance.run = AsyncMock(return_value=ActionOutput(content="Default action run response"))
    generic_role.set_todo(default_action_instance)

    await generic_role.run(with_message="Test message for no reasoner")

    default_action_instance.run.assert_called_once()
    final_message = generic_role.rc.memory.get()[-1]
    assert final_message.content == "Default action run response"
    assert "reasoner_used" not in final_message.metadata # No reasoner metadata should be present


@pytest.mark.asyncio
@patch('metagpt.agent.reasoners.got_reasoner.GraphOfThoughtsReasoner') # Using GoT for this test
async def test_role_multi_step_reasoning(MockGoTReasoner, generic_role: Role, mock_context):
    # Configure for GoT and multi-step
    mock_context.config.get = Mock(side_effect=lambda key, default: {"reasoning_method": "GoT", "max_reasoning_steps": 3}.get(key, default))
    generic_role._process_role_extra()

    mock_got_instance = MockGoTReasoner.return_value
    mock_got_instance.setup = Mock()
    mock_got_instance.execute_reasoning_step = AsyncMock(return_value={"status": "success"})
    mock_got_instance.get_current_solution = AsyncMock(return_value="Multi-step GoT solution")

    # Simulate is_finished: False, False, True for 3 steps
    mock_got_instance.is_finished = Mock(side_effect=[False, False, True])
    generic_role.rc.reasoner = mock_got_instance

    generic_role.set_todo(generic_role.actions[0])
    await generic_role.run(with_message="Test for multi-step")

    mock_got_instance.setup.assert_called_once()
    assert mock_got_instance.execute_reasoning_step.call_count == 3 # Called 3 times
    mock_got_instance.get_current_solution.assert_called_once() # Called once after loop

    final_message = generic_role.rc.memory.get()[-1]
    assert final_message.content == "Multi-step GoT solution"
    assert final_message.metadata.get("reasoner_used") == "GraphOfThoughtsReasoner"
