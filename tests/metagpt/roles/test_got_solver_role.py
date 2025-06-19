import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from metagpt.roles.got_solver_role import GoTSolverRole
from metagpt.schema import Message, AIMessage
from metagpt.context import Context # Using actual Context
from metagpt.memory import Memory
from metagpt.config2 import Config # Using actual Config for role.config

# Minimal mock for Config object to be used by Role's context
class MockRoleConfig(Config):
    def __init__(self, config_dict=None, **data):
        super().__init__(**data) # Pass other potential Config args if any
        self._config_data = config_dict if config_dict else {}

    def get(self, key, default=None):
        return self._config_data.get(key, default)

    # Add other methods if Role's usage of self.config becomes more complex
    # For now, only `get` is essential for GoTSolverRole's _act method.


@pytest.fixture
def mock_llm_for_role():
    # This LLM is for the Role itself, not directly for the Action if Action is fully mocked.
    # If Role's _think or other parts used LLM, this would be important.
    # For GoTSolverRole, self.llm is passed to GoTReasoningAction, so it needs to be a valid LLM mock.
    llm = MagicMock()
    llm.config = MagicMock() # Mocking llm.config as it's accessed by Role's _process_role_extra
    llm.config.api_key = "test_key"
    llm.config.model = "test_model"
    llm.config.base_url = "test_url"
    llm.config.temperature = 0.7
    llm.config.max_token = 1500
    return llm


@pytest.fixture
def got_solver_role_with_mocks(mock_llm_for_role):
    # Patch GoTReasoningAction at the point of import for got_solver_role module
    with patch('metagpt.roles.got_solver_role.GoTReasoningAction') as MockGoTReasoningActionClass:
        # Configure the instance that will be created by the Role
        mock_action_instance = MockGoTReasoningActionClass.return_value
        mock_action_instance.run = AsyncMock(return_value="Mocked GoT Solution from Role Test")
        mock_action_instance.name = "MockedGoTActionForRole"

        # Setup context for the Role
        ctx = Context() # Using a real Context
        # Configure the context's config object for the Role
        role_specific_config_data = {
            "got_max_iterations": 2,
            "got_num_thoughts_to_generate": 3, # Changed from got_num_dummy_thoughts
            "got_generation_prompt_template": "Role-level generation prompt template",
            "got_evaluation_scoring_criteria": "Role-level evaluation criteria"
            # Add other got_* keys as needed to test they are passed
        }
        ctx.config = MockRoleConfig(config_dict=role_specific_config_data)
        ctx.llm = mock_llm_for_role # Role will get its LLM from context if not passed directly

        # Initialize the role
        # The Role's __init__ will create an instance of GoTReasoningAction (which is mocked)
        # and its self.config will be ctx.config
        role = GoTSolverRole(context=ctx)
        # Ensure role.llm is set, Role's __init__ should handle this via context.
        # If role.llm is not set by __init__ based on context.llm, then:
        # role.llm = mock_llm_for_role # Explicitly if needed

        # The role's self.got_action will be the mocked instance due to the patch
        yield role # Yield the role instance for use in tests


@pytest.mark.asyncio
async def test_gotsolverrole_act_with_message(got_solver_role_with_mocks: GoTSolverRole):
    role = got_solver_role_with_mocks
    problem_desc = "Solve this complex problem via integration test."
    test_message = Message(content=problem_desc, role="User")

    role.rc.memory = Memory()
    role.rc.memory.add(test_message)

    response_message = await role._act()

    mock_action_instance = role.got_action

    # Assert that the mocked action's run method was called with correct parameters
    # from the role's config, plus the problem_description.
    # Direct params to action.run take precedence, then role.config, then action defaults.
    # GoTSolverRole._act currently passes parameters explicitly from its own config.
    expected_run_args = {
        "problem_description": problem_desc,
        "max_iterations": role.config.get("got_max_iterations"),
        "num_thoughts_to_generate": role.config.get("got_num_thoughts_to_generate"),
        # These are not explicitly passed by GoTSolverRole._act from its config in the current version,
        # so they would be None if not set in action.run defaults or if action.config itself is not populated.
        # GoTReasoningAction.run now has defaults or gets from its *own* config for these.
        # What we are testing here is that GoTSolverRole *could* pass them if it chose to.
        # The current GoTSolverRole._act only passes problem_desc, max_iter, num_thoughts.
        # So, we verify those.
    }
    # Check only the args explicitly passed by GoTSolverRole._act
    mock_action_instance.run.assert_called_once()
    called_kwargs = mock_action_instance.run.call_args.kwargs
    assert called_kwargs.get("problem_description") == problem_desc
    assert called_kwargs.get("max_iterations") == role.config.get("got_max_iterations")
    assert called_kwargs.get("num_thoughts_to_generate") == role.config.get("got_num_thoughts_to_generate")


    assert isinstance(response_message, AIMessage)
    assert response_message.content == "Mocked GoT Solution from Role Test"
    assert response_message.role == role.profile
    assert response_message.sent_from == role.name
    assert response_message.cause_by == mock_action_instance.name


@pytest.mark.asyncio
async def test_gotsolverrole_act_no_message_in_memory(got_solver_role_with_mocks: GoTSolverRole):
    role = got_solver_role_with_mocks
    role.rc.memory = Memory()

    response_message = await role._act()

    mock_action_instance = role.got_action
    mock_action_instance.run.assert_not_called()

    assert isinstance(response_message, AIMessage)
    assert response_message.content == "No problem received to solve."
    assert response_message.cause_by == mock_action_instance.name


@pytest.mark.asyncio
async def test_gotsolverrole_act_message_no_content(got_solver_role_with_mocks: GoTSolverRole):
    role = got_solver_role_with_mocks
    test_message_no_content = Message(content="", role="User")
    role.rc.memory = Memory()
    role.rc.memory.add(test_message_no_content)

    response_message = await role._act()

    mock_action_instance = role.got_action
    mock_action_instance.run.assert_not_called()

    assert isinstance(response_message, AIMessage)
    assert response_message.content == "Incoming message had no problem description."


@pytest.mark.asyncio
@patch('metagpt.roles.got_solver_role.logger')
async def test_gotsolverrole_act_action_exception(mock_logger, got_solver_role_with_mocks: GoTSolverRole):
    role = got_solver_role_with_mocks
    problem_desc = "Problem that causes action error in role test."
    test_message = Message(content=problem_desc)
    role.rc.memory = Memory()
    role.rc.memory.add(test_message)

    mock_action_instance = role.got_action
    error_message = "Simulated action error from role test"
    mock_action_instance.run = AsyncMock(side_effect=Exception(error_message))

    response_message = await role._act()

    mock_action_instance.run.assert_called_once()

    # Check that an error was logged by the Role's _act method
    logged_error = False
    for call in mock_logger.error.call_args_list:
        if error_message in call.args[0]:
            logged_error = True
            break
    assert logged_error, "Error message not logged by role's _act"

    assert isinstance(response_message, AIMessage)
    assert f"An error occurred while trying to solve the problem: {error_message}" in response_message.content


def test_gotsolverrole_initialization_and_config(mock_llm_for_role):
    # Test if the role initializes and sets up its action, and picks up config
    role_config_data = {
        "name": "TestGoTRole", # Override default name
        "profile": "TestProfile",
        "goal": "TestGoal",
        "got_max_iterations": 5,
        "got_num_thoughts_to_generate": 6,
        "some_other_config": "value"
    }
    ctx = Context()
    ctx.config = MockRoleConfig(config_dict=role_config_data)
    ctx.llm = mock_llm_for_role


    with patch('metagpt.roles.got_solver_role.GoTReasoningAction') as MockActionForInit:
        # Pass the **role_config_data to Role init which should populate self.config
        # Role's __init__ takes **kwargs, which are passed to BaseModel.
        # If these keys (name, profile, goal) are direct fields of Role, they'll be set.
        # Other keys will go into self.config if Role's __init__ handles it, or self.context.config is used.
        # MetaGPT's Role takes context, and self.config is self.context.config.
        # So, the config is primarily through context.
        role = GoTSolverRole(context=ctx, **role_config_data) # Pass kwargs for name, profile, goal

        MockActionForInit.assert_called_once()
        assert role.name == "TestGoTRole" # Check if name from config was used
        assert role.profile == "TestProfile"
        assert role.goal == "TestGoal"
        assert isinstance(role.got_action, MagicMock)

        # Check if the role's self.config correctly reflects the passed config
        assert role.config.get("got_max_iterations") == 5
        assert role.config.get("got_num_thoughts_to_generate") == 6
        assert role.config.get("some_other_config") == "value"
        # Ensure the got_action (which is a mock here) has its self.config set if it needs it
        # The actual GoTReasoningAction will have its self.config set by its own __init__
        # using the context it receives from the role.
        # Here, role.got_action is a MagicMock, so it doesn't have a config unless we assign it.
        # What's important is that the role.config is correct, as the action will use that.

        # Test that GoTReasoningAction (if it were real) would get the right config from context.
        # We can simulate this by checking the context passed to its constructor if we didn't patch it.
        # Since it's patched, we assume the role correctly passes its context to the action.
        # The action's `self.config` will be `role.context.config`.
        assert role.got_action.config is role.context.config # Check if action would get role's context config
                                                            # This test is a bit indirect for the action's config
                                                            # as got_action is a mock.
                                                            # A better test would be to instantiate the real action
                                                            # with the role's context and check action.config.
                                                            # However, for this integration test, focusing on the Role's behavior.
                                                            # The Action's own unit tests cover its config handling.

    # A more direct test of what config the action would see:
    real_action_config_test = MockRoleConfig({"test_action_specific_key": "action_val"})
    real_action_context = Context()
    real_action_context.config = real_action_config_test

    # from metagpt.actions.got_reasoning_action import GoTReasoningAction # Not patching here
    # real_action = GoTReasoningAction(context=real_action_context)
    # assert real_action.config.get("test_action_specific_key") == "action_val"
    # This confirms Action gets config from its context. Role ensures Action gets context.
