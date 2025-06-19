"""
Example MetaGPT Role for Graph of Thoughts (GoT) Reasoning.

This module defines `GoTSolverRole`, a Role that utilizes the `GoTReasoningAction`
to solve problems presented to it. It demonstrates how a Role can integrate a
complex reasoning action based on LangGraph for its primary task. The GoT process
itself is now LLM-powered for thought generation and evaluation.
"""
import asyncio
from metagpt.roles.role import Role
from metagpt.actions.got_reasoning_action import GoTReasoningAction
from metagpt.schema import Message, AIMessage # AIMessage for responses
from metagpt.logs import logger

class GoTSolverRole(Role):
    """
    A Role specialized in solving problems using a Graph of Thoughts (GoT)
    reasoning process, orchestrated by the LLM-powered `GoTReasoningAction`.

    This Role expects to receive a problem description via a Message in its memory.
    It then uses `GoTReasoningAction` to find a solution. The behavior of the
    GoT process (e.g., number of iterations, thoughts per step, prompt templates)
    can be configured through the Role's `self.config` object, typically loaded
    from a YAML file.

    Attributes:
        name (str): Name of the role.
        profile (str): Profile description of the role.
        goal (str): The primary goal of this role.
        got_action (GoTReasoningAction): An instance of the GoT reasoning action.

    Expected `self.config` keys for GoT customization (see `GoTReasoningAction` for more):
        got_max_iterations (int): Maximum iterations for the GoT process.
        got_num_thoughts_to_generate (int): Number of thoughts for LLM to generate per step.
        got_generation_prompt_template (str, optional): Custom prompt for thought generation.
        got_evaluation_prompt_template (str, optional): Custom prompt for thought evaluation.
        # ... and other keys supported by GoTReasoningAction.
    """
    name: str = "GoTSolver"
    profile: str = "GoTSolver"
    goal: str = "Solve complex problems using an LLM-driven Graph of Thoughts reasoning process."
    # constraints: str = "Solutions should be well-justified and clearly explained."

    def __init__(self, **kwargs):
        """
        Initializes the GoTSolverRole.

        Args:
            **kwargs: Arguments to be passed to the parent Role class. The `context`
                      argument, if provided, should contain an LLM instance (`context.llm`)
                      and configuration (`context.config`).
        """
        super().__init__(**kwargs)
        # Instantiate the GoTReasoningAction.
        # The Action will use self.llm and self.config from the Role's context.
        self.got_action = GoTReasoningAction()

        # This role is designed to primarily use the GoTReasoningAction.
        # For simplicity in this example, `_act` directly invokes `self.got_action`.
        # In a more complex agent, `_think` would determine `self.rc.todo`.
        # If this role were to always use GoTReasoningAction when activated:
        # self.set_actions([self.got_action])
        # And `_think` would set `self.rc.todo = self.got_action`.


    async def _act(self) -> Message:
        """
        The primary action execution method for the GoTSolverRole.

        This method retrieves a problem description from the latest message in its
        memory. It then invokes the `GoTReasoningAction` with this problem and
        any relevant configurations (like `max_iterations`, `num_thoughts_to_generate`,
        custom prompt templates, etc.) sourced from `self.config`. The resulting
        solution from the GoT process is then wrapped in an AIMessage.

        Returns:
            An AIMessage containing the solution from the GoT process or an error/status message.
        """
        logger.info(f"{self._setting}: Starting GoT problem-solving action.")

        if not self.rc.memory or self.rc.memory.count() == 0:
            logger.warning(f"{self._setting}: No incoming message in memory to process for GoT.")
            return AIMessage(content="No problem received to solve.", role=self.profile, cause_by=self.got_action.name)

        latest_message = self.rc.memory.get_last()
        if not latest_message:
            logger.warning(f"{self._setting}: Memory count > 0 but get_last() returned None.")
            return AIMessage(content="Could not retrieve message from memory.", role=self.profile, cause_by=self.got_action.name)

        problem_description = latest_message.content
        if not problem_description:
            logger.warning(f"{self._setting}: Incoming message (ID: {latest_message.id}) has no content.")
            return AIMessage(content="Incoming message had no problem description.", role=self.profile, cause_by=self.got_action.name)

        logger.info(f"{self._setting}: Received problem for GoT: '{problem_description[:100]}...'")

        # Retrieve GoT-specific configurations from the Role's config object.
        # These will be passed to GoTReasoningAction.run(), which can further
        # override them if specific values are passed directly to its run method.
        # Here, we rely on GoTReasoningAction.run() to pick up its defaults from its
        # own self.config if these are not explicitly passed or are None.

        # Explicitly pass configurations from role's config to the action's run method.
        # This makes it clear how role-level config translates to action parameters.
        run_params = {
            "problem_description": problem_description,
            "max_iterations": self.config.get("got_max_iterations"),
            "num_thoughts_to_generate": self.config.get("got_num_thoughts_to_generate"),
            "generation_prompt_template": self.config.get("got_generation_prompt_template"),
            "generation_instruction_modifier": self.config.get("got_generation_instruction_modifier"),
            "evaluation_prompt_template": self.config.get("got_evaluation_prompt_template"),
            "evaluation_score_scale_description": self.config.get("got_evaluation_score_scale_description"),
            "evaluation_scoring_criteria": self.config.get("got_evaluation_scoring_criteria"),
            # Not passing initial_thoughts, generation_llm_config, evaluation_llm_config here
            # to let GoTReasoningAction use its defaults or its own config for these if not set at Role level.
        }
        # Filter out None values so that defaults in GoTReasoningAction.run signature apply if config not set
        run_params_cleaned = {k: v for k, v in run_params.items() if v is not None}

        logger.debug(f"{self._setting}: Invoking GoTReasoningAction with params: {run_params_cleaned}")

        try:
            solution = await self.got_action.run(**run_params_cleaned)
            logger.info(f"{self._setting}: GoT Action produced solution: '{solution[:100]}...'")
        except Exception as e:
            logger.error(f"{self._setting}: Error running GoTReasoningAction: {e}", exc_info=True)
            solution = f"An error occurred while trying to solve the problem using Graph of Thoughts: {str(e)}"

        response_message = AIMessage(
            content=solution,
            role=self.profile,
            sent_from=self.name,
            cause_by=self.got_action.name
        )
        return response_message

# Example of how this Role might be run (for testing purposes)
async def run_role_example():
    """
    An example function to demonstrate running the GoTSolverRole.
    This sets up a minimal context and message for the role to process.
    It uses a mock LLM for the GoT nodes for this test.
    """
    from metagpt.context import Context
    from metagpt.provider.base_llm import BaseLLM # For mock
    from metagpt.config2 import Config # For Role's config
    from unittest.mock import MagicMock
    import re # For mock LLM response parsing

    # 1. Create a problem message
    problem_content = "What are the main challenges in achieving AGI, and suggest three potential research directions?"
    problem_msg = Message(content=problem_content, role="User", sent_from="User")

    # 2. Initialize the Role with a context and LLM
    ctx = Context()

    # Mock LLM for the GoT nodes (passed via Action to LangGraph state)
    class MockGoTRoleLLM(BaseLLM):
        async def aask(self, prompt: str, system_msgs=None, format_msgs=None):
            logger.debug(f"MockGoTRoleLLM aask called with prompt (first 60 chars): '{prompt[:60].replace('\n', ' ')}...'")
            if "Generate" in prompt.splitlines()[0]:
                num_thoughts = 2
                try:
                    match = re.search(r"Generate (\d+) distinct thoughts", prompt)
                    if match: num_thoughts = int(match.group(1))
                except: pass
                thoughts_output = [f"{i+1}. Mock LLM thought {i+1} for '{problem_content[:20]}...' (Role Test)" for i in range(num_thoughts)]
                return "\n".join(thoughts_output)
            elif "Evaluate" in prompt.splitlines()[0]:
                thought_being_evaluated = "Unknown thought"
                match = re.search(r"Thought to Evaluate:\s*\"(.*?)\"", prompt, re.DOTALL)
                if match: thought_being_evaluated = match.group(1)
                score = len(thought_being_evaluated) / (len(thought_being_evaluated) + 70.0)
                score = min(max(score, 0.1), 0.98)
                return f"Score: {score:.2f}\nJustification: Mock LLM justification for '{thought_being_evaluated[:20]}...' (Role Test)."
            return "Default Mock LLM response (Role Test)."
        # Minimal BaseLLM abstract methods
        async def _achat_completion(self, messages: list[dict], options) -> dict: return {}
        async def _achat_completion_stream(self, messages: list[dict], options) -> None: pass

    mock_llm_instance = MockGoTRoleLLM()
    ctx.llm = mock_llm_instance # Set LLM in context, Role will pick it up

    # Configure the Role using a dictionary (simulating YAML loading)
    role_config_dict = {
        "name": "TestGoTRoleInstance", # Override default Role name for clarity
        "got_max_iterations": 2,
        "got_num_thoughts_to_generate": 3,
        "got_evaluation_scoring_criteria": "novelty and direct applicability (Role Test Config)"
    }
    # MetaGPT's config system loads YAML into a Config object.
    # Role's self.config will be this Config object.
    ctx.config = Config(config_dict=role_config_dict)

    # Instantiate the role, passing the context.
    # The Role's __init__ will set self.llm and self.config from the context.
    solver_role = GoTSolverRole(context=ctx, **role_config_dict) # Pass kwargs for name, profile, goal

    # 3. Simulate adding the message to the role's memory
    if not hasattr(solver_role.rc, 'memory') or solver_role.rc.memory is None:
        from metagpt.memory import Memory
        solver_role.rc.memory = Memory()
    solver_role.rc.memory.add(problem_msg)

    # 4. Execute the role's action logic
    logger.info(f"--- Running GoTSolverRole instance '{solver_role.name}' for problem: '{problem_content}' ---")
    response = await solver_role._act()

    print("\n--- GoTSolverRole Test Run Complete ---")
    print(f"Role: {solver_role.name}, Profile: {solver_role.profile}")
    print(f"Problem: {problem_content}")
    if response:
        print(f"GoTSolverRole Solution: {response.content}")
    else:
        print("GoTSolverRole did not return a response.")

if __name__ == '__main__':
    asyncio.run(run_role_example())
