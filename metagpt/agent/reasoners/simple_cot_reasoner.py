import asyncio
from metagpt.agent.reasoners.base_reasoner import BaseReasoner
from metagpt.provider.base_llm import BaseLLM
from metagpt.schema import AIMessage # Assuming AIMessage can be used for LLM interaction result

class SimpleCotReasoner(BaseReasoner):
    def __init__(self, llm: BaseLLM, agent_config=None):
        """
        :param llm: The language model instance to use for reasoning.
        :param agent_config: Optional configuration for the agent/reasoner.
        """
        super().__init__(llm_config=None, agent_config=agent_config) # llm_config is not directly used here as llm is passed
        self.llm: BaseLLM = llm # Store the passed LLM instance
        self.problem_description: str = ""
        self.solution: str = ""
        self.finished: bool = False

        if not self.llm:
            raise ValueError("SimpleCotReasoner requires an LLM instance.")

    def setup(self, problem_description: str, initial_context: dict = None):
        """
        Set up the reasoner with the problem description and any initial context.
        :param problem_description: The problem to be solved.
        :param initial_context: Optional dictionary containing initial context.
        """
        self.problem_description = problem_description
        self.initial_context = initial_context if initial_context else {}
        self.solution = ""
        self.finished = False

    async def execute_reasoning_step(self) -> dict:
        """
        Execute a single step of CoT reasoning.
        For SimpleCoT, this will be the only step.
        :return: A dictionary containing the thought process or result.
        """
        if not self.problem_description:
            return {"status": "error", "message": "Problem not set up."}

        if self.finished:
            return {"status": "already_finished", "solution": self.solution}

        # Basic CoT: prepend "Think step-by-step:"
        # More advanced CoT might involve multiple prompts or self-correction.
        prompt = f"Problem: {self.problem_description}\n\nLet's think step by step to arrive at a solution."

        # Add initial context to prompt if available
        if self.initial_context:
            prompt += "\n\nInitial Context:\n"
            for key, value in self.initial_context.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\nNow, let's think step by step:"


        # Use the llm instance passed during __init__
        try:
            # Assuming the llm has an aask method or similar for prompting
            # The specific method might vary (e.g., aask, acompletion)
            # For simplicity, using aask which is common in MetaGPT.
            response = await self.llm.aask(prompt) # Simple ask

            # In a more complex CoT, this might involve parsing the LLM's thought process
            # and then deriving a final answer. For SimpleCoT, the response is the solution.
            self.solution = response # The raw response is considered the solution
            self.finished = True
            return {"status": "success", "thought_process": response, "solution": self.solution}
        except Exception as e:
            return {"status": "error", "message": f"LLM call failed: {str(e)}"}

    async def get_current_solution(self) -> any:
        """
        Get the current best solution.
        :return: The solution string, or None if not yet available.
        """
        return self.solution if self.finished else None

    def is_finished(self) -> bool:
        """
        Check if the reasoning process is finished.
        For SimpleCoT, it's finished after one step.
        :return: True if finished, False otherwise.
        """
        return self.finished

# Example Usage (for testing, can be removed or commented out)
# async def main():
#     from metagpt.llm import LLM # Assuming LLM can be imported and configured
#     # Configure LLM (ensure API keys are set in environment or via config)
#     llm_instance = LLM()
#
#     cot_reasoner = SimpleCotReasoner(llm=llm_instance)
#     cot_reasoner.setup(problem_description="If I have 3 apples and eat 1, how many are left?",
#                        initial_context={"scenario": "A simple math question."})
#
#     step_result = await cot_reasoner.execute_reasoning_step()
#     print("Step Result:", step_result)
#
#     solution = await cot_reasoner.get_current_solution()
#     print("Solution:", solution)
#
#     print("Is Finished:", cot_reasoner.is_finished())
#
# if __name__ == '__main__':
#    asyncio.run(main())
