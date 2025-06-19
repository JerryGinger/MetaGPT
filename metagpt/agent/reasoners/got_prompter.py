# Assuming graph_of_thoughts.prompter.Prompter exists and can be a base class
# If not, GoTPrompter will be a standalone class.
try:
    from graph_of_thoughts.prompter import Prompter
except ImportError:
    Prompter = object # Fallback if Prompter base class is not found

class GoTPrompter(Prompter):
    def __init__(self, prompter_config=None):
        """
        Initializes the GoTPrompter.
        :param prompter_config: Optional dictionary containing configuration for the prompter.
        """
        super().__init__() # Call to base class constructor if it exists and needs it
        self.prompter_config = prompter_config if prompter_config else {}

    # The GoT library's Prompter class methods often take `state: State` as the first argument.
    # We'll adapt to a dictionary `state_dict` for now, assuming it contains necessary info.
    # The actual State object might have specific attributes we need to access.

    def generate_initial_prompt(self, state_dict: dict) -> str:
        """
        Generates the initial prompt to start the thought generation process.
        :param state_dict: A dictionary representing the current state,
                           expected to contain 'original_problem'.
        :return: A string representing the initial prompt.
        """
        problem = state_dict.get('original_problem', state_dict.get('original', 'No problem specified'))
        # initial_context = state_dict.get('initial_context', {}) # If needed

        # prompt = f"Problem: {problem}\n"
        # if initial_context:
        #     prompt += "Initial Context:\n"
        #     for key, value in initial_context.items():
        #         prompt += f"- {key}: {value}\n"
        # prompt += "\nPlease generate initial thoughts or potential approaches to solve this problem."
        return f"Given the problem: {problem}, generate a possible solution."

    def generate_evaluation_prompt(self, state_dict: dict, thoughts: list[dict]) -> str:
        """
        Generates a prompt to evaluate a list of thoughts based on the current state.
        :param state_dict: A dictionary representing the current state.
        :param thoughts: A list of thought dictionaries to be evaluated. Each thought dict is expected to have a 'value'.
        :return: A string representing the evaluation prompt.
        """
        # current_state_description = state_dict.get('current', 'Current situation') # Example

        if not thoughts:
            return "No thoughts provided for evaluation."

        # Assuming thoughts is a list of dictionaries, and each has a 'value' key
        thought_values = []
        for thought in thoughts:
            if isinstance(thought, dict) and 'value' in thought:
                thought_values.append(str(thought['value']))
            elif isinstance(thought, str): # If it's just a list of strings
                thought_values.append(thought)
            else:
                thought_values.append("Invalid thought format")

        # prompt = f"Current State: {current_state_description}\n"
        # prompt += "Please evaluate the following thoughts. For each thought, provide a score (e.g., 0.0 to 1.0) and a brief justification.\n"
        # for i, value in enumerate(thought_values):
        #     prompt += f"\nThought {i+1}: {value}\n"
        # prompt += "\nEvaluation Format: For each thought, provide 'Score: [score], Justification: [text]'"

        # Note: If using a programmatic scoring function, this prompt might not be directly used by the Score operation.
        # It's kept for completeness or for other evaluation strategies.
        return f"Evaluate the following solution(s): {'; '.join(thought_values)}. Score it from 0.0 to 1.0."


    def generate_aggregate_prompt(self, state_dict: dict, thoughts: list[dict]) -> str:
        """
        Generates a prompt to aggregate or synthesize information from a list of thoughts.
        :param current_state_description: A description of the current state of reasoning.
        :param thoughts_to_aggregate: A list of thoughts (strings) to be aggregated.
        :return: A string representing the aggregation prompt.
        """
        # current_state_description = state_dict.get('current', 'Current situation')
        if not thoughts:
            return "No thoughts provided for aggregation."

        thought_values = []
        for thought in thoughts: # Changed from thoughts_to_aggregate to thoughts
            if isinstance(thought, dict) and 'value' in thought:
                thought_values.append(str(thought['value']))
            elif isinstance(thought, str):
                thought_values.append(thought)
            else:
                thought_values.append("Invalid thought format")

        # prompt = f"Current State: {current_state_description}\n"
        # prompt += "Please synthesize the following thoughts into a more refined thought or a summary:\n"
        # for i, value in enumerate(thought_values):
        #     prompt += f"\nThought {i+1}: {value}\n"
        # prompt += "\nProvide the synthesized thought."
        return "Aggregate these thoughts: " + ", ".join(thought_values) # Simplified

    def generate_next_step_prompt(self, state_dict: dict, available_operations: list[str]) -> str:
        """
        Generates a prompt to decide the next operation or step in the reasoning process.
        :param current_state_description: A description of the current state of reasoning.
        :param available_operations: A list of possible operations that can be performed.
        :return: A string representing the next step prompt.
        """
        # current_state_description = state_dict.get('current', 'Current situation')
        # prompt = f"Current State: {current_state_description}\n"
        # prompt += "Given the current state, what is the most logical next step or operation to perform?\n"
        # if available_operations:
        #     prompt += "Available operations:\n"
        #     for op in available_operations:
        #         prompt += f"- {op}\n"
        # prompt += "Please suggest the next operation and any parameters if needed."
        return "What is the next step?" # Simplified

# Example usage (optional, can be removed)
# if __name__ == '__main__':
#     prompter = GoTPrompter()
#     sample_state = {"original_problem": "Solve 2x+5=11", "current": "Initial state"}
#     initial_prompt = prompter.generate_initial_prompt(sample_state)
#     print("Initial Prompt:\n", initial_prompt)

#     sample_thoughts_for_eval = [{"value": "Let x = 3."},{"value": "2*3 + 5 = 11, which is 6 + 5 = 11. This is correct."}]
#     evaluation_prompt = prompter.generate_evaluation_prompt(sample_state, sample_thoughts_for_eval)
#     print("\nEvaluation Prompt:\n", evaluation_prompt)

#     sample_thoughts_for_agg = [{"value": "Thought A"}, {"value": "Thought B"}]
#     aggregate_prompt = prompter.generate_aggregate_prompt(sample_state, sample_thoughts_for_agg)
#     print("\nAggregate Prompt:\n", aggregate_prompt)

#     next_step_prompt = prompter.generate_next_step_prompt(sample_state, ["Generate", "Score", "Aggregate"])
#     print("\nNext Step Prompt:\n", next_step_prompt)
