import re

# Assuming graph_of_thoughts.parser.Parser exists and can be a base class
# If not, GoTParser will be a standalone class.
try:
    from graph_of_thoughts.parser import Parser
except ImportError:
    Parser = object  # Fallback if Parser base class is not found

class GoTParser(Parser):
    def __init__(self, parser_config=None):
        """
        Initializes the GoTParser.
        :param parser_config: Optional dictionary containing configuration for the parser.
        """
        super().__init__()  # Call to base class constructor if it exists and needs it
        self.parser_config = parser_config if parser_config else {}

    # The GoT library's Parser class methods often take `state: State` as the second argument.
    # We'll adapt to a dictionary `state_dict` for now.

    def parse_generation_output(self, llm_output: str, state_dict: dict) -> list[dict]:
        """
        Parses the LLM output for generated thoughts.
        :param llm_output: The raw output string from the language model.
        :param state_dict: A dictionary describing the current state.
        :return: A list of thought dictionaries, e.g., [{'value': 'thought text'}].
        """
        # This is a basic placeholder implementation.
        # It assumes the entire llm_output is a single thought.
        # More sophisticated parsing might split the output or extract specific parts.

        # For GoT, thoughts are often dictionaries. The minimal is usually {'value': <thought_string>}.
        # It can also include 'score', 'id', 'children', 'parents', etc.
        # The Generate operation typically produces the 'value'.

        cleaned_output = llm_output.strip()
        if not cleaned_output:
            return [] # Return empty list if output is empty

        # Return a list containing a single thought dictionary
        return [{'value': cleaned_output}]


    def parse_evaluation_output(self, llm_output: str, state_dict: dict) -> list[float]:
        """
        Parses the LLM output for thought evaluations (scores).
        :param llm_output: The raw output string from the language model.
        :param state_dict: A dictionary describing the current state.
        :return: A list of floats, where each float is a score for a thought.
                 The order should correspond to the thoughts that were evaluated.
        """
        # This method is primarily for when an LLM generates the scores.
        # If a programmatic scoring_function is used with the Score operation,
        # this parser method might not be directly invoked by that operation.
        # However, it's good practice to have a basic implementation.

        # Assuming the LLM output is a single number (score) or a list of numbers.
        # For simplicity, let's try to parse one float.
        try:
            # If multiple scores are expected, LLM output format and parsing here would be more complex.
            # E.g., "0.8" or "Score: 0.8"
            # A simple extraction of the first float found.
            match = re.search(r"([0-9.]+)", llm_output)
            if match:
                return [float(match.group(1))]
            else:
                # Fallback if no number is found
                print(f"Warning: Could not parse score from LLM output: '{llm_output}'")
                return [0.0]
        except ValueError:
            print(f"Warning: ValueError parsing score from LLM output: '{llm_output}'")
            return [0.0] # Default score in case of parsing error


    def parse_synthesis_output(self, llm_output: str, state_dict: dict) -> str:
        """
        Parses the LLM output for a synthesized thought.
        :param llm_output: The raw output string from the language model.
        :param state_dict: A dictionary describing the current state.
        :return: A string representing the synthesized thought.
        """
        # Assuming the LLM directly outputs the synthesized thought.
        # May require cleaning or stripping.
        return llm_output.strip() # Return the raw output as the synthesized thought value

    def parse_next_step_output(self, llm_output: str, state_dict: dict, available_operations: list[str]) -> dict:
        """
        Parses the LLM output for the suggested next step/operation.
        :param llm_output: The raw output string from the language model.
        :param state_dict: A dictionary describing the current state.
        :param available_operations: List of valid operations for context.
        :return: A dictionary like {'operation_name': 'generate', 'parameters': {...}}
        """
        # This is highly dependent on how the LLM is prompted to suggest next steps.
        # For example, LLM might output: "Next Operation: generate_thoughts"
        # Or "Operation: evaluate_thought, Thought ID: 3"

        cleaned_output = llm_output.lower().strip()

        for op in available_operations:
            if op.lower() in cleaned_output: # Simple check if operation name is in output
                # This is very basic. Parameters would need more sophisticated parsing.
                return {"operation_name": op, "parameters": {}}

        # Fallback if no operation is clearly identified
        # Default to the first available operation or 'generate' if none specified
        default_op = available_operations[0] if available_operations else "generate"
        print(f"Warning: Could not parse next step from LLM output: '{llm_output}'. Defaulting to '{default_op}'.")
        return {"operation_name": default_op, "parameters": {}, "raw_output": llm_output}

# Example usage (optional, can be removed)
# if __name__ == '__main__':
#     parser = GoTParser()
#     sample_state = {} # Dummy state
#
#     # Test parse_generation_output
#     gen_llm_output = "This is a generated solution to the problem."
#     generated_thoughts_list = parser.parse_generation_output(gen_llm_output, sample_state)
#     print("Parsed Generation Output:\n", generated_thoughts_list) # Expected: [{'value': 'This is a generated solution to the problem.'}]
#
#     # Test parse_evaluation_output (if LLM provides score)
#     eval_llm_output = "The score for this thought is 0.75."
#     parsed_scores = parser.parse_evaluation_output(eval_llm_output, sample_state)
#     print("\nParsed Evaluation Output (Scores):\n", parsed_scores) # Expected: [0.75]

#     eval_llm_output_no_score = "This thought is good."
#     parsed_scores_no_score = parser.parse_evaluation_output(eval_llm_output_no_score, sample_state)
#     print("\nParsed Evaluation Output (No Score):\n", parsed_scores_no_score) # Expected: [0.0]

#     # Test parse_synthesis_output
#     synth_llm_output = "  This is the synthesized thought, combining previous ideas.  "
#     synthesized_thought_value = parser.parse_synthesis_output(synth_llm_output, sample_state)
#     print("\nParsed Synthesis Output:\n", synthesized_thought_value)

#     # Test parse_next_step_output
#     next_step_llm_output = "I suggest we use the Aggregate operation next."
#     next_step_action = parser.parse_next_step_output(next_step_llm_output, sample_state, ["Generate", "Score", "Aggregate"])
#     print("\nParsed Next Step Output:\n", next_step_action)
#
#     next_step_llm_output_fallback = "Hmm, what to do..."
#     next_step_action_fallback = parser.parse_next_step_output(next_step_llm_output_fallback, sample_state, ["Generate", "Score", "Aggregate"])
#     print("\nParsed Next Step Output (Fallback):\n", next_step_action_fallback)
