import os
from metagpt.agent.reasoners.base_reasoner import BaseReasoner
from metagpt.agent.reasoners.got_prompter import GoTPrompter
from metagpt.agent.reasoners.got_parser import GoTParser

# Import necessary components from graph_of_thoughts
try:
    from graph_of_thoughts import Controller, GraphOfOperations
    from graph_of_thoughts.language_models import OpenAILanguageModel
    from graph_of_thoughts.operations import Generate, Score, Aggregate # Using Score instead of Evaluate
    # Import other necessary GoT components like Thought, State, etc. if directly used.
except ImportError as e:
    # Handle cases where graph_of_thoughts might not be installed yet or has different structure
    print(f"Error importing graph_of_thoughts: {e}. Ensure the library is installed and configured.")
    Controller = None
    GraphOfOperations = None
    OpenAILanguageModel = None
    Generate, Score, Aggregate = None, None, None # Adjusted imports


class GraphOfThoughtsReasoner(BaseReasoner):
    @staticmethod
    def _simple_scoring_function(thought_value: str) -> float:
        """A very basic scoring function."""
        if not isinstance(thought_value, str):
            return 0.0
        score = len(thought_value) / 100.0  # Example: score based on length
        if "solution" in thought_value.lower():
            score += 0.5
        return min(max(score, 0.0), 1.0) # Clamp score between 0.0 and 1.0

    def __init__(self, llm_config, agent_config):
        super().__init__(llm_config, agent_config)
        self.llm_config = llm_config
        self.agent_config = agent_config
        self.problem_description = None
        self.initial_context = None

        self.language_model = None
        self.prompter = None
        self.parser = None
        self.gop = None
        self.controller = None

        if not OpenAILanguageModel or not Controller or not GraphOfOperations or not Generate or not Score:
            print("Warning: Core graph_of_thoughts components not imported. Reasoner will not function.")
            return

        # 1. Instantiate GoTPrompter and GoTParser
        self.prompter = GoTPrompter()
        self.parser = GoTParser()

        # 2. Instantiate LanguageModel
        self.config_path = self._create_dummy_lm_config()
        if self.config_path:
            try:
                self.language_model = OpenAILanguageModel(config_path=self.config_path)
            except Exception as e:
                print(f"Error instantiating OpenAILanguageModel: {e}. Check config and API key.")
                self.language_model = None
        else:
            self.language_model = None
            print("Warning: Language model could not be instantiated due to missing config path.")

        if not self.language_model:
            print("Warning: LanguageModel not instantiated. Controller setup will be skipped.")
            return

        # 3. Create GraphOfOperations
        self.gop = GraphOfOperations()
        # Add Generate and Score operations
        if Generate and Score:
            self.gop.append_operation(Generate(self.language_model, self.prompter, self.parser))
            # Pass the static scoring function to the Score operation
            self.gop.append_operation(Score(scoring_function=self._simple_scoring_function))
        else:
            print("Warning: GoT Operations (Generate, Score) not available.")

        # 4. Instantiate Controller
        # The initial state for the controller. This is a very basic placeholder.
        # The actual structure will depend on how GoT defines states.
        initial_thought_state = {
            "original_problem": "", # Will be set in setup, used by prompter
            "current": "",  # Will be set in setup
            "method": "generate_score", # Example method
            "history": [],
            "result": None,
        }

        try:
            self.controller = Controller(
                lm=self.language_model,
                gop=self.gop,
                prompter=self.prompter,
                parser=self.parser,
                initial_state=initial_thought_state,
                max_thoughts=self.agent_config.get("max_thoughts", 10), # Example: get from agent_config
                max_depth=self.agent_config.get("max_depth", 5)      # Example: get from agent_config
            )
        except Exception as e:
            print(f"Error instantiating Controller: {e}")
            self.controller = None

    def _create_dummy_lm_config(self) -> str:
        """Creates a dummy openai_config.json for the GoT Language Model."""
        api_key = self.llm_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        model_name = self.llm_config.get("model", "gpt-3.5-turbo")

        if not api_key:
            print("Error: OpenAI API key not found in llm_config or environment variables.")
            return None

        config_content = {
            "api_key": api_key,
            "model_name": model_name,
            "temperature": self.llm_config.get("temperature", 0.7),
            "max_tokens": self.llm_config.get("max_tokens", 150)
        }

        config_dir = "./metagpt_got_configs" # Temp directory for config
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "openai_config.json")

        try:
            with open(config_path, 'w') as f:
                import json
                json.dump(config_content, f, indent=4)
            return config_path
        except Exception as e:
            print(f"Error creating dummy LM config file: {e}")
            return None


    def setup(self, problem_description: str, initial_context: dict):
        self.problem_description = problem_description
        self.initial_context = initial_context

        if not self.controller:
            print("Warning: Controller not initialized. Cannot setup GoTReasoner.")
            # raise NotImplementedError("Controller not initialized. Cannot setup GoTReasoner.")
            return

        # Prepare the initial thought state for the GoT Controller
        # This structure is an assumption and needs to match GoT's expected state format.
        current_thought_value = f"Problem: {problem_description}. Context: {initial_context}"
        initial_thought_state = {
            "original": problem_description,
            "current": current_thought_value,
            "method": self.agent_config.get("reasoning_method", "generate_evaluate"), # e.g., could be "dfs", "bfs"
            "history": [(None, current_thought_value)], # (operation, state_value)
            "result": None,
            # Add any other fields required by the specific GoT Controller implementation
        }

        # The GoT Controller might have a method to set or update the initial state
        # or it might need to be re-instantiated if the GOP or other critical parts change.
        # For now, let's assume we update the existing controller's state if possible,
        # or simply store it for when controller.run() is called.
        # If GoT's Controller needs re-init:
        # self.controller = Controller(...) with new initial_state

        # The provided GoT Controller seems to take initial_state at __init__.
        # If problem_description changes how GOP is structured, GOP might need re-init too.
        # For this iteration, we assume GOP is static after __init__.
        # We will update the controller's internal state if such a method exists,
        # or rely on run() taking the state. The current GoT Controller.run() takes `initial_thoughts`.
        # We will prepare this to be passed to run().

        # Storing for use in execute_reasoning_step, as controller.run() expects initial_thoughts
        self.controller.initial_state = initial_thought_state
        # print(f"GoT Reasoner setup with problem: {problem_description}")


    def execute_reasoning_step(self) -> dict:
        if not self.controller:
            # raise NotImplementedError("Controller not initialized. Cannot execute reasoning step.")
            print("Error: Controller not initialized in execute_reasoning_step.")
            return {"status": "error", "message": "Controller not initialized."}

        try:
            # The controller.run() method in the reference GoT implementation
            # takes `initial_thoughts` (a list of strings).
            # We'll use the 'current' value from our prepared initial_state.
            initial_thoughts_for_run = [self.controller.initial_state.get("current", self.problem_description)]

            # The `run` method executes the graph of operations.
            # We need to ensure the `gop` is configured with operations that make sense.
            # The `run` method itself might manage the loop of operations.
            print(f"GoT Reasoner executing step with initial thoughts: {initial_thoughts_for_run}")
            self.controller.run(initial_thoughts=initial_thoughts_for_run)

            # After run, results should be in controller.thoughts or a specific result attribute.
            # This is a placeholder for extracting actual results.
            # final_thoughts = self.controller.get_final_thoughts() # Assuming such a method
            # For now, let's return a generic status.
            return {"status": "success", "message": "GoT step executed.", "thoughts_count": len(self.controller.thoughts)}
        except Exception as e:
            print(f"Error during GoT controller.run(): {e}")
            return {"status": "error", "message": str(e)}


    def get_current_solution(self) -> any:
        if not self.controller or not self.controller.thoughts:
            # raise NotImplementedError("Controller not initialized or no thoughts available.")
            print("Warning: Controller not initialized or no thoughts available in get_current_solution.")
            return None

        # This depends heavily on how GoT's Controller stores results.
        # It might be the last thought, a thought marked as 'solution', or the highest-scored thought.
        # Assuming thoughts is a list of Thought objects, and each has a 'value' and 'score'.

        if not self.controller.thoughts:
            return None

        # Example: Return the value of the last thought generated, or highest scored one
        # This is a simplistic approach. A more robust way would be to find a thought
        # marked as 'solution' or with the highest score if evaluation is part of the GOP.
        try:
            # Assuming thoughts are stored in controller.thoughts
            # And thoughts are objects with 'value' and potentially 'score' attributes
            # Let's find the "best" thought. If scores exist, use that. Otherwise, the last one.

            best_thought = None
            if hasattr(self.controller.thoughts[0], 'score'): # Check if thoughts have scores
                # Sort by score descending if scores are present
                sorted_thoughts = sorted(self.controller.thoughts, key=lambda t: t.score, reverse=True)
                if sorted_thoughts:
                    best_thought = sorted_thoughts[0]
            else:
                # If no scores, take the last thought
                if self.controller.thoughts:
                    best_thought = self.controller.thoughts[-1]

            return best_thought.value if best_thought else None
        except AttributeError as e:
            print(f"Error accessing thought attributes: {e}. Thought structure might be different.")
            # Fallback: return the raw last thought if possible
            if self.controller.thoughts:
                return self.controller.thoughts[-1] # Could be a Thought object or just a string
            return None
        except Exception as e:
            print(f"Generic error in get_current_solution: {e}")
            return None


    def is_finished(self) -> bool:
        if not self.controller:
            # raise NotImplementedError("Controller not initialized. Cannot determine if finished.")
            print("Warning: Controller not initialized in is_finished.")
            return False # Or True, depending on desired behavior for uninitialized controller

        # Determine if the GoT process has reached a terminal state.
        # This could be based on:
        # 1. Max depth or max thoughts reached (controller might track this).
        # 2. A thought being marked as a final solution.
        # 3. No more operations can be applied.
        # For now, returning False as a placeholder.

        # Example based on GoT controller having properties like max_depth_reached or solution_found
        # if hasattr(self.controller, 'max_depth_reached') and self.controller.max_depth_reached():
        #     return True
        # if hasattr(self.controller, 'is_solution_found') and self.controller.is_solution_found():
        #     return True

        return False # Placeholder

# Example of how llm_config might look (to be passed from outside):
# llm_config_example = {
#     "api_key": "YOUR_OPENAI_API_KEY",
#     "model": "gpt-3.5-turbo" # Or any other model supported by OpenAILanguageModel
# }
# agent_config_example = {}

# To test instantiation (optional, can be removed)
# if __name__ == '__main__':
#     try:
#         reasoner = GraphOfThoughtsReasoner(llm_config=llm_config_example, agent_config=agent_config_example)
#         print("GraphOfThoughtsReasoner instantiated.")
#         reasoner.setup("Solve 2+2", {})
#     except NotImplementedError as e:
#         print(f"Caught expected error: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred during instantiation: {e}")

# Note: The actual usage of Controller, GraphOfOperations, and language models
# will depend on the specific API and design of the `graph-of-thoughts` library.
# The above implementation makes some assumptions and might need adjustments
# once the library is integrated and its usage patterns are clearer.
