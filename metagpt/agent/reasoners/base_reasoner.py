import abc

class BaseReasoner(abc.ABC):
    @abc.abstractmethod
    def __init__(self, llm_config, agent_config):
        pass

    @abc.abstractmethod
    def setup(self, problem_description: str, initial_context: dict):
        pass

    @abc.abstractmethod
    def execute_reasoning_step(self) -> dict:
        pass

    @abc.abstractmethod
    def get_current_solution(self) -> any:
        pass

    @abc.abstractmethod
    def is_finished(self) -> bool:
        pass
