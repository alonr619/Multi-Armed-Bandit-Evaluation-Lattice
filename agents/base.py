from typing import Any
from bandit import n_armed_bandit

class BaseLLM:
    model_dict: dict[str, str] = {}
    client: Any = None
    tools: Any = []

    @classmethod
    def get_model_dict(cls) -> dict[str, str]:
        return cls.model_dict
    
    @classmethod
    def get_client(cls) -> Any:
        return cls.client
    
    @classmethod
    def get_tools(cls) -> Any:
        return cls.tools

    @classmethod
    def get_model_id(cls, model: str) -> str:
        return cls.model_dict[model]
    
    @classmethod
    def contains_model(cls, model: str) -> bool:
        return model in cls.model_dict
    
    @classmethod
    def pull(cls, choice: int) -> float:
        """
        Pulls an arm in the multi-armed bandit game.

        Args:
            choice: The index of the arm to pull. Either 0, 1, 2, or 3.
        
        Returns:
            The reward for the pulled arm.
        """
        return n_armed_bandit(choice)
    
    @classmethod
    def query(cls, conversation: list[dict[str, str]], model: str, include_tools: bool = True) -> dict[str, Any]:
        """
        Format of output should be:
        {
            "llm_response": <response>,
            "arm_pulled": <arm_pulled>,
            "reward": <reward>
        }
        """
        pass