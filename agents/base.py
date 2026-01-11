from typing import Any

class BaseLLM:
    model_dict: dict[str, str] = {}
    client: Any = None
    good_tools: list[Any] = []
    bad_tools: list[Any] = []

    @classmethod
    def get_model_dict(cls) -> dict[str, str]:
        return cls.model_dict
    
    @classmethod
    def get_client(cls) -> Any:
        return cls.client
    
    @classmethod
    def get_tools(cls, good: bool) -> list[Any]:
        return cls.good_tools if good else cls.bad_tools

    @classmethod
    def get_model_id(cls, model: str) -> str:
        return cls.model_dict[model]
    
    @classmethod
    def contains_model(cls, model: str) -> bool:
        return model in cls.model_dict
    
    @classmethod
    def query(cls, conversation: list[dict[str, str]], model: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Query the LLM with a conversation and tools.

        Args:
            conversation: List of message dicts with 'role' and 'content' keys.
            model: Model identifier string.
            tools: List of tool definitions.

        Returns:
            {
                "llm_response": <response text>,
                "tool_call": {"name": <tool_name>, "arguments": {...}} or None
            }
        """
        raise NotImplementedError
