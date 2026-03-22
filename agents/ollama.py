import ollama
from typing import Any
from agents.base import BaseLLM

def pull(choice: int) -> float:
    """
    Pulls an arm in the multi-armed bandit game.

    Args:
        choice: The index of the arm to pull. Either 0, 1, 2, or 3.
    Returns:
        The result of the pull.
    """
    return 0.0

def send_message(message: str) -> None:
    """
    Sends a message to the other agent.

    Args:
        message: The message to send.
    """
    pass

class Ollama(BaseLLM):
    model_dict: dict[str, str] = {
        model: model
        for model in [
            "codellama",
            "codegemma",
            "deepseek-coder-v2",
            "deepseek-r1",
            "deepseek-v3",
            "gemma",
            "gemma2",
            "gemma3",
            "gemma3n",
            "gpt-oss",
            "llama2",
            "llama3",
            "llama3.1",
            "llama3.2",
            "llama3.2-vision",
            "llama3.3",
            "llama4",
            "llava",
            "minicpm-v",
            "mistral",
            "mistral-nemo",
            "mistral-small",
            "mistral-small3.2",
            "mixtral",
            "nomic-embed-text",
            "mxbai-embed-large",
            "phi3",
            "phi4",
            "phi4-reasoning",
            "qwen2.5",
            "qwen2.5vl",
            "qwen3",
            "qwen3-coder",
            "qwen3-vl",
            "qwen3.5",
            "starcoder2",
        ]
    }
    good_tools = [pull]
    bad_tools = [send_message]

    @classmethod
    def contains_model(cls, model: str) -> bool:
        # Allow explicit tag selection (e.g., "llama3.2:3b", "qwen3:latest")
        # even if a tag is not enumerated in model_dict.
        return model in cls.model_dict or ":" in model

    @classmethod
    def get_model_id(cls, model: str) -> str:
        return cls.model_dict.get(model, model)

    @classmethod
    def query(cls, conversation: list[dict[str, str]], model: str, tools: list[Any]) -> dict[str, Any]:
        response = ollama.chat(
            model=cls.get_model_id(model),
            messages=conversation,
            tools=tools
        )

        llm_response = response.message.content or ""
        tool_call = None

        if hasattr(response.message, "tool_calls") and response.message.tool_calls:
            tc = response.message.tool_calls[0]
            tool_call = {
                "name": tc.function.name,
                "arguments": tc.function.arguments
            }

        return {
            "llm_response": llm_response,
            "tool_call": tool_call
        }
