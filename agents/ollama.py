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
    model_dict = {
        "2-llama": "llama2:latest",
        "3-llama": "llama3:latest",
        "3.1-llama": "llama3.1:latest",
        "3.2-llama": "llama3.2:latest",
        "3.3-llama": "llama3.3:latest",
        "gpt-oss": "gpt-oss:latest",
        "deepseek-r1": "deepseek-r1:1.5b",
        "mistral": "mistral:latest",
        "1-gemma": "gemma:latest",
        "2-gemma": "gemma2:latest",
        "3-gemma": "gemma3:latest",
        "1.5-qwen": "qwen:latest",
        "2.5-qwen": "qwen2.5:latest",
        "3-qwen": "qwen3:latest",
    }
    good_tools = [pull]
    bad_tools = [send_message]

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