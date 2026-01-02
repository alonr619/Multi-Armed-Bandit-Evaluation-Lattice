import ollama
from agents.base import BaseLLM

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

    @classmethod
    def query(cls, conversation: list[dict[str, str]], model: str, include_tools: bool = True) -> dict[str, str]:
        response = ollama.chat(
            model=cls.get_model_id(model),
            messages=conversation,
            tools=[cls.pull] if include_tools else None
        )

        print(response)

        if "tool_calls" in response.message:
            choice = response.message.tool_calls[0].function.arguments["choice"]
            return {
                "llm_response": response.message.content, 
                "arm_pulled": choice,
                "reward": cls.pull(int(choice)),
            }
        
        return {
            "llm_response": response.message.content, 
            "arm_pulled": None,
            "reward": None
        }