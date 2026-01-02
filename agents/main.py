from agents.anthropic import Anthropic
from agents.ollama import Ollama
from agents.base import BaseLLM
from typing import Any

clients: list[BaseLLM] = [Anthropic, Ollama]

def call_agent(conversation: list[dict[str, str]], model: str, include_tools: bool = True) -> dict[str, Any]:
    for client in clients:
        if client.contains_model(model):
            return client.query(conversation, model, include_tools)
    raise ValueError(f"Model {model} not found")