import anthropic
import os
import dotenv
from typing import Any
from agents.base import BaseLLM
from config import MAX_TOKENS
from agents.tools import ANTHROPIC_GOOD_TOOLS, ANTHROPIC_BAD_TOOLS

dotenv.load_dotenv()

class Anthropic(BaseLLM):
    model_dict: dict[str, str] = {
        "4.6-opus": "claude-opus-4-6",
        "4.5-sonnet": "claude-sonnet-4-5",
        "4.5-haiku": "claude-haiku-4-5",
        "4.5-opus": "claude-opus-4-6",
        "4.1-opus": "claude-opus-4-1",
        "4-sonnet": "claude-sonnet-4-0",
        "4-opus": "claude-opus-4-0",
        "3.7-sonnet": "claude-3-7-sonnet-latest",
        "3-haiku": "claude-3-haiku-20240307",
    }
    client: anthropic.Anthropic | None = None
    good_tools: list[dict[str, Any]] = ANTHROPIC_GOOD_TOOLS
    bad_tools: list[dict[str, Any]] = ANTHROPIC_BAD_TOOLS

    @classmethod
    def get_client(cls) -> anthropic.Anthropic:
        if cls.client is not None:
            return cls.client

        api_key = os.environ.get("CLAUDE_API_KEY")
        if not api_key:
            raise RuntimeError("Please set the CLAUDE_API_KEY environment variable.")

        cls.client = anthropic.Anthropic(api_key=api_key)
        return cls.client

    @classmethod
    def query(cls, conversation: list[dict[str, str]], model: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
        system_parts, payload = [], []
        for msg in conversation:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                payload.append({"role": msg["role"], "content": msg["content"]})
        
        kwargs = {
            "model": cls.get_model_id(model),
            "max_tokens": MAX_TOKENS,
            "messages": payload,
            "tools": tools,
        }
        if system_parts:
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": "\n\n".join(system_parts),
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        
        response = cls.get_client().messages.create(**kwargs)

        llm_response = ""
        tool_call = None

        for block in response.content:
            if block.type == "text":
                llm_response += block.text
            elif block.type == "tool_use":
                tool_call = {
                    "name": block.name,
                    "arguments": block.input
                }

        usage = response.usage
        cache_creation_input_tokens = getattr(usage, "cache_creation_input_tokens", None)
        cache_read_input_tokens = getattr(usage, "cache_read_input_tokens", None)
        cache_discount_available = (
            cache_creation_input_tokens is not None or cache_read_input_tokens is not None
        )

        return {
            "llm_response": llm_response,
            "tool_call": tool_call,
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "cache_creation_input_tokens": cache_creation_input_tokens,
                "cache_read_input_tokens": cache_read_input_tokens,
            },
            "cache_discount_available": cache_discount_available,
            "cache_discount_note": (
                None
                if cache_discount_available
                else "Cached token discount reporting is not available for this Anthropic response."
            ),
        }
