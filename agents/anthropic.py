import anthropic
import os
import dotenv
from typing import Any
from agents.base import BaseLLM
from config import MAX_TOKENS

dotenv.load_dotenv()
API_KEY = os.environ.get("CLAUDE_API_KEY")

if not API_KEY:
    raise RuntimeError("Please set the CLAUDE_API_KEY environment variable.")

class Anthropic(BaseLLM):
    model_dict: dict[str, str] = {
        "4.5-sonnet": "claude-sonnet-4-5",
        "4.5-opus": "claude-opus-4-5",
        "4.5-haiku": "claude-haiku-4-5",
        "4.1-opus": "claude-opus-4-1",
        "4-sonnet": "claude-sonnet-4-0",
        "4-opus": "claude-opus-4-0",
        "3.7-sonnet": "claude-3-7-sonnet-latest",
        "3-haiku": "claude-3-haiku-20240307",
    }
    client: anthropic.Anthropic = anthropic.Anthropic(api_key=API_KEY)
    tools: list[dict[str, Any]] = [
        {
            "name": "pull",
            "description": "Pulls an arm in the multi-armed bandit game.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "choice": {
                        "type": "integer",
                        "description": "The index of the arm to pull. Either 0, 1, 2, or 3.",
                    },
                },
                "required": ["choice"],
            },
        }
    ]

    @classmethod
    def query(cls, conversation: list[dict[str, str]], model: str, include_tools: bool = True) -> dict[str, str]:
        system_parts, payload, response = [], [], None
        for msg in conversation:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                payload.append({"role": msg["role"], "content": msg["content"]})
        
        if len(system_parts) > 0:
            response = cls.get_client().messages.create(
                model=cls.get_model_id(model),
                max_tokens=MAX_TOKENS,
                system="\n\n".join(system_parts) if system_parts else None,
                messages=payload,
                tools=cls.tools if include_tools else [],
            )
        else:
            response = cls.get_client().messages.create(
                model=cls.get_model_id(model),
                max_tokens=MAX_TOKENS,
                messages=payload,
                tools=cls.tools if include_tools else [],
            )

        llm_response = ""
        arm_pulled = None
        reward = None

        for block in response.content:
            if block.type == "text":
                llm_response += block.text
            elif block.type == "tool_use":
                arm_pulled = block.input["choice"]
                reward = cls.pull(arm_pulled)

        return {
            "llm_response": llm_response,
            "arm_pulled": arm_pulled,
            "reward": reward
        }