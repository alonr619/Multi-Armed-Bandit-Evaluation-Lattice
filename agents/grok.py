from agents.openai_compatible import OpenAICompatible
from agents.tools import OPENAI_BAD_TOOLS, OPENAI_GOOD_TOOLS


class Grok(OpenAICompatible):
    api_key_env_var = "XAI_API_KEY"
    base_url = "https://api.x.ai/v1"
    provider_name = "Grok/xAI"
    model_dict: dict[str, str] = {
        "grok-4-latest": "grok-4-latest",
        "grok-4": "grok-4",
        "grok-4-0709": "grok-4-0709",
        "grok-4-fast-reasoning": "grok-4-fast-reasoning",
        "grok-4-1-fast-reasoning": "grok-4-1-fast-reasoning",
        "grok-3-latest": "grok-3-latest",
        "grok-3": "grok-3",
        "grok-3-fast": "grok-3-fast",
        "grok-3-fast-latest": "grok-3-fast-latest",
        "grok-3-mini": "grok-3-mini",
        "grok-3-mini-latest": "grok-3-mini-latest",
    }
    good_tools = OPENAI_GOOD_TOOLS
    bad_tools = OPENAI_BAD_TOOLS
