from agents.openai_compatible import OpenAICompatible
from agents.tools import OPENAI_BAD_TOOLS, OPENAI_GOOD_TOOLS


class OpenAI(OpenAICompatible):
    api_key_env_var = "OPENAI_API_KEY"
    provider_name = "OpenAI"
    token_limit_param = "max_completion_tokens"
    model_dict: dict[str, str] = {
        "openai-5.1": "gpt-5.1",
        "openai-5": "gpt-5",
        "openai-5-mini": "gpt-5-mini",
        "openai-5-nano": "gpt-5-nano",
        "openai-5-codex": "gpt-5-codex",
        "openai-4o": "gpt-4o",
        "openai-4o-mini": "gpt-4o-mini",
        "openai-4.1": "gpt-4.1",
        "openai-4.1-mini": "gpt-4.1-mini",
        "openai-4.1-nano": "gpt-4.1-nano",
        "openai-o3": "o3",
        "openai-o3-mini": "o3-mini",
        "openai-o4-mini": "o4-mini",
    }
    good_tools = OPENAI_GOOD_TOOLS
    bad_tools = OPENAI_BAD_TOOLS
