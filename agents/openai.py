from agents.openai_compatible import OpenAICompatible
from agents.tools import OPENAI_BAD_TOOLS, OPENAI_GOOD_TOOLS


class OpenAI(OpenAICompatible):
    api_key_env_var = "OPENAI_API_KEY"
    provider_name = "OpenAI"
    token_limit_param = "max_completion_tokens"
    model_dict: dict[str, str] = {
        model: model
        for model in [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-instruct",
            "gpt-3.5-turbo-instruct-0914",
            "gpt-4",
            "gpt-4-0125-preview",
            "gpt-4-0613",
            "gpt-4-1106-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo-preview",
            "gpt-4.1",
            "gpt-4.1-2025-04-14",
            "gpt-4.1-mini",
            "gpt-4.1-mini-2025-04-14",
            "gpt-4.1-nano",
            "gpt-4.1-nano-2025-04-14",
            "gpt-4o",
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-11-20",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-5",
            "gpt-5-2025-08-07",
            "gpt-5-chat-latest",
            "gpt-5-codex",
            "gpt-5-mini",
            "gpt-5-mini-2025-08-07",
            "gpt-5-nano",
            "gpt-5-nano-2025-08-07",
            "gpt-5-pro",
            "gpt-5-pro-2025-10-06",
            "gpt-5.1",
            "gpt-5.1-2025-11-13",
            "gpt-5.1-chat-latest",
            "gpt-5.1-codex",
            "gpt-5.1-codex-max",
            "gpt-5.1-codex-mini",
            "gpt-5.2",
            "gpt-5.2-2025-12-11",
            "gpt-5.2-chat-latest",
            "gpt-5.2-codex",
            "gpt-5.2-pro",
            "gpt-5.2-pro-2025-12-11",
            "gpt-5.3-chat-latest",
            "gpt-5.3-codex",
            "gpt-5.4",
            "gpt-5.4-2026-03-05",
            "gpt-5.4-mini",
            "gpt-5.4-mini-2026-03-17",
            "gpt-5.4-nano",
            "gpt-5.4-nano-2026-03-17",
            "gpt-5.4-pro",
            "gpt-5.4-pro-2026-03-05",
            "o1",
            "o1-2024-12-17",
            "o1-pro",
            "o1-pro-2025-03-19",
            "o3",
            "o3-2025-04-16",
            "o3-mini",
            "o3-mini-2025-01-31",
            "o4-mini",
            "o4-mini-2025-04-16",
        ]
    }
    reasoning_effort_alias_map: dict[str, str] = {
        "gpt-5.4-reasoning-none": "none",
        "gpt-5.4-reasoning-low": "low",
        "gpt-5.4-reasoning-medium": "medium",
        "gpt-5.4-reasoning-high": "high",
    }
    model_dict.update({alias: "gpt-5.4" for alias in reasoning_effort_alias_map})

    @classmethod
    def get_reasoning_effort_for_alias(cls, model: str) -> str | None:
        return cls.reasoning_effort_alias_map.get(model)

    good_tools = OPENAI_GOOD_TOOLS
    bad_tools = OPENAI_BAD_TOOLS
