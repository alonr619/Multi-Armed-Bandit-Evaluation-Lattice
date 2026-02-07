from agents.openai_compatible import OpenAICompatible
from agents.tools import OPENAI_BAD_TOOLS, OPENAI_GOOD_TOOLS


class Gemini(OpenAICompatible):
    api_key_env_var = "GEMINI_API_KEY"
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    provider_name = "Gemini"
    model_dict: dict[str, str] = {
        "gemini-3-pro-preview": "gemini-3-pro-preview",
        "gemini-3-flash-preview": "gemini-3-flash-preview",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash-preview": "gemini-2.5-flash-preview-09-2025",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.5-flash-lite-preview": "gemini-2.5-flash-lite-preview-09-2025",
        "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
        "gemini-2.0-flash": "gemini-2.0-flash",
        "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
    }
    good_tools = OPENAI_GOOD_TOOLS
    bad_tools = OPENAI_BAD_TOOLS
