import uuid
import os

from agents.openai_compatible import OpenAICompatible
from agents.tools import OPENAI_BAD_TOOLS, OPENAI_GOOD_TOOLS


class Grok(OpenAICompatible):
    api_key_env_var = "XAI_API_KEY"
    base_url = "https://api.x.ai/v1"
    provider_name = "Grok/xAI"
    # Stable per-session ID that tells xAI to cache under one conversation.
    # A new UUID is generated each time a fresh client is created (i.e. once
    # per process), so separate evaluation runs get independent cache scopes.
    _conv_id: str = str(uuid.uuid4())
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

    @classmethod
    def get_client(cls):
        if cls.client is not None:
            return cls.client
        # Call super() to benefit from the import-error check in OpenAICompatible.
        # We can't use super().get_client() because we need to pass extra kwargs,
        # so we replicate the minimal setup here.
        from agents.openai_compatible import OPENAI_IMPORT_ERROR
        if OPENAI_IMPORT_ERROR is not None:
            raise RuntimeError(
                "The 'openai' package is required for Grok models. "
                "Install it with: pip install openai"
            ) from OPENAI_IMPORT_ERROR
        from openai import OpenAI
        api_key = os.environ.get(cls.api_key_env_var)
        if not api_key:
            raise RuntimeError(f"Please set the {cls.api_key_env_var} environment variable.")
        cls.client = OpenAI(
            api_key=api_key,
            base_url=cls.base_url,
            default_headers={"x-grok-conv-id": cls._conv_id},
        )
        return cls.client
