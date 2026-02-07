import json
import os
from typing import Any

import dotenv
try:
    from openai import OpenAI as OpenAIClient
    OPENAI_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    OpenAIClient = Any  # type: ignore[assignment]
    OPENAI_IMPORT_ERROR = exc

from agents.base import BaseLLM
from config import MAX_TOKENS

dotenv.load_dotenv()


class OpenAICompatible(BaseLLM):
    api_key_env_var: str = ""
    base_url: str | None = None
    provider_name: str = "provider"
    token_limit_param: str = "max_tokens"
    client: OpenAIClient | None = None

    @classmethod
    def get_client(cls) -> OpenAIClient:
        if OPENAI_IMPORT_ERROR is not None:
            raise RuntimeError(
                "The 'openai' package is required for OpenAI/Grok/Gemini models. "
                "Install it with: pip install openai"
            ) from OPENAI_IMPORT_ERROR

        if cls.client is not None:
            return cls.client

        api_key = os.environ.get(cls.api_key_env_var)
        if not api_key:
            raise RuntimeError(f"Please set the {cls.api_key_env_var} environment variable.")

        kwargs: dict[str, Any] = {"api_key": api_key}
        if cls.base_url:
            kwargs["base_url"] = cls.base_url

        cls.client = OpenAIClient(**kwargs)
        return cls.client

    @classmethod
    def _normalize_conversation(cls, conversation: list[dict[str, str]]) -> list[dict[str, str]]:
        system_parts: list[str] = []
        payload: list[dict[str, str]] = []

        for msg in conversation:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                payload.append({"role": msg["role"], "content": msg["content"]})

        if system_parts:
            return [{"role": "system", "content": "\n\n".join(system_parts)}, *payload]
        return payload

    @classmethod
    def _parse_tool_call(cls, message: Any) -> dict[str, Any] | None:
        if not getattr(message, "tool_calls", None):
            return None

        tc = message.tool_calls[0]
        raw_args = tc.function.arguments or "{}"

        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError:
            arguments = {}

        return {
            "name": tc.function.name,
            "arguments": arguments,
        }

    @classmethod
    def _parse_usage(cls, usage: Any) -> tuple[dict[str, int | None], bool, str | None]:
        if not usage:
            return {
                "input_tokens": None,
                "output_tokens": None,
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None,
            }, False, f"Cached token discount reporting is not available for {cls.provider_name}."

        prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = None
        if prompt_tokens_details is not None:
            cached_tokens = getattr(prompt_tokens_details, "cached_tokens", None)

        cache_discount_available = cached_tokens is not None
        cache_discount_note = None
        if not cache_discount_available:
            cache_discount_note = (
                f"Cached token discount reporting is not available for {cls.provider_name} "
                "for this model/response."
            )

        return {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": cached_tokens,
        }, cache_discount_available, cache_discount_note

    @classmethod
    def query(
        cls, conversation: list[dict[str, str]], model: str, tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        messages = cls._normalize_conversation(conversation)
        kwargs: dict[str, Any] = {
            "model": cls.get_model_id(model),
            cls.token_limit_param: MAX_TOKENS,
            "messages": messages,
            "tools": tools,
        }

        try:
            response = cls.get_client().chat.completions.create(**kwargs)
        except Exception as exc:
            error_text = str(exc).lower()
            current_param = cls.token_limit_param
            alt_param = "max_completion_tokens" if current_param == "max_tokens" else "max_tokens"
            should_retry = current_param in error_text or alt_param in error_text

            if not should_retry:
                raise

            kwargs.pop(current_param, None)
            kwargs[alt_param] = MAX_TOKENS
            response = cls.get_client().chat.completions.create(**kwargs)

        message = response.choices[0].message
        usage, cache_discount_available, cache_discount_note = cls._parse_usage(response.usage)

        return {
            "llm_response": message.content or "",
            "tool_call": cls._parse_tool_call(message),
            "usage": usage,
            "cache_discount_available": cache_discount_available,
            "cache_discount_note": cache_discount_note,
        }
