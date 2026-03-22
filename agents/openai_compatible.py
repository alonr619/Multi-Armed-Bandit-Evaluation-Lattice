import json
import os
import re
import ast
from typing import Any

import dotenv
try:
    from openai import OpenAI as OpenAIClient
    OPENAI_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    OpenAIClient = Any  # type: ignore[assignment]
    OPENAI_IMPORT_ERROR = exc

from agents.base import BaseLLM
from config import MAX_TOKENS, OPENAI_COMPAT_REASONING_EFFORT

dotenv.load_dotenv()


class OpenAICompatible(BaseLLM):
    api_key_env_var: str = ""
    base_url: str | None = None
    provider_name: str = "provider"
    token_limit_param: str = "max_tokens"
    reasoning_effort_override: str | None = None
    client: OpenAIClient | None = None
    unsupported_reasoning_effort_models: set[str] = set()
    resolved_token_limit_param: str | None = None

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
    def _normalize_conversation(cls, conversation: list[dict]) -> list[dict]:
        system_parts: list[str] = []
        payload: list[dict] = []

        for msg in conversation:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                payload.append(msg)

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
    def _extract_json_objects(cls, text: str) -> list[str]:
        """Extract likely JSON object snippets from free-form text."""
        objects: list[str] = []

        # First, prefer fenced blocks (```json ... ``` or ``` ... ```).
        for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE):
            block = match.group(1).strip()
            if block:
                objects.append(block)

        # Also scan for balanced {...} objects in plain text.
        start = -1
        depth = 0
        in_string = False
        escape = False
        for idx, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_string = False
                continue

            if ch == "\"":
                in_string = True
            elif ch == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    objects.append(text[start:idx + 1].strip())
                    start = -1

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique_objects: list[str] = []
        for obj in objects:
            if obj not in seen:
                seen.add(obj)
                unique_objects.append(obj)
        return unique_objects

    @classmethod
    def _parse_object_candidate(cls, candidate: str) -> dict[str, Any] | None:
        """Parse dict-like text (JSON / Python / JS-ish) into a Python dict."""
        text = candidate.strip()
        if not text:
            return None

        # Strict JSON first.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Python literal dict support (single quotes, trailing commas).
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass

        # JS-ish objects: quote unquoted keys and normalize booleans/null.
        normalized = re.sub(
            r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)',
            r'\1"\2"\3',
            text,
        )
        normalized_py = re.sub(r"\btrue\b", "True", normalized, flags=re.IGNORECASE)
        normalized_py = re.sub(r"\bfalse\b", "False", normalized_py, flags=re.IGNORECASE)
        normalized_py = re.sub(r"\bnull\b", "None", normalized_py, flags=re.IGNORECASE)

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(normalized if parser is json.loads else normalized_py)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue

        return None

    @classmethod
    def _coerce_tool_arguments(
        cls,
        args: dict[str, Any],
        properties: dict[str, Any],
        required: list[str],
    ) -> dict[str, Any] | None:
        coerced = dict(args)

        for key, schema in properties.items():
            if key not in coerced:
                continue
            value = coerced[key]
            expected_type = schema.get("type")

            if expected_type == "integer":
                if isinstance(value, int):
                    continue
                if isinstance(value, float) and value.is_integer():
                    coerced[key] = int(value)
                    continue
                if isinstance(value, str):
                    try:
                        coerced[key] = int(value.strip())
                        continue
                    except ValueError:
                        return None
                return None

            if expected_type == "string" and not isinstance(value, str):
                coerced[key] = str(value)

        if any(req not in coerced for req in required):
            return None
        return coerced

    @classmethod
    def _parse_tool_call_from_text(
        cls, content: str, tools: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Best-effort parse for cases where model prints tool args in text."""
        if not content or not tools:
            return None

        tool_specs: list[dict[str, Any]] = []
        for tool in tools:
            fn = tool.get("function", {})
            name = fn.get("name")
            params = fn.get("parameters", {}) or {}
            props = params.get("properties", {}) or {}
            required = params.get("required", []) or []
            if isinstance(name, str) and name:
                tool_specs.append(
                    {
                        "name": name,
                        "properties": props,
                        "required": required,
                    }
                )

        if not tool_specs:
            return None

        parsed_dicts: list[dict[str, Any]] = []
        for candidate in cls._extract_json_objects(content):
            obj = cls._parse_object_candidate(candidate)
            if obj is not None:
                parsed_dicts.append(obj)

        for obj in parsed_dicts:
            # Explicit shape: {"name": "...", "arguments": {...}}
            name = obj.get("name")
            arguments = obj.get("arguments")
            if isinstance(name, str) and isinstance(arguments, dict):
                for spec in tool_specs:
                    if spec["name"] != name:
                        continue
                    coerced = cls._coerce_tool_arguments(
                        arguments, spec["properties"], spec["required"]
                    )
                    if coerced is not None:
                        return {"name": name, "arguments": coerced}

            # Implicit shape: {"choice": 0} for a single tool.
            if len(tool_specs) == 1:
                spec = tool_specs[0]
                coerced = cls._coerce_tool_arguments(
                    obj, spec["properties"], spec["required"]
                )
                if coerced is not None:
                    return {"name": spec["name"], "arguments": coerced}

        # Legacy fallback for pull-style tags.
        if len(tool_specs) == 1 and tool_specs[0]["name"] == "pull":
            match = re.search(r"<Pull>\s*(\d+)\s*</Pull>", content, flags=re.DOTALL)
            if match:
                return {"name": "pull", "arguments": {"choice": int(match.group(1))}}

            # Function-style calls in text:
            #   pull({ choice: 2 })
            #   pull({"choice": 2})
            #   pull(choice=2)
            #   pull(choice: 2)
            pull_call_match = re.search(
                r"""\bpull\s*\(\s*(?:\{\s*)?(?:["']?choice["']?\s*(?::|=)\s*)?(-?\d+)""",
                content,
                flags=re.IGNORECASE,
            )
            if pull_call_match:
                return {
                    "name": "pull",
                    "arguments": {"choice": int(pull_call_match.group(1))},
                }

            # Natural-language fallback (e.g., "I'll pull Arm 0", "Pulling arm 2").
            pull_matches = list(
                re.finditer(
                    r"\bpull(?:ing)?\s+arm\s*(\d+)\b",
                    content,
                    flags=re.IGNORECASE,
                )
            )
            if pull_matches:
                return {
                    "name": "pull",
                    "arguments": {"choice": int(pull_matches[-1].group(1))},
                }

        return None

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
        completion_tokens_details = getattr(usage, "completion_tokens_details", None)
        cached_tokens = None
        reasoning_tokens = None
        if prompt_tokens_details is not None:
            cached_tokens = getattr(prompt_tokens_details, "cached_tokens", None)
        if completion_tokens_details is not None:
            reasoning_tokens = getattr(completion_tokens_details, "reasoning_tokens", None)

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
            "reasoning_output_tokens": reasoning_tokens,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": cached_tokens,
        }, cache_discount_available, cache_discount_note

    @classmethod
    def query(
        cls, conversation: list[dict], model: str, tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        messages = cls._normalize_conversation(conversation)
        resolved_model = cls.get_model_id(model)
        reasoning_effort = cls.reasoning_effort_override or OPENAI_COMPAT_REASONING_EFFORT
        normalized_reasoning_effort = reasoning_effort.strip().lower()
        reasoning_effort_key = f"{cls.__name__}:{resolved_model}"
        token_limit_param = cls.resolved_token_limit_param or cls.token_limit_param
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            token_limit_param: MAX_TOKENS,
            "messages": messages,
            "tools": tools,
        }
        if normalized_reasoning_effort and (
            reasoning_effort_key not in cls.unsupported_reasoning_effort_models
        ):
            kwargs["reasoning_effort"] = reasoning_effort

        while True:
            try:
                response = cls.get_client().chat.completions.create(**kwargs)
                break
            except Exception as exc:
                error_text = str(exc).lower()
                current_param = (
                    "max_completion_tokens"
                    if "max_completion_tokens" in kwargs
                    else "max_tokens"
                )
                alt_param = (
                    "max_completion_tokens"
                    if current_param == "max_tokens"
                    else "max_tokens"
                )
                retried = False

                # Some OpenAI-compatible endpoints reject reasoning_effort.
                if "reasoning_effort" in kwargs and "reasoning_effort" in error_text:
                    kwargs.pop("reasoning_effort", None)
                    cls.unsupported_reasoning_effort_models.add(reasoning_effort_key)
                    retried = True

                # Providers differ on token limit parameter name.
                if current_param in error_text or alt_param in error_text:
                    kwargs.pop(current_param, None)
                    kwargs[alt_param] = MAX_TOKENS
                    cls.resolved_token_limit_param = alt_param
                    retried = True

                if not retried:
                    raise

        message = response.choices[0].message
        tool_call = cls._parse_tool_call(message)
        if tool_call is None:
            tool_call = cls._parse_tool_call_from_text(message.content or "", tools)
        usage, cache_discount_available, cache_discount_note = cls._parse_usage(response.usage)

        # Build provider-native history turn for caching in subsequent calls.
        tc_list = getattr(message, "tool_calls", None) or []
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": message.content,
        }
        if tc_list:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tc_list
            ]

        if tc_list:
            tool_result_msgs = [
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "ok",
                }
                for tc in tc_list
            ]
        else:
            tool_result_msgs = []

        history_turn = {
            "assistant": assistant_msg,
            "tool_result": tool_result_msgs,  # list for OpenAI (can be multiple)
        }

        return {
            "llm_response": message.content or "",
            "tool_call": tool_call,
            "history_turn": history_turn,
            "usage": usage,
            "cache_discount_available": cache_discount_available,
            "cache_discount_note": cache_discount_note,
        }
