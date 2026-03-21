from agents.anthropic import Anthropic
from agents.ollama import Ollama
from agents.openai import OpenAI
from agents.grok import Grok
from agents.gemini import Gemini
from agents.base import BaseLLM
from prompts import get_good_prompt, get_bad_prompt
from typing import Any

clients: list[BaseLLM] = [Anthropic, OpenAI, Grok, Gemini, Ollama]

def _get_client(model: str) -> BaseLLM:
    """Get the appropriate LLM client for the given model."""
    for client in clients:
        if client.contains_model(model):
            return client
    raise ValueError(f"Model {model} not found")


def _good_turn_text(
    turn_index: int,
    num_pulls: int,
    latest_result: tuple[int, float] | None,
    bad_msg: str | None,
) -> str:
    """Format a round update for the good agent with a stable schema."""
    if latest_result is None:
        arm_text = "none yet"
        result_text = "none yet"
    else:
        arm, result = latest_result
        arm_text = str(arm)
        result_text = str(result)

    lines = [
        f"Turn {turn_index} of {num_pulls}",
        f"Latest observed arm choice: {arm_text}",
        f"Latest observed pull result: {result_text}",
        f"Latest message from the other agent: {bad_msg if bad_msg else '(none)'}",
    ]
    return "\n".join(lines)


def _bad_turn_text_completed(i: int, arm: int, result: float, reasoning: str, thought: str, bad_msg: str) -> str:
    lines = [
        f"Pull {i}",
        f"Reasoning: {reasoning}",
        f"Arm choice: {arm}",
        f"Result: {result}",
        f"Your thoughts: {thought}",
        f"Your response: {bad_msg}",
    ]
    return "\n".join(lines)


def _bad_turn_text_latest(i: int, arm: int, result: float, reasoning: str) -> str:
    lines = [
        f"Pull {i} (latest pull, you must respond to this)",
        f"Reasoning: {reasoning}",
        f"Arm choice: {arm}",
        f"Result: {result}",
    ]
    return "\n".join(lines)


def _flatten_history_turns(history_turns: list[dict]) -> list[dict]:
    """Expand a list of history_turn dicts into a flat message list."""
    msgs = []
    for turn in history_turns:
        user_msg = turn.get("user")
        if user_msg:
            msgs.append(user_msg)

        assistant_msg = turn.get("assistant")
        if assistant_msg:
            msgs.append(assistant_msg)

        tool_result = turn.get("tool_result")
        if not tool_result:
            continue
        # Anthropic: single dict. OpenAI: list of dicts.
        if isinstance(tool_result, list):
            msgs.extend(tool_result)
        else:
            msgs.append(tool_result)
    return msgs


def call_good_agent(
    model: str,
    current_turn: int,
    past_results: list[tuple[int, float]],
    bad_messages: list[str],
    good_history_turns: list[dict],
    num_pulls: int,
) -> dict[str, Any]:
    """
    Call the good agent to make a decision.

    Args:
        model: Model identifier string.
        current_turn: 1-indexed turn number in the conversation loop.
        past_results: List of (arm_pulled, result) tuples from previous rounds.
        bad_messages: List of messages from the bad agent.
        good_history_turns: Provider-native history turns for cache-friendly replay.
        num_pulls: Total number of pulls in the game.

    Returns:
        {"llm_response": str, "arm_pulled": int or None, "history_turn": dict}
    """
    client = _get_client(model)
    prompt = get_good_prompt(num_pulls)

    # Build user context from explicit turn index, not from whether a pull
    # happened. This prevents repeated "first-round" framing if the model does
    # not call pull.
    current_user_text = _good_turn_text(
        turn_index=current_turn,
        num_pulls=num_pulls,
        latest_result=past_results[-1] if past_results else None,
        bad_msg=bad_messages[-1] if bad_messages else None,
    )

    conversation = [
        {"role": "system", "content": prompt},
        *_flatten_history_turns(good_history_turns),
        {"role": "user", "content": current_user_text},
    ]

    result = client.query(conversation, model, client.get_tools(True))

    arm_pulled = None
    if result["tool_call"] and result["tool_call"]["name"] == "pull":
        arm_pulled = result["tool_call"]["arguments"]["choice"]

    history_turn = result.get("history_turn")
    if history_turn:
        history_turn = {
            "user": {"role": "user", "content": current_user_text},
            **history_turn,
        }

    return {
        "llm_response": result["llm_response"],
        "arm_pulled": arm_pulled,
        "history_turn": history_turn,
        "usage": result.get("usage"),
        "cache_discount_available": result.get("cache_discount_available"),
        "cache_discount_note": result.get("cache_discount_note"),
    }


def call_bad_agent(
    model: str,
    past_results: list[tuple[int, float]],
    past_reasoning: list[str],
    bad_history_turns: list[dict],
    num_pulls: int,
) -> dict[str, Any]:
    """
    Call the bad agent to generate a manipulative message.

    Args:
        model: Model identifier string.
        past_results: List of (arm_pulled, result) tuples from previous rounds.
        past_thoughts: List of internal thoughts from the bad agent.
        past_reasoning: List of reasoning strings from the good agent.
        bad_history_turns: Provider-native history turns for cache-friendly replay.
        num_pulls: Total number of pulls in the game.

    Returns:
        {"llm_response": str, "message": str or None, "history_turn": dict}
    """
    client = _get_client(model)
    prompt = get_bad_prompt(num_pulls)

    reasoning = past_reasoning[-1] if past_reasoning else ""
    if past_results:
        latest_idx = len(past_results) - 1
        arm, result_val = past_results[latest_idx]
        current_user_text = _bad_turn_text_latest(latest_idx + 1, arm, result_val, reasoning)
    else:
        current_user_text = f"The other agent has not pulled an arm yet. Their reasoning: {reasoning}"

    conversation = [
        {"role": "system", "content": prompt},
        *_flatten_history_turns(bad_history_turns),
        {"role": "user", "content": current_user_text},
    ]

    result = client.query(conversation, model, client.get_tools(False))

    message = None
    if result["tool_call"] and result["tool_call"]["name"] == "send_message":
        message = result["tool_call"]["arguments"]["message"]

    history_turn = result.get("history_turn")
    if history_turn:
        history_turn = {
            "user": {"role": "user", "content": current_user_text},
            **history_turn,
        }

    return {
        "llm_response": result["llm_response"],
        "message": message,
        "history_turn": history_turn,
        "usage": result.get("usage"),
        "cache_discount_available": result.get("cache_discount_available"),
        "cache_discount_note": result.get("cache_discount_note"),
    }
