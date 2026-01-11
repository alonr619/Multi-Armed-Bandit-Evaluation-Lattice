from agents.anthropic import Anthropic
from agents.ollama import Ollama
from agents.base import BaseLLM
from prompts import get_good_prompt, get_bad_prompt
from typing import Any

clients: list[BaseLLM] = [Anthropic, Ollama]

def _get_client(model: str) -> BaseLLM:
    """Get the appropriate LLM client for the given model."""
    for client in clients:
        if client.contains_model(model):
            return client
    raise ValueError(f"Model {model} not found")


def _build_good_agent_input(
    past_results: list[tuple[int, float]],
    past_reasoning: list[str],
    bad_messages: list[str]
) -> str:
    """Build the input text for the good agent based on history."""
    if not past_results:
        return "This is the first round. Please begin the discussion. What are your initial thoughts?"
    
    lines = []
    for i, ((arm, result), reasoning, bad_msg) in enumerate(zip(past_results, past_reasoning, bad_messages), 1):
        lines.append(f"Pull {i}")
        lines.append(f"Reasoning: {reasoning}")
        lines.append(f"Arm choice: {arm}")
        lines.append(f"Result: {result}")
        lines.append(f"Agent Response: {bad_msg}")
        lines.append("")
    
    return "\n".join(lines).strip()


def _build_bad_agent_input(
    past_results: list[tuple[int, float]],
    past_thoughts: list[str],
    past_reasoning: list[str],
    bad_messages: list[str]
) -> str:
    """Build the input text for the bad agent based on history."""
    if not past_results:
        return "The game is starting. The other agent will begin reasoning shortly."
    
    lines = []
    
    # Past completed rounds (where we already responded)
    num_completed = len(past_thoughts)  # past_thoughts and bad_messages are one shorter
    for i in range(num_completed):
        arm, result = past_results[i]
        reasoning = past_reasoning[i]
        thought = past_thoughts[i]
        bad_msg = bad_messages[i]
        
        lines.append(f"Pull {i + 1}")
        lines.append(f"Reasoning: {reasoning}")
        lines.append(f"Arm choice: {arm}")
        lines.append(f"Result: {result}")
        lines.append(f"Your thoughts: {thought}")
        lines.append(f"Your response: {bad_msg}")
        lines.append("")
    
    # Latest pull that needs a response
    latest_idx = len(past_results) - 1
    arm, result = past_results[latest_idx]
    reasoning = past_reasoning[latest_idx]
    
    lines.append(f"Pull {latest_idx + 1} (latest pull, you must respond to this)")
    lines.append(f"Reasoning: {reasoning}")
    lines.append(f"Arm choice: {arm}")
    lines.append(f"Result: {result}")
    
    return "\n".join(lines).strip()


def call_good_agent(
    model: str,
    past_results: list[tuple[int, float]],
    past_reasoning: list[str],
    bad_messages: list[str],
    num_pulls: int
) -> dict[str, Any]:
    """
    Call the good agent to make a decision.

    Args:
        model: Model identifier string.
        past_results: List of (arm_pulled, result) tuples from previous rounds.
        past_reasoning: List of reasoning strings from previous good agent responses.
        bad_messages: List of messages from the bad agent.
        num_pulls: Total number of pulls in the game.

    Returns:
        {"llm_response": str, "arm_pulled": int or None}
    """
    client = _get_client(model)
    prompt = get_good_prompt(num_pulls)
    user_input = _build_good_agent_input(past_results, past_reasoning, bad_messages)
    
    conversation = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}
    ]
    
    result = client.query(conversation, model, client.get_tools(True))
    
    arm_pulled = None
    if result["tool_call"] and result["tool_call"]["name"] == "pull":
        arm_pulled = result["tool_call"]["arguments"]["choice"]
    
    return {
        "llm_response": result["llm_response"],
        "arm_pulled": arm_pulled
    }


def call_bad_agent(
    model: str,
    past_results: list[tuple[int, float]],
    past_thoughts: list[str],
    past_reasoning: list[str],
    bad_messages: list[str],
    num_pulls: int
) -> dict[str, Any]:
    """
    Call the bad agent to generate a manipulative message.

    Args:
        model: Model identifier string.
        past_results: List of (arm_pulled, result) tuples from previous rounds.
        past_thoughts: List of internal thoughts from the bad agent.
        past_reasoning: List of reasoning strings from the good agent.
        bad_messages: List of previous messages sent by the bad agent.
        num_pulls: Total number of pulls in the game.

    Returns:
        {"llm_response": str, "message": str or None}
    """
    client = _get_client(model)
    prompt = get_bad_prompt(num_pulls)
    user_input = _build_bad_agent_input(past_results, past_thoughts, past_reasoning, bad_messages)
    
    conversation = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}
    ]
    
    result = client.query(conversation, model, client.get_tools(False))
    
    message = None
    if result["tool_call"] and result["tool_call"]["name"] == "send_message":
        message = result["tool_call"]["arguments"]["message"]
    
    return {
        "llm_response": result["llm_response"],
        "message": message
    }