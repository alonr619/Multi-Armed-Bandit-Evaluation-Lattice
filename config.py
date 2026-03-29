ARMS: list[dict[float, float]] = [ # generate a random number from 0 to 1. if it's less than the first number, give the reward in the second number.
    {1: 8}, # always give 8 reward
    {0.5: 18, 1: 6}, # 50% chance of giving 18 reward, 50% chance of giving 6 reward
    {0.6: 24, 1: 4}, # 60% chance of giving 24 reward, 40% chance of giving 4 reward
    {0.75: 26, 1: 2}, # 75% chance of giving 26 reward, 25% chance of giving 2 reward
]

NUM_PULLS: int = 5

MODEL_IDS: list[str] = ["llama3.2", "claude-haiku-4-5", "gpt-4.1-mini"]

MAX_TOKENS: int = 1024

# Reasoning defaults. To switch back later:
# - OpenAI-compatible: set OPENAI_COMPAT_REASONING_EFFORT to "none"
# - Anthropic: set ANTHROPIC_REASONING_EFFORT to None and
#   ANTHROPIC_THINKING to {"type": "disabled"}
OPENAI_COMPAT_REASONING_EFFORT: str = "low"
ANTHROPIC_REASONING_EFFORT: str | None = "low"
ANTHROPIC_THINKING: dict[str, str] = {"type": "adaptive"}
