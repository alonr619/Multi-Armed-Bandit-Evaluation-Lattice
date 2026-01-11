ARMS: list[dict[float, float]] = [ # generate a random number from 0 to 1. if it's less than the first number, give the reward in the second number.
    {1: 8}, # always give 8 reward
    {0.5: 18, 1: 6}, # 50% chance of giving 18 reward, 50% chance of giving 6 reward
    {0.6: 24, 1: 4}, # 60% chance of giving 24 reward, 40% chance of giving 4 reward
    {0.75: 26, 1: 2}, # 75% chance of giving 26 reward, 25% chance of giving 2 reward
]

NUM_PULLS: int = 2

MODEL_IDS: list[str] = ["3.2-llama", "3-haiku", "3.5-haiku"]

MAX_TOKENS: int = 1024