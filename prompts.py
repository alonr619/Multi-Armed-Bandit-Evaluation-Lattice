from bandit import ARMS
from util import load_prompt, expected_score

def get_good_prompt(num_pulls: int) -> str:
    num_arms = len(ARMS)
    arms_1 = ", ".join([f"Arm {i}" for i in range(num_arms-1)]) + f", and Arm {num_arms-1}"
    arms_2 = ", ".join([f"{i}" for i in range(num_arms-1)]) + f", or {num_arms-1}"
    arms_3 = ", ".join([f"{i}" for i in range(num_arms-1)]) + f", AND {num_arms-1}"
    return load_prompt("good_prompt.txt").format(num_pulls=num_pulls, num_arms=num_arms, arms_1=arms_1, arms_2=arms_2, arms_3=arms_3)

def arm_description(choice: int) -> str:
    tmp = [0] + sorted(ARMS[choice].keys())
    s = f"Arm {choice}: " + ", ".join([f"{int(tmp[i]*100)-int(tmp[i-1]*100)}% chance of giving {ARMS[choice][tmp[i]]} points" for i in range(1, len(tmp))])
    s += f" - expected value is {expected_score(choice)}."
    return s

def get_bad_prompt(num_pulls: int) -> str:
    info = "\n".join([arm_description(i) for i in range(len(ARMS))])
    ev_info = "\n".join(
        [f"Arm {i}: expected value {expected_score(i)}" for i in range(len(ARMS))]
    )
    return load_prompt("bad_prompt.txt").format(
        num_pulls=num_pulls,
        info=info,
        ev_info=ev_info,
    )
