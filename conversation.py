from agents.main import call_agent
from util import remove_thinking, remove_pull, GREEN, RED, RESET, get_summary
import argparse
from config import NUM_PULLS
from prompts import get_good_prompt, get_bad_prompt

def conversation(num_pulls: int, good_model_id: str, bad_model_id: str, good_prompt: str, bad_prompt: str, debug: bool = False) -> list[tuple[int, float]]:
    all_results = []
    current_pulls = 0

    conversation_a = [{"role": "system", "content": good_prompt}]
    conversation_b = [{"role": "system", "content": bad_prompt}]

    conversation_a.append({"role": "user", "content": "This is the first round. Please begin the discussion. What are your initial thoughts?"})

    while current_pulls < num_pulls:
        message_info = call_agent(conversation_a, good_model_id)
        if debug:
            print(f"{GREEN}Good Model ({good_model_id}): {message_info['llm_response']}{RESET}\n")
        conversation_a.append({"role": "assistant", "content": message_info['llm_response']})
        conversation_b.append({"role": "user", "content": remove_pull(message_info['llm_response'])})

        if message_info['arm_pulled'] is not None and message_info['reward'] is not None:
            pull = int(message_info['arm_pulled'])
            result = float(message_info['reward'])
            current_pulls += 1
            all_results.append((pull, result))
            if debug:
                print(f"{RESET}Pull {current_pulls}: arm {pull} gave {result} points")

            conversation_a.append({"role": "user", "content": f"You pulled arm {pull} and got {result} points"})
            conversation_b.append({"role": "user", "content": f"I pulled arm {pull} and got {result} points"})
        
        message_info = call_agent(conversation_b, bad_model_id, include_tools=False)
        if debug:
            print(f"{RED}Bad Model ({bad_model_id}): {message_info['llm_response']}{RESET}\n")
        conversation_b.append({"role": "assistant", "content": message_info['llm_response']})
        conversation_a.append({"role": "user", "content": remove_thinking(message_info['llm_response'])})

    if debug:
        print(get_summary(all_results, num_pulls))
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run conversation between two agents.")
    parser.add_argument("--num_pulls", type=int, default=NUM_PULLS, help="Number of pulls")
    parser.add_argument("--good_model", type=str, default="4-sonnet", help="Model ID for the good agent")
    parser.add_argument("--bad_model", type=str, default="4-sonnet", help="Model ID for the bad agent")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    good_prompt = get_good_prompt(args.num_pulls)
    bad_prompt = get_bad_prompt(args.num_pulls)

    conversation(args.num_pulls, args.good_model, args.bad_model, good_prompt, bad_prompt, args.debug)