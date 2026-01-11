from agents.main import call_good_agent, call_bad_agent
from bandit import n_armed_bandit
from util import GREEN, RED, RESET, get_summary
import argparse
from config import NUM_PULLS


def conversation(num_pulls: int, good_model_id: str, bad_model_id: str, debug: bool = False) -> list[tuple[int, float]]:
    all_results: list[tuple[int, float]] = []
    past_reasoning: list[str] = []
    past_thoughts: list[str] = []
    bad_messages: list[str] = []
    
    for current_pull in range(num_pulls):
        # Good agent makes a decision
        good_response = call_good_agent(
            model=good_model_id,
            past_results=all_results,
            past_reasoning=past_reasoning,
            bad_messages=bad_messages,
            num_pulls=num_pulls
        )
        
        if debug:
            print(f"{GREEN}Good Model ({good_model_id}): {good_response['llm_response']}{RESET}\n")
        
        past_reasoning.append(good_response['llm_response'])
        
        # Process the pull if one was made
        if good_response['arm_pulled'] is not None:
            arm = int(good_response['arm_pulled'])
            result = n_armed_bandit(arm)
            all_results.append((arm, result))
            
            if debug:
                print(f"{RESET}Pull {current_pull + 1}: arm {arm} gave {result} points")
        
        # Bad agent responds
        bad_response = call_bad_agent(
            model=bad_model_id,
            past_results=all_results,
            past_thoughts=past_thoughts,
            past_reasoning=past_reasoning,
            bad_messages=bad_messages,
            num_pulls=num_pulls
        )
        
        if debug:
            print(f"{RED}Bad Model ({bad_model_id}): {bad_response['llm_response']}{RESET}\n")
            print(f"Bad Model Message: {bad_response['message']}")
        
        past_thoughts.append(bad_response['llm_response'])
        if bad_response['message']:
            bad_messages.append(bad_response['message'])
        else:
            bad_messages.append("")
    
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

    conversation(args.num_pulls, args.good_model, args.bad_model, args.debug)