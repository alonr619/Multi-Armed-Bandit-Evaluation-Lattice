from agents.main import call_good_agent, call_bad_agent
from bandit import n_armed_bandit
from util import GREEN, RED, RESET, get_summary
import argparse
from config import NUM_PULLS


def conversation(num_pulls: int, good_model_id: str, bad_model_id: str, debug: bool = False) -> list[tuple[int, float]]:
    all_results: list[tuple[int, float]] = []
    past_reasoning: list[str] = []

    bad_messages: list[str] = []
    good_history_turns: list[dict] = []
    bad_history_turns: list[dict] = []
    cache_discount_warnings_shown: set[str] = set()

    for current_pull in range(num_pulls):
        # Good agent makes a decision
        good_response = call_good_agent(
            model=good_model_id,
            current_turn=current_pull + 1,
            past_results=all_results,
            bad_messages=bad_messages,
            good_history_turns=good_history_turns,
            num_pulls=num_pulls,
        )

        good_cache_note = good_response.get("cache_discount_note")
        good_warning_key = f"{good_model_id}:{good_cache_note}"
        if good_cache_note and good_warning_key not in cache_discount_warnings_shown:
            print(f"[cache-warning][{good_model_id}] {good_cache_note}")
            cache_discount_warnings_shown.add(good_warning_key)
        
        if debug:
            print(f"{GREEN}Good Model ({good_model_id}): {good_response['llm_response']}{RESET}\n")
            if good_response.get("usage"):
                print(f"Usage ({good_model_id}): {good_response['usage']}")
        
        past_reasoning.append(good_response['llm_response'])
        if good_response.get("history_turn"):
            good_history_turns.append(good_response["history_turn"])

        # Process the pull if one was made
        if good_response['arm_pulled'] is not None:
            arm = int(good_response['arm_pulled'])
            result = n_armed_bandit(arm)
            all_results.append((arm, result))

            if debug:
                print(f"{RESET}Pull {current_pull + 1}: arm {arm} gave {result} points")

        # Bad agent responds
        if current_pull < num_pulls - 1:
            bad_response = call_bad_agent(
                model=bad_model_id,
                past_results=all_results,
                past_reasoning=past_reasoning,
                bad_history_turns=bad_history_turns,
                num_pulls=num_pulls,
            )

            bad_cache_note = bad_response.get("cache_discount_note")
            bad_warning_key = f"{bad_model_id}:{bad_cache_note}"
            if bad_cache_note and bad_warning_key not in cache_discount_warnings_shown:
                print(f"[cache-warning][{bad_model_id}] {bad_cache_note}")
                cache_discount_warnings_shown.add(bad_warning_key)
            
            if debug:
                print(f"{RED}Bad Model ({bad_model_id}): {bad_response['llm_response']}{RESET}\n")
                print(f"Bad Model Message: {bad_response['message']}")
                if bad_response.get("usage"):
                    print(f"Usage ({bad_model_id}): {bad_response['usage']}")
            
            if bad_response.get("history_turn"):
                bad_history_turns.append(bad_response["history_turn"])
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
    parser.add_argument("--good_model", type=str, default="claude-sonnet-4-6", help="Model ID for the good agent")
    parser.add_argument("--bad_model", type=str, default="claude-sonnet-4-6", help="Model ID for the bad agent")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    conversation(args.num_pulls, args.good_model, args.bad_model, args.debug)
