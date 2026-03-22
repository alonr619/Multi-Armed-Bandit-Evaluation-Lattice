import argparse
from agents.main import call_good_agent
from bandit import n_armed_bandit
from util import GREEN, RESET, get_summary
from config import NUM_PULLS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline agent.")
    parser.add_argument("--num_pulls", type=int, default=NUM_PULLS, help="Number of pulls in the game")
    parser.add_argument("--model", type=str, default="llama3.2", help="Model ID to use for the agent")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    MODEL_ID = args.model
    num_pulls = args.num_pulls
    
    all_results: list[tuple[int, float]] = []
    bad_messages: list[str] = []
    good_history_turns: list[dict] = []
    
    for i in range(num_pulls):
        print(f"{RESET}{'='*25} PULL {i+1} OF {num_pulls} {'='*25}")
        
        response = call_good_agent(
            model=MODEL_ID,
            current_turn=i + 1,
            past_results=all_results,
            bad_messages=bad_messages,
            good_history_turns=good_history_turns,
            num_pulls=num_pulls
        )
        
        if args.debug:
            print(response)
        print(f"{GREEN}{MODEL_ID}: {response['llm_response']}{RESET}\n")
        
        if response.get("history_turn"):
            good_history_turns.append(response["history_turn"])
        
        if response['arm_pulled'] is not None:
            arm = int(response['arm_pulled'])
            result = n_armed_bandit(arm)
            all_results.append((arm, result))
            bad_messages.append("The other agent chose not to say anything.")
            
            print(f"{RESET}Chosen Arm: {arm}")
            print(f"Official System Result: {result} points\n")
    
    print(get_summary(all_results, num_pulls))
