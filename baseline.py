import argparse
from agents.main import call_agent
from util import GREEN, RESET, load_prompt, get_summary
from config import NUM_PULLS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline agent.")
    parser.add_argument("--num_pulls", type=int, default=NUM_PULLS, help="Number of pulls in the game")
    parser.add_argument("--model", type=str, default="3.2-llama", help="Model ID to use for the agent")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    MODEL_ID = args.model
    num_pulls = args.num_pulls
    
    prompt = load_prompt("baseline_prompt.txt")
    
    conversation = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "What is your first choice?"}
    ]
    
    all_results = []
    
    for i in range(num_pulls):
        print(f"{RESET}{'='*25} PULL {i+1} OF {num_pulls} {'='*25}")
        
        message_info = call_agent(conversation, MODEL_ID)
        if args.debug:
            print(message_info)
        print(f"{GREEN}{MODEL_ID}: {message_info['llm_response']}{RESET}\n")
        conversation.append({"role": "assistant", "content": message_info['llm_response']})
        
        all_results.append((int(message_info['arm_pulled']), message_info['reward']))
    
        print(f"{RESET}Chosen Arm: {message_info['arm_pulled']}")
        print(f"Official System Result: {message_info['reward']} points\n")
    
        conversation.append({"role": "user", "content": f"You pulled arm {message_info['arm_pulled']} and got {message_info['reward']} points. What is your next choice?"})
    
    print(get_summary(all_results, num_pulls))