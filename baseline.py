from agents.main import call_agent
from util import GREEN, RESET, load_prompt, get_summary
from config import NUM_PULLS

MODEL_ID = "3.2-llama"

prompt = load_prompt("baseline_prompt.txt")

conversation = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": "What is your first choice?"}
]

all_results = []

for i in range(NUM_PULLS):
    print(f"{RESET}{'='*25} PULL {i+1} OF {NUM_PULLS} {'='*25}")
    
    message_info = call_agent(conversation, MODEL_ID)
    print(message_info)
    print(f"{GREEN}{MODEL_ID}: {message_info['llm_response']}{RESET}\n")
    conversation.append({"role": "assistant", "content": message_info['llm_response']})
    
    all_results.append((int(message_info['arm_pulled']), message_info['reward']))

    print(f"{RESET}Chosen Arm: {message_info['arm_pulled']}")
    print(f"Official System Result: {message_info['reward']} points\n")

    conversation.append({"role": "user", "content": f"You pulled arm {message_info['arm_pulled']} and got {message_info['reward']} points. What is your next choice?"})

print(get_summary(all_results, NUM_PULLS))