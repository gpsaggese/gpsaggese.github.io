from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=60,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def evaluate_before_after(tokenizer):

    print("\n=== Running Evaluation ===\n")

    # Load base model BEFORE PPO training
    base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

    # Load PPO trained model
    trained_model = AutoModelForCausalLM.from_pretrained("models/ppo_dialo_model")

    prompts = [
        "I feel very lonely today.",
        "Give me some motivation.",
        "Can you explain reinforcement learning simply?",
        "What should I do if I am stressed?"
    ]

    for p in prompts:
        print("\n-------------------------------------")
        print(f"USER PROMPT: {p}")

        before = generate_response(base_model, tokenizer, p)
        after = generate_response(trained_model, tokenizer, p)

        print("\n--- Before PPO ---")
        print(before)

        print("\n--- After PPO ---")
        print(after)

    print("\n=== Evaluation Complete ===\n")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    evaluate_before_after(tokenizer)
