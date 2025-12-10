import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead

from preprocess import load_dataset_for_rl
from reward import RewardFunction
from config import get_ppo_config
from evaluate import evaluate_before_after


# Keep training small while debugging
MAX_STEPS = 3


def main():
    print("\n=== Loading tokenizer & model ===")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (CPU/GPU auto)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            "microsoft/DialoGPT-small"
        ).to(device)
    except Exception:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            "microsoft/DialoGPT-small"
        ).to(device)

    print("\n=== Loading dataset ===")
    dataset = load_dataset_for_rl()   # flat dataset
    print("Total training samples:", len(dataset))

    ppo_config = get_ppo_config()
    batch_size = ppo_config.batch_size

    print("\n=== Setting up PPO trainer ===")
    trainer = PPOTrainer(
        ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer
    )

    reward_fn = RewardFunction(device=device)

    gen_kwargs = {
        "max_new_tokens": 32,
        "pad_token_id": tokenizer.pad_token_id,
    }

    print("\n=== Starting PPO training ===")
    step_count = 0

    for i in range(0, len(dataset), batch_size):
        if step_count >= MAX_STEPS:
            print(f"\nReached MAX_STEPS={MAX_STEPS}, stopping training loop.")
            break

        batch = dataset[i : i + batch_size]   # dict-of-lists (correct)
        prompts = batch["first"]              # list of strings
        responses_gt = batch["second"]        # list of strings


        if len(prompts) < batch_size:  # skip last incomplete batch
            continue

        print(f"\n---- PPO Step {step_count + 1} ----")
        print(f"Batch size: {len(prompts)}")

        # 1. Convert prompts to tensors
        query_tensors = []
        for p in prompts:
            enc = tokenizer(
                p,
                return_tensors="pt",
                truncation=True,
                padding=False
            )
            query_tensors.append(enc["input_ids"][0].to(device))

        # 2. Generate responses
        with torch.no_grad():
            response_tensors = trainer.generate(query_tensors, **gen_kwargs)

        # Ensure response tensors have correct shape
        clean_responses = []
        for r in response_tensors:
            if r.dim() == 2:
                r = r.squeeze(0)
            clean_responses.append(r)

        decoded = [
            tokenizer.decode(r, skip_special_tokens=True)
            for r in clean_responses
        ]

        print("Sample response:", decoded[0])

        # 3. Compute rewards
        rewards = reward_fn(prompts, decoded, gen_tokenizer=tokenizer)

        reward_tensors = [
            torch.tensor(r, dtype=torch.float32).to(device)
            for r in rewards
        ]

        # 4. PPO update
        trainer.step(query_tensors, clean_responses, reward_tensors)

        avg_reward = float(sum(rewards)) / len(rewards)
        print(f"Step {step_count+1}: Avg reward = {avg_reward:.4f}")

        step_count += 1

    print("\n=== Saving trained model ===")
    trainer.model.save_pretrained("models/ppo_dialo_model")
    tokenizer.save_pretrained("models/ppo_dialo_model")

    print("\n=== Evaluating improved responses ===")
    evaluate_before_after(tokenizer)

    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()
