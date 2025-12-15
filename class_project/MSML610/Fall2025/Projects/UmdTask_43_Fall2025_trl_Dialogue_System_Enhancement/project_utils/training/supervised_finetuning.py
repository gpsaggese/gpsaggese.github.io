import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "microsoft/DialoGPT-small"
SAVE_PATH = "models/sft_baseline"


def train_sft_baseline():

    print("\n========== TRAINING TRADITIONAL FINE-TUNING BASELINE ==========\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    # -----------------------------
    # LOAD **YOUR** DATASET
    # -----------------------------
    ds = load_dataset("pixelsandpointers/daily_dialog_w_turn_templates", split="train")

    # take tiny sample for fast SFT
    N = 64
    prompts = ds["first"][:N]
    responses = ds["second"][:N]

    pairs = [f"User: {p}\nAssistant: {r}" for p, r in zip(prompts, responses)]

    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()
    for i, text in enumerate(pairs):
        if i >= 30:
            break

        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)

        outputs = model(**encoding, labels=encoding["input_ids"])
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 5 == 0:
            print(f"Step {i} | Loss: {loss.item():.4f}")

    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print("\nSFT baseline model saved to:", SAVE_PATH)
    print("\n==============================================================\n")
    return SAVE_PATH


def main():
    print("Function is running!")
    train_sft_baseline()


if __name__ == "__main__":
    main()
