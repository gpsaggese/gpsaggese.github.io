import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import Dataset, concatenate_datasets
import utils
import sys
from math import exp

# ==========================================
# CONFIG
# ==========================================
DEBUG_MODE = True            # Set to False for full training
SFT_MODEL_DIR = "../models/gpt2-improved-sft"
OUTPUT_DIR = "../models/gpt2-dpo-active-learning"
NUM_ACTIVE_SAMPLES = 5       # How many tweets to label
POOL_SIZE = 200              # How many samples to check for uncertainty (speed vs accuracy trade-off)
utils.set_seed(42)
device = utils.get_device()

print(f"=== Running DPO + Active Learning (Uncertainty Sampling) on {device} ===")

# -----------------------------------------------------
# 1. LOAD DATA & MODELS
# -----------------------------------------------------
df, label2id = utils.load_and_clean_data()
LABELS = list(label2id.keys())
_, dpo_df, _, _ = utils.get_data_splits(df, split_type="hybrid")

if DEBUG_MODE:
    print("⚠️ DEBUG MODE: Using only 5% of data")
    dpo_df = dpo_df.sample(frac=0.05, random_state=42)

print(f"DPO Pool Size: {len(dpo_df)}")

print(f"Loading SFT model from {SFT_MODEL_DIR}...")
try:
    policy_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_DIR).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_DIR).to(device)
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_DIR)
except OSError:
    print(f"❌ Error: SFT model not found at '{SFT_MODEL_DIR}'.")
    sys.exit(1)

if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
policy_model.eval()

# -----------------------------------------------------
# 2. UNCERTAINTY SAMPLING FUNCTIONS
# -----------------------------------------------------

def get_label_probabilities(model, tokenizer, text):
    """
    Computes probabilities for [negative, neutral, positive] given the text.
    Uses the log-likelihood of the first token of each label.
    """
    prompt = f'Tweet: "{text}"\nSentiment:'
    
    # We want to score the likelihood of the next token being " negative", " neutral", " positive"
    # Note: GPT-2 usually prepends a space.
    candidate_tokens = [" negative", " neutral", " positive"]
    candidate_ids = [tokenizer.encode(t)[0] for t in candidate_tokens]
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Get logits of the last token
        next_token_logits = outputs.logits[0, -1, :]
        
    # Extract logits for our specific candidate labels
    label_logits = next_token_logits[candidate_ids]
    
    # Softmax to get probabilities over just these 3 options
    probs = F.softmax(label_logits, dim=0).cpu().tolist()
    
    return {
        "negative": probs[0],
        "neutral": probs[1],
        "positive": probs[2]
    }

def find_uncertain_examples(df, model, tokenizer, k=5, pool_limit=200):
    """
    Scans 'pool_limit' examples from df, computes margins, and returns k most uncertain.
    Margin = Prob(Best) - Prob(2ndBest). Small margin = High Uncertainty.
    """
    # Sample a pool to scan (scanning 100% of dataset might be slow)
    if len(df) > pool_limit:
        pool = df.sample(pool_limit, random_state=42)
    else:
        pool = df

    scored_samples = []
    
    print(f"\nScanning {len(pool)} samples for uncertainty...")
    for idx, row in pool.iterrows():
        text = row["clean_text"]
        probs = get_label_probabilities(model, tokenizer, text)
        
        # Sort probs: [('negative', 0.4), ('neutral', 0.35), ('positive', 0.25)]
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        top_1_conf = sorted_probs[0][1]
        top_2_conf = sorted_probs[1][1]
        margin = top_1_conf - top_2_conf
        
        scored_samples.append({
            "text": text,
            "probs": probs,
            "margin": margin,
            "pred": sorted_probs[0][0]
        })
    
    # Sort by margin ascending (smallest margin = most uncertain)
    scored_samples.sort(key=lambda x: x["margin"])
    
    return scored_samples[:k]

# -----------------------------------------------------
# 3. ACTIVE LEARNING LOOP
# -----------------------------------------------------

print("\n" + "="*50)
print(f"🧠 ACTIVE LEARNING PHASE (Uncertainty Sampling)")
print(f"Scanning for the {NUM_ACTIVE_SAMPLES} most confusing tweets...")
print("="*50)

uncertain_candidates = find_uncertain_examples(dpo_df, policy_model, tokenizer, k=NUM_ACTIVE_SAMPLES, pool_limit=POOL_SIZE)
active_pairs = []

for i, sample in enumerate(uncertain_candidates):
    text = sample["text"]
    pred = sample["pred"]
    margin = sample["margin"]
    
    print(f"\n--- Tweet {i+1}/{NUM_ACTIVE_SAMPLES} ---")
    print(f"Content: \"{text}\"")
    print(f"Model Prediction: {pred} (Margin: {margin:.4f})")
    print(f"Probabilities: Neg={sample['probs']['negative']:.2f}, Neu={sample['probs']['neutral']:.2f}, Pos={sample['probs']['positive']:.2f}")
    
    while True:
        user_input = input("Your Label (negative/neutral/positive) [Enter to accept model pred]: ").strip().lower()
        if user_input == "":
            user_input = pred
            print(f"Accepted: {user_input}")
            break
        if user_input in LABELS:
            break
        print(f"❌ Invalid label. Choose from: {LABELS}")
    
    # Create Preference Pairs
    prompt = f'Tweet: "{text}"\nSentiment:'
    
    for rej in LABELS:
        if rej != user_input:
            active_pairs.append({
                "prompt": prompt,
                "chosen": " " + user_input,
                "rejected": " " + rej
            })

print(f"\n✅ Collected {len(active_pairs)} new preference pairs from your feedback.")

# -----------------------------------------------------
# 4. PREPARE DPO DATASET
# -----------------------------------------------------
print("Building standard DPO dataset...")
dpo_data = {"prompt": [], "chosen": [], "rejected": []}

for _, row in dpo_df.iterrows():
    text = row["clean_text"]
    true_lab = row["airline_sentiment"]
    prompt = f'Tweet: "{text}"\nSentiment:'
    
    for wrong_lab in LABELS:
        if wrong_lab != true_lab:
            dpo_data["prompt"].append(prompt)
            dpo_data["chosen"].append(" " + true_lab)
            dpo_data["rejected"].append(" " + wrong_lab)

main_dpo_ds = Dataset.from_dict(dpo_data)

if active_pairs:
    active_ds = Dataset.from_list(active_pairs)
    combined_ds = concatenate_datasets([main_dpo_ds, active_ds])
    print(f"Original DPO size: {len(main_dpo_ds)} -> Combined size: {len(combined_ds)}")
else:
    combined_ds = main_dpo_ds

dpo_train, dpo_val = combined_ds.train_test_split(test_size=0.1, seed=42).values()

# -----------------------------------------------------
# 5. TRAIN DPO
# -----------------------------------------------------
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-5,
    beta=0.1,
    logging_steps=10,
    save_strategy="no",
    use_mps_device=(device.type == "mps"),
    no_cuda=(device.type == "cpu"),
    report_to=[]
)

dpo_trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dpo_train,
    eval_dataset=dpo_val,
    processing_class=tokenizer,
)

print("\n=== Starting DPO Training ===")
dpo_trainer.train()
dpo_trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Active Learning DPO Model saved to {OUTPUT_DIR}")