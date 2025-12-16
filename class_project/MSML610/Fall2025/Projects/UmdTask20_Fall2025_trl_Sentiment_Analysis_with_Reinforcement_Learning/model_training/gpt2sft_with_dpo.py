import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import utils

# ==========================================
# CONFIG
# ==========================================
DEBUG_MODE = True
SFT_MODEL_DIR = "../models/gpt2-improved-sft" # Must run gpt2_improved_sft.py first!
OUTPUT_DIR = "../models/gpt2-dpo"
utils.set_seed(42)
device = utils.get_device()

print(f"=== Running DPO Training on {device} ===")

# 1. Load Data (Hybrid Split: 40/40/10/10)
df, label2id = utils.load_and_clean_data()
LABELS = list(label2id.keys())
_, dpo_df, _, _ = utils.get_data_splits(df, split_type="hybrid")

# --- DEBUG: USE ONLY 5% OF DATA ---
if DEBUG_MODE:
    print("⚠️ DEBUG MODE: Using only 5% of data")
    dpo_df = dpo_df.sample(frac=0.05, random_state=42)
# ----------------------------------

print(f"DPO Dataset Size: {len(dpo_df)} (Unbalanced)")

# 2. Build Preference Pairs
print("Building DPO pairs...")
dpo_data = {"prompt": [], "chosen": [], "rejected": []}

for _, row in dpo_df.iterrows():
    text = row["clean_text"]
    true_lab = row["airline_sentiment"]
    
    prompt = f'Tweet: "{text}"\nSentiment:'
    
    # Create pairs against all wrong labels
    for wrong_lab in LABELS:
        if wrong_lab != true_lab:
            dpo_data["prompt"].append(prompt)
            dpo_data["chosen"].append(" " + true_lab)
            dpo_data["rejected"].append(" " + wrong_lab)

dpo_dataset = Dataset.from_dict(dpo_data)
# Split internal DPO set for validation
dpo_train, dpo_val = dpo_dataset.train_test_split(test_size=0.1, seed=42).values()

# 3. Load Models
print(f"Loading SFT model from {SFT_MODEL_DIR}...")
try:
    policy_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_DIR).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_DIR).to(device)
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_DIR)
except OSError:
    raise FileNotFoundError("SFT Model not found! Run 'gpt2_improved_sft.py' first.")

if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

# 4. Train DPO
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-5,
    beta=0.1,
    logging_steps=10,
    save_strategy="no",  # Don't save intermediate checkpoints to save space
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

dpo_trainer.train()
dpo_trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"DPO Model saved to {OUTPUT_DIR}")