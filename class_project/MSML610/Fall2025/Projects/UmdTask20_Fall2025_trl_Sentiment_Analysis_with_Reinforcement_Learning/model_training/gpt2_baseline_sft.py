import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import Dataset
import utils

# ==========================================
# CONFIG
# ==========================================
DEBUG_MODE = True   # Set to False for full training
MODEL_NAME = "gpt2"
OUTPUT_DIR = "../models/baseline-gpt2-sft"
utils.set_seed(42)
device = utils.get_device()

print(f"=== Running GPT-2 Baseline on {device} ===")

# 1. Load Data (Baseline Split: 80/10/10)
df, _ = utils.load_and_clean_data()
train_df, val_df, test_df = utils.get_data_splits(df, split_type="baseline")

# --- DEBUG: USE ONLY 5% OF DATA ---
if DEBUG_MODE:
    print("⚠️ DEBUG MODE: Using only 5% of data")
    train_df = train_df.sample(frac=0.05, random_state=42)
    val_df = val_df.sample(frac=0.05, random_state=42)
# ----------------------------------

print(f"Train Size: {len(train_df)} | Val Size: {len(val_df)}")

# 2. Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# 3. Prepare Datasets
def process_data(df):
    prompts = [utils.build_gpt2_prompt(t, l) for t, l in zip(df["clean_text"], df["airline_sentiment"])]
    return Dataset.from_dict({"text": prompts})

train_ds = process_data(train_df)
val_ds = process_data(val_df)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
val_tok = val_ds.map(tokenize, batched=True, remove_columns=["text"])

# 4. Train
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4 if device.type=="cuda" else 2,
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=50,
    use_mps_device=(device.type == "mps"),
    no_cuda=(device.type == "cpu"),
    report_to=[]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Baseline GPT-2 saved to {OUTPUT_DIR}")