import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import utils

# ==========================================
# CONFIG
# ==========================================
DEBUG_MODE = False
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "../models/baseline-bert"
utils.set_seed(42)
device = utils.get_device()

print(f"=== Running BERT Baseline on {device} ===")

# 1. Load Data (Baseline Split: 80/10/10)
df, label2id = utils.load_and_clean_data()
train_df, val_df, test_df = utils.get_data_splits(df, split_type="baseline")

# --- DEBUG: USE ONLY 5% OF DATA ---
if DEBUG_MODE:
    print("⚠️ DEBUG MODE: Using only 5% of data")
    train_df = train_df.sample(frac=0.05, random_state=42)
    val_df = val_df.sample(frac=0.05, random_state=42)
# ----------------------------------

# 2. Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label={v: k for k, v in label2id.items()}, label2id=label2id
).to(device)

# 3. Prepare Datasets
def tokenize_bert(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

def df_to_ds(df):
    ds = Dataset.from_dict({
        "text": df["clean_text"],
        "label": [label2id[l] for l in df["airline_sentiment"]]
    })
    return ds.map(tokenize_bert, batched=True)

train_ds = df_to_ds(train_df)
val_ds = df_to_ds(val_df)

# 4. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}

# 5. Train
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    eval_strategy="epoch",
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
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Baseline BERT saved to {OUTPUT_DIR}")