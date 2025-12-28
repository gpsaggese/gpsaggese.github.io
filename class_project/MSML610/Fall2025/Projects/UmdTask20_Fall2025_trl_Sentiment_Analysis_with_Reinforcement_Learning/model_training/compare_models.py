import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import utils
import os

# ==========================================
# CONFIG
# ==========================================
DEBUG_MODE = False  # Set to False to evaluate on the full test set
PREDICTIONS_CSV = "../data/predictions.csv"
METRICS_CSV = "../data/evaluation_results.csv"

utils.set_seed(42)
device = utils.get_device()
print(f"=== Running Detailed Evaluation on {device} ===")

# -----------------------------------------------------
# 1. LOAD TEST DATA
# -----------------------------------------------------
df, label2id = utils.load_and_clean_data()
id2label = {v: k for k, v in label2id.items()}
# Ensure we get the label names in the correct ID order (0, 1, 2)
target_names = [k for k, v in sorted(label2id.items(), key=lambda item: item[1])]

_, _, _, test_df = utils.get_data_splits(df, split_type="hybrid")

if DEBUG_MODE:
    print("⚠️ DEBUG MODE: Using only 5% of test data")
    test_df = test_df.sample(frac=0.05, random_state=42)

# Prepare Master DataFrame to store all results
master_df = test_df[["clean_text", "airline_sentiment"]].copy().reset_index(drop=True)
print(f"Test Set Size: {len(master_df)} samples")

# -----------------------------------------------------
# 2. DEFINITION OF MODELS
# -----------------------------------------------------
models_config = {
    "Baseline BERT": {
        "path": "blank4hd/airline-sentiment-bert-baseline",
        "type": "bert"
    },
    "Baseline GPT-2": {
        "path": "blank4hd/airline-sentiment-baseline-gpt2-sft",
        "type": "gpt2"
    },
    "Improved SFT (GPT-2)": {
        "path": "blank4hd/airline-sentiment-gpt2-improved-sft",
        "type": "gpt2"
    },
    "DPO Model (Active Learning)": {
        "path": "blank4hd/airline-sentiment-gpt2-dpo-active-learning",
        "type": "gpt2"
    }
}

# -----------------------------------------------------
# 3. PREDICTION FUNCTIONS
# -----------------------------------------------------
def predict_bert(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id]  # Return Label String

def predict_gpt2(model, tokenizer, text):
    prompt = utils.build_gpt2_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=3, 
            pad_token_id=tokenizer.eos_token_id, 
            do_sample=False
        )
    gen_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    label_str = utils.extract_label(gen_text)
    return label_str if label_str in label2id else "neutral"

# -----------------------------------------------------
# 4. EVALUATION LOOP
# -----------------------------------------------------
metrics_list = []

for name, config in models_config.items():
    print(f"\n" + "-"*40)
    print(f"Evaluating: {name}")
    print("-"*40)
    
    # Load Model
    try:
        print(f"📥 Loading from Hub: {config['path']}...")
        if config["type"] == "bert":
            tokenizer = AutoTokenizer.from_pretrained(config["path"])
            model = AutoModelForSequenceClassification.from_pretrained(config["path"]).to(device)
        else:
            tokenizer = AutoTokenizer.from_pretrained(config["path"])
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(config["path"]).to(device)
            model.config.pad_token_id = tokenizer.eos_token_id
        model.eval()
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")
        continue

    # Predict Loop
    preds = []
    for text in tqdm(master_df["clean_text"], desc=f"Predicting"):
        try:
            if config["type"] == "bert":
                p = predict_bert(model, tokenizer, text)
            else:
                p = predict_gpt2(model, tokenizer, text)
        except:
            p = "neutral" # Fallback
        preds.append(p)
    
    # Store Predictions in Master DF
    master_df[name] = preds
    
    # Calculate Metrics
    y_true_ids = [label2id[l] for l in master_df["airline_sentiment"]]
    y_pred_ids = [label2id[l] for l in preds]
    
    acc = accuracy_score(y_true_ids, y_pred_ids)
    f1 = f1_score(y_true_ids, y_pred_ids, average="macro")
    
    # --- PRINT DETAILED REPORT ---
    print(f"\n📊 Classification Report for {name}:")
    print(classification_report(y_true_ids, y_pred_ids, target_names=target_names, digits=4))
    # -----------------------------
    
    metrics_list.append({
        "Model": name,
        "Accuracy": acc,
        "Macro F1": f1
    })

    # Cleanup
    del model
    del tokenizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# -----------------------------------------------------
# 5. SAVE RESULTS
# -----------------------------------------------------
print("\n" + "="*60)
print("FINAL METRICS SUMMARY")
print("="*60)
metrics_df = pd.DataFrame(metrics_list)
print(metrics_df)

# Save Files
metrics_df.to_csv(METRICS_CSV, index=False)
master_df.to_csv(PREDICTIONS_CSV, index=False)

print(f"\n✅ Metrics saved to: {METRICS_CSV}")
print(f"✅ Detailed predictions saved to: {PREDICTIONS_CSV}")