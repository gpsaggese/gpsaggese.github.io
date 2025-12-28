import os
import re
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------
# 1. CONFIG & DEVICE
# -----------------------------------------------------
def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------
# 2. DATA CLEANING
# -----------------------------------------------------
def clean_tweet(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def load_and_clean_data(path="../data/Tweets.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    
    df = pd.read_csv(path)
    df = df[["text", "airline_sentiment"]].dropna()
    
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    LABELS = list(label2id.keys())
    
    df = df[df["airline_sentiment"].isin(LABELS)].copy()
    df["clean_text"] = df["text"].apply(clean_tweet)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    
    return df, label2id

# -----------------------------------------------------
# 3. SPLITTING LOGIC
# -----------------------------------------------------
def get_data_splits(df, split_type="baseline", seed=42):
    """
    split_type="baseline": Returns Train(80) / Val(10) / Test(10)
    split_type="hybrid":   Returns SFT(40) / DPO(40) / Val(10) / Test(10)
    """
    if split_type == "baseline":
        # 80% Train, 20% Temp
        train_df, temp = train_test_split(df, test_size=0.20, stratify=df["airline_sentiment"], random_state=seed)
        # Split Temp into 10% Val, 10% Test
        val_df, test_df = train_test_split(temp, test_size=0.50, stratify=temp["airline_sentiment"], random_state=seed)
        return train_df, val_df, test_df
    
    elif split_type == "hybrid":
        # 1. Split off SFT set (40%) -> Leaves 60%
        sft_df, remaining = train_test_split(df, test_size=0.60, stratify=df["airline_sentiment"], random_state=seed)
        
        # 2. Split DPO set (40% of original = 66.6% of remaining)
        dpo_df, rest = train_test_split(remaining, test_size=0.3333, stratify=remaining["airline_sentiment"], random_state=seed)
        
        # 3. Split Val/Test (10% each of original)
        val_df, test_df = train_test_split(rest, test_size=0.50, stratify=rest["airline_sentiment"], random_state=seed)
        
        return sft_df, dpo_df, val_df, test_df

# -----------------------------------------------------
# 4. PROMPTS & EVALUATION
# -----------------------------------------------------
def build_gpt2_prompt(text, label=None):
    prompt = f'Tweet: "{text}"\nSentiment:'
    if label:
        prompt += f" {label}"
    return prompt

def extract_label(gen_text):
    txt = gen_text.strip().lower()
    if not txt: return None
    first = re.split(r"\s+", txt)[0]
    first = re.sub(r"[^a-z]", "", first)
    if first in {"negative", "neutral", "positive"}:
        return first
    return None