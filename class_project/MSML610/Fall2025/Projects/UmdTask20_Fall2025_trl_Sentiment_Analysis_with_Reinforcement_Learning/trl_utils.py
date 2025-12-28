import torch
import pandas as pd
import numpy as np
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# ==========================================
# 1. DEVICE CONFIGURATION
# ==========================================
def get_device():
    """Returns the best available device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# ==========================================
# 2. DATA PROCESSING WRAPPERS
# ==========================================
def clean_tweet(text):
    """
    Cleans tweet text:
    - Removes @mentions
    - Removes URLS
    - Removes special characters
    """
    if not isinstance(text, str): return ""
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

def load_airline_data(csv_path="Tweets.csv"):
    """
    Loads and cleans the Twitter US Airline Sentiment dataset.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    df = df[["text", "airline_sentiment"]]
    df["clean_text"] = df["text"].apply(clean_tweet)
    
    # Map labels to IDs
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    df = df[df["airline_sentiment"].isin(label2id.keys())]
    
    return df, label2id

def get_data_splits(df, seed=42):
    """
    Splits data into a 'hybrid' held-out test set structure.
    Returns: train_df, val_df, test_df
    """
    # Simple random split for demonstration
    train_val = df.sample(frac=0.9, random_state=seed)
    test = df.drop(train_val.index)
    
    train = train_val.sample(frac=0.9, random_state=seed)
    val = train_val.drop(train.index)
    
    return train, val, test

# ==========================================
# 3. MODEL WRAPPER API
# ==========================================
class SentimentModel:
    """
    A unified wrapper for both BERT (SequenceClassification) 
    and GPT-2 (CausalLM) sentiment models.
    """
    def __init__(self, repo_id, model_type="gpt2", device=None):
        self.repo_id = repo_id
        self.model_type = model_type
        self.device = device if device else get_device()
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Internal method to load resources from Hugging Face Hub."""
        print(f"Loading {self.repo_id} ({self.model_type})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
        
        if self.model_type == "bert":
            self.model = AutoModelForSequenceClassification.from_pretrained(self.repo_id).to(self.device)
        else:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(self.repo_id).to(self.device)
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
            self.model.eval()

    def predict(self, text):
        """
        Predicts sentiment for a single string.
        Returns: (label_string, confidence_score)
        """
        if self.model_type == "bert":
            return self._predict_bert(text)
        else:
            return self._predict_gpt2(text)

    def _predict_bert(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=1).item()
            
        # Hardcoded mapping based on training
        id2label = {0: "negative", 1: "neutral", 2: "positive"}
        return id2label.get(pred_id, "neutral"), probs[0][pred_id].item()

    def _predict_gpt2(self, text):
        # Prompt engineering for GPT-2
        prompt = f'Tweet: "{text}"\nSentiment:'
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=3, 
                pad_token_id=self.tokenizer.eos_token_id, 
                do_sample=False
            )
        
        # Decode only the new tokens
        gen_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        cleaned_label = gen_text.lower().strip().split()[0] if gen_text.strip() else "neutral"
        
        # Normalize output
        valid_labels = ["negative", "neutral", "positive"]
        for label in valid_labels:
            if label in cleaned_label:
                return label, 0.0 # GPT-2 generation confidence is complex to normalize quickly
        return "neutral", 0.0