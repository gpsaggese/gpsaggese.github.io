"""
BERT-based Fake News Detection Pipeline

This module implements fake news classification using BERT (Bidirectional Encoder
Representations from Transformers) fine-tuned on real and fake news datasets.

Features:
- Data loading and preprocessing from data/true.csv and data/fake.csv
- BERT tokenization and dataset preparation
- Fine-tuning BERT on fake news detection task
- Comprehensive metrics: accuracy, precision, recall, ROC-AUC, F1
- Model persistence and inference
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from tqdm import tqdm

from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FakeNewsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_base_dataset(data_dir: str = "data") -> pd.DataFrame:
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    real_csv = data_path / "true.csv"
    if not real_csv.exists():
        raise FileNotFoundError(f"File not found: {real_csv}")

    true_df = pd.read_csv(real_csv)
    logger.info(f"Loaded {len(true_df)} real articles from {real_csv}")

    fake_csv = data_path / "fake.csv"
    if not fake_csv.exists():
        raise FileNotFoundError(f"File not found: {fake_csv}")

    fake_df = pd.read_csv(fake_csv)
    logger.info(f"Loaded {len(fake_df)} fake articles from {fake_csv}")

    dfs = []

    if 'title' in true_df.columns and 'text' in true_df.columns:
        true_df['content'] = true_df['title'].fillna('') + " " + true_df['text'].fillna('')
        true_df['label'] = 1
        dfs.append(true_df[['content', 'label']])

    if 'title' in fake_df.columns and 'text' in fake_df.columns:
        fake_df['content'] = fake_df['title'].fillna('') + " " + fake_df['text'].fillna('')
        fake_df['label'] = 0
        dfs.append(fake_df[['content', 'label']])

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df[combined_df['content'].str.strip() != '']
    combined_df = combined_df.dropna(subset=['content', 'label'])
    combined_df['content'] = combined_df['content'].apply(lambda x: x[:2048])

    logger.info(f"Combined dataset: {len(combined_df)} samples")
    logger.info(f"  Real: {(combined_df['label'] == 1).sum()}")
    logger.info(f"  Fake: {(combined_df['label'] == 0).sum()}")

    return combined_df


def get_device(device: str = None) -> str:
    if device is not None:
        return device
    if torch.backends.mps.is_available():
        logger.info("MacBook GPU (MPS) detected")
        return 'mps'
    if torch.cuda.is_available():
        logger.info("CUDA GPU detected")
        return 'cuda'
    logger.info("Using CPU")
    return 'cpu'


def train_bert_model(
    texts: List[str],
    labels: List[int],
    model_name: str = "bert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    device: str = None
) -> Tuple[BertForSequenceClassification, BertTokenizer]:
    device = get_device(device)
    logger.info(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(device)

    dataset = FakeNewsDataset(texts, labels, tokenizer, max_len=128)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    logger.info(f"Starting BERT training for {epochs} epochs...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=True,
            unit="batch"
        )

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    logger.info("BERT training complete")
    return model, tokenizer


def evaluate_bert_model(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    texts: List[str],
    labels: List[int],
    batch_size: int = 16,
    device: str = None
) -> Dict[str, Any]:
    
    if device is None:
        device = get_device()  # Use GPU auto-detection (MPS > CUDA > CPU)

    model.to(device)
    model.eval()

    dataset = FakeNewsDataset(texts, labels, tokenizer, max_len=128)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    predictions = []
    probabilities = []

    progress_bar = tqdm(loader, desc="Evaluating", unit="batch", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    labels = np.array(labels)

    results = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0),
        'roc_auc': roc_auc_score(labels, probabilities),
        'confusion_matrix': confusion_matrix(labels, predictions),
        'predictions': predictions,
        'probabilities': probabilities,
        'ground_truth': labels
    }

    logger.info(f"Accuracy:  {results['accuracy']:.4f}")
    logger.info(f"Precision: {results['precision']:.4f}")
    logger.info(f"Recall:    {results['recall']:.4f}")
    logger.info(f"F1 Score:  {results['f1']:.4f}")
    logger.info(f"ROC-AUC:   {results['roc_auc']:.4f}")

    return results


def predict_text(
    text: str,
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    device: str = None
) -> Tuple[int, float]:
    
    if device is None:
        device = get_device()  # Use GPU auto-detection (MPS > CUDA > CPU)

    model.to(device)
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    prediction = torch.argmax(logits, dim=1).item()
    confidence = probs[0, prediction].item()

    return prediction, confidence


def save_model(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    save_dir: str = "models/bert_fake_news"
) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    logger.info(f"Model saved to {save_dir}")


def load_model(
    model_dir: str = "models/bert_fake_news",
    device: str = None
) -> Tuple[BertForSequenceClassification, BertTokenizer]:
    if device is None:
        device = get_device()

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    logger.info(f"Model loaded from {model_dir}")
    return model, tokenizer


def full_pipeline(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    model_name: str = "bert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    save_dir: str = "models/bert_fake_news"
) -> Dict[str, Any]:
    
    logger.info("\nBERT FAKE NEWS DETECTION PIPELINE")

    device = get_device() 
    logger.info(f"Device: {device}")

    logger.info("\n[1] Training BERT model...")
    model, tokenizer = train_bert_model(
        train_texts, train_labels,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        device=device
    )

    logger.info("\n[2] Evaluating on test set...")
    results = evaluate_bert_model(
        model, tokenizer,
        test_texts, test_labels,
        batch_size=batch_size,
        device=device
    )

    logger.info("\n[3] Saving model...")
    save_model(model, tokenizer, save_dir)


    return {
        'model': model,
        'tokenizer': tokenizer,
        'results': results,
        'device': device
    }
