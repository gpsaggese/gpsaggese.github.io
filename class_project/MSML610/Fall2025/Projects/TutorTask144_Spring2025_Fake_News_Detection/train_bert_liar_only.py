#!/usr/bin/env python3
"""
train_bert_liar_only.py

Fine-tune DistilBERT on LIAR dataset only for faster training on CPU.
Once validated, can be extended to multi-dataset.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project path
PROJECT_DIR = Path(__file__).parent


class BertNewsDataset(Dataset):
    """Minimal dataset for BERT fine-tuning."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        """Store texts and labels, tokenize on-the-fly."""
        self.texts = [str(t)[:256] for t in texts]  # Pre-process texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize on-the-fly (lazy tokenization)
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_liar_dataset() -> Tuple[List[str], List[int]]:
    """Load LIAR dataset."""
    logger.info("Loading LIAR dataset...")

    liar_path = PROJECT_DIR / 'data' / 'LIAR'
    texts = []
    labels = []

    # Load all LIAR files
    for tsv_file in ['train.tsv', 'valid.tsv', 'test.tsv']:
        filepath = liar_path / tsv_file
        if filepath.exists():
            df = pd.read_csv(filepath, sep='\t', low_memory=False, header=None)

            # LIAR: column 1 = label, column 2 = statement
            file_texts = df[2].fillna('').astype(str).values
            file_labels = [0 if x in ['false', 'half-true', 'mostly-false'] else 1
                          for x in df[1].fillna('').astype(str).values]

            texts.extend(file_texts)
            labels.extend(file_labels)
            logger.info(f"  ✓ Loaded {tsv_file}: {len(file_texts)} samples")

    logger.info(f"✅ Total LIAR samples: {len(texts)}")
    fake_count = sum(1 for l in labels if l == 0)
    real_count = sum(1 for l in labels if l == 1)
    logger.info(f"   Fake: {fake_count} ({100*fake_count/len(texts):.1f}%)")
    logger.info(f"   Real: {real_count} ({100*real_count/len(texts):.1f}%)")

    return texts, labels


def train_bert_model():
    """Main training function."""
    logger.info(f"\n{'='*80}")
    logger.info("BERT FINE-TUNING FOR FAKE NEWS DETECTION (LIAR Dataset)")
    logger.info(f"{'='*80}")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load data
    texts, labels = load_liar_dataset()

    # Train-val-test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(X_train)} samples")
    logger.info(f"  Val:   {len(X_val)} samples")
    logger.info(f"  Test:  {len(X_test)} samples")

    # Load model and tokenizer
    logger.info(f"\nLoading DistilBERT model...")
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    ).to(device)

    # Create datasets and dataloaders
    logger.info(f"Preparing datasets...")
    train_dataset = BertNewsDataset(X_train, y_train, tokenizer)
    val_dataset = BertNewsDataset(X_val, y_val, tokenizer)
    test_dataset = BertNewsDataset(X_test, y_test, tokenizer)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Optimizer and scheduler
    num_epochs = 2
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Training loop
    logger.info(f"\n{'='*80}")
    logger.info(f"Training DistilBERT for {num_epochs} epochs")
    logger.info(f"{'='*80}")
    logger.info(f"Batch size: {batch_size}, Learning rate: 2e-05\n")

    best_val_acc = 0
    patience = 1
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (batch_idx + 1) % 50 == 0:
                logger.info(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_loss += outputs.loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, average='weighted')
        val_recall = recall_score(all_labels, all_preds, average='weighted')
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_roc_auc = roc_auc_score(all_labels, all_scores)

        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss:   {avg_val_loss:.4f}")
        logger.info(f"  Val Accuracy:  {val_acc:.4f}")
        logger.info(f"  Val Precision: {val_precision:.4f}")
        logger.info(f"  Val Recall:    {val_recall:.4f}")
        logger.info(f"  Val F1:        {val_f1:.4f}")
        logger.info(f"  Val ROC-AUC:   {val_roc_auc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save best model
            model_path = PROJECT_DIR / 'models' / 'distilbert_best_liar'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(model_path))
            logger.info(f"  ✓ Best model saved (accuracy: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  ⏸ Early stopping at epoch {epoch+1}")
                break

    # Evaluation on test set
    logger.info(f"\n{'='*80}")
    logger.info("Evaluating on test set")
    logger.info(f"{'='*80}")

    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            test_loss += outputs.loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='weighted')
    test_recall = recall_score(all_labels, all_preds, average='weighted')
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    test_roc_auc = roc_auc_score(all_labels, all_scores)

    logger.info(f"\nTest Results:")
    logger.info(f"  Accuracy:  {test_acc:.4f}")
    logger.info(f"  Precision: {test_precision:.4f}")
    logger.info(f"  Recall:    {test_recall:.4f}")
    logger.info(f"  F1:        {test_f1:.4f}")
    logger.info(f"  ROC-AUC:   {test_roc_auc:.4f}")

    # Save results
    results = {
        'model': 'distilbert-base-uncased',
        'dataset': 'LIAR',
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'test_roc_auc': float(test_roc_auc),
        'num_epochs': epoch + 1,
        'timestamp': datetime.now().isoformat()
    }

    results_path = PROJECT_DIR / 'bert_results_liar.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results saved to: {results_path}")

    return 0


if __name__ == '__main__':
    sys.exit(train_bert_model())
