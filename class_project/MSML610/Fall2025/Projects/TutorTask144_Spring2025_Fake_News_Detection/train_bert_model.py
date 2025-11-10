#!/usr/bin/env python3
"""
train_bert_model.py

Fine-tune BERT/RoBERTa for fake news detection on combined datasets.
Uses HuggingFace transformers library for state-of-the-art performance.
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
from torch.utils.data import Dataset, DataLoader, TensorDataset
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

class FakeNewsDataset(Dataset):
    """Custom dataset for fake news detection with lazy tokenization."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = torch.tensor(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Tokenize on-the-fly to avoid memory issues with large datasets
        text = str(self.texts[idx])[:512]  # Limit text length before tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': self.labels[idx]
        }


class BERTFinetuner:
    """Fine-tune BERT for fake news detection."""

    def __init__(self, model_name='distilbert-base-uncased', device=None):
        """
        Initialize BERT fine-tuner.

        Args:
            model_name: HuggingFace model (distilbert-base-uncased or roberta-base)
            device: torch device (cuda or cpu)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Loading {model_name} model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(self.device)

        logger.info(f"Model loaded on {self.device}")

    def prepare_data(self) -> Tuple[List, List, List, List, List, List]:
        """Prepare combined dataset from all three sources."""
        logger.info("Preparing combined dataset...")

        texts = []
        labels = []

        # Load LIAR (no headers, column 2 is statement, column 1 is label)
        liar_path = PROJECT_DIR / 'data' / 'LIAR'
        if liar_path.exists():
            for tsv_file in ['train.tsv', 'valid.tsv', 'test.tsv']:
                df = pd.read_csv(liar_path / tsv_file, sep='\t', low_memory=False, header=None)
                # Column 2 (index 2) is the statement, column 1 (index 1) is the label
                texts.extend(df[2].fillna('').astype(str).values)
                # LIAR labels: 0=false, 1=half-true, 2=mostly-true, 3=true
                # Convert to binary: 0,1,2 = fake (0), 3 = real (1)
                labels.extend([0 if x in ['false', 'half-true', 'mostly-false'] else 1
                              for x in df[1].fillna('').astype(str).values])
                logger.info(f"  ✓ Loaded LIAR {tsv_file}: {len(df)} samples")

        # Load ISOT
        isot_path = PROJECT_DIR / 'data' / 'ISOT'
        if isot_path.exists():
            for csv_file in ['True.csv', 'Fake.csv']:
                df = pd.read_csv(isot_path / csv_file, low_memory=False)
                text_col = 'text' if 'text' in df.columns else df.columns[1]
                texts.extend(df[text_col].fillna('').values)
                # True.csv = real (1), Fake.csv = fake (0)
                label = 1 if csv_file == 'True.csv' else 0
                labels.extend([label] * len(df))
                logger.info(f"  ✓ Loaded ISOT {csv_file}: {len(df)} samples")

        # Load FakeNewsNet
        fakenewsnet_path = PROJECT_DIR / 'data' / 'FakeNewsNet' / 'fakenewsnet_combined.csv'
        if fakenewsnet_path.exists():
            df = pd.read_csv(fakenewsnet_path)
            texts.extend(df['text'].fillna('').values)
            labels.extend(df['label'].values)
            logger.info(f"  ✓ Loaded FakeNewsNet: {len(df)} samples")

        total_samples = len(texts)
        logger.info(f"\n✅ Total samples loaded: {total_samples}")

        # Class distribution
        fake_count = sum(1 for l in labels if l == 0)
        real_count = sum(1 for l in labels if l == 1)
        logger.info(f"   Fake: {fake_count} ({100*fake_count/total_samples:.1f}%)")
        logger.info(f"   Real: {real_count} ({100*real_count/total_samples:.1f}%)")

        # Split: 70% train, 15% val, 15% test
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

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train(self, X_train, y_train, X_val, y_val, epochs=3, batch_size=16, lr=2e-5):
        """
        Fine-tune BERT on training data.

        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate

        Returns:
            dict: Training history
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Training {self.model_name} for {epochs} epochs")
        logger.info(f"{'='*80}")
        logger.info(f"Batch size: {batch_size}, Learning rate: {lr}")

        # Create datasets
        train_dataset = FakeNewsDataset(X_train, y_train, self.tokenizer)
        val_dataset = FakeNewsDataset(X_val, y_val, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_roc_auc': []
        }

        best_val_accuracy = 0
        patience = 2
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            # Validation phase
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            all_logits = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    val_loss += outputs.loss.item()

                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_logits.extend(logits[:, 1].detach().cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = accuracy_score(all_labels, all_preds)
            val_precision = precision_score(all_labels, all_preds, average='weighted')
            val_recall = recall_score(all_labels, all_preds, average='weighted')
            val_f1 = f1_score(all_labels, all_preds, average='weighted')
            val_roc_auc = roc_auc_score(all_labels, all_logits)

            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_f1'].append(val_f1)
            history['val_roc_auc'].append(val_roc_auc)

            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            logger.info(f"  Train Loss:     {avg_train_loss:.4f}")
            logger.info(f"  Val Loss:       {avg_val_loss:.4f}")
            logger.info(f"  Val Accuracy:   {val_accuracy:.4f}")
            logger.info(f"  Val Precision:  {val_precision:.4f}")
            logger.info(f"  Val Recall:     {val_recall:.4f}")
            logger.info(f"  Val F1:         {val_f1:.4f}")
            logger.info(f"  Val ROC-AUC:    {val_roc_auc:.4f}")

            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                # Save best model
                best_model_path = PROJECT_DIR / 'models' / f'best_bert_{self.model_name.split("/")[-1]}'
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(str(best_model_path))
                logger.info(f"  ✓ Best model saved (accuracy: {best_val_accuracy:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"  ⏸ Early stopping at epoch {epoch+1}")
                    break

        return history

    def evaluate(self, X_test, y_test, batch_size=16) -> Dict:
        """Evaluate model on test set."""
        logger.info(f"\n{'='*80}")
        logger.info("Evaluating on test set")
        logger.info(f"{'='*80}")

        test_dataset = FakeNewsDataset(X_test, y_test, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        self.model.eval()
        all_preds = []
        all_labels = []
        all_logits = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits[:, 1].detach().cpu().numpy())

        # Metrics
        test_accuracy = accuracy_score(all_labels, all_preds)
        test_precision = precision_score(all_labels, all_preds, average='weighted')
        test_recall = recall_score(all_labels, all_preds, average='weighted')
        test_f1 = f1_score(all_labels, all_preds, average='weighted')
        test_roc_auc = roc_auc_score(all_labels, all_logits)

        results = {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'roc_auc': test_roc_auc
        }

        logger.info(f"\nTest Results:")
        logger.info(f"  Accuracy:   {test_accuracy:.4f}")
        logger.info(f"  Precision:  {test_precision:.4f}")
        logger.info(f"  Recall:     {test_recall:.4f}")
        logger.info(f"  F1:         {test_f1:.4f}")
        logger.info(f"  ROC-AUC:    {test_roc_auc:.4f}")

        return results


def main():
    """Main execution."""
    logger.info(f"\n{'='*80}")
    logger.info("BERT FINE-TUNING FOR FAKE NEWS DETECTION")
    logger.info(f"{'='*80}")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Initialize fine-tuner with DistilBERT (faster)
    model_name = 'distilbert-base-uncased'
    fine_tuner = BERTFinetuner(model_name=model_name, device=device)

    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = fine_tuner.prepare_data()

    # Train (reduced epochs and batch size for faster CPU training)
    history = fine_tuner.train(
        X_train, y_train, X_val, y_val,
        epochs=2,
        batch_size=32,
        lr=2e-5
    )

    # Evaluate
    results = fine_tuner.evaluate(X_test, y_test)

    logger.info(f"\n{'='*80}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
