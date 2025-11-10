#!/usr/bin/env python3
"""
eval_bert_liar.py

Evaluate the saved BERT model on LIAR test set and generate detailed results.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project path
PROJECT_DIR = Path(__file__).parent


class BertNewsDataset(Dataset):
    """Dataset for BERT evaluation."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = [str(t)[:256] for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

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


def load_liar_dataset():
    """Load LIAR dataset."""
    liar_path = PROJECT_DIR / 'data' / 'LIAR'
    texts = []
    labels = []

    for tsv_file in ['train.tsv', 'valid.tsv', 'test.tsv']:
        filepath = liar_path / tsv_file
        if filepath.exists():
            df = pd.read_csv(filepath, sep='\t', low_memory=False, header=None)
            texts.extend(df[2].fillna('').astype(str).values)
            labels.extend([0 if x in ['false', 'half-true', 'mostly-false'] else 1
                          for x in df[1].fillna('').astype(str).values])

    return texts, labels


def evaluate_model():
    """Evaluate saved BERT model."""
    logger.info(f"\n{'='*80}")
    logger.info("BERT MODEL EVALUATION - LIAR DATASET")
    logger.info(f"{'='*80}")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("\nLoading LIAR dataset...")
    texts, labels = load_liar_dataset()

    # Create same split as training
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info(f"Test set size: {len(X_test)} samples")

    # Load model and tokenizer
    model_path = PROJECT_DIR / 'models' / 'distilbert_best_liar'
    logger.info(f"\nLoading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(device)

    # Create dataset and dataloader
    test_dataset = BertNewsDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Evaluate
    logger.info("\nEvaluating on test set...")
    model.eval()

    all_preds = []
    all_labels = []
    all_scores = []
    test_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            test_loss += outputs.loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())

            if (batch_idx + 1) % 20 == 0:
                logger.info(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")

    avg_test_loss = test_loss / len(test_loader)

    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='weighted')
    test_recall = recall_score(all_labels, all_preds, average='weighted')
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    test_roc_auc = roc_auc_score(all_labels, all_scores)

    # Per-class metrics
    class_report = classification_report(all_labels, all_preds,
                                        target_names=['Fake', 'Real'],
                                        output_dict=True)

    cm = confusion_matrix(all_labels, all_preds)

    # Log results
    logger.info(f"\n{'='*80}")
    logger.info("TEST RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Loss:      {avg_test_loss:.4f}")
    logger.info(f"  Accuracy:  {test_acc:.4f}")
    logger.info(f"  Precision: {test_precision:.4f}")
    logger.info(f"  Recall:    {test_recall:.4f}")
    logger.info(f"  F1-Score:  {test_f1:.4f}")
    logger.info(f"  ROC-AUC:   {test_roc_auc:.4f}")

    logger.info(f"\nPer-Class Metrics:")
    for class_name in ['Fake', 'Real']:
        metrics = class_report[class_name]
        logger.info(f"  {class_name}:")
        logger.info(f"    Precision: {metrics['precision']:.4f}")
        logger.info(f"    Recall:    {metrics['recall']:.4f}")
        logger.info(f"    F1-Score:  {metrics['f1-score']:.4f}")

    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  Fake  (actual) | {cm[0][0]:4d} correctly classified as fake, {cm[0][1]:4d} misclassified as real")
    logger.info(f"  Real  (actual) | {cm[1][0]:4d} misclassified as fake, {cm[1][1]:4d} correctly classified as real")

    # Save results
    results = {
        'model': 'distilbert-base-uncased',
        'dataset': 'LIAR',
        'test_size': len(X_test),
        'test_loss': float(avg_test_loss),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'test_roc_auc': float(test_roc_auc),
        'per_class_metrics': {
            'fake': {
                'precision': float(class_report['Fake']['precision']),
                'recall': float(class_report['Fake']['recall']),
                'f1-score': float(class_report['Fake']['f1-score']),
                'support': int(class_report['Fake']['support'])
            },
            'real': {
                'precision': float(class_report['Real']['precision']),
                'recall': float(class_report['Real']['recall']),
                'f1-score': float(class_report['Real']['f1-score']),
                'support': int(class_report['Real']['support'])
            }
        },
        'confusion_matrix': {
            'fake_as_fake': int(cm[0][0]),
            'fake_as_real': int(cm[0][1]),
            'real_as_fake': int(cm[1][0]),
            'real_as_real': int(cm[1][1])
        },
        'timestamp': datetime.now().isoformat()
    }

    results_path = PROJECT_DIR / 'bert_eval_results_liar.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results saved to: {results_path}")

    return 0


if __name__ == '__main__':
    sys.exit(evaluate_model())
