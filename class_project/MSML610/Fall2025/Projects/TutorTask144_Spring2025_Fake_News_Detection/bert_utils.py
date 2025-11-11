"""
bert_utils.py

Utility functions and wrapper layer for BERT-based fake news detection.
This module provides reusable components for data loading, model training,
and evaluation of transformer-based text classification models.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    max_text_length: int = 256
    stratify: bool = True


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_name: str = 'distilbert-base-uncased'
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 2
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    patience: int = 1
    device: str = 'cpu'
    use_class_weights: bool = False
    max_text_length: int = 256


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    loss: float
    per_class_metrics: Dict
    confusion_matrix: Dict


class BertTextDataset(Dataset):
    """PyTorch Dataset for text classification with lazy tokenization."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 256
    ):
        """
        Initialize dataset.

        Args:
            texts: List of text samples
            labels: List of labels (0 or 1)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum token length
        """
        self.texts = [str(t)[:512] for t in texts]  # Pre-process texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        """Get item with lazy tokenization."""
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


class BertModelWrapper:
    """Wrapper for BERT-based text classification."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize BERT model wrapper.

        Args:
            config: TrainingConfig instance with model parameters
        """
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.tokenizer = None
        self.training_history = None
        self.class_weights = None

        self._load_model()

    def _load_model(self):
        """Load pre-trained BERT model and tokenizer."""
        logger.info(f"Loading {self.config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2
        ).to(self.device)
        logger.info(f"Model loaded on {self.device}")

    def train(
        self,
        X_train: List[str],
        y_train: List[int],
        X_val: List[str],
        y_val: List[int]
    ) -> Dict:
        """
        Fine-tune BERT model.

        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels

        Returns:
            Training history dictionary
        """
        # Compute class weights if enabled
        if self.config.use_class_weights:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            self.class_weights = torch.tensor(
                class_weights, dtype=torch.float
            ).to(self.device)
            logger.info(f"Class weights: {class_weights}")
        else:
            self.class_weights = None

        # Create datasets and dataloaders
        train_dataset = BertTextDataset(
            X_train, y_train, self.tokenizer, self.config.max_text_length
        )
        val_dataset = BertTextDataset(
            X_val, y_val, self.tokenizer, self.config.max_text_length
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size
        )

        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(self.config.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training loop
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'val_roc_auc': []
        }

        best_val_accuracy = 0
        patience_counter = 0

        logger.info(f"Training {self.config.model_name} for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                # Apply class weights if enabled
                if self.class_weights is not None:
                    from torch.nn import CrossEntropyLoss
                    criterion = CrossEntropyLoss(weight=self.class_weights)
                    loss = criterion(outputs.logits, labels)

                train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()

                if (batch_idx + 1) % 50 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.config.num_epochs}, "
                        f"Batch {batch_idx + 1}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}"
                    )

            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            # Validation phase
            val_metrics = self._evaluate(val_loader)
            history['val_loss'].append(val_metrics.loss)
            history['val_accuracy'].append(val_metrics.accuracy)
            history['val_f1'].append(val_metrics.f1)
            history['val_roc_auc'].append(val_metrics.roc_auc)

            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_metrics.loss:.4f}, "
                f"Val Accuracy: {val_metrics.accuracy:.4f}"
            )

            # Early stopping
            if val_metrics.accuracy > best_val_accuracy:
                best_val_accuracy = val_metrics.accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        self.training_history = history
        return history

    def _evaluate(self, data_loader: DataLoader) -> ModelMetrics:
        """
        Evaluate model on a dataset.

        Args:
            data_loader: PyTorch DataLoader

        Returns:
            ModelMetrics instance
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_scores = []
        total_loss = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs.loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(
                    torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                )

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        roc_auc = roc_auc_score(all_labels, all_scores)
        avg_loss = total_loss / len(data_loader)

        # Per-class metrics
        class_report = classification_report(
            all_labels, all_preds,
            target_names=['Fake', 'Real'],
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        cm_dict = {
            'fake_as_fake': int(cm[0][0]),
            'fake_as_real': int(cm[0][1]),
            'real_as_fake': int(cm[1][0]),
            'real_as_real': int(cm[1][1])
        }

        per_class = {
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
        }

        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
            loss=avg_loss,
            per_class_metrics=per_class,
            confusion_matrix=cm_dict
        )

    def predict_with_threshold(
        self,
        texts: List[str],
        threshold: float = 0.5
    ) -> Tuple[List[int], List[float]]:
        """
        Make predictions with custom threshold.

        Args:
            texts: List of texts to predict
            threshold: Decision threshold (0.5 = default, higher = more "fake" predictions)

        Returns:
            Tuple of (predictions, probabilities)
        """
        dataset = BertTextDataset(
            texts, [0] * len(texts), self.tokenizer, self.config.max_text_length
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size)

        all_preds = []
        all_probs = []

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = (probs > threshold).astype(int)

                all_preds.extend(preds)
                all_probs.extend(probs)

        return all_preds, all_probs

    def save_model(self, path: str):
        """Save model to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from disk."""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path
        ).to(self.device)
        logger.info(f"Model loaded from {path}")


class DataLoader:
    """Utility for loading and preprocessing fake news datasets."""

    @staticmethod
    def load_liar(data_dir: Path) -> Tuple[List[str], List[int]]:
        """
        Load LIAR dataset.

        Args:
            data_dir: Path to LIAR directory

        Returns:
            Tuple of (texts, labels)
        """
        texts = []
        labels = []

        for tsv_file in ['train.tsv', 'valid.tsv', 'test.tsv']:
            filepath = data_dir / tsv_file
            if filepath.exists():
                df = pd.read_csv(filepath, sep='\t', low_memory=False, header=None)
                texts.extend(df[2].fillna('').astype(str).values)
                labels.extend([
                    0 if x in ['false', 'half-true', 'mostly-false'] else 1
                    for x in df[1].fillna('').astype(str).values
                ])
                logger.info(f"Loaded {tsv_file}: {len(df)} samples")

        return texts, labels

    @staticmethod
    def load_isot(data_dir: Path) -> Tuple[List[str], List[int]]:
        """
        Load ISOT dataset.

        Args:
            data_dir: Path to ISOT directory

        Returns:
            Tuple of (texts, labels)
        """
        texts = []
        labels = []

        for csv_file, label in [('True.csv', 1), ('Fake.csv', 0)]:
            filepath = data_dir / csv_file
            if filepath.exists():
                df = pd.read_csv(filepath, low_memory=False)
                text_col = 'text' if 'text' in df.columns else df.columns[1]
                texts.extend(df[text_col].fillna('').values)
                labels.extend([label] * len(df))
                logger.info(f"Loaded {csv_file}: {len(df)} samples")

        return texts, labels

    @staticmethod
    def load_fakenewsnet(filepath: Path) -> Tuple[List[str], List[int]]:
        """
        Load FakeNewsNet dataset.

        Args:
            filepath: Path to combined CSV file

        Returns:
            Tuple of (texts, labels)
        """
        texts = []
        labels = []

        if filepath.exists():
            df = pd.read_csv(filepath)
            texts.extend(df['text'].fillna('').values)
            labels.extend(df['label'].values)
            logger.info(f"Loaded FakeNewsNet: {len(df)} samples")

        return texts, labels

    @staticmethod
    def split_data(
        texts: List[str],
        labels: List[int],
        config: DataConfig
    ) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
        """
        Split data into train/validation/test sets.

        Args:
            texts: List of text samples
            labels: List of labels
            config: DataConfig instance

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        stratify = labels if config.stratify else None

        # Train/temp split
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, labels,
            test_size=(1 - config.train_size),
            random_state=config.random_state,
            stratify=stratify
        )

        # Val/test split
        val_test_ratio = config.val_size / (config.val_size + config.test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_test_ratio),
            random_state=config.random_state,
            stratify=y_temp if config.stratify else None
        )

        logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test
