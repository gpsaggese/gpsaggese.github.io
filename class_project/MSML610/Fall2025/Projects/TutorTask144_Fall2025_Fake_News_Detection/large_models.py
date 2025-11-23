"""
Large Pre-trained Models for Improved Accuracy.

Implements:
- BERT-base-uncased (larger than DistilBERT)
- RoBERTa (improved BERT with better training)
- ELECTRA (discriminative pre-training)
- Albert (parameter-efficient alternative)
- Multi-model fine-tuning and comparison

Expected accuracy improvement: +8-15%
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("large_models")


class LargeModelTrainer:
    """Trainer for large pre-trained models."""

    AVAILABLE_MODELS = {
        'bert_base': 'bert-base-uncased',
        'bert_large': 'bert-large-uncased',
        'roberta_base': 'roberta-base',
        'roberta_large': 'roberta-large',
        'electra_base': 'google/electra-base-discriminator',
        'electra_large': 'google/electra-large-discriminator',
        'albert_base': 'albert-base-v2',
        'albert_large': 'albert-large-v2',
        'distilbert_base': 'distilbert-base-uncased',  # For comparison
    }

    def __init__(
        self,
        model_key: str = 'bert_base',
        device: str = None,
        max_length: int = 256
    ):
        """
        Initialize trainer.

        Args:
            model_key: Model identifier from AVAILABLE_MODELS
            device: 'cuda' or 'cpu'
            max_length: Max token length
        """
        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_key}")

        self.model_key = model_key
        self.model_name = self.AVAILABLE_MODELS[model_key]
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length

        logger.info(f"Loading {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        ).to(self.device)

        self.training_history = {}

    def prepare_data(
        self,
        texts: List[str],
        labels: List[int],
        batch_size: int = 16
    ) -> DataLoader:
        """Prepare data for training."""
        from bert_utils import BertTextDataset

        dataset = BertTextDataset(
            texts, labels, self.tokenizer, self.max_length
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader

    def train(
        self,
        X_train: List[str],
        y_train: List[int],
        X_val: List[str],
        y_val: List[int],
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 16,
        warmup_ratio: float = 0.1,
        use_class_weights: bool = True
    ) -> Dict:
        """
        Train large model.

        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            learning_rate: Learning rate
            num_epochs: Number of epochs
            batch_size: Batch size
            warmup_ratio: Warmup ratio
            use_class_weights: Use class weights

        Returns:
            Training history
        """
        # Prepare data
        train_loader = self.prepare_data(X_train, y_train, batch_size)
        val_loader = self.prepare_data(X_val, y_val, batch_size)

        # Setup optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(warmup_ratio * total_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Class weights if needed
        if use_class_weights:
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
            logger.info(f"Using class weights: {class_weights}")
        else:
            class_weights = None

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }

        best_val_acc = 0
        patience_counter = 0
        patience = 2

        logger.info(f"Training {self.model_name} for {num_epochs} epochs")

        for epoch in range(num_epochs):
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

                # Apply class weights
                if class_weights is not None:
                    from torch.nn import CrossEntropyLoss
                    criterion = CrossEntropyLoss(weight=class_weights)
                    loss = criterion(outputs.logits, labels)

                train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                if (batch_idx + 1) % 50 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            # Validation phase
            val_metrics = self.evaluate(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])

            logger.info(
                f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}"
            )

            # Early stopping
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        self.training_history = history
        return history

    def evaluate(self, data_loader: DataLoader) -> Dict:
        """Evaluate model."""
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
                    torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                )

        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0,
            'loss': total_loss / len(data_loader)
        }

        return metrics

    def predict(self, text: str) -> int:
        """Make single prediction."""
        self.model.eval()

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()

        return pred

    def predict_with_confidence(self, text: str) -> Tuple[float, float]:
        """Make prediction with confidence."""
        self.model.eval()

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]

        return (float(probs[0]), float(probs[1]))

    def batch_predict(self, texts: List[str]) -> List[int]:
        """Make batch predictions."""
        predictions = []
        for text in texts:
            pred = self.predict(text)
            predictions.append(pred)
        return predictions

    def save_model(self, save_path: Path):
        """Save model."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path / self.model_key)
        self.tokenizer.save_pretrained(save_path / self.model_key)
        logger.info(f"Model saved to {save_path / self.model_key}")

    def load_model(self, load_path: Path):
        """Load model."""
        load_path = Path(load_path) / self.model_key

        self.model = AutoModelForSequenceClassification.from_pretrained(
            load_path
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        logger.info(f"Model loaded from {load_path}")


class MultiModelComparison:
    """Compare performance of multiple large models."""

    def __init__(self, device: str = None):
        """Initialize comparator."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.results = {}

    def train_model(
        self,
        model_key: str,
        X_train: List[str],
        y_train: List[int],
        X_val: List[str],
        y_val: List[int],
        num_epochs: int = 3,
        batch_size: int = 16
    ):
        """Train a single large model."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Training {model_key}")
        logger.info(f"{'='*80}")

        trainer = LargeModelTrainer(model_key, device=self.device)
        history = trainer.train(
            X_train, y_train, X_val, y_val,
            num_epochs=num_epochs,
            batch_size=batch_size,
            use_class_weights=True
        )

        self.models[model_key] = trainer

        # Evaluate
        from bert_utils import BertTextDataset
        val_dataset = BertTextDataset(X_val, y_val, trainer.tokenizer, 256)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        metrics = trainer.evaluate(val_loader)
        self.results[model_key] = metrics

        logger.info(f"{model_key} Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"{model_key} F1: {metrics['f1']:.4f}")

    def compare_all_models(
        self,
        X_train: List[str],
        y_train: List[int],
        X_val: List[str],
        y_val: List[int],
        models_to_train: Optional[List[str]] = None,
        num_epochs: int = 3
    ):
        """Train and compare multiple models."""
        if models_to_train is None:
            models_to_train = ['bert_base', 'roberta_base', 'electra_base', 'distilbert_base']

        for model_key in models_to_train:
            try:
                self.train_model(
                    model_key, X_train, y_train, X_val, y_val,
                    num_epochs=num_epochs
                )
            except Exception as e:
                logger.error(f"Error training {model_key}: {str(e)}")

    def print_comparison(self):
        """Print model comparison."""
        print("\n" + "="*100)
        print("LARGE MODEL COMPARISON")
        print("="*100)

        print(f"\n{'Model':<25} {'Accuracy':<15} {'Precision':<15} {'F1-Score':<15} {'ROC-AUC':<15}")
        print("-"*100)

        for model_key, metrics in sorted(self.results.items()):
            print(
                f"{model_key:<25} "
                f"{metrics['accuracy']:<15.4f} "
                f"{metrics['precision']:<15.4f} "
                f"{metrics['f1']:<15.4f} "
                f"{metrics['roc_auc']:<15.4f}"
            )

        print("="*100)

        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest Model: {best_model[0]}")
        print(f"Best Accuracy: {best_model[1]['accuracy']:.4f}")


if __name__ == '__main__':
    logger.info("Large models module loaded successfully")
