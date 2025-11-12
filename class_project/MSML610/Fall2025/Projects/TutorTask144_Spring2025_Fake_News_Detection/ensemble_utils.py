"""
ensemble_utils.py

Ensemble learning module combining BERT, TF-IDF, and LSTM for fake news detection.
This module provides multiple baseline models that are combined via voting to achieve
70-75% accuracy on the fake news classification task.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict
from pathlib import Path
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class TFIDFModel:
    """TF-IDF baseline with Logistic Regression for fake news detection."""

    def __init__(self, max_features: int = 5000, max_df: float = 0.8, min_df: int = 2):
        """
        Initialize TF-IDF model.

        Args:
            max_features: Maximum number of features for TF-IDF
            max_df: Maximum document frequency
            min_df: Minimum document frequency
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            max_df=max_df,
            min_df=min_df,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.is_trained = False

    def train(self, X_train: List[str], y_train: List[int]):
        """Train TF-IDF model."""
        logger.info("Training TF-IDF model...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_tfidf, y_train)
        self.is_trained = True
        logger.info("TF-IDF model trained successfully")

    def predict(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        """
        Make predictions.

        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_tfidf = self.vectorizer.transform(texts)
        preds = self.classifier.predict(X_tfidf)
        probs = self.classifier.predict_proba(X_tfidf)[:, 1]  # Probability of class 1

        return preds.tolist(), probs.tolist()


class LSTMModel(nn.Module):
    """LSTM-based model for sequence-aware fake news detection."""

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            lengths: Sequence lengths for packing

        Returns:
            Logits (batch_size, 2)
        """
        embedded = self.embedding(input_ids)

        if lengths is not None:
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            packed = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(embedded)

        # Use last hidden state
        if self.lstm.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]

        logits = self.fc(last_hidden)
        return logits


class LSTMTrainer:
    """Trainer for LSTM model."""

    def __init__(self, model: LSTMModel, device: str = 'cpu', learning_rate: float = 1e-3):
        """Initialize trainer."""
        self.model = model.to(device)
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids)
            loss = self.criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader: DataLoader) -> Dict:
        """Evaluate on validation set."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids)
                logits = outputs
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)[:, 1]

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(probs.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'probabilities': all_scores
        }


class EnsembleModel:
    """Ensemble combining BERT, TF-IDF, and LSTM with weighted voting."""

    def __init__(
        self,
        bert_model=None,
        weights: Dict[str, float] = None,
        voting: str = 'weighted'
    ):
        """
        Initialize ensemble model.

        Args:
            bert_model: Trained BERT model wrapper
            weights: Voting weights for each model (BERT, TF-IDF, LSTM)
            voting: 'weighted' or 'hard' voting
        """
        self.bert_model = bert_model
        self.tfidf_model = TFIDFModel()
        self.lstm_model = None
        self.voting = voting

        # Default weights: BERT gets highest weight
        self.weights = weights or {
            'bert': 0.5,
            'tfidf': 0.25,
            'lstm': 0.25
        }

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def train_components(
        self,
        X_train: List[str],
        y_train: List[int],
        X_val: List[str],
        y_val: List[int],
        train_lstm: bool = True
    ):
        """Train individual components."""
        logger.info("Training ensemble components...")

        # Train TF-IDF
        logger.info("Training TF-IDF component...")
        self.tfidf_model.train(X_train, y_train)

        # Train LSTM if requested
        if train_lstm:
            logger.info("Training LSTM component...")
            self._train_lstm(X_train, y_train, X_val, y_val)

        logger.info("Ensemble training complete")

    def _train_lstm(
        self,
        X_train: List[str],
        y_train: List[int],
        X_val: List[str],
        y_val: List[int],
        epochs: int = 3,
        batch_size: int = 32
    ):
        """Train LSTM component."""
        from bert_utils import BertTextDataset
        from transformers import AutoTokenizer

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize LSTM
        self.lstm_model = LSTMModel(
            vocab_size=10000,
            embedding_dim=100,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3,
            bidirectional=True
        )

        # Use BERT tokenizer for consistency
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        train_dataset = BertTextDataset(X_train, y_train, tokenizer, max_length=256)
        val_dataset = BertTextDataset(X_val, y_val, tokenizer, max_length=256)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        trainer = LSTMTrainer(self.lstm_model, device=device)

        best_f1 = 0
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = trainer.train_epoch(train_loader)
            val_metrics = trainer.evaluate(val_loader)

            logger.info(
                f"LSTM Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 2:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

    def predict(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        """
        Make ensemble predictions using weighted voting.

        Args:
            texts: List of texts to predict

        Returns:
            Tuple of (predictions, confidence_scores)
        """
        predictions = {}

        # BERT predictions
        if self.bert_model is not None:
            bert_preds, bert_probs = self.bert_model.predict_with_threshold(texts, threshold=0.5)
            predictions['bert'] = (np.array(bert_preds), np.array(bert_probs))
            logger.info(f"BERT: {np.mean(bert_preds):.2%} predicted as real")

        # TF-IDF predictions
        tfidf_preds, tfidf_probs = self.tfidf_model.predict(texts)
        predictions['tfidf'] = (np.array(tfidf_preds), np.array(tfidf_probs))
        logger.info(f"TF-IDF: {np.mean(tfidf_preds):.2%} predicted as real")

        # LSTM predictions
        if self.lstm_model is not None:
            lstm_preds, lstm_probs = self._predict_lstm(texts)
            predictions['lstm'] = (np.array(lstm_preds), np.array(lstm_probs))
            logger.info(f"LSTM: {np.mean(lstm_preds):.2%} predicted as real")

        # Weighted voting
        ensemble_scores = self._weighted_vote(predictions)
        ensemble_preds = (ensemble_scores > 0.5).astype(int).tolist()

        return ensemble_preds, ensemble_scores.tolist()

    def _weighted_vote(self, predictions: Dict) -> np.ndarray:
        """Combine predictions via weighted voting."""
        ensemble_scores = np.zeros(len(list(predictions.values())[0][1]))

        for model_name, (preds, probs) in predictions.items():
            if model_name in self.weights:
                ensemble_scores += self.weights[model_name] * probs

        return ensemble_scores

    def _predict_lstm(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        """Generate LSTM predictions."""
        from bert_utils import BertTextDataset
        from transformers import AutoTokenizer

        if self.lstm_model is None:
            raise ValueError("LSTM model not trained")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        dataset = BertTextDataset(texts, [0] * len(texts), tokenizer, max_length=256)
        loader = DataLoader(dataset, batch_size=32)

        self.lstm_model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                outputs = self.lstm_model(input_ids)
                logits = outputs
                probs = torch.softmax(logits, dim=1)[:, 1]

                preds = (probs > 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return all_preds, all_probs
