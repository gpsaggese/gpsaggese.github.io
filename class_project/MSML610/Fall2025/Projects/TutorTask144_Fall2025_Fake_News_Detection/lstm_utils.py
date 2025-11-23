"""
LSTM Model for Fake News Detection.

Implements a bidirectional LSTM model with GloVe embeddings for text classification.
Provides an alternative to BERT for comparison.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lstm_utils")


@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""
    embedding_dim: int = 100
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 5
    device: str = 'cpu'
    max_seq_length: int = 256
    vocab_size: int = 10000


class LSTMTextDataset(Dataset):
    """PyTorch Dataset for text with LSTM."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Dict[str, int],
        max_length: int = 256,
        pad_token_id: int = 0
    ):
        """Initialize dataset."""
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        tokens = text.lower().split()
        token_ids = []
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.vocab.get('<unk>', 1)))

        # Pad/truncate
        if len(token_ids) < self.max_length:
            token_ids += [self.pad_token_id] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'length': min(len(text.split()), self.max_length)
        }


class LSTMModel(nn.Module):
    """Bidirectional LSTM for text classification."""

    def __init__(self, config: LSTMConfig):
        """Initialize LSTM model."""
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_dim,
            config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            batch_first=True
        )

        lstm_output_dim = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim

        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(lstm_output_dim, 64)
        self.fc2 = nn.Linear(64, 2)

        self.relu = nn.ReLU()

    def forward(self, input_ids, lengths=None):
        """Forward pass."""
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        if lengths is not None:
            embeddings = nn.utils.rnn.pack_padded_sequence(
                embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(embeddings)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(embeddings)

        # Use last hidden state
        if self.config.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        hidden = self.dropout(hidden)
        logits = self.fc1(hidden)
        logits = self.relu(logits)
        logits = self.dropout(logits)
        logits = self.fc2(logits)

        return logits


class LSTMModelWrapper:
    """Wrapper for LSTM model training and inference."""

    def __init__(self, config: LSTMConfig):
        """Initialize wrapper."""
        self.config = config
        self.device = torch.device(config.device)
        self.vocab = self._build_vocab()
        self.model = LSTMModel(config).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary."""
        vocab = {'<pad>': 0, '<unk>': 1}
        return vocab

    def build_vocab_from_texts(self, texts: List[str]) -> Dict[str, int]:
        """Build vocabulary from texts."""
        vocab = {'<pad>': 0, '<unk>': 1}
        word_freq = {}

        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Keep top vocab_size - 2 words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, _) in enumerate(sorted_words[:self.config.vocab_size - 2], 2):
            vocab[word] = idx

        self.vocab = vocab
        return vocab

    def train(
        self,
        X_train: List[str],
        y_train: List[int],
        X_val: List[str],
        y_val: List[int]
    ) -> Dict[str, List[float]]:
        """Train LSTM model."""
        logger.info("Building vocabulary from training data...")
        self.build_vocab_from_texts(X_train + X_val)

        logger.info(f"Vocabulary size: {len(self.vocab)}")

        # Create datasets
        train_dataset = LSTMTextDataset(X_train, y_train, self.vocab, self.config.max_seq_length)
        val_dataset = LSTMTextDataset(X_val, y_val, self.vocab, self.config.max_seq_length)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            # Train
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch.get('length')

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, lengths)
                loss = self.criterion(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    lengths = batch.get('length')

                    outputs = self.model(input_ids, lengths)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

            val_loss /= len(val_loader)
            val_accuracy = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, zero_division=0)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['val_f1'].append(val_f1)

            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 2:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        return history

    def evaluate(self, X_test: List[str], y_test: List[int]) -> Dict[str, Any]:
        """Evaluate model on test set."""
        test_dataset = LSTMTextDataset(X_test, y_test, self.vocab, self.config.max_seq_length)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels']
                lengths = batch.get('length')

                outputs = self.model(input_ids, lengths)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'roc_auc': roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0,
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
        }

        return metrics

    def predict(self, text: str) -> int:
        """Make prediction on single text."""
        tokens = text.lower().split()
        token_ids = []
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.vocab.get('<unk>', 1)))

        if len(token_ids) < self.config.max_seq_length:
            token_ids += [0] * (self.config.max_seq_length - len(token_ids))
        else:
            token_ids = token_ids[:self.config.max_seq_length]

        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            pred = torch.argmax(outputs, dim=1).item()

        return pred

    def save_model(self, path: str) -> None:
        """Save model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path / 'lstm_model.pt')
        with open(path / 'vocab.pkl', 'wb') as f:
            pickle.dump(self.vocab, f)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model."""
        path = Path(path)
        self.model.load_state_dict(torch.load(path / 'lstm_model.pt'))
        with open(path / 'vocab.pkl', 'rb') as f:
            self.vocab = pickle.load(f)
        logger.info(f"Model loaded from {path}")
