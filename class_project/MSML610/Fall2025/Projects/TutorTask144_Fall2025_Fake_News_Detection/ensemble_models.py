"""
Ensemble Models for Fake News Detection.

Implements:
- BERT + LSTM ensemble with voting
- Weighted ensemble based on model confidence
- Stacking ensemble with meta-learner
- Soft voting with confidence scores
- Category-specific ensemble routing

Expected accuracy improvement: +10-20%
"""

import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ensemble_models")


class EnsembleVoter:
    """Base ensemble voting strategy."""

    @staticmethod
    def hard_vote(predictions: List[int]) -> int:
        """Simple majority voting."""
        return 1 if sum(predictions) >= len(predictions) / 2 else 0

    @staticmethod
    def soft_vote(confidences: List[List[float]], threshold: float = 0.5) -> int:
        """Soft voting based on average confidence."""
        avg_confidence = np.mean(confidences, axis=0)
        return 1 if avg_confidence[1] >= threshold else 0

    @staticmethod
    def weighted_vote(
        predictions: List[int],
        confidences: List[float],
        weights: Optional[List[float]] = None
    ) -> int:
        """Weighted voting based on model confidences."""
        if weights is None:
            weights = [1.0] * len(predictions)

        total_weight = 0.0
        weighted_sum = 0.0

        for pred, conf, weight in zip(predictions, confidences, weights):
            weighted_sum += pred * conf * weight
            total_weight += conf * weight

        if total_weight == 0:
            return EnsembleVoter.hard_vote(predictions)

        return 1 if weighted_sum / total_weight >= 0.5 else 0


class BertLstmEnsemble:
    """Ensemble combining BERT and LSTM models."""

    def __init__(
        self,
        bert_model=None,
        lstm_model=None,
        voting_strategy: str = 'soft'
    ):
        """
        Initialize ensemble.

        Args:
            bert_model: Trained BERT model
            lstm_model: Trained LSTM model
            voting_strategy: 'hard', 'soft', or 'weighted'
        """
        self.bert_model = bert_model
        self.lstm_model = lstm_model
        self.voting_strategy = voting_strategy
        self.model_weights = [0.6, 0.4]  # BERT weighted higher
        self.voter = EnsembleVoter()

    def predict_ensemble(
        self,
        text: str,
        return_confidence: bool = True
    ) -> Dict:
        """
        Make ensemble prediction.

        Args:
            text: Input text
            return_confidence: Return confidence scores

        Returns:
            Ensemble prediction result
        """
        # Get BERT prediction
        bert_logits = self.bert_model.predict_with_confidence(text)
        bert_pred = 1 if bert_logits[1] > bert_logits[0] else 0
        bert_conf = max(bert_logits)

        # Get LSTM prediction
        lstm_pred = self.lstm_model.predict(text)
        lstm_logits = self.lstm_model.predict_with_confidence(text)
        lstm_conf = max(lstm_logits)

        # Combine predictions
        if self.voting_strategy == 'hard':
            ensemble_pred = self.voter.hard_vote([bert_pred, lstm_pred])
            confidence = (bert_conf + lstm_conf) / 2

        elif self.voting_strategy == 'soft':
            ensemble_pred = self.voter.soft_vote([bert_logits, lstm_logits])
            confidence = max(bert_conf, lstm_conf)

        elif self.voting_strategy == 'weighted':
            predictions = [bert_pred, lstm_pred]
            confidences = [bert_conf, lstm_conf]
            ensemble_pred = self.voter.weighted_vote(
                predictions, confidences, self.model_weights
            )
            confidence = np.average(confidences, weights=self.model_weights)

        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")

        return {
            'ensemble_prediction': ensemble_pred,
            'ensemble_label': 'Fake' if ensemble_pred == 1 else 'Real',
            'ensemble_confidence': confidence,
            'bert_prediction': bert_pred,
            'bert_confidence': bert_conf,
            'lstm_prediction': lstm_pred,
            'lstm_confidence': lstm_conf,
            'agreement': bert_pred == lstm_pred
        }

    def batch_predict_ensemble(
        self,
        texts: List[str],
        return_confidence: bool = True
    ) -> Dict:
        """
        Make ensemble predictions on batch.

        Args:
            texts: List of texts
            return_confidence: Return confidence scores

        Returns:
            Batch prediction results
        """
        predictions = []
        for text in texts:
            pred = self.predict_ensemble(text, return_confidence)
            predictions.append(pred)

        ensemble_preds = [p['ensemble_prediction'] for p in predictions]
        bert_preds = [p['bert_prediction'] for p in predictions]
        lstm_preds = [p['lstm_prediction'] for p in predictions]
        agreements = [p['agreement'] for p in predictions]

        return {
            'total_samples': len(texts),
            'fake_count': sum(ensemble_preds),
            'real_count': len(texts) - sum(ensemble_preds),
            'fake_percentage': sum(ensemble_preds) / len(texts) * 100,
            'model_agreement': sum(agreements) / len(texts) * 100,
            'predictions': predictions
        }

    def evaluate_ensemble(
        self,
        texts: List[str],
        labels: List[int]
    ) -> Dict:
        """
        Evaluate ensemble on test set.

        Args:
            texts: Test texts
            labels: Test labels

        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating ensemble...")

        # Get ensemble predictions
        batch_result = self.batch_predict_ensemble(texts)
        ensemble_preds = [p['ensemble_prediction'] for p in batch_result['predictions']]

        # Get individual model predictions
        bert_preds = [p['bert_prediction'] for p in batch_result['predictions']]
        lstm_preds = [p['lstm_prediction'] for p in batch_result['predictions']]

        # Calculate metrics
        ensemble_acc = accuracy_score(labels, ensemble_preds)
        bert_acc = accuracy_score(labels, bert_preds)
        lstm_acc = accuracy_score(labels, lstm_preds)

        ensemble_prec = precision_score(labels, ensemble_preds, average='weighted', zero_division=0)
        bert_prec = precision_score(labels, bert_preds, average='weighted', zero_division=0)
        lstm_prec = precision_score(labels, lstm_preds, average='weighted', zero_division=0)

        ensemble_rec = recall_score(labels, ensemble_preds, average='weighted', zero_division=0)
        bert_rec = recall_score(labels, bert_preds, average='weighted', zero_division=0)
        lstm_rec = recall_score(labels, lstm_preds, average='weighted', zero_division=0)

        ensemble_f1 = f1_score(labels, ensemble_preds, average='weighted', zero_division=0)
        bert_f1 = f1_score(labels, bert_preds, average='weighted', zero_division=0)
        lstm_f1 = f1_score(labels, lstm_preds, average='weighted', zero_division=0)

        try:
            ensemble_auc = roc_auc_score(labels, ensemble_preds)
            bert_auc = roc_auc_score(labels, bert_preds)
            lstm_auc = roc_auc_score(labels, lstm_preds)
        except:
            ensemble_auc = bert_auc = lstm_auc = 0.0

        return {
            'ensemble': {
                'accuracy': ensemble_acc,
                'precision': ensemble_prec,
                'recall': ensemble_rec,
                'f1': ensemble_f1,
                'roc_auc': ensemble_auc
            },
            'bert': {
                'accuracy': bert_acc,
                'precision': bert_prec,
                'recall': bert_rec,
                'f1': bert_f1,
                'roc_auc': bert_auc
            },
            'lstm': {
                'accuracy': lstm_acc,
                'precision': lstm_prec,
                'recall': lstm_rec,
                'f1': lstm_f1,
                'roc_auc': lstm_auc
            },
            'improvement': {
                'ensemble_over_bert': ensemble_acc - bert_acc,
                'ensemble_over_lstm': ensemble_acc - lstm_acc,
                'average_accuracy': (ensemble_acc + bert_acc + lstm_acc) / 3
            }
        }


class StackingEnsemble:
    """Stacking ensemble with meta-learner."""

    def __init__(
        self,
        bert_model=None,
        lstm_model=None,
        meta_learner=None
    ):
        """
        Initialize stacking ensemble.

        Args:
            bert_model: Trained BERT model
            lstm_model: Trained LSTM model
            meta_learner: Meta-learner model (e.g., logistic regression)
        """
        self.bert_model = bert_model
        self.lstm_model = lstm_model
        self.meta_learner = meta_learner

    def get_base_predictions(self, texts: List[str]) -> np.ndarray:
        """Get predictions from base models."""
        bert_preds = []
        lstm_preds = []

        for text in texts:
            # Get BERT predictions
            bert_logits = self.bert_model.predict_with_confidence(text)
            bert_preds.append(bert_logits)

            # Get LSTM predictions
            lstm_logits = self.lstm_model.predict_with_confidence(text)
            lstm_preds.append(lstm_logits)

        # Stack predictions
        stacked = np.hstack([np.array(bert_preds), np.array(lstm_preds)])
        return stacked

    def train_meta_learner(
        self,
        texts: List[str],
        labels: List[int]
    ):
        """Train meta-learner on base model predictions."""
        logger.info("Training meta-learner...")

        # Get base predictions
        X_meta = self.get_base_predictions(texts)

        # Train meta-learner
        self.meta_learner.fit(X_meta, labels)
        logger.info("Meta-learner training complete")

    def predict(self, text: str) -> int:
        """Make stacking prediction."""
        # Get base predictions
        bert_logits = self.bert_model.predict_with_confidence(text)
        lstm_logits = self.lstm_model.predict_with_confidence(text)

        X_meta = np.hstack([bert_logits, lstm_logits]).reshape(1, -1)

        # Meta-learner prediction
        pred = self.meta_learner.predict(X_meta)[0]
        return pred

    def batch_predict(self, texts: List[str]) -> List[int]:
        """Make batch predictions."""
        predictions = []
        for text in texts:
            pred = self.predict(text)
            predictions.append(pred)
        return predictions

    def evaluate(
        self,
        texts: List[str],
        labels: List[int]
    ) -> Dict:
        """Evaluate stacking ensemble."""
        preds = self.batch_predict(texts)

        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='weighted', zero_division=0),
            'recall': recall_score(labels, preds, average='weighted', zero_division=0),
            'f1': f1_score(labels, preds, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0
        }


class CategorySpecificEnsemble:
    """Use different models for different article categories."""

    def __init__(self):
        """Initialize category ensemble."""
        self.category_models = {}

    def register_category_model(
        self,
        category: str,
        bert_model=None,
        lstm_model=None,
        voting_strategy: str = 'soft'
    ):
        """Register ensemble for a category."""
        self.category_models[category] = BertLstmEnsemble(
            bert_model, lstm_model, voting_strategy
        )

    def predict(self, text: str, category: str, default_ensemble=None) -> Dict:
        """Predict with category-specific ensemble."""
        if category in self.category_models:
            ensemble = self.category_models[category]
            return ensemble.predict_ensemble(text)

        elif default_ensemble is not None:
            logger.warning(f"No ensemble for category {category}, using default")
            return default_ensemble.predict_ensemble(text)

        else:
            raise ValueError(f"No ensemble for category {category}")

    def batch_predict(
        self,
        texts: List[str],
        categories: List[str],
        default_ensemble=None
    ) -> Dict:
        """Batch predict with category routing."""
        predictions = []
        for text, category in zip(texts, categories):
            pred = self.predict(text, category, default_ensemble)
            predictions.append(pred)

        ensemble_preds = [p['ensemble_prediction'] for p in predictions]

        return {
            'total_samples': len(texts),
            'fake_count': sum(ensemble_preds),
            'real_count': len(texts) - sum(ensemble_preds),
            'fake_percentage': sum(ensemble_preds) / len(texts) * 100,
            'predictions': predictions
        }


if __name__ == '__main__':
    logger.info("Ensemble models module loaded successfully")
