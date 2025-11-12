"""
confidence_utils.py

Uncertainty quantification module for adding confidence scores to predictions.
Provides Bayesian approximation via dropout-based uncertainty estimation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ConfidenceEstimator:
    """
    Estimate prediction confidence using Monte Carlo dropout.

    This approach performs multiple forward passes through the model with
    dropout enabled, treating dropout as Bayesian approximation to sample
    from the posterior distribution.
    """

    def __init__(self, model, device: str = 'cpu', num_samples: int = 10):
        """
        Initialize confidence estimator.

        Args:
            model: Trained model with dropout
            device: Device to run on
            num_samples: Number of MC samples for uncertainty estimation
        """
        self.model = model
        self.device = torch.device(device)
        self.num_samples = num_samples

    def enable_dropout(self):
        """Enable dropout in eval mode (MC dropout)."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def disable_dropout(self):
        """Disable dropout for standard evaluation."""
        self.model.eval()

    def estimate_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate prediction uncertainty using MC dropout.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask

        Returns:
            Tuple of (mean_logits, std_logits, entropy)
        """
        samples = []

        # Perform multiple forward passes
        for i in range(self.num_samples):
            self.enable_dropout()

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device)
                )
                logits = outputs.logits
                samples.append(logits.cpu().numpy())

            self.disable_dropout()

        # Convert samples to array
        samples = np.array(samples)  # (num_samples, batch_size, num_classes)

        # Compute statistics
        mean_logits = np.mean(samples, axis=0)
        std_logits = np.std(samples, axis=0)

        # Compute entropy as measure of uncertainty
        probs = self._softmax(mean_logits)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

        return mean_logits, std_logits, entropy

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)


class ConfidenceScorer:
    """
    Generate confidence scores for predictions using multiple methods.
    """

    def __init__(self, model=None, device: str = 'cpu'):
        """
        Initialize scorer.

        Args:
            model: Trained BERT model
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.estimator = None
        if model:
            self.estimator = ConfidenceEstimator(model.model, device=device)

    def predict_with_confidence(
        self,
        texts: List[str],
        method: str = 'probability',
        threshold: float = 0.5
    ) -> Dict:
        """
        Make predictions with confidence scores.

        Args:
            texts: List of texts to predict
            method: Confidence estimation method
                   - 'probability': Max softmax probability
                   - 'entropy': Information entropy
                   - 'margin': Margin between top 2 predictions
            threshold: Decision threshold for predictions

        Returns:
            Dictionary with predictions, probabilities, and confidence scores
        """
        if self.model is None:
            raise ValueError("Model must be provided to scorer")

        # Get predictions and probabilities
        preds, probs = self.model.predict_with_threshold(texts, threshold=threshold)
        probs = np.array(probs)

        if method == 'probability':
            # Confidence = max probability
            confidence = np.max([probs, 1 - probs], axis=0)

        elif method == 'entropy':
            # Uncertainty from entropy
            # Low entropy = high confidence
            p0 = 1 - probs
            p1 = probs
            entropy = -(p0 * np.log(p0 + 1e-10) + p1 * np.log(p1 + 1e-10))
            confidence = 1 - (entropy / np.log(2))  # Normalize by max entropy

        elif method == 'margin':
            # Margin between top 2 probabilities
            top_2 = np.sort(np.array([probs, 1 - probs]))[::-1]
            margin = top_2[0] - top_2[1]
            confidence = margin

        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            'predictions': preds,
            'probabilities': probs.tolist(),
            'confidence': confidence.tolist(),
            'high_confidence': [c >= 0.7 for c in confidence],
            'method': method
        }

    def predict_with_uncertainty(
        self,
        tokenizer,
        texts: List[str],
        batch_size: int = 16,
        threshold: float = 0.5,
        num_samples: int = 10
    ) -> Dict:
        """
        Make predictions with Bayesian uncertainty estimates.

        Args:
            tokenizer: BERT tokenizer
            texts: List of texts
            batch_size: Batch size
            threshold: Decision threshold
            num_samples: Number of MC samples

        Returns:
            Predictions with uncertainty estimates
        """
        from bert_utils import BertTextDataset
        from torch.utils.data import DataLoader as TorchDataLoader

        if self.estimator is None:
            raise ValueError("Estimator not initialized")

        dataset = BertTextDataset(texts, [0] * len(texts), tokenizer, max_length=256)
        loader = TorchDataLoader(dataset, batch_size=batch_size)

        all_means = []
        all_stds = []
        all_entropy = []
        all_preds = []

        for batch in loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            mean_logits, std_logits, entropy = self.estimator.estimate_uncertainty(
                input_ids, attention_mask
            )

            all_means.append(mean_logits)
            all_stds.append(std_logits)
            all_entropy.append(entropy)

            # Get predictions from mean logits
            probs = self.estimator._softmax(mean_logits)
            preds = (probs[:, 1] > threshold).astype(int)
            all_preds.extend(preds)

        # Concatenate results
        mean_logits = np.concatenate(all_means, axis=0)
        std_logits = np.concatenate(all_stds, axis=0)
        entropy = np.concatenate(all_entropy, axis=0)

        # Convert logits to probabilities
        mean_probs = self.estimator._softmax(mean_logits)[:, 1]
        std_probs = std_logits[:, 1]  # Approximate std of probabilities

        return {
            'predictions': all_preds,
            'mean_probabilities': mean_probs.tolist(),
            'std_probabilities': std_probs.tolist(),
            'entropy': entropy.tolist(),
            'confidence': (1 - entropy / np.log(2)).tolist(),
            'high_confidence': [e < np.log(2) * 0.3 for e in entropy],  # Low entropy
            'uncertain': [e > np.log(2) * 0.7 for e in entropy]  # High entropy
        }


class CalibrationMetrics:
    """Compute calibration metrics for confidence scores."""

    @staticmethod
    def expected_calibration_error(
        predictions: List[int],
        probabilities: List[float],
        labels: List[int],
        num_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE measures average difference between confidence and accuracy.

        Args:
            predictions: Model predictions
            probabilities: Predicted probabilities
            labels: True labels
            num_bins: Number of bins for calibration

        Returns:
            ECE score (0 = perfectly calibrated, 1 = poorly calibrated)
        """
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        labels = np.array(labels)

        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

        total_samples = len(labels)
        ece = 0

        for i in range(num_bins):
            bin_mask = (probabilities >= bin_boundaries[i]) & (probabilities < bin_boundaries[i + 1])

            if np.sum(bin_mask) == 0:
                continue

            bin_accuracy = np.mean(predictions[bin_mask] == labels[bin_mask])
            bin_confidence = np.mean(probabilities[bin_mask])
            bin_size = np.sum(bin_mask)

            ece += np.abs(bin_accuracy - bin_confidence) * (bin_size / total_samples)

        return ece

    @staticmethod
    def maximum_calibration_error(
        predictions: List[int],
        probabilities: List[float],
        labels: List[int],
        num_bins: int = 10
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).

        MCE is the maximum difference between accuracy and confidence
        across all bins.

        Args:
            predictions: Model predictions
            probabilities: Predicted probabilities
            labels: True labels
            num_bins: Number of bins

        Returns:
            MCE score
        """
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        labels = np.array(labels)

        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        mce = 0

        for i in range(num_bins):
            bin_mask = (probabilities >= bin_boundaries[i]) & (probabilities < bin_boundaries[i + 1])

            if np.sum(bin_mask) == 0:
                continue

            bin_accuracy = np.mean(predictions[bin_mask] == labels[bin_mask])
            bin_confidence = np.mean(probabilities[bin_mask])

            mce = max(mce, np.abs(bin_accuracy - bin_confidence))

        return mce

    @staticmethod
    def brier_score(
        probabilities: List[float],
        labels: List[int]
    ) -> float:
        """
        Compute Brier Score (mean squared error of probabilities).

        Lower is better. Range: [0, 1]

        Args:
            probabilities: Predicted probabilities
            labels: True labels

        Returns:
            Brier score
        """
        probabilities = np.array(probabilities)
        labels = np.array(labels)

        return np.mean((probabilities - labels) ** 2)


class ConfidenceThresholdAnalyzer:
    """Analyze performance across different confidence thresholds."""

    @staticmethod
    def analyze(
        predictions: List[int],
        confidences: List[float],
        labels: List[int],
        thresholds: List[float] = None
    ) -> Dict:
        """
        Analyze predictions filtered by confidence threshold.

        Args:
            predictions: Model predictions
            confidences: Confidence scores
            labels: True labels
            thresholds: Thresholds to analyze (default: [0.5, 0.6, 0.7, 0.8, 0.9])

        Returns:
            Dictionary with metrics per threshold
        """
        if thresholds is None:
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

        predictions = np.array(predictions)
        confidences = np.array(confidences)
        labels = np.array(labels)

        results = {}

        for threshold in thresholds:
            mask = confidences >= threshold
            if np.sum(mask) == 0:
                results[threshold] = {
                    'samples': 0,
                    'coverage': 0,
                    'accuracy': None
                }
                continue

            high_conf_preds = predictions[mask]
            high_conf_labels = labels[mask]

            accuracy = np.mean(high_conf_preds == high_conf_labels)
            coverage = np.sum(mask) / len(labels)

            results[threshold] = {
                'samples': int(np.sum(mask)),
                'coverage': float(coverage),
                'accuracy': float(accuracy)
            }

        return results


def add_confidence_to_predictions(
    model,
    texts: List[str],
    threshold: float = 0.5,
    confidence_method: str = 'probability'
) -> Dict:
    """
    Convenience function to add confidence scores to predictions.

    Args:
        model: Trained BertModelWrapper
        texts: List of texts to predict
        threshold: Decision threshold
        confidence_method: Method for confidence scoring

    Returns:
        Dictionary with predictions and confidence scores
    """
    scorer = ConfidenceScorer(model=model)
    return scorer.predict_with_confidence(
        texts,
        method=confidence_method,
        threshold=threshold
    )
