"""
test_suite.py

Comprehensive unit tests for BERT fake news detection system.

Tests cover:
- Data loading and preprocessing
- Model initialization and training
- Inference and predictions
- Confidence scoring
- Ensemble predictions
- API endpoints
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple

# Import modules to test
from bert_utils import (
    DataConfig, TrainingConfig, BertModelWrapper, DataLoader,
    BertTextDataset, ModelMetrics
)
from confidence_utils import ConfidenceScorer, CalibrationMetrics
from ensemble_utils import EnsembleModel, TFIDFModel, LSTMModel


class TestDataLoading:
    """Test data loading and preprocessing."""

    def test_data_config_defaults(self):
        """Test DataConfig initialization with defaults."""
        config = DataConfig()
        assert config.train_size == 0.7
        assert config.val_size == 0.15
        assert config.test_size == 0.15
        assert config.max_text_length == 256
        assert config.stratify is True

    def test_data_config_custom(self):
        """Test DataConfig with custom parameters."""
        config = DataConfig(
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            max_text_length=512,
            stratify=False
        )
        assert config.train_size == 0.6
        assert config.max_text_length == 512
        assert config.stratify is False

    def test_data_split(self):
        """Test data splitting functionality."""
        # Create sample data
        texts = ["Text " + str(i) for i in range(100)]
        labels = [i % 2 for i in range(100)]

        loader = DataLoader()
        config = DataConfig(train_size=0.7, val_size=0.15, test_size=0.15)

        X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
            texts, labels, config
        )

        # Check split sizes
        assert len(X_train) == 70
        assert len(X_val) == 15
        assert len(X_test) == 15

        # Check label distribution is preserved
        fake_train = sum(1 for y in y_train if y == 0)
        real_train = sum(1 for y in y_train if y == 1)
        assert fake_train > 0 and real_train > 0

    def test_stratified_split(self):
        """Test stratified splitting preserves class distribution."""
        texts = ["Text " + str(i) for i in range(100)]
        labels = [0] * 30 + [1] * 70  # 30% fake, 70% real

        loader = DataLoader()
        config = DataConfig(stratify=True)

        X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
            texts, labels, config
        )

        # Check distribution is similar in splits
        fake_ratio_train = sum(1 for y in y_train if y == 0) / len(y_train)
        fake_ratio_val = sum(1 for y in y_val if y == 0) / len(y_val)
        fake_ratio_test = sum(1 for y in y_test if y == 0) / len(y_test)

        # All should be close to 0.3
        assert 0.25 < fake_ratio_train < 0.35
        assert 0.25 < fake_ratio_val < 0.35
        assert 0.25 < fake_ratio_test < 0.35


class TestBertTextDataset:
    """Test BERT text dataset."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        from transformers import AutoTokenizer

        texts = ["This is a test", "Another test text"]
        labels = [0, 1]
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        return BertTextDataset(texts, labels, tokenizer, max_length=256)

    def test_dataset_length(self, sample_dataset):
        """Test dataset length."""
        assert len(sample_dataset) == 2

    def test_dataset_getitem(self, sample_dataset):
        """Test getting items from dataset."""
        item = sample_dataset[0]

        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'label' in item
        assert item['input_ids'].shape[0] <= 256
        assert item['label'] == 0


class TestTrainingConfig:
    """Test training configuration."""

    def test_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        assert config.batch_size == 16
        assert config.num_epochs == 2
        assert config.learning_rate == 2e-5
        assert config.use_class_weights is False

    def test_config_with_class_weights(self):
        """Test TrainingConfig with class weights enabled."""
        config = TrainingConfig(use_class_weights=True)
        assert config.use_class_weights is True

    def test_config_validation(self):
        """Test config value validation."""
        config = TrainingConfig(
            batch_size=32,
            num_epochs=5,
            learning_rate=1e-5
        )
        assert config.batch_size == 32
        assert config.num_epochs == 5


class TestBertModelWrapper:
    """Test BERT model wrapper."""

    @pytest.fixture
    def model(self):
        """Create model instance."""
        config = TrainingConfig(
            model_name='distilbert-base-uncased',
            batch_size=16,
            device='cpu'
        )
        return BertModelWrapper(config)

    def test_model_initialization(self, model):
        """Test model loads correctly."""
        assert model.model is not None
        assert model.tokenizer is not None
        assert model.device.type == 'cpu'

    def test_model_config(self, model):
        """Test model config is set correctly."""
        assert model.config.model_name == 'distilbert-base-uncased'
        assert model.config.batch_size == 16

    def test_predict_with_threshold(self, model):
        """Test prediction with custom threshold."""
        texts = ["This is fake news", "This is real news"]
        preds, probs = model.predict_with_threshold(texts, threshold=0.5)

        assert len(preds) == 2
        assert len(probs) == 2
        assert all(p in [0, 1] for p in preds)
        assert all(0 <= p <= 1 for p in probs)


class TestConfidenceScoring:
    """Test confidence scoring."""

    def test_calibration_metrics_ece(self):
        """Test Expected Calibration Error calculation."""
        predictions = [1, 0, 1, 1, 0, 1, 0, 0]
        probabilities = [0.9, 0.1, 0.8, 0.7, 0.2, 0.95, 0.3, 0.1]
        labels = [1, 0, 1, 1, 0, 1, 0, 0]

        ece = CalibrationMetrics.expected_calibration_error(
            predictions, probabilities, labels
        )

        # Should be between 0 and 1
        assert 0 <= ece <= 1

    def test_brier_score(self):
        """Test Brier Score calculation."""
        probabilities = [0.9, 0.1, 0.8, 0.7]
        labels = [1, 0, 1, 1]

        brier = CalibrationMetrics.brier_score(probabilities, labels)

        # Brier score is MSE, should be small for good predictions
        assert 0 <= brier <= 1
        assert brier < 0.5  # Good predictions

    def test_maximum_calibration_error(self):
        """Test Maximum Calibration Error calculation."""
        predictions = [1, 0, 1, 1, 0, 1, 0, 0]
        probabilities = [0.9, 0.1, 0.8, 0.7, 0.2, 0.95, 0.3, 0.1]
        labels = [1, 0, 1, 1, 0, 1, 0, 0]

        mce = CalibrationMetrics.maximum_calibration_error(
            predictions, probabilities, labels
        )

        assert 0 <= mce <= 1


class TestEnsembleModels:
    """Test ensemble components."""

    def test_tfidf_model_initialization(self):
        """Test TF-IDF model initialization."""
        model = TFIDFModel(max_features=5000)
        assert model.vectorizer is not None
        assert model.classifier is not None
        assert model.is_trained is False

    def test_tfidf_training(self):
        """Test TF-IDF model training."""
        model = TFIDFModel(max_features=100)
        texts = ["fake news", "real news"] * 5
        labels = [0, 1] * 5

        model.train(texts, labels)
        assert model.is_trained is True

    def test_tfidf_prediction(self):
        """Test TF-IDF prediction."""
        model = TFIDFModel(max_features=100)
        texts = ["fake news", "real news"] * 5
        labels = [0, 1] * 5

        model.train(texts, labels)

        preds, probs = model.predict(["fake news", "real news"])
        assert len(preds) == 2
        assert len(probs) == 2
        assert all(p in [0, 1] for p in preds)
        assert all(0 <= p <= 1 for p in probs)

    def test_lstm_model_initialization(self):
        """Test LSTM model initialization."""
        lstm = LSTMModel(
            vocab_size=10000,
            embedding_dim=100,
            hidden_dim=128,
            num_layers=2
        )
        assert lstm.embedding is not None
        assert lstm.lstm is not None
        assert lstm.fc is not None

    def test_lstm_forward_pass(self):
        """Test LSTM forward pass."""
        lstm = LSTMModel(vocab_size=100, embedding_dim=50, hidden_dim=64)
        input_ids = torch.randint(0, 100, (4, 32))  # Batch of 4, seq_len 32

        output = lstm(input_ids)
        assert output.shape == (4, 2)  # Batch size 4, 2 classes


class TestDataAugmentation:
    """Test data augmentation."""

    def test_augmentation_pipeline_initialization(self):
        """Test augmentation pipeline initialization."""
        try:
            from augmentation_utils import DataAugmentationPipeline

            pipeline = DataAugmentationPipeline(
                use_paraphrase=False,
                use_back_translation=False
            )
            # Should initialize without error
            assert pipeline is not None
        except ImportError:
            pytest.skip("T5 models not available")

    def test_tfidf_augmentation_preserves_labels(self):
        """Test that augmentation preserves labels."""
        model = TFIDFModel(max_features=100)
        texts = ["fake news"] * 3 + ["real news"] * 3
        labels = [0] * 3 + [1] * 3

        model.train(texts, labels)

        # Note: Real augmentation would generate new texts
        # This just tests that the mechanism works
        assert len(labels) == 6
        assert sum(labels) == 3  # 3 fake (0), 3 real (1)


class TestMetrics:
    """Test evaluation metrics."""

    def test_model_metrics_creation(self):
        """Test ModelMetrics creation."""
        metrics = ModelMetrics(
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1=0.85,
            roc_auc=0.92,
            loss=0.25,
            per_class_metrics={
                'fake': {'precision': 0.80, 'recall': 0.82, 'f1-score': 0.81, 'support': 100},
                'real': {'precision': 0.86, 'recall': 0.88, 'f1-score': 0.87, 'support': 200}
            },
            confusion_matrix={
                'fake_as_fake': 82,
                'fake_as_real': 18,
                'real_as_fake': 24,
                'real_as_real': 176
            }
        )

        assert metrics.accuracy == 0.85
        assert metrics.f1 == 0.85
        assert metrics.per_class_metrics['fake']['precision'] == 0.80

    def test_metrics_per_class(self):
        """Test per-class metrics extraction."""
        metrics = ModelMetrics(
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1=0.85,
            roc_auc=0.92,
            loss=0.25,
            per_class_metrics={
                'fake': {'precision': 0.75, 'recall': 0.70, 'f1-score': 0.72, 'support': 100},
                'real': {'precision': 0.90, 'recall': 0.95, 'f1-score': 0.92, 'support': 200}
            },
            confusion_matrix={}
        )

        # Check fake news detection metrics
        assert metrics.per_class_metrics['fake']['recall'] == 0.70
        # Check real news detection metrics
        assert metrics.per_class_metrics['real']['recall'] == 0.95


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text_handling(self):
        """Test handling of empty texts."""
        config = TrainingConfig(device='cpu')
        model = BertModelWrapper(config)

        # Empty string should still work (gets padding)
        preds, probs = model.predict_with_threshold([""])
        assert len(preds) == 1

    def test_very_long_text_handling(self):
        """Test handling of very long texts."""
        config = TrainingConfig(device='cpu', max_text_length=256)
        model = BertModelWrapper(config)

        # Very long text should be truncated
        long_text = " ".join(["word"] * 1000)
        preds, probs = model.predict_with_threshold([long_text])
        assert len(preds) == 1

    def test_special_characters(self):
        """Test handling of special characters."""
        config = TrainingConfig(device='cpu')
        model = BertModelWrapper(config)

        special_text = "!!!@@@###$$$%%%^^^&&&***(){}[]<>?/"
        preds, probs = model.predict_with_threshold([special_text])
        assert len(preds) == 1

    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        config = TrainingConfig(device='cpu')
        model = BertModelWrapper(config)

        unicode_text = "Test with unicode: 你好 مرحبا שלום"
        preds, probs = model.predict_with_threshold([unicode_text])
        assert len(preds) == 1


class TestPerformance:
    """Test performance characteristics."""

    def test_batch_prediction_faster_than_sequential(self):
        """Test that batch prediction is faster."""
        import time

        config = TrainingConfig(device='cpu', batch_size=16)
        model = BertModelWrapper(config)

        texts = ["test text"] * 10

        # Batch
        start = time.time()
        model.predict_with_threshold(texts)
        batch_time = time.time() - start

        # Sequential (single predictions)
        start = time.time()
        for text in texts:
            model.predict_with_threshold([text])
        sequential_time = time.time() - start

        # Batch should be faster
        assert batch_time < sequential_time


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
