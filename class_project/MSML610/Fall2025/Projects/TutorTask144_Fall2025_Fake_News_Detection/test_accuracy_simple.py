"""
Simplified Accuracy Testing Script.

Tests core models without dependencies issues.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("accuracy_test")

import random
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def create_synthetic_data(n_samples: int = 1000) -> Tuple[List[str], List[int]]:
    """Create synthetic fake news data."""
    fake_claims = [
        "The election was rigged and stolen",
        "Vaccines contain microchips to control people",
        "The government is hiding the truth",
        "This product cures all diseases",
        "Scientists discovered a hidden cure being suppressed",
        "The moon landing was faked",
        "5G towers cause coronavirus",
        "The earth is flat",
        "Jet fuel can't melt steel beams",
        "The deep state controls everything",
    ]

    real_claims = [
        "Scientists confirmed the vaccine is safe and effective",
        "The election results were verified by multiple audits",
        "COVID-19 is a real disease caused by a virus",
        "The government released official statements",
        "New study shows climate change is accelerating",
        "Researchers published findings in peer-reviewed journal",
        "The election was conducted according to law",
        "Officials confirmed the safety standards",
        "Studies show vaccine effectiveness exceeds 90%",
        "Experts agree on the scientific consensus",
    ]

    texts = []
    labels = []

    for _ in range(n_samples // 2):
        texts.append(random.choice(fake_claims))
        labels.append(1)

        texts.append(random.choice(real_claims))
        labels.append(0)

    return texts, labels


def test_bert_accuracy():
    """Test BERT model accuracy."""
    logger.info("\n" + "="*80)
    logger.info("TESTING BERT MODEL ACCURACY")
    logger.info("="*80)

    try:
        from bert_utils import BertModelWrapper, TrainingConfig, BertTextDataset
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader

        # Create synthetic data
        logger.info("Creating synthetic test data...")
        texts, labels = create_synthetic_data(1000)
        logger.info(f"Total samples: {len(texts)}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=labels
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Train BERT
        config = TrainingConfig(
            model_name='distilbert-base-uncased',
            batch_size=16,
            learning_rate=2e-5,
            num_epochs=2,
            device='cpu',
            use_class_weights=True,
            max_text_length=256
        )

        logger.info("Initializing DistilBERT model...")
        model = BertModelWrapper(config)

        logger.info("Training model...")
        history = model.train(X_train, y_train, X_val, y_val)

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_dataset = BertTextDataset(X_test, y_test, model.tokenizer, config.max_text_length)
        test_loader = DataLoader(test_dataset, batch_size=16)

        metrics = model._evaluate(test_loader)

        logger.info("\n" + "="*80)
        logger.info("BERT TEST RESULTS")
        logger.info("="*80)
        logger.info(f"Accuracy:  {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)")
        logger.info(f"Precision: {metrics.precision:.4f}")
        logger.info(f"Recall:    {metrics.recall:.4f}")
        logger.info(f"F1-Score:  {metrics.f1:.4f}")
        logger.info(f"ROC-AUC:   {metrics.roc_auc:.4f}")
        logger.info("="*80)

        return {
            'model': 'DistilBERT',
            'accuracy': float(metrics.accuracy),
            'precision': float(metrics.precision),
            'recall': float(metrics.recall),
            'f1': float(metrics.f1),
            'roc_auc': float(metrics.roc_auc),
            'per_class_metrics': metrics.per_class_metrics,
            'confusion_matrix': metrics.confusion_matrix
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_lstm_accuracy():
    """Test LSTM model accuracy."""
    logger.info("\n" + "="*80)
    logger.info("TESTING LSTM MODEL ACCURACY")
    logger.info("="*80)

    try:
        from lstm_utils import LSTMModelWrapper, LSTMConfig
        from sklearn.model_selection import train_test_split

        # Create synthetic data
        logger.info("Creating synthetic test data...")
        texts, labels = create_synthetic_data(1000)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=labels
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Train LSTM
        config = LSTMConfig(
            embedding_dim=100,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3,
            batch_size=16,
            learning_rate=1e-3,
            num_epochs=2,
            device='cpu'
        )

        logger.info("Initializing LSTM model...")
        model = LSTMModelWrapper(config)

        logger.info("Training model...")
        history = model.train(X_train, y_train, X_val, y_val)

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        metrics = model.evaluate(X_test, y_test)

        logger.info("\n" + "="*80)
        logger.info("LSTM TEST RESULTS")
        logger.info("="*80)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info("="*80)

        metrics['model'] = 'LSTM'
        return metrics

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run accuracy tests."""
    logger.info("\n" + "█"*80)
    logger.info("COMPREHENSIVE MODEL ACCURACY TESTING")
    logger.info("█"*80 + "\n")

    results = {}

    # Test BERT
    bert_result = test_bert_accuracy()
    if bert_result:
        results['distilbert'] = bert_result

    # Test LSTM
    lstm_result = test_lstm_accuracy()
    if lstm_result:
        results['lstm'] = lstm_result

    # Save results
    with open('accuracy_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "█"*80)
    logger.info("FINAL ACCURACY SUMMARY")
    logger.info("█"*80)

    for model_name, metrics in results.items():
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    logger.info("\n" + "█"*80)
    logger.info("Results saved to: accuracy_test_results.json")
    logger.info("█"*80 + "\n")


if __name__ == '__main__':
    main()
