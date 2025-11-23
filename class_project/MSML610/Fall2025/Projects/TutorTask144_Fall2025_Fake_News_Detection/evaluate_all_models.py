"""
Comprehensive Evaluation of All Models.

Evaluates and compares accuracies of:
- BERT model
- LSTM model
- Optimized BERT
- Ensemble models
"""

import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import json

from bert_utils import BertModelWrapper, TrainingConfig, DataLoader, BertTextDataset
from lstm_utils import LSTMModelWrapper, LSTMConfig
from cross_validation import CrossValidationEvaluator
from train_optimized import OptimizedFakeNewsTrainer
from category_adaptation import CategoryDetector, analyze_fake_news_by_category

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate_all_models")


class ModelEvaluationSuite:
    """Comprehensive evaluation of all models."""

    def __init__(self, data_path: Path = Path('data/LIAR')):
        """Initialize evaluation suite."""
        self.data_path = data_path
        self.loader = DataLoader()
        self.results = {
            'dataset_info': {},
            'model_results': {},
            'comparison': {}
        }

    def load_data(self, dataset_name: str = "LIAR") -> tuple:
        """Load dataset."""
        logger.info(f"Loading {dataset_name} dataset...")

        if dataset_name == "LIAR":
            texts, labels = self.loader.load_liar(self.data_path)
        elif dataset_name == "ISOT":
            texts, labels = self.loader.load_isot(self.data_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Store dataset info
        self.results['dataset_info'] = {
            'name': dataset_name,
            'total_samples': len(texts),
            'fake_count': sum(labels),
            'real_count': len(labels) - sum(labels),
            'fake_percentage': sum(labels) / len(labels) * 100
        }

        logger.info(f"Loaded {len(texts)} samples")
        logger.info(f"Fake: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
        logger.info(f"Real: {len(labels) - sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")

        return texts, labels

    def evaluate_standard_bert(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """Evaluate standard BERT model."""
        logger.info("\n" + "="*80)
        logger.info("EVALUATING STANDARD BERT MODEL")
        logger.info("="*80)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Train config
        train_config = TrainingConfig(
            model_name='distilbert-base-uncased',
            batch_size=16,
            learning_rate=2e-5,
            num_epochs=3,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            patience=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        logger.info(f"Training BERT for {train_config.num_epochs} epochs...")

        # Train
        model = BertModelWrapper(train_config)
        history = model.train(X_train, y_train, X_test, y_test)

        # Evaluate
        from torch.utils.data import DataLoader as TorchDataLoader
        test_dataset = BertTextDataset(X_test, y_test, model.tokenizer, max_length=256)
        test_loader = TorchDataLoader(test_dataset, batch_size=train_config.batch_size)
        metrics = model._evaluate(test_loader)

        result = {
            'model_type': 'Standard BERT',
            'configuration': {
                'model': train_config.model_name,
                'epochs': train_config.num_epochs,
                'batch_size': train_config.batch_size,
                'learning_rate': train_config.learning_rate
            },
            'metrics': {
                'accuracy': float(metrics.accuracy),
                'precision': float(metrics.precision),
                'recall': float(metrics.recall),
                'f1': float(metrics.f1),
                'roc_auc': float(metrics.roc_auc)
            },
            'model': model
        }

        logger.info(f"BERT Accuracy: {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)")
        logger.info(f"BERT Precision: {metrics.precision:.4f}")
        logger.info(f"BERT F1-Score: {metrics.f1:.4f}")

        return result

    def evaluate_lstm(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """Evaluate LSTM model."""
        logger.info("\n" + "="*80)
        logger.info("EVALUATING LSTM MODEL")
        logger.info("="*80)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # LSTM config
        lstm_config = LSTMConfig(
            embedding_dim=100,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            batch_size=32,
            learning_rate=1e-3,
            num_epochs=5,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            max_seq_length=256,
            vocab_size=10000
        )

        logger.info(f"Training LSTM for {lstm_config.num_epochs} epochs...")

        # Train
        lstm_model = LSTMModelWrapper(lstm_config)
        history = lstm_model.train(X_train, y_train, X_test, y_test)

        # Evaluate
        metrics = lstm_model.evaluate(X_test, y_test)

        result = {
            'model_type': 'LSTM',
            'configuration': {
                'embedding_dim': lstm_config.embedding_dim,
                'hidden_dim': lstm_config.hidden_dim,
                'num_layers': lstm_config.num_layers,
                'bidirectional': lstm_config.bidirectional,
                'epochs': lstm_config.num_epochs,
                'batch_size': lstm_config.batch_size,
                'learning_rate': lstm_config.learning_rate
            },
            'metrics': {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1']),
                'roc_auc': float(metrics['roc_auc'])
            },
            'model': lstm_model
        }

        logger.info(f"LSTM Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"LSTM Precision: {metrics['precision']:.4f}")
        logger.info(f"LSTM F1-Score: {metrics['f1']:.4f}")

        return result

    def evaluate_optimized_bert(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """Evaluate optimized BERT for >80% accuracy."""
        logger.info("\n" + "="*80)
        logger.info("EVALUATING OPTIMIZED BERT (TARGET: >80% ACCURACY)")
        logger.info("="*80)

        trainer = OptimizedFakeNewsTrainer()
        opt_results = trainer.train_with_optimization(
            self.data_path,
            dataset_name="LIAR",
            target_accuracy=0.80
        )

        if opt_results['best_model']:
            best = opt_results['best_model']
            result = {
                'model_type': 'Optimized BERT',
                'configuration': best['config'],
                'metrics': {
                    'accuracy': float(best['accuracy']),
                    'precision': float(best['precision']),
                    'recall': float(best['recall']),
                    'f1': float(best['f1']),
                    'roc_auc': float(best['roc_auc'])
                },
                'model': best['model']
            }

            logger.info(f"Optimized BERT Accuracy: {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
            logger.info(f"Optimized BERT Precision: {best['precision']:.4f}")
            logger.info(f"Optimized BERT F1-Score: {best['f1']:.4f}")

            return result
        else:
            logger.warning("Failed to train optimized BERT")
            return None

    def evaluate_kfold(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """Evaluate BERT with k-fold cross-validation."""
        logger.info("\n" + "="*80)
        logger.info("EVALUATING BERT WITH 5-FOLD CROSS-VALIDATION")
        logger.info("="*80)

        train_config = TrainingConfig(
            model_name='distilbert-base-uncased',
            batch_size=16,
            learning_rate=2e-5,
            num_epochs=2,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            patience=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        evaluator = CrossValidationEvaluator(n_splits=5)
        cv_results = evaluator.evaluate_bert(texts, labels, train_config)

        agg = cv_results['aggregate_metrics']

        result = {
            'model_type': 'BERT (5-Fold CV)',
            'configuration': {
                'model': train_config.model_name,
                'n_splits': 5,
                'epochs_per_fold': train_config.num_epochs,
                'batch_size': train_config.batch_size,
                'learning_rate': train_config.learning_rate
            },
            'metrics': {
                'accuracy_mean': float(agg.get('accuracy_mean', 0)),
                'accuracy_std': float(agg.get('accuracy_std', 0)),
                'precision_mean': float(agg.get('precision_mean', 0)),
                'precision_std': float(agg.get('precision_std', 0)),
                'f1_mean': float(agg.get('f1_mean', 0)),
                'f1_std': float(agg.get('f1_std', 0)),
                'roc_auc_mean': float(agg.get('roc_auc_mean', 0)),
                'roc_auc_std': float(agg.get('roc_auc_std', 0)),
                'overall_accuracy': float(agg.get('overall_accuracy', 0))
            },
            'per_fold_results': cv_results['fold_results']
        }

        logger.info(f"BERT (5-Fold) Mean Accuracy: {agg.get('accuracy_mean', 0):.4f} ± {agg.get('accuracy_std', 0):.4f}")
        logger.info(f"BERT (5-Fold) Overall Accuracy: {agg.get('overall_accuracy', 0):.4f}")
        logger.info(f"BERT (5-Fold) Mean F1: {agg.get('f1_mean', 0):.4f} ± {agg.get('f1_std', 0):.4f}")

        return result

    def run_all_evaluations(self) -> Dict[str, Any]:
        """Run all model evaluations."""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE MODEL EVALUATION SUITE")
        logger.info("="*80)

        # Load data
        texts, labels = self.load_data("LIAR")

        # Evaluate each model
        evaluations = {}

        try:
            evaluations['standard_bert'] = self.evaluate_standard_bert(texts, labels)
        except Exception as e:
            logger.error(f"Standard BERT evaluation failed: {str(e)}")

        try:
            evaluations['lstm'] = self.evaluate_lstm(texts, labels)
        except Exception as e:
            logger.error(f"LSTM evaluation failed: {str(e)}")

        try:
            evaluations['optimized_bert'] = self.evaluate_optimized_bert(texts, labels)
        except Exception as e:
            logger.error(f"Optimized BERT evaluation failed: {str(e)}")

        try:
            evaluations['kfold'] = self.evaluate_kfold(texts, labels)
        except Exception as e:
            logger.error(f"K-Fold evaluation failed: {str(e)}")

        self.results['model_results'] = evaluations

        return evaluations

    def print_comparison(self):
        """Print model comparison."""
        print("\n" + "="*100)
        print("MODEL ACCURACY COMPARISON")
        print("="*100)

        print(f"\nDataset: {self.results['dataset_info']['name']}")
        print(f"Total Samples: {self.results['dataset_info']['total_samples']}")
        print(f"Fake: {self.results['dataset_info']['fake_count']} ({self.results['dataset_info']['fake_percentage']:.1f}%)")
        print(f"Real: {self.results['dataset_info']['real_count']} ({100-self.results['dataset_info']['fake_percentage']:.1f}%)")

        print("\n" + "-"*100)
        print(f"{'Model':<30} {'Accuracy':<15} {'Precision':<15} {'F1-Score':<15} {'ROC-AUC':<15}")
        print("-"*100)

        for model_name, result in self.results['model_results'].items():
            if result is None:
                continue

            metrics = result['metrics']
            model_type = result['model_type']

            if 'accuracy_mean' in metrics:  # K-Fold
                acc_str = f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}"
                prec_str = f"{metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}"
                f1_str = f"{metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}"
                roc_str = f"{metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}"
            else:
                acc_str = f"{metrics['accuracy']:.4f}"
                prec_str = f"{metrics['precision']:.4f}"
                f1_str = f"{metrics['f1']:.4f}"
                roc_str = f"{metrics['roc_auc']:.4f}"

            print(f"{model_type:<30} {acc_str:<15} {prec_str:<15} {f1_str:<15} {roc_str:<15}")

        print("="*100)

        # Category analysis
        logger.info("\nAnalyzing fake news by category...")
        texts, labels = self.load_data("LIAR")
        category_analysis = analyze_fake_news_by_category(texts, labels, "LIAR")

        print("\nFAKE NEWS DISTRIBUTION BY CATEGORY")
        print("-"*100)
        print(f"{'Category':<20} {'Total':<15} {'Fake':<15} {'Real':<15} {'Fake %':<15}")
        print("-"*100)

        for category, stats in category_analysis['categories'].items():
            print(f"{category:<20} {stats['total']:<15} {stats['fake']:<15} {stats['real']:<15} {stats['fake_percentage']:.1f}%")

        print("="*100)

    def save_results(self, output_path: Path = Path('evaluation_results.json')):
        """Save results to JSON."""
        output_path = Path(output_path)

        # Convert models to strings for JSON serialization
        serializable_results = self.results.copy()
        for model_name in serializable_results['model_results']:
            if 'model' in serializable_results['model_results'][model_name]:
                del serializable_results['model_results'][model_name]['model']

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def main():
    """Run complete evaluation suite."""
    suite = ModelEvaluationSuite(Path('data/LIAR'))
    suite.run_all_evaluations()
    suite.print_comparison()
    suite.save_results()


if __name__ == '__main__':
    main()
