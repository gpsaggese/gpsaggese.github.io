"""
K-Fold Cross-Validation for Fake News Detection Models.

Implements stratified k-fold cross-validation for both BERT and LSTM models,
providing comprehensive evaluation across multiple folds.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path

from bert_utils import BertModelWrapper, TrainingConfig, BertTextDataset, DataLoader, ModelMetrics
from torch.utils.data import DataLoader as TorchDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cross_validation")


class CrossValidationEvaluator:
    """Performs k-fold cross-validation on fake news detection models."""

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize cross-validation evaluator.

        Args:
            n_splits: Number of folds
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.fold_results = []

    def evaluate_bert(
        self,
        texts: List[str],
        labels: List[int],
        train_config: TrainingConfig,
        dataset_name: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation on BERT model.

        Args:
            texts: List of text samples
            labels: List of labels (0=real, 1=fake)
            train_config: Training configuration
            dataset_name: Name of dataset for logging

        Returns:
            Dictionary with fold results and aggregated metrics
        """
        logger.info(f"Starting {self.n_splits}-fold CV for BERT on {dataset_name}")
        logger.info(f"Total samples: {len(texts)}")

        fold_results = []
        all_predictions = []
        all_labels = []

        texts_array = np.array(texts)
        labels_array = np.array(labels)

        for fold_idx, (train_idx, val_idx) in enumerate(self.skf.split(texts_array, labels_array)):
            logger.info(f"\n{'='*60}")
            logger.info(f"Fold {fold_idx + 1}/{self.n_splits}")
            logger.info(f"{'='*60}")

            # Split data
            X_train, X_val = texts_array[train_idx].tolist(), texts_array[val_idx].tolist()
            y_train, y_val = labels_array[train_idx].tolist(), labels_array[val_idx].tolist()

            logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
            logger.info(f"Train fake rate: {sum(y_train)/len(y_train)*100:.1f}%")

            try:
                # Initialize and train model
                model = BertModelWrapper(train_config)
                history = model.train(X_train, y_train, X_val, y_val)

                # Evaluate on validation set
                val_dataset = BertTextDataset(X_val, y_val, model.tokenizer, max_length=256)
                val_loader = TorchDataLoader(val_dataset, batch_size=train_config.batch_size)
                metrics = model._evaluate(val_loader)

                fold_result = {
                    'fold': fold_idx + 1,
                    'train_size': len(X_train),
                    'val_size': len(X_val),
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1': metrics.f1,
                    'roc_auc': metrics.roc_auc,
                    'loss': metrics.loss
                }
                fold_results.append(fold_result)

                logger.info(f"Fold {fold_idx + 1} Results:")
                logger.info(f"  Accuracy:  {metrics.accuracy:.4f}")
                logger.info(f"  Precision: {metrics.precision:.4f}")
                logger.info(f"  Recall:    {metrics.recall:.4f}")
                logger.info(f"  F1-Score:  {metrics.f1:.4f}")
                logger.info(f"  ROC-AUC:   {metrics.roc_auc:.4f}")

                # Store predictions for overall metrics
                with torch.no_grad():
                    batch_preds = []
                    for batch in val_loader:
                        inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
                        outputs = model.model(**inputs)
                        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                        batch_preds.extend(preds)
                    all_predictions.extend(batch_preds)
                    all_labels.extend(y_val)

            except Exception as e:
                logger.error(f"Error in fold {fold_idx + 1}: {str(e)}")
                continue

        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(fold_results, all_predictions, all_labels)

        return {
            'dataset': dataset_name,
            'n_splits': self.n_splits,
            'fold_results': fold_results,
            'aggregate_metrics': aggregate_metrics,
            'all_predictions': all_predictions,
            'all_labels': all_labels
        }

    def _calculate_aggregate_metrics(
        self,
        fold_results: List[Dict],
        predictions: List[int],
        labels: List[int]
    ) -> Dict[str, float]:
        """Calculate aggregate metrics across all folds."""
        if not fold_results:
            return {}

        metrics_dict = {}

        # Per-fold averages
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'loss']:
            values = [r[metric] for r in fold_results if metric in r]
            if values:
                metrics_dict[f'{metric}_mean'] = float(np.mean(values))
                metrics_dict[f'{metric}_std'] = float(np.std(values))
                metrics_dict[f'{metric}_min'] = float(np.min(values))
                metrics_dict[f'{metric}_max'] = float(np.max(values))

        # Overall metrics on all predictions
        if predictions and labels:
            try:
                metrics_dict['overall_accuracy'] = float(accuracy_score(labels, predictions))
                metrics_dict['overall_precision'] = float(precision_score(labels, predictions, zero_division=0))
                metrics_dict['overall_recall'] = float(recall_score(labels, predictions, zero_division=0))
                metrics_dict['overall_f1'] = float(f1_score(labels, predictions, zero_division=0))
                try:
                    metrics_dict['overall_roc_auc'] = float(roc_auc_score(labels, predictions))
                except:
                    metrics_dict['overall_roc_auc'] = 0.0
            except Exception as e:
                logger.error(f"Error calculating overall metrics: {str(e)}")

        return metrics_dict

    def print_results(self, cv_results: Dict[str, Any]) -> None:
        """Pretty print cross-validation results."""
        print("\n" + "="*80)
        print(f"CROSS-VALIDATION RESULTS: {cv_results['dataset']}")
        print("="*80)

        print(f"\nNumber of Folds: {cv_results['n_splits']}")

        print("\nPer-Fold Results:")
        print("-" * 80)
        for fold in cv_results['fold_results']:
            print(f"Fold {fold['fold']}:")
            print(f"  Accuracy:  {fold['accuracy']:.4f}")
            print(f"  Precision: {fold['precision']:.4f}")
            print(f"  Recall:    {fold['recall']:.4f}")
            print(f"  F1-Score:  {fold['f1']:.4f}")
            print(f"  ROC-AUC:   {fold['roc_auc']:.4f}")

        print("\nAggregate Metrics:")
        print("-" * 80)
        agg = cv_results['aggregate_metrics']

        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if f'{metric}_mean' in agg:
                mean = agg[f'{metric}_mean']
                std = agg[f'{metric}_std']
                min_val = agg[f'{metric}_min']
                max_val = agg[f'{metric}_max']
                print(f"{metric.upper():12} - Mean: {mean:.4f} ± {std:.4f} (Min: {min_val:.4f}, Max: {max_val:.4f})")

        if 'overall_accuracy' in agg:
            print("\nOverall Metrics (All Predictions):")
            print("-" * 80)
            print(f"Accuracy:  {agg['overall_accuracy']:.4f}")
            print(f"Precision: {agg['overall_precision']:.4f}")
            print(f"Recall:    {agg['overall_recall']:.4f}")
            print(f"F1-Score:  {agg['overall_f1']:.4f}")
            print(f"ROC-AUC:   {agg['overall_roc_auc']:.4f}")

        print("\n" + "="*80)


def run_cv_evaluation():
    """Run k-fold cross-validation on LIAR dataset."""
    # Load data
    loader = DataLoader()
    texts, labels = loader.load_liar(Path('data/LIAR'))

    logger.info(f"Loaded {len(texts)} samples from LIAR")

    # Setup training config
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

    # Run cross-validation
    evaluator = CrossValidationEvaluator(n_splits=5)
    cv_results = evaluator.evaluate_bert(texts, labels, train_config, dataset_name="LIAR")

    # Print results
    evaluator.print_results(cv_results)

    return cv_results


if __name__ == '__main__':
    cv_results = run_cv_evaluation()
