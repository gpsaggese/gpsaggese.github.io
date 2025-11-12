#!/usr/bin/env python3
"""
Train ensemble model combining BERT + TF-IDF + LSTM for fake news detection.

This script demonstrates ensemble learning which combines three complementary
approaches:
1. BERT (transformer, context-aware)
2. TF-IDF (bag-of-words, statistical)
3. LSTM (sequence-aware, RNN-based)

Expected improvements:
- Accuracy: 60.92% → 70-75%
- More robust predictions through model diversity
- Better handling of edge cases
"""

import torch
from pathlib import Path
from bert_utils import DataConfig, TrainingConfig, BertModelWrapper, DataLoader
from ensemble_utils import EnsembleModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    # Setup paths
    data_dir = Path('data/LIAR')
    bert_model_dir = Path('models/distilbert_ensemble')
    ensemble_model_dir = Path('models/ensemble_combined')

    # Load data
    print("Loading LIAR dataset...")
    loader = DataLoader()
    texts, labels = loader.load_liar(data_dir)
    print(f"Total samples: {len(texts)}")

    # Calculate class distribution
    fake_count = sum(1 for l in labels if l == 0)
    real_count = sum(1 for l in labels if l == 1)
    print(f"Class distribution: {fake_count} fake ({100*fake_count/len(labels):.1f}%), "
          f"{real_count} real ({100*real_count/len(labels):.1f}%)")

    # Configure data loading
    data_config = DataConfig(
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        max_text_length=256,
        stratify=True
    )

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
        texts, labels, data_config
    )

    # Step 1: Train BERT component
    print("\n" + "="*60)
    print("Step 1: Training BERT Component")
    print("="*60)

    train_config = TrainingConfig(
        model_name='distilbert-base-uncased',
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=3,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        patience=2,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_class_weights=True,
        max_text_length=256
    )

    print(f"Training Configuration:")
    print(f"  Model: {train_config.model_name}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Epochs: {train_config.num_epochs}")
    print(f"  Class weights: True")
    print(f"  Device: {train_config.device}")

    bert_model = BertModelWrapper(train_config)
    print("\nTraining BERT model...")
    bert_history = bert_model.train(X_train, y_train, X_val, y_val)

    # Evaluate BERT alone
    print("\n" + "="*60)
    print("BERT Evaluation")
    print("="*60)
    from torch.utils.data import DataLoader as TorchDataLoader
    from bert_utils import BertTextDataset

    test_dataset = BertTextDataset(X_test, y_test, bert_model.tokenizer, max_length=256)
    test_loader = TorchDataLoader(test_dataset, batch_size=16)
    bert_metrics = bert_model._evaluate(test_loader)

    print(f"BERT Accuracy: {bert_metrics.accuracy:.4f}")
    print(f"BERT Precision: {bert_metrics.precision:.4f}")
    print(f"BERT Recall: {bert_metrics.recall:.4f}")
    print(f"BERT F1-Score: {bert_metrics.f1:.4f}")

    # Step 2: Create and train ensemble
    print("\n" + "="*60)
    print("Step 2: Training Ensemble Components")
    print("="*60)

    # Initialize ensemble with BERT
    ensemble = EnsembleModel(
        bert_model=bert_model,
        weights={'bert': 0.5, 'tfidf': 0.25, 'lstm': 0.25},
        voting='weighted'
    )

    print("\nTraining ensemble components (TF-IDF and LSTM)...")
    print("This may take 5-10 minutes. Progress will be displayed for LSTM training.\n")

    ensemble.train_components(
        X_train, y_train, X_val, y_val,
        train_lstm=True
    )

    # Step 3: Evaluate ensemble
    print("\n" + "="*60)
    print("Step 3: Ensemble Evaluation")
    print("="*60)

    ensemble_preds, ensemble_probs = ensemble.predict(X_test)

    ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
    ensemble_precision = precision_score(y_test, ensemble_preds, average='weighted')
    ensemble_recall = recall_score(y_test, ensemble_preds, average='weighted')
    ensemble_f1 = f1_score(y_test, ensemble_preds, average='weighted')

    print(f"\nEnsemble Results:")
    print(f"  Accuracy:  {ensemble_accuracy:.4f}")
    print(f"  Precision: {ensemble_precision:.4f}")
    print(f"  Recall:    {ensemble_recall:.4f}")
    print(f"  F1-Score:  {ensemble_f1:.4f}")

    # Step 4: Per-class analysis
    print("\n" + "="*60)
    print("Per-Class Performance Comparison")
    print("="*60)

    fake_indices = [i for i, y in enumerate(y_test) if y == 0]
    real_indices = [i for i, y in enumerate(y_test) if y == 1]

    # BERT
    bert_preds, _ = bert_model.predict_with_threshold(X_test, threshold=0.5)
    bert_fake_recall = recall_score(
        [y_test[i] for i in fake_indices],
        [bert_preds[i] for i in fake_indices],
        zero_division=0
    )
    bert_real_recall = recall_score(
        [y_test[i] for i in real_indices],
        [bert_preds[i] for i in real_indices],
        zero_division=0
    )

    # Ensemble
    ensemble_fake_recall = recall_score(
        [y_test[i] for i in fake_indices],
        [ensemble_preds[i] for i in fake_indices],
        zero_division=0
    )
    ensemble_real_recall = recall_score(
        [y_test[i] for i in real_indices],
        [ensemble_preds[i] for i in real_indices],
        zero_division=0
    )

    print(f"\nFake News Recall:")
    print(f"  BERT:     {bert_fake_recall:.4f}")
    print(f"  Ensemble: {ensemble_fake_recall:.4f}")
    print(f"  Improvement: {(ensemble_fake_recall - bert_fake_recall):.4f}")

    print(f"\nReal News Recall:")
    print(f"  BERT:     {bert_real_recall:.4f}")
    print(f"  Ensemble: {ensemble_real_recall:.4f}")
    print(f"  Difference: {(ensemble_real_recall - bert_real_recall):.4f}")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, ensemble_preds)
    print(f"\nConfusion Matrix:")
    print(f"  Fake predicted as Fake: {cm[0][0]}")
    print(f"  Fake predicted as Real: {cm[0][1]}")
    print(f"  Real predicted as Fake: {cm[1][0]}")
    print(f"  Real predicted as Real: {cm[1][1]}")

    # Step 5: Model comparison
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)

    print(f"\n{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 62)

    print(f"{'BERT':<15} {bert_metrics.accuracy:<12.4f} {bert_metrics.precision:<12.4f} "
          f"{bert_metrics.recall:<12.4f} {bert_metrics.f1:<12.4f}")
    print(f"{'Ensemble':<15} {ensemble_accuracy:<12.4f} {ensemble_precision:<12.4f} "
          f"{ensemble_recall:<12.4f} {ensemble_f1:<12.4f}")

    improvement = ensemble_accuracy - bert_metrics.accuracy
    print(f"\nAccuracy Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")

    # Save models
    print("\n" + "="*60)
    print("Saving Models")
    print("="*60)

    print(f"Saving BERT model to {bert_model_dir}...")
    bert_model.save_model(str(bert_model_dir))

    print(f"Saving ensemble configuration...")
    ensemble_model_dir.mkdir(parents=True, exist_ok=True)

    # Save ensemble metadata
    import json
    metadata = {
        'ensemble_type': 'weighted_voting',
        'weights': ensemble.weights,
        'components': ['bert', 'tfidf', 'lstm'],
        'bert_model_dir': str(bert_model_dir),
        'test_metrics': {
            'bert_accuracy': float(bert_metrics.accuracy),
            'ensemble_accuracy': float(ensemble_accuracy),
            'improvement': float(improvement)
        }
    }

    with open(ensemble_model_dir / 'ensemble_config.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Ensemble configuration saved to {ensemble_model_dir}")

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print(f"\nBERT model: {bert_model_dir}")
    print(f"Ensemble config: {ensemble_model_dir}/ensemble_config.json")
    print(f"\nAccuracy achieved: {ensemble_accuracy:.2%}")
    print(f"Target accuracy: 70-75%")
    print(f"Status: {'GOAL ACHIEVED' if ensemble_accuracy >= 0.70 else 'On track for further improvements'}")


if __name__ == '__main__':
    main()
