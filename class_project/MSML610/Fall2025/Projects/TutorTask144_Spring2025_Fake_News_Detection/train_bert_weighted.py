#!/usr/bin/env python3
"""
Train BERT with class-weighted loss for improved fake news detection.

This script demonstrates class-weighted loss which addresses the class imbalance
issue in the LIAR dataset (40% fake, 60% real). The weights make the model pay
more attention to minority class (fake news) during training.
"""

import torch
from pathlib import Path
from bert_utils import DataConfig, TrainingConfig, BertModelWrapper, DataLoader

def main():
    # Setup paths
    data_dir = Path('data/LIAR')
    model_dir = Path('models/distilbert_weighted')

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

    # Configure training WITH CLASS WEIGHTS
    train_config = TrainingConfig(
        model_name='distilbert-base-uncased',
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=3,  # Slightly longer training
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        patience=2,  # More patience
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_class_weights=True,  # KEY: Enable class weighting
        max_text_length=256
    )

    print(f"\nTraining Configuration:")
    print(f"  Model: {train_config.model_name}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Epochs: {train_config.num_epochs}")
    print(f"  Class weights: {train_config.use_class_weights}")
    print(f"  Device: {train_config.device}")

    # Initialize model
    model = BertModelWrapper(train_config)

    # Train model
    print("\nTraining model with class-weighted loss...")
    history = model.train(X_train, y_train, X_val, y_val)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    from torch.utils.data import DataLoader as TorchDataLoader
    from bert_utils import BertTextDataset

    test_dataset = BertTextDataset(X_test, y_test, model.tokenizer, max_length=256)
    test_loader = TorchDataLoader(test_dataset, batch_size=16)
    metrics = model._evaluate(test_loader)

    print(f"\nTest Results (Class-Weighted Loss):")
    print(f"  Accuracy:  {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1-Score:  {metrics.f1:.4f}")
    print(f"  ROC-AUC:   {metrics.roc_auc:.4f}")

    print(f"\nPer-Class Performance:")
    print(f"  Fake - Precision: {metrics.per_class_metrics['fake']['precision']:.4f}, "
          f"Recall: {metrics.per_class_metrics['fake']['recall']:.4f}, "
          f"F1: {metrics.per_class_metrics['fake']['f1-score']:.4f}")
    print(f"  Real - Precision: {metrics.per_class_metrics['real']['precision']:.4f}, "
          f"Recall: {metrics.per_class_metrics['real']['recall']:.4f}, "
          f"F1: {metrics.per_class_metrics['real']['f1-score']:.4f}")

    # Test threshold optimization
    print("\n" + "="*60)
    print("Testing Threshold Optimization")
    print("="*60)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    predictions, probabilities = model.predict_with_threshold(X_test, threshold=0.5)

    print(f"\nAdjusting decision threshold:")
    for threshold in thresholds:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        preds_threshold = (probabilities > threshold).astype(int)
        acc = accuracy_score(y_test, preds_threshold)
        prec = precision_score(y_test, preds_threshold, average='weighted')
        rec = recall_score(y_test, preds_threshold, average='weighted')
        f1 = f1_score(y_test, preds_threshold, average='weighted')
        print(f"  Threshold {threshold}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    # Save model
    print(f"\nSaving model to {model_dir}...")
    model.save_model(str(model_dir))

    print("\nTraining completed successfully!")

if __name__ == '__main__':
    main()
