#!/usr/bin/env python3
"""
Extended BERT training with advanced learning rate scheduling.

This script trains BERT for 5+ epochs with:
1. Class-weighted loss to handle imbalance
2. Cosine annealing learning rate scheduling
3. Increased patience for early stopping
4. Gradient accumulation for effective batch size

Expected improvements over baseline:
- More stable convergence
- Better final performance
- Reduced overfitting risk
"""

import torch
from pathlib import Path
from bert_utils import DataConfig, TrainingConfig, BertModelWrapper, DataLoader

def main():
    # Setup paths
    data_dir = Path('data/LIAR')
    model_dir = Path('models/distilbert_extended')

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

    # Configure training with extended settings
    train_config = TrainingConfig(
        model_name='distilbert-base-uncased',
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=5,  # INCREASED: 2 → 5 epochs
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        patience=3,  # INCREASED: More patience for longer training
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_class_weights=True,
        max_text_length=256
    )

    print(f"\nTraining Configuration:")
    print(f"  Model: {train_config.model_name}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Epochs: {train_config.num_epochs}")
    print(f"  Warmup ratio: {train_config.warmup_ratio}")
    print(f"  Class weights: True")
    print(f"  Patience: {train_config.patience}")
    print(f"  Device: {train_config.device}")
    print(f"\nEstimated training time:")
    print(f"  CPU: ~5-7 hours")
    print(f"  GPU: ~30-45 minutes")

    # Initialize model
    model = BertModelWrapper(train_config)

    # Train model
    print("\nTraining model with extended epochs and class weights...")
    print("This may take a while. Progress will be displayed every 50 batches.\n")

    history = model.train(X_train, y_train, X_val, y_val)

    # Display training history
    print("\n" + "="*60)
    print("Training History")
    print("="*60)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val Acc':<12} {'Val F1':<12}")
    print("-"*60)
    for i, (train_loss, val_loss, val_acc, val_f1) in enumerate(zip(
        history['train_loss'],
        history['val_loss'],
        history['val_accuracy'],
        history['val_f1']
    )):
        print(f"{i+1:<8} {train_loss:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f} {val_f1:<12.4f}")

    # Evaluate on test set
    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60)

    from torch.utils.data import DataLoader as TorchDataLoader
    from bert_utils import BertTextDataset

    test_dataset = BertTextDataset(X_test, y_test, model.tokenizer, max_length=256)
    test_loader = TorchDataLoader(test_dataset, batch_size=16)
    metrics = model._evaluate(test_loader)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1-Score:  {metrics.f1:.4f}")
    print(f"  ROC-AUC:   {metrics.roc_auc:.4f}")

    print(f"\nPer-Class Performance:")
    print(f"  Fake News:")
    print(f"    Precision: {metrics.per_class_metrics['fake']['precision']:.4f}")
    print(f"    Recall:    {metrics.per_class_metrics['fake']['recall']:.4f}")
    print(f"    F1-Score:  {metrics.per_class_metrics['fake']['f1-score']:.4f}")
    print(f"    Support:   {metrics.per_class_metrics['fake']['support']}")

    print(f"  Real News:")
    print(f"    Precision: {metrics.per_class_metrics['real']['precision']:.4f}")
    print(f"    Recall:    {metrics.per_class_metrics['real']['recall']:.4f}")
    print(f"    F1-Score:  {metrics.per_class_metrics['real']['f1-score']:.4f}")
    print(f"    Support:   {metrics.per_class_metrics['real']['support']}")

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"  Fake predicted as Fake: {metrics.confusion_matrix['fake_as_fake']}")
    print(f"  Fake predicted as Real: {metrics.confusion_matrix['fake_as_real']}")
    print(f"  Real predicted as Fake: {metrics.confusion_matrix['real_as_fake']}")
    print(f"  Real predicted as Real: {metrics.confusion_matrix['real_as_real']}")

    # Test threshold optimization
    print("\n" + "="*60)
    print("Threshold Optimization Results")
    print("="*60)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    predictions, probabilities = model.predict_with_threshold(X_test, threshold=0.5)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    print(f"\n{'Threshold':<12} {'Accuracy':<12} {'Fake Prec':<12} {'Fake Rec':<12} {'Real Rec':<12} {'F1':<12}")
    print("-"*72)

    for threshold in thresholds:
        preds_threshold = (probabilities > threshold).astype(int)
        acc = accuracy_score(y_test, preds_threshold)

        # Per-class metrics
        fake_indices = [i for i, y in enumerate(y_test) if y == 0]
        real_indices = [i for i, y in enumerate(y_test) if y == 1]

        fake_prec = precision_score([y_test[i] for i in fake_indices],
                                   [preds_threshold[i] for i in fake_indices],
                                   zero_division=0) if fake_indices else 0
        fake_rec = recall_score([y_test[i] for i in fake_indices],
                               [preds_threshold[i] for i in fake_indices],
                               zero_division=0) if fake_indices else 0
        real_rec = recall_score([y_test[i] for i in real_indices],
                               [preds_threshold[i] for i in real_indices],
                               zero_division=0) if real_indices else 0

        f1 = f1_score(y_test, preds_threshold, average='weighted')

        print(f"{threshold:<12.1f} {acc:<12.4f} {fake_prec:<12.4f} {fake_rec:<12.4f} {real_rec:<12.4f} {f1:<12.4f}")

    # Save model
    print(f"\nSaving model to {model_dir}...")
    model.save_model(str(model_dir))

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print(f"\nModel saved to: {model_dir}")
    print(f"Best model: {model_dir}/pytorch_model.bin")

if __name__ == '__main__':
    main()
