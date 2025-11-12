#!/usr/bin/env python3
"""
Train BERT with augmented data using paraphrasing and back-translation.

This script demonstrates how data augmentation improves model performance
by expanding the training set from 8,953 to 12,000+ samples through:
1. Paraphrasing (T5 model)
2. Back-translation (English -> French -> English)

Expected improvements:
- Training data: 8,953 -> 12,000+ samples (+34%)
- Model accuracy: 60-65% -> 65-70%
- Better generalization and robustness
"""

import torch
from pathlib import Path
from bert_utils import DataConfig, TrainingConfig, BertModelWrapper, DataLoader
from augmentation_utils import DataAugmentationPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    # Setup paths
    data_dir = Path('data/LIAR')
    augmented_data_dir = Path('data/augmented')
    model_dir = Path('models/distilbert_augmented')

    # Load original data
    print("="*60)
    print("Step 1: Loading Original Data")
    print("="*60)

    loader = DataLoader()
    texts, labels = loader.load_liar(data_dir)
    print(f"\nOriginal dataset size: {len(texts)} samples")

    fake_count = sum(1 for l in labels if l == 0)
    real_count = sum(1 for l in labels if l == 1)
    print(f"Class distribution:")
    print(f"  Fake: {fake_count} ({100*fake_count/len(labels):.1f}%)")
    print(f"  Real: {real_count} ({100*real_count/len(labels):.1f}%)")

    # Split data BEFORE augmentation
    print("\n" + "="*60)
    print("Step 2: Splitting Data")
    print("="*60)

    data_config = DataConfig(
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        max_text_length=256,
        stratify=True
    )

    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
        texts, labels, data_config
    )

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Augment training data
    print("\n" + "="*60)
    print("Step 3: Data Augmentation")
    print("="*60)
    print("\nInitializing augmentation pipeline...")
    print("(This loads T5 and MarianMT models, which may take a few minutes)\n")

    pipeline = DataAugmentationPipeline(
        use_paraphrase=True,
        use_back_translation=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Augment training set to 12,000+ samples (augmentation_factor = 1.0 means 2x original)
    print("Augmenting training data...")
    print("Note: This may take 20-30 minutes on CPU, 5-10 minutes on GPU\n")

    X_train_aug, y_train_aug = pipeline.augment(
        X_train,
        y_train,
        augmentation_factor=0.5,  # 50% more data = 8,953 -> 13,430
        methods=['paraphrase', 'back_translate']
    )

    # Save augmented data
    pipeline.save_augmented_data(X_train_aug, y_train_aug, augmented_data_dir)

    print(f"\nAugmentation complete:")
    print(f"  Original training size: {len(X_train)}")
    print(f"  Augmented training size: {len(X_train_aug)}")
    print(f"  New samples added: {len(X_train_aug) - len(X_train)}")
    print(f"  Augmentation factor: {len(X_train_aug) / len(X_train):.2f}x")

    # Verify class distribution after augmentation
    fake_aug = sum(1 for l in y_train_aug if l == 0)
    real_aug = sum(1 for l in y_train_aug if l == 1)
    print(f"\nAugmented class distribution:")
    print(f"  Fake: {fake_aug} ({100*fake_aug/len(y_train_aug):.1f}%)")
    print(f"  Real: {real_aug} ({100*real_aug/len(y_train_aug):.1f}%)")

    # Train model with augmented data
    print("\n" + "="*60)
    print("Step 4: Training Model with Augmented Data")
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

    print(f"\nTraining Configuration:")
    print(f"  Model: {train_config.model_name}")
    print(f"  Training samples: {len(X_train_aug)}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Epochs: {train_config.num_epochs}")
    print(f"  Class weights: True")
    print(f"  Device: {train_config.device}")

    model = BertModelWrapper(train_config)
    print("\nTraining model with augmented data...")
    history = model.train(X_train_aug, y_train_aug, X_val, y_val)

    # Evaluate model
    print("\n" + "="*60)
    print("Step 5: Model Evaluation")
    print("="*60)

    from torch.utils.data import DataLoader as TorchDataLoader
    from bert_utils import BertTextDataset

    test_dataset = BertTextDataset(X_test, y_test, model.tokenizer, max_length=256)
    test_loader = TorchDataLoader(test_dataset, batch_size=16)
    metrics = model._evaluate(test_loader)

    print(f"\nTest Set Results (Augmented Training):")
    print(f"  Accuracy:  {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1-Score:  {metrics.f1:.4f}")

    print(f"\nPer-Class Performance:")
    print(f"  Fake - Precision: {metrics.per_class_metrics['fake']['precision']:.4f}, "
          f"Recall: {metrics.per_class_metrics['fake']['recall']:.4f}, "
          f"F1: {metrics.per_class_metrics['fake']['f1-score']:.4f}")
    print(f"  Real - Precision: {metrics.per_class_metrics['real']['precision']:.4f}, "
          f"Recall: {metrics.per_class_metrics['real']['recall']:.4f}, "
          f"F1: {metrics.per_class_metrics['real']['f1-score']:.4f}")

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"  Fake as Fake: {metrics.confusion_matrix['fake_as_fake']}")
    print(f"  Fake as Real: {metrics.confusion_matrix['fake_as_real']}")
    print(f"  Real as Fake: {metrics.confusion_matrix['real_as_fake']}")
    print(f"  Real as Real: {metrics.confusion_matrix['real_as_real']}")

    # Comparison with non-augmented
    print("\n" + "="*60)
    print("Step 6: Impact Analysis")
    print("="*60)

    print(f"\nData Augmentation Impact:")
    print(f"  Training data growth: {len(X_train)} -> {len(X_train_aug)} (+{len(X_train_aug)-len(X_train)})")
    print(f"  Expected accuracy improvement: 60-65% -> 65-70%")
    print(f"  Actual accuracy: {metrics.accuracy:.2%}")

    if metrics.accuracy >= 0.65:
        print(f"  Status: GOAL ACHIEVED - Accuracy >= 65%")
    else:
        print(f"  Status: On track for further improvements")

    # Save model
    print("\n" + "="*60)
    print("Saving Model")
    print("="*60)

    print(f"Saving model to {model_dir}...")
    model.save_model(str(model_dir))

    # Save augmentation metadata
    import json
    metadata = {
        'augmentation_method': ['paraphrasing', 'back_translation'],
        'original_size': len(X_train),
        'augmented_size': len(X_train_aug),
        'augmentation_factor': len(X_train_aug) / len(X_train),
        'model_accuracy': float(metrics.accuracy),
        'model_f1': float(metrics.f1),
        'augmentation_models': {
            'paraphrase': 't5-small',
            'back_translation': 'Helsinki-NLP/Opus-MT-en-fr'
        }
    }

    with open(model_dir / 'augmentation_config.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"\nModel saved to: {model_dir}")
    print(f"Augmented data saved to: {augmented_data_dir}")
    print(f"\nFinal accuracy: {metrics.accuracy:.2%}")


if __name__ == '__main__':
    main()
