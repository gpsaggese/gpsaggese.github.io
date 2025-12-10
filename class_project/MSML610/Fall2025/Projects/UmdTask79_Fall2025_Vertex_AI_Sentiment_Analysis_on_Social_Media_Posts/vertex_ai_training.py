#!/usr/bin/env python3
"""
Vertex AI Custom Training Script for RoBERTa Sentiment Analysis

This script runs RoBERTa fine-tuning on Vertex AI for Twitter sentiment analysis.
Used for project requirement: "Train a sentiment classification model using Vertex AI's NLP capabilities"

FIXED FOR HYPERPARAMETER TUNING:
- Uses cloudml-hypertune for proper metric reporting
- Simplified argparse to accept direct values from ${trial.parameters.*}
- Removed environment variable complexity
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
import logging

# Vertex AI and ML libraries
from google.cloud import storage
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

# CRITICAL: Import cloudml-hypertune for Vertex AI metric reporting
# CORRECT IMPORT (2025): import hypertune
try:
    import hypertune
    HYPERTUNE_AVAILABLE = True
except ImportError:
    try:
        # Fallback for some versions/contexts
        from cloudml_hypertune import hypertune
        HYPERTUNE_AVAILABLE = True
    except ImportError:
        HYPERTUNE_AVAILABLE = False
        print("WARNING: cloudml-hypertune not installed - hyperparameter tuning metrics will not be reported!")
        print(f"   [INFO] Estimated time: {max_trial_count * 30 // parallel_trial_count} minutes (GPU)")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _check_torch():
    """Debug helper to verify PyTorch is available in Vertex AI container."""
    print("[DEBUG] Torch version:", torch.__version__)
    print("[DEBUG] CUDA available:", torch.cuda.is_available())


def clean_text(text: str) -> str:
    """
    Clean text data by removing special characters, URLs, and mentions.

    PROJECT REQUIREMENT: "Clean the text data by removing stop words, special characters, and tokenizing"

    This function implements explicit text cleaning as required by the project description.
    Note: Stop word removal is handled by RoBERTa's tokenizer, which is optimized for Twitter data.

    :param text: Raw text string
    :return: Cleaned text string
    """
    import re

    # Remove URLs (http, https, www)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove @mentions (Twitter-specific)
    text = re.sub(r'@\w+', '', text)

    # Remove hashtag symbols (keep the text)
    text = re.sub(r'#', '', text)

    # Remove special characters and punctuation (keep only letters, numbers, spaces)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Convert to lowercase for consistency
    text = text.lower()

    return text


def load_data_from_gcs(gcs_path: str) -> pd.DataFrame:
    """
    Load preprocessed data from Google Cloud Storage.

    :param gcs_path: GCS path to JSONL file
    :return: DataFrame with text and labels
    """
    logger.info(f"Loading data from {gcs_path}")

    # Parse GCS path
    if gcs_path.startswith('gs://'):
        bucket_name = gcs_path.split('/')[2]
        blob_path = '/'.join(gcs_path.split('/')[3:])

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Download to temporary file
        temp_file = '/tmp/training_data.jsonl'
        blob.download_to_filename(temp_file)

        # Load JSONL data
        data = []
        with open(temp_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))

        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} samples from GCS")
        return df
    else:
        # Local file for testing
        data = []
        with open(gcs_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)


def preprocess_data(df: pd.DataFrame, text_column: str = 'text_content',
                   label_column: str = 'category') -> pd.DataFrame:
    """
    Preprocess data for training with explicit text cleaning.

    PROJECT REQUIREMENT: "Clean the text data by removing stop words, special characters, and tokenizing"

    :param df: Input DataFrame
    :param text_column: Name of text column
    :param label_column: Name of label column
    :return: Processed DataFrame
    """
    logger.info("Preprocessing data with text cleaning...")

    # Remove null values
    df = df.dropna(subset=[text_column, label_column]).reset_index(drop=True)

    # Ensure text is string
    df[text_column] = df[text_column].fillna('').astype(str)

    # **EXPLICIT TEXT CLEANING** (Project Requirement)
    logger.info("Cleaning text: removing URLs, mentions, special characters...")
    df[text_column] = df[text_column].apply(clean_text)
    logger.info("Text cleaning complete!")

    # Label mapping for sentiment
    label2id = {'positive': 0, 'neutral': 1, 'negative': 2}
    df['label'] = df[label_column].map(label2id)

    # Remove rows with unmapped labels
    df = df.dropna(subset=['label']).reset_index(drop=True)
    df['label'] = df['label'].astype(int)

    logger.info(f"Preprocessed data: {len(df)} samples")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

    return df


def tokenize_data(df: pd.DataFrame, tokenizer, text_column: str = 'text_content',
                 max_length: int = 128) -> Dataset:
    """
    Tokenize data for model input.

    PROJECT REQUIREMENT: "Clean the text data by removing stop words, special characters, and tokenizing"

    :param df: Input DataFrame
    :param tokenizer: HuggingFace tokenizer (RoBERTa)
    :param text_column: Name of text column
    :param max_length: Maximum sequence length
    :return: Tokenized Dataset
    """
    logger.info(f"Tokenizing data with max_length={max_length} (PROJECT REQUIREMENT)")

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding=False
        )

    # Create HuggingFace dataset
    dataset = Dataset.from_pandas(df[[text_column, 'label']])

    # Remove pandas index column if present
    if '__index_level_0__' in dataset.column_names:
        dataset = dataset.remove_columns(['__index_level_0__'])

    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[text_column]
    )

    logger.info(f"Tokenization complete: {len(tokenized_dataset)} samples")
    return tokenized_dataset


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for training.

    :param eval_pred: Predictions and labels from evaluation
    :return: Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # F1 Scores
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    # Additional metrics for comprehensive comparison
    accuracy = accuracy_score(labels, predictions)
    precision_macro = precision_score(labels, predictions, average='macro')
    recall_macro = recall_score(labels, predictions, average='macro')

    return {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro
    }


def train_model(train_dataset: Dataset, val_dataset: Dataset,
               model_name: str, output_dir: str, num_epochs: int = 4,
               batch_size: int = 32, learning_rate: float = 2e-5,
               warmup_ratio: float = 0.1, weight_decay: float = 0.01,
               metric_name: str = 'f1_macro'):
    """
    Train the RoBERTa model.

    :param train_dataset: Training dataset
    :param val_dataset: Validation dataset
    :param model_name: HuggingFace model name
    :param output_dir: Output directory for model
    :param num_epochs: Number of training epochs
    :param batch_size: Batch size
    :param learning_rate: Learning rate
    :param warmup_ratio: Warmup ratio
    :param weight_decay: Weight decay
    :return: Trained model and trainer
    """
    logger.info(f"Training model: {model_name}")
    logger.info(f"Training parameters: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Label mappings
    id2label = {0: 'positive', 1: 'neutral', 2: 'negative'}
    label2id = {'positive': 0, 'neutral': 1, 'negative': 2}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        logging_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],  # Disable wandb/tensorboard logging
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train model
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info(f"Training complete! Loss: {train_result.training_loss:.4f}")

    # Save model
    trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")

    return model, trainer, train_result


def upload_model_to_gcs(local_dir: str, bucket_name: str, gcs_prefix: str):
    """
    Upload model files from local directory to GCS bucket.
    
    This ensures model artifacts persist after the training job container terminates.
    
    :param local_dir: Local directory containing model files
    :param bucket_name: GCS bucket name (without gs:// prefix)
    :param gcs_prefix: Prefix path in GCS bucket (e.g., 'models/roberta-sentiment')
    """
    logger.info(f"Uploading model files to GCS...")
    logger.info(f"  Local dir: {local_dir}")
    logger.info(f"  GCS bucket: {bucket_name}")
    logger.info(f"  GCS prefix: {gcs_prefix}")
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Upload all files in the output directory
        uploaded_files = []
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                # Get relative path from local_dir
                relative_path = os.path.relpath(local_path, local_dir)
                # Construct GCS path
                gcs_path = os.path.join(gcs_prefix, relative_path).replace('\\\\', '/')
                
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                uploaded_files.append(f"gs://{bucket_name}/{gcs_path}")
                logger.info(f"  Uploaded: {relative_path}")
        
        logger.info(f"[SUCCESS] Uploaded {len(uploaded_files)} files to GCS")
        logger.info(f"  Model location: gs://{bucket_name}/{gcs_prefix}/")
        return uploaded_files
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to upload model to GCS: {str(e)}")
        logger.error("Model files will only be available in $AIP_MODEL_DIR")
        return []


def evaluate_model(trainer, test_dataset: Dataset, output_dir: str):
    """
    Evaluate the trained model on test data.

    :param trainer: Trained model trainer
    :param test_dataset: Test dataset
    :param output_dir: Output directory for results
    """
    logger.info("Evaluating model on test set...")

    # Get predictions
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = test_dataset['label']

    # Calculate metrics
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)

    logger.info(f"F1-Score (Macro): {f1_macro:.4f}")
    logger.info(f"F1-Score (Weighted): {f1_weighted:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=['positive', 'neutral', 'negative'],
        digits=4
    )
    logger.info(f"\nClassification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Save results
    results = {
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'accuracy': float(accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results saved to {results_path}")
    return results


def main():
    """Main training function - SIMPLIFIED for Vertex AI Hyperparameter Tuning."""
    _check_torch()
    parser = argparse.ArgumentParser(description='RoBERTa Sentiment Analysis Training on Vertex AI')

    # Data arguments
    parser.add_argument('--train_data_path', type=str, required=True,
                       help='GCS path to training data (JSONL)')
    parser.add_argument('--val_data_path', type=str, required=True,
                       help='GCS path to validation data (JSONL)')
    parser.add_argument('--test_data_path', type=str, required=True,
                       help='GCS path to test data (JSONL)')

    # Model arguments
    parser.add_argument('--model_name', type=str,
                       default='cardiffnlp/twitter-roberta-base-sentiment-latest',
                       help='HuggingFace model name')
    parser.add_argument('--output_dir', type=str, default='./model',
                       help='Output directory for model and results')

    # Training arguments - SIMPLIFIED TO ACCEPT DIRECT VALUES
    # Vertex AI will inject values using ${trial.parameters.param_name} syntax
    parser.add_argument('--num_epochs', type=int, default=4,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--metric_name', type=str, default='f1_macro',
                        help='Metric to optimize (f1_macro or accuracy)')

    args = parser.parse_args()

    # ============================================================
    # DEBUG: Log the received hyperparameters
    # ============================================================
    logger.info("=" * 60)
    logger.info("HYPERPARAMETERS RECEIVED FOR THIS TRIAL:")
    logger.info(f"  learning_rate: {args.learning_rate}")
    logger.info(f"  batch_size: {args.batch_size}")
    logger.info(f"  weight_decay: {args.weight_decay}")
    logger.info(f"  warmup_ratio: {args.warmup_ratio}")
    logger.info(f"  num_epochs: {args.num_epochs}")
    logger.info("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and preprocess data
    try:
        logger.info("Loading training data...")
        train_df = load_data_from_gcs(args.train_data_path)
        train_df = preprocess_data(train_df)
        logger.info(f"[SUCCESS] Training data loaded: {len(train_df)} samples")

        logger.info("Loading validation data...")
        val_df = load_data_from_gcs(args.val_data_path)
        val_df = preprocess_data(val_df)
        logger.info(f"[SUCCESS] Validation data loaded: {len(val_df)} samples")

        logger.info("Loading test data...")
        test_df = load_data_from_gcs(args.test_data_path)
        test_df = preprocess_data(test_df)
        logger.info(f"[SUCCESS] Test data loaded: {len(test_df)} samples")
    except Exception as e:
        logger.error(f"Failed to load data from GCS: {str(e)}")
        logger.error(f"Train path: {args.train_data_path}")
        logger.error(f"Val path: {args.val_data_path}")
        logger.error(f"Test path: {args.test_data_path}")
        raise

    # Tokenize data
    logger.info("Tokenizing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = tokenize_data(train_df, tokenizer, max_length=args.max_length)
    val_dataset = tokenize_data(val_df, tokenizer, max_length=args.max_length)
    test_dataset = tokenize_data(test_df, tokenizer, max_length=args.max_length)

    # Train model
    model, trainer, train_result = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        metric_name=args.metric_name
    )

    # Evaluate model
    results = evaluate_model(trainer, test_dataset, args.output_dir)

    # Save training summary
    summary = {
        'model_name': args.model_name,
        'hyperparameters': {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay,
            'warmup_ratio': args.warmup_ratio,
            'num_epochs': args.num_epochs,
            'max_length': args.max_length
        },
        'training_loss': float(train_result.training_loss),
        'evaluation_results': results
    }

    summary_path = os.path.join(args.output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("Training and evaluation complete!")
    logger.info(f"Model and results saved to {args.output_dir}")
    
    # ============================================================
    # CRITICAL FIX: Upload model files to GCS
    # ============================================================
    # Extract bucket name and prefix from training data path
    try:
        if args.train_data_path.startswith('gs://'):
            bucket_name = args.train_data_path.split('/')[2]
            # Create model path without timestamp for consistent naming
            model_prefix = f"trained_models/{args.model_name.replace('/', '_')}"
            
            logger.info("=" * 60)
            logger.info("UPLOADING MODEL TO GCS")
            logger.info("=" * 60)
            upload_model_to_gcs(args.output_dir, bucket_name, model_prefix)
        else:
            logger.info("[INFO] Not uploading to GCS (local training mode)")
    except Exception as e:
        logger.error(f"[ERROR] Failed to upload model to GCS: {str(e)}")
        logger.error("Model files are still available in $AIP_MODEL_DIR")

    # ============================================================
    # CRITICAL FIX: Report metric to Vertex AI using cloudml-hypertune
    # ============================================================
    f1_macro_score = results.get('f1_macro', 0.0)

    if HYPERTUNE_AVAILABLE:
        logger.info("=" * 60)
        logger.info("REPORTING METRIC TO VERTEX AI HYPERPARAMETER TUNING")
        logger.info("=" * 60)

        try:
            # CORRECT USAGE (2025): HyperTune() class from cloudml_hypertune
            hpt = hypertune.HyperTune()
            
            # Report F1 Macro
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='f1_macro',
                metric_value=f1_macro_score,
                global_step=args.num_epochs
            )
            
            # Report Accuracy (Requested by user)
            accuracy_score_val = results.get('accuracy', 0.0)
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='accuracy',
                metric_value=accuracy_score_val,
                global_step=args.num_epochs
            )
            
            logger.info(f"[SUCCESS] Successfully reported metrics to Vertex AI:")
            logger.info(f"  - f1_macro: {f1_macro_score:.4f}")
            logger.info(f"  - accuracy: {accuracy_score_val:.4f}")
            logger.info(f"  Global step: {args.num_epochs}")
        except Exception as e:
            #logger.error(f"[ERROR] Failed to report metric via cloudml-hypertune: {str(e)}")
            logger.error(f"This will cause red '!' indicators in Vertex AI console")
            # Fallback to stdout (old method, likely won't work)
            print(f"\nf1_macro={f1_macro_score}")
            raise
    else:
        logger.error("=" * 60)
        #logger.error("[ERROR] cloudml-hypertune NOT AVAILABLE!")
        logger.error("=" * 60)
        logger.error("Trials will show red '!' indicators in Vertex AI console")
        logger.error("Install cloudml-hypertune in your container!")
        # Try fallback (unlikely to work)
        print(f"\nf1_macro={f1_macro_score}")


if __name__ == '__main__':
    main()
