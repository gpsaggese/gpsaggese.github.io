"""
PEFT Sentiment Analysis Utilities

This module provides utility functions for fine-tuning transformer models using
Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters for binary text classification.

The pipeline includes:
- Data loading and preprocessing
- Text tokenization and cleaning
- HuggingFace dataset preparation
- LoRA-adapted RoBERTa model setup
- Training and evaluation
- SHAP-based explainability

Author: UmdTask96
Date: December 2025
"""

import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

from imblearn.over_sampling import SMOTE

from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer
)

from peft import LoraConfig, get_peft_model, TaskType

import shap
from transformers import pipeline


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_fake_true(fake_path="Data/Fake.csv", true_path="Data/True.csv"):
    """
    Load and combine fake and true news datasets.
    
    Args:
        fake_path (str): Path to the fake news CSV file. Default: "Data/Fake.csv"
        true_path (str): Path to the true news CSV file. Default: "Data/True.csv"
    
    Returns:
        pd.DataFrame: Combined dataframe with columns including 'text' and 'label'
                     where label=1 for fake news and label=0 for true news
    
    Note:
        The function concatenates both datasets and resets the index to ensure
        a clean sequential index for further processing.
    """
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 1
    true_df["label"] = 0

    df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
    return df


# ============================================================
# 2. TEXT CLEANING AND PREPROCESSING
# ============================================================

def remove_punct(text):
    """
    Remove all punctuation from text.
    
    Args:
        text (str): Input text containing punctuation
    
    Returns:
        str: Text with all punctuation marks removed
    
    Example:
        >>> remove_punct("Hello, world!")
        "Hello world"
    """
    return "".join([char for char in text if char not in string.punctuation])

def preprocess_text(df):
    """
    Preprocess text data through multiple cleaning steps.
    
    This function performs comprehensive text preprocessing including:
    1. Lowercase conversion and punctuation removal
    2. Tokenization into individual words
    3. Stopword removal to reduce noise
    4. Lemmatization to normalize word forms
    
    Args:
        df (pd.DataFrame): DataFrame with a 'text' column containing raw text
    
    Returns:
        pd.DataFrame: Original dataframe with additional columns:
                     - text_clean: lowercased, punctuation-removed text
                     - text_tokens: list of processed tokens
                     - text_final: final preprocessed text as string
    
    Note:
        This function downloads required NLTK data if not already present.
        Processing is done in-place on the input dataframe.
    """
    df["text_clean"] = df["text"].apply(lambda x: remove_punct(x.lower()))

    # Download required NLTK data quietly
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    stopwords_en = stopwords.words("english")
    df["text_tokens"] = df["text_clean"].apply(word_tokenize)

    # Remove common stopwords (the, and, is, etc.) to reduce noise
    df["text_tokens"] = df["text_tokens"].apply(
        lambda tokens: [w for w in tokens if w not in stopwords_en]
    )

    # Lemmatize: convert words to base form (running → run, better → good)
    wn = WordNetLemmatizer()
    df["text_tokens"] = df["text_tokens"].apply(
        lambda tokens: [wn.lemmatize(w) for w in tokens]
    )

    # Rejoin tokens into a single string for model input
    df["text_final"] = df["text_tokens"].apply(lambda tokens: " ".join(tokens))

    return df


# ============================================================
# 3. TRAIN/TEST SPLIT
# ============================================================

def split_data(df):
    """
    Split dataset into training and testing sets.
    
    Args:
        df (pd.DataFrame): DataFrame with 'text_final' and 'label' columns
    
    Returns:
        tuple: (train_texts, test_texts, train_labels, test_labels)
               80/20 split with stratification to maintain class balance
    
    Note:
        Uses stratified splitting to ensure equal representation of both
        classes (fake/true) in training and test sets. Random state is
        fixed at 42 for reproducibility.
    """
    X = df["text_final"].tolist()
    y = df["label"].tolist()

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


# ============================================================
# 4. TOKENIZATION + HUGGINGFACE DATASETS
# ============================================================

def tokenize_function(batch, tokenizer):
    """
    Tokenize text batch using RoBERTa tokenizer.
    
    Args:
        batch (dict): Batch dictionary containing 'text' key
        tokenizer (RobertaTokenizer): Pre-initialized tokenizer
    
    Returns:
        dict: Tokenized batch with input_ids, attention_mask, etc.
    
    Note:
        Max length is set to 256 tokens as a balance between capturing
        sufficient context and computational efficiency. Longer sequences
        are truncated, shorter ones are padded.
    """
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256  # Balance between context and efficiency
    )

def prepare_hf_dataset(train_texts, test_texts, train_labels, test_labels):
    """
    Prepare HuggingFace datasets from raw text and labels.
    
    Args:
        train_texts (list): List of training text strings
        test_texts (list): List of testing text strings
        train_labels (list): List of training labels (0 or 1)
        test_labels (list): List of testing labels (0 or 1)
    
    Returns:
        tuple: (train_dataset, test_dataset, tokenizer)
               - train_dataset: Tokenized HF Dataset for training
               - test_dataset: Tokenized HF Dataset for testing
               - tokenizer: RobertaTokenizer instance for later use
    
    Note:
        Datasets are converted to PyTorch tensors format for compatibility
        with the HuggingFace Trainer API.
    """
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Create HuggingFace datasets from raw lists
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    # Apply tokenization to entire dataset (batched for efficiency)
    train_dataset = train_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        batch_size=len(train_dataset)  # Process entire dataset in one batch
    )
    test_dataset = test_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        batch_size=len(test_dataset)
    )

    # Convert to PyTorch tensors - required format for Trainer
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return train_dataset, test_dataset, tokenizer


# ============================================================
# 5. LOAD MODEL + APPLY LORA
# ============================================================

def load_roberta_lora():
    """
    Load RoBERTa model and apply LoRA (Low-Rank Adaptation) for efficient fine-tuning.
    
    Returns:
        PeftModel: RoBERTa model with LoRA adapters applied
    
    LoRA Configuration:
        - r=8: Rank of low-rank matrices (controls adapter capacity)
        - lora_alpha=16: Scaling factor for LoRA updates (alpha/r determines learning rate scale)
        - lora_dropout=0.1: Dropout probability for regularization
        - target_modules: ['query', 'value'] - Apply LoRA only to attention Q and V matrices
    
    Design Rationale:
        - r=8 chosen as sweet spot between performance and efficiency
        - Targeting only Q/V reduces parameters while maintaining accuracy
        - This configuration trains <1% of parameters vs full fine-tuning
    
    Note:
        Prints the number of trainable parameters for transparency.
    """
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2  # Binary classification: fake (1) vs true (0)
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,  # Low-rank dimension - balance between efficiency and capacity
        lora_alpha=16,  # Scaling factor - controls magnitude of LoRA updates
        lora_dropout=0.1,  # Regularization to prevent overfitting
        bias="none",  # Don't adapt bias terms
        target_modules=["query", "value"]  # Apply LoRA to attention Q/V only
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ============================================================
# 6. TRAINING SETUP
# ============================================================

def get_training_args():
    """
    Configure training hyperparameters for HuggingFace Trainer.
    
    Returns:
        TrainingArguments: Configuration object for model training
    
    Hyperparameter Choices:
        - batch_size=8: Balanced for CPU/small GPU memory constraints
        - max_steps=800: Sufficient for convergence on this dataset size
        - save_steps=200: Checkpoints every 200 steps for recovery
        - logging_steps=50: Frequent logging for monitoring progress
    
    Note:
        Training is capped at 800 steps (overrides num_train_epochs if reached earlier).
        This ensures consistent training duration across different dataset sizes.
    """
    return TrainingArguments(
        output_dir="./roberta_lora_results",
        per_device_train_batch_size=8,  # Optimized for CPU/small GPU
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        max_steps=800,  # Cap training for reproducibility
        save_steps=200,  # Checkpoint every 200 steps
        logging_dir="./logs",
        logging_steps=50  # Frequent logging for monitoring
    )

def get_trainer(model, train_dataset, test_dataset, training_args):
    """
    Create HuggingFace Trainer instance with custom metrics.
    
    Args:
        model (PeftModel): LoRA-adapted RoBERTa model
        train_dataset (Dataset): Tokenized training dataset
        test_dataset (Dataset): Tokenized testing dataset
        training_args (TrainingArguments): Training configuration
    
    Returns:
        Trainer: Configured trainer ready for model.train()
    
    Note:
        compute_metrics callback calculates accuracy during evaluation.
        Additional metrics (precision, recall, F1) are computed separately
        via evaluate_model() for comprehensive reporting.
    """
    def compute_metrics(eval_pred):
        """Calculate accuracy from evaluation predictions."""
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {"accuracy": accuracy_score(labels, preds)}

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )


# ============================================================
# 7. EVALUATION HELPERS
# ============================================================

def evaluate_model(trainer, test_dataset, test_labels):
    """
    Evaluate trained model with comprehensive metrics.
    
    Args:
        trainer (Trainer): Trained HuggingFace Trainer instance
        test_dataset (Dataset): Tokenized test dataset
        test_labels (list): Ground truth labels for test set
    
    Returns:
        dict: Dictionary containing:
              - accuracy: Overall classification accuracy
              - precision: Precision score for positive class
              - recall: Recall score for positive class
              - f1: F1-score (harmonic mean of precision/recall)
              - roc_auc: Area under ROC curve
              - classification_report: Detailed per-class metrics
              - confusion_matrix: 2x2 confusion matrix
    
    Note:
        Uses probability scores for ROC-AUC calculation to assess
        model's discrimination ability beyond hard predictions.
    """
    preds_output = trainer.predict(test_dataset)
    y_pred = preds_output.predictions.argmax(-1)  # Hard predictions
    y_true = test_labels
    y_prob = preds_output.predictions[:, 1]  # Probability scores for class 1

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "classification_report": classification_report(y_true, y_pred, digits=4),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }


# ============================================================
# 8. SHAP EXPLAINABILITY
# ============================================================

def setup_shap(model, tokenizer):
    """
    Initialize SHAP explainer for model interpretability.
    
    Args:
        model (PeftModel): Trained LoRA-adapted RoBERTa model
        tokenizer (RobertaTokenizer): Tokenizer matching the model
    
    Returns:
        shap.Explainer: SHAP explainer instance for generating explanations
    
    Note:
        SHAP (SHapley Additive exPlanations) provides token-level importance
        scores showing which words contributed most to the prediction.
        This helps understand and debug model decisions.
    """
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        truncation=True,
        max_length=256
    )
    return shap.Explainer(classifier)

def shap_explain(explainer, text):
    """
    Generate SHAP explanation for a single text input.
    
    Args:
        explainer (shap.Explainer): Initialized SHAP explainer
        text (str): Input text to explain
    
    Returns:
        shap.Explanation: SHAP values showing token-level contributions
    
    Usage:
        Use shap.plots.text() or shap.plots.waterfall() to visualize
        the explanation and understand which tokens drove the prediction.
    """
    shap_values = explainer([text])
    return shap_values
