"""
utils_model_training.py

Utility functions for model training, evaluation, and management with MCP integration.

This module provides:
- Model context management using MCP concepts
- Model training with various architectures (BERT, LSTM, Sklearn)
- Cross-validation and performance evaluation
- Model versioning and persistence
"""

import logging
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MCP Concepts: Model Context Management
# ============================================================================


@dataclass
class ModelContext:
    """
    Represents the context in which a model operates.

    This dataclass encapsulates metadata about a model and its training conditions,
    following MCP principles for context-aware model management.
    """
    model_id: str
    model_name: str
    model_type: str  # e.g., 'logistic_regression', 'random_forest', 'lstm'
    feature_type: str  # e.g., 'tfidf', 'embeddings'
    created_at: str
    training_samples: int
    validation_samples: int
    test_samples: int
    preprocessed: bool
    hyperparameters: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert context to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ModelContextManager:
    """
    Manages model contexts following MCP principles.

    Ensures models are used in appropriate scenarios and tracks their metadata.
    """

    def __init__(self):
        """Initialize the context manager."""
        self.contexts: Dict[str, ModelContext] = {}
        logger.info("ModelContextManager initialized")

    def register_context(self, context: ModelContext) -> None:
        """
        Register a new model context.

        :param context: ModelContext object to register
        """
        self.contexts[context.model_id] = context
        logger.info(f"Registered context for model: {context.model_id}")

    def get_context(self, model_id: str) -> Optional[ModelContext]:
        """
        Retrieve a model context by ID.

        :param model_id: unique model identifier
        :return: ModelContext or None if not found
        """
        return self.contexts.get(model_id)

    def is_context_compatible(
        self,
        model_id: str,
        feature_type: str,
        data_preprocessing: bool
    ) -> bool:
        """
        Check if a model's context is compatible with current scenario.

        :param model_id: model identifier
        :param feature_type: type of features being used
        :param data_preprocessing: whether preprocessing is applied
        :return: True if compatible, False otherwise
        """
        context = self.get_context(model_id)
        if context is None:
            logger.warning(f"Context not found for model: {model_id}")
            return False

        compatibility = (
            context.feature_type == feature_type and
            context.preprocessed == data_preprocessing
        )

        if not compatibility:
            logger.warning(
                f"Model context mismatch for {model_id}. "
                f"Expected: feature_type={context.feature_type}, "
                f"preprocessed={context.preprocessed}. "
                f"Got: feature_type={feature_type}, preprocessed={data_preprocessing}"
            )

        return compatibility

    def list_contexts(self) -> List[ModelContext]:
        """
        List all registered contexts.

        :return: list of ModelContext objects
        """
        return list(self.contexts.values())


# ============================================================================
# Model Training and Evaluation
# ============================================================================


def create_model(model_type: str, hyperparameters: Dict[str, Any] = None):
    """
    Create a classification model with specified type and hyperparameters.

    Supported types:
    - 'logistic_regression': Logistic Regression classifier
    - 'random_forest': Random Forest classifier
    - 'gradient_boosting': Gradient Boosting classifier

    :param model_type: type of model to create
    :param hyperparameters: dict of hyperparameters (optional)
    :return: initialized model
    """
    logger.info(f"Creating model of type: {model_type}")

    if hyperparameters is None:
        hyperparameters = {}

    if model_type == 'logistic_regression':
        return LogisticRegression(
            max_iter=hyperparameters.get('max_iter', 1000),
            random_state=42,
            **{k: v for k, v in hyperparameters.items() if k != 'max_iter'}
        )
    elif model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=hyperparameters.get('n_estimators', 100),
            max_depth=hyperparameters.get('max_depth', 20),
            random_state=42,
            n_jobs=-1,
            **{k: v for k, v in hyperparameters.items() if k not in ['n_estimators', 'max_depth']}
        )
    elif model_type == 'gradient_boosting':
        return GradientBoostingClassifier(
            n_estimators=hyperparameters.get('n_estimators', 100),
            max_depth=hyperparameters.get('max_depth', 5),
            learning_rate=hyperparameters.get('learning_rate', 0.1),
            random_state=42,
            **{k: v for k, v in hyperparameters.items() if k not in ['n_estimators', 'max_depth', 'learning_rate']}
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = 'unknown'
) -> Dict[str, Any]:
    """
    Train a model on training data.

    :param model: sklearn model object
    :param X_train: training features
    :param y_train: training labels
    :param model_type: name of model type (for logging)
    :return: training results dictionary
    """
    logger.info(f"Training {model_type} model on {X_train.shape[0]} samples...")

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    logger.info(f"Training accuracy: {train_acc:.4f}")

    return {
        'model': model,
        'training_accuracy': train_acc
    }


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = 'Model'
) -> Dict[str, float]:
    """
    Evaluate model performance on test data.

    Metrics include: Accuracy, Precision, Recall, F1, ROC-AUC

    :param model: trained sklearn model
    :param X_test: test features
    :param y_test: test labels
    :param model_name: name of model (for logging)
    :return: dictionary of evaluation metrics
    """
    logger.info(f"Evaluating {model_name} on {X_test.shape[0]} test samples...")

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

    logger.info(f"{model_name} Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    return metrics


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    model_name: str = 'Model'
) -> Dict[str, float]:
    """
    Perform k-fold cross-validation on a model.

    :param model: sklearn model object
    :param X: feature matrix
    :param y: target labels
    :param cv: number of cross-validation folds
    :param model_name: name of model (for logging)
    :return: cross-validation scores
    """
    logger.info(f"Performing {cv}-fold cross-validation for {model_name}...")

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Cross-validate multiple metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = {}

    for metric in metrics:
        try:
            scores = cross_val_score(
                model, X, y,
                cv=cv_splitter,
                scoring=metric
            )
            cv_results[f'{metric}_mean'] = scores.mean()
            cv_results[f'{metric}_std'] = scores.std()
            logger.info(f"  {metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        except Exception as e:
            logger.warning(f"Could not compute {metric}: {e}")

    return cv_results


def compare_models(
    models_config: Dict[str, Dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Train and evaluate multiple models, returning a comparison table.

    :param models_config: dict with model names and their configs
                         e.g., {'logistic_regression': {'hyperparameters': {...}}}
    :param X_train: training features
    :param y_train: training labels
    :param X_test: test features
    :param y_test: test labels
    :return: pandas DataFrame with results
    """
    logger.info(f"Comparing {len(models_config)} models...")

    results = []

    for model_name, config in models_config.items():
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_name}")
            logger.info(f"{'='*60}")

            # Create and train model
            model_type = config.get('model_type', model_name)
            hyperparams = config.get('hyperparameters', {})
            model = create_model(model_type, hyperparams)

            train_results = train_model(model, X_train, y_train, model_name)

            # Evaluate
            metrics = evaluate_model(train_results['model'], X_test, y_test, model_name)

            # Prepare result row
            result_row = {
                'model_name': model_name,
                'model_type': model_type,
                **metrics
            }

            results.append(result_row)

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            continue

    results_df = pd.DataFrame(results)
    logger.info(f"\nModel Comparison Results:\n{results_df.to_string()}")

    return results_df


# ============================================================================
# Model Persistence
# ============================================================================


def save_model(model, filepath: str) -> None:
    """
    Save trained model to disk.

    :param model: trained model object
    :param filepath: path to save model
    """
    logger.info(f"Saving model to {filepath}")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"Model saved successfully")


def load_model(filepath: str):
    """
    Load trained model from disk.

    :param filepath: path to saved model
    :return: loaded model object
    """
    logger.info(f"Loading model from {filepath}")

    with open(filepath, 'rb') as f:
        model = pickle.load(f)

    logger.info(f"Model loaded successfully")
    return model


def save_context(context: ModelContext, filepath: str) -> None:
    """
    Save model context metadata to JSON file.

    :param context: ModelContext object
    :param filepath: path to save context
    """
    logger.info(f"Saving model context to {filepath}")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        f.write(context.to_json())

    logger.info(f"Context saved successfully")


def load_context(filepath: str) -> ModelContext:
    """
    Load model context metadata from JSON file.

    :param filepath: path to saved context
    :return: ModelContext object
    """
    logger.info(f"Loading model context from {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    context = ModelContext(**data)
    logger.info(f"Context loaded successfully")
    return context


# ============================================================================
# Confusion Matrix and Classification Report
# ============================================================================


def get_confusion_matrix_data(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Generate confusion matrix and related metrics.

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: dictionary with confusion matrix and metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        output_dict=True,
        zero_division=0
    )

    return {
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
