"""
Model evaluation utilities for the house price prediction pipeline

This module contains:
- Evaluation metrics calculation (RMSE, R², MAE)
- Cross-validation functions
- Model comparison utilities
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, Tuple

from . import config


def calculate_rmse(y_true, y_pred) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_r2(y_true, y_pred) -> float:
    """
    Calculate R² score.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        R² score
    """
    return r2_score(y_true, y_pred)


def calculate_mae(y_true, y_pred) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def evaluate_model(model, X, y) -> Dict[str, float]:
    """
    Evaluate model on given data.

    Args:
        model: Trained model
        X: Features
        y: Target values

    Returns:
        Dictionary of evaluation metrics

    TODO: Implement in Phase 5
    """
    y_pred = model.predict(X)

    metrics = {
        "rmse": calculate_rmse(y, y_pred),
        "r2": calculate_r2(y, y_pred),
        "mae": calculate_mae(y, y_pred)
    }

    return metrics


def cross_validate_model(model, X, y, cv_folds=None) -> Dict[str, Any]:
    """
    Perform cross-validation on the model.

    Args:
        model: Model to evaluate
        X: Features
        y: Target values
        cv_folds: Number of CV folds (optional)

    Returns:
        Dictionary of cross-validation results

    TODO: Implement in Phase 5
    """
    if cv_folds is None:
        cv_folds = config.CV_FOLDS

    # Placeholder - will be implemented in Phase 5
    results = {
        "cv_folds": cv_folds,
        "scores": [],
        "mean_score": 0.0,
        "std_score": 0.0
    }

    return results


def compare_models(model1_results: Dict, model2_results: Dict) -> Dict[str, Any]:
    """
    Compare two models based on evaluation metrics.

    Args:
        model1_results: Evaluation results for model 1
        model2_results: Evaluation results for model 2

    Returns:
        Comparison summary

    TODO: Implement in Phase 5
    """
    comparison = {
        "model1": model1_results,
        "model2": model2_results,
        "winner": None
    }

    # Determine winner based on RMSE (lower is better)
    if model1_results.get("rmse", float('inf')) < model2_results.get("rmse", float('inf')):
        comparison["winner"] = "model1"
    else:
        comparison["winner"] = "model2"

    return comparison


def test_evaluation():
    """Test function to verify evaluation utilities work."""
    print("Evaluation utilities placeholder - will be implemented in Phase 5")

    # Simple test
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 390, 510])

    rmse = calculate_rmse(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)

    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Test MAE: {mae:.2f}")


if __name__ == "__main__":
    test_evaluation()
