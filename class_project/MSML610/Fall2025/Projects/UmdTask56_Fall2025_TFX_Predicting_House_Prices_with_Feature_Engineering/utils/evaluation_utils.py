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


def cross_validate_model(model, X, y, cv_folds=None, scoring='neg_root_mean_squared_error') -> Dict[str, Any]:
    """
    Perform cross-validation on the model.

    Args:
        model: Model to evaluate
        X: Features
        y: Target values (can be log-transformed)
        cv_folds: Number of CV folds (optional)
        scoring: Scoring metric for cross-validation

    Returns:
        Dictionary of cross-validation results with RMSE scores
    """
    if cv_folds is None:
        cv_folds = config.CV_FOLDS

    # Perform cross-validation with negative RMSE (sklearn convention)
    cv_scores = cross_val_score(
        model, X, y,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=-1  # Use all CPU cores
    )

    # Convert negative RMSE back to positive
    cv_scores = -cv_scores

    results = {
        "cv_folds": cv_folds,
        "scores": cv_scores.tolist(),
        "mean_rmse": float(np.mean(cv_scores)),
        "std_rmse": float(np.std(cv_scores)),
        "min_rmse": float(np.min(cv_scores)),
        "max_rmse": float(np.max(cv_scores)),
        "scoring": scoring
    }

    return results


def compare_models(models_results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Compare multiple models based on evaluation metrics.

    Args:
        models_results: Dictionary mapping model names to their evaluation results
                       Each result should have 'rmse', 'mae', 'r2' keys

    Returns:
        Comparison summary with rankings and best model

    Example:
        results = {
            'XGBoost': {'rmse': 25000, 'mae': 18000, 'r2': 0.89},
            'RandomForest': {'rmse': 27000, 'mae': 19000, 'r2': 0.87},
            'TF_DNN': {'rmse': 30000, 'mae': 20000, 'r2': 0.85}
        }
        comparison = compare_models(results)
    """
    if not models_results:
        return {"error": "No models to compare"}

    comparison = {
        "models": models_results,
        "rankings": {},
        "best_model": None,
        "summary": {}
    }

    # Rank by RMSE (lower is better)
    rmse_ranking = sorted(
        models_results.items(),
        key=lambda x: x[1].get('rmse', float('inf'))
    )
    comparison["rankings"]["by_rmse"] = [name for name, _ in rmse_ranking]

    # Rank by MAE (lower is better)
    mae_ranking = sorted(
        models_results.items(),
        key=lambda x: x[1].get('mae', float('inf'))
    )
    comparison["rankings"]["by_mae"] = [name for name, _ in mae_ranking]

    # Rank by R² (higher is better)
    r2_ranking = sorted(
        models_results.items(),
        key=lambda x: x[1].get('r2', -float('inf')),
        reverse=True
    )
    comparison["rankings"]["by_r2"] = [name for name, _ in r2_ranking]

    # Best model is top by RMSE (primary metric)
    comparison["best_model"] = rmse_ranking[0][0]

    # Calculate performance summary
    comparison["summary"] = {
        "best_rmse": rmse_ranking[0][1].get('rmse'),
        "worst_rmse": rmse_ranking[-1][1].get('rmse'),
        "rmse_improvement": (
            rmse_ranking[-1][1].get('rmse', 0) -
            rmse_ranking[0][1].get('rmse', 0)
        )
    }

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
