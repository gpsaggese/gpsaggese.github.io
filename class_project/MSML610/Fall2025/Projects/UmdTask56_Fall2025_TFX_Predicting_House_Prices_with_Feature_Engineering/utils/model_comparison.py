"""
Model Comparison Module for House Price Prediction

This module contains:
- Multiple regression model implementations (XGBoost, RF, GBM, Ridge, Lasso, ElasticNet)
- Ensemble methods (Voting Regressor, Stacking Regressor)
- Model training and comparison utilities
- Cross-validation support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import pickle
import time

# Scikit-learn models
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score

# XGBoost
import xgboost as xgb

# Import local modules
from . import config
from .evaluation_utils import (
    calculate_rmse,
    calculate_mae,
    calculate_r2,
    cross_validate_model
)


class ModelRegistry:
    """
    Registry for all regression models used in comparison.
    """

    @staticmethod
    def get_xgboost_model() -> xgb.XGBRegressor:
        """
        Get configured XGBoost model.

        Returns:
            XGBoost regressor with optimized hyperparameters
        """
        params = config.XGBOOST_PARAMS.copy()
        # Remove early_stopping_rounds from init params
        params.pop('early_stopping_rounds', None)

        model = xgb.XGBRegressor(
            n_estimators=params.get('n_estimators', 1000),
            max_depth=params.get('max_depth', 7),
            learning_rate=params.get('learning_rate', 0.01),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            min_child_weight=params.get('min_child_weight', 3),
            gamma=params.get('gamma', 0),
            reg_alpha=params.get('reg_alpha', 0.1),
            reg_lambda=params.get('reg_lambda', 1),
            random_state=params.get('random_state', 42),
            objective=params.get('objective', 'reg:squarederror'),
            n_jobs=-1
        )
        return model

    @staticmethod
    def get_random_forest_model() -> RandomForestRegressor:
        """
        Get configured Random Forest model.

        Returns:
            Random Forest regressor
        """
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        return model

    @staticmethod
    def get_gradient_boosting_model() -> GradientBoostingRegressor:
        """
        Get configured Gradient Boosting model.

        Returns:
            Gradient Boosting regressor
        """
        model = GradientBoostingRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        return model

    @staticmethod
    def get_ridge_model() -> Ridge:
        """
        Get configured Ridge regression model.

        Returns:
            Ridge regressor
        """
        model = Ridge(
            alpha=10.0,
            random_state=42
        )
        return model

    @staticmethod
    def get_lasso_model() -> Lasso:
        """
        Get configured Lasso regression model.

        Returns:
            Lasso regressor
        """
        model = Lasso(
            alpha=0.001,
            max_iter=10000,
            random_state=42
        )
        return model

    @staticmethod
    def get_elasticnet_model() -> ElasticNet:
        """
        Get configured ElasticNet regression model.

        Returns:
            ElasticNet regressor
        """
        model = ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=10000,
            random_state=42
        )
        return model

    @staticmethod
    def get_voting_ensemble(base_models: Optional[List[Tuple[str, Any]]] = None) -> VotingRegressor:
        """
        Get Voting Regressor ensemble.

        Args:
            base_models: List of (name, model) tuples. If None, uses default models.

        Returns:
            Voting regressor ensemble
        """
        if base_models is None:
            base_models = [
                ('xgb', ModelRegistry.get_xgboost_model()),
                ('rf', ModelRegistry.get_random_forest_model()),
                ('gbm', ModelRegistry.get_gradient_boosting_model())
            ]

        model = VotingRegressor(
            estimators=base_models,
            n_jobs=-1
        )
        return model

    @staticmethod
    def get_stacking_ensemble(
        base_models: Optional[List[Tuple[str, Any]]] = None,
        meta_model: Optional[Any] = None
    ) -> StackingRegressor:
        """
        Get Stacking Regressor ensemble.

        Args:
            base_models: List of (name, model) tuples. If None, uses default models.
            meta_model: Meta-learner model. If None, uses Ridge.

        Returns:
            Stacking regressor ensemble
        """
        if base_models is None:
            base_models = [
                ('xgb', ModelRegistry.get_xgboost_model()),
                ('rf', ModelRegistry.get_random_forest_model()),
                ('gbm', ModelRegistry.get_gradient_boosting_model()),
                ('ridge', ModelRegistry.get_ridge_model())
            ]

        if meta_model is None:
            meta_model = Ridge(alpha=1.0)

        model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            n_jobs=-1
        )
        return model

    @classmethod
    def get_all_models(cls) -> Dict[str, Any]:
        """
        Get all available models for comparison.

        Returns:
            Dictionary mapping model names to model instances
        """
        models = {
            'XGBoost': cls.get_xgboost_model(),
            'RandomForest': cls.get_random_forest_model(),
            'GradientBoosting': cls.get_gradient_boosting_model(),
            'Ridge': cls.get_ridge_model(),
            'Lasso': cls.get_lasso_model(),
            'ElasticNet': cls.get_elasticnet_model(),
            'VotingEnsemble': cls.get_voting_ensemble(),
            'StackingEnsemble': cls.get_stacking_ensemble()
        }
        return models


def train_and_evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Train a model and evaluate its performance.

    Args:
        model: Scikit-learn compatible model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_name: Name of the model (for logging)

    Returns:
        Dictionary with training time and evaluation metrics
    """
    print(f"\nTraining {model_name}...")
    start_time = time.time()

    # Train the model
    model.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"{model_name} training completed in {training_time:.2f} seconds")

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    results = {
        'model_name': model_name,
        'training_time': training_time,
        'train_rmse': calculate_rmse(y_train, y_pred_train),
        'train_mae': calculate_mae(y_train, y_pred_train),
        'train_r2': calculate_r2(y_train, y_pred_train),
        'test_rmse': calculate_rmse(y_test, y_pred_test),
        'test_mae': calculate_mae(y_test, y_pred_test),
        'test_r2': calculate_r2(y_test, y_pred_test),
        'rmse': calculate_rmse(y_test, y_pred_test),  # Alias for comparison
        'mae': calculate_mae(y_test, y_pred_test),    # Alias for comparison
        'r2': calculate_r2(y_test, y_pred_test)       # Alias for comparison
    }

    print(f"{model_name} Results:")
    print(f"  Train RMSE: {results['train_rmse']:.4f}")
    print(f"  Test RMSE:  {results['test_rmse']:.4f}")
    print(f"  Test MAE:   {results['test_mae']:.4f}")
    print(f"  Test R²:    {results['test_r2']:.4f}")

    return results


def train_and_evaluate_with_cv(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "Model",
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Train a model with cross-validation.

    Args:
        model: Scikit-learn compatible model
        X: Features
        y: Target
        model_name: Name of the model
        cv_folds: Number of CV folds

    Returns:
        Dictionary with cross-validation results and metrics
    """
    print(f"\nTraining {model_name} with {cv_folds}-fold cross-validation...")
    start_time = time.time()

    # Perform cross-validation
    cv_results = cross_validate_model(model, X, y, cv_folds=cv_folds)

    # Train on full dataset
    model.fit(X, y)
    training_time = time.time() - start_time

    # Make predictions on training set
    y_pred = model.predict(X)

    results = {
        'model_name': model_name,
        'training_time': training_time,
        'cv_mean_rmse': cv_results['mean_rmse'],
        'cv_std_rmse': cv_results['std_rmse'],
        'cv_min_rmse': cv_results['min_rmse'],
        'cv_max_rmse': cv_results['max_rmse'],
        'cv_scores': cv_results['scores'],
        'train_rmse': calculate_rmse(y, y_pred),
        'train_mae': calculate_mae(y, y_pred),
        'train_r2': calculate_r2(y, y_pred),
        'rmse': cv_results['mean_rmse'],  # Use CV RMSE for comparison
        'mae': calculate_mae(y, y_pred),
        'r2': calculate_r2(y, y_pred)
    }

    print(f"{model_name} CV Results:")
    print(f"  CV RMSE: {results['cv_mean_rmse']:.4f} (+/- {results['cv_std_rmse']:.4f})")
    print(f"  Train RMSE: {results['train_rmse']:.4f}")
    print(f"  Train R²: {results['train_r2']:.4f}")

    return results


def compare_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    use_cv: bool = True,
    cv_folds: int = 5,
    models_to_compare: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compare all regression models.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features (optional, required if use_cv=False)
        y_test: Test target (optional, required if use_cv=False)
        use_cv: Whether to use cross-validation
        cv_folds: Number of CV folds
        models_to_compare: List of model names to compare (None = all)

    Returns:
        Dictionary with all model results and comparison
    """
    print("=" * 80)
    print("MODEL COMPARISON: HOUSE PRICE PREDICTION")
    print("=" * 80)

    # Get all models
    all_models = ModelRegistry.get_all_models()

    # Filter models if specified
    if models_to_compare is not None:
        all_models = {k: v for k, v in all_models.items() if k in models_to_compare}

    results = {}

    # Train and evaluate each model
    for model_name, model in all_models.items():
        try:
            if use_cv:
                # Use cross-validation
                model_results = train_and_evaluate_with_cv(
                    model, X_train, y_train,
                    model_name=model_name,
                    cv_folds=cv_folds
                )
            else:
                # Use train-test split
                if X_test is None or y_test is None:
                    raise ValueError("X_test and y_test required when use_cv=False")

                model_results = train_and_evaluate_model(
                    model, X_train, y_train, X_test, y_test,
                    model_name=model_name
                )

            results[model_name] = model_results

        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            results[model_name] = {
                'error': str(e),
                'rmse': float('inf'),
                'mae': float('inf'),
                'r2': -float('inf')
            }

    # Compare models using evaluation_utils
    from .evaluation_utils import compare_models
    comparison = compare_models(results)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\nBest Model: {comparison['best_model']}")
    print(f"Best RMSE: {comparison['summary']['best_rmse']:.4f}")
    print(f"Worst RMSE: {comparison['summary']['worst_rmse']:.4f}")
    print(f"Improvement: {comparison['summary']['rmse_improvement']:.4f}")

    print("\nRankings by RMSE (lower is better):")
    for i, model_name in enumerate(comparison['rankings']['by_rmse'], 1):
        rmse = results[model_name].get('rmse', float('inf'))
        print(f"  {i}. {model_name:20s} - RMSE: {rmse:.4f}")

    print("\nRankings by R² (higher is better):")
    for i, model_name in enumerate(comparison['rankings']['by_r2'], 1):
        r2 = results[model_name].get('r2', -float('inf'))
        print(f"  {i}. {model_name:20s} - R²: {r2:.4f}")

    return {
        'results': results,
        'comparison': comparison
    }


def save_model(model: Any, filepath: str):
    """
    Save a trained model to disk.

    Args:
        model: Trained model
        filepath: Path to save the model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.

    Args:
        filepath: Path to the saved model

    Returns:
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    print("Model Comparison Module")
    print("This module provides utilities for comparing multiple regression models")
    print("\nAvailable models:")
    models = ModelRegistry.get_all_models()
    for i, model_name in enumerate(models.keys(), 1):
        print(f"  {i}. {model_name}")
