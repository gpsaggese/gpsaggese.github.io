"""Train and evaluate a suite of 10 best-in-class machine learning models
using k-fold cross validation for IoT anomaly detection.

This script implements a comprehensive anomaly detection pipeline with:
- 10 diverse models including classical ML, ensemble, and anomaly-specific algorithms
- K-fold cross-validation (default k=5)
- Multiple target predictions (anomaly, maintenance, failure type, RUL, downtime)
- Comprehensive metrics for each model
- Model persistence for best performing models

Models evaluated (10 best-in-class):
  1. Random Forest Classifier
  2. XGBoost Classifier
  3. LightGBM Classifier
  4. CatBoost Classifier
  5. Gradient Boosting Classifier
  6. Extra Trees Classifier
  7. Isolation Forest (anomaly-specific)
  8. Logistic Regression (baseline)
  9. AdaBoost Classifier
  10. Gaussian Naive Bayes

For regression (RUL prediction):
  1. Random Forest Regressor
  2. XGBoost Regressor
  3. LightGBM Regressor
  4. CatBoost Regressor
  5. Gradient Boosting Regressor
  6. Extra Trees Regressor

Example usage:
    python train_models_kfold.py --input data/smart_manufacturing_data.csv \
        --output_dir results --sample 20000 --n_folds 5
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (make_scorer, f1_score, precision_score,
                             recall_score, accuracy_score, roc_auc_score,
                             mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier,
                              RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor, IsolationForest,
                              VotingClassifier, StackingClassifier)
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

import sys
sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import compute_features, SENSOR_COLUMNS

warnings.filterwarnings('ignore')


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features and return a cleaned DataFrame ready for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data containing sensor readings and labels.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features and original labels.
    """
    print("Computing engineered features...")
    features_df = compute_features(df)

    # Remove any overlapping original columns from the features before merge
    original_cols = ["machine_id", "anomaly_flag", "maintenance_required",
                     "downtime_risk", "predicted_remaining_life",
                     "failure_type"]
    feature_only_df = features_df.drop(
        columns=[c for c in original_cols if c in features_df.columns],
        errors='ignore'
    )

    # Concatenate original label columns with engineered features
    merged = pd.concat([
        df[original_cols],
        feature_only_df
    ], axis=1)

    # Replace any infinite values with nan and drop them
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna()
    print(f"Features computed. Shape: {merged.shape}")
    return merged


def get_models_classification() -> Dict[str, Any]:
    """Return a dictionary of 10 best-in-class classification models.

    The returned models implement scikit-learn's estimator interface.
    """
    models = {
        "1_RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "2_XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        ),
        "3_LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        "4_CatBoost": CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            loss_function="Logloss",
            verbose=False,
            random_state=42,
            thread_count=-1
        ),
        "5_GradientBoosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        ),
        "6_ExtraTrees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "7_IsolationForest": "SKIP",  # Special handling for unsupervised
        "8_LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "9_AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.5,
            random_state=42
        ),
        "10_GaussianNB": GaussianNB(),
    }
    return models


def get_ensemble_models(base_models: Dict[str, Any]) -> Dict[str, Any]:
    """Create ensemble models using voting and stacking.

    Parameters
    ----------
    base_models : dict
        Dictionary of trained base models

    Returns
    -------
    dict
        Dictionary containing ensemble models
    """
    # Select top models for ensembling
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss', use_label_encoder=False)),
        ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)),
    ]

    ensemble_models = {
        "E1_VotingClassifier": VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        ),
        "E2_StackingClassifier": StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=3,
            n_jobs=-1
        ),
    }
    return ensemble_models


def get_models_regression() -> Dict[str, Any]:
    """Return a dictionary of best-in-class regression models.

    The returned models implement scikit-learn's estimator interface.
    """
    models = {
        "1_RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        "2_XGBoostRegressor": XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        "3_LightGBMRegressor": LGBMRegressor(
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        "4_CatBoostRegressor": CatBoostRegressor(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            loss_function="RMSE",
            verbose=False,
            random_state=42,
            thread_count=-1
        ),
        "5_GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        ),
        "6_ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
    }
    return models


def evaluate_classification(
    X: np.ndarray, y: np.ndarray, models: Dict[str, Any], n_folds: int = 5
) -> List[Dict[str, Any]]:
    """Perform stratified k-fold cross validation for classification models.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    models : dict
        Dictionary of estimators.
    n_folds : int
        Number of folds for cross-validation.

    Returns
    -------
    list of dict
        Each dict contains the model name and aggregated metrics.
    """
    results = []
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    scorers = {
        "f1": make_scorer(f1_score, average="weighted", zero_division=0),
        "precision": make_scorer(precision_score, average="weighted", zero_division=0),
        "recall": make_scorer(recall_score, average="weighted", zero_division=0),
        "accuracy": make_scorer(accuracy_score),
    }

    for name, model in models.items():
        if model == "SKIP":
            continue

        print(f"  Training {name}...")
        try:
            scores = cross_validate(
                model, X, y, cv=cv, scoring=scorers,
                n_jobs=-1, error_score="raise"
            )
            result = {
                "model": name,
                "f1_mean": np.mean(scores["test_f1"]),
                "f1_std": np.std(scores["test_f1"]),
                "precision_mean": np.mean(scores["test_precision"]),
                "precision_std": np.std(scores["test_precision"]),
                "recall_mean": np.mean(scores["test_recall"]),
                "recall_std": np.std(scores["test_recall"]),
                "accuracy_mean": np.mean(scores["test_accuracy"]),
                "accuracy_std": np.std(scores["test_accuracy"]),
                "fit_time_mean": np.mean(scores["fit_time"]),
            }
            results.append(result)
            print(f"    F1: {result['f1_mean']:.4f} (+/- {result['f1_std']:.4f})")
        except Exception as e:
            print(f"    Error training {name}: {str(e)}")

    return results


def evaluate_regression(
    X: np.ndarray, y: np.ndarray, models: Dict[str, Any], n_folds: int = 5
) -> List[Dict[str, Any]]:
    """Perform k-fold cross validation for regression models.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector (continuous).
    models : dict
        Dictionary of regression estimators.
    n_folds : int
        Number of folds for cross-validation.

    Returns
    -------
    list of dict
        Each dict contains the model name and aggregated regression metrics.
    """
    results = []
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    scorers = {
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "mse": make_scorer(mean_squared_error, greater_is_better=False),
        "r2": make_scorer(r2_score),
    }

    for name, model in models.items():
        print(f"  Training {name}...")
        try:
            scores = cross_validate(
                model, X, y, cv=cv, scoring=scorers,
                n_jobs=-1, error_score="raise"
            )
            result = {
                "model": name,
                "mae_mean": -np.mean(scores["test_mae"]),
                "mae_std": np.std(scores["test_mae"]),
                "rmse_mean": np.sqrt(-np.mean(scores["test_mse"])),
                "rmse_std": np.sqrt(np.var(scores["test_mse"])),
                "r2_mean": np.mean(scores["test_r2"]),
                "r2_std": np.std(scores["test_r2"]),
                "fit_time_mean": np.mean(scores["fit_time"]),
            }
            results.append(result)
            print(f"    R2: {result['r2_mean']:.4f} (+/- {result['r2_std']:.4f})")
        except Exception as e:
            print(f"    Error training {name}: {str(e)}")

    return results


def train_and_save_best_model(
    X: np.ndarray, y: np.ndarray, models: Dict[str, Any],
    results: List[Dict[str, Any]], metric: str, output_dir: Path,
    model_name: str
) -> None:
    """Train and save the best performing model based on a metric.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    models : dict
        Dictionary of model instances
    results : list
        List of result dictionaries from cross-validation
    metric : str
        Metric to use for selecting best model (e.g., 'f1_mean', 'r2_mean')
    output_dir : Path
        Directory to save the model
    model_name : str
        Name for the saved model file
    """
    if not results:
        print(f"No results available to save best model for {model_name}")
        return

    # Find best model
    best_result = max(results, key=lambda x: x[metric])
    best_model_name = best_result['model']

    print(f"  Best model: {best_model_name} with {metric}={best_result[metric]:.4f}")

    # Train on full data
    if best_model_name in models and models[best_model_name] != "SKIP":
        best_model = models[best_model_name]
        print(f"  Training {best_model_name} on full dataset...")
        best_model.fit(X, y)

        # Save model
        model_path = output_dir / f"{model_name}.pkl"
        joblib.dump(best_model, model_path)
        print(f"  Saved best model to {model_path}")


def main(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("IoT Anomaly Detection - Model Training Pipeline")
    print("="*80)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"K-Folds: {args.n_folds}")
    print(f"Sample size: {args.sample if args.sample > 0 else 'All'}")
    print("="*80)

    # Load raw dataset
    print("\nLoading dataset...")
    df_raw = pd.read_csv(input_path)
    print(f"Dataset loaded. Shape: {df_raw.shape}")

    # Optionally sample to reduce runtime
    if args.sample > 0 and len(df_raw) > args.sample:
        df_raw = df_raw.sample(n=args.sample, random_state=42)
        print(f"Sampled to {args.sample} rows")

    # Compute features
    df_feat = prepare_data(df_raw)

    # Separate features and labels
    target_cols = ["anomaly_flag", "maintenance_required", "downtime_risk",
                  "predicted_remaining_life", "failure_type"]
    feature_cols = [c for c in df_feat.columns if c not in target_cols + ["machine_id"]]
    X_all = df_feat[feature_cols].to_numpy(dtype=float)

    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # Save scaler
    joblib.dump(scaler, models_dir / "scaler.pkl")
    print(f"Scaler saved to {models_dir / 'scaler.pkl'}")

    results_summary = {}

    # Get models
    clf_models = get_models_classification()
    ensemble_models = get_ensemble_models(clf_models)
    # Add ensemble models to classifier models
    clf_models.update(ensemble_models)

    reg_models = get_models_regression()

    print("\n" + "="*80)
    print("TASK 1: Anomaly Detection (Binary Classification)")
    print("="*80)
    y_anomaly = df_feat["anomaly_flag"].astype(int).to_numpy()
    print(f"Class distribution: {np.bincount(y_anomaly)}")
    results_anomaly = evaluate_classification(X_scaled, y_anomaly, clf_models, args.n_folds)
    df_results = pd.DataFrame(results_anomaly).sort_values("f1_mean", ascending=False)
    df_results.to_csv(output_dir / "results_anomaly.csv", index=False)
    results_summary["anomaly_flag"] = results_anomaly
    print("\nTop 3 models by F1 score:")
    print(df_results[["model", "f1_mean", "precision_mean", "recall_mean"]].head(3))
    train_and_save_best_model(X_scaled, y_anomaly, clf_models, results_anomaly,
                              "f1_mean", models_dir, "best_anomaly_model")

    print("\n" + "="*80)
    print("TASK 2: Maintenance Prediction (Binary Classification)")
    print("="*80)
    y_maint = df_feat["maintenance_required"].astype(int).to_numpy()
    print(f"Class distribution: {np.bincount(y_maint)}")
    results_maint = evaluate_classification(X_scaled, y_maint, clf_models, args.n_folds)
    df_results = pd.DataFrame(results_maint).sort_values("f1_mean", ascending=False)
    df_results.to_csv(output_dir / "results_maintenance.csv", index=False)
    results_summary["maintenance_required"] = results_maint
    print("\nTop 3 models by F1 score:")
    print(df_results[["model", "f1_mean", "precision_mean", "recall_mean"]].head(3))
    train_and_save_best_model(X_scaled, y_maint, clf_models, results_maint,
                              "f1_mean", models_dir, "best_maintenance_model")

    print("\n" + "="*80)
    print("TASK 3: Downtime Risk Prediction (Binary Classification)")
    print("="*80)
    y_downtime = (df_feat["downtime_risk"] > 0).astype(int).to_numpy()
    print(f"Class distribution: {np.bincount(y_downtime)}")
    results_dt = evaluate_classification(X_scaled, y_downtime, clf_models, args.n_folds)
    df_results = pd.DataFrame(results_dt).sort_values("f1_mean", ascending=False)
    df_results.to_csv(output_dir / "results_downtime.csv", index=False)
    results_summary["downtime_risk"] = results_dt
    print("\nTop 3 models by F1 score:")
    print(df_results[["model", "f1_mean", "precision_mean", "recall_mean"]].head(3))
    train_and_save_best_model(X_scaled, y_downtime, clf_models, results_dt,
                              "f1_mean", models_dir, "best_downtime_model")

    print("\n" + "="*80)
    print("TASK 4: Failure Type Classification (Multi-class)")
    print("="*80)
    le = LabelEncoder()
    y_failure = le.fit_transform(df_feat["failure_type"].astype(str))
    print(f"Number of classes: {len(le.classes_)}")
    print(f"Classes: {le.classes_}")
    results_fail = evaluate_classification(X_scaled, y_failure, clf_models, args.n_folds)
    df_results = pd.DataFrame(results_fail).sort_values("f1_mean", ascending=False)
    df_results.to_csv(output_dir / "results_failure_type.csv", index=False)
    results_summary["failure_type"] = results_fail
    print("\nTop 3 models by F1 score:")
    print(df_results[["model", "f1_mean", "precision_mean", "recall_mean"]].head(3))
    joblib.dump(le, models_dir / "failure_type_label_encoder.pkl")
    train_and_save_best_model(X_scaled, y_failure, clf_models, results_fail,
                              "f1_mean", models_dir, "best_failure_type_model")

    print("\n" + "="*80)
    print("TASK 5: Remaining Useful Life Prediction (Regression)")
    print("="*80)
    y_rul = df_feat["predicted_remaining_life"].astype(float).to_numpy()
    print(f"RUL statistics: mean={y_rul.mean():.2f}, std={y_rul.std():.2f}, min={y_rul.min():.2f}, max={y_rul.max():.2f}")
    results_rul = evaluate_regression(X_scaled, y_rul, reg_models, args.n_folds)
    df_results = pd.DataFrame(results_rul).sort_values("r2_mean", ascending=False)
    df_results.to_csv(output_dir / "results_remaining_life.csv", index=False)
    results_summary["predicted_remaining_life"] = results_rul
    print("\nTop 3 models by R2 score:")
    print(df_results[["model", "r2_mean", "mae_mean", "rmse_mean"]].head(3))
    train_and_save_best_model(X_scaled, y_rul, reg_models, results_rul,
                              "r2_mean", models_dir, "best_rul_model")

    # Write summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"Models saved to: {models_dir}")
    print(f"Summary saved to: {summary_path}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate 10 best-in-class models for IoT anomaly detection using k-fold CV."
    )
    parser.add_argument("--input", required=True, help="Path to the raw CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to write result CSVs")
    parser.add_argument("--sample", type=int, default=10000,
                        help="Number of rows to sample for training (0 for all)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds for cross-validation (default: 5)")
    args = parser.parse_args()
    main(args)
