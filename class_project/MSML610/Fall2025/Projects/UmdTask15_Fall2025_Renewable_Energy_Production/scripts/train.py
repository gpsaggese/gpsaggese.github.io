"""
Train a baseline model to forecast solar energy (energy_mwh).

This script uses the feature-enriched dataset created by make_features.py.

Flow:
1. Load data/processed/train.csv
2. Separate features (X) and target (y = energy_mwh)
3. Do a time-aware train/validation split (no shuffling)
4. Train a RandomForestRegressor as a baseline
5. Evaluate with MAE and RMSE
6. Log everything to MLflow (params, metrics, model)
"""

import sys
from pathlib import Path

# Ensure I can import from the project root (for RenewableEnergy_utils, etc.)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RenewableEnergy_utils import PROCESSED_DIR  # type: ignore

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn


def load_processed_data() -> pd.DataFrame:
    """
    Load the processed training data from data/processed/train.csv.
    """
    path = PROCESSED_DIR / "train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. "
            "I need to run `python3 scripts/make_features.py` first."
        )

    df = pd.read_csv(path)
    return df


def train_baseline_model(df: pd.DataFrame, target_col: str = "energy_mwh") -> None:
    """
    Train a simple baseline model, log everything to MLflow,
    and print MAE and RMSE.
    """
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in columns: {list(df.columns)}"
        )

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Use only numeric columns as features.
    X = X.select_dtypes(include=[np.number])

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found for modeling.")

    # Time-aware split: last 20% of rows as validation, no shuffling.
    test_size = 0.2
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=False,
    )

    n_estimators = 200
    random_state = 42

    # Tell MLflow which experiment this belongs to.
    mlflow.set_experiment("RenewableEnergy_Baseline")

    with mlflow.start_run(run_name="random_forest_baseline"):
        # ---- Log parameters ----
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("shuffle", False)
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("n_features", X.shape[1])

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_valid)

        mae = mean_absolute_error(y_valid, preds)
        rmse = mean_squared_error(y_valid, preds, squared=False)

        # ---- Log metrics ----
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("n_train_samples", float(len(X_train)))
        mlflow.log_metric("n_valid_samples", float(len(X_valid)))

        # ---- Log the model itself ----
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        # (Optional) log feature importances as an artifact
        importances = model.feature_importances_
        fi = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        fi_path = PROJECT_ROOT / "feature_importances_rf.csv"
        fi.to_csv(fi_path)
        mlflow.log_artifact(str(fi_path), artifact_path="artifacts")

    # Also print to terminal for quick inspection
    print(f"Number of training samples:   {len(X_train)}")
    print(f"Number of validation samples: {len(X_valid)}")
    print(f"Validation MAE:  {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")


def main() -> None:
    """
    End-to-end training entry point.

    This runs when I call:
    python3 scripts/train.py
    """
    df = load_processed_data()
    train_baseline_model(df, target_col="energy_mwh")


if __name__ == "__main__":
    main()
