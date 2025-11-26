#!/usr/bin/env python3
"""
Training script for the Solar / Renewable Energy Forecasting project.

This script:
- Loads the raw hourly solar dataset
- Builds time-based + weather features using `make_basic_time_features`
- Splits into train / validation sets with `train_val_split`
- Trains a RandomForestRegressor baseline model
- Logs parameters, metrics, feature importances, plots, and the model to MLflow
"""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------
# Import project utilities from the project root
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from RenewableEnergy_utils import (  # noqa: E402
    TIME_COL,
    TARGET_COL,
    load_data,
    make_basic_time_features,
    train_val_split,
)

# ---------------------------------------------------------------------
# MLflow experiment + config
# ---------------------------------------------------------------------
EXPERIMENT_NAME = "solar_energy_forecasting"

CONFIG = {
    "csv_path": str(PROJECT_ROOT / "data" / "raw" / "solar_energy.csv"),
    "test_size_days": 7,
    "rf_n_estimators": 200,
    "rf_random_state": 42,
}


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, R^2 as floats."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def run_random_forest(X_train, y_train, X_val, y_val, feature_cols):
    """
    Train and evaluate a RandomForest baseline, logging rich artifacts to MLflow:
    - Params
    - Metrics (rmse, mae, r2)
    - Feature importances (as metrics)
    - Validation predictions (CSV)
    - Plots:
        * Actual vs predicted on validation
        * Feature importances bar plot
    - Model with signature + input_example
    """
    model_name = "RandomForest_baseline"

    model = RandomForestRegressor(
        n_estimators=CONFIG.get("rf_n_estimators", 200),
        random_state=CONFIG.get("rf_random_state", 42),
        n_jobs=-1,
    )

    with mlflow.start_run(run_name=model_name):
        # ----- log params -----
        mlflow.log_param("csv_path", CONFIG["csv_path"])
        mlflow.log_param("test_size_days", CONFIG["test_size_days"])
        mlflow.log_param("rf_n_estimators", CONFIG.get("rf_n_estimators", 200))
        mlflow.log_param("rf_random_state", CONFIG.get("rf_random_state", 42))

        # ----- fit & predict -----
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # ----- metrics -----
        rmse, mae, r2 = compute_metrics(y_val, y_pred)
        print(f"[{model_name}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # ----- feature importances as metrics -----
        importances = model.feature_importances_
        for name, val in zip(feature_cols, importances):
            mlflow.log_metric(f"fi_{name}", float(val))

        # ----- artifacts (CSV + plots) -----
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # 1) CSV with predictions vs actuals
            preds_df = pd.DataFrame(
                {
                    "y_true": y_val.values,
                    "y_pred": y_pred,
                }
            )
            preds_path = tmpdir_path / "val_predictions.csv"
            preds_df.to_csv(preds_path, index=False)
            mlflow.log_artifact(str(preds_path), artifact_path="analysis")

            # 2) Plot: actual vs predicted
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(y_val.values, label="Actual")
            ax1.plot(y_pred, label="Predicted", alpha=0.8)
            ax1.set_xlabel("Validation index")
            ax1.set_ylabel("Energy (MWh)")
            ax1.set_title("Actual vs Predicted – Validation period")
            ax1.legend()
            fig1.tight_layout()
            fig1_path = tmpdir_path / "val_predictions_plot.png"
            fig1.savefig(fig1_path)
            plt.close(fig1)
            mlflow.log_artifact(str(fig1_path), artifact_path="figures")

            # 3) Plot: feature importances
            fi_series = pd.Series(importances, index=feature_cols).sort_values(
                ascending=False
            )
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            fi_series.plot(kind="bar", ax=ax2)
            ax2.set_ylabel("Importance")
            ax2.set_title("RandomForest feature importances")
            fig2.tight_layout()
            fig2_path = tmpdir_path / "feature_importance.png"
            fig2.savefig(fig2_path)
            plt.close(fig2)
            mlflow.log_artifact(str(fig2_path), artifact_path="figures")

        # ----- log model with signature + input_example -----
        input_example = X_val[:5]
        signature = infer_signature(X_val, y_pred)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

    return model, y_pred


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def main() -> None:
    # 1. Set up MLflow tracking
    tracking_uri = f"file:{PROJECT_ROOT / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"[train] Project root       : {PROJECT_ROOT}")
    print(f"[train] Tracking URI       : {tracking_uri}")
    print(f"[train] Experiment name    : {EXPERIMENT_NAME}")
    print(f"[train] Raw CSV path       : {CONFIG['csv_path']}")

    csv_path = Path(CONFIG["csv_path"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {csv_path}")

    # 2. Load raw data
    df_raw = load_data(str(csv_path))
    print(f"[train] Raw shape          : {df_raw.shape}")
    print(f"[train] Columns            : {list(df_raw.columns)}")

    # 3. Build features
    df_feats = make_basic_time_features(df_raw)
    print(f"[train] Feature shape      : {df_feats.shape}")

    # 4. Train / validation split
    X_train, X_val, y_train, y_val, feature_cols = train_val_split(
        df_feats,
        test_size_days=CONFIG["test_size_days"],
    )
    print(f"[train] X_train: {X_train.shape}, X_val: {X_val.shape}")
    print(f"[train] y_train: {y_train.shape}, y_val: {y_val.shape}")
    print(f"[train] Num features        : {len(feature_cols)}")

    # 5. Train RF baseline + log to MLflow
    _model, _y_pred = run_random_forest(
        X_train, y_train, X_val, y_val, feature_cols
    )

    print("[train] Done.")


if __name__ == "__main__":
    main()
