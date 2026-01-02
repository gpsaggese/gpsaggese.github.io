#!/usr/bin/env python3
"""
Training script for the Solar / Renewable Energy Forecasting project.

This script:
- Loads the raw hourly solar dataset
- Builds time-based + weather features using `make_basic_time_features`
- Trains:
    * RandomForestRegressor baseline
    * LSTM baseline (deep learning)
    * GRU baseline (deep learning)
- Uses MLflow to log metrics, feature importances (RF), plots, and models
"""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from mlflow.models.signature import infer_signature

import tensorflow as tf  # noqa: F401
from tensorflow import keras
from tensorflow.keras import layers

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
    train_val_split,  # still used for RF
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
    "seq_window": 24,      # sequence length (hours) for LSTM/GRU
    "batch_size": 32,
    "epochs": 15,
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


def make_sequences(df, feature_cols, target_col, window_size, val_start_time):
    """
    Convert a feature DataFrame into 3D sequences for LSTM/GRU.

    Parameters
    ----------
    df : DataFrame indexed by time
    feature_cols : list of feature column names
    target_col : target column name
    window_size : int, number of timesteps per sequence
    val_start_time : timestamp where validation period begins

    Returns
    -------
    (X_train, y_train, X_val, y_val)
      X_* shape: (n_samples, window_size, n_features)
    """
    df = df.sort_index()
    values = df[feature_cols].values
    targets = df[target_col].values
    times = df.index.to_list()

    X_list, y_list, t_list = [], [], []
    for i in range(window_size, len(df)):
        X_list.append(values[i - window_size : i])
        y_list.append(targets[i])
        t_list.append(times[i])

    X = np.array(X_list)
    y = np.array(y_list)
    t = np.array(t_list)

    # Split into train/val using time threshold
    mask_val = t >= val_start_time
    mask_train = ~mask_val

    X_train = X[mask_train]
    y_train = y[mask_train]
    X_val = X[mask_val]
    y_val = y[mask_val]

    return X_train, y_train, X_val, y_val


def build_lstm_model(input_shape):
    """
    Build a simple LSTM regression model.
    input_shape: (window_size, n_features)
    """
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=False),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_gru_model(input_shape):
    """
    Build a simple GRU regression model.
    input_shape: (window_size, n_features)
    """
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.GRU(64, return_sequences=False),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ---------------------------------------------------------------------
# Model runs with MLflow
# ---------------------------------------------------------------------
def run_random_forest(X_train, y_train, X_val, y_val, feature_cols):
    """
    Train and evaluate a RandomForest baseline, logging rich artifacts to MLflow.

    IMPORTANT CHANGE:
    - Log model with a DataFrame signature/input_example so MLflow serving knows column names.
    - Save feature columns list to artifacts (metadata/feature_columns.json).
    """
    model_name = "RandomForest_baseline"

    model = RandomForestRegressor(
        n_estimators=CONFIG.get("rf_n_estimators", 200),
        random_state=CONFIG.get("rf_random_state", 42),
        n_jobs=-1,
    )

    with mlflow.start_run(run_name=model_name):
        # ----- log params -----
        mlflow.log_param("model_type", "RandomForestRegressor")
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

        # ----- artifacts (CSV + plots + feature list) -----
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # 0) Save feature columns for reproducible serving requests
            feature_path = tmpdir_path / "feature_columns.json"
            feature_path.write_text(json.dumps(feature_cols, indent=2))
            mlflow.log_artifact(str(feature_path), artifact_path="metadata")

            # 1) CSV with predictions vs actuals
            preds_df = pd.DataFrame(
                {
                    "y_true": y_val.values,
                    "y_pred": y_pred,
                }
            )
            preds_path = tmpdir_path / "val_predictions_rf.csv"
            preds_df.to_csv(preds_path, index=False)
            mlflow.log_artifact(str(preds_path), artifact_path="analysis")

            # 2) Plot: actual vs predicted
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(y_val.values, label="Actual")
            ax1.plot(y_pred, label="Predicted", alpha=0.8)
            ax1.set_xlabel("Validation index")
            ax1.set_ylabel("Energy (MWh)")
            ax1.set_title("RF: Actual vs Predicted – Validation period")
            ax1.legend()
            fig1.tight_layout()
            fig1_path = tmpdir_path / "rf_val_predictions_plot.png"
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
            fig2_path = tmpdir_path / "rf_feature_importance.png"
            fig2.savefig(fig2_path)
            plt.close(fig2)
            mlflow.log_artifact(str(fig2_path), artifact_path="figures")

        # ----- log model with signature + input_example (DataFrame with columns!) -----
        X_val_df = pd.DataFrame(X_val, columns=feature_cols)
        input_example = X_val_df.head(5)
        signature = infer_signature(X_val_df, y_pred)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

    return model, y_pred


def run_deep_model(model_name, build_fn, X_train, y_train, X_val, y_val):
    """
    Train an LSTM or GRU model and log it to MLflow.
    """
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("csv_path", CONFIG["csv_path"])
        mlflow.log_param("test_size_days", CONFIG["test_size_days"])
        mlflow.log_param("seq_window", CONFIG["seq_window"])
        mlflow.log_param("batch_size", CONFIG["batch_size"])
        mlflow.log_param("epochs", CONFIG["epochs"])

        input_shape = X_train.shape[1:]  # (window_size, n_features)
        model = build_fn(input_shape)

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            verbose=1,
        )

        # Predictions
        y_pred = model.predict(X_val).reshape(-1)

        # Metrics
        rmse, mae, r2 = compute_metrics(y_val, y_pred)
        print(f"[{model_name}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Artifacts: loss curves + prediction plot
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # 1) Loss curves
            hist_df = pd.DataFrame(history.history)
            hist_path = tmpdir_path / f"{model_name}_history.csv"
            hist_df.to_csv(hist_path, index=False)
            mlflow.log_artifact(str(hist_path), artifact_path="analysis")

            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(history.history["loss"], label="train_loss")
            ax1.plot(history.history["val_loss"], label="val_loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("MSE loss")
            ax1.set_title(f"{model_name} Training Loss")
            ax1.legend()
            fig1.tight_layout()
            fig1_path = tmpdir_path / f"{model_name}_loss_curve.png"
            fig1.savefig(fig1_path)
            plt.close(fig1)
            mlflow.log_artifact(str(fig1_path), artifact_path="figures")

            # 2) Prediction vs actual
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(y_val, label="Actual")
            ax2.plot(y_pred, label="Predicted", alpha=0.8)
            ax2.set_xlabel("Validation index")
            ax2.set_ylabel("Energy (MWh)")
            ax2.set_title(f"{model_name}: Actual vs Predicted – Validation period")
            ax2.legend()
            fig2.tight_layout()
            fig2_path = tmpdir_path / f"{model_name}_val_predictions.png"
            fig2.savefig(fig2_path)
            plt.close(fig2)
            mlflow.log_artifact(str(fig2_path), artifact_path="figures")

        # Log model
        mlflow.tensorflow.log_model(model, artifact_path="model")

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

    # ---- Random Forest (using existing 2D feature split) ----
    X_train_rf, X_val_rf, y_train_rf, y_val_rf, feature_cols = train_val_split(
        df_feats,
        test_size_days=CONFIG["test_size_days"],
    )
    print(f"[train] RF X_train: {X_train_rf.shape}, X_val: {X_val_rf.shape}")
    print(f"[train] RF y_train: {y_train_rf.shape}, y_val: {y_val_rf.shape}")
    print(f"[train] Num features        : {len(feature_cols)}")
    print(f"[train] Feature cols (first 10): {feature_cols[:10]}")

    run_random_forest(X_train_rf, y_train_rf, X_val_rf, y_val_rf, feature_cols)

    # ---- LSTM / GRU (sequence-based) ----
    # Determine validation start time
    max_time = df_feats.index.max()
    val_start_time = max_time - pd.Timedelta(days=CONFIG["test_size_days"])
    print(f"[train] Validation starts at: {val_start_time}")

    X_train_seq, y_train_seq, X_val_seq, y_val_seq = make_sequences(
        df_feats,
        feature_cols,
        TARGET_COL,
        window_size=CONFIG["seq_window"],
        val_start_time=val_start_time,
    )
    print(f"[train] Seq X_train: {X_train_seq.shape}, X_val: {X_val_seq.shape}")
    print(f"[train] Seq y_train: {y_train_seq.shape}, y_val: {y_val_seq.shape}")

    # LSTM run
    run_deep_model(
        model_name="LSTM_baseline",
        build_fn=build_lstm_model,
        X_train=X_train_seq,
        y_train=y_train_seq,
        X_val=X_val_seq,
        y_val=y_val_seq,
    )

    # GRU run
    run_deep_model(
        model_name="GRU_baseline",
        build_fn=build_gru_model,
        X_train=X_train_seq,
        y_train=y_train_seq,
        X_val=X_val_seq,
        y_val=y_val_seq,
    )

    print("[train] Done.")


if __name__ == "__main__":
    main()
