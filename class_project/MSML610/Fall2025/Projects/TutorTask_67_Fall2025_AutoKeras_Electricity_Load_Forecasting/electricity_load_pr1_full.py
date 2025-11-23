#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PR1 — Electricity Load Forecasting (Full)
----------------------------------------
Adds ARIMA, Prophet, and LSTM to the simple baselines.
- Gracefully skips a model if its dependency is missing.
- Keeps outputs identical format with extra columns for new models.

Run:
python electricity_load_pr1_full.py --file "/path/to/your.csv" --timestamp-col "datetime" --target-col "nat_demand"

Install (optional):
pip install statsmodels prophet tensorflow
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------
# Helpers
# ----------------------------
CANDIDATE_TIME_COLS = ["timestamp","datetime","date","ds","time","hour","Date","Datetime","Timestamp"]
CANDIDATE_TARGET_COLS = ["target","load","demand","y","y_value","value","power","usage","consumption","nat_demand"]

def find_time_col(df: pd.DataFrame):
    cols_lower = {c.lower(): c for c in df.columns}
    for c in CANDIDATE_TIME_COLS:
        if c in cols_lower:
            return cols_lower[c]
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() > 0.9:
                return c
        except Exception:
            continue
    return None

def find_target_col(df: pd.DataFrame, time_col: str|None):
    cols_lower = {c.lower(): c for c in df.columns}
    for c in CANDIDATE_TARGET_COLS:
        if c in cols_lower:
            return cols_lower[c]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != time_col]
    if len(num_cols) == 0:
        return None
    variances = df[num_cols].var().sort_values(ascending=False)
    return variances.index[0]

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean()) * 100.0

def train_test_split_time(df, test_size_hours: int = 7*24):
    if len(df) <= test_size_hours + 400:
        test_size_hours = max(int(len(df)*0.2), 24*2)
    train = df.iloc[:-test_size_hours].copy()
    test  = df.iloc[-test_size_hours:].copy()
    return train, test

def build_features(df, target_col):
    out = df.copy()
    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    for lag in [1, 2, 24, 48, 168]:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    out["roll_mean_24"] = out[target_col].shift(1).rolling(24, min_periods=12).mean()
    out["roll_mean_168"] = out[target_col].shift(1).rolling(168, min_periods=24).mean()
    out = out.dropna()
    return out

def safe_eval_metrics(y_true, y_pred):
    if np.isnan(y_pred).any():
        return (np.nan, np.nan, np.nan)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mp = mape(y_true, y_pred)
    return mae, rmse, mp

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--timestamp-col", type=str, default="")
    parser.add_argument("--target-col", type=str, default="")
    parser.add_argument("--freq", type=str, default="H")
    parser.add_argument("--outdir", type=str, default="")
    args = parser.parse_args()

    candidate_paths = []
    if args.file:
        candidate_paths.append(Path(args.file))
    candidate_paths += [
        Path("/mnt/data/continuous dataset.csv"),
        Path("/mnt/data/weekly pre-dispatch forecast.csv"),
        Path("/mnt/data/train_dataframes.xlsx"),
        Path("/mnt/data/test_dataframes.xlsx"),
    ]
    file_path = None
    for p in candidate_paths:
        if p.exists():
            file_path = p
            break
    if file_path is None:
        print("ERROR: Could not find a data file. Pass --file pointing to your CSV/Excel.", file=sys.stderr)
        sys.exit(1)

    # Read
    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        xls = pd.ExcelFile(file_path)
        df = None
        for sheet in xls.sheet_names:
            tmp = xls.parse(sheet)
            if tmp.shape[1] >= 2:
                df = tmp
                break
        if df is None:
            print("ERROR: No usable sheet found in Excel file.", file=sys.stderr)
            sys.exit(1)
    else:
        df = pd.read_csv(file_path)

    time_col = args.timestamp_col or find_time_col(df)
    if time_col is None:
        print(f"ERROR: Could not detect a timestamp column in {list(df.columns)}. Use --timestamp-col.", file=sys.stderr)
        sys.exit(1)
    target_col = args.target_col or find_target_col(df, time_col)
    if target_col is None:
        print(f"ERROR: Could not detect a numeric target column in {list(df.columns)}. Use --target-col.", file=sys.stderr)
        sys.exit(1)

    # Parse & index
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", infer_datetime_format=True)
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df[[time_col, target_col]].copy()
    df = df.set_index(time_col).sort_index()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    # Resample and impute
    df = df.resample(args.freq).mean()
    df[target_col] = df[target_col].interpolate(limit_direction="both").fillna(df[target_col].median())

    # Features
    feat_df = build_features(df, target_col)

    # Split
    train_df, test_df = train_test_split_time(feat_df, test_size_hours=7*24)

    FEATURES = [c for c in train_df.columns if c != target_col]
    X_train = train_df[FEATURES].values
    y_train = train_df[target_col].values
    X_test  = test_df[FEATURES].values
    y_test  = test_df[target_col].values

    # Baselines
    lr = LinearRegression().fit(X_train, y_train)
    ridge = Ridge(alpha=1.0, random_state=42).fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42).fit(X_train, y_train)

    pred_lr = lr.predict(X_test)
    pred_ridge = ridge.predict(X_test)
    pred_rf = rf.predict(X_test)

    naive_24 = df[target_col].shift(24).reindex(test_df.index)
    naive_24 = naive_24.fillna(method="bfill").fillna(method="ffill")
    pred_naive = naive_24.values

    # ---------------------------
    # Additional Models
    # ---------------------------
    # ARIMA
    try:
        from statsmodels.tsa.arima.model import ARIMA
        print("\nTraining ARIMA model...")
        arima_model = ARIMA(train_df[target_col], order=(2,1,2))
        arima_fit = arima_model.fit()
        pred_arima = arima_fit.forecast(steps=len(test_df))
        pred_arima = np.asarray(pred_arima, dtype=float)
    except Exception as e:
        print("Skipping ARIMA (error):", e)
        pred_arima = np.full(len(test_df), np.nan)

    # Prophet
    try:
        from prophet import Prophet
        print("\nTraining Prophet model...")
        prophet_df = train_df.reset_index()[[train_df.index.name, target_col]].rename(columns={train_df.index.name: "ds", target_col: "y"})
        m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=len(test_df), freq="H")
        forecast = m.predict(future)
        pred_prophet = forecast.set_index("ds")["yhat"].reindex(test_df.index).values
    except Exception as e:
        print("Skipping Prophet (error):", e)
        pred_prophet = np.full(len(test_df), np.nan)

    # LSTM
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from sklearn.preprocessing import MinMaxScaler
        print("\nTraining LSTM model...")
        scaler = MinMaxScaler()
        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
        # reshape features to [samples, timesteps, features]; using 1 timestep with feature vector
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        model_lstm = Sequential([
            LSTM(64, activation="relu", input_shape=(1, X_train.shape[1])),
            Dense(1)
        ])
        model_lstm.compile(optimizer="adam", loss="mse")
        model_lstm.fit(X_train_lstm, y_train_scaled, epochs=10, batch_size=32, verbose=0)
        pred_lstm_scaled = model_lstm.predict(X_test_lstm, verbose=0)
        pred_lstm = scaler.inverse_transform(pred_lstm_scaled).flatten()
    except Exception as e:
        print("Skipping LSTM (error):", e)
        pred_lstm = np.full(len(test_df), np.nan)

    # ---------------------------
    # Evaluation
    # ---------------------------
    def eval_all(y_true, y_pred):
        return safe_eval_metrics(y_true, y_pred)

    metrics = {
        "seasonal_naive_24": eval_all(y_test, pred_naive),
        "linear_regression": eval_all(y_test, pred_lr),
        "ridge": eval_all(y_test, pred_ridge),
        "random_forest": eval_all(y_test, pred_rf),
        "arima": eval_all(y_test, pred_arima),
        "prophet": eval_all(y_test, pred_prophet),
        "lstm": eval_all(y_test, pred_lstm),
    }

    # Outdir
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = file_path.parent if file_path.parent.exists() else Path("./outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    # Metrics file
    metrics_txt = outdir / "pr1_metrics.txt"
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write("Model\tMAE\tRMSE\tMAPE(%)\n")
        for name, (mae, rmse, mp) in metrics.items():
            mae_s = f"{mae:.4f}" if mae==mae else "nan"
            rmse_s = f"{rmse:.4f}" if rmse==rmse else "nan"
            mp_s = f"{mp:.2f}" if mp==mp else "nan"
            f.write(f"{name}\t{mae_s}\t{rmse_s}\t{mp_s}\n")

    # Predictions CSV
    pred_df = pd.DataFrame({
        "timestamp": test_df.index,
        "y_actual": y_test,
        "y_hat_naive24": pred_naive,
        "y_hat_lr": pred_lr,
        "y_hat_ridge": pred_ridge,
        "y_hat_rf": pred_rf,
        "y_hat_arima": pred_arima,
        "y_hat_prophet": pred_prophet,
        "y_hat_lstm": pred_lstm,
    }).set_index("timestamp")
    pred_csv = outdir / "pr1_predictions.csv"
    pred_df.to_csv(pred_csv, index=True)

    # Best model by RMSE (ignore NaNs)
    rmse_map = {name: vals[1] for name, vals in metrics.items()}
    best_model = min((k for k,v in rmse_map.items() if v==v), key=lambda k: rmse_map[k], default="random_forest")
    best_pred = pred_df[f"y_hat_{'naive24' if best_model=='seasonal_naive_24' else best_model if best_model!='linear_regression' else 'lr'}"].values

    # Plot
    plt.figure(figsize=(12,4))
    plt.plot(test_df.index, y_test, label="Actual")
    plt.plot(test_df.index, best_pred, label=f"Predicted ({best_model})")
    plt.title("Electricity Load — Actual vs Predicted (Test)")
    plt.xlabel("Time")
    plt.ylabel(target_col)
    plt.legend()
    plot_path = outdir / "pr1_actual_vs_pred.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print("=== PR1 Metrics ===")
    for name, (mae, rmse, mp) in metrics.items():
        mae_s = f"{mae:.4f}" if mae==mae else "nan"
        rmse_s = f"{rmse:.4f}" if rmse==rmse else "nan"
        mp_s = f"{mp:.2f}" if mp==mp else "nan"
        print(f"{name:>20s} | MAE={mae_s} | RMSE={rmse_s} | MAPE={mp_s}%")

    print(f"\nSaved metrics to: {metrics_txt}")
    print(f"Saved predictions to: {pred_csv}")
    print(f"Saved plot to: {plot_path}")
    print(f"\nDetected columns | time: '{time_col}'  target: '{target_col}'")
    print(f"Train size: {len(train_df):,} rows  | Test size: {len(test_df):,} rows")

if __name__ == "__main__":
    main()
