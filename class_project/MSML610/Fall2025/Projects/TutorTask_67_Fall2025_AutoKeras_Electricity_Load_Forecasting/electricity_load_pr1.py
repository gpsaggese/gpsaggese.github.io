#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PR1 — Electricity Load Forecasting (Simple, No AutoKeras)
--------------------------------------------------------
What this script does
- Loads an hourly (or higher frequency) electricity consumption dataset.
- Handles missing values.
- Creates time features and lag/rolling features.
- Trains simple models: LinearRegression and RandomForestRegressor.
- Adds a seasonal naive baseline (y_hat(t) = y(t-24)).
- Evaluates MAE, RMSE, and MAPE.
- Plots Actual vs Predicted (test window) and saves CSV of predictions.

How to run
$ python electricity_load_pr1.py --file "/mnt/data/continuous dataset.csv" --timestamp-col "timestamp" --target-col "target"

If you are unsure of column names, just run without args; the script will try to infer them.
$ python electricity_load_pr1.py

Outputs (saved next to your data file or ./outputs if not writable):
- pr1_metrics.txt
- pr1_predictions.csv
- pr1_actual_vs_pred.png

Dependencies
- pandas, numpy, scikit-learn, matplotlib
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import math
import warnings

warnings.filterwarnings("ignore")

# Matplotlib for plotting (no specific style/colors per project rules)
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------
# Helpers
# ----------------------------
CANDIDATE_TIME_COLS = ["timestamp","datetime","date","ds","time","hour","Date","Datetime","Timestamp"]
CANDIDATE_TARGET_COLS = ["target","load","demand","y","y_value","value","power","usage","consumption"]

def find_time_col(df: pd.DataFrame):
    # First, exact matches (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}
    for c in CANDIDATE_TIME_COLS:
        if c in cols_lower:
            return cols_lower[c]
    # Next, any column that can be parsed as datetime with low failure rate
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            frac_ok = parsed.notna().mean()
            if frac_ok > 0.9:
                return c
        except Exception:
            continue
    return None

def find_target_col(df: pd.DataFrame, time_col: str|None):
    # Prefer known names
    cols_lower = {c.lower(): c for c in df.columns}
    for c in CANDIDATE_TARGET_COLS:
        if c in cols_lower:
            return cols_lower[c]
    # Otherwise pick the first numeric column that's not the time col
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != time_col]
    if len(num_cols) == 0:
        return None
    # If there's more than one, prefer columns with larger variance (heuristic)
    variances = df[num_cols].var().sort_values(ascending=False)
    return variances.index[0]

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Avoid divide-by-zero by masking zeros
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean()) * 100.0

def train_test_split_time(df, test_size_hours: int = 7*24):
    if len(df) <= test_size_hours + 400:
        # If dataset is small, shrink test size to 20%
        test_size_hours = max(int(len(df)*0.2), 24*2)
    train = df.iloc[:-test_size_hours].copy()
    test  = df.iloc[-test_size_hours:].copy()
    return train, test

def build_features(df, target_col):
    """Create time-based, lag, and rolling features. Assumes df has DatetimeIndex."""
    out = df.copy()
    # Calendar features
    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    # Lags
    for lag in [1, 2, 24, 48, 168]:
        out[f"lag_{lag}"] = out[target_col].shift(lag)

    # Rolling means
    out["roll_mean_24"] = out[target_col].shift(1).rolling(24, min_periods=12).mean()
    out["roll_mean_168"] = out[target_col].shift(1).rolling(168, min_periods=24).mean()

    # Drop rows that became NaN due to feature creation
    out = out.dropna()
    return out

def seasonal_naive_forecast(test_df, target_col):
    # Predict y(t) = y(t-24). If unavailable, fallback to yesterday's mean.
    preds = []
    for i, idx in enumerate(test_df.index):
        prev_day_idx = idx - pd.Timedelta(hours=24)
        if prev_day_idx in test_df.index:
            preds.append(test_df.loc[prev_day_idx, target_col])
        else:
            # Fallback: previous 24 hours mean from the test start
            window = test_df[target_col].iloc[max(0, i-24):i]
            preds.append(window.mean() if len(window) else test_df[target_col].iloc[0])
    return np.asarray(preds, dtype=float)

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="",
                        help="Path to CSV/Excel file (defaults to trying known uploaded files).")
    parser.add_argument("--timestamp-col", type=str, default="",
                        help="Name of timestamp column (auto-detected if empty).")
    parser.add_argument("--target-col", type=str, default="",
                        help="Name of target column (auto-detected if empty).")
    parser.add_argument("--freq", type=str, default="H",
                        help="Resample frequency (default: H for hourly).")
    parser.add_argument("--outdir", type=str, default="",
                        help="Where to write outputs (default: alongside input file or ./outputs).")
    args = parser.parse_args()

    # 1) Locate input file
    candidate_paths = []
    if args.file:
        candidate_paths.append(Path(args.file))
    # Try common uploaded file names
    candidate_paths += [
        Path("/mnt/data/continuous dataset.csv"),
        Path("/mnt/data/weekly pre-dispatch forecast.csv"),
        Path("/mnt/data/train_dataframes.xlsx"),  # May contain a sheet with needed columns
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

    # 2) Read the data
    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        # Try first sheet that yields a timestamp/target
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
        # CSV
        df = pd.read_csv(file_path)

    # 3) Detect/confirm columns
    time_col = args.timestamp_col or find_time_col(df)
    if time_col is None:
        print(f"ERROR: Could not detect a timestamp column in {list(df.columns)}. Use --timestamp-col.", file=sys.stderr)
        sys.exit(1)
    target_col = args.target_col or find_target_col(df, time_col)
    if target_col is None:
        print(f"ERROR: Could not detect a numeric target column in {list(df.columns)}. Use --target-col.", file=sys.stderr)
        sys.exit(1)

    # 4) Parse datetime, sort, set index, resample to hourly (if needed), and impute
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", infer_datetime_format=True)
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    # Keep only time+target for simplicity
    df = df[[time_col, target_col]].copy()
    df = df.set_index(time_col).sort_index()

    # Coerce target to numeric
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    # Resample to requested frequency (hourly by default)
    df = df.resample(args.freq).mean()

    # Fill missing timestamps via resample; impute small gaps by forward/back fill then median
    df[target_col] = df[target_col].interpolate(limit_direction="both").fillna(df[target_col].median())

    # 5) Feature engineering
    feat_df = build_features(df, target_col)

    # 6) Split train/test
    train_df, test_df = train_test_split_time(feat_df, test_size_hours=7*24)

    FEATURES = [c for c in train_df.columns if c != target_col]

    X_train = train_df[FEATURES].values
    y_train = train_df[target_col].values
    X_test  = test_df[FEATURES].values
    y_test  = test_df[target_col].values

    # 7) Train simple models
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # 8) Predictions
    pred_lr = lr.predict(X_test)
    pred_ridge = ridge.predict(X_test)
    pred_rf = rf.predict(X_test)
    # Seasonal naive on original (align with test index)
    naive_24 = df[target_col].shift(24).reindex(test_df.index)
    if naive_24.isna().any():
        # fill remaining with simple backfill from available history
        naive_24 = naive_24.fillna(method="bfill").fillna(method="ffill")
    pred_naive = naive_24.values

    # ---------------------------
    # 8B) Additional Models
    # ---------------------------

import math

    # --- ARIMA ---
    try:
        from statsmodels.tsa.arima.model import ARIMA
        print("\nTraining ARIMA model...")
        arima_model = ARIMA(train_df[target_col], order=(2,1,2))
        arima_fit = arima_model.fit()
        pred_arima = arima_fit.forecast(steps=len(test_df))
    pred_arima.index = test_df.index
except Exception as e:
    print("Skipping ARIMA (error):", e)
    pred_arima = np.full(len(test_df), np.nan)

# --- Prophet ---
try:
    from prophet import Prophet
    print("\nTraining Prophet model...")
    prophet_df = train_df.reset_index()[[time_col, target_col]].rename(columns={time_col: "ds", target_col: "y"})
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=len(test_df), freq="H")
    forecast = m.predict(future)
    pred_prophet = forecast.set_index("ds")["yhat"].reindex(test_df.index)
except Exception as e:
    print("Skipping Prophet (error):", e)
    pred_prophet = np.full(len(test_df), np.nan)

# --- LSTM ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler

    print("\nTraining LSTM model...")
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(y_train.reshape(-1, 1))
    scaled_test = scaler.transform(y_test.reshape(-1, 1))

    # reshape input to [samples, timesteps, features]
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model_lstm = Sequential([
        LSTM(64, activation='relu', input_shape=(1, X_train.shape[1])),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_train_lstm, scaled_train, epochs=10, batch_size=32, verbose=0)

    pred_lstm_scaled = model_lstm.predict(X_test_lstm)
    pred_lstm = scaler.inverse_transform(pred_lstm_scaled).flatten()
except Exception as e:
    print("Skipping LSTM (error):", e)
    pred_lstm = np.full(len(test_df), np.nan)


    # 9) Evaluation
    def eval_all(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mp = mape(y_true, y_pred)
        return mae, rmse, mp

    metrics = {
        "seasonal_naive_24": eval_all(y_test, pred_naive),
        "linear_regression": eval_all(y_test, pred_lr),
        "ridge": eval_all(y_test, pred_ridge),
        "random_forest": eval_all(y_test, pred_rf),
    }

    # 10) Outputs
    # Decide outdir
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = file_path.parent if file_path.parent.exists() else Path("./outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_txt = outdir / "pr1_metrics.txt"
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write("Model\tMAE\tRMSE\tMAPE(%)\n")
        for name, (mae, rmse, mp) in metrics.items():
            f.write(f"{name}\t{mae:.4f}\t{rmse:.4f}\t{mp:.2f}\n")

    # Save predictions CSV
    pred_df = pd.DataFrame({
        "timestamp": test_df.index,
        "y_actual": y_test,
        "y_hat_naive24": pred_naive,
        "y_hat_lr": pred_lr,
        "y_hat_ridge": pred_ridge,
        "y_hat_rf": pred_rf,
    }).set_index("timestamp")
    pred_csv = outdir / "pr1_predictions.csv"
    pred_df.to_csv(pred_csv, index=True)

    # Plot actual vs best model on test (choose model with lowest RMSE)
    best_model = min(metrics.items(), key=lambda kv: kv[1][1])[0]  # by RMSE
    y_hat_best = {
        "seasonal_naive_24": pred_naive,
        "linear_regression": pred_lr,
        "ridge": pred_ridge,
        "random_forest": pred_rf,
    }[best_model]

    plt.figure(figsize=(12, 4))
    plt.plot(test_df.index, y_test, label="Actual")
    plt.plot(test_df.index, y_hat_best, label=f"Predicted ({best_model})")
    plt.title("Electricity Load — Actual vs Predicted (Test)")
    plt.xlabel("Time")
    plt.ylabel(target_col)
    plt.legend()
    plot_path = outdir / "pr1_actual_vs_pred.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    # Print a short console summary
    print("=== PR1 Metrics ===")
    for name, (mae, rmse, mp) in metrics.items():
        print(f"{name:>20s} | MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mp:.2f}%")
    print(f"\nSaved metrics to: {metrics_txt}")
    print(f"Saved predictions to: {pred_csv}")
    print(f"Saved plot to: {plot_path}")
    print(f"\nDetected columns | time: '{time_col}'  target: '{target_col}'")
    print(f"Train size: {len(train_df):,} rows  | Test size: {len(test_df):,} rows")

if __name__ == "__main__":
    main()
