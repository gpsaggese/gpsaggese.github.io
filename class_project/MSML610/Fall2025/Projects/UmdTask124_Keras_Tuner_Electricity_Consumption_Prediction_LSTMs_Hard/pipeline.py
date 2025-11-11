import os
import numpy as np
from src.data_preprocessing import load_and_engineer
from src.sequence_builder import create_sequences, time_split
from src.model_baseline import build_baseline
from src.evaluate import compute_metrics, plot_predictions
from src.utils import save_scaler

import tensorflow as tf

def main():
    # ---- Step 1: Load and preprocess ----
    data_path = "data/PJM_Load_hourly.csv"
    print(f"Loading and processing data from {data_path}...")
    df, scaler = load_and_engineer(data_path)
    save_scaler(scaler)
    print(f"Data shape after preprocessing: {df.shape}")

    # ---- Step 2: Build sequences ----
    features = df[[
        'load_scaled', 'rolling_24h_mean', 'rolling_7d_mean',
        'hour_norm', 'dayofweek_norm', 'month_norm', 'is_weekend'
    ]].values
    target = df['load_scaled'].values

    X, y = create_sequences(features, target, seq_len=24, target_step=1)
    X_train, y_train, X_val, y_val, X_test, y_test = time_split(X, y)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ---- Step 3: Train baseline LSTM ----
    model = build_baseline(X_train.shape[1:])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=64,
        verbose=1
    )

    # ---- Step 4: Evaluate ----
    y_pred = model.predict(X_test)
    mae, rmse = compute_metrics(y_test, y_pred)
    print(f"\n📊 Baseline LSTM Performance: MAE={mae:.4f}, RMSE={rmse:.4f}")

    # ---- Step 5: Plot results ----
    plot_predictions(y_test, y_pred, n=500, title="Baseline LSTM Forecast")

    # ---- Step 6: Save model ----
    os.makedirs("results/models", exist_ok=True)
    model.save("results/models/baseline_lstm.h5")
    print("Model saved to results/models/baseline_lstm.h5")


if __name__ == "__main__":
    main()
