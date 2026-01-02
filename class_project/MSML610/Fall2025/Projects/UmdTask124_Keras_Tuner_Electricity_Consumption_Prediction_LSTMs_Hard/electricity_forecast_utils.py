# electricity_forecast_utils.py


from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_first_csv_in_data(data_dir: str = "data") -> pd.DataFrame:
    """Find first CSV under data/ and load it (simple helper)."""
    p = Path(data_dir)
    csvs = sorted(p.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {data_dir}/")
    df = pd.read_csv(csvs[0])
    df.columns = [c.strip() for c in df.columns]
    print("Loaded:", csvs[0])
    return df

def prepare_data_from_df(
    df: pd.DataFrame,
    timestamp_col: str,
    value_col: str,
    freq: str = "H",
    window_size: int = 24,
    horizon: int = 1,
    val_fraction: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Minimal pipeline:
    - expects df with timestamp_col & value_col
    - resamples to freq, fills forward, splits by time
    - returns X_train,y_train,X_val,y_val,scaler
    """
    df = df[[timestamp_col, value_col]].rename(columns={timestamp_col: "timestamp", value_col: "value"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    if freq:
        df = df.resample(freq).mean()
        df["value"] = df["value"].ffill()

    n = len(df)
    split = int(n * (1 - val_fraction))
    train = df.iloc[:split]["value"].values
    val = df.iloc[split:]["value"].values

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1,1)).reshape(-1)
    val_scaled = scaler.transform(val.reshape(-1,1)).reshape(-1)

    def _make_windows(arr, w, h=1):
        X, y = [], []
        max_start = len(arr) - w - (h - 1)
        for i in range(max_start):
            X.append(arr[i:i+w])
            y.append(arr[i+w+(h-1)])
        X = np.array(X).reshape((-1, w, 1))
        y = np.array(y)
        return X, y

    X_train, y_train = _make_windows(train_scaled, window_size, horizon)
    X_val, y_val = _make_windows(val_scaled, window_size, horizon)

    return X_train, y_train, X_val, y_val, scaler

def build_lstm_model(input_shape: tuple, units: int = 64, dropout: float = 0.1, lr: float = 1e-3) -> keras.Model:
    """Return compiled Keras LSTM model."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(units),
            layers.Dropout(dropout),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model

def train_model(model: keras.Model, X_train, y_train, X_val, y_val, epochs=5, batch_size=32, patience=3):
    """Train with early stopping and return history."""
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)
    return history

def invert_and_eval(y_true_scaled, y_pred_scaled, scaler):
    """Return (y_true_real, y_pred_real, mae, rmse)."""
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1,1)).reshape(-1)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    return y_true, y_pred, mae, rmse