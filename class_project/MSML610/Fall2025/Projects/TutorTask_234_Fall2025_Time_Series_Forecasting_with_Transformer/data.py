import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================================================
# LOAD AMZN CSV FILE
# ============================================================
def load_ticker_csv(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").set_index("Date")
    return df


# ============================================================
# CREATE 5-STEP FUTURE TARGETS FROM CLOSE PRICE
# ============================================================
def create_price_targets(df, horizon=5):
    """
    horizon = number of future steps (e.g., next 5 days)
    Returns: y shape = (num_samples, horizon)
    """

    close = df["Close"].values
    y = []

    for i in range(len(close) - horizon):
        y.append(close[i+1 : i+1+horizon])

    return np.array(y)


# ============================================================
# CREATE INPUT WINDOWS (X) USING SLIDING WINDOW
# ============================================================
def create_windows(X, y, seq_len):
    """
    X: scaled features array
    y: scaled targets array
    seq_len: number of lookback timesteps (e.g., 60)
    """
    X_out, y_out = [], []
    for i in range(len(X) - seq_len):
        X_out.append(X[i:i+seq_len])
        y_out.append(y[i+seq_len-1])  # match alignment

    return np.array(X_out), np.array(y_out)


# ============================================================
# FULL DATA PREPARATION PIPELINE FOR AMZN ONLY
# ============================================================
def prepare_dataset(path, seq_len=60, horizon=5):
    print("🔹 Loading AMZN data...")
    df = load_ticker_csv(path)

    print("🔹 Creating future targets...")
    y_raw = create_price_targets(df, horizon)

    print("🔹 Aligning features with targets...")
    X_raw = df.iloc[:-horizon].values

    print("🔹 Scaling features and targets...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Scale y using CLOSE column's mean/std
    close = df["Close"].values
    close_mean = close.mean()
    close_std = close.std()
    y_scaled = (y_raw - close_mean) / close_std

    print("🔹 Building sliding windows...")
    X, y = create_windows(X_scaled, y_scaled, seq_len)

    print("🔹 Splitting dataset (70/10/20)...")
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print("✅ AMZN dataset ready!")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler
