import joblib
import os


def save_scaler(scaler, path="data/processed/scaler.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path="data/processed/scaler.pkl"):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        raise FileNotFoundError(f"Scaler not found at {path}")
