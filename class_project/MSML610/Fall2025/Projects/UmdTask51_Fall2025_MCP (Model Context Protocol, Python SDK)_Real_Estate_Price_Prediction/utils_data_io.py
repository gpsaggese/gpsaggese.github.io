import pandas as pd
import joblib
import os

def load_data(path: str) -> pd.DataFrame:
    """
    Loads the King County dataset from a CSV file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please download it from Kaggle.")
    
    df = pd.read_csv(path)
    print(f"Data loaded from {path}. Shape: {df.shape}")
    return df

def save_model(model, path: str):
    """
    Saves a trained model to disk using joblib.
    """
    print(f"Saving model to {path}...")
    joblib.dump(model, path)
    print("Model saved.")
