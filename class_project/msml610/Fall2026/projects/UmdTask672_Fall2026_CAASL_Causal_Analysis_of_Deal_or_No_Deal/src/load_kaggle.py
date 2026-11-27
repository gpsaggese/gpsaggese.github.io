from pathlib import Path
import pandas as pd

def load_kaggle(path="data/raw/kaggle/dond_game_data.csv"):
    """Load the Jared Stock/Kaggle Deal or No Deal dataset."""
    path = Path(path)
    df = pd.read_csv(path)

    # Standardize names without changing the underlying data.
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df

if __name__ == "__main__":
    df = load_kaggle()
    print(df.head())
    print("\nShape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
