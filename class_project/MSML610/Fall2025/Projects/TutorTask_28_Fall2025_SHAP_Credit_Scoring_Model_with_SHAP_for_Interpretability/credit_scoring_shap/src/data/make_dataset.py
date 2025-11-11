"""
Create interim train/test splits from the raw German Credit dataset.

Expected raw CSV: data/raw/german_credit_data.csv
Columns vary by source; adjust mapping as needed.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import Paths
from src.utils.io import ensure_dirs
from src.utils.seed import set_seed

TARGET_COL = "target"  # TODO: align with your CSV's target column name

def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # TODO: Map/rename columns here if your CSV uses different names.
    # Example (commented):
    # df = df.rename(columns={'Creditability':'target'})
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Minimal cleaning: drop obvious duplicates, strip strings.
    df = df.drop_duplicates().copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
    return df

def main():
    set_seed(42)
    P = Paths()
    ensure_dirs(P.data_dir / "raw", P.interim_dir, P.processed_dir)
    if not P.data_raw.exists():
        raise FileNotFoundError(f"Raw dataset not found at {P.data_raw}. Place the CSV and retry.")

    df = load_raw(P.data_raw)
    df = basic_clean(df)
    if TARGET_COL not in df.columns:
        raise KeyError(f"Expected target column '{TARGET_COL}' not found. Update TARGET_COL in make_dataset.py.")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET_COL])
    (P.interim_dir / 'split_train.parquet').parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(P.interim_dir / 'split_train.parquet', index=False)
    test_df.to_parquet(P.interim_dir / 'split_test.parquet', index=False)
    print("Saved interim splits to:", P.interim_dir)

if __name__ == "__main__":
    main()
