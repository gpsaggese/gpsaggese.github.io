from pathlib import Path
import pandas as pd

LABEL_MAP = {0: 0, 2: 1, 4: 2}  # neg, neu, pos → 0,1,2
LABEL_STR = {0: "negative", 1: "neutral", 2: "positive"}

def load_clean_csv(path: str,
                   text_col: str = "text_clean",
                   label_col: str = "target") -> pd.DataFrame:
    """Load the cleaned CSV exported from OpenRefine."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cleaned CSV not found: {p}")
    df = pd.read_csv(p)
    # keep only rows with text & label
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[label_col] = df[label_col].astype(int).map(LABEL_MAP)
    df = df[df[label_col].isin([0, 1, 2])]
    return df

def save_csv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
