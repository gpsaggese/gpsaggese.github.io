"""
Handles dataset I/O and preprocessing for Fake News Detection.
"""

import pandas as pd
import re

def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(path)

def clean_text(text: str) -> str:
    """Basic text cleaning function."""
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()
