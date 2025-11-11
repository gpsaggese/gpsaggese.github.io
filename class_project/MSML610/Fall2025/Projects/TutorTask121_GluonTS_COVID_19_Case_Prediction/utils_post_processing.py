"""Post-processing utilities for the COVID project.

Keep this file small: helpers for cleaning and summarizing dataframes used by the
notebooks. The notebooks should import functions from here rather than embedding
large logic inline.
"""
from __future__ import annotations

import pandas as pd


def summarize(df: pd.DataFrame) -> dict:
    """Return a small summary of dataframe useful for sanity checks.

    The returned dict contains row count, column count, and null counts per
    column.
    """
    return {
        "rows": len(df),
        "cols": df.shape[1] if len(df.shape) > 1 else 1,
        "null_counts": df.isnull().sum().to_dict(),
    }
