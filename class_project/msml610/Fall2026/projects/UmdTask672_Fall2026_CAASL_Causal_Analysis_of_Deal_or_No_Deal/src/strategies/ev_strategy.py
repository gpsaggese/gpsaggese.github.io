
"""Simple expected-value benchmark for Deal or No Deal."""

import pandas as pd


def apply_ev_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recommend accepting the banker's offer when it is
    greater than or equal to the remaining board's EV.

    Returns a copy of the dataset with strategy recommendations.
    """
    required_columns = {"bank_offer", "board_ev"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[["bank_offer", "board_ev"]].isna().any().any():
        raise ValueError("Bank offers and board EV must not be missing.")

    results = df.copy()

    results["ev_strategy"] = (
        results["bank_offer"] >= results["board_ev"]
    ).astype(int)

    return results
