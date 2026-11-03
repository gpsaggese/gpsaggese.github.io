
"""Dataset loading and validation for the Housing Price Analyzer."""

import pandas as pd
from ucimlrepo import fetch_ucirepo

DATASET_ID = 477


def load_housing_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the UCI Real Estate Valuation dataset."""
    dataset = fetch_ucirepo(id=DATASET_ID)

    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()

    # Remove the transaction identifier if present.
    X = X.drop(columns=["No"], errors="ignore")

    if X.empty or y.empty:
        raise ValueError("The housing dataset is empty.")

    if len(X) != len(y):
        raise ValueError("Feature and target row counts do not match.")

    return X, y


if __name__ == "__main__":
    features, target = load_housing_data()

    print("Dataset loaded successfully!")
    print(f"Rows: {len(features)}")
    print(f"Features: {features.shape[1]}")
    print(f"Target: {target.name}")
    print("\nMissing values:")
    print(features.isna().sum())
    print(f"Target missing values: {target.isna().sum()}")
