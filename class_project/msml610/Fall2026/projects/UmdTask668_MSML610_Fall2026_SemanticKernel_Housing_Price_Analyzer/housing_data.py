
"""Dataset loading, validation, and preprocessing for Housing Price Analyzer."""

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

DATASET_ID = 477


def check_missing_values(X: pd.DataFrame, y: pd.Series) -> dict:
    """Report missing values in features and target."""
    return {
        "feature_missing": X.isna().sum().to_dict(),
        "target_missing": int(y.isna().sum()),
        "total_feature_missing": int(X.isna().sum().sum()),
    }


def check_duplicates(X: pd.DataFrame, y: pd.Series) -> dict:
    """Report duplicate feature rows and complete duplicate records."""
    combined = pd.concat(
        [X.reset_index(drop=True), y.reset_index(drop=True)],
        axis=1,
    )

    return {
        "duplicate_feature_rows": int(X.duplicated().sum()),
        "duplicate_records": int(combined.duplicated().sum()),
    }


def validate_housing_data(
    X: pd.DataFrame,
    y: pd.Series,
    required_features=None,
) -> dict:
    """
    Validate housing features and target.

    Raises:
        TypeError: If inputs have incorrect types.
        ValueError: If data is empty, invalid, or inconsistent.

    Returns:
        Dictionary containing validation results.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")

    if X.empty or y.empty:
        raise ValueError("Housing features and target cannot be empty.")

    if len(X) != len(y):
        raise ValueError("Feature and target row counts do not match.")

    if not X.columns.is_unique:
        raise ValueError("Feature column names must be unique.")

    if not X.index.equals(y.index):
        raise ValueError("Feature and target indexes must match.")

    if required_features is not None:
        missing_columns = set(required_features) - set(X.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required features: {sorted(missing_columns)}"
            )

    if not all(pd.api.types.is_numeric_dtype(dtype)
               for dtype in X.dtypes):
        raise ValueError("All feature columns must be numeric.")

    if not pd.api.types.is_numeric_dtype(y.dtype):
        raise ValueError("Target must be numeric.")

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Missing values detected in housing data.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Features contain infinite or invalid values.")

    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("Target contains infinite or invalid values.")

    return {
        "valid": True,
        "rows": len(X),
        "features": X.shape[1],
        "target": y.name,
        "missing_values": check_missing_values(X, y),
        "duplicates": check_duplicates(X, y),
    }


def preprocess_housing_data(
    X: pd.DataFrame,
    y: pd.Series,
    missing_strategy: str = "error",
    duplicate_strategy: str = "keep",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare housing data using explicit missing-value and
    duplicate-handling strategies.

    missing_strategy: 'error' or 'drop'
    duplicate_strategy: 'keep' or 'drop'
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")

    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("Feature and target rows must align.")

    if missing_strategy not in {"error", "drop"}:
        raise ValueError("Invalid missing_strategy.")

    if duplicate_strategy not in {"keep", "drop"}:
        raise ValueError("Invalid duplicate_strategy.")

    # Validate structure and numeric values, allowing missing values
    # temporarily when the explicit drop strategy is selected.
    X = X.copy()
    y = y.copy()

    if X.empty or y.empty:
        raise ValueError("Housing data cannot be empty.")

    if not X.columns.is_unique:
        raise ValueError("Feature column names must be unique.")

    if not all(pd.api.types.is_numeric_dtype(dtype)
               for dtype in X.dtypes):
        raise ValueError("All feature columns must be numeric.")

    if not pd.api.types.is_numeric_dtype(y.dtype):
        raise ValueError("Target must be numeric.")

    combined = pd.concat(
        [X.reset_index(drop=True), y.reset_index(drop=True)],
        axis=1,
    )

    if not np.isfinite(
        X.dropna().to_numpy(dtype=float)
    ).all() or not np.isfinite(
        y.dropna().to_numpy(dtype=float)
    ).all():
        raise ValueError("Data contains infinite values.")

    if missing_strategy == "error":
        validate_housing_data(X, y)
    else:
        valid_rows = combined.notna().all(axis=1)
        X = X.loc[valid_rows.to_numpy()].copy()
        y = y.loc[valid_rows.to_numpy()].copy()

    if duplicate_strategy == "drop":
        combined = pd.concat(
            [X.reset_index(drop=True), y.reset_index(drop=True)],
            axis=1,
        )
        keep = ~combined.duplicated()
        X = X.iloc[np.flatnonzero(keep.to_numpy())].copy()
        y = y.iloc[np.flatnonzero(keep.to_numpy())].copy()

    validate_housing_data(X, y)

    return X, y


def load_housing_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the UCI Real Estate Valuation dataset."""
    dataset = fetch_ucirepo(id=DATASET_ID)

    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()

    # Remove transaction identifier if present.
    X = X.drop(columns=["No"], errors="ignore")

    validate_housing_data(X, y)

    return X, y


if __name__ == "__main__":
    features, target = load_housing_data()

    report = validate_housing_data(features, target)

    print("Dataset loaded and validated successfully!")
    print(f"Rows: {report['rows']}")
    print(f"Features: {report['features']}")
    print(f"Target: {report['target']}")
    print(f"Missing values: {report['missing_values']}")
    print(f"Duplicates: {report['duplicates']}")
