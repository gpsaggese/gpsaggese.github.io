from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DataConfig
from ucimlrepo import fetch_ucirepo
from ucimlrepo import fetch_ucirepo
import pandas as pd

def load_raw_data(cfg: DataConfig) -> pd.DataFrame:
    """
    Load the German Credit Data from the UCI Machine Learning Repository
    via the ucimlrepo package (dataset ID = 144).

    Returns:
        DataFrame with meaningful feature names + target named cfg.target_col.
    """
    print("Fetching German Credit Data from UCI ML Repository...")

    dataset = fetch_ucirepo(id=144)  # Statlog German Credit Data

    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    rename_map = {
        "Attribute1":  "status_checking_account",
        "Attribute2":  "duration_months",
        "Attribute3":  "credit_history",
        "Attribute4":  "purpose",
        "Attribute5":  "credit_amount",
        "Attribute6":  "savings_account",
        "Attribute7":  "employment_since",
        "Attribute8":  "installment_rate",
        "Attribute9":  "personal_status_sex",
        "Attribute10": "guarantors",
        "Attribute11": "residence_since",
        "Attribute12": "property",
        "Attribute13": "age_years",
        "Attribute14": "other_installment_plans",
        "Attribute15": "housing",
        "Attribute16": "existing_credits",
        "Attribute17": "job",
        "Attribute18": "num_dependents",
        "Attribute19": "telephone",
        "Attribute20": "foreign_worker",
    }
    df = df.rename(columns=rename_map)

    # Standardize target column name (UCI uses "class" or similar)
    df = df.rename(columns={df.columns[-1]: cfg.target_col})

    # UCI labels: 1 = good, 2 = bad → 0 = good, 1 = bad
    df[cfg.target_col] = df[cfg.target_col].map({1: 0, 2: 1})


    return df

def _prepare_features_and_target(df: pd.DataFrame, target_col: str):
    """
    Split features and target.

    Assumptions:
    - Target has already been semantically encoded upstream as:
        0 = Good
        1 = Bad (positive class)

    This function does NOT perform label remapping.
    It only validates and enforces numeric correctness.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    df = df.copy()
    df = df.drop_duplicates()

    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    # Enforce numeric target with correct semantics
    if not np.issubdtype(y_raw.dtype, np.number):
        raise ValueError(
            "Target column must be numeric at this stage. "
            "Expected encoding: 0 = Good, 1 = Bad."
        )

    # Enforce binary values
    unique_vals = set(y_raw.unique())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"Unexpected target values {unique_vals}. "
            "Expected only {0, 1} where 1 = Bad."
        )

    y = y_raw.astype(int)

    return X, y



def _build_preprocessor(
    X: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Infer numeric vs categorical columns and build a ColumnTransformer
    with scaling + one-hot encoding.
    """
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if X[c].dtype != "object"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    return preprocessor, num_cols, cat_cols


def load_and_preprocess(
    cfg: DataConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer, List[str]]:
    """
    Load the CSV, clean, encode, and split into train/test.

    Returns:
        X_train_df, X_test_df, y_train, y_test, preprocessor, feature_names
    """
    df = load_raw_data(cfg)
    X, y = _prepare_features_and_target(df, cfg.target_col)

    # Handle missing values, numeric to median, categorical to 'Unknown'
    num_cols = [c for c in X.columns if X[c].dtype != "object"]
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]

    if num_cols:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for col in cat_cols:
        X[col] = X[col].fillna("Unknown")

    preprocessor, num_cols, cat_cols = _build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    # Fit on train, transform both
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    feature_names: List[str] = []
    if num_cols:
        feature_names.extend(num_cols)
    if cat_cols:
        ohe: OneHotEncoder = preprocessor.named_transformers_["cat"]
        ohe_feature_names = list(ohe.get_feature_names_out(cat_cols))
        feature_names.extend(ohe_feature_names)

    X_train_df = pd.DataFrame(
        X_train_arr, columns=feature_names, index=X_train.index
    )
    X_test_df = pd.DataFrame(
        X_test_arr, columns=feature_names, index=X_test.index
    )

    return X_train_df, X_test_df, y_train, y_test, preprocessor, feature_names
