"""
Utility functions for data loading and preprocessing setup
for the Employee Attrition project.
"""

import os
from typing import Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

import kagglehub  # Always download from Kaggle


def load_hr_dataset() -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Load and prepare the IBM HR Analytics Employee Attrition dataset.

    Data source:
    Always downloads the dataset from Kaggle using kagglehub.
    No local CSV is required.

    Steps:
    - Download from Kaggle
    - Drop ID / constant columns
    - Map Attrition to binary target
    - Split into X, y
    - Identify categorical and numeric columns

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels (0 = No Attrition, 1 = Attrition).
    categorical_cols : list of str
        Names of categorical columns.
    numeric_cols : list of str
        Names of numeric columns.
    """

    print("Downloading IBM HR Analytics dataset from Kaggle via kagglehub...")
    path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
    csv_path = os.path.join(path, "WA_Fn-UseC_-HR-Employee-Attrition.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find expected CSV inside Kaggle dataset folder: {csv_path}"
        )

    print("Dataset path:", csv_path)
    df = pd.read_csv(csv_path)

    # Drop non-informative / constant columns
    drop_cols = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode target
    if "Attrition" not in df.columns:
        raise ValueError("Expected 'Attrition' column in dataset.")
    df["AttritionFlag"] = df["Attrition"].map({"No": 0, "Yes": 1})
    y = df["AttritionFlag"]
    X = df.drop(columns=["Attrition", "AttritionFlag"])

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    return X, y, categorical_cols, numeric_cols


def train_test_split_stratified(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Wrapper around train_test_split with stratification."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer for scaling and encoding."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor
