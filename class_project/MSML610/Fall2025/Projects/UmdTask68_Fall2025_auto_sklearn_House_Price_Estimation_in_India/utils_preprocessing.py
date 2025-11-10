import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from utils_transformers import AmenitiesEncoder
from utils_data_io import load_housing_data
from utils_feature_engineering import get_column_groups, encode_binary_columns


def create_preprocessor(column_groups: Dict[str, list]) -> ColumnTransformer:
    """
    create a feature engineering preprocessing pipeline
    
    handles:
    - numeric columns: imputation with median + standardization
    - categorical columns: imputation + one-hot encoding
    - text columns (amenities): custom binary encoding
    
    args:
        column_groups: Dictionary with column groups from get_column_groups()
        
    returns:
        ColumnTransformer pipeline for preprocessing
    """
    numeric_cols = column_groups["numeric"]
    categorical_cols = column_groups["categorical"]
    text_cols = column_groups["text"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            # numeric: impute missing with median and standardize
            ("num", Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler())
            ]), numeric_cols),
            
            # categorical: impute with most frequent and one-hot encode
            ("cat", Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_cols),
            
            # amenities: expand into multiple binary columns
            ("amenities", Pipeline(steps=[
                ("bin", AmenitiesEncoder())
            ]), text_cols[0]),
        ],
        remainder="drop",                   # discard unhandled columns
        verbose_feature_names_out=False,    # keep feature names clean
    )
    
    return preprocessor

def prepare_data(
    data_path: str,
    target_col: str = "Price_in_Lakhs",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, ColumnTransformer, np.ndarray]:
    """
    load and prepare data for modeling: load, split, encode, and preprocess
    
    this is the main function to call for data preparation
    
    args:
        data_path: path to the CSV file
        target_col: name of the target column (default: "Price_in_Lakhs")
        test_size: proportion of data for testing (default: 0.2)
        random_state: random seed for reproducibility (default: 42)
        
    returns:
        tuple of:
        - X_train_processed: Preprocessed training features (numpy array)
        - X_test_processed: Preprocessed test features (numpy array)
        - y_train: Training target values (pandas Series)
        - y_test: Test target values (pandas Series)
        - preprocessor: Fitted ColumnTransformer for making predictions on new data
        - feature_names: Array of engineered feature names
    """
    # load data
    df = load_housing_data(data_path)
    
    # esparate features and target (drop ID as it's a surrogate key)
    drop_cols = ["ID", target_col]
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    # get column groups for preprocessing
    column_groups = get_column_groups(X)
    
    # encode binary columns (Yes/No -> 1/0)
    X = encode_binary_columns(X, column_groups["binary"])
    
    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # create preprocessor and transform data
    preprocessor = create_preprocessor(column_groups)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # get feature names for interpretability
    feature_names = preprocessor.get_feature_names_out()
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names