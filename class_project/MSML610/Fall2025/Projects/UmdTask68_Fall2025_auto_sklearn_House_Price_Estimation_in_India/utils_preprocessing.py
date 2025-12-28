import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import inspect

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

    # OneHotEncoder API changed across sklearn versions:
    # - older versions: sparse=
    # - newer versions: sparse_output=
    ohe_kwargs = {"handle_unknown": "ignore"}
    try:
        ohe_params = set(inspect.signature(OneHotEncoder).parameters.keys())
        if "sparse_output" in ohe_params:
            ohe_kwargs["sparse_output"] = False
        else:
            ohe_kwargs["sparse"] = False
    except Exception:
        # Best-effort fallback (older sklearn)
        ohe_kwargs["sparse"] = False
    
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
                ("onehot", OneHotEncoder(**ohe_kwargs))
            ]), categorical_cols),
            
            # amenities: expand into multiple binary columns
            ("amenities", Pipeline(steps=[
                ("bin", AmenitiesEncoder())
            ]), text_cols[0]),
        ],
        remainder="drop"                   # discard unhandled columns
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
    #
    # sklearn >= 1.0: ColumnTransformer.get_feature_names_out() generally works
    # sklearn 0.24.x (auto-sklearn compatible): ColumnTransformer doesn't expose
    # get_feature_names_out, and Pipelines don't expose stable feature-name APIs.
    # We therefore build names manually from each fitted sub-transformer.
    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()
    else:
        ct = preprocessor  # fitted ColumnTransformer

        # NUM: original numeric columns + missing-indicator columns (from SimpleImputer(add_indicator=True))
        num_cols = list(column_groups["numeric"])
        num_pipe = ct.named_transformers_.get("num")
        if num_pipe is None:
            num_feature_names = []
        else:
            imp = num_pipe.named_steps.get("impute")
            num_feature_names = list(num_cols)
            if getattr(imp, "add_indicator", False) and getattr(imp, "indicator_", None) is not None:
                miss_idx = getattr(imp.indicator_, "features_", [])
                miss_cols = [num_cols[i] for i in miss_idx]
                num_feature_names += [f"{c}__missing" for c in miss_cols]

        # CAT: OneHotEncoder feature names
        cat_cols = list(column_groups["categorical"])
        cat_pipe = ct.named_transformers_.get("cat")
        if cat_pipe is None or not cat_cols:
            cat_feature_names = []
        else:
            ohe = cat_pipe.named_steps.get("onehot")
            if ohe is None:
                cat_feature_names = []
            elif hasattr(ohe, "get_feature_names_out"):
                cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
            else:
                # sklearn 0.24.x
                cat_feature_names = list(ohe.get_feature_names(cat_cols))

        # AMENITIES: names from the custom transformer
        amen_pipe = ct.named_transformers_.get("amenities")
        if amen_pipe is None:
            amen_feature_names = []
        else:
            amen_enc = amen_pipe.named_steps.get("bin")
            if amen_enc is None:
                amen_feature_names = []
            elif hasattr(amen_enc, "get_feature_names_out"):
                amen_feature_names = list(amen_enc.get_feature_names_out())
            else:
                amen_feature_names = ["amenities"]

        feature_names = np.array(num_feature_names + cat_feature_names + amen_feature_names, dtype=object)

        # sanity check: names length should match transformed feature count
        if X_train_processed.shape[1] != len(feature_names):
            raise ValueError(
                f"Feature name count mismatch: X has {X_train_processed.shape[1]} columns "
                f"but feature_names has {len(feature_names)} entries."
            )
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names