from typing import Dict
import pandas as pd


def get_column_groups(X: pd.DataFrame) -> Dict[str, list]:
    """
    categorize columns into preprocessing groups
    
    args:
        X: feature DataFrame
        
    returns:
        dictionary with keys: 'numeric', 'binary', 'categorical', 'text'
    """
    numeric_cols = [
        "BHK", "Size_in_SqFt", "Price_per_SqFt", "Year_Built",
        "Floor_No", "Total_Floors", "Age_of_Property",
        "Nearby_Schools", "Nearby_Hospitals"
    ]
    binary_cols = ["Parking_Space", "Security"]
    text_cols = ["Amenities"]
    categorical_cols = list(
        set(X.columns) - set(numeric_cols) - set(binary_cols) - set(text_cols)
    )
    
    return {
        "numeric": numeric_cols,
        "binary": binary_cols,
        "categorical": categorical_cols,
        "text": text_cols
    }

def encode_binary_columns(X: pd.DataFrame, binary_cols: list) -> pd.DataFrame:
    """
    map yes/no binary columns to numeric 1/0.
    
    args:
        X: feature DataFrame
        binary_cols: List of binary column names
        
    returns:
        dataFrame with binary columns encoded as 1/0
    """
    X_copy = X.copy()
    for col in binary_cols:
        X_copy[col] = X_copy[col].map({"Yes": 1, "No": 0})
    return X_copy
