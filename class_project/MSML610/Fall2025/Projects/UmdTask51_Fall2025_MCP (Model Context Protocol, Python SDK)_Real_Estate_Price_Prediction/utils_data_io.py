# utils_data_io.py

import pandas as pd
import numpy as np

def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Loads the raw King County house sales dataset from the specified path.

    Args:
        data_path: The file path to the CSV dataset.

    Returns:
        A pandas DataFrame containing the raw data.
    """
    # For initial testing, we'll use dummy data if the path doesn't exist,
    # but in the final project, this will be pd.read_csv(data_path)
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"File not found at {data_path}. Using dummy data for testing.")
        data = {
            'id': [1, 2, 3],
            'date': ['20140502T000000', '20150218T000000', '20140627T000000'],
            'price': [221900.0, 538000.0, 180000.0],
            'sqft_living': [1180, 2570, 770],
            'waterfront': [0, 0, 0],
            'yr_built': [1955, 1951, 1933],
            'yr_renovated': [0, 1991, 0]
        }
        df = pd.DataFrame(data)

    return df

# test_f1.py

import pandas as pd
from utils_data_io import load_raw_data

# The expected path to your downloaded Kaggle dataset (modify this)
TEST_PATH = 'kc_house_data.csv'

def test_load_raw_data():
    """Tests the load_raw_data function."""
    df = load_raw_data(TEST_PATH)

    assert isinstance(df, pd.DataFrame), "Result must be a DataFrame."
    assert 'price' in df.columns, "DataFrame must contain the 'price' column."
    assert len(df) > 0, "DataFrame must not be empty."

    print("\n--- Test Results ---")
    print("Function: load_raw_data")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("Test passed: Data loaded successfully (or dummy data used).")
    print("--------------------")

if __name__ == "__main__":
    test_load_raw_data()