import pandas as pd


# data loading and preparation functions
def load_housing_data(data_path: str) -> pd.DataFrame:
    """
    load the India housing prices dataset
    
    args:
        data_path: path to the CSV file containing housing data
        
    returns:
        DataFrame with raw housing data + blank strings converted to NaN
    """
    df = pd.read_csv(data_path)
    # harmonize empty strings to NaN so they can be detected consistently
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    return df