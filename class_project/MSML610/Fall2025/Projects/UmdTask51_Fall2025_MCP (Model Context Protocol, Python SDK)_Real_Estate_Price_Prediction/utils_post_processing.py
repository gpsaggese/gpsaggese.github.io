import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values and cleans the raw data.
    """
    # Simple cleaning: drop rows with missing values
    df_clean = df.dropna()
    print(f"Data cleaned. Original rows: {len(df)}, Cleaned rows: {len(df_clean)}")
    return df_clean

def feature_engineer(df: pd.DataFrame):
    """
    Creates new features and prepares data for modeling.
    """
    print("Starting feature engineering...")
    
    # Log-transform the target variable
    df['price_log'] = np.log1p(df['price'])
    
    # Convert date to features
    df['date'] = pd.to_datetime(df['date'])
    df['yr_sold'] = df['date'].dt.year
    df['month_sold'] = df['date'].dt.month
    
    # Create 'age' of house
    df['age'] = df['yr_sold'] - df['yr_built']
    
    # Handle 'yr_renovated'
    df['renovated'] = (df['yr_renovated'] > 0).astype(int)

    # Define features and target
    features = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
        'waterfront', 'view', 'condition', 'grade', 'sqft_above',
        'sqft_basement', 'age', 'renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15'
    ]
    target = 'price_log'
    
    X = df[features]
    y = df[target]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("Feature engineering complete.")
    
    return X_train, X_test, y_train, y_test