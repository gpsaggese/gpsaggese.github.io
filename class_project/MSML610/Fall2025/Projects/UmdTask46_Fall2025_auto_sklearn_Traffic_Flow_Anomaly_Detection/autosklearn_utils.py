#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_and_prep_data(csv_path):
    """
    Loads the Metro Traffic data, cleans it, and engineers features.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        print("Please make sure 'Metro_Interstate_Traffic_Volume.csv' is in the same folder.")
        return None, None, None, None

    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # Engineer time-based features
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    
    # Drop original datetime and redundant columns
    df = df.drop(['date_time', 'weather_description'], axis=1)
    
    # Separate features and target
    X = df.drop('traffic_volume', axis=1)
    y = df['traffic_volume']
    
    # One-hot encode categorical features
    # Use pandas get_dummies for simplicity
    X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns, drop_first=True)
    
    # Split the data (don't shuffle time series data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Align columns in case test set is missing categories from train set
    train_cols = X_train.columns
    test_cols = X_test.columns
    
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test[c] = 0
        
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        X_train[c] = 0
        
    X_test = X_test[train_cols]
    
    return X_train, X_test, y_train, y_test

def define_anomalies_from_residuals(y_true, y_pred, threshold_std=3):
    """
    Defines anomalies as points where the prediction error (residual)
    is more than 'threshold_std' standard deviations from the mean error.
    
    Returns a boolean Series (True == Anomaly)
    """
    residuals = y_true - y_pred
    mean_error = residuals.mean()
    std_error = residuals.std()
    
    upper_bound = mean_error + (std_error * threshold_std)
    lower_bound = mean_error - (std_error * threshold_std)
    
    anomalies = (residuals > upper_bound) | (residuals < lower_bound)
    return anomalies

