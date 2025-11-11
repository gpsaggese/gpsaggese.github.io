# WIT_utils.py
# Helper functions for WIT Credit Card Fraud Detection

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path="creditcard.csv"):
    df = pd.read_csv(path)
    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y

