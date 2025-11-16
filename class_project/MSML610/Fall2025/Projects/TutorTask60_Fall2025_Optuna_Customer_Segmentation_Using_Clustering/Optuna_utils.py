import pandas as pd
import numpy as np

def load_online_retail_data(filepath='online_retail.xlsx'):
    df = pd.read_excel(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_data(df):
    # Remove missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Remove negative quantities and prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    print(f"After cleaning: {df.shape[0]} rows remaining")
    return df
