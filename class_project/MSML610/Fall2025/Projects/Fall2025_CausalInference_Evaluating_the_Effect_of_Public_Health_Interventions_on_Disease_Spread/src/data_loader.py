import pandas as pd
import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except:
        return {
            'dataset': {
                'url': "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
            }
        }

def download_owid_data():
    """Download OWID COVID dataset"""
    url = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
    raw_csv = "owid_covid_data.csv"
    
    print("Downloading dataset from OWID compact link...")
    df = pd.read_csv(url, parse_dates=['date'], low_memory=False)
    
    # Save raw data
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"data/raw/{raw_csv}", index=False)
    
    print("Download successful!")
    print(f"Rows: {df.shape[0]} Columns: {df.shape[1]}")
    print(f"Saved local copy: data/raw/{raw_csv}")
    
    return df