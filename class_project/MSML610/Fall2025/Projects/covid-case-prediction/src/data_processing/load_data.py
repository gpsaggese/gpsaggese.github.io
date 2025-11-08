"""
Load COVID-19 and mobility data from CSV files
"""

import pandas as pd
from pathlib import Path


class DataLoader:
    """Load COVID-19 case, death, vaccine, and mobility data"""
    
    def __init__(self, data_dir='data'):
        """
        Initialize data loader
        
        Args:
            data_dir (str): Path to data directory
        """
        self.data_dir = Path(data_dir)
    
    def load_cases(self):
        """Load JHU CSSE confirmed cases data"""
        filepath = self.data_dir / 'time_series_covid19_confirmed_US.csv'
        print(f"Loading cases data from {filepath}")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows")
        return df
    
    def load_deaths(self):
        """Load JHU CSSE deaths data"""
        filepath = self.data_dir / 'time_series_covid19_deaths_US.csv'
        print(f"Loading deaths data from {filepath}")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows")
        return df
    
    def load_vaccines(self):
        """Load JHU CSSE vaccine data"""
        filepath = self.data_dir / 'time_series_covid19_vaccine_us.csv'
        print(f"Loading vaccine data from {filepath}")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows")
        return df
    
    def load_mobility(self):
        """Load Google Mobility data from ActiveConclusion"""
        filepath = self.data_dir / 'mobility_report_US.csv'
        print(f"Loading mobility data from {filepath}")
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded {len(df)} rows")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        return df


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Load all datasets
    cases_df = loader.load_cases()
    deaths_df = loader.load_deaths()
    mobility_df = loader.load_mobility()
    
    print("\n" + "="*60)
    print("DATA LOADED SUCCESSFULLY")
    print("="*60)

