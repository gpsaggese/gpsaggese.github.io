"""
Load COVID-19 and mobility data from CSV files
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import re


class DataLoader:
    """Load COVID-19 and mobility data from Google Drive CSVs.

    This class downloads CSVs from Google Drive (by shareable link or
    direct file id) into a local `data/` directory and loads them into
    pandas DataFrames for later preprocessing and modeling.
    """

    def __init__(
        self,
        gdrive_urls: Optional[Dict[str, str]] = None,
        data_dir: str = "data",
    ) -> None:
        """
        Args:
            gdrive_urls: mapping name -> Google Drive URL or file-id-based URL.
                Example keys: 'cases', 'deaths', 'vaccine', 'mobility'.
            data_dir: local directory where CSVs will be stored.
        """
        self.gdrive_urls = gdrive_urls or {}
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # name -> pandas.DataFrame
        self.dataframes: Dict[str, pd.DataFrame] = {}

    @staticmethod
    def _normalize_gdrive_url(url: str) -> str:
        """Return a gdown-friendly Google Drive URL (uses `uc?id=` form).

        Accepts either full share link or `open?id=` form. If a raw file id
        is provided, this will convert it to the `uc?id=` format.
        """
        # If the user provided a plain file id (alphanumeric id only)
        if re.match(r"^[A-Za-z0-9_-]{10,}$", url):
            return f"https://drive.google.com/uc?id={url}"
        # If already in open?id= or uc?id= form, try to extract id
        m = re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)
        if m:
            return f"https://drive.google.com/uc?id={m.group(1)}"
        # If it looks like a share link with /file/d/ID/
        m2 = re.search(r"/d/([A-Za-z0-9_-]+)/", url)
        if m2:
            return f"https://drive.google.com/uc?id={m2.group(1)}"
        # Fallback: return the original URL and let gdown decide
        return url

    def download_all(self, overwrite: bool = False) -> None:
        """Download all configured Google Drive CSVs into the data dir.

        Args:
            overwrite: if True, re-download even if local file exists.
        """
        try:
            import gdown
        except Exception as e:
            raise ImportError("gdown is required. Install with 'pip install gdown'.") from e

        for name, url in self.gdrive_urls.items():
            local_path = self.data_dir / f"{name}.csv"
            if local_path.exists() and not overwrite:
                print(f"Skipping download for '{name}' - exists at {local_path}")
                continue
            friendly_url = self._normalize_gdrive_url(url)
            print(f"Downloading '{name}' from {friendly_url} -> {local_path}")
            # gdown can raise if file is not found or permission denied
            gdown.download(friendly_url, str(local_path), quiet=False)

    def load_all(self) -> None:
        """Load all CSVs present in the data directory into DataFrames.

        Files are expected to be named <name>.csv where <name> is a key in
        the provided gdrive_urls mapping (or any CSV in the dir will be
        loaded with its stem as key).
        """
        for csv_path in sorted(self.data_dir.glob("*.csv")):
            name = csv_path.stem
            print(f"Loading '{name}' from {csv_path}")
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} rows for '{name}'")
            self.dataframes[name] = df

    def get_dataframe(self, name: str) -> pd.DataFrame:
        """Return the DataFrame for the given name.

        Raises KeyError if the name is not loaded.
        """
        if name not in self.dataframes:
            raise KeyError(f"DataFrame '{name}' not loaded. Call load_all() first.")
        return self.dataframes[name]
    
    def load_cases(self):
        """Load JHU CSSE confirmed cases data"""
        filepath = self.data_dir / 'cases.csv'
        print(f"Loading cases data from {filepath}")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows")
        return df
    
    def load_deaths(self):
        """Load JHU CSSE deaths data"""
        filepath = self.data_dir / 'deaths.csv'
        print(f"Loading deaths data from {filepath}")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows")
        return df
    
    def load_vaccines(self):
        """Load JHU CSSE vaccine data"""
        filepath = self.data_dir / 'vaccine.csv'
        print(f"Loading vaccine data from {filepath}")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows")
        return df
    
    def load_mobility(self):
        """Load Google Mobility data from ActiveConclusion"""
        filepath = self.data_dir / 'mobility.csv'
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
    vaccines_df = loader.load_vaccines()
    
    print("\n" + "="*60)
    print("DATA LOADED SUCCESSFULLY")
    print("="*60)

