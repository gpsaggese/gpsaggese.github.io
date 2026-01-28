"""
Load COVID-19 Data Utilities

Load COVID-19 data from various sources: cases, deaths, vaccines, and mobility reports.

Import as:

import tutorials.tutorial_GluonTS_COVID19_Prediction.GluonTS_utils_data_io as ttgcpgudi
"""

import pandas as pd
from pathlib import Path
from typing import Dict


def load_jhu_cases(data_dir: str = "data") -> pd.DataFrame:
    """Load JHU CSSE COVID-19 cases data from CSV."""
    filepath = Path(data_dir) / "cases.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"cases.csv not found in {data_dir}")

    print(f"Loading cases data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_jhu_deaths(data_dir: str = "data") -> pd.DataFrame:
    """Load JHU CSSE COVID-19 deaths data from CSV."""
    filepath = Path(data_dir) / "deaths.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"deaths.csv not found in {data_dir}")

    print(f"Loading deaths data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_jhu_vaccines(data_dir: str = "data") -> pd.DataFrame:
    """Load JHU CSSE COVID-19 vaccine data from CSV."""
    filepath = Path(data_dir) / "vaccine.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"vaccine.csv not found in {data_dir}")

    print(f"Loading vaccine data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_google_mobility(data_dir: str = "data") -> pd.DataFrame:
    """Load Google COVID-19 Community Mobility Reports from CSV."""
    filepath = Path(data_dir) / "mobility.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"mobility.csv not found in {data_dir}")

    print(f"Loading mobility data from {filepath}")
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    return df


def load_all_data(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """Load all COVID-19 datasets at once."""
    print("Loading all COVID-19 datasets...")
    print("=" * 60)

    data = {}
    data["cases"] = load_jhu_cases(data_dir)
    data["deaths"] = load_jhu_deaths(data_dir)
    data["vaccines"] = load_jhu_vaccines(data_dir)
    data["mobility"] = load_google_mobility(data_dir)

    print("=" * 60)
    print("All datasets loaded successfully")
    return data


def verify_data_exists(data_dir: str = "data") -> bool:
    """Verify that all required data files exist."""
    required_files = ["cases.csv", "deaths.csv", "vaccine.csv", "mobility.csv"]
    data_path = Path(data_dir)

    missing_files = []
    for filename in required_files:
        if not (data_path / filename).exists():
            missing_files.append(filename)

    if missing_files:
        print(f"Missing files: {', '.join(missing_files)}")
        print(f"Expected location: {data_path.absolute()}")
        return False

    print(f"All required data files present in {data_dir}")
    return True

# #############################################################################
# DataLoader
# #############################################################################

class DataLoader:
    """Convenience class for loading COVID-19 data."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def load_cases(self) -> pd.DataFrame:
        return load_jhu_cases(self.data_dir)

    def load_deaths(self) -> pd.DataFrame:
        return load_jhu_deaths(self.data_dir)

    def load_vaccines(self) -> pd.DataFrame:
        return load_jhu_vaccines(self.data_dir)

    def load_mobility(self) -> pd.DataFrame:
        return load_google_mobility(self.data_dir)

    def load_all(self) -> Dict[str, pd.DataFrame]:
        return load_all_data(self.data_dir)
