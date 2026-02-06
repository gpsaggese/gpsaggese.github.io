"""
Load COVID-19 Data Utilities.

Load COVID-19 data from various sources: cases, deaths, vaccines, and mobility reports.

Import as:

import tutorials.tutorial_GluonTS_COVID19_Prediction.GluonTS_utils_data_io as ttgcpgudi
"""

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

_LOG = logging.getLogger(__name__)


def load_jhu_cases(
    *,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Load JHU CSSE COVID-19 cases data from CSV.

    :param data_dir: Directory containing data files
    :return: DataFrame with cases data
    """
    filepath = Path(data_dir) / "cases.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"cases.csv not found in {data_dir}")
    _LOG.info("Loading cases data from %s", filepath)
    df = pd.read_csv(filepath)
    _LOG.info("Loaded %s rows, %s columns", len(df), len(df.columns))
    return df


def load_jhu_deaths(
    *,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Load JHU CSSE COVID-19 deaths data from CSV.

    :param data_dir: Directory containing data files
    :return: DataFrame with deaths data
    """
    filepath = Path(data_dir) / "deaths.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"deaths.csv not found in {data_dir}")
    _LOG.info("Loading deaths data from %s", filepath)
    df = pd.read_csv(filepath)
    _LOG.info("Loaded %s rows, %s columns", len(df), len(df.columns))
    return df


def load_jhu_vaccines(
    *,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Load JHU CSSE COVID-19 vaccine data from CSV.

    :param data_dir: Directory containing data files
    :return: DataFrame with vaccine data
    """
    filepath = Path(data_dir) / "vaccine.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"vaccine.csv not found in {data_dir}")
    _LOG.info("Loading vaccine data from %s", filepath)
    df = pd.read_csv(filepath)
    _LOG.info("Loaded %s rows, %s columns", len(df), len(df.columns))
    return df


def load_google_mobility(
    *,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Load Google COVID-19 Community Mobility Reports from CSV.

    :param data_dir: Directory containing data files
    :return: DataFrame with mobility data
    """
    filepath = Path(data_dir) / "mobility.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"mobility.csv not found in {data_dir}")
    _LOG.info("Loading mobility data from %s", filepath)
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    _LOG.info("Loaded %s rows, %s columns", len(df), len(df.columns))
    _LOG.info("  Date range: %s to %s", df["date"].min(), df["date"].max())
    return df


def load_all_data(
    *,
    data_dir: str = "data",
) -> Dict[str, pd.DataFrame]:
    """
    Load all COVID-19 datasets at once.

    :param data_dir: Directory containing data files
    :return: Dictionary with keys 'cases', 'deaths', 'vaccines', 'mobility'
    """
    _LOG.info("Loading all COVID-19 datasets...")
    _LOG.info("=" * 60)
    data = {}
    data["cases"] = load_jhu_cases(data_dir=data_dir)
    data["deaths"] = load_jhu_deaths(data_dir=data_dir)
    data["vaccines"] = load_jhu_vaccines(data_dir=data_dir)
    data["mobility"] = load_google_mobility(data_dir=data_dir)
    _LOG.info("=" * 60)
    _LOG.info("All datasets loaded successfully")
    return data


def verify_data_exists(
    *,
    data_dir: str = "data",
) -> bool:
    """
    Verify that all required data files exist.

    :param data_dir: Directory containing data files
    :return: True if all files exist, False otherwise
    """
    required_files = ["cases.csv", "deaths.csv", "vaccine.csv", "mobility.csv"]
    data_path = Path(data_dir)
    missing_files = []
    for filename in required_files:
        if not (data_path / filename).exists():
            missing_files.append(filename)
    if missing_files:
        _LOG.info("Missing files: %s", ", ".join(missing_files))
        _LOG.info("Expected location: %s", data_path.absolute())
        return False
    _LOG.info("All required data files present in %s", data_dir)
    return True

# #############################################################################
# DataLoader
# #############################################################################

class DataLoader:
    """Convenience class for loading COVID-19 data."""

    def __init__(
        self,
        *,
        data_dir: str = "data",
    ):
        """
        Initialize DataLoader.

        :param data_dir: Directory containing data files
        """
        self.data_dir = data_dir

    def load_cases(self) -> pd.DataFrame:
        """
        Load cases data.

        :return: DataFrame with cases data
        """
        return load_jhu_cases(data_dir=self.data_dir)

    def load_deaths(self) -> pd.DataFrame:
        """
        Load deaths data.

        :return: DataFrame with deaths data
        """
        return load_jhu_deaths(data_dir=self.data_dir)

    def load_vaccines(self) -> pd.DataFrame:
        """
        Load vaccines data.

        :return: DataFrame with vaccine data
        """
        return load_jhu_vaccines(data_dir=self.data_dir)

    def load_mobility(self) -> pd.DataFrame:
        """
        Load mobility data.

        :return: DataFrame with mobility data
        """
        return load_google_mobility(data_dir=self.data_dir)

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets.

        :return: Dictionary with all COVID-19 datasets
        """
        return load_all_data(data_dir=self.data_dir)
