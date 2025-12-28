"""
pmdarima_utils.py

This file contains utility functions that support the tutorial notebooks for the
Energy Consumption Forecasting using PMDARIMA project.

- Notebooks should call these functions instead of writing raw logic inline.
- This helps keep the notebooks clean, modular, and easier to debug.
- Students should implement functions here for data preprocessing,
  model setup, evaluation, or any reusable logic.
"""

import os
import urllib.request
import zipfile
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Load and preprocess the dataset
# -----------------------------------------------------------------------------
def load_energy_data(path: str = "data/household_power_consumption.txt") -> pd.DataFrame:
    """
    Load and preprocess the UCI Individual Household Electric Power Consumption dataset.
        Automatically downloads the dataset if not found locally.
    - Combines 'Date' and 'Time' columns into a single datetime index.
    - Converts missing values ('?') to NaN and drops invalid rows.
    - Resamples the target variable to hourly frequency for forecasting.

    :param path: Path to the dataset (.txt)
    :return: Cleaned and hourly-resampled DataFrame containing 'Global_active_power'
    """
    # Auto-download if file missing
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = (
             "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
        )
        zip_path = os.path.join(os.path.dirname(path), "household_power_consumption.zip")
        logger.info("Dataset not found locally. Downloading from UCI...")
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(path))
        os.remove(zip_path)
        logger.info("Dataset successfully downloaded and extracted to %s", path)

    logger.info("Loading dataset from %s", path)

    # Reading dataset
    df = pd.read_csv(
        filepath_or_buffer=path, sep=";", na_values=["?"], low_memory=False
    )

    logger.info("Initial dataset shape: %s", df.shape)

    # Combining 'Date' and 'Time' columns into a datetime index
    if "Date" not in df.columns or "Time" not in df.columns:
        raise KeyError("Expected 'Date' and 'Time' columns in dataset")

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )

    df.drop(columns=["Date", "Time"], inplace=True)
    df.dropna(subset=["datetime"], inplace=True)

    # Keep only target variable and handle missing values
    if "Global_active_power" not in df.columns:
        raise KeyError("'Global_active_power' column not found in dataset")

    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"], errors="coerce"
    )
    df = df.dropna(subset=["Global_active_power"])

    # Resample hourly
    df = df.set_index("datetime").resample("h").mean().dropna()

    logger.info("After resampling: %s", df.shape)
    return df


# -----------------------------------------------------------------------------
# Split the dataset into train and test sets
# -----------------------------------------------------------------------------


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Split the dataset chronologically into training and testing sets.

    :param df: preprocessed DataFrame
    :param train_ratio: proportion of data used for training (default = 0.8)

    :return: train_set, test_set
    """
    logger.info("Splitting dataset with train ratio = %.2f", train_ratio)

    split_point = int(len(df) * train_ratio)
    train = df.iloc[:split_point]
    test = df.iloc[split_point:]

    logger.info("Train size: %d | Test size: %d", len(train), len(test))
    return train, test


# -----------------------------------------------------------------------------
# Evaluate PMDARIMA model performance
# -----------------------------------------------------------------------------


def evaluate_forecast(model, test_series: pd.Series) -> dict:
    """
    Evaluate a trained PMDARIMA model on test data.

    :param model: trained PMDARIMA model
    :param test_series: actual observed test values

    :return: dictionary containing MAE and RMSE metrics
    """
    logger.info("Generating forecasts for %d periods", len(test_series))

    forecast = model.predict(n_periods=len(test_series))

    mae = mean_absolute_error(test_series, forecast)
    rmse = np.sqrt(mean_squared_error(test_series, forecast))

    logger.info("Evaluation complete → MAE: %.4f | RMSE: %.4f", mae, rmse)
    return {"MAE": mae, "RMSE": rmse}


# -----------------------------------------------------------------------------
# Quick demo to verify workflow
# -----------------------------------------------------------------------------


def quick_demo(path: str):
    """
    Quick test to verify the data loading and splitting workflow.

    :param path: path to the dataset
    :return: full dataset, train subset, test subset
    """
    df = load_energy_data(path)
    train, test = split_train_test(df)
    print(f"Loaded {len(df)} records → Train: {len(train)}, Test: {len(test)}")
    return df, train, test

# -----------------------------------------------------------------------------
# Load weather data 
# -----------------------------------------------------------------------------

def load_weather_data(path: str = "data/weather.csv") -> pd.DataFrame:
    """
    Load local weather dataset. Auto-download from Open-Meteo only if missing.
    """

    # If file already exists → just load it
    if os.path.exists(path):
        logger.info("Loading cached weather data from %s", path)
        return pd.read_csv(path, parse_dates=["datetime"])

    # Otherwise → download and save to CSV
    logger.info("Weather file not found. Downloading real hourly weather from Open-Meteo...")

    import requests

    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        "latitude=48.86&longitude=2.35&"
        "start_date=2006-12-16&end_date=2010-11-26&"
        "hourly=temperature_2m"
    )

    weather = requests.get(url).json()

    weather_df = pd.DataFrame({
        "datetime": pd.to_datetime(weather["hourly"]["time"]),
        "temperature": weather["hourly"]["temperature_2m"]
    })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    weather_df.to_csv(path, index=False)

    logger.info("Weather dataset downloaded and saved → %s", path)

    return weather_df

