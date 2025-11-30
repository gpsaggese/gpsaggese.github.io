"""Feature engineering utilities for the IoT anomaly detection project.

This module exposes a single function, ``compute_features``, which
accepts a pandas DataFrame containing the raw IoT sensor data and
returns a new DataFrame enriched with a variety of hand‑crafted
features.  The goal of this module is to provide a lightweight
implementation reminiscent of TSFRESH's automated feature extraction
without relying on external dependencies.  All of the transformations
are performed using standard NumPy and pandas operations, making it
easy to integrate into batch jobs or streaming pipelines.

The engineered features include:

* **Basic sensor transformations** – squares, square roots, and
  logarithms of the original sensor readings.
* **Per‑machine z‑scores** – each sensor is standardized by machine
  identifier to highlight deviations from a machine's typical range.
* **Row statistics** – the mean, standard deviation, minimum and
  maximum across all sensors for each time step.
* **Interaction terms** – products and ratios of selected sensor
  combinations that often signal emerging faults.
* **Temporal encodings** – hour of day and day of week (cyclic
  encodings) derived from the timestamp column.

The return value preserves the ``machine_id`` and original target
labels; only new feature columns are added.  Users should merge the
resulting DataFrame back to the labels prior to training models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SENSOR_COLUMNS = [
    "temperature",
    "vibration",
    "humidity",
    "pressure",
    "energy_consumption",
]

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered features for IoT sensor data.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with at least the columns
        ``machine_id``, ``timestamp`` and the five sensor columns
        listed in ``SENSOR_COLUMNS``.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed the same as ``df`` containing the original
        machine_id and timestamp along with newly created feature
        columns.  The target labels present in ``df`` are also
        included unchanged.
    """
    # Ensure timestamp is datetime
    data = df.copy()
    if not np.issubdtype(data["timestamp"].dtype, np.datetime64):
        data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Basic sensor transformations
    for col in SENSOR_COLUMNS:
        val = data[col].astype(float)
        data[f"{col}_squared"] = val ** 2
        data[f"{col}_sqrt"] = np.sqrt(np.abs(val))
        # log1p to avoid issues with zero/negative values
        data[f"{col}_log1p"] = np.log1p(np.abs(val))

    # Per‑machine z‑scores
    for col in SENSOR_COLUMNS:
        grp = data.groupby("machine_id")[col]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, 1)
        data[f"{col}_zscore"] = (data[col] - mean) / std

    # Row statistics across sensors
    sensor_values = data[SENSOR_COLUMNS].astype(float)
    data["sensor_mean"] = sensor_values.mean(axis=1)
    data["sensor_std"] = sensor_values.std(axis=1)
    data["sensor_min"] = sensor_values.min(axis=1)
    data["sensor_max"] = sensor_values.max(axis=1)
    data["sensor_range"] = data["sensor_max"] - data["sensor_min"]

    # Interaction terms (pairwise products and ratios)
    data["temp_vibration_product"] = data["temperature"] * data["vibration"]
    data["pressure_energy_ratio"] = data["pressure"] / (data["energy_consumption"] + 1e-3)
    data["humidity_pressure_product"] = data["humidity"] * data["pressure"]

    # Temporal encodings
    data["hour"] = data["timestamp"].dt.hour.astype(int)
    data["dayofweek"] = data["timestamp"].dt.dayofweek.astype(int)
    # Cyclical encoding for hour (0–23) and dayofweek (0–6)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["day_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7)
    data["day_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7)

    # Advanced rolling window features (by machine)
    for col in SENSOR_COLUMNS:
        # Sort by machine_id and timestamp for proper rolling computation
        data_sorted = data.sort_values(["machine_id", "timestamp"])
        grouped = data_sorted.groupby("machine_id")[col]

        # Rolling mean and std (window=5)
        data[f"{col}_rolling_mean_5"] = grouped.transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        data[f"{col}_rolling_std_5"] = grouped.transform(lambda x: x.rolling(window=5, min_periods=1).std().fillna(0))

        # Rate of change
        data[f"{col}_diff"] = grouped.transform(lambda x: x.diff().fillna(0))

        # Exponential weighted mean
        data[f"{col}_ewm"] = grouped.transform(lambda x: x.ewm(span=5, adjust=False).mean())

    # Anomaly score features (distance from machine mean)
    for col in SENSOR_COLUMNS:
        grp_mean = data.groupby("machine_id")[col].transform("mean")
        grp_std = data.groupby("machine_id")[col].transform("std").replace(0, 1)
        data[f"{col}_deviation"] = np.abs(data[col] - grp_mean) / grp_std

    # Composite anomaly score
    deviation_cols = [c for c in data.columns if c.endswith("_deviation")]
    data["composite_anomaly_score"] = data[deviation_cols].mean(axis=1)

    # Preserve targets if present
    target_columns = [c for c in df.columns if c not in SENSOR_COLUMNS + ["timestamp", "machine_id"]]
    # Create return DataFrame with only engineered features, machine_id and targets
    feature_cols = [c for c in data.columns if c not in df.columns or c in target_columns or c == "machine_id"]
    return data[feature_cols].copy()

if __name__ == "__main__":
    # Simple CLI: load CSV path and save features as CSV
    import argparse
    parser = argparse.ArgumentParser(description="Compute engineered features for IoT sensor data.")
    parser.add_argument("--input", required=True, help="Path to input CSV file containing raw data")
    parser.add_argument("--output", required=True, help="Path to output CSV file to save engineered features")
    args = parser.parse_args()

    df_input = pd.read_csv(args.input)
    df_features = compute_features(df_input)
    df_features.to_csv(args.output, index=False)
    print(f"Saved engineered features to {args.output}")