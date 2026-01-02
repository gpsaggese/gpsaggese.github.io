"""Feature engineering utilities for the IoT anomaly detection project.

This module exposes a single function, ``compute_features``, which
accepts a pandas DataFrame containing the raw IoT sensor data and
returns a new DataFrame enriched with features extracted using the
tsfresh library.  tsfresh is a Python package for automated extraction
of relevant features from time series data.

The engineered features include:

* **tsfresh extracted features** - comprehensive time series features
  automatically extracted from sensor readings including statistical,
  spectral, and complexity-based features.
* **Per-machine features** - features are computed per machine identifier
  to capture machine-specific patterns.
* **Temporal encodings** - hour of day and day of week (cyclic
  encodings) derived from the timestamp column.

The return value preserves the ``machine_id`` and original target
labels; only new feature columns are added.  Users should merge the
resulting DataFrame back to the labels prior to training models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters

SENSOR_COLUMNS = [
    "temperature",
    "vibration",
    "humidity",
    "pressure",
    "energy_consumption",
]

def compute_features(df: pd.DataFrame, use_minimal: bool = True) -> pd.DataFrame:
    """Compute engineered features for IoT sensor data using tsfresh.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with at least the columns
        ``machine_id``, ``timestamp`` and the five sensor columns
        listed in ``SENSOR_COLUMNS``.
    use_minimal : bool, default=True
        If True, use MinimalFCParameters for faster computation.
        If False, use ComprehensiveFCParameters for more features.

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

    # Add row identifier for joining later
    data["row_id"] = np.arange(len(data))

    # Prepare data for tsfresh: needs columns [id, time, value]
    # We'll create separate dataframes for each sensor and stack them
    tsfresh_data_list = []

    for sensor in SENSOR_COLUMNS:
        sensor_df = data[["row_id", "machine_id", "timestamp", sensor]].copy()
        sensor_df["sensor_type"] = sensor
        sensor_df = sensor_df.rename(columns={sensor: "value"})
        tsfresh_data_list.append(sensor_df)

    # Combine all sensors into tsfresh format
    tsfresh_data = pd.concat(tsfresh_data_list, ignore_index=True)
    tsfresh_data = tsfresh_data.sort_values(["row_id", "machine_id", "sensor_type", "timestamp"])

    # Extract features using tsfresh
    # Use row_id as the id and machine_id + sensor_type as grouping
    tsfresh_data["ts_id"] = tsfresh_data["row_id"].astype(str) + "_" + tsfresh_data["sensor_type"]

    # Select feature extraction parameters
    if use_minimal:
        extraction_settings = MinimalFCParameters()
    else:
        extraction_settings = ComprehensiveFCParameters()

    # Extract features
    extracted_features = extract_features(
        tsfresh_data,
        column_id="ts_id",
        column_sort="timestamp",
        column_value="value",
        default_fc_parameters=extraction_settings,
        impute_function=impute,
        disable_progressbar=True
    )

    # Parse ts_id back to row_id
    extracted_features["row_id"] = extracted_features.index.str.split("_").str[0].astype(int)
    extracted_features["sensor_type"] = extracted_features.index.str.split("_").str[1]

    # Pivot features to have one row per row_id with all sensor features
    # First, add sensor prefix to column names
    feature_cols = [c for c in extracted_features.columns if c not in ["row_id", "sensor_type"]]

    # Reshape so each row_id has features from all sensors
    pivoted_features = []
    for sensor in SENSOR_COLUMNS:
        sensor_features = extracted_features[extracted_features["sensor_type"] == sensor].copy()
        sensor_features = sensor_features.drop(columns=["sensor_type"])
        sensor_features = sensor_features.set_index("row_id")
        # Rename columns to include sensor name
        sensor_features.columns = [f"{sensor}_{col}" for col in sensor_features.columns]
        pivoted_features.append(sensor_features)

    # Join all sensor features
    all_features = pivoted_features[0]
    for feat_df in pivoted_features[1:]:
        all_features = all_features.join(feat_df, how="outer")

    # Add temporal encodings (not in tsfresh by default)
    data["hour"] = data["timestamp"].dt.hour.astype(int)
    data["dayofweek"] = data["timestamp"].dt.dayofweek.astype(int)
    # Cyclical encoding for hour (0-23) and dayofweek (0-6)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["day_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7)
    data["day_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7)

    # Join tsfresh features back to original data
    data = data.set_index("row_id")
    data = data.join(all_features, how="left")
    data = data.reset_index(drop=True)

    # Fill any remaining NaN values with 0
    feature_columns = all_features.columns.tolist()
    data[feature_columns] = data[feature_columns].fillna(0)

    # Replace infinite values with 0
    data = data.replace([np.inf, -np.inf], 0)

    # Preserve targets, timestamp, and original sensor columns if present
    target_columns = [c for c in df.columns if c not in SENSOR_COLUMNS + ["timestamp", "machine_id"]]

    # Return with engineered features + original sensors, machine_id, timestamp and targets
    keep_cols = ["machine_id", "timestamp"] + SENSOR_COLUMNS + target_columns + feature_columns + ["hour_sin", "hour_cos", "day_sin", "day_cos"]
    return_cols = [c for c in keep_cols if c in data.columns]

    return data[return_cols].copy()

if __name__ == "__main__":
    # Simple CLI: load CSV path and save features as CSV
    import argparse
    parser = argparse.ArgumentParser(description="Compute engineered features for IoT sensor data using tsfresh.")
    parser.add_argument("--input", required=True, help="Path to input CSV file containing raw data")
    parser.add_argument("--output", required=True, help="Path to output CSV file to save engineered features")
    parser.add_argument("--comprehensive", action="store_true", help="Use comprehensive feature set (slower)")
    args = parser.parse_args()

    df_input = pd.read_csv(args.input)
    df_features = compute_features(df_input, use_minimal=not args.comprehensive)
    df_features.to_csv(args.output, index=False)
    print(f"Saved engineered features to {args.output}")
