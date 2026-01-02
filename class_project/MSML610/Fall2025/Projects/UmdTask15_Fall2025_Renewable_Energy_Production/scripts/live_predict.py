#!/usr/bin/env python3
"""
Live prediction demo for the Renewable Energy Forecasting project.

What this script does:
1) Reads the most recent engineered lag/rolling features from data/processed/train.csv
   so the model input is realistic (instead of filling lags with 0.0).
2) Fetches current weather from OpenWeather (temp/cloud/wind) using OPENWEATHER_API_KEY.
3) Builds a MLflow /invocations payload using the exact feature schema used in training.
4) Calls the deployed MLflow model server at http://127.0.0.1:1234/invocations.
"""

import json
import datetime as dt
from pathlib import Path

import pandas as pd
import requests

from scripts.openweather_client import (
    fetch_openweather_current,
    extract_features_from_openweather,
)

MODEL_URL = "http://127.0.0.1:1234/invocations"

FEATURE_COLUMNS = [
    "temp_c",
    "cloud_cover",
    "solar_radiation",
    "wind_speed",
    "hour",
    "dayofweek",
    "month",
    "energy_mwh_lag_1",
    "energy_mwh_lag_2",
    "energy_mwh_lag_24",
    "energy_mwh_rollmean_3",
    "energy_mwh_rollmean_24",
]


def load_latest_lag_features(project_root: Path) -> dict:
    """
    Load last available lag/rolling features from processed train.csv.

    This makes the inference payload realistic because the deployed model
    was trained with these lag/rolling columns.
    """
    processed_path = project_root / "data" / "processed" / "train.csv"
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed features file not found: {processed_path}\n"
            f"Run: python3 scripts/make_features.py"
        )

    df = pd.read_csv(processed_path)

    required = [
        "energy_mwh_lag_1",
        "energy_mwh_lag_2",
        "energy_mwh_lag_24",
        "energy_mwh_rollmean_3",
        "energy_mwh_rollmean_24",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required lag/rolling columns in {processed_path}: {missing}"
        )

    last = df.iloc[-1]
    return {c: float(last[c]) for c in required}


def main() -> None:
    # Location (Maryland-ish coords; you can change)
    lat, lon = 39.0, -77.0

    # Resolve project root: .../UmdTask15... (parent of scripts/)
    project_root = Path(__file__).resolve().parent.parent

    # 0) Load realistic lag/rolling features from processed data
    lag_feats = load_latest_lag_features(project_root)

    # 1) Build time features from "now"
    now = dt.datetime.now()
    time_feats = {
        "hour": int(now.hour),
        "dayofweek": int(now.weekday()),
        "month": int(now.month),
    }

    # 2) Fetch live weather
    weather_raw = fetch_openweather_current(lat, lon, units="metric")
    weather_feats = extract_features_from_openweather(weather_raw)

    # 3) Assemble model input row
    row = {**weather_feats, **time_feats, **lag_feats}

    # Ensure ordering matches training schema
    data_row = [row[c] for c in FEATURE_COLUMNS]

    payload = {
        "dataframe_split": {
            "columns": FEATURE_COLUMNS,
            "data": [data_row],
        }
    }

    # 4) Call deployed MLflow model
    r = requests.post(MODEL_URL, json=payload, timeout=20)
    r.raise_for_status()

    print("Prediction result:")
    print(json.dumps(r.json(), indent=2))


if __name__ == "__main__":
    main()
