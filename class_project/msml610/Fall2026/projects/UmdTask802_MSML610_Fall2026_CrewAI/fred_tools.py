"""
Utilities for downloading and profiling FRED macroeconomic data.
"""

from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests


FRED_SERIES = {
    "CPIAUCSL": "Consumer Price Index",
    "UNRATE": "Unemployment Rate",
    "FEDFUNDS": "Federal Funds Rate",
    "T10Y2Y": "10-Year Minus 2-Year Treasury Spread",
}

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def download_fred_series(
    series_id: str,
    *,
    data_dir: str = "data/fred",
    download_date: Optional[str] = None,
) -> Path:
    """
    Download and save one FRED series as a dated CSV snapshot.

    :param series_id: FRED series identifier.
    :param data_dir: Root directory for saved snapshots.
    :param download_date: Optional ISO date used for the snapshot directory.
    :return: Path to the saved CSV file.
    """
    if series_id not in FRED_SERIES:
        raise ValueError(f"Unsupported FRED series: {series_id}")

    snapshot_date = download_date or date.today().isoformat()
    snapshot_dir = Path(data_dir) / snapshot_date
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    output_path = snapshot_dir / f"{series_id}.csv"

    response = requests.get(
        FRED_CSV_URL,
        params={"id": series_id},
        timeout=30,
    )
    response.raise_for_status()

    # Validate the response before saving it.
    dataframe = pd.read_csv(BytesIO(response.content))
    expected_columns = {"observation_date", series_id}
    if not expected_columns.issubset(dataframe.columns):
        raise ValueError(
            f"Unexpected columns for {series_id}: "
            f"{dataframe.columns.tolist()}"
        )

    output_path.write_bytes(response.content)
    return output_path


def download_all_fred_series(
    *,
    data_dir: str = "data/fred",
    download_date: Optional[str] = None,
) -> dict[str, Path]:
    """
    Download all required FRED series into one dated snapshot directory.

    :param data_dir: Root directory for saved snapshots.
    :param download_date: Optional ISO date used for the snapshot directory.
    :return: Mapping from series identifier to saved CSV path.
    """
    return {
        series_id: download_fred_series(
            series_id,
            data_dir=data_dir,
            download_date=download_date,
        )
        for series_id in FRED_SERIES
    }


def load_fred_series(csv_path: Path, series_id: str) -> pd.Series:
    """
    Load and clean a saved FRED series.

    :param csv_path: Path to a saved FRED CSV file.
    :param series_id: FRED series identifier.
    :return: Numeric series indexed by observation date.
    """
    dataframe = pd.read_csv(csv_path)

    if "observation_date" not in dataframe.columns:
        raise ValueError(f"Missing observation_date in {csv_path}")
    if series_id not in dataframe.columns:
        raise ValueError(f"Missing {series_id} in {csv_path}")

    dataframe["observation_date"] = pd.to_datetime(
        dataframe["observation_date"],
        errors="coerce",
    )
    dataframe[series_id] = pd.to_numeric(
        dataframe[series_id],
        errors="coerce",
    )

    cleaned = (
        dataframe.dropna(subset=["observation_date"])
        .set_index("observation_date")[series_id]
        .sort_index()
    )
    return cleaned


def profile_fred_series(
    csv_path: Path,
    series_id: str,
) -> dict[str, Any]:
    """
    Compute a deterministic profile of one saved FRED series.

    :param csv_path: Path to a saved FRED CSV file.
    :param series_id: FRED series identifier.
    :return: Latest value, prior-year value, dates, and missing-value count.
    """
    series = load_fred_series(csv_path, series_id)
    valid_series = series.dropna()

    if valid_series.empty:
        raise ValueError(f"No valid observations found for {series_id}")

    latest_date = valid_series.index[-1]
    latest_value = float(valid_series.iloc[-1])

    target_date = latest_date - pd.DateOffset(years=1)
    prior_values = valid_series.loc[:target_date]
    if prior_values.empty:
        prior_date = None
        prior_value = None
    else:
        prior_date = prior_values.index[-1]
        prior_value = float(prior_values.iloc[-1])

    return {
        "series_id": series_id,
        "series_name": FRED_SERIES[series_id],
        "latest_date": latest_date.date().isoformat(),
        "latest_value": latest_value,
        "prior_year_date": (
            prior_date.date().isoformat() if prior_date is not None else None
        ),
        "prior_year_value": prior_value,
        "missing_values": int(series.isna().sum()),
        "csv_path": str(csv_path),
    }

def profile_fred_snapshot(
    snapshot_dir: Path,
) -> dict[str, dict[str, Any]]:
    """
    Profile all four FRED series in a saved snapshot.

    :param snapshot_dir: Directory containing the four CSV files.
    :return: Mapping from series identifier to its profile.
    """
    snapshot_dir = Path(snapshot_dir)

    return {
        series_id: profile_fred_series(
            snapshot_dir / f"{series_id}.csv",
            series_id,
        )
        for series_id in FRED_SERIES
    }

def calculate_macro_statistics(
    snapshot_dir: Path,
) -> dict[str, Any]:
    """
    Calculate the macroeconomic statistics used in the briefing.

    :param snapshot_dir: Directory containing the four saved CSV files.
    :return: Deterministic macroeconomic statistics.
    """
    profiles = profile_fred_snapshot(snapshot_dir)

    cpi = profiles["CPIAUCSL"]
    unemployment = profiles["UNRATE"]
    fed_funds = profiles["FEDFUNDS"]
    yield_spread = profiles["T10Y2Y"]

    if cpi["prior_year_value"] is None:
        raise ValueError("CPI does not contain a prior-year observation.")
    if unemployment["prior_year_value"] is None:
        raise ValueError(
            "UNRATE does not contain a prior-year observation."
        )
    if fed_funds["prior_year_value"] is None:
        raise ValueError(
            "FEDFUNDS does not contain a prior-year observation."
        )

    inflation_yoy_pct = (
        (cpi["latest_value"] / cpi["prior_year_value"]) - 1
    ) * 100

    unemployment_change_pp = (
        unemployment["latest_value"]
        - unemployment["prior_year_value"]
    )

    fed_funds_change_pp = (
        fed_funds["latest_value"]
        - fed_funds["prior_year_value"]
    )

    spread_value = yield_spread["latest_value"]
    if spread_value > 0:
        yield_curve_sign = "positive"
    elif spread_value < 0:
        yield_curve_sign = "negative"
    else:
        yield_curve_sign = "flat"

    return {
        "inflation_yoy_pct": round(inflation_yoy_pct, 2),
        "inflation_latest_date": cpi["latest_date"],
        "unemployment_rate_pct": round(
            unemployment["latest_value"],
            2,
        ),
        "unemployment_change_pp": round(
            unemployment_change_pp,
            2,
        ),
        "unemployment_latest_date": unemployment["latest_date"],
        "fed_funds_rate_pct": round(
            fed_funds["latest_value"],
            2,
        ),
        "fed_funds_change_pp": round(
            fed_funds_change_pp,
            2,
        ),
        "fed_funds_latest_date": fed_funds["latest_date"],
        "yield_spread_pp": round(spread_value, 2),
        "yield_curve_sign": yield_curve_sign,
        "yield_spread_latest_date": yield_spread["latest_date"],
    }