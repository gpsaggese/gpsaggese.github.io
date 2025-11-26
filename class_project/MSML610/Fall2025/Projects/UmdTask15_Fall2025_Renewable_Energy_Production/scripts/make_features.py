#!/usr/bin/env python3
"""
Generate processed features for the Renewable Energy Forecasting project.
"""

from pathlib import Path
import sys

# ---------------------------------------------------------------------
# Make sure the project root is on sys.path so we can import RenewableEnergy_utils
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from RenewableEnergy_utils import (
    load_data,
    make_basic_time_features,
    TIME_COL,
    TARGET_COL,
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "solar_energy.csv"
PROCESSED_DIR = DATA_DIR / "processed"


def main() -> None:
    print(f"[make_features] Project root      : {PROJECT_ROOT}")
    print(f"[make_features] Raw data path     : {RAW_PATH}")

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found at: {RAW_PATH}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_data(str(RAW_PATH))
    print(f"[make_features] Raw shape         : {df_raw.shape}")

    df_feats = make_basic_time_features(df_raw)
    print(f"[make_features] Features shape    : {df_feats.shape}")
    print(f"[make_features] Columns           : {list(df_feats.columns)}")

    out_path = PROCESSED_DIR / "train.csv"
    df_feats.to_csv(out_path, index=True)
    print(f"[make_features] Saved features to : {out_path}")


if __name__ == "__main__":
    main()
