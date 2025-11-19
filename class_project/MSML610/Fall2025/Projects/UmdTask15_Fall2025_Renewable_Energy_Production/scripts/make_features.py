"""
Script to turn the raw solar CSV into a feature-enriched training dataset.

I am keeping this script very simple on purpose:
1. Load the raw data from data/raw/solar_energy.csv
2. Add basic time features using the helper functions
3. Save the result into data/processed/train.csv
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RenewableEnergy_utils import (
    load_raw_solar,
    add_basic_time_features,
    save_processed,
)


def main() -> None:
    """
    End-to-end feature creation pipeline.

    This function is what I run from the command line:
    python3 scripts/make_features.py
    """
    # 1. Load the raw solar data
    df_raw = load_raw_solar()

    # 2. Choose the name of the target column in the raw data.
    #    For this dataset, the target I want to predict is energy_mwh.
    target_col = "energy_mwh"

    # 3. Create basic time features (year, month, day_of_week, hour, etc.)
    df_features = add_basic_time_features(df_raw, target_col=target_col)

    # 4. Save the processed dataset to data/processed/train.csv
    output_path = save_processed(df_features, filename="train.csv")

    print(f"Saved processed training data to: {output_path}")


if __name__ == "__main__":
    main()
