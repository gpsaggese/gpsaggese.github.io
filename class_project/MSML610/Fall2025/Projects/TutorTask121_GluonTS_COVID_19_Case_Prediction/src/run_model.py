"""Placeholder model runner.

This script demonstrates the place where you'd call model training/prediction
once the data has been generated (downloaded and preprocessed). It checks for
presence of expected data files under `data/` and exits with an error if any
is missing.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXPECTED = [
    "cases.csv",
    "deaths.csv",
    "mobility.csv",
    "vaccine.csv",
]


missing = [f for f in EXPECTED if not (DATA_DIR / f).exists()]
if missing:
    print("Missing data files:", missing)
    print("Please run: make generate-data (or place the CSVs under ./data/)")
    sys.exit(2)

print("All required data files are present. Running model(s)...")

# TODO: Import and run models
print("(Model run completed - placeholder)")
