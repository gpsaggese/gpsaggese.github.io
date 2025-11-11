"""Utilities for data I/O used by the COVID case prediction project.

This module provides a thin wrapper around the data loader implemented in
`src/data_processing/load_data.py` so that notebooks and the top-level project
files can import a single stable module (`utils_data_io`) as required by the
project submission template.

The wrapper is intentionally small and only adjusts `sys.path` so `src` can be
imported when running code from the project root or from Jupyter notebooks.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

# Make sure `src` directory is importable when running from the project root.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import the DataLoader implemented under src/data_processing.
from data_processing.load_data import DataLoader


def download_and_load_all(gdrive_urls: Dict[str, str], data_dir: str = "data") -> Dict[str, "pandas.DataFrame"]:
    """Download all CSVs from Google Drive and load them into DataFrames.

    Args:
        gdrive_urls: mapping name -> google drive url
        data_dir: local directory where files will be stored

    Returns:
        mapping name -> pandas.DataFrame
    """
    loader = DataLoader(gdrive_urls=gdrive_urls, data_dir=data_dir)
    loader.download_all()
    loader.load_all()
    return loader.dataframes
