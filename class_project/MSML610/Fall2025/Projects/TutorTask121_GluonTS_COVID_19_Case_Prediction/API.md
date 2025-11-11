# COVID Case Prediction — API

This document describes the API provided by this project (stable, small wrapper surface used by notebooks and examples).

Public entry points

- `utils_data_io.download_and_load_all(gdrive_urls, data_dir="data") -> dict`
  - Downloads the CSV files from Google Drive into `data/` and returns a
    mapping of names to pandas DataFrames.
- `DataLoader` (class in `src/data_processing/load_data.py`)
  - `DataLoader(gdrive_urls: dict, data_dir: str)`
  - `download_all(overwrite: bool=False)`
  - `load_all()`
  - `get_dataframe(name: str) -> pandas.DataFrame`

Notes

- The implementation uses `gdown` to fetch public Google Drive files. The files in the Google Drive are public allowing gdown to pull from it. If, for some reason a file is private, you must either make it shareable or download it manually and put it under the `data/` directory.
- Keep the `utils_data_io` functions thin: notebooks should call these helpers rather than embedding file-download logic.
