# COVID Case Prediction

This project provides a small pipeline to download public COVID-19 CSV datasets
and load them into pandas DataFrames for preprocessing and model building.

Structure

- `src/` contains the source code (data loading and processing)
- `utils_data_io.py` thin helper for notebooks and examples
- `utils_post_processing.py` post-processing helpers
- `data/` where CSV files are stored (data loaded via `load_data.py`)

See `API.md` and `example.md` for usage.

Data preparation
----------------

We separate out data acquisition and preprocessing from modeling. Use the following sequence to prepare data for GluonTS training:

1. Download the raw CSVs into `data/` (uses Google Drive links configured in `scripts/download_data.py`):

```bash
python scripts/download_data.py
```

2. Preprocess raw CSVs into a single national-level file (`data/processed/national_data.csv`):

```bash
python src/data_processing/preprocess_data.py
```


2. Preprocess raw CSVs into a single national-level file (`data/processed/national_data.csv`):

```bash
python src/data_processing/preprocess_data.py
```

Alternatively, open `example.ipynb` or `API.ipynb` where we show the same steps inline (you can run the notebook cells to execute the loader + preprocessing flow).

Notes
-----
- The downloader uses `gdown` under the hood. Please make sure that you have this installed in your environment (`pip install gdown`).
- The preprocessing step expects the JHU time-series CSVs to be present in `data/` with filenames that `load_data.py` expects (see `scripts/download_data.py` for default names).
- If you have the processed file already (`data/processed/national_data.csv`), `prepare_gluonts.py` will skip reprocessing and proceed to create GluonTS datasets.
