# Cleaning & Classifying Twitter Sentiment — PR 1 (Cleaning)

This folder contains a reproducible **OpenRefine cleaning workflow** for Sentiment140 and minimal utilities to load/inspect the cleaned CSV.

## What’s in this PR
- `API.md` (with the **OpenRefine steps JSON**) and `API.ipynb` (walkthrough).
- `utils_data_io.py`, `utils_post_processing.py` (helpers).
- `example.md`, `example.ipynb` (sanity checks on the cleaned file).
- `Dockerfile` (optional local run for notebooks).

## Expected input / output
- **Input (raw)**: `training.1600000.processed.noemoticon.csv` (not tracked)
- **Output (clean)**: `Sentiment140_raw.csv` (not tracked)

## Next PR (planned)
- Baseline model (TF-IDF + Logistic Regression) + tests + CI.
