# 📘 Anomaly Detection API Documentation

This document describes the functions defined in `anomaly_utils.py`, designed for reproducible
machine-learning workflows on the UNSW-NB15 network intrusion dataset.

## Functions Overview
- **load_unsw_data(paths, columns)** → loads and merges dataset parts
- **preprocess_data(df)** → builds numeric & categorical preprocessing pipelines
- **train_rf() / train_xgb()** → train supervised models
- **evaluate_model()** → compute ROC-AUC & PR-AUC metrics

The API layer allows quick experimentation with consistent preprocessing and evaluation standards.
