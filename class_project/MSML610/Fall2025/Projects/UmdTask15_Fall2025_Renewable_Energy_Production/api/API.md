# API Documentation — Renewable Energy Forecasting

This document summarizes how the core project functions in
`RenewableEnergy_utils.py` act as a small, reusable API for the
Renewable Energy Forecasting system.

The purpose of this API is to provide simple, consistent functions that
can be used across:

- Python scripts (training, feature generation)
- Jupyter notebooks (analysis and demonstration)
- Streamlit app (interactive forecasting)

These functions allow any user to load data, engineer features, create
a time-based train/validation split, and build models without having to
manually rewrite preprocessing code.

---

## API Functions Overview

### **1. load_data(path)**
Loads the raw solar energy dataset from a CSV file, parses timestamps,
sorts the data, and returns a clean DataFrame ready for feature
engineering.

### **2. make_basic_time_features(df)**
Transforms the raw dataset by adding:

- Time features (hour, day of week, month)  
- Lag features (1-hour, 2-hour, 24-hour lags)  
- Rolling averages  

This produces the main feature-engineered dataset used for modeling.

### **3. train_val_split(df, test_size_days=7)**
Creates a **time-aware** split for forecasting by assigning the last
N days of data to the validation set.  
Returns training features, validation features, targets, and the list of
feature columns.

---

## Purpose of This API Notebook

The API notebook (`API.ipynb`) demonstrates:

- How to import and use the API functions  
- How to load and transform the raw solar dataset  
- How to generate features and split the data  
- How to train a simple model using the API outputs  
- How to compute and view basic evaluation metrics  

This notebook serves as a quick reference showing that the project code
can be used like a small forecasting library, not just as standalone
scripts.

---

## Summary

The project exposes a clean and modular API consisting of three core
functions:

1. **load_data** – ingest the raw dataset  
2. **make_basic_time_features** – build forecasting features  
3. **train_val_split** – produce a proper time-based split  

The API notebook walks through how to use these functions, confirming
that the system is reusable, consistent, and easy to integrate into
pipelines, notebooks, or applications such as the Streamlit frontend.
