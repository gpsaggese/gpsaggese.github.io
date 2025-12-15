# Time-Series Forecasting with Transformers (AAPL Stock Prediction)

## Project Overview

This project implements a **Transformer-based time-series forecasting system** using **Xformers** to predict **future Apple (AAPL) stock prices**.
The model learns patterns from the past **60 days** and predicts the next **7 days** for **10 financial features**, including:

* Open, High, Low, Close
* Volume
* Moving averages (MA5, MA10, MA20)
* Bollinger Bands (Upper/Lower)
* Returns

The system includes a clean API layer, reusable utility functions, and demonstration notebooks.

---

# Folder Structure & Files to Submit

Your professor requires the following files, and all of them are included:

```
xformers_timeseries_utils.py
xformers_timeseries.API.md
xformers_timeseries.API.ipynb
xformers_timeseries.example.md
xformers_timeseries.example.ipynb
requirements.txt
```

### What each file does

| File | Purpose |
| --- | --- |
| **xformers_timeseries_utils.py** | Core utilities: data loading, feature engineering, sequence creation, transformer model definition, training loop, forecasting helpers |
| **xformers_timeseries.API.md** | Documentation of your “internal API”: classes, functions, how the wrapper layer works |
| **xformers_timeseries.API.ipynb** | Small notebook showing minimal usage of the API (not full training) |
| **xformers_timeseries.example.md** | Written end-to-end example of how to use the system |
| **xformers_timeseries.example.ipynb** | Full runnable notebook: loads data → trains model → evaluates → plots |
| **requirements.txt** | All necessary Python libraries |

---

# How to Run the Project

### 1. Install requirements

```
pip install -r requirements.txt
```

### 2. Run the example notebook

Open:

```
xformers_timeseries.example.ipynb
```

and click **Runtime → Restart & Run All**.

This notebook performs:

* Data download from Yahoo Finance.
* Feature engineering
* Sequence preparation
* Transformer training (Xformers attention)
* 7-day forecasting
* RMSE & MAE evaluation
* Plot generation

Artifacts generated include:

```
forecast_close.png
metrics.txt
preds.csv
actual_next_block.csv
```

---

# Model Description

This project uses a **Transformer Encoder** with:

* Multi-Head Attention (Xformers memory-efficient attention)
* Position embeddings
* Feedforward blocks
* Layer normalization
* Dropout

Model input: **60-day window × 10 features**
Model output: **7-day window × 10 features**

This is a **multi-step, multi-feature forecasting problem**.

---

# Evaluation Metrics

You will see two main metrics:

### **RMSE – Root Mean Squared Error**

* Measures error magnitude
* Penalizes large errors
* Lower = better

### **MAE – Mean Absolute Error**

* Measures average deviation
* Interpretation: “on average, predictions differ by X units”

Metrics are reported **per feature**, e.g.:

* RMSE_Close
* MAE_Open
* RMSE_Volume
* etc.

This allows understanding which indicators are easier or harder to predict.

---

# Visualization

The example notebook includes:

### **Predicted vs Actual Close Price (7 days)**

Saved as:

```
forecast_close.png
```

### **Prediction tables**

* `preds.csv`
* `actual_next_block.csv`

These are useful in your presentation.

---

# Why Multi-Feature Forecasting?

The model predicts **all 10 features** because:

* Features influence each other
* Capturing relationships improves realism
* Better than predicting only “Close”

Example:
Bollinger bands depend on moving averages, which depend on price.

---

# Why 60-Day Input Windows?

60 days:

* Capture medium-term trends
* Avoid overfitting
* Keep sequence length manageable for Transformers

A shorter window misses trends.
A longer one becomes unstable without huge training datasets.

---

# Why 7-Day Forecast Horizon?

Predicting 1 day is easy but not useful.
Predicting 30 days is unstable.

**7 days** gives a reasonable balance between:

* Stability
* Practical usefulness
* Good evaluation capability

---

# Reproducibility

This repo is fully reproducible:

* Notebooks run start-to-finish
* All core code exists in `xformers_timeseries_utils.py`
* External dependencies are listed in `requirements.txt`
* Includes baseline comparison (ARIMA)
