# FLAML Example: Time-Series Forecasting of Energy Consumption

## Overview

This notebook (`FLAML_Example.ipynb`) demonstrates a complete, real-world application
of the **FLAML AutoML library** for time-series forecasting. While the companion
`FLAML_API.ipynb` introduces FLAML's interface and capabilities in a general context,
this notebook applies those concepts to solve an actual forecasting problem:
**predicting household energy consumption**.

The project uses the UCI Household Electric Power Consumption dataset and showcases
how AutoML can accelerate the development of accurate forecasting models while
handling the complexities of temporal data.

---

## Project Objective

Forecast daily household energy consumption using historical usage data. The project
focuses on:

1. Experimenting with multiple forecasting approaches through FLAML
2. Handling seasonality and temporal patterns in energy data
3. Comparing AutoML-selected models against traditional baselines
4. Providing actionable insights for energy management

---

## Key Results

### Model Performance Summary

| Model | Test RMSE | Test MAPE | Test R² | Accuracy |
|-------|-----------|-----------|---------|----------|
| **FLAML (XGBoost)** | **0.0388** | **3.36%** | **0.9843** | **96.64%** |
| FLAML (LightGBM) | 0.0422 | 3.59% | 0.9815 | 96.41% |
| FLAML (Random Forest) | 0.0666 | 3.87% | 0.9538 | 96.13% |
| FLAML (Extra Trees) | 0.0679 | 5.22% | 0.9520 | 94.78% |
| Prophet | 0.2585 | 22.14% | 0.3034 | 77.86% |
| ARIMA | 0.3344 | 37.86% | -0.1658 | 62.14% |
| Ensemble (60-40) | 0.1143 | 9.41% | 0.8639 | 90.59% |

### Key Findings

1. **FLAML's XGBoost** achieved 96.64% accuracy — outperforming Prophet by 18.78% and ARIMA by 34.50%

2. **EMA and rolling features** were the most important predictors (ema_7: 23.5% importance)

3. **Feature engineering was crucial** — 41 engineered features enabled FLAML to dramatically outperform simpler models

4. **Ensemble didn't improve over FLAML** — when one model dominates, ensembling dilutes performance

5. **Rolling forecast evaluation** confirmed temporal stability (CV = 0.201, moderate stability)

---

## Dataset

**Source:** UCI Machine Learning Repository  
**Name:** Individual Household Electric Power Consumption  
**URL:** https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

### Dataset Characteristics

| Property | Value |
|----------|-------|
| Time Period | December 2006 – November 2010 (47 months) |
| Original Frequency | 1-minute intervals |
| Processed Frequency | Daily aggregates |
| Raw Records | 2,075,259 |
| Processed Records | 1,442 (daily) → 1,412 (after feature engineering) |
| Target Variable | Global Active Power (kW) |
| Train/Test Split | 80/20 (1,129 / 283 samples) |

### Features in Raw Data

- `Date` and `Time`: Timestamp components
- `Global_active_power`: Household global minute-averaged active power (kW)
- `Global_reactive_power`: Household global minute-averaged reactive power (kW)
- `Voltage`: Minute-averaged voltage (V)
- `Global_intensity`: Household global minute-averaged current intensity (A)
- `Sub_metering_1`: Kitchen energy (Wh)
- `Sub_metering_2`: Laundry room energy (Wh)
- `Sub_metering_3`: Water heater and air conditioner energy (Wh)

---

## Methodology

### 1. Data Preparation

The raw minute-level data undergoes several preprocessing steps:

- **Resampling:** Aggregate 2M+ records to 1,442 daily averages (1,439x compression)
- **Missing Value Handling:** Hybrid approach (forward fill → backward fill → linear interpolation)
- **Date Parsing:** Convert to proper datetime index with Prophet-compatible format (ds, y)

### 2. Feature Engineering

41 features engineered from the target variable:

| Category | Count | Features | Purpose |
|----------|-------|----------|---------|
| **Temporal** | 13 | day_of_week, month, quarter, year, season, is_weekend, cyclical encodings | Calendar patterns |
| **Lag** | 6 | lag_1, lag_2, lag_3, lag_7, lag_14, lag_30 | Autocorrelation |
| **Rolling** | 12 | rolling_mean/std/min/max for 7, 14, 30 days | Trends and volatility |
| **EMA** | 2 | ema_7, ema_30 | Weighted recent trends |
| **Difference** | 2 | diff_1, diff_7 | Momentum/change |

### 3. Model Training

Three modeling approaches are compared:

#### FLAML AutoML (4 Estimators)

- **Estimators:** LightGBM, XGBoost, Random Forest, Extra Trees
- **Time Budget:** 120 seconds per model
- **Metric:** RMSE
- **Cross-validation:** 3-fold
- **Best Model:** XGBoost (96.64% accuracy)

#### Facebook Prophet (Baseline)

- Explicit trend and seasonality decomposition
- Weekly and yearly seasonality components
- Multiplicative seasonality mode

#### ARIMA (Statistical Baseline)

- Augmented Dickey-Fuller test for stationarity
- ARIMA(1, 0, 1) configuration
- Classical time series approach

#### Ensemble Model (BONUS)

- Weighted combination: 60% FLAML + 40% Prophet
- Tests if model diversity improves predictions

### 4. Evaluation Metrics

| Metric | Description | Best Value |
|--------|-------------|------------|
| RMSE | Root Mean Square Error | Lower |
| MAE | Mean Absolute Error | Lower |
| MAPE | Mean Absolute Percentage Error | Lower |
| R² | Coefficient of Determination | Higher (max 1.0) |
| Accuracy | 100% - MAPE | Higher |

### 5. Advanced Analysis (BONUS Features)

**Feature Importance Analysis:**

- XGBoost gain-based importance scores
- Category breakdown visualization
- Top 20 features ranked

**Rolling Forecast Evaluation (BONUS):**

- 30-day windows with 7-day steps
- 37 evaluation windows across test set
- Temporal stability assessment (CV metric)

**Ensemble Forecasting (BONUS):**

- 60% FLAML + 40% Prophet weighted average
- Comparison against individual models

**Seasonality & Volatility Analysis:**

- Seasonal error breakdown (Winter, Spring, Summer, Fall)
- Model approach comparison for handling seasonality

---

## Notebook Structure

The notebook is organized into 8 clearly defined sections:

### Section 1: Setup and Configuration

- Library imports with error handling
- Configuration dictionary (DRY principle)
- Output directory setup
- Random seed for reproducibility

### Section 2: Data Loading and Exploration

- Load UCI dataset with proper parsing
- Data quality assessment (missing values, data types)
- 9-panel exploratory analysis visualization
- Statistical summary and skewness analysis

### Section 3: Data Preprocessing

- Resampling from minute to daily frequency
- Missing value imputation (hybrid approach)
- Prophet-compatible format (ds, y columns)
- Cleaned time series visualization with moving averages

### Section 4: Feature Engineering

- 41 features created using utils.py functions
- Feature correlation analysis
- Top 20 features by correlation with target
- Category breakdown visualization

### Section 5: Train-Test Split

- Chronological split (80/20) — no shuffling
- Data leakage prevention verification
- Distribution comparison (train vs test)
- Feature matrices preparation (X_train, y_train, X_test, y_test)

### Section 6: Model Training & Comparison

- 6.1: FLAML AutoML (4 estimators trained individually)
- 6.2: Prophet baseline with explicit seasonality
- 6.3: ARIMA statistical baseline with stationarity test
- 6.4: Comprehensive model comparison and ranking

### Section 7: Advanced Analysis (BONUS)

- 7.1: Feature importance analysis
- 7.2: Ensemble forecasting (60-40 weighted) — BONUS 
- 7.3: Rolling forecast evaluation — BONUS 
- 7.4: Seasonality & volatility analysis

### Section 8: Business Impact & Final Summary

- 8.1: Business impact analysis (ROI, payback period)
- 8.2: Save all results (CSV, JSON, PNG)
- 8.3: Final summary and project requirements checklist

---

## How to Run

### Prerequisites

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Key packages:

- `flaml>=2.1.0`
- `prophet>=1.1.5`
- `pandas>=2.0.0`
- `scikit-learn>=1.3.0`
- `lightgbm>=4.0.0`
- `xgboost>=2.0.0`
- `statsmodels>=0.14.0`

### Running the Notebook

1. **Ensure the dataset exists:**

   ```
   data/household_power_consumption.txt
   ```

2. **Open the notebook:**

   ```bash
   jupyter notebook FLAML_Example.ipynb
   ```

3. **Execute all cells:**
   - Use "Kernel → Restart & Run All"
   - Total runtime: ~10-15 minutes (depending on FLAML time budget)

4. **Check outputs:**
   - Visualizations saved to `outputs/` directory
   - Results in `outputs/summary.json`

### Running with Docker

```bash
cd docker/
docker compose up --build
```

Then access JupyterLab at `http://localhost:8888`

---

## Output Files

The notebook generates the following outputs in the `outputs/` directory:

### Visualizations (6 PNG files, ~4 MB total)

| File | Description |
|------|-------------|
| `exploratory_analysis.png` | 9-panel EDA visualization |
| `cleaned_timeseries.png` | Daily consumption with moving averages |
| `feature_correlations.png` | Feature correlation heatmap |
| `train_test_split.png` | Train/test split visualization |
| `feature_importance.png` | XGBoost feature importance |
| `rolling_forecast.png` | Rolling evaluation results |

### Data Files (3 CSV files)

| File | Description |
|------|-------------|
| `predictions.csv` | Actual vs predicted for all models |
| `model_comparison.csv` | Final model performance metrics |
| `flaml_candidates_comparison.csv` | 4 FLAML models comparison |

### Summary (1 JSON file)

| File | Description |
|------|-------------|
| `summary.json` | Complete project results and configuration |

---

## Utility Functions

The `utils.py` module provides reusable functions:

```python
from utils import (
    add_temporal_features,
    add_lag_features,
    add_rolling_features,
    add_ema_features,
    calculate_metrics,
    create_comparison_table
)
```

Key functions:

- `add_temporal_features()` - Add calendar-based features (13 features)
- `add_lag_features()` - Add historical value lags (6 features)
- `add_rolling_features()` - Add rolling statistics (12 features)
- `add_ema_features()` - Add exponential moving averages (2 features)
- `calculate_metrics()` - Compute RMSE, MAE, MAPE, R²

---

## Best Practices Demonstrated

1. **Temporal Data Handling:**
   - Chronological train-test split (no shuffling)
   - Lag features respect temporal ordering
   - No future data leakage

2. **Feature Engineering:**
   - Domain-appropriate features (temporal, cyclical)
   - Shift operations to prevent leakage
   - Feature importance analysis for interpretability

3. **Model Evaluation:**
   - Multiple metrics (RMSE, MAPE, R²)
   - Rolling window validation for temporal stability
   - Comparison against multiple baselines (Prophet, ARIMA)

4. **Code Quality:**
   - DRY principle with utils.py
   - Configuration dictionary for parameters
   - Fixed random seeds for reproducibility

5. **Documentation:**
   - Comprehensive markdown documentation before/after each code section
   - Self-contained notebook with explanations for beginners
   - Clear output interpretations

---

## Educational Value

This notebook serves as a comprehensive learning resource for students and practitioners interested in time series forecasting and AutoML.

### Concepts Learned

| Topic | What You'll Learn |
|-------|-------------------|
| **AutoML** | How FLAML automates model selection and hyperparameter tuning |
| **Time Series Fundamentals** | Temporal splits, avoiding data leakage, handling autocorrelation |
| **Feature Engineering** | Creating lag, rolling, EMA, and cyclical features from raw data |
| **Model Comparison** | Evaluating multiple models with appropriate metrics |
| **Statistical Baselines** | Understanding ARIMA and stationarity testing |
| **Ensemble Methods** | When and why to combine model predictions |

### Skills Developed

| Skill | Application in Notebook |
|-------|------------------------|
| **Data Preprocessing** | Resampling, imputation, handling 2M+ records |
| **Exploratory Data Analysis** | Multi-panel visualizations, distribution analysis |
| **Python Libraries** | pandas, scikit-learn, FLAML, Prophet, statsmodels |
| **Model Interpretation** | Feature importance, error analysis, seasonality decomposition |
| **Business Translation** | Converting technical metrics to ROI and business impact |
| **Documentation** | Writing clear, reproducible, self-contained notebooks |

### Key Takeaways for Beginners

1. **Feature engineering often matters more than model choice** — Our 41 engineered features enabled 96.64% accuracy

2. **Always use temporal splits for time series** — Random splits cause data leakage and inflated metrics

3. **Simpler isn't always better** — ARIMA (simple) achieved 62% accuracy vs XGBoost (complex) at 97%

4. **Ensemble models aren't magic** — They only help when component models have complementary strengths

5. **Smoothed features beat raw values** — EMA and rolling features outperformed raw lag features

6. **Validation strategy matters** — Rolling window evaluation reveals temporal stability issues

### Who Should Use This Notebook?

| Audience | What They'll Gain |
|----------|-------------------|
| **ML Students** | End-to-end project experience, proper methodology |
| **Data Scientists** | FLAML implementation patterns, time series best practices |
| **Energy Analysts** | Domain-specific forecasting techniques |
| **AutoML Beginners** | Practical introduction to automated machine learning |
| **Researchers** | Reproducible baseline for energy forecasting studies |

---

## Limitations and Future Work

### Current Limitations

- Single household data (may not generalize to other households)
- No external features (weather, holidays)
- Daily granularity only (loses intraday patterns)
- Negative ROI for single household deployment

### Potential Improvements

1. **External Features:**
   - Weather data (temperature, humidity)
   - Holiday calendars (French holidays)
   - Economic indicators

2. **Model Enhancements:**
   - Deep learning baselines (LSTM, Transformer)
   - Multi-step ahead forecasting
   - Probabilistic predictions with uncertainty quantification

3. **Scaling:**
   - Multi-household aggregation for positive ROI
   - Utility-scale deployment

4. **Deployment:**
   - Real-time prediction API
   - Automated retraining pipeline
   - Monitoring dashboard (Streamlit app provided)

---

## Contact

**Author:** Anisha Katiyar  
**Course:** MSML610 - Advanced Machine Learning  
**Institution:** University of Maryland  
**Term:** Fall 2025

---