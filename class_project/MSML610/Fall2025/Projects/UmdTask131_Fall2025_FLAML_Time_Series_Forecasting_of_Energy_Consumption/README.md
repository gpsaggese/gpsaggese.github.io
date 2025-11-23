# MSML610 - Midterm Project Submission

**Project Title:** Time-Series Forecasting of Energy Consumption using FLAML  
**Student Name:** Anisha Katiyar  
**Issue Number:** #131  
**Branch Name:** `UmdTask131_Fall2025_FLAML_Time_Series_Forecasting_of_Energy_Consumption`  
**Date:** November 10, 2025

---

## 📋 Project Overview

This project implements automated time-series forecasting for household energy consumption using **FLAML (Fast and Lightweight AutoML)**. It compares the AutoML-generated model with **Facebook Prophet** and a **hybrid ensemble**, analyzing forecast accuracy, robustness, and business impact.

**Project Difficulty:** 3 (Hard)

### Objectives
- Forecast household energy consumption using historical usage data  
- Automate model selection through FLAML  
- Compare multiple forecasting approaches (FLAML AutoML vs Prophet vs Ensemble)  
- Analyze seasonality and volatility handling capabilities  
- Implement bonus features (rolling forecast, ensemble averaging, business impact)

---

## 📊 Dataset

**Source:** UCI Machine Learning Repository – Household Electric Power Consumption  
**Link:** [https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

**Dataset Characteristics:**
- **Time Period:** December 2006 – November 2010 (47 months)  
- **Frequency:** Originally 1-minute, resampled to daily  
- **Records:** ~2 million observations  
- **Target Variable:** Global Active Power (kW)  
- **Features:** Date, Time, Voltage, Sub-metering readings, etc.  

---

## 🗂️ Folder Structure

```
Fall2025_FLAML_Time_Series_Forecasting_of_Energy_Consumption/
│
├── README.md                          # Main project documentation
├── README_MIDTERM.md                  # This midterm submission document
│
├── notebooks/
│   └── Fall2025_FLAML_Energy_Forecasting.ipynb
│
├── src/
│   ├── __init__.py
│   ├── energy_forecasting.py
│   ├── enhanced_analysis.py
│   ├── dashboard.py
│   └── config.py
│
├── data/
│   └── household_power_consumption.txt
│
├── outputs/
│   ├── cleaned_timeseries.png
│   ├── exploratory_analysis.png
│   ├── feature_importance.png
│   ├── model_comparison_results.csv
│   ├── performance_comparison.png
│   ├── predictions_comparison.png
│   ├── rolling_forecast.png
│   ├── predictions.csv
│   ├── summary.json
│   └── README.md
│
├── requirements.txt
└── .gitignore
```

---

## 📓 Midterm Submission Content

### Primary Deliverable: Jupyter Notebook

**File:** `notebooks/Fall2025_FLAML_Energy_Forecasting.ipynb`

This notebook implements a full pipeline for automated energy forecasting.

#### 1. **Data Loading & Exploration**
- Loaded and cleaned UCI dataset  
- Visualized consumption trends and seasonal cycles  
- Identified missing values (~1.2%) and interpolated them  

#### 2. **Data Preparation**
- Parsed timestamps, indexed by date  
- Resampled to daily averages  
- Applied forward fill and interpolation for missing data  
- Split into 80/20 train-test (chronological order)

#### 3. **Feature Engineering**
- **Temporal:** Day of week, month, quarter  
- **Cyclical Encoding:** sin/cos transformations for periodic features  
- **Lag Features:** 1, 7, 14, and 30-day lags  
- **Rolling Windows:** 7/14/30-day mean and std  
- **Calendar Flags:** Weekend, holiday, and seasonal indicators  

#### 4. **Model Training**
- **FLAML AutoML:**
  - Best estimator: **LightGBM**
  - Time budget: 300 seconds  
  - Test RMSE: **0.2380**, Test MAPE: **19.47%**
- **Prophet:**
  - Test RMSE: **0.2585**, Test MAPE: **22.14%**
- **Ensemble (60% FLAML + 40% Prophet):**
  - Test RMSE: **0.2356**, Test MAPE: **19.62%**

#### 5. **Evaluation & Visualizations**
- Model comparison (FLAML vs Prophet vs Ensemble)  
- Feature importance plots  
- Rolling forecast stability analysis  
- Seasonal error decomposition (weekday/month trends)

#### 6. **Bonus Analyses**
- Rolling forecast evaluation (30-day window, 7-day stride)  
- Ensemble averaging  
- Business cost impact estimation based on predictive error  

---

## 🎯 Key Tasks Completed

### ✅ Required Tasks

| Task | Status | Description |
|------|--------|-------------|
| Data Preparation | ✅ | Cleaned, resampled, interpolated missing data |
| Feature Engineering | ✅ | Lag, rolling stats, cyclical features |
| FLAML Model Training | ✅ | AutoML with LightGBM as best model |
| Prophet Baseline | ✅ | Seasonal trend modeling |
| Model Comparison | ✅ | Comprehensive RMSE & MAPE analysis |
| Visualization | ✅ | Predictions, feature importance, and error plots |
| Discussion | ✅ | Interpreted seasonality and volatility patterns |

### ⭐ Bonus Tasks

| Task | Status | Description |
|------|--------|-------------|
| Rolling Forecast | ✅ | 30-day window, 7-day step |
| Ensemble Model | ✅ | 60-40 weighted average |
| Feature Importance | ✅ | LightGBM feature ranking visualization |
| Business Impact | ✅ | Estimated cost savings from forecast accuracy |

---

## 📈 Results Summary

### Model Performance

| Model | Train RMSE | Test RMSE | Test MAPE | Notes |
|-------|-------------|-----------|-----------|-------|
| **FLAML AutoML (LightGBM)** | 0.2256 | 0.2380 | 19.47% | Best overall performer |
| **Prophet** | 0.2912 | 0.2585 | 22.14% | Baseline model |
| **Ensemble** | 0.2423 | 0.2356 | 19.62% | Slightly improved RMSE over FLAML |

### Rolling Forecast Stability

| Model | Avg RMSE | Std Dev | Comment |
|-------|-----------|----------|----------|
| FLAML | 0.2188 | ±0.066 | Strong stability across windows |
| Prophet | 0.2421 | ±0.055 | Higher variability |

### Key Findings

1. **FLAML’s LightGBM** outperformed Prophet by ~8% in RMSE and 12% in MAPE.  
2. **Ensemble averaging** yielded marginal improvement in generalization.  
3. **Weekday-weekend differences** were significant; weekend consumption patterns more volatile.  
4. **FLAML’s rolling performance** showed high temporal stability.  
5. **Economic analysis** indicated potential **annual cost reduction of ~$4.7 per household** with improved accuracy.

---

## 🚀 How to Run

### Setup

```bash
cd Fall2025_FLAML_Time_Series_Forecasting_of_Energy_Consumption/
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Run Notebook

```bash
jupyter notebook notebooks/Fall2025_FLAML_Energy_Forecasting.ipynb
```

### Optional Dashboard (later)

```bash
streamlit run src/dashboard.py
```

---

## 📚 Dependencies

Core packages:
- Python 3.8+
- pandas, numpy, matplotlib, seaborn  
- scikit-learn, lightgbm  
- flaml, prophet  
- jupyter, streamlit  

---

## 🔄 Current Status

### Completed for Midterm:
- [x] Data pipeline and EDA  
- [x] FLAML and Prophet implementation  
- [x] Rolling evaluation and ensembling  
- [x] Key results visualizations  
- [x] Cost impact analysis  

### In Progress:
- [ ] Extended parameter tuning  
- [ ] Dashboard UI refinements  
- [ ] Final report polishing  

### Planned for Final:
- [ ] Deep learning baseline (LSTM)  
- [ ] Multistep forecasting evaluation  
- [ ] Deployment pipeline integration  

---

## 💡 Challenges & Learnings

### Challenges:
1. Managing large dataset (2M+ records) efficiently  
2. Handling gaps and irregular timestamps  
3. Balancing feature complexity with compute limits  
4. Prophet tuning for daily-level granularity  

### Learnings:
1. AutoML can drastically reduce experimentation time.  
2. Lag-based temporal features are crucial for high accuracy.  
3. Model ensembling smooths extreme variations.  
4. Rolling forecasts provide realistic robustness estimates.  

---

## 📝 Next Steps

### For Final Submission:
1. Integrate weather/temperature data as exogenous features.  
2. Add more model baselines (ARIMA, LSTM).  
3. Perform cross-validation for temporal consistency.  
4. Expand business metric analysis.  
5. Finalize dashboard and documentation.  

---

## 🎓 Academic Integrity Statement

This project is completed individually as part of MSML610 coursework.  
All code is original or properly cited.  
Dataset sourced from UCI Machine Learning Repository (acknowledged below).

---

## 🔗 References

1. UCI Machine Learning Repository – Household Electric Power Consumption Dataset  
2. Microsoft FLAML Documentation – https://microsoft.github.io/FLAML/  
3. Facebook Prophet Documentation – https://facebook.github.io/prophet/  
4. Scikit-learn User Guide – https://scikit-learn.org/  

---

**Last Updated:** November 10, 2025  
**Status:** Midterm Submission – Completed ✅