# Energy Consumption Forecasting for Smart Grids (PMDARIMA)
**Level:** Hard · **Project 3**

## Objective
Develop a forecasting model to predict **hourly energy consumption** for a smart grid system using the **PMDARIMA** library.  
The goal is to optimize the model for accuracy while handling large-scale and noisy time-series data.

## Dataset
**UCI Machine Learning Repository:**  
[Individual Household Electric Power Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

- Time range: 2006–2010  
- Sampling rate: 1 minute (aggregated to hourly)  
- Target variable: `Global_active_power`

*Note*: The dataset is automatically downloaded from the UCI repository by `load_energy_data()` if not present locally.

## Project Structure

```
class_project/MSML610/Fall2025/Projects/UmdTask48_Fall2025_pmdarima_Energy_Consumption_Forecasting_for_Smart_Grids/
├── pmdarima.API.ipynb        # PMDARIMA API demonstration
├── pmdarima.API.md           # API documentation and explanation
├── pmdarima.example.ipynb    # End-to-end project implementation
├── pmdarima.example.md       # Example explanation and evaluation
├── pmdarima_utils.py         # Utility functions for data loading and metrics
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration
└── README.md                 # Project overview
```

## Project Status
✅ Completed 

✔ Data Preparation

- Loaded the UCI Power Consumption dataset
- Parsed dates and times
- Handled missing values
- Resampled to hourly frequency

✔ Time Series Decomposition

- Performed trend, seasonality and residual decomposition

✔ Model Development

- Applied pmdarima.auto_arima
- Selected best ARIMA model automatically

✔ Model Validation

- Implemented rolling cross-validation (RollingForecastCV)

✔ Forecasting and Analysis

- Generated full 7-day (168-hour) forecast
- Visualized actual vs. forecast
- Included confidence intervals

✔ Bonus Extensions 

- Integrated weather data as exogenous regressors
- Implemented Hybrid ARIMA + ML (Random Forest residual model)

## How to Run
```bash
docker build -t pmdarima-energy-forecast .
docker run -p 8888:8888 pmdarima-energy-forecast
```

Then open Jupyter in your browser and run the notebooks sequentially.
  
## Dependencies
- numpy
- pandas
- matplotlib
- scikit-learn
- statsmodels
- pmdarima
