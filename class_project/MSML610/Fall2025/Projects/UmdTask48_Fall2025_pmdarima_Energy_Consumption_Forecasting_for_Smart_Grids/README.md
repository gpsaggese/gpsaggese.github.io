# Energy Consumption Forecasting for Smart Grids (PMDARIMA)

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
class_project/MSML610/Fall2025/Projects/UmdTask48_Fall2025_pmdarima_Energy_Consumption_Forecasting_for_Smart_Grids/  
├── pmdarima.API.ipynb  # PMDARIMA API demonstration  
├── pmdarima.API.md  # API documentation and explanation  
├── pmdarima.example.ipynb  # End-to-end forecasting example  
├── pmdarima.example.md  # Example explanation and evaluation  
├── pmdarima_utils.py  # Utility functions for data loading and metrics  
├── requirements.txt  # Python dependencies   
└── README.md  # Project overview

## Completed
- ✅ Project initialized
- ✅ Data loading and helper functions implemented
- ✅ Model training and forecasting completed

## In progress
- Adding 7-day forecast evaluation and decomposition plots.
- Implementation of cross-validation for model validation.
- Bonus extensions (weather data, hybrid ARIMA + ML)
  
## Dependencies
- numpy
- pandas
- matplotlib
- scikit-learn
- statsmodels
- pmdarima
