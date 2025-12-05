# electricity_forecast.example.md

This file describes the example pipeline used in the project notebook.
The goal is to build and evaluate LSTM models for hourly electricity
consumption using historical load values.

The notebook follows these steps:

### 1. Data loading and preparation
- Load the raw CSV (AEP hourly load).
- Convert timestamps to datetime, sort by time.
- Resample to hourly and fill any gaps.
- Split into train / validation / test based on time.
- Build sliding-window sequences for LSTM input.

### 2. Baseline LSTM model
A simple one-layer LSTM is trained to predict the next hour of load.
The results (MAE/RMSE) form a baseline against which tuned models are
compared.

### 3. Hyperparameter tuning
Keras Tuner is used to search over:
- number of LSTM units  
- dropout  
- learning rate  

The tuned model improves over the baseline.

### 4. Multi-step forecasting (bonus)
A separate model is trained to predict the next 24 hours at once.
Errors are higher, which is expected for longer horizons.

### 5. Classical baseline: Prophet
Prophet is fit as a reference model. It performs worse than LSTM on
high-frequency hourly load, but provides a useful comparison.

### 6. Model comparison
The notebook ends with a table comparing:

- Baseline LSTM  
- Tuned LSTM  
- 24-step multi-output LSTM  
- Prophet baseline  

The tuned 1-step LSTM is the best-performing model.

### How to run
Simply run the cells in `electricity_forecast.example.ipynb`.  
All data preparation, training, and evaluation steps are contained there.