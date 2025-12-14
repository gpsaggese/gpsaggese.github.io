# Prophet API Documentation

## 🐳 Installation & Docker Setup
To ensure reproducibility, this project is containerized. Follow these steps to build and run the analysis.

### 1. Build the Image
Run this command in the project root (where the `Dockerfile` is located):
```bash
docker build -t prophet_project .
```

### 2. Run the Container
Start the Jupyter environment with volume mounting (to save your notebook changes):
```bash
# Mac/Linux/WSL
docker run -p 8888:8888 -v "$(pwd)":/app prophet_project
```

### 3. Access the Project
- Click the `http://127.0.0.1:8888...` link in your terminal to open JupyterLab.
- Open `Prophet_API.ipynb` to test the tool.
- Open `Prophet_example.ipynb` to see the full COVID-19 analysis with multi-model comparison.

---

## Overview
The Prophet utilities module (`utils.py`) provides a high-level wrapper around Facebook Prophet and comparison models for time series forecasting. It is designed to simplify the workflow of COVID-19 case prediction, specifically focusing on incorporating interventions and comparing multiple forecasting approaches.

While Prophet provides powerful decomposable time series modeling, the native API requires significant boilerplate code for data preprocessing, intervention handling, non-negative constraints, and visualization. This API standardizes that workflow into a single `ProphetWrapper` class with additional support for ARIMA, SARIMA, and LSTM models.

**Author**: Ibrahim Ahmed Mohammed  
**Course**: DATA610  
**Dataset**: Johns Hopkins University COVID-19 Time Series (Jan 2020 - March 2023)

---

## Architecture
The core component is the `ProphetWrapper` class, which orchestrates the following pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    COVID-19 Forecasting Pipeline                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ Data Prep   │───▶│   Prophet   │───▶│    Evaluation       │ │
│  │   Layer     │    │   Model     │    │    & Comparison     │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│        │                  │                      │              │
│        ▼                  ▼                      ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ Baseline    │    │ Interven-   │    │   Visualization     │ │
│  │ Models      │    │ tions       │    │   & Reporting       │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

1. **Data Preparation Layer:** Loads JHU time series data, converts cumulative to daily cases, and formats for Prophet (`ds`, `y` columns).
2. **Modeling Layer:** Wraps Prophet with sensible defaults (weekly seasonality, non-negative constraints) and provides ARIMA/SARIMA/LSTM baselines.
3. **Evaluation Layer:** Provides RMSE, MAE, SMAPE metrics and multi-model comparison utilities.
4. **Visualization Layer:** Built-in methods for forecast plots, intervention effects, and model comparisons.

---

## Class Reference: `ProphetWrapper`

### Initialization
```python
wrapper = ProphetWrapper(
    growth='linear',              # Options: 'linear', 'logistic'
    weekly_seasonality=True,      # Capture Mon-Sun reporting patterns
    yearly_seasonality=True,      # Seasonal disease patterns
    daily_seasonality=False,      # Not needed for daily aggregates
    changepoint_prior_scale=0.05, # Trend flexibility (higher = more flexible)
    seasonality_prior_scale=10.0, # Seasonality strength
    holidays_prior_scale=10.0,    # Intervention effect strength
    interval_width=0.95,          # Confidence interval width
    floor=0.0,                    # Minimum prediction value
    cap=None                      # Maximum for logistic growth
)
```

### Methods

#### `set_holidays(holidays_df)`
**Purpose:** Register interventions/holidays that affect the time series.
- **Inputs:** DataFrame with columns `holiday`, `ds`, `lower_window`, `upper_window`.
- **Output:** Returns `self` for method chaining.
- **Why use this:** COVID-19 cases are heavily influenced by policy interventions (lockdowns, vaccine rollouts). This method allows Prophet to model these effects explicitly.

#### `fit(df)`
**Purpose:** Train the Prophet model on historical data.
- **Inputs:** DataFrame with `ds` (date) and `y` (target) columns.
- **Output:** Returns `self` for method chaining.
- **Design Choice:** Automatically adds floor constraint to training data to ensure non-negative predictions, which is essential for count data like COVID cases.

#### `predict(periods=28, freq='D', include_history=True)`
**Purpose:** Generate forecasts for future periods.
- **Inputs:** 
  - `periods`: Number of days to forecast (default 28 = 4 weeks)
  - `freq`: Frequency ('D' for daily)
  - `include_history`: Whether to include historical predictions
- **Output:** DataFrame with `ds`, `yhat`, `yhat_lower`, `yhat_upper`, `trend`, `weekly`, `yearly` columns.
- **Critical Feature:** All predictions are automatically clipped to be ≥ floor (default 0), preventing nonsensical negative case counts.

#### `cross_validate(initial='365 days', period='30 days', horizon='28 days')`
**Purpose:** Perform time series cross-validation for robust model evaluation.
- **Inputs:** Initial training period, spacing between cutoffs, forecast horizon.
- **Output:** DataFrame with predictions and actuals for each CV fold.
- **Why use this:** Standard train/test splits can be misleading for time series. CV provides more reliable performance estimates.

#### `get_performance_metrics(cv_results)`
**Purpose:** Calculate performance metrics from cross-validation results.
- **Inputs:** Output from `cross_validate()`.
- **Output:** DataFrame with RMSE, MAE, MAPE, SMAPE by horizon.

#### `get_components()`
**Purpose:** Extract trend and seasonal components from the fitted model.
- **Inputs:** None (uses stored forecast).
- **Output:** Dictionary with `trend`, `weekly`, `yearly`, and `holidays` DataFrames.
- **Interpretation:** Useful for understanding what drives the forecast.

---

## Baseline Model Functions

#### `fit_arima(df, order=(5,1,0), enforce_non_negative=True)`
**Purpose:** Fit ARIMA model for statistical baseline comparison.
- **Inputs:** Prophet-formatted DataFrame, ARIMA order tuple.
- **Output:** Tuple of (fitted_model, fitted_values).
- **Design Choice:** Default order (5,1,0) captures short-term autocorrelation with first differencing for stationarity.

#### `fit_sarima(df, order=(1,1,1), seasonal_order=(1,1,1,7))`
**Purpose:** Fit Seasonal ARIMA with weekly seasonality.
- **Inputs:** DataFrame, ARIMA order, seasonal order (P,D,Q,s).
- **Output:** Tuple of (fitted_model, fitted_values).
- **Why s=7:** COVID reporting follows strong weekly cycles (lower weekend reporting). SARIMA explicitly models this.

#### `forecast_arima(model, periods=28)` / `forecast_sarima(model, periods=28)`
**Purpose:** Generate multi-step ahead forecasts from fitted ARIMA/SARIMA models.
- **Inputs:** Fitted model, number of periods.
- **Output:** NumPy array of forecasted values (non-negative).

---

## Class Reference: `LSTMForecaster`

### Initialization
```python
lstm = LSTMForecaster(
    sequence_length=14,       # Lookback window (2 weeks)
    n_features=1,             # Univariate time series
    lstm_units=[64, 32],      # Units per LSTM layer
    dropout_rate=0.2,         # Regularization
    learning_rate=0.001       # Adam optimizer LR
)
```

### Methods

#### `fit(df, epochs=100, batch_size=32, validation_split=0.1)`
**Purpose:** Train the LSTM neural network.
- **Inputs:** DataFrame, training parameters.
- **Output:** Returns `self` for method chaining.
- **Design Choice:** Uses early stopping with patience=10 to prevent overfitting.

#### `forecast(df, periods=28)`
**Purpose:** Generate multi-step recursive forecasts.
- **Inputs:** Historical DataFrame, forecast horizon.
- **Output:** NumPy array of forecasted values.
- **Logic:** Each prediction is fed back as input for the next step (recursive forecasting).

#### `get_training_history()`
**Purpose:** Retrieve training metrics for diagnostics.
- **Output:** Dictionary with `loss`, `val_loss`, `mae`, `val_mae` per epoch.

---

## Evaluation Functions

#### `calculate_rmse(actual, predicted)`
**Purpose:** Compute Root Mean Squared Error.
- **Formula:** `sqrt(mean((actual - predicted)²))`
- **Interpretation:** Penalizes large errors heavily. Same units as target variable.

#### `calculate_mae(actual, predicted)`
**Purpose:** Compute Mean Absolute Error.
- **Formula:** `mean(|actual - predicted|)`
- **Interpretation:** Average absolute deviation. More robust to outliers than RMSE.

#### `calculate_smape(actual, predicted)`
**Purpose:** Compute Symmetric Mean Absolute Percentage Error.
- **Formula:** `100 × mean(|F - A| / ((|A| + |F|) / 2))`
- **Interpretation:** Scale-independent percentage error. Bounded [0, 200%].

#### `compare_models(actual, predictions_dict)`
**Purpose:** Compare multiple models side-by-side.
- **Inputs:** Actual values, dictionary mapping model names to predictions.
- **Output:** DataFrame with RMSE, MAE, SMAPE for each model.
- **Example:**
```python
comparison = compare_models(actual, {
    'Prophet': prophet_pred,
    'ARIMA': arima_pred,
    'SARIMA': sarima_pred,
    'LSTM': lstm_pred
})
```

---

## Helper Functions

#### `load_jhu_timeseries(filepath, country='US')`
**Purpose:** Load and transform Johns Hopkins COVID-19 time series data.
- **Inputs:** Path to CSV file, country name.
- **Behavior:** 
  1. Filters to specified country
  2. Sums across provinces/states
  3. Converts cumulative to daily new cases
  4. Clips negative values (data corrections) to zero
- **Output:** Prophet-formatted DataFrame with `ds` and `y` columns.

#### `get_available_countries(filepath)`
**Purpose:** List all countries available in the JHU dataset.
- **Output:** Sorted list of 200+ country names.

#### `create_intervention_dataframe(interventions)`
**Purpose:** Convert intervention dictionary to Prophet holidays format.
- **Inputs:** Dictionary mapping intervention names to dates.
- **Output:** DataFrame with `holiday`, `ds`, `lower_window`, `upper_window`.
- **Design Choice:** Default `upper_window=14` assumes intervention effects last ~2 weeks.

#### `get_us_covid_interventions()`
**Purpose:** Pre-defined dictionary of major US COVID-19 policy dates.
- **Output:** Dictionary with 9 key interventions:
  - `national_emergency`: 2020-03-13
  - `lockdowns_begin`: 2020-03-19
  - `reopening_phase1`: 2020-05-01
  - `vaccine_auth`: 2020-12-11
  - `omicron_surge`: 2021-12-15
  - *(and 4 more)*

#### `get_country_interventions(country)`
**Purpose:** Get intervention dates for US, Germany, Brazil, or India.
- **Output:** Country-specific intervention dictionary.

#### `train_test_split_temporal(df, test_size=28)`
**Purpose:** Split time series data preserving temporal order.
- **Inputs:** DataFrame, number of days for test set.
- **Output:** Tuple of (train_df, test_df).
- **Why use this:** Random splits cause data leakage in time series. This function ensures test data is always *after* training data.

---

## Visualization Functions

#### `plot_forecast(df, forecast, title, ylabel, figsize, show_intervals=True)`
**Purpose:** Plot actual vs forecasted values with confidence intervals.
- **Output:** Matplotlib Figure with actual points, forecast line, and shaded CI.

#### `plot_intervention_effects(forecast, interventions)`
**Purpose:** Visualize how interventions affected the trend.
- **Output:** Trend line with vertical markers at each intervention date.

#### `plot_model_comparison(actual_dates, actual_values, predictions_dict)`
**Purpose:** Overlay multiple model predictions against actuals.
- **Output:** Line plot comparing all models.

#### `plot_training_history(history)`
**Purpose:** Plot LSTM training loss curves.
- **Output:** Two-panel figure with Loss and MAE over epochs.

#### `plot_forecast_comparison(df, periods, prophet, arima, sarima, lstm)`
**Purpose:** Multi-model future forecast visualization.
- **Output:** Historical data with all four model forecasts extending into the future.

---

## References

1. Taylor, S.J. and Letham, B. (2018). *Forecasting at Scale*. The American Statistician.
2. Johns Hopkins University COVID-19 Data Repository: https://github.com/CSSEGISandData/COVID-19
3. Prophet Documentation: https://facebook.github.io/prophet/
