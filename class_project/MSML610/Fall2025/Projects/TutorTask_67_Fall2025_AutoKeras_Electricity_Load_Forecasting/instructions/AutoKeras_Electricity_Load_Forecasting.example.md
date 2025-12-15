# Electricity Load Forecasting with AutoKeras - Complete Example

## Project Overview

This example demonstrates a complete electricity load forecasting pipeline using AutoKeras. We'll predict hourly electricity demand for the PJM (Pennsylvania-New Jersey-Maryland) Interconnection, one of the largest regional transmission organizations in the United States.

**Objective**: Build an accurate hourly electricity load forecasting system that:
1. Preprocesses historical energy consumption data
2. Engineers time-series-specific features
3. Uses AutoKeras to automatically find the best model
4. Compares results with baseline forecasting methods
5. Evaluates performance using industry-standard metrics

---

## Why Electricity Load Forecasting?

### Real-World Importance

Accurate electricity load forecasting is critical for:

1. **Grid Stability**
   - Prevents blackouts by balancing supply and demand
   - Ensures reliable power delivery to consumers
   
2. **Economic Efficiency**
   - Optimizes power generation scheduling
   - Reduces operational costs
   - Minimizes waste from over-generation

3. **Renewable Integration**
   - Enables better integration of solar and wind power
   - Manages intermittent renewable sources

4. **Market Operations**
   - Supports electricity trading and pricing
   - Helps utilities plan capacity

### Technical Challenges

Electricity demand exhibits:
- **Strong seasonality**: Daily, weekly, and yearly patterns
- **Weather dependency**: Temperature strongly affects demand
- **Special events**: Holidays and events create anomalies
- **Non-stationarity**: Long-term trends change over time

---

## Dataset Description

### PJM Hourly Energy Consumption Dataset

**Source**: Kaggle - https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption

**About PJM Interconnection**:
- Coordinates electricity movement across 13 states + DC
- Serves 65 million people
- One of the largest RTOs in North America

**Dataset Characteristics**:
- **Time Period**: 1998-2018 (20 years)
- **Resolution**: Hourly measurements
- **Size**: ~130,000+ records
- **Variables**: Datetime, PJME_MW (Megawatts)
- **Format**: CSV file

**Data Features**:
```
Datetime           PJME_MW
2002-01-01 00:00   19745.0
2002-01-01 01:00   19213.0
2002-01-01 02:00   18694.0
...
```

**Patterns in the Data**:
1. **Daily Pattern**: Peak during daytime, low at night
2. **Weekly Pattern**: Lower demand on weekends
3. **Seasonal Pattern**: Higher in summer (AC) and winter (heating)
4. **Trend**: Gradual increase over years

---

## Project Workflow

```
┌─────────────────────────────────────────────────┐
│              1. DATA LOADING                    │
│  Load CSV → Parse dates → Set index → Sort     │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│           2. DATA PREPROCESSING                 │
│  Handle missing values → Interpolation          │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│         3. FEATURE ENGINEERING                  │
│  Time features → Lags → Rolling stats           │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│           4. DATA SPLITTING                     │
│  Train (70%) → Validation (10%) → Test (20%)   │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│          5. FEATURE SCALING                     │
│  Fit scaler on train → Transform all sets      │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│         6. AUTOKERAS TRAINING                   │
│  Initialize → Build → Train → Export best      │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│          7. BASELINE MODELS                     │
│  Naive → Seasonal Naive → Moving Average       │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│            8. EVALUATION                        │
│  Calculate MAE, RMSE, MAPE → Compare models    │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│          9. VISUALIZATION                       │
│  Predictions plot → Error analysis → Metrics   │
└─────────────────────────────────────────────────┘
```

---

## Step-by-Step Implementation

### Step 1: Data Loading

```python
from autokeras_utils import ElectricityDataPreprocessor

# Initialize preprocessor
preprocessor = ElectricityDataPreprocessor()

# Load and prepare data
df = preprocessor.load_and_prepare_data(
    filepath='data/PJME_hourly.csv',
    datetime_col='Datetime',
    target_col='PJME_MW'
)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
```

**What happens here**:
- Loads CSV file into pandas DataFrame
- Converts datetime strings to datetime objects
- Sets datetime as index for time series operations
- Sorts by time to ensure chronological order
- Handles any missing values using forward/backward fill

### Step 2: Exploratory Data Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot full time series
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['PJME_MW'])
plt.xlabel('Date')
plt.ylabel('Load (MW)')
plt.title('PJM Electricity Load Over Time')
plt.show()

# Basic statistics
print(df['PJME_MW'].describe())
```

**Key observations**:
- Identify overall trends
- Spot anomalies or outliers
- Understand value ranges
- Check for missing data patterns

### Step 3: Feature Engineering

This is where the magic happens! We transform raw timestamps into meaningful features.

```python
# Complete feature engineering pipeline
df_features = preprocessor.prepare_features(
    df,
    target_col='PJME_MW',
    create_lags=True,
    create_rolling=True
)

print(f"Original features: {df.shape[1]}")
print(f"After engineering: {df_features.shape[1]}")
print(f"New features: {df_features.shape[1] - df.shape[1]}")
```

**Features created**:

1. **Time Components** (13 features):
   - hour, day_of_week, day_of_month, month, quarter, year
   - hour_sin, hour_cos (cyclical encoding)
   - day_sin, day_cos (cyclical encoding)
   - month_sin, month_cos (cyclical encoding)
   - is_weekend

2. **Lag Features** (6 features):
   - Load_lag_1, Load_lag_2, Load_lag_3
   - Load_lag_24 (yesterday same hour)
   - Load_lag_48 (2 days ago same hour)
   - Load_lag_168 (last week same hour)

3. **Rolling Statistics** (20 features):
   For windows [3, 6, 12, 24, 168]:
   - rolling_mean, rolling_std
   - rolling_min, rolling_max

**Total: ~40 engineered features from 1 original feature!**

### Step 4: Data Splitting

Time series requires special splitting (no shuffling!):

```python
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
    df_features,
    target_col='PJME_MW',
    test_size=0.2,    # Last 20% for testing
    val_size=0.1      # 10% of training for validation
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")
```

**Split strategy**:
```
|────────── 72% ──────────|─ 8% ─|──── 20% ────|
|       Training          | Val  |     Test    |
└──────────── Past ────────────┴──── Future ──►
```

**Why this matters**:
- Training: Learn patterns from past
- Validation: Tune hyperparameters on recent past
- Test: Evaluate on true future (unseen data)

### Step 5: Feature Scaling

Normalize features to similar ranges:

```python
X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.normalize_features(
    X_train, X_val, X_test
)
```

**Why scale**:
- Neural networks train better with normalized inputs
- Prevents features with large values from dominating
- Speeds up convergence

**Critical**: Fit scaler ONLY on training data to avoid data leakage!

### Step 6: AutoKeras Training

Now for the automated machine learning:

```python
from autokeras_utils import AutoKerasForecaster

# Initialize forecaster
forecaster = AutoKerasForecaster(
    max_trials=10,      # Try 10 different architectures
    epochs=100,         # Train each for 100 epochs
    objective='val_loss',
    seed=42
)

# Build model
forecaster.build_model(X_train_scaled, y_train)

# Train with validation monitoring
forecaster.train(
    X_train_scaled, y_train,
    X_val_scaled, y_val,
    verbose=1
)
```

**What AutoKeras does**:
1. **Trial 1**: Tests a simple neural network
2. **Trial 2**: Tries a deeper network
3. **Trial 3**: Experiments with different activations
4. **Trial 4-10**: Continues search based on performance
5. **Selection**: Picks the best performing model

**Each trial involves**:
- Architecture selection (layers, neurons)
- Hyperparameter tuning (learning rate, batch size)
- Training and validation
- Performance tracking

### Step 7: Baseline Models

Compare AutoKeras with simple baselines:

```python
from autokeras_utils import BaselineModels

# Naive: Repeat last value
naive_pred = BaselineModels.naive_forecast(y_train, len(y_test))

# Seasonal naive: Repeat last week
seasonal_pred = BaselineModels.seasonal_naive_forecast(
    y_train, len(y_test), season_length=24
)

# Moving average: Average of last 24 hours
ma_pred = BaselineModels.moving_average_forecast(
    y_train, len(y_test), window=24
)
```

**Why baselines**:
- Establish minimum acceptable performance
- Sometimes simple methods work surprisingly well
- Validate that complexity adds value

### Step 8: Evaluation

Calculate comprehensive metrics:

```python
from autokeras_utils import print_evaluation_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

# AutoKeras predictions
ak_pred = forecaster.predict(X_test_scaled)

# Calculate metrics for all models
def evaluate_model(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    print_evaluation_metrics(metrics, name)
    return metrics

# Evaluate all models
ak_metrics = evaluate_model(y_test, ak_pred, "AutoKeras")
naive_metrics = evaluate_model(y_test, naive_pred, "Naive")
seasonal_metrics = evaluate_model(y_test, seasonal_pred, "Seasonal Naive")
ma_metrics = evaluate_model(y_test, ma_pred, "Moving Average")
```

**Understanding the metrics**:

1. **MAE (Mean Absolute Error)**:
   - Average error in Megawatts
   - "On average, predictions are off by X MW"
   - Easy to interpret

2. **RMSE (Root Mean Squared Error)**:
   - Penalizes large errors more heavily
   - Higher than MAE if large errors exist
   - Standard forecasting metric

3. **MAPE (Mean Absolute Percentage Error)**:
   - Percentage error
   - Scale-independent
   - < 10% is excellent, < 20% is good

### Step 9: Visualization

Visualize results for insights:

```python
from autokeras_utils import ForecastVisualizer

viz = ForecastVisualizer()

# 1. Predictions vs Actuals
fig1 = viz.plot_predictions(
    y_test, ak_pred,
    title="AutoKeras: Actual vs Predicted Load"
)
plt.show()

# 2. Error Distribution
fig2 = viz.plot_error_distribution(y_test, ak_pred)
plt.show()

# 3. Model Comparison
metrics_dict = {
    'AutoKeras': ak_metrics,
    'Naive': naive_metrics,
    'Seasonal': seasonal_metrics,
    'Moving Avg': ma_metrics
}
fig3 = viz.plot_metrics_comparison(metrics_dict)
plt.show()
```

---

## Expected Results

### Typical Performance

Based on the PJM dataset, you can expect:

**AutoKeras**:
- MAE: ~800-1200 MW
- RMSE: ~1200-1600 MW
- MAPE: 3-6%

**Baselines**:
- Naive: MAPE ~10-15%
- Seasonal Naive: MAPE ~6-10%
- Moving Average: MAPE ~7-12%

**Interpretation**:
- AutoKeras typically outperforms baselines by 30-50%
- MAPE < 5% is considered excellent for energy forecasting
- Seasonal patterns are captured well

### What the Visualizations Show

1. **Predictions Plot**:
   - AutoKeras closely tracks actual load
   - Captures daily and weekly patterns
   - Some divergence during extreme events

2. **Error Distribution**:
   - Errors centered around zero (unbiased)
   - Normal distribution (good model)
   - Few large outliers

3. **Residual Plot**:
   - Random scatter (no pattern)
   - Constant variance (homoscedastic)
   - Few systematic biases

---

## Key Learnings

### 1. Feature Engineering is Critical

The jump from 1 to ~40 features dramatically improves performance:
- Time components capture patterns
- Lags enable autoregression
- Rolling stats smooth noise

**Lesson**: Good features > complex models

### 2. AutoML Saves Time

Without AutoKeras, you'd need to:
- Manually design network architecture
- Tune hyperparameters via grid search
- Train multiple models for comparison
- Select the best performing one

AutoKeras does all this automatically in minutes!

### 3. Baselines Provide Context

Simple methods can be surprisingly effective:
- Seasonal naive achieves ~6-10% MAPE
- Only ~50% worse than AutoKeras
- Much faster to compute

**Lesson**: Start simple, add complexity only if needed

### 4. Time Series Requires Special Handling

Key differences from standard ML:
- No random shuffling
- Sequential train/test split
- Temporal features
- Lag-based features

**Lesson**: Respect temporal ordering

---

## Some Common Issues I faced and Solutions


### Issue 1: AutoKeras Takes Too Long

**Symptoms**:
- Training runs for hours
- Each trial is very slow

**Solutions**:
```python
# Reduce search space
forecaster = AutoKerasForecaster(
    max_trials=5,      # Fewer trials
    epochs=50          # Fewer epochs
)

# Use subset of data for development
df_subset = df.iloc[:10000]
```

### Issue 2: Memory Errors

**Symptoms**:
- Kernel crashes
- Out of memory errors

**Solutions**:
- Use smaller batch size
- Reduce max_trials
- Use fewer features
- Increase Docker memory limit

---

## Extensions and Next Steps

### 1. Multi-Step Forecasting

Predict multiple hours ahead:

```python
# Create features for t+1, t+2, t+3 forecasts
for horizon in [1, 2, 3]:
    # Shift target variable
    df[f'target_h{horizon}'] = df['PJME_MW'].shift(-horizon)
```

### 2. Incorporate External Features

Add weather data:

```python
# Example with temperature
df['temperature'] = load_weather_data()
df['temperature_lag_1'] = df['temperature'].shift(1)
df['temperature_rolling_24'] = df['temperature'].rolling(24).mean()
```

### 3. Probabilistic Forecasting

Generate prediction intervals:

```python
# Use quantile regression or ensemble methods
# Predict 5th, 50th, 95th percentiles
```

### 4. Real-Time Forecasting System

Build production pipeline:

```python
# 1. Load latest data
# 2. Engineer features
# 3. Load saved model
# 4. Generate forecasts
# 5. Store/visualize results
```

### 5. Ensemble Methods

Combine multiple models:

```python
# Average predictions from:
# - AutoKeras model
# - LSTM model
# - Prophet model
# - ARIMA model
final_pred = (ak_pred + lstm_pred + prophet_pred + arima_pred) / 4
```

---

## Conclusion

This example demonstrated:

1. **Complete forecasting pipeline**: From raw data to predictions
2. **AutoKeras integration**: Automated model selection and tuning
3. **Feature engineering**: Time series-specific transformations
4. **Model comparison**: AutoML vs traditional baselines
5. **Comprehensive evaluation**: Multiple metrics and visualizations

**Key takeaways**:
- AutoKeras simplifies time series forecasting
- Feature engineering is crucial for performance
- Baseline comparisons provide context
- Proper evaluation requires multiple perspectives

**Your results show**:
- AutoKeras achieved X% MAPE on test data
- Outperformed baselines by Y%
- Captured daily and weekly patterns effectively
- Ready for deployment with further refinement

---

## Further Reading

### Academic Papers
- "Auto-Keras: An Efficient Neural Architecture Search System" (KDD 2019)
- "Neural Architecture Search: A Survey" (JMLR 2019)
- "Electricity Load Forecasting: A Survey" (IEEE, 2020)

### Books
- "Forecasting: Principles and Practice" by Hyndman & Athanasopoulos
- "Deep Learning" by Goodfellow, Bengio, and Courville

### Online Resources
- AutoKeras Documentation: https://autokeras.com/
- Time Series with Python: https://otexts.com/fpp3/
- PJM Data Portal: https://www.pjm.com/

---

## Acknowledgments

- Dataset: PJM Interconnection via Kaggle
- Framework: AutoKeras team
- Tutorial: Course project for Big Data/AI class

---

**Ready to run the complete example? Open `AutoKeras.example.ipynb` and execute the cells!** 🚀
