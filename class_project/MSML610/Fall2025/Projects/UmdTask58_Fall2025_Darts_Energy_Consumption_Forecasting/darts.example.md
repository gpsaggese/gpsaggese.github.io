<!-- toc -->

- [Energy Consumption Forecasting](#energy-consumption-forecasting)
  * [Project Overview](#project-overview)
  * [Dataset](#dataset)
  * [Architecture](#architecture)
    + [Data Pipeline](#data-pipeline)
    + [Feature Engineering](#feature-engineering)
    + [Model Training](#model-training)
  * [Models Implemented](#models-implemented)
  * [Hyperparameter Tuning](#hyperparameter-tuning)
  * [Results and Evaluation](#results-and-evaluation)
  * [Usage](#usage)
  * [References](#references)

<!-- tocstop -->

# Energy Consumption Forecasting

A comprehensive time series forecasting project for predicting energy consumption
in the PJM East region using the Darts library.

## Project Overview

**Objective:** Forecast energy consumption for a region based on historical
usage patterns, optimizing for the model that provides the most accurate
multi-step forecasts.

**Key Tasks:**
1. **Data Ingestion:** Load the dataset and parse date-time information
2. **Feature Engineering:** Create temporal features, lagged values, rolling averages
3. **Model Comparison:** Compare Prophet, N-BEATS, LSTM, and statistical models
4. **Hyperparameter Tuning:** Optimize using grid search and cross-validation
5. **Visualization:** Plot predicted vs. actual consumption across time windows

**Implementation Files:**
- `darts.example.py` - Main project implementation script
- `darts.example.ipynb` - Interactive notebook with visualizations
- `darts_utils.py` - Utility functions for data processing and evaluation

## Dataset

**PJME Hourly Energy Consumption**

| Attribute | Value |
|-----------|-------|
| Source | [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) |
| Region | PJM East (Pennsylvania-New Jersey-Maryland) |
| Frequency | Hourly |
| Time Range | 2002-2018 |
| Records | ~145,000 hourly observations |
| Target Variable | Energy consumption in Megawatts (MW) |

**Data Characteristics:**
- Strong daily seasonality (24-hour cycle)
- Weekly seasonality (weekday vs. weekend patterns)
- Yearly seasonality (summer peaks, winter variations)
- Some missing timestamps requiring interpolation

## Architecture

### Data Pipeline

The project follows a modular pipeline architecture:

```
Raw Data → Load & Parse → Handle Missing → Create TimeSeries → Feature Engineering
                                                    ↓
                                            Train/Test Split
                                                    ↓
                                              Scale Data
                                                    ↓
                                         Train Multiple Models
                                                    ↓
                                           Evaluate & Compare
                                                    ↓
                                        Select Best Model
```

**Key Components:**

1. **DataPipeline Class:**
   - Handles data loading from CSV
   - Parses datetime and creates proper index
   - Fills missing timestamps via interpolation
   - Creates Darts TimeSeries objects
   - Manages train/test splitting and scaling

2. **ModelTrainer Class:**
   - Trains multiple forecasting models
   - Evaluates predictions against test data
   - Stores results for comparison

3. **HyperparameterTuner Class:**
   - Performs grid search for model optimization
   - Tracks best parameters

### Feature Engineering

The following features are engineered from the datetime index:

**Temporal Features:**
| Feature | Description |
|---------|-------------|
| `hour` | Hour of day (0-23) |
| `dayofweek` | Day of week (0=Monday, 6=Sunday) |
| `month` | Month (1-12) |
| `quarter` | Quarter (1-4) |
| `dayofyear` | Day of year (1-365/366) |
| `weekofyear` | Week number |
| `is_weekend` | Binary: 1 if Saturday/Sunday |
| `is_peak_hour` | Binary: 1 if 7 AM - 10 PM |
| `season` | Season encoding (0-3) |

**Lag Features:**
| Feature | Description |
|---------|-------------|
| `lag_1h` | Energy consumption 1 hour ago |
| `lag_24h` | Energy consumption 24 hours ago |
| `lag_48h` | Energy consumption 48 hours ago |
| `lag_168h` | Energy consumption 1 week ago |

**Rolling Features:**
| Feature | Description |
|---------|-------------|
| `rolling_mean_24h` | 24-hour rolling average |
| `rolling_std_24h` | 24-hour rolling std deviation |
| `rolling_mean_168h` | 1-week rolling average |

### Model Training

**Training Configuration:**
- Training data: Last 3 years of data
- Test data: Last 30 days (720 hours)
- Forecast horizon: 7 days (168 hours)
- Neural network data is scaled using StandardScaler

## Models Implemented

### 1. Naive Seasonal (Baseline)

Simple baseline that repeats the pattern from the previous week.

```python
model = NaiveSeasonal(K=168)  # Weekly seasonality
```

### 2. Exponential Smoothing

Classical statistical model with additive daily seasonality.

```python
model = ExponentialSmoothing(
    seasonal_periods=24,
    trend=None,
    seasonal='add'
)
```

### 3. Prophet

Facebook's additive forecasting model with multiple seasonalities.

```python
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    seasonality_mode='multiplicative'
)
```

### 4. N-BEATS

Neural Basis Expansion Analysis - deep learning model designed for
interpretable time series forecasting.

```python
model = NBEATSModel(
    input_chunk_length=168,   # 1 week lookback
    output_chunk_length=24,   # 1 day forecast
    num_stacks=10,
    num_layers=4,
    layer_widths=256,
    n_epochs=50
)
```

### 5. LSTM

Long Short-Term Memory recurrent neural network.

```python
model = RNNModel(
    model='LSTM',
    input_chunk_length=168,
    output_chunk_length=24,
    hidden_dim=64,
    n_rnn_layers=2,
    dropout=0.1,
    n_epochs=50
)
```

## Hyperparameter Tuning

Grid search is performed on N-BEATS with the following parameter ranges:

| Parameter | Values Tested |
|-----------|--------------|
| `input_chunk_length` | [72, 168] |
| `output_chunk_length` | [24, 48] |
| `num_stacks` | [5, 10] |
| `num_layers` | [2, 4] |
| `layer_widths` | [128, 256] |

**Tuning Process:**
1. Split training data into train/validation sets
2. Train models with reduced epochs (20) for each parameter combination
3. Evaluate on validation set using MAPE
4. Select parameters with lowest MAPE
5. Retrain final model with optimal parameters and full epochs

## Results and Evaluation

**Evaluation Metrics:**

| Metric | Formula | Description |
|--------|---------|-------------|
| MAPE | Mean Absolute Percentage Error | Percentage error |
| RMSE | Root Mean Squared Error | Absolute error in MW |
| MAE | Mean Absolute Error | Average absolute error |
| SMAPE | Symmetric MAPE | Symmetric percentage error |

**Expected Performance Range:**
- Best models typically achieve MAPE < 5%
- Deep learning models (N-BEATS, LSTM) generally outperform statistical methods
- Prophet provides strong baseline with automatic seasonality detection

**Error Analysis:**
- Error patterns analyzed by hour of day and day of week
- Higher errors typically occur during:
  - Peak demand hours (mornings and evenings)
  - Weekend transitions
  - Seasonal transitions

## Usage

### Running the Script

```bash
python darts.example.py
```

### Using the Classes

```python
from darts.example import ForecastConfig, DataPipeline, ModelTrainer

# Configure pipeline
config = ForecastConfig()
config.test_size = 24 * 14  # 2 weeks test

# Initialize components
data_pipeline = DataPipeline(config)
trainer = ModelTrainer(config)

# Load data
data_pipeline.load_data()
series = data_pipeline.create_time_series()
train, test = data_pipeline.split_data()

# Train models
trainer.train_prophet(train, test)
results = trainer.get_comparison_summary()
```

### Using Utility Functions

```python
import darts_utils as utils

# Load data
df = utils.load_energy_data('data/PJME_hourly.csv')

# Feature engineering
df = utils.create_temporal_features(df)
df = utils.add_lag_features(df)
df = utils.add_rolling_features(df)

# Create TimeSeries
series = utils.create_darts_series(df)

# Evaluate forecast
metrics = utils.evaluate_forecast(actual, predicted, "Model Name")
```

## References

- **Darts Library:** https://unit8co.github.io/darts/
- **Dataset:** https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
- **N-BEATS Paper:** Oreshkin et al. (2019). "N-BEATS: Neural basis expansion
  analysis for interpretable time series forecasting"
- **Prophet Paper:** Taylor & Letham (2018). "Forecasting at Scale"
- **PJM Interconnection:** https://www.pjm.com/

