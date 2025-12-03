# ⚡ Energy Consumption Forecasting with Darts

A comprehensive time series forecasting project for predicting energy consumption in the PJM East region using the Darts library.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Models Implemented](#models-implemented)
- [Feature Engineering](#feature-engineering)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [References](#references)

---

## Project Overview

**Objective:** Forecast energy consumption for a region based on historical usage patterns, optimizing for the model that provides the most accurate multi-step forecasts.

**Key Tasks:**

| Task | Description |
|------|-------------|
| **Data Ingestion** | Load the dataset and parse date-time information for time series analysis |
| **Feature Engineering** | Create temporal features, lagged values, and rolling averages |
| **Model Comparison** | Compare Prophet, N-BEATS, LSTM, and statistical models |
| **Hyperparameter Tuning** | Optimize using grid search and cross-validation |
| **Visualization** | Plot predicted vs. actual consumption across time windows |

---

## Project Structure

```
Darts/
│
├── 📂 data/
│   └── PJME_hourly.csv              # Dataset (download from Kaggle)
│
├── 📓 Notebooks
│   ├── darts.example.ipynb          # Main project notebook (interactive)
│   └── darts.API.ipynb              # Darts API tutorial notebook
│
├── 🐍 Python Scripts
│   ├── darts.example.py             # Project implementation script
│   ├── darts.API.py                 # API exploration script
│   └── darts.utils.py               # Utility functions module
│
├── 📖 Documentation
│   ├── darts.example.md             # Project documentation
│   ├── darts.API.md                 # API documentation
│   └── README.md                    # This file
│
├── 🐳 Docker
│   ├── Dockerfile                   # Multi-stage Docker build
│   ├── docker-compose.yml           # Docker Compose configuration
│   └── .dockerignore                # Docker ignore patterns
│
├── 📂 scripts/
│   └── docker-entrypoint.sh         # Docker entrypoint script
│
├── 📂 output/                       # Generated outputs (charts, CSVs)
├── 📂 models/                       # Saved model files
│
└── 📦 Configuration
    └── requirements.txt             # Python dependencies
```

---

## File Descriptions

### Core Project Files

| File | Type | Description |
|------|------|-------------|
| `darts.example.ipynb` | Notebook | **Main project notebook** - Complete interactive analysis with data ingestion, EDA, feature engineering, model training, hyperparameter tuning, and visualizations |
| `darts.example.py` | Script | **Project implementation** - Object-oriented Python script with `ForecastConfig`, `DataPipeline`, `ModelTrainer`, and `HyperparameterTuner` classes |
| `darts.example.md` | Docs | **Project documentation** - Detailed explanation of architecture, models, tuning process, and usage examples |

### API Tutorial Files

| File | Type | Description |
|------|------|-------------|
| `darts.API.ipynb` | Notebook | **API tutorial notebook** - Interactive exploration of Darts library features: TimeSeries creation, preprocessing, and model training |
| `darts.API.py` | Script | **API script** - Demonstrates Darts API usage with wrapper classes for each model type |
| `darts.API.md` | Docs | **API documentation** - Comprehensive guide to Darts library concepts, model configurations, and code examples |

### Utility Files

| File | Type | Description |
|------|------|-------------|
| `darts.utils.py` | Module | **Utility functions** - Reusable functions for data loading, feature engineering, model evaluation, and visualization |
| `requirements.txt` | Config | **Dependencies** - All required Python packages with versions |

### Docker Files

| File | Type | Description |
|------|------|-------------|
| `Dockerfile` | Docker | **Multi-stage build** - Optimized container with base, builder, production, and development stages |
| `docker-compose.yml` | Docker | **Compose config** - Services for dev (JupyterLab), prod (Notebook), script, and training |
| `.dockerignore` | Docker | **Ignore patterns** - Excludes unnecessary files from Docker build context |
| `scripts/docker-entrypoint.sh` | Script | **Entrypoint** - Container initialization script with data validation |

### Utility Functions in `darts.utils.py`

```python
# Data Loading & Preprocessing
load_energy_data(file_path)           # Load PJME dataset from CSV
handle_missing_timestamps(df)          # Fill missing timestamps with interpolation
create_darts_series(df, value_col)    # Convert DataFrame to Darts TimeSeries

# Feature Engineering
create_temporal_features(df)           # Add hour, day, month, season features
add_lag_features(df, target_col, lags) # Add lagged value features
add_rolling_features(df, target_col)   # Add rolling statistics features

# Model Evaluation
evaluate_forecast(actual, predicted)   # Calculate MAPE, RMSE, MAE, SMAPE
compare_models(model_results)          # Create comparison summary DataFrame

# Visualization
plot_time_series(series, title)        # Plot TimeSeries
plot_seasonality_analysis(df)          # Plot hourly, daily, monthly patterns
plot_predictions_vs_actual(actual, predictions)  # Compare model outputs
plot_error_analysis(actual, predicted) # Analyze prediction errors

# Train/Test Split
train_test_split_series(series, test_size)  # Split TimeSeries
scale_series(train, test)              # Scale data for neural networks
```

---

## Dataset

**PJME Hourly Energy Consumption**

| Attribute | Value |
|-----------|-------|
| **Source** | [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) |
| **Region** | PJM East (Pennsylvania-New Jersey-Maryland Interconnection) |
| **Frequency** | Hourly |
| **Time Range** | 2002-2018 |
| **Records** | ~145,000 hourly observations |
| **Target Variable** | Energy consumption in Megawatts (MW) |

**Data Characteristics:**
- Strong daily seasonality (24-hour cycle)
- Weekly seasonality (weekday vs. weekend patterns)
- Yearly seasonality (summer peaks, winter variations)
- Some missing timestamps requiring interpolation

---

## Architecture

### Pipeline Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Raw Data   │ ──▶ │ Load & Parse│ ──▶ │Handle Missing│
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│Create Series│ ◀── │Feature Eng. │ ◀── │ TimeSeries  │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│Train/Test   │ ──▶ │ Scale Data  │ ──▶ │Train Models │
│   Split     │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Best Model  │ ◀── │  Compare    │ ◀── │  Evaluate   │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Key Classes (in `darts.example.py`)

| Class | Purpose |
|-------|---------|
| `ForecastConfig` | Configuration settings (paths, test size, epochs) |
| `DataPipeline` | Data loading, preprocessing, train/test split |
| `ModelTrainer` | Train and evaluate multiple forecasting models |
| `HyperparameterTuner` | Grid search for model optimization |

---

## Models Implemented

| Model | Type | Description | Key Parameters |
|-------|------|-------------|----------------|
| **Naive Seasonal** | Baseline | Repeats previous week's pattern | `K=168` (weekly) |
| **Exponential Smoothing** | Statistical | Classical model with seasonality | `seasonal_periods=24` |
| **Prophet** | ML | Facebook's additive model | `yearly/weekly/daily_seasonality` |
| **N-BEATS** | Deep Learning | Neural Basis Expansion | `input_chunk=168, stacks=10` |
| **LSTM** | Deep Learning | Recurrent Neural Network | `hidden_dim=64, layers=2` |

### Model Code Examples

```python
# Naive Seasonal (Baseline)
model = NaiveSeasonal(K=168)

# Exponential Smoothing
model = ExponentialSmoothing(seasonal_periods=24, seasonal='add')

# Prophet
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True
)

# N-BEATS
model = NBEATSModel(
    input_chunk_length=168,
    output_chunk_length=24,
    num_stacks=10,
    num_layers=4,
    n_epochs=50
)

# LSTM
model = RNNModel(
    model='LSTM',
    input_chunk_length=168,
    output_chunk_length=24,
    hidden_dim=64,
    n_rnn_layers=2
)
```

---

## Feature Engineering

### Temporal Features

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
| `season` | Season encoding (0=Winter, 1=Spring, 2=Summer, 3=Fall) |

### Lag Features

| Feature | Description |
|---------|-------------|
| `lag_1h` | Energy consumption 1 hour ago |
| `lag_24h` | Energy consumption 24 hours ago |
| `lag_48h` | Energy consumption 48 hours ago |
| `lag_168h` | Energy consumption 1 week ago |

### Rolling Features

| Feature | Description |
|---------|-------------|
| `rolling_mean_24h` | 24-hour rolling average |
| `rolling_std_24h` | 24-hour rolling standard deviation |
| `rolling_mean_168h` | 1-week rolling average |

---

## Hyperparameter Tuning

Grid search is performed on N-BEATS with the following parameter ranges:

| Parameter | Values Tested | Description |
|-----------|--------------|-------------|
| `input_chunk_length` | [72, 168] | Lookback window (3 days or 1 week) |
| `output_chunk_length` | [24, 48] | Forecast horizon (1-2 days) |
| `num_stacks` | [5, 10] | Model capacity |
| `num_layers` | [2, 4] | Layers per block |
| `layer_widths` | [128, 256] | Hidden layer width |

**Tuning Process:**
1. Split training data into train/validation sets
2. Train models with reduced epochs (20) for each combination
3. Evaluate on validation set using MAPE
4. Select parameters with lowest MAPE
5. Retrain final model with optimal parameters

---

## Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **MAPE** | Mean Absolute Percentage Error | Percentage error (lower is better) |
| **RMSE** | Root Mean Squared Error | Absolute error in MW |
| **MAE** | Mean Absolute Error | Average absolute error |
| **SMAPE** | Symmetric MAPE | Symmetric percentage error |

**Expected Performance:**
- Best models typically achieve **MAPE < 5%**
- Deep learning models (N-BEATS, LSTM) generally outperform statistical methods
- Prophet provides strong baseline with automatic seasonality detection

---

## Installation

### Prerequisites
- Python 3.9+
- pip or conda

### Steps

```bash
# 1. Navigate to project directory
cd /Users/manan/Documents/Darts

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Download Dataset

1. Visit [Kaggle PJME Dataset](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
2. Download `PJME_hourly.csv`
3. Place it in the `data/` folder

---

## Docker Setup

Run the project in a containerized environment using Docker.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed

### Quick Start

```bash
# 1. Build and start development environment with JupyterLab
docker-compose up dev

# 2. Open browser at http://localhost:8888
```

### Available Services

| Service | Command | Description |
|---------|---------|-------------|
| **dev** | `docker-compose up dev` | JupyterLab with hot reload (development) |
| **prod** | `docker-compose up prod` | Jupyter Notebook (production) |
| **script** | `docker-compose run --rm script` | Run `darts.example.py` directly |
| **train** | `docker-compose --profile train up train` | Long-running training with resource limits |

### Docker Commands

```bash
# Build the image
docker-compose build

# Start development environment (JupyterLab)
docker-compose up dev

# Start production environment (Jupyter Notebook)
docker-compose up prod

# Run the Python script
docker-compose run --rm script

# Run with GPU support (if available)
docker-compose up dev  # Uses "accelerator": "auto"

# Stop all services
docker-compose down

# Remove all containers and volumes
docker-compose down -v

# Rebuild from scratch
docker-compose build --no-cache
```

### Volume Mounts

| Host Path | Container Path | Description |
|-----------|----------------|-------------|
| `./data` | `/app/data` | Dataset files |
| `./output` | `/app/output` | Generated outputs |
| `./models` | `/app/models` | Saved model files |
| `.` (dev only) | `/app` | Source code (live editing) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JUPYTER_ENABLE_LAB` | `yes` | Enable JupyterLab interface |
| `PYTHONPATH` | `/app` | Python module path |

### Building Custom Image

```bash
# Build production image only
docker build --target production -t darts-forecast:prod .

# Build development image
docker build --target development -t darts-forecast:dev .

# Run standalone container
docker run -p 8888:8888 -v $(pwd)/data:/app/data darts-forecast:dev
```

### Dockerfile Stages

| Stage | Description |
|-------|-------------|
| `base` | Python 3.10 with system dependencies |
| `builder` | Installs Python packages in virtual environment |
| `production` | Minimal image with Jupyter Notebook |
| `development` | Full image with JupyterLab + dev tools |

---

## Usage

### Option 1: Interactive Notebook (Recommended)

```bash
jupyter notebook darts.example.ipynb
```

### Option 2: Python Script

```bash
python darts.example.py
```

### Option 3: Using Classes Programmatically

```python
from darts.example import ForecastConfig, DataPipeline, ModelTrainer

# Configure pipeline
config = ForecastConfig()
config.test_size = 24 * 14  # 2 weeks test

# Initialize components
data_pipeline = DataPipeline(config)
trainer = ModelTrainer(config)

# Load and prepare data
data_pipeline.load_data()
series = data_pipeline.create_time_series()
train, test = data_pipeline.split_data()
train_scaled, test_scaled = data_pipeline.scale_data(train, test)

# Train models
trainer.train_prophet(train, test)
trainer.train_nbeats(train_scaled, test, data_pipeline.scaler)

# Get comparison
results = trainer.get_comparison_summary()
print(results)
```

### Option 4: Using Utility Functions

```python
import darts.utils as utils

# Load data
df = utils.load_energy_data('data/PJME_hourly.csv')

# Feature engineering
df = utils.create_temporal_features(df)
df = utils.add_lag_features(df)
df = utils.add_rolling_features(df)

# Create TimeSeries
series = utils.create_darts_series(df)

# Evaluate forecast
metrics = utils.evaluate_forecast(actual, predicted, "My Model")
```

---

## API Reference

### Darts Core Classes

| Class | Import | Description |
|-------|--------|-------------|
| `TimeSeries` | `from darts import TimeSeries` | Fundamental time series data structure |
| `Scaler` | `from darts.dataprocessing.transformers import Scaler` | Scale time series data |

### Model Classes

| Model | Import |
|-------|--------|
| `NaiveSeasonal` | `from darts.models import NaiveSeasonal` |
| `ExponentialSmoothing` | `from darts.models import ExponentialSmoothing` |
| `Prophet` | `from darts.models import Prophet` |
| `NBEATSModel` | `from darts.models import NBEATSModel` |
| `RNNModel` | `from darts.models import RNNModel` |

### Metrics

| Metric | Import |
|--------|--------|
| `mape` | `from darts.metrics import mape` |
| `rmse` | `from darts.metrics import rmse` |
| `mae` | `from darts.metrics import mae` |
| `smape` | `from darts.metrics import smape` |

---

## References

### Libraries & Tools
- **Darts Library:** https://unit8co.github.io/darts/
- **Darts GitHub:** https://github.com/unit8co/darts
- **Prophet:** https://facebook.github.io/prophet/

### Papers
- **Darts:** Herzen et al. (2022). "Darts: User-Friendly Modern Machine Learning for Time Series" JMLR 23(124):1−6
- **N-BEATS:** Oreshkin et al. (2019). "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"
- **Prophet:** Taylor & Letham (2018). "Forecasting at Scale"

### Data
- **Dataset:** https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
- **PJM Interconnection:** https://www.pjm.com/

---
