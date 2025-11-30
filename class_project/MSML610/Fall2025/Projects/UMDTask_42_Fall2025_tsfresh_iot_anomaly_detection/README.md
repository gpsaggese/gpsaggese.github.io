# IoT Anomaly Detection for Smart Manufacturing
PREMAL SHAH UID: 121293596 (tsfresh difficulty: 3)
Dataset: https://www.kaggle.com/datasets/ziya07/smart-manufacturing-iot-cloud-monitoring-dataset

Machine learning system for predictive maintenance in IoT-enabled manufacturing equipment using time-series feature extraction and ensemble learning.

**Course**: MSML610 Fall 2025
**Project**: TutorTask 42

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Docker Setup](#docker-setup)
- [Usage](#usage)
- [Results](#results)
- [Requirements](#requirements)

---

## Overview

This project implements an end-to-end anomaly detection system for smart manufacturing with:

- Real-time anomaly detection (99.97% accuracy)
- Predictive models for 1-2 day advance warning (99.8% detection rate)
- Multi-target classification for 5 different prediction targets
- Feature engineering pipeline with 165+ time-series features
- Docker-based deployment

### Dataset

- 100,000 IoT sensor readings from 50 machines
- 5 sensors: Temperature, Vibration, Humidity, Pressure, Energy Consumption
- 6 target variables for prediction

---

## Quick Start

### Installation

```bash
cd UMDTask_42_Fall2025_tsfresh_iot_anomaly_detection
pip install -r requirements.txt
```

### Run with Docker

```bash
cd docker
./docker_build.sh
./docker_jupyter.sh
```

Access Jupyter at http://localhost:8888

### Run Main Notebook

```bash
jupyter notebook iot_anomaly.example.ipynb
```

---

## Project Structure

```
UMDTask_42_Fall2025_tsfresh_iot_anomaly_detection/
├── iot_anomaly_utils.py          # Utility functions and API wrappers
├── iot_anomaly.API.md            # API documentation
├── iot_anomaly.API.ipynb         # API demonstration
├── iot_anomaly.example.md        # Example walkthrough
├── iot_anomaly.example.ipynb     # Main example (fully executed)
├── Dockerfile                     # Docker configuration
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── data/
│   ├── raw/
│   │   └── smart_manufacturing_data.csv
│   └── processed/
│       └── engineered/
│           └── features_sample_30k.csv
│
├── models/
│   ├── isolation_forest_anomaly.pkl
│   ├── failure_type_classifier.pkl
│   ├── maintenance_required_model.pkl
│   ├── machine_status_model.pkl
│   ├── downtime_risk_model.pkl
│   ├── all_targets_performance_summary.csv
│   └── registry/
│       └── model_registry.json
│
├── outputs/
│   └── figures/
│       ├── data_exploration.png
│       ├── isolation_forest_evaluation.png
│       ├── time_series_anomaly_visualization.png
│       ├── multi_sensor_anomaly_detection.png
│       ├── failure_type_classification.png
│       ├── multi_target_performance.png
│       └── realtime_dashboard.png
│
├── scripts/
│   ├── feature_engineering.py     # tsfresh-based feature extraction
│   ├── train_predictive_models.py
│   ├── train_models_kfold.py
│   ├── train_lstm_xgb_ensemble.py
│   └── create_visualizations.py
│
└── docker/
    ├── requirements.txt
    ├── docker_build.sh
    ├── docker_bash.sh
    └── docker_jupyter.sh
```

---

## Technical Details

### Feature Engineering

Uses **tsfresh library** for automated time-series feature extraction from raw sensor data:

- Comprehensive feature extraction via tsfresh.extract_features()
- MinimalFCParameters for fast extraction (default)
- ComprehensiveFCParameters for extensive features (optional)
- Temporal encodings: cyclical hour/day features
- Automatic imputation of missing values
- Feature extraction per machine and sensor

### Models

**Real-time Detection**:
- Random Forest Classifier (300 estimators)
- SMOTE balancing for class imbalance
- StandardScaler normalization

**Predictive Models**:
- Gradient Boosting Classifier (200 estimators)
- Forward-looking target labels (24h and 48h horizons)
- Temporal train/test split

### Implementation

All reusable code is in `iot_anomaly_utils.py`:
- Data loading and validation
- Feature engineering functions
- Model training and evaluation
- Visualization helpers
- Model persistence

---

## Docker Setup

### Build the Image

```bash
cd docker
./docker_build.sh
```

### Run the Container

```bash
./docker_bash.sh
```

### Start Jupyter

```bash
./docker_jupyter.sh
```

---

## Usage

### Load and Use Pre-trained Models

```python
from iot_anomaly_utils import load_model

# Load model
model = load_model('models/isolation_forest_anomaly.pkl')
scaler = load_model('models/scaler.pkl')

# Predict
X_scaled = scaler.transform(features)
predictions = model.predict(X_scaled)
```

### Feature Engineering

```python
from iot_anomaly_utils import (
    load_iot_data,
    compute_basic_features,
    compute_rolling_features
)

# Load data
df = load_iot_data('data/raw/smart_manufacturing_data.csv')

# Engineer features
sensors = ['temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption']
df = compute_basic_features(df, sensors)
df = compute_rolling_features(df, sensors, windows=[6, 12, 24])
```

### Train New Models

```python
from iot_anomaly_utils import train_anomaly_detector, evaluate_model

# Train
model, scaler = train_anomaly_detector(X_train, y_train)

# Evaluate
metrics = evaluate_model(y_true, y_pred)
```

### Predictive Analysis

```python
# Load 48-hour predictor
predictor = load_model('models/predictive/anomaly_predictor_48h.pkl')
scaler = load_model('models/predictive/scaler_48h.pkl')

# Predict
predictions = predictor.predict(scaler.transform(features))
probabilities = predictor.predict_proba(scaler.transform(features))[:, 1]
```

---

## Results

### Note on Model Performance

The dataset used is synthetically generated with clear feature separation, resulting in unrealistically high accuracy (99.97%). A simple threshold on temperature alone achieves 96% accuracy. In production IoT systems, anomaly detection typically achieves 85-92% accuracy due to sensor noise, edge cases, and subtle failure patterns. The high performance here demonstrates the methodology works correctly, but real-world deployment would require more challenging data with realistic noise and overlap.

### Real-Time Detection

| Model | Accuracy | F1 Score | Type |
|-------|----------|----------|------|
| Anomaly Detection | 99.97% | 99.81% | Binary |
| Failure Type | 91.93% | 88.13% | 5-class |
| Maintenance Required | 89.05% | 87.34% | Binary |
| Machine Status | 79.93% | 71.19% | 3-class |
| Downtime Risk | 99.97% | 99.97% | Binary |

Average Accuracy: 92.17%

### Predictive Models

| Model | Accuracy | Detection Rate | Lead Time |
|-------|----------|----------------|-----------|
| 24h Prediction | 92.03% | 99.9% | 1 day |
| 48h Prediction | 98.86% | 99.8% | 2 days |

### Confusion Matrices

**Anomaly Detection (Real-time)**:
```
              Predicted
            No    Anomaly
Actual No   5460      2
       Yes     0    538
```

**48h Prediction**:
```
              Predicted
            No    Anomaly
Actual No      3     194
       Yes    34  19769
```

---

## Requirements

```
Python >= 3.8
numpy >= 1.19.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
imbalanced-learn >= 0.10.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
jupyter >= 1.0.0
joblib >= 1.0.0
```

See `requirements.txt` for complete dependencies.

---

## Files Description

### Core Files

- `iot_anomaly_utils.py`: Reusable utility functions and API wrappers
- `iot_anomaly.API.ipynb`: Demonstrates native API and wrapper layer
- `iot_anomaly.API.md`: API documentation
- `iot_anomaly.example.ipynb`: Complete end-to-end example (39 cells, fully executed)
- `iot_anomaly.example.md`: Detailed walkthrough

### Scripts

- `scripts/feature_engineering.py`: Feature extraction pipeline
- `scripts/train_predictive_models.py`: Predictive model training

---

## Project Information

**Course**: MSML610 - Fall 2025
**Institution**: University of Maryland
**Project**: UMDTask 42 - IoT Anomaly Detection
**Last Updated**: November 28, 2025
