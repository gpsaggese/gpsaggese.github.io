# Renewable Energy Production Forecasting

---

## Overview

This project implements an **end-to-end renewable energy forecasting system** for predicting **hourly solar energy production** using historical generation data, weather variables, and time-series features.  

The goal of the project is twofold:

1. **Technical**: Build and evaluate machine learning and deep learning models for time-series forecasting.
2. **Educational**: Provide a **hands-on tutorial-style project** demonstrating how modern ML tooling (MLflow, Docker, Streamlit) fits into a real-world data science workflow.

The project is designed following the **MSML610 / DATA605 class project guidelines**, with a clear separation between:
- **Reusable API utilities**
- **Runnable example workflows**
- **Experiment tracking and deployment**

---

## Why Solar Energy Forecasting Matters

Solar energy production is inherently variable due to:
- Weather conditions (cloud cover, temperature, wind)
- Daily and seasonal cycles
- Sudden environmental changes

Accurate forecasting enables:

- **Grid stability** by balancing supply and demand  
- **Efficient energy storage planning**  
- **Reduced reliance on fossil-fuel backup generation**  
- **Higher penetration of renewable energy sources**  

This project demonstrates how **machine learning models** can learn temporal and environmental patterns to produce accurate short-term solar energy forecasts.

---

## Key Outcomes

- A **RandomForestRegressor** provides the strongest performance and serves as the production model.
- **LSTM** and **GRU** models successfully capture temporal dependencies but require more data to outperform tree-based methods.
- A fully reproducible workflow using **Docker** and **MLflow**.
- A **Streamlit application** for interactive prediction.
- A **live prediction script** (bonus) demonstrating real-time inference using external weather data.

---

## Project structure

```text
UmdTask15_Fall2025_Renewable_Energy_Production/
├── README.md
├── Dockerfile
├── compose.yaml
├── RenewableEnergy_utils.py
├── api/
│   ├── API.md
│   └── API.ipynb
├── examples/
│   ├── example.md
│   └── example.ipynb
├── scripts/
│   ├── make_features.py
│   ├── train.py
│   ├── live_predict.py
│   └── openweather_client.py
├── app/
│   └── streamlit_app.py
```
----

## Dataset Description

The dataset consists of **hourly solar energy production** with associated weather variables:

| Column           | Description                         |
|------------------|-------------------------------------|
| energy_mwh       | Solar energy output (target)        |
| temp_c           | Temperature (°C)                    |
| cloud_cover      | Cloud coverage (%)                  |
| solar_radiation  | Solar irradiance (W/m²)             |
| wind_speed       | Wind speed (m/s)                    |

---

## Feature Engineering

Feature engineering is implemented in **`RenewableEnergy_utils.py`** via `make_basic_time_features()`.

### Generated Features
- **Calendar features**
  - Hour of day
  - Day of week
  - Month
- **Lag features**
  - 1-hour, 2-hour, 24-hour lags
- **Rolling statistics**
  - 3-hour rolling mean
  - 24-hour rolling mean

This design captures:
- Diurnal solar cycles
- Short-term persistence
- Daily seasonal patterns

### Generate Features
```bash
python3 scripts/make_features.py
data/processed/train.csv
```
## Modeling Approach

### 1. Random Forest Regressor (Production Model)
- Strong baseline for tabular time-series data  
- Interpretable via feature importance  
- Achieves the best overall accuracy  

### 2. LSTM (Deep Learning)
- Sequence-based neural network  
- Uses a 24-hour sliding window  
- Captures temporal dependencies in solar production  

### 3. GRU (Deep Learning)
- Similar to LSTM but computationally lighter  
- Comparable performance on this dataset  

All models use a **time-aware train/validation split**, holding out the **last 7 days** of data for validation to simulate real forecasting conditions.

---

## Model Performance

| Model          | RMSE | MAE  | R²    |
|----------------|------|------|-------|
| Random Forest  | 0.51 | 0.39 | 0.997 |
| LSTM           | 3.37 | 2.07 | 0.90  |
| GRU            | 3.36 | 2.16 | 0.90  |

The **Random Forest model** is selected as the **production model** for deployment and serving.

---

## MLflow Experiment Tracking

All experiments are tracked using **MLflow**, including:

- Hyperparameters  
- Evaluation metrics (RMSE, MAE, R²)  
- Feature importance scores  
- Diagnostic plots (loss curves, predictions)  
- Serialized model artifacts  

### Start MLflow UI

```bash
mlflow ui \
  --backend-store-uri file:./mlruns \
  --host 0.0.0.0 \
  --port 5000
```
http://localhost:5001

### Dockerized Environment
The project runs entirely inside Docker to ensure full reproducibility across systems.
Build & Run the Container
```
docker run --rm -it \
  -p 8893:8890 \
  -p 5001:5000 \
  -p 8501:8501 \
  -p 1234:1234 \
  -v "$PWD":/work \
  -w /work/class_project/MSML610/Fall2025/projects/UmdTask15_Fall2025_Renewable_Energy_Production \
  umd-dev:latest \
  bash
```
### Training the Models
python3 scripts/train.py
This command:
Trains Random Forest, LSTM, and GRU models
Logs all metrics and artifacts to MLflow
Saves models with proper input signatures for serving
Streamlit Application
The Streamlit application provides an interactive solar energy forecasting dashboard.
Run the App
```
streamlit run app/streamlit_app.py \
  --server.port=8501 \
  --server.address=0.0.0.0
```
Open in browser:
http://localhost:8501

### Application Features
User-controlled weather and time inputs
Real-time energy production prediction
Feature importance visualization
Recent production trend plots


#### API (api/API.*)
Demonstrates the stable internal programming interface
Uses reusable functions from RenewableEnergy_utils.py
Shows how to:
Load data
Engineer features
Perform time-series splits
Train a baseline model
Contains no heavy runtime logic or deployment code

### Example (examples/example.*)
End-to-end runnable workflow
Includes:
Exploratory Data Analysis (EDA)
Feature engineering
Model training
Visualization of results
Represents a realistic application use case

### Bonus: Live Prediction (Real-Time Inference)
A bonus live prediction pipeline demonstrates real-time inference using external weather data.
Components
openweather_client.py
Fetches real-time weather data from the OpenWeather API
live_predict.py
Builds a feature vector
Sends an HTTP request to the MLflow model server
Receives and prints the prediction

### Run MLflow Model Serving
```mlflow models serve \
  -m runs:/<RUN_ID>/model \
  -p 1234
```
### Docker Compose (Bonus)
The included compose.yaml orchestrates:
MLflow UI
Streamlit application
Model serving endpoint
This allows the full system to be launched with a single command:
docker compose up

### Summary
## Project Highlights

- Clean and well-documented **API layer** for model inference  
- Complete **end-to-end example workflow**, from data preprocessing to prediction  
- Strong **forecasting performance** using engineered temporal and weather features  
- **MLflow-based experiment tracking** for reproducibility and model comparison  
- Fully **Dockerized pipeline** ensuring consistent environments across systems  
- Interactive **Streamlit dashboard** for visualization and live predictions  
- **Bonus real-time inference pipeline** integrating live weather data  
- Demonstrates practical **machine learning, MLOps, and software engineering** best practices aligned with modern data science workflows  
