# Renewable Energy Production Forecasting  

---

## Overview

This project develops an end-to-end machine learning system for forecasting hourly solar energy production using historical weather and generation data. Accurate forecasting is essential for grid stability, energy storage planning, and large-scale integration of renewable energy into power systems.

The project includes:

- Data ingestion and feature engineering  
- Classical and deep learning forecasting models  
- Time-series modeling with lagged and rolling features  
- MLflow experiment tracking and model logging  
- A Dockerized, reproducible development environment  
- A Streamlit application for interactive predictions  
- API and example notebooks documenting code usage  


---
## Forecasting Purpose and Significance

Solar energy production is highly variable because it depends on weather conditions and natural cycles such as time of day and seasonality. This variability makes it difficult for power systems to rely on renewable energy unless accurate forecasts are available. Forecasting plays a central role in ensuring that solar energy can be integrated into the electrical grid in a stable and efficient manner.

Accurate solar forecasting supports several key objectives:

- **Grid Stability:** Power system operators must match supply and demand in real time. Reliable forecasts help prevent shortages and overloads.
- **Energy Storage Planning:** Battery systems can be charged or discharged more effectively when future production is known.
- **Operational Efficiency:** Solar plant operators can plan maintenance and deployment more intelligently.
- **Cost Reduction:** Forecasting reduces the need for backup fossil-fuel generation, lowering operational costs.
- **Renewable Integration:** Improved predictability enables higher penetration of renewable energy into the grid.

This project demonstrates how machine learning and deep learning methods can model the relationship between weather conditions, temporal patterns, and historical production. By training and comparing multiple models, the system identifies which approaches provide the most accurate estimates of short-term solar output.

### Project Outcomes

Through experimentation and evaluation, this project achieves the following:

- The **RandomForestRegressor** delivers the strongest forecasting performance, achieving an RMSE near 0.51 and an R² of approximately 0.9976. It serves as the primary production model.
- **LSTM** and **GRU** neural networks effectively model sequential behavior and daily cycles, although they require more data to outperform the Random Forest on this dataset.
- The system successfully predicts hourly solar energy output using engineered features such as weather variables, lagged production values, and rolling averages.
- The accompanying **Streamlit application** enables users to interactively adjust weather conditions and immediately view forecasted energy values.

Overall, this project provides a complete, realistic forecasting solution that reflects the challenges and requirements of real-world renewable energy operations.


## Project Structure
UmdTask15_Fall2025_Renewable_Energy_Production/
│
├── README.md
├── Dockerfile
├── RenewableEnergy_utils.py
│
├── api/
│ ├── API.md
│ └── API.ipynb
│
├── examples/
│ ├── example.md
│ └── example.ipynb
│
├── scripts/
│ ├── make_features.py
│ └── train.py
│
├── app/
│ └── streamlit_app.py
│
└── (not committed to Git)
├── data/
├── mlruns/
├── artifacts/


---

## Dataset Description

The raw dataset contains hourly observations of solar energy production along with weather variables:

| Column           | Description                          |
|------------------|--------------------------------------|
| energy_mwh       | Solar energy output (target)         |
| temp_c           | Temperature (°C)                     |
| cloud_cover      | Cloud percentage                     |
| solar_radiation  | Solar irradiance (W/m²)              |
| wind_speed       | Wind speed (m/s)                     |

### Engineered Features

Additional features are created to capture temporal patterns and historical dependencies:

- Time features: hour, day of week, month  
- Lag features: 1-hour, 2-hour, 24-hour lags  
- Rolling averages: 3-hour and 24-hour windows  

The final feature dataset consists of 4296 rows and 13 columns.

---

## Feature Engineering

Feature engineering is implemented in `make_basic_time_features` within `RenewableEnergy_utils.py`.  
It generates time-based, lagged, and smoothed (rolling) features.  
Running the following command produces the processed dataset:

python3 scripts/make_features.py

The processed file is stored at:
data/processed/train.csv


---

## Modeling Approach

The project trains three forecasting models:

### 1. Random Forest Regressor
A strong baseline model for tabular data.  
This model achieves the best overall accuracy and acts as the production model.

### 2. Long Short-Term Memory Network (LSTM)
A deep learning model suited for sequence modeling.  
Uses a 24-hour sliding window as input.

### 3. Gated Recurrent Unit Network (GRU)
A recurrent deep learning model that performs similarly to the LSTM while being more efficient.

All models follow a time-based train/validation split, where the final seven days of data serve as the validation period.

---

## Model Performance Summary

| Model        | RMSE  | MAE   | R²       |
|--------------|-------|-------|----------|
| Random Forest | 0.51 | 0.39 | 0.9976   |
| LSTM         | 3.37  | 2.07 | 0.90     |
| GRU          | 3.36  | 2.16 | 0.90     |

The Random Forest model performs best and is used in the Streamlit application.

---

## MLflow Experiment Tracking

All model training runs are tracked using MLflow, including:

- Parameters  
- Metrics (RMSE, MAE, R²)  
- Plots (loss curves, actual vs. predicted)  
- Feature importances  
- Saved model artifacts  

### Launch MLflow UI

Inside Docker:

mlflow ui
--backend-store-uri file:./mlruns
--host 0.0.0.0
--port 5000


Open in a local browser:

http://localhost:5001


---

## Reproducible Environment (Docker)

The project is fully containerized. The Docker image installs Python, MLflow, TensorFlow (CPU), scikit-learn, Jupyter, and Streamlit.

### Run the Docker Environment

docker run --rm -it
-p 8893:8890
-p 5001:5000
-p 8501:8501
-v "$PWD":/work
-w /work/class_project/MSML610/Fall2025/Projects/UmdTask15_Fall2025_Renewable_Energy_Production
umd-dev:latest
bash

---

## Training the Models

### Generate Features

python3 scripts/make_features.py

### Train All Models

python3 scripts/train.py


All outputs are logged under `mlruns/`.

---

## Streamlit Application

A simple user interface is provided for real-time solar energy forecasting.

### Run the App

streamlit run app/streamlit_app.py
--server.port=8501
--server.address=0.0.0.0


Open in a local browser:

http://localhost:8501


### Application Features

- Input controls for weather and time conditions  
- Real-time prediction of solar energy production  
- Recent trend visualization  
- Feature importance chart  

---

## API and Example Notebooks

### API Notebook (`api/API.ipynb`, `api/API.md`)
Demonstrates how to use the core utilities as a lightweight API for:

- Loading raw data  
- Engineering features  
- Splitting datasets  
- Training a baseline model  
- Computing evaluation metrics  

### Example Notebook (`examples/example.ipynb`, `examples/example.md`)
Provides a complete workflow including:

- Exploratory data analysis  
- Feature engineering  
- Model training  
- Validation  
- Visualization of actual vs. predicted energy output  

---

## Summary

This project delivers a complete renewable energy forecasting pipeline with:

- Structured feature engineering  
- Classical and deep learning models  
- MLflow experiment tracking  
- Docker-based reproducibility  
- A Streamlit deployment interface  
- Well-documented API and example notebooks  

It demonstrates practical machine learning and MLOps skills for real-world energy forecasting tasks.






