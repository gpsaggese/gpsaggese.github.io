Airport Congestion Level Prediction

MSML 610 – Final Project
Author: Varun Parashar

⸻

Overview

This repository contains a complete machine learning pipeline to predict hourly airport congestion levels (Low, Medium, High) using the U.S. DOT Airline On-Time Performance Dataset.

The project includes:
    •    Data preprocessing
    •    Feature engineering
    •    XGBoost multi-class classifier
    •    Jupyter notebooks for EDA and modeling
    •    Streamlit dashboard for interactive visualization

⸻

Repository Structure

 project_root/
│
├── data/
│   ├── raw/                     # flights.csv, airlines.csv, airports.csv
│   ├── processed/               # hourly_congestion.csv
│   └── models/                  # model.pkl
│
├── src/
│   ├── preprocess_hourly.py     # Create hourly congestion dataset
│   └── train_model.py           # Train and save XGBoost model
│
├── app/
│   └── app.py                   # Streamlit dashboard
│
└── notebooks/
    ├── 01_eda_flights.ipynb
    ├── 02_feature_engineering.ipynb
    └── 03_training_xgboost.ipynb
    
    
Dataset

Source:
U.S. DOT Airline On-Time Performance Dataset (Kaggle)

Files Required:
    •    flights.csv
    •    airlines.csv
    •    airports.csv

Place these inside:  data/raw/

Installation

Clone the repository

git clone https://github.com/Varun-22/umd_classes
cd umd_classes

Create a virtual environment (recommended)

python3 -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows

Install dependencies

If using a requirements file:

pip install -r requirements.txt

1. Preprocessing

Generates hourly congestion data by computing:
    •    Departures per hour
    •    Arrivals per hour
    •    Total flights per hour
    •    Congestion level label

Run the preprocessing script: python3 src/preprocess_hourly.py

Output:  data/processed/hourly_congestion.csv

2. Model Training

Trains an XGBoost classifier using engineered features.

Run:  python3 src/train_model.py

Output: data/models/model.pkl

3. Streamlit Dashboard

Launch the dashboard:

streamlit run app/app.py

Features:
    •    Airport selection
    •    Date selection
    •    Predicted hourly congestion levels
    •    Peak congestion hour
    •    Interactive bar chart
    •    Flight details table

⸻

Notebooks

Three Jupyter notebooks document the full workflow:
    1.    01_eda_flights.ipynb – dataset exploration
    2.    02_feature_engineering.ipynb – build hourly dataset
    3.    03_training_xgboost.ipynb – model fitting and evaluation

⸻

Model Details
    •    Algorithm: XGBoost (multi-class classification)
    •    Label Encoding:
    •    Low = 0
    •    Medium = 1
    •    High = 2
    •    Airport encoded using OneHotEncoder
    •    Evaluation metrics: precision, recall, F1-score

⸻

Author

Varun Parashar
Master of Science in Artificial Intelligence
University of Maryland, College Park
