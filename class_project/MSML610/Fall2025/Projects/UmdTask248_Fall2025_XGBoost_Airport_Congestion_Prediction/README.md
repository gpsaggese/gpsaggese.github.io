# Airport Congestion Level Prediction  
## Using XGBoost and Flight Operations Data

---

## Project Overview

Airports experience congestion when too many flights arrive and depart within a short time window.  
This project predicts **hourly airport congestion levels** at major U.S. airports using historical flight operations data and machine learning.

The final output is an **interactive web application** where users can:
- Select an airport
- Select a date
- View hourly congestion levels
- Identify peak congestion hours

---

## Project Objective

The main objectives of this project are:

- Aggregate flight-level data into **hourly airport summaries**
- Engineer traffic and delay-based features
- Train an **XGBoost multi-class classifier**
- Predict congestion levels: **Low, Medium, High**
- Visualize results through a **Streamlit web dashboard**

---

## Dataset Description

This project uses the **U.S. Airline On-Time Performance Dataset**.

### Raw files used (not included in GitHub due to size limits):

- `flights.csv` – Flight-level records (500MB+)
- `airlines.csv` – Airline metadata
- `airports.csv` – Airport metadata

Due to GitHub’s 100MB file size restriction, raw datasets are **excluded from the repository**.

---

UmdTask248_Fall2025_XGBoost_Airport_Congestion_Prediction/
│
├── app/
│   └── app.py
│       # Streamlit web application for hourly congestion visualization
│
├── src/
│   ├── preprocess_hourly.py
│   │   # Aggregates raw flight-level data into hourly airport summaries
│   │
│   └── train_model.py
│       # Trains the XGBoost multi-class congestion model
│
├── data/
│   ├── raw/
│   │   └── README.md
│   │       # Instructions for downloading Kaggle datasets
│   │       # (CSV files are ignored in GitHub due to size limits)
│   │
│   ├── processed/
│   │   └── hourly_congestion.csv
│   │       # Generated hourly dataset (ignored in Git)
│   │
│   └── models/
│       └── model.pkl
│           # Trained XGBoost model (ignored in Git)
│
├── notebooks/
│   ├── 01_eda_flights.ipynb
│   │   # Exploratory data analysis
│   │
│   ├── 02_feature_engineering.ipynb
│   │   # Feature creation and congestion logic
│   │
│   └── 03_training_xgboost.ipynb
│       # Model training and evaluation
│
├── .gitignore
│   # Prevents large datasets, models, and cache files from being committed
│
├── .dockerignore
│   # Excludes unnecessary files from Docker images
│
├── README.md
│   # Project overview and documentation
│
└── requirements.txt  # Python dependencies

---

## Explanation of Each Folder and File

### app/app.py — Streamlit Web Application

This file creates the **interactive web dashboard**.

Responsibilities:
- Loads processed hourly data
- Allows users to select an airport and date
- Displays hourly congestion levels
- Highlights peak congestion hour
- Updates results dynamically based on user input

---

### src/preprocess_hourly.py — Data Preprocessing

This script converts raw flight-level data into an **hourly congestion dataset**.

Steps performed:
1. Load raw CSV files
2. Extract date and hour from timestamps
3. Aggregate flights by airport and hour
4. Compute traffic and delay metrics
5. Label congestion levels (Low / Medium / High)
6. Save processed data to data/processed/hourly_congestion.csv

---

### src/train_model.py — Model Training

This script trains the machine learning model.

Steps:
1. Load processed hourly dataset
2. Encode congestion categories numerically
3. Train an XGBoost multi-class classifier
4. Evaluate performance metrics
5. Save trained model to data/models/model.pkl

---

## Notebooks

### 01_eda_flights.ipynb

- Initial exploration of flight data
- Distribution of delays
- Traffic volume across airports

---

### 02_feature_engineering.ipynb

- Creation of congestion-related features
- Delay thresholds
- Hourly aggregation analysis

---

### 03_training_xgboost.ipynb

- Model training experiments
- Hyperparameter tuning
- Performance evaluation

---

## data/raw/README.md

This file explains how to obtain the dataset.

Raw data files must be downloaded manually and placed in data/raw/

These files are ignored by GitHub.

---

## Git Ignore Files

### .gitignore

Prevents the following from being uploaded:
- Raw CSV datasets
- Processed data
- Model files
- Temporary files and caches

---

### .dockerignore

Excludes unnecessary files during Docker image creation:
- Data folders
- Notebooks
- Git metadata
- Cache files

---

## How the Website Works

1. User opens the Streamlit web app
2. Selects an airport
3. Selects a date
4. Application filters hourly data
5. Displays:
   - Hourly congestion levels
   - Peak congestion time
6. Results update instantly

---

## How to Run the Project Locally

### Step 1: Install dependencies pip install -r requirements.txt

---

### Step 2: Place raw datasets

data/raw/flights.csv
data/raw/airlines.csv
data/raw/airports.csv

---

### Step 3: Preprocess data

python src/preprocess_hourly.py

---

### Step 4: Train the model

python src/train_model.py

---

### Step 5: Run the Streamlit app

streamlit run app/app.py

---

## Output

- Hourly congestion predictions
- Interactive dashboard
- Clear visualization of peak congestion hours

---

## Key Takeaways

- Large datasets require careful preprocessing
- Hourly aggregation reveals congestion patterns
- XGBoost performs well for multi-class classification
- Streamlit enables rapid deployment of ML applications

---

## Author

**Varun Parashar**  
MSML610 — Fall 2025  
University of Maryland
