<!-- toc -->

- [Project Files](#project-files)
- [Setup and Dependencies](#setup-and-dependencies)
  * [Building and Running the Docker Container](#building-and-running-the-docker-container)
    + [Environment Setup](#environment-setup)

<!-- tocstop -->

# Electricity Consumption Forecasting with LSTMs & Keras Tuner

- **Team Members:**  
  - Mohit Saluru  
  - Harshit Vinay Gadge  

- **Course:** MSML 610 — Fall 2025  
- **Date:** 2025-03-15  

This project implements **hourly electricity consumption forecasting** using:

- LSTM neural networks  
- Hyperparameter tuning with **Keras Tuner**  
- Multi-step forecasting  
- Prophet baseline comparison  
- A clean API layer + demonstration notebooks  

The dataset used is the **AEP Hourly Electricity Load** from PJM (Kaggle).

---

# Project Files

This directory contains the following major components.

## API Layer Files

- **electricity_forecast_utils.py**  
  Utility module containing:
  - Data preparation & resampling  
  - Sliding-window generation  
  - Scaling helpers  
  - LSTM model builders (baseline + tuned)  
  - Training helpers with early stopping  
  - Inverse-scaling + evaluation utilities  

- **keras_tuner_API.ipynb**  
  Notebook demonstrating:
  - Native Keras API  
  - Keras Tuner API  
  - How to use utility functions

- **keras_tuner_API.md**  
  Markdown describing the API layer and exposed functions.

## Example (End-to-End Project)

- **keras_tuner_example.ipynb**  
  Full workflow including:
  - data cleaning  
  - window generation  
  - baseline LSTM model  
  - hyperparameter tuning  
  - tuned LSTM model  
  - multi-step forecasting  
  - Prophet baseline  
  - evaluation + metrics  
  - comparison table + plots  

- **keras_tuner_example.md**  
  Markdown summary of the full project.

## Supporting Files

- **data_preprocessing.py** — Additional preprocessing helpers  
- **sequence_model.py** — Early experiment script  

## Docker & Environment

- **Dockerfile**  
  Defines reproducible environment with:
  - Python 3.10  
  - TensorFlow (CPU)  
  - Keras Tuner  
  - Prophet  
  - Jupyter Lab  

- **docker-compose.yml**  
  Allows running the environment via:
  ```
  docker compose up --build
  ```

- **requirements.txt**  
  Python dependencies.

## Data & Results

- **data/** — folder containing `AEP_hourly.csv`  
- **tuner_results/** — Keras Tuner output directory (ignored in git)

---

# Setup and Dependencies

You may run this project in two ways:

1. **Docker (recommended)**  
2. **Local Python environment**

---

## Building and Running the Docker Container

From the top-level repo directory:

```
cd $GIT_ROOT
```

### Build the Docker image

```
docker compose up --build
```

This will:

- build the image  
- mount your project into `/workspace`  
- launch Jupyter Lab on port **8888**

### Launch Jupyter Notebook

Once running, open:

```
http://localhost:8888
```

Run:

- `keras_tuner_API.ipynb`  
- `keras_tuner_example.ipynb`  

---

## Environment Setup

This project **does not require external API keys**.

Optional TensorFlow noise reduction:

```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
```

To install dependencies locally (no Docker):

```
pip install -r requirements.txt
```

---

# End of README
