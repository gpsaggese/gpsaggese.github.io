# Project 3: Traffic Flow Anomaly Detection

**Project Tag:** `Fall2025_Traffic_Flow_Anomaly_Detection`
**Branch:** `UmdTask46_Fall2025_auto_sklearn_Traffic_Flow_Anomaly_Detection`

---

## **Project Status: Completed**


This directory contains the complete code implementation, Docker environment, and results for the Traffic Flow Anomaly Detection project.

* **Docker:** Fully built and optimized for Apple Silicon (Mac M-series) using Python 3.9.
* **Analysis:** `auto-sklearn` regression implemented with a **Hybrid Anomaly Detection** strategy.
* **Dashboard:** An interactive Streamlit app (`dashboard.py`) is included for real-time visualization, heatmaps, and weather correlation.

---

## Project Overview

The goal of this project is to use **Auto-Sklearn** (Automated Machine Learning) to detect anomalies in the Metro Interstate Traffic Volume dataset.

Unlike traditional unsupervised methods (which struggle with complex seasonality), this project treats anomaly detection as a **regression problem**. We train a model to learn "normal" traffic patterns based on time and weather. Anomalies are defined as deviations from this learned normality.

The `auto-sklearn` model's performance is compared against two traditional baselines:
1.  **Isolation Forest** (Unsupervised)
2.  **One-Class SVM** (Unsupervised)

---

## Methodology & Hybrid Detection

This project follows the structure defined in `autosklearn.API.md`.

### 1. The Hypothesis
Normal traffic is predictable (cyclical). Anomalies are unpredictable. Therefore, a strong regression model should accurately predict normal traffic, and fail (produce high errors) on anomalies.

### 2. The "Hybrid" Detection Logic
Standard residual analysis initially failed (produced 0.0 Recall) because `auto-sklearn` was *too* accurate—it correctly predicted low-traffic events. To fix this, we implemented a **Hybrid Strategy**:

* **Criteria A (Model Failure):** The prediction error (Residual) is in the top 3% (High Error).
* **Criteria B (Predicted Low Value):** The model correctly predicts an extremely low traffic event (Bottom 3%).

**Definition:** `Anomaly = (High Residual) OR (Predicted Low Value)`

---

## Setup & Installation (Mac M4 Optimized)

### 1. Build the Docker Image
We use a custom `Dockerfile` based on `python:3.9-slim` to ensure compatibility with `auto-sklearn` and Numpy < 2.0.0 on Apple Silicon.

```bash
docker build -t autosklearn-project .
````

### 2\. Run the Container

This command mounts your current folder to `/data` and exposes ports for Jupyter (8888) and Streamlit (8501).

```bash
docker run --rm -p 8888:8888 -p 8501:8501 -v "$(pwd)":/data autosklearn-project
```

-----

## How to Run

### Step 1: Execute the Analysis (Jupyter)

1.  Copy the URL printed in the terminal (e.g., `http://127.0.0.1:8888/?token=...`) and open it in your browser.
2.  Open **`autosklearn.example.ipynb`**.
3.  Click **Run -\> Run All Cells**.
      * *Note:* The training step takes **15 minutes**.
      * *Note:* `n_jobs=1` is used to prevent multiprocessing crashes on Mac.

### Step 2: Launch the Dashboard (Streamlit)

While the container is running, open a **new terminal window** on your computer and run:

1.  Find your container ID:
    ```bash
    docker ps
    ```
2.  Launch the dashboard inside that container:
    ```bash
    docker exec -it <CONTAINER_ID> streamlit run dashboard.py
    ```
3.  Open **http://localhost:8501** in your browser.

-----

## Final Results

The table below shows the performance of the models. The **Hybrid Auto-Sklearn** approach significantly outperformed the unsupervised baselines.

| Model | Precision | Recall | F1-Score |
|:---|---:|---:|---:|
| Isolation Forest | 0.05 | 0.06 | 0.06 |
| One-Class SVM | 0.02 | 0.53 | 0.05 |
| **Auto-Sklearn (Hybrid)** |0.27 | 0.56 | 0.37 |


### Analysis of Results

  * **Baselines Failed:** Isolation Forest and SVM struggled because they lack context. They flagged *any* statistical outlier as an anomaly, even if it was just normal midnight traffic (High False Positives).
  * **Auto-Sklearn Succeeded:** By using regression, the model understood the context (Time of Day + Weather). The Hybrid Logic allowed it to capture both unexpected events (Accidents) and expected anomalies (Road Closures/Zero Traffic).

-----

## File Structure

  * **`autosklearn_utils.py`**: Helper functions for loading data and calculating residuals.
  * **`autosklearn.example.ipynb`**: The main analysis notebook (Training & Evaluation).
  * **`dashboard.py`**: The interactive Streamlit application (Heatmaps & Live Simulation).
  * **`Dockerfile`**: Configuration for the reproducible Python 3.9 environment.
  * **`autosklearn.API.md`**: The API contract documentation.
  * **`anomaly_plot.png`**: Visualization of detected anomalies vs. actual traffic.
