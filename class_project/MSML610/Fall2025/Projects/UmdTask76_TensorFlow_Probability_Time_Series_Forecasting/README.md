# Project 2: Time Series Forecasting with Probabilistic Models

## 📌 Project Overview
This project explores **Time Series Forecasting with Probabilistic Models** using **TensorFlow Probability (TFP)**.
The primary objective is to forecast **Air Quality (CO concentrations)** while accounting for the inherent uncertainty in predictions—something standard regression models cannot do.

This repository demonstrates how to build a custom **Gaussian Process (GP)** model to capture complex seasonal patterns and provide calibrated confidence intervals.

## ✅ Project Status: Completed (100%)

### **Phase 1: Data Preparation & EDA**
* **Dataset:** [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/air+quality) (Italian city sensor data).
* **Challenges Solved:**
    * Parsed non-standard timestamps (`DD/MM/YYYY` format).
    * Handled sensor errors (values marked as `-200`).
    * **Crucial Step:** Resampled irregular data to a strict **Hourly Frequency** and interpolated missing gaps to ensure a continuous timeline for the Gaussian Process.

### **Phase 2: Probabilistic Modeling**
* **Model:** Gaussian Process Regression using `tensorflow_probability`.
* **Kernel Design:** A composite kernel to capture real-world physics:
    * `ExponentiatedQuadratic` (RBF): For smooth, long-term trends.
    * `ExpSinSquared`: For the strong **24-hour daily seasonality** (traffic patterns).
* **Uncertainty:** The model outputs a full probability distribution, allowing us to visualize **95% Confidence Intervals** (Error Bands).

### **Phase 3: Performance Evaluation & Bonus**
* **Objective:** Compare the Probabilistic Model against a classical Time Series baseline.
* **Bonus Task:** Implemented **ARIMA (Order 24,1,0)** to benchmark performance.
* **Final Results:**
    * **ARIMA RMSE:** `1.4053`
    * **Gaussian Process RMSE:** `1.3922`
    * **Verdict:** The Gaussian Process **outperformed the baseline** while providing critical uncertainty estimates that ARIMA lacks.

---

## 📂 Repository Structure
This project follows a flat, modular structure designed for reusability.

```text
├── TFP_Time_Series.API.md        # Technical documentation of the GP Class
├── TFP_Time_Series.API.ipynb     # Demo notebook showing the tool's syntax on synthetic data
├── TFP_Time_Series.example.md    # Full project report (Methodology, Diagrams, Results)
├── TFP_Time_Series.example.ipynb # Main execution notebook (Data Prep -> Modeling -> Evaluation)
├── utils_tfp.py                  # Core Logic Library (Contains Data Loading, GP Class, and ARIMA Class)
├── README.md                     # Project Status and Instructions
├── requirements.txt              # Python dependencies
└── data/                         # Contains 'air_quality.csv'
```

---

## 🚀 How to Run

### 1. Install Dependencies
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

### 2. Run the Main Analysis (The "Example")
To see the full end-to-end analysis, including data cleaning, the Gaussian Process training, and the **ARIMA vs. GP comparison plot**:
* Open and run **`TFP_Time_Series.example.ipynb`**.

### 3. Test the Tool (The "API")
To see how the `TFP_GaussianProcess_Forecaster` class can be reused on other datasets (demonstrated on a synthetic sine wave):
* Open and run **`TFP_Time_Series.API.ipynb`**.

---

## 🛠 Tech Stack
* **TensorFlow Probability (TFP):** For Gaussian Process layers and kernels.
* **TensorFlow:** Backend computation (configured for CPU stability).
* **Statsmodels:** For the ARIMA baseline.
* **Pandas/NumPy:** Data manipulation.
* **Matplotlib:** Visualization of forecasts and uncertainty bands.

---
**Author:** Satyam Rai