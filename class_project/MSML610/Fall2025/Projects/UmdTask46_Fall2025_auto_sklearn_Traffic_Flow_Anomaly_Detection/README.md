# Project 3: Traffic Flow Anomaly Detection
**Project Tag:** `Fall2025_Traffic_Flow_Anomaly_Detection`
**Branch:** `UndTask66_Fall2025_auto_sklearn_Traffic_Flow_Anomaly_Detection`

---

## **Current Status: Mid-Submission Checkpoint**
**(As of: November 10, 2025)**

This directory contains the complete code implementation for the project, structured according to the class guidelines.

* **Code:** All 5 required files (`autosklearn_utils.py`, `autosklearn.API.ipynb`, `autosklearn.API.md`, `autosklearn.example.ipynb`, `autosklearn.example.md`) are implemented.
* **Docker:** The Docker environment is **not yet built**. This is the main task remaining for the final submission.
* **Results:** The `autosklearn.example.ipynb` notebook is fully coded and ready to run, but it **has not been executed yet**. The results table and plots in `autosklearn.example.md` are currently placeholders.

---

## Project Overview

The goal of this project is to use `auto-sklearn` to detect unusual traffic conditions and anomalies in the Metro Interstate Traffic Volume dataset [cite: 1.2].

The `auto-sklearn` model's performance will be compared against two traditional anomaly detection methods:
1.  Isolation Forest
2.  One-Class SVM

## File Structure

This project follows the 5-file structure required by the class instructions:

* **`autosklearn_utils.py`:** [cite: 1.29]
    * Contains helper functions for loading data (`load_and_prep_data`) and processing results (`define_anomalies_from_residuals`).

* **`autosklearn.API.ipynb` & `autosklearn.API.md`:** [cite: 1.25, 1.26]
    * Defines the "API contract" for our models.
    * It specifies a Python `Protocol` named `AnomalyModel` that all of our models must implement.

* **`autosklearn.example.ipynb` & `autosklearn.example.md`:** [cite: 1.25, 1.26]
    * This is the main application.
    * It implements the `AnomalyModel` protocol for `IsolationForest`, `OneClassSVM`, and our `auto-sklearn` (regression-based) approach.
    * It runs the comparison and is set up to generate the final plots and metrics.

## Plan to Final Submission

The following steps are required to complete the project:

1.  **Build Docker Environment:** Create and build a `Dockerfile` that installs all necessary dependencies (pandas, scikit-learn, auto-sklearn, matplotlib).
2.  **Run Notebook:** Start the Docker container and execute the `autosklearn.example.ipynb` notebook from top to bottom.
3.  **Generate Results:** The notebook will run for 15+ minutes (during the `auto-sklearn` training) and save the final plot as `anomaly_plot.png`.
4.  **Finalize Documentation:** Copy the output results (the markdown table) from the notebook and paste it into `autosklearn.example.md` to complete the report.
5.  **Submit:** Push the final, runnable, and documented project.