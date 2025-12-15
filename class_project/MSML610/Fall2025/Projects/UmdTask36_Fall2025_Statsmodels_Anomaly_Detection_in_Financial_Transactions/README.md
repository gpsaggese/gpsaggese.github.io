

# Anomaly Detection in Financial Transactions

**Statistical Modeling and Anomaly Detection for Fraud Detection**

---

## Overview

This project demonstrates how **statistical models** and **anomaly detection techniques**
can be applied to detect fraudulent credit card transactions in **highly imbalanced financial data**.

Fraud detection is inherently an anomaly detection problem:
fraudulent transactions are rare, costly to miss, and difficult to model using standard
classification approaches alone.

The project combines:
- Statistical modeling using a Generalized Linear Model (GLM)
- Residual and influence diagnostics for anomaly detection
- Supervised and unsupervised evaluation techniques

The emphasis is on **methodological correctness, interpretability, and reproducibility**.

---

## Dataset

The project uses the **Credit Card Fraud Detection** dataset from Kaggle
(European cardholders, September 2013).

**Key characteristics**
- Total transactions: 284,807
- Fraud cases: 492 (≈ 0.17%)
- Features:
  - `V1–V28`: anonymized PCA components
  - `Time`: seconds since the first transaction
  - `Amount`: transaction amount
- Target:
  - `Class = 1`: fraud
  - `Class = 0`: legitimate

The dataset is **not included** in this repository due to licensing.
Download `creditcard.csv` from Kaggle and place it in the project root directory.

---

## Repository Structure

```

AnomalyDetection/
│
├── AnomalyDetection_utils.py      # Reusable utility functions and API logic
│
├── AnomalyDetection.API.ipynb     # API usage demonstrations
├── AnomalyDetection.API.md        # API documentation
│
├── AnomalyDetection.example.ipynb # End-to-end fraud detection workflow
├── AnomalyDetection.example.md    # Narrative explanation of the workflow
│
├── Dockerfile                     # Docker environment for reproducibility
└── README.md                      # Project overview and instructions

````

---

## Running the Project (Docker)

This project is designed to run **entirely inside Docker** to ensure
consistent dependencies and reproducibility.

### 1. Build the Docker image

From the project root directory:

```bash
docker build -t anomaly-detection .
````

### 2. Run the Docker container

Mount the project directory into the container and expose Jupyter:

```bash
docker run -it \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  anomaly-detection
```

### 3. Access Jupyter Notebook

The container will start Jupyter Notebook and print a URL such as:

```
http://127.0.0.1:8888/?token=...
```

Open the URL in your browser, then run the notebooks:

* `AnomalyDetection.API.ipynb`
* `AnomalyDetection.example.ipynb`

Both notebooks are designed to run **top-to-bottom** without modification.

---

## Notes

* The dataset file `creditcard.csv` should not be committed to version control.
* Results may vary slightly due to randomness in SMOTE and Isolation Forest.
* The project focuses on **workflow design and interpretability**, not tuning for maximum accuracy.

---

## Author

**Harshini Karella**
Masters in Data Science
University of Maryland, College Park

---

