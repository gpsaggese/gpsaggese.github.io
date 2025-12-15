
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
├── Dockerfile                     # Docker environment (used by class Docker scripts)
└── README.md                      # Project overview and instructions

````

---

## Running the Project (Docker)

This project is designed to run **entirely inside Docker** using the
**class-provided simple Docker setup**, which is the recommended workflow
used in the course tutorials.

### 1. Navigate to the Docker template directory

From the repository root, navigate to:

```bash
cd class_project/instructions/tutorial_template/tutorial_github_data605_style
````

### 2. Build the Docker container

Run the provided build script:

```bash
./docker_build.sh
```

This script builds a Docker image with Python, Jupyter, and all required
dependencies for the project.

### 3. Start Jupyter Notebook inside the container

Run:

```bash
./docker_jupyter.sh
```

This starts Jupyter Notebook inside the Docker container and exposes it on port `8888`.

### 4. Open Jupyter in the browser

Open the following URL in your web browser:

```
http://localhost:8888
```

The Jupyter Notebook interface will open at the path:

```
class_project/MSML610/Fall2025/Projects
```

From there, run the notebooks:

* `AnomalyDetection.API.ipynb`
* `AnomalyDetection.example.ipynb`

Both notebooks are designed to run **top-to-bottom** without modification.

**Note:**
There is a minor typo in the template documentation regarding directory names.
The correct directory is `tutorial_github_data605_style`, not `docker_simple`.
The Docker setup functions correctly when following the actual repository structure.

---

## Notes

* The dataset file `creditcard.csv` should not be committed to version control.
* Results may vary slightly due to randomness in SMOTE and Isolation Forest.
* The project focuses on **workflow design and interpretability**, not tuning for maximum accuracy.

---

## Author

**Harshini Karella**
Master’s in Data Science
University of Maryland, College Park

---


