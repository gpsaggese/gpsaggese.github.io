# Employee Attrition Prediction (IBM HR Analytics)

This project trains and evaluates an **XGBoost-based classifier** to predict whether an employee is likely to leave the company, using the **IBM HR Analytics Employee Attrition** dataset from Kaggle.

The goal is to show how to:
- Load and preprocess the dataset (with automatic download from Kaggle)
- Build train/test splits with proper preprocessing
- Train multiple classifiers (e.g., XGBoost, Random Forest, Logistic Regression)
- Evaluate models with Accuracy, F1, ROC-AUC and classification reports
- Inspect feature importance and model explanations

---

## 1. Folder structure (inside this project directory)

```
Fall2025_xgboost_Employee_Attrition_Prediction
|
├── docker_build.sh          # Build Docker image
├── docker_bash.sh           # Optional: open bash shell inside container
├── docker_jupyter.sh        # Run Jupyter Notebook in container
├── Dockerfile               # Image definition (Python 3.10 + libs + Jupyter)
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── xgboost.API.ipynb        # Small notebook showing XGBoost API + wrapper usage
├── xgboost.API.md           # Text documentation of XGBoost API + wrapper
├── xgboost.example.ipynb    # MAIN notebook: full E2E project (EDA + models)
└── xgboost.example.md       # Text summary of the example and results
```

All grading-relevant work is in **`xgboost.example.ipynb`**.

---

## 2. Prerequisites

- Docker installed and running (Docker Desktop or equivalent).
- Internet access (first run downloads the Kaggle dataset via `kagglehub`).
- Port **8888** free on the host.

---

## 3. Getting the project & one-time setup

All commands below are meant to be run in a terminal.

### 3.1 Clone the repository and go to the project folder

```bash
git clone https://github.com/gpsaggese-org/umd_classes.git
cd umd_classes/class_project/MSML610/Fall2025/Projects/Fall2025_xgboost_Employee_Attrition_Prediction
```

Make sure you see files like `Dockerfile`, `docker_build.sh`, `xgboost.example.ipynb`, etc. in this directory.

### 3.2 Make the Docker helper scripts executable (first time only)

```bash
chmod +x docker_build.sh docker_bash.sh docker_jupyter.sh
```

### 3.3 Build the Docker image

```bash
./docker_build.sh
```

This builds an image called:

```text
xgboost-docker-image
```

The image is based on `python:3.10-slim` and installs everything in `requirements.txt` plus Jupyter.

---

## 4. Running the notebooks in Docker

### 4.1 Start Jupyter in the container

From the **same project folder**:

```bash
./docker_jupyter.sh
```

This will:

- Start a container named `attrition-ml-jupyter`
- Mount the **current folder** into `/workspace` inside the container
- Expose Jupyter on **http://127.0.0.1:8888/**
- Launch Jupyter Notebook **without a token** (no password)

Leave this terminal **open** while using the notebook.

You should see output similar to:

```text
Serving notebooks from local directory: /workspace
Jupyter Server 2.x is running at:
    http://127.0.0.1:8888/tree
```

### 4.2 Open Jupyter in the browser

1. Open a web browser.
2. Go to: **http://127.0.0.1:8888/**
3. You should see the contents of `/workspace`, which is the same as this project folder.

---

## 5. What to run 

### 5.1 Main notebook

Open **`xgboost.example.ipynb`** and run:

- Menu: **Kernel → Restart & Run All**

The notebook will:

1. **Load the IBM HR dataset** via `kagglehub`  
   - Downloads `WA_Fn-UseC_-HR-Employee-Attrition.csv` from Kaggle on the first run and caches it.

2. **Perform EDA**  
   - Overall attrition rate and class imbalance  
   - Attrition by age group, job role, work–life balance, overtime  
   - Monthly income distribution and simple clustering

3. **Preprocess the data**  
   - Split into train/test with stratification  
   - Separate categorical and numeric features  
   - Apply `OneHotEncoder` + `StandardScaler` via a `ColumnTransformer`

4. **Train the main model (XGBoost)**  
   - `XGBClassifier` with `scale_pos_weight` to handle imbalanced labels  
   - Evaluate with Accuracy, F1-score, ROC-AUC and confusion matrix

5. **Tune the decision threshold**  
   - Sweep thresholds 0.1–0.9  
   - Compare **0.4** (best F1, better accuracy) vs **0.3** (higher recall for leavers)

6. **Compare with other models**  
   - Logistic Regression (class_weight="balanced")  
   - Random Forest (class_weight="balanced")

7. **Interpret the model**  
   - XGBoost feature importances  
   - SHAP summary plot for key drivers of attrition

### 5.2 API demo notebook 

Open **`xgboost.API.ipynb`** to see a compact demo of:

- Native XGBoost API calls  
- The lightweight wrapper layer used in the main notebook

This notebook is for API demonstration only; the full analysis lives in `xgboost.example.ipynb`.

### 5.3 API demo notebook 
- `xgboost.API.md` – Text documentation of the native XGBoost API and the wrapper layer.

### 5.4 API demo notebook 
- `xgboost.example.md` – Markdown description of the full example, results, and conclusions.

### 5.5 API demo notebook 
- `requirements.txt` – Python dependencies.

### 5.6 Dockerfile 
- `Dockerfile` – Lightweight image (Python 3.10 + requirements + Jupyter).

### 5.7 docker_build.sh
- `docker_build.sh` – Builds the Docker image.

### 5.8 docker_jupyter.sh
- `docker_jupyter.sh` – Starts Jupyter Notebook inside the container.

### 5.8 docker_bash.sh
- `docker_bash.sh` – Opens an interactive bash shell inside the container.

---

## 6. Optional: interactive shell inside the container

If you prefer a terminal **inside** the same environment:

```bash
./docker_bash.sh
```

This starts a container named `attrition-ml-bash` and drops you into `/workspace`.  
From there you can run, for example:

```bash
python
# or
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

---

## 7. Stopping the environment

- To stop Jupyter: press `Ctrl + C` in the terminal where `./docker_jupyter.sh` is running.
- Containers are run with `--rm`, so they are automatically removed when stopped.  
  The project files remain on the host in the cloned folder.

---


