# Airline-Delay-Prediction-XgBoost-Level-Hard-Difficulty-MSML610 Abstract
Airline Delay Prediction project (hard difficulty) for MSML 610

<br> Here is a link for the instructions (so I can access it easily instead of digging for it): https://github.com/gpsaggese-org/umd_classes/blob/master/class_project/instructions/README.md

<br> Here is a link for the specific project I have (level hard on the bottom): https://github.com/gpsaggese-org/umd_classes/blob/master/class_project/MSML610/Fall2025/project_descriptions/xgboost_Project_Description.md 

## Note: 
If you clone my repository, all you really have to do see the visuals and all my work is run 05_running_app.ipynb. There, you will see the summaries of my XgBoost as well as the bonus points of comparing it with Catboost alongside GBM. Likewise, you can see the indivdual plots and visuals for each of the 3 models and performance metrics such as confusion matrices, lost plot on train/val, ROC/AUC, etc. Of course, there was a lot that went into cleaning and preparing the data so we have "good data in, good data out". Similarly, you can see the backbone and deep underlying of my work that went into modeling and tuning the models - specifically XGBoost. Lukcily, you won't have to suffer the dozens of hours that it took me to run the models and tune them since you can clone it and see the outputs in the folders and .ipynb files.

## Reproduction On Command Line (Once you have cloned my repository:

### 1) Create env

conda create -n airline-delay-prediction python=3.10 -y

conda activate airline-delay-prediction

pip install -r requirements.txt

(CatBoost sometimes needs OpenMP on Mac; LightGBM wheel covers most setups)

### 2) Produce features & train (XGB tuned + baselines):

Chose 1 of the ways to run:

#### a. If you wanna run .py files:

python -m src.spark_etl          

python -m src.train_xgb           

python -m src.train_baselines    


#### b. Or if you wanna run .ipynb files:

notebooks/01_spark_etl_and_features.ipynb

notebooks/03_train_evaluate_model.ipynb

notebooks/04_tuning_models_ex.ipynb

### 3) Launch app

streamlit run src/app.py

Or:

notebooks/05_running_app.ipynb



- My "XYZ.example.ipynb" is each of the 5 .ipynb in the Notebooks (5 .ipynb notebooks). Using them all, this create the "XYZ.API.ipynb"


- My code which is used above is stored in the src folders with .py files


### Docker:

- I have included Docker (4 files at the root of this repository), you can run them once you have cloned and are in my repository in the command/terminal: chmod +x docker_build.sh docker_bash.sh docker_jupyter.sh docker_streamlit.sh

- run order:

bash docker_build.sh

for a shell: bash docker_bash.sh

for Jupyter: bash docker_jupyter.sh → open http://localhost:8888

for the app: bash docker_streamlit.sh → open http://localhost:8501







# Airline-Delay-Prediction — XGBoost (Hard) Finer Details

Predicting US airline **arrival delays** using gradient-boosted trees on merged **flights + weather** data. I built a full pipeline (ETL → features → training → Bayesian tuning → model comparison → app) and I’m checking this into the repo with **artifacts included** so you can run and evaluate everything **without re-training for hours**.

- **Course**: MSML 610  
- **Difficulty**: Hard  
- **Bonus**: Side-by-side comparison with **LightGBM** and **CatBoost**

### Quick links
- Project guidelines:  
  https://github.com/gpsaggese-org/umd_classes/blob/master/class_project/instructions/README.md
- Project description (XGBoost – Hard):  
  https://github.com/gpsaggese-org/umd_classes/blob/master/class_project/MSML610/Fall2025/project_descriptions/xgboost_Project_Description.md

---

## What’s in this repo (high level)

- **ETL & Features**: Spark job to join flight + weather, clean types, and build model-ready features  
- **Modeling**  
  - `src/train_xgb.py`: strong XGBoost baseline  
  - `src/tuning_models.py`: **Bayesian tuning** (Optuna TPE) with **time-aware CV**, optimizing AP (PR-AUC)  
  - `src/train_baselines.py`: LightGBM + CatBoost baselines with the same split and logging  
- **Artifacts** (already included): saved `.joblib` models, metrics JSONs, and plots (ROC/PR, loss, confusion matrix, feature importance)  
- **App**: Streamlit UI to browse metrics/plots and **score new flights** (single-row form by default)  
- **Docs for grading**: API + Example (both **.md** and **.ipynb**) and a tiny **smoke test** script

I made sure you can get to real results quickly. If you don’t want to retrain, you don’t have to.

---

## TL;DR (fastest way to see results)

If you only want to **see visuals / compare models / try a one-row score** and skip training:

```bash
# 1) env
conda create -n airline-delay-prediction python=3.10 -y
conda activate airline-delay-prediction
pip install -r requirements.txt

# 2) app (opens at http://localhost:8501 and you can see everything including the viz, models, predictions, performance tables, etc. However, look at the Notebooks folder for finer details and sanity checks)
streamlit run src/app.py
```

Or open **`notebooks/00,01,02,03,04, and 05_running_app.ipynb`** and just run all the cells there and get the requirements, setup, output, and sanity checks such as viewing the merged dataframe or generating the models updates after certain intervals while they are being trained/ran.

> I intentionally included trained models + metrics in `models/`, so you can review immediately.

---

## Reproduce (full, command line)

> Only do this if you want to regenerate features and retrain.

### 1) Create env

```bash
conda create -n airline-delay-prediction python=3.10 -y
conda activate airline-delay-prediction
pip install -r requirements.txt
# Note: CatBoost may need OpenMP on some Macs; LightGBM wheel usually just works.
```

### 2) Produce features & train

Pick **one** path:

**a) Python modules**
```bash
python -m src.spark_etl
python -m src.train_xgb
python -m src.train_baselines
python -m src.tuning_models   # Bayesian tuning (Optuna)
```

**b) Notebooks (this is what I did and advise you to know because you can run all the cells once you open them in VS Code or anywhere once you cloned this repo in whatever directory. Doing so, you can see my thought process for the data side by doing some checks after running the .py with sanity checks and viewing the df's as well as seeing the models and their outputs one by one after x amount of iterations. It gives a more in depth analysis and also ensures that my code scans your directory and places you in the proper file paths as well as downloading the required versions + modeling shenanigins)**
- `notebooks/00_colab_setup.ipynb`
- `notebooks/01_spark_etl_and_features.ipynb`
- `notebooks/02_EDA_and_analysis.ipynb`
- `notebooks/03_train_evaluate_model.ipynb`  
- `notebooks/04_tuning_models_ex.ipynb`
- `notebooks/05_running_app.ipynb`  

### 3) App
```bash
streamlit run src/app.py
# or open notebooks/05_running_app.ipynb
```

---

## Docker (zero local setup)

At repo root:

```bash
chmod +x docker_build.sh docker_bash.sh docker_jupyter.sh docker_streamlit.sh

# Build image
bash docker_build.sh

# Shell in container
bash docker_bash.sh

# Jupyter (http://localhost:8888)
bash docker_jupyter.sh

# Streamlit app (http://localhost:8501)
bash docker_streamlit.sh
```

---

## Smoke test (no retraining)

I added a tiny script that:
1) Confirms `models/` has metrics for **XGB tuned / LGBM / CatBoost**  
2) Prints a metrics summary and writes `models/model_comparison.csv` if it doesn’t exist  
3) Loads each model and scores **one** synthetic flight

Run:
```bash
python scripts/smoke_test.py
```

---

## Results snapshot (hold-out validation)

Same time-aware split and identical evaluation across models:

| model        | best_iter | AUC   | AP    | F1    | Precision | Recall | threshold | learning_rate | max_depth | log_loss |
|--------------|-----------|-------|-------|-------|-----------|--------|-----------|---------------|-----------|----------|
| **cat**      | 974       | 0.962 | **0.920** | **0.844** | 0.899     | 0.795  | 0.511059  | –             | –         | **0.16098** |
| **lgbm**     | 318       | 0.962 | 0.918 | 0.841 | 0.885     | 0.801  | 0.475464  | –             | –         | 0.16276  |
| **xgb_tuned**| 983       | 0.962 | 0.918 | 0.839 | **0.895** | 0.790  | 0.807907  | 0.0876        | 5         | 0.24172  |

- **CatBoost** is best on **AP** and **F1**, and has the lowest log loss.  
- **LightGBM** is a very close second overall.  
- **Tuned XGBoost** ties on AUC/AP but trails slightly on F1/log loss at the selected operating point.

You can view ROC/PR curves, loss curves, confusion matrices, and feature importance plots in the app or directly in `models/`.

---

## App overview (`src/app.py`)

- **Model Leaderboard**: AUC, AP, F1, precision/recall, log_loss, best_iter  
- **Artifacts**: PR/ROC, loss curve, confusion matrix (at chosen threshold), feature importance  
- **Score new flights**: single-row form with reasonable defaults; uses the model’s stored threshold

I removed the CSV upload by default to keep the UX clean for grading.

---

## Project structure (mapped to rubric)

```
.
├── API.md
├── API.ipynb
├── example.md
├── example.ipynb
├── Dockerfile
├── docker_build.sh
├── docker_bash.sh
├── docker_jupyter.sh
├── docker_streamlit.sh
├── screenshot.pdf    # a google doc that shows screenshots of me running some parts of my programs with outputs and some notes included
├── models/                       # artifacts (no retraining needed)
│   ├── tuned_all_features_bo_model.joblib
│   ├── tuned_all_features_bo_metrics.json
│   ├── tuned_all_features_bo_{pr,roc,loss_curve,confusion_matrix,feature_importance}.png
│   ├── lgbm_all_features_model.joblib
│   ├── lgbm_all_features_metrics.json
│   ├── lgbm_all_features_{pr,roc,loss_curve,confusion_matrix,feature_importance}.png
│   ├── cat_all_features_model.joblib
│   ├── cat_all_features_metrics.json
│   ├── cat_all_features_{pr,roc,loss_curve,confusion_matrix,feature_importance}.png
│   └── model_comparison.csv     
├── notebooks/
│   ├── 01_spark_etl_and_features.ipynb
│   ├── 03_train_evaluate_model.ipynb
│   ├── 04_tuning_models_ex.ipynb
│   └── 05_running_app.ipynb
├── requirements.txt
└── src/
    ├── app.py
    ├── spark_etl.py
    ├── train_xgb.py
    ├── train_baselines.py
    ├── tuning_models.py
    └── utils_model.py            # shared API layer (load, coerce, predict, etc.)
```

---

## API vs Example (as required)

- **`API.md` / `API.ipynb`**  
  Describe the tool’s **native API** and my **wrapper** in `utils_model.py` (schema coercion, loading artifacts, predicting probabilities, picking thresholds, etc.). Notebooks call functions—no heavy logic inline.

- **`example.md` / `example.ipynb`**  
  Runnable, end-to-end reference that uses the API: load artifacts, show metrics/plots, and export a comparison table (`model_comparison.csv`) when needed.

---

## Design choices (short version)

- **Time-aware split** and forward CV to avoid temporal leakage  
- **Objective**: binary:logistic; **metrics**: ROC-AUC and **AP** (class imbalance), and F1 at the chosen threshold  
- **Tuning**: Optuna TPE with pruning; bounds surfaced as CLI args for reproducibility  
- **Explainability**: feature importance (gain), confusion matrix with counts + overall %  
- **App**: lightweight by design; single-row scoring; artifacts panel for quick inspection

---

You’ll see metrics, plots, and be able to score a row **without** re-training anything.


