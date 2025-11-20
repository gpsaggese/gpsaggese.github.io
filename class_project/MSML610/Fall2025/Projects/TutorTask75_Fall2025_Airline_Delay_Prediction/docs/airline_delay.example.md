# Airline Delay — Example (End-to-End, No Retraining)

This is the practical “how I run it” guide for my project. It shows how I launch the Streamlit app, load the already-trained models (XGBoost tuned, LightGBM, CatBoost), score new flights, and reproduce the comparison figures — **without** re-running any long training jobs.

---

## 0) What’s in my repo (shortened because there's hella files)

```
.
├── Dockerfile
├── requirements.txt
├── README.md
├── docker_build.sh
├── docker_bash.sh
├── docker_jupyter.sh
├── docker_streamlit.sh
├── configs/
├── dashboard/
├── docs/
│   ├── airline_delay.API.md
│   ├── airline_delay.example.md      # ← this file
│   └── (other writeups)
├── notebooks/
│   ├── 00_colab_setup.ipynb
│   ├── 01_spark_etl_and_features.ipynb
│   ├── 02_EDA_and_analysis.ipynb
│   ├── 03_train_evaluate_models.ipynb
│   ├── 04_tuning_models_ex.ipynb
│   └── 05_running_app.ipynb
├── src/
│   ├── app.py                        # Streamlit UI
│   ├── utils_model.py                # shared helpers (schema, I/O, scoring)
│   ├── train_xgb.py                  # tuned XGBoost
│   ├── train_baselines.py            # LightGBM + CatBoost
│   └── tuning_models.py              # Optuna tuning
├── models/                           # artifacts used by the app
│   ├── tuned_all_features_bo_model.joblib
│   ├── tuned_all_features_bo_metrics.json
│   ├── tuned_all_features_bo_pr.png
│   ├── tuned_all_features_bo_roc.png
│   ├── tuned_all_features_bo_loss_curve.png
│   ├── tuned_all_features_bo_confusion_matrix.png
│   ├── tuned_all_features_bo_feature_importance.png
│   ├── lgbm_all_features_model.joblib
│   ├── lgbm_all_features_metrics.json
│   ├── lgbm_all_features_pr.png
│   ├── lgbm_all_features_roc.png
│   ├── lgbm_all_features_loss_curve.png
│   ├── lgbm_all_features_confusion_matrix.png
│   ├── lgbm_all_features_feature_importance.png
│   ├── cat_all_features_model.joblib
│   ├── cat_all_features_metrics.json
│   ├── cat_all_features_pr.png
│   ├── cat_all_features_roc.png
│   ├── cat_all_features_loss_curve.png
│   ├── cat_all_features_confusion_matrix.png
│   └── cat_all_features_feature_importance.png
├── reports/
│   └── figures/
└── data/
    ├── raw/
    └── processed/
        └── flights_with_weather.parquet
```

> The **app** only needs what’s in `models/`. You don’t need `data/` to use the UI.

---

## 1) Run it locally (Conda)

```bash
# Create and activate env
conda create -n airline-delay-prediction python=3.10 -y
conda activate airline-delay-prediction

# Install deps
python -m pip install -r requirements.txt
# macOS + CatBoost JVM (if CatBoost complains):
# conda install -n airline-delay-prediction -c conda-forge openjdk=11 -y || true

# Launch app
streamlit run src/app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

---

## 2) Or run it via Docker

From the repo root:

```bash
# Build image
bash docker_build.sh
# (Equivalent) docker build -t airline-delay-prediction:latest .

# Run Streamlit on :8501 and mount the repo so the container sees ./models and ./src
bash docker_streamlit.sh
# (Equivalent)
# docker run --rm -p 8501:8501 -v "$PWD":/app airline-delay-prediction:latest
```

Navigate to http://localhost:8501.

---

## 3) What the app shows

### 3.1 Model leaderboard (validation hold-out)

The table comes from the three `*_metrics.json` files in `models/`. My saved results:

- **CatBoost** — AUC ≈ **0.962**, AP ≈ **0.920**, F1 ≈ **0.844**, log_loss ≈ **0.161**, best_iter ≈ **974**  
- **LightGBM** — AUC ≈ **0.962**, AP ≈ **0.918**, F1 ≈ **0.841**, log_loss ≈ **0.163**, best_iter ≈ **318**  
- **XGBoost (tuned)** — AUC ≈ **0.962**, AP ≈ **0.918**, F1 ≈ **0.839**, log_loss ≈ **0.242**, best_iter ≈ **983**

All three tie on AUC/AP; **CatBoost** is best on **F1** and **log-loss** at my selected operating point.

### 3.2 Per-model artifacts

For whichever model I select in the sidebar, the app loads any of these images it finds under `models/`:

- Precision–Recall curve (`*_pr.png`)
- ROC curve (`*_roc.png`)
- Loss vs boosting rounds (`*_loss_curve.png`)
- Confusion Matrix at chosen threshold (`*_confusion_matrix.png`)
- Feature Importance (`*_feature_importance.png`)

Missing files are simply skipped (no crash).

---

## 4) Score new flights

I set **Single example form** as the default in the “Score new flights” section. There’s also a CSV tab if needed.

### A) Single example form (default)

1. Provide airline, airports, states, and numeric weather/geo features.
2. Click **Predict**.
3. The app shows:
   - `P(delay)` — predicted probability for class 1
   - Predicted class using the saved `threshold` from the model’s metrics JSON (fallback 0.5 if missing)

### B) Upload CSV (optional)

**Expected columns** (any order):

```
AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, ORIGIN_STATE, DEST_STATE,
DEPARTURE_DELAY, AIR_TIME, DISTANCE,
ORIGIN_LAT, ORIGIN_LON, DEST_LAT, DEST_LON,
temp, rhum, prcp, snow, wspd, pres
```

I coerce dtypes with `src/utils_model.py::coerce_schema`. The output adds `proba_delay` and `pred_delay`, and there’s a **Download scored CSV** button.

---

## 5) Programmatic use (helpers)

I centralized I/O and schema logic in `src/utils_model.py`, so notebooks and the app call the same functions.

```python
from src.utils_model import (
    SCHEMA, BASE_CATEGORICAL, BASE_NUMERIC,
    load_model, load_metrics, predict_proba, coerce_schema,
    pick_threshold, load_all_metrics_table, score_row, score_dataframe
)

# Load CatBoost and its operating threshold
m_cat = load_model("cat")
th = pick_threshold(load_metrics("cat"), default=0.5)

# One-row scoring
row = {
  "AIRLINE":"AA","ORIGIN_AIRPORT":"JFK","DESTINATION_AIRPORT":"LAX",
  "ORIGIN_STATE":"NY","DEST_STATE":"CA",
  "DEPARTURE_DELAY":12.0,"AIR_TIME":360.0,"DISTANCE":2475.0,
  "ORIGIN_LAT":40.64,"ORIGIN_LON":-73.78,"DEST_LAT":33.94,"DEST_LON":-118.41,
  "temp":22.0,"rhum":60.0,"prcp":0.0,"snow":0.0,"wspd":4.0,"pres":1013.0
}
score_row(m_cat, row, threshold=th)

```

---

## 6) Troubleshooting I ran into (and how I fixed it)

- **“No metrics JSONs found”** in the leaderboard  
  → Make sure the three `*_metrics.json` files are present in `models/`.

- **CatBoost Java error on macOS**  
  → `conda install -c conda-forge openjdk=11 -y` before installing CatBoost.

- **Images don’t render**  
  → Double-check filenames under `models/` match what `src/app.py` expects. Missing images are okay; they just don’t render.

- **Older Streamlit versions**  
  → I’m using `st.image(path, caption=...)` without `use_container_width` to be compatible with older/newer versions.

---

## 7) (Optional) Re-generating artifacts

If I need to retrain, these are the exact commands I used.

### XGBoost with Optuna (time-aware CV; maximize AP)

```bash
%run src/tuning_models.py \
  --in_path data/processed/flights_with_weather.parquet \
  --out_dir models \
  --split time --eval_size 0.20 \
  --use_departure_delay true \
  --tag tuned_all_features_bo \
  --cv_folds 5 \
  --bo_trials 10 \
  --bo_startup_trials 10 \
  --n_rounds 1001 \
  --early_stopping 100 \
  --lr_low 0.03 --lr_high 0.2 \
  --max_depth_low 5 --max_depth_high 9 \
  --min_child_weight_low 1 --min_child_weight_high 8 \
  --subsample_low 0.6 --subsample_high 1.0 \
  --colsample_bytree_low 0.6 --colsample_bytree_high 1.0 \
  --reg_alpha_low 1e-8 --reg_alpha_high 1.0 \
  --reg_lambda_low 1e-2 --reg_lambda_high 10.0
```

### LightGBM + CatBoost baselines (same split/knobs)

```bash
%run src/train_baselines.py \
  --in_path data/processed/flights_with_weather.parquet \
  --out_dir models \
  --split time --eval_size 0.20 \
  --use_departure_delay true \
  --model all \
  --tag all_features \
  --n_estimators 1000 \
  --learning_rate 0.1 \
  --early_stopping 100 \
  --log_period 25 \
  --lgbm_max_depth 7 \
  --cat_depth 7
```

These overwrite `models/` with fresh `*.joblib`, `*_metrics.json`, and plots. The app picks them up automatically.

---

## 8) What to expect after launch

- Sidebar lets me toggle between **CatBoost**, **LightGBM**, and **XGBoost (tuned)**.  
- The **leaderboard** uses exactly the numbers from the saved metrics.  
- The **artifacts** section shows PR, ROC, loss curve, confusion matrix, and feature importance for the selected model (if those files exist).  
- Under **Score new flights**, the **Single example form** is the default. I can still switch to the **Upload CSV** tab if I have a file.  
- The bottom **Model comparison snapshot** echoes the compact table for quick reporting.

That’s the complete runnable example that accompanies my project. It’s minimal, fast to verify, and aligned with the rubric’s “example.md + example.ipynb” requirement.
