# Credit Scoring Model with SHAP (MSML610 – Fall 2025)

**Branch / Folder:** `TutorTask28_Fall2025_SHAP_Credit_Scoring_Model_with_SHAP_for_Interpretability`  
**Goal:** Tutorial-style project that builds a credit scoring model and explains it with **SHAP**.

> This commit focuses on **environment + runnable skeleton**, not final modeling. Code is deliberately light so the structure, Docker, and notebooks run end‑to‑end. Future PRs will enrich the modeling, SHAP plots, and analysis.

---

## Project layout

```
TutorTask28_Fall2025_SHAP_Credit_Scoring_Model_with_SHAP_for_Interpretability/
├─ SHAP_Credit.API.md
├─ SHAP_Credit.API.ipynb
├─ SHAP_Credit.example.md
├─ SHAP_Credit.example.ipynb
├─ SHAP_Credit_utils.py
├─ requirements.txt
├─ Dockerfile
├─ docker_build.sh
├─ docker_bash.sh
├─ docker_jupyter.sh
├─ .dockerignore
└─ data/
   └─ raw/
      └─ german_credit_data.csv   # <- place file here (not tracked by git)
```

---

## Quick start (Linux)

### 1) Build Docker image
```bash
bash docker_build.sh
```

### 2) Open a shell in the container
```bash
bash docker_bash.sh
# inside container:
python -V
pip list | grep shap
```

### 3) (Optional) Launch Jupyter Lab
```bash
bash docker_jupyter.sh
# open the printed http://127.0.0.1:<PORT>/ URL in your browser
```

### 4) Run the example notebook
Open **`SHAP_Credit.example.ipynb`** in Jupyter and run top-to-bottom.  
It will: import the utils module, generate a tiny synthetic dataset, fit a toy model, and print a few metrics.

> For the real project, add the UCI German Credit CSV at `data/raw/german_credit_data.csv` and update the notebook to use it. The utils already contain TODO hooks for loading a CSV and running SHAP on a real model.

---

## What’s included (skeleton code)
- **`SHAP_Credit_utils.py`** — tiny library with placeholders:
  - `load_credit_data()` (CSV or synthetic fallback)
  - `build_preprocessor()` (ColumnTransformer)
  - `train_xgb()` (baseline model)
  - `evaluate_model()` (AUC + confusion matrix)
  - `compute_shap()` (stubs ready for SHAP TreeExplainer)
- **`SHAP_Credit.API.ipynb`** — demonstrates calling the utils (API layer).
- **`SHAP_Credit.example.ipynb`** — runnable end‑to‑end demo using the utils.
- **Docker scripts** — `docker_build.sh`, `docker_bash.sh`, `docker_jupyter.sh` for a simple, classroom‑style workflow.

---

## Next steps (planned)
- Replace synthetic data with **German Credit** dataset and map target to {0,1}.
- Add robust SHAP visuals (summary, beeswarm, dependence).
- Threshold tuning (business cost), calibration, and subgroup/fairness slices.
- Write up **`SHAP_Credit.API.md`** and **`SHAP_Credit.example.md`** with diagrams and narrative.
