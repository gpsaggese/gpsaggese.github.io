# Credit Scoring Model with SHAP (Work-in-Progress)

This repository contains a **work-in-progress** credit scoring pipeline using the classic **German Credit** dataset (UCI Machine Learning Repository).  
The aim is to build a transparent and auditable scoring model with **XGBoost** and **SHAP** for interpretability.

> ⚠️ Status: Not fully complete. Some steps are stubbed with TODOs. See **Future Steps** for the planned roadmap.

---

## Why this project?
- Train a baseline credit risk classifier (good/bad) with **XGBoost**.
- Produce **explainable** predictions using **SHAP** (global + local).
- Offer a reproducible pipeline (data → features → model → evaluation → explanations).

---

## Dataset
Use the **German Credit dataset** from the UCI ML Repository. You can download a CSV or load it manually and place it under `data/raw/`.

- Expected filename: `german_credit_data.csv` (you can change this in `src/utils/config.py`)

> The project intentionally avoids automatic downloading to respect the dataset's hosting terms. Place the file locally before running.

---

## Project Structure

```
credit_scoring_shap/
├─ README.md
├─ rqt.txt
├─ scripts/
│  └─ run_all.sh
└─ src/
   ├─ data/
   │  └─ make_dataset.py
   ├─ features/
   │  └─ build_features.py
   ├─ models/
   │  ├─ train_model.py
   │  └─ evaluate_model.py
   ├─ interpret/
   │  └─ shap_explain.py
   └─ utils/
      ├─ io.py
      ├─ seed.py
      └─ config.py
```

**Generated outputs (after running scripts):**
```
data/
  raw/
    german_credit_data.csv        # <- you provide this
  interim/
    split_train.parquet
    split_test.parquet
  processed/
    X_train.npz, y_train.npy
    X_test.npz,  y_test.npy
models/
  xgb_model.json
  pipeline.joblib
reports/
  metrics.json
  confusion_matrix.png
  shap_summary.png
  shap_beeswarm.png
  shap_dependence_<feature>.png
```

---

## Quickstart

1) **Create & activate a virtual environment (Python 3.10+)**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

2) **Install requirements**
```bash
pip install -r rqt.txt
```

3) **Add the dataset**
- Put the `german_credit_data.csv` file in `data/raw/` (create the folder if needed).
- Update `src/utils/config.py` paths if your filenames differ.

4) **Run pipeline (WIP)**
```bash
bash scripts/run_all.sh
```
or run individual steps:
```bash
python -m src.data.make_dataset
python -m src.features.build_features
python -m src.models.train_model
python -m src.models.evaluate_model
python -m src.interpret.shap_explain
```

---

## Current Features (MVP-ish)
- Deterministic train/test split with a fixed random seed.
- Basic preprocessing via `ColumnTransformer` and `Pipeline`:
  - One-Hot encode categoricals, standardize numerics.
  - Keeps feature names for SHAP plots.
- Baseline **XGBoost** classifier with sensible defaults (not tuned).
- Model evaluation: AUC, confusion matrix, simple report saved to `reports/metrics.json`.
- SHAP global summary and beeswarm plots.
- Example dependence plot for a chosen feature.

---

## Future Steps (Roadmap)
- **Data validation**: schema checks (pydantic / pandera), missing data audits.
- **Feature store**: persist feature lists and encoders with versioning.
- **Hyperparameter tuning**: Optuna/Optiuna-style search with CV and AUC-PR.
- **Fairness analysis**: subgroup metrics, equal opportunity / demographic parity diagnostics.
- **Robust SHAP**: background dataset selection, sampling strategies for speed.
- **What-if / Sensitivity tools**: UI sliders for top features, scorecards for business users.
- **Risk policy layer**: threshold selection, reject inference notes, challenger models (LightGBM, Logistic Regression baseline).
- **Model monitoring**: drift detection, performance dashboards.
- **Packaging**: CLI with `typer`, `pre-commit` hooks, and unit tests (pytest) for pipeline steps.
- **Deployment**: REST endpoint (FastAPI) with `/score` and `/explain` routes; batch scoring job.

---

## Ethics & Risk Management (Draft)
- Avoid disparate impact: evaluate metrics across relevant subgroups.
- Prefer transparent documentation: record rationale for threshold choices.
- Keep explanations human-friendly: pair SHAP visuals with plain-language summaries.
- Data handling: document sources, consent, and retention policies.

---

## Notes
- The code is intentionally **not** a finished product. Expect TODOs and small gaps.
- SHAP plots can be slow on big datasets; this project uses sampling when needed.
- Contributions welcome—open PRs with clear commit messages.



## Modular Pipeline (New)
Key modules:
- `src/schemas/german_credit_schema.py` — centralized target detection + feature typing.
- `src/pipeline/steps.py` — reusable functions (load→clean→split→features→train→evaluate→SHAP).
- `scripts/run_pipeline.py` — CLI to execute the pipeline end-to-end.
- `notebooks/credit_scoring_thin.ipynb` — *thin* notebook that imports and runs the pipeline functions.

### CLI usage
```bash
python scripts/run_pipeline.py --root .
```
