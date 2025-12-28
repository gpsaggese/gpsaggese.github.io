# Example: Anomaly Detection in Network Traffic (UNSW‑NB15)

This example shows the full end‑to‑end workflow (data → preprocessing → feature selection → models → evaluation) using the reusable API in `anomaly_utils.py`.

**What’s in this example**
1. Load dataset (local zip)
2. Quick EDA + sanity checks
3. Preprocess + train/test split (reproducible pipeline)
4. Feature selection (MI + EFS)
5. Train supervised models (RandomForest, XGBoost)
6. Train unsupervised baselines (Isolation Forest, LOF)
7. Evaluate and export metrics + plots to `outputs/`

---

## 0) Setup

Install deps:

```bash
pip install -r requirements.txt
```

Dataset:
- Put the UNSW‑NB15 `archive.zip` locally (recommended).
- Keep the raw dataset out of Git via `.gitignore`.

---

## 1) Load data + quick EDA

```python
from anomaly_utils import load_unsw_from_zip, basic_eda

df = load_unsw_from_zip("archive.zip", extract_dir="./data")
basic_eda(df)
```

---

## 2) Preprocess + train/test split

```python
from anomaly_utils import build_preprocess_and_split

(preprocess, X_train, X_test, y_train, y_test,
 X_train_proc, X_test_proc, num_cols, cat_cols) = build_preprocess_and_split(
    df, test_size=0.25, random_state=42
)

print("Train:", X_train_proc.shape, "Test:", X_test_proc.shape)
```

This keeps preprocessing consistent across all models.

---

## 3) Feature selection (numeric)

We first rank numeric features via Mutual Information (fast screening), then run Exhaustive Feature Selection (EFS) on a sample to find a compact subset.

```python
from anomaly_utils import fast_numeric_feature_selection

selected_numeric = fast_numeric_feature_selection(
    X_train=X_train,
    y_train=y_train,
    numeric_columns=num_cols,
    out_dir="outputs",
    top_k_mi=8,
    min_k=3,
    max_k=5,
    sample_size=50000,
)

print("Selected numeric:", selected_numeric)
```

Writes:
- `outputs/selected_numeric.json`

---

## 4) Supervised models (RF + XGBoost)

Train supervised classifiers and evaluate with ROC‑AUC and PR‑AUC.

Suggested outputs:
- `outputs/metrics_supervised_efs.csv`
- ROC / PR curves under `outputs/plots/`

---

## 5) Unsupervised baselines (Isolation Forest + LOF)

Train label-free anomaly detectors for a realistic baseline:
- IsolationForest
- LocalOutlierFactor (LOF)

Suggested outputs:
- `outputs/metrics_unsupervised.csv`

---

## 6) Results summary (fill with your exported metrics)

Example structure:

| Model | Type | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| RandomForest_EFS | Supervised | ~1.00 | ~1.00 |
| XGBoost_EFS | Supervised | ~1.00 | ~1.00 |
| IsolationForest | Unsupervised | ~0.95 | ~0.55 |
| LOF | Unsupervised | ~0.45 | ~0.12 |

## **Interpretation.**  
The supervised models achieve very strong performance because the UNSW-NB15 dataset provides explicit attack labels and strong discriminative features, allowing tree-based models to directly optimize for the target signal. In contrast, unsupervised methods such as Isolation Forest and LOF operate without label guidance and are more sensitive to class imbalance, high dimensionality, and overlapping normal/attack behavior, which naturally degrades their anomaly separation performance.

---

## How to run

### Local Jupyter
```bash
jupyter lab
```

Run the example notebook end-to-end to generate all outputs.

### Docker/Jupyter
Use `docker_build.sh` then `docker_jupyter.sh`, and run the notebooks inside the container.

