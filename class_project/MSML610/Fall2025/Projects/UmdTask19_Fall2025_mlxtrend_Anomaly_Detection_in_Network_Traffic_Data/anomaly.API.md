# Anomaly Detection API

This document describes the reusable **data + preprocessing API** implemented in `anomaly_utils.py`.  
It is designed to make the UNSW‑NB15 workflow reproducible and easy to reuse across notebooks/scripts.

This API is intentionally model-agnostic and dataset-aware, enabling consistent preprocessing, feature selection, and evaluation across multiple experiments.

> Scope: **only the tool** (loading, quick EDA, preprocessing + split, and numeric feature selection).  
> Project-specific model training, evaluation, plots, and conclusions belong in `anomaly.example.*`.

---

## What this API provides

`anomaly_utils.py` exposes four main functions:

1. **`load_unsw_from_zip`** – Load UNSW‑NB15 CSV parts from a zip, normalize columns, and return a single DataFrame.
2. **`basic_eda`** – Lightweight sanity checks (shape, dtypes, missingness, sample rows).
3. **`build_preprocess_and_split`** – Build a scikit‑learn `ColumnTransformer` pipeline (impute + scale numeric, impute + one‑hot categorical) and return a train/test split.
4. **`fast_numeric_feature_selection`** – A quick feature selection routine for numeric columns (MI ranking + optional EFS on a sample).

---

## Installation / environment

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data expectations

- The UNSW‑NB15 dataset is expected to be available as a **zip file** (example: `archive.zip`).
- The zip can contain multiple CSV parts in subfolders.
- The loader assigns official UNSW‑NB15 column names (see `UNSW_COLS` inside `anomaly_utils.py`) and returns a combined DataFrame.

Because datasets are often large, it’s normal to keep them out of Git (via `.gitignore`) and place them locally under a folder like `data/`.

---

## Function reference

### 1) `load_unsw_from_zip(zip_path, extract_dir="./data")`

Extract the zip, find CSV files, read them, and concatenate into one DataFrame.

**Parameters**
- `zip_path` (str): Path to the dataset zip (e.g., `archive.zip`)
- `extract_dir` (str): Where to extract files (default `./data`)

**Returns**
- `pd.DataFrame`: Combined UNSW‑NB15 records

**Example**
```python
from anomaly_utils import load_unsw_from_zip

df = load_unsw_from_zip("archive.zip", extract_dir="./data")
print(df.shape)
```

---

### 2) `basic_eda(df)`

Quick sanity check for shape, dtypes, missing values, and a preview.

**Example**
```python
from anomaly_utils import basic_eda

basic_eda(df)
```

---

### 3) `build_preprocess_and_split(df, label_col="label", test_size=0.2, random_state=42)`

Build a reusable preprocessing pipeline + return train/test splits.

Typical steps:
- Separate target label
- Identify numeric vs categorical columns
- Build `ColumnTransformer`:
  - Numeric: impute + scale
  - Categorical: impute + one‑hot encode
- Return split data and a pipeline ready to `fit_transform/transform`

**Example**
```python
from anomaly_utils import build_preprocess_and_split

(preprocess, X_train, X_test, y_train, y_test,
 X_train_proc, X_test_proc, num_cols, cat_cols) = build_preprocess_and_split(df)

print(X_train_proc.shape, X_test_proc.shape)
```

---

### 4) `fast_numeric_feature_selection(X_train, y_train, numeric_columns, out_dir="outputs", top_k_mi=10, min_k=3, max_k=5, sample_size=50000, random_state=42)`

Quickly select a strong numeric subset:
- rank numeric features via **Mutual Information**
- optionally apply **Exhaustive Feature Selection (EFS)** on a *sample* for tractability
- save selected features to `outputs/selected_numeric.json`

**Returns**
- `List[str]` selected numeric feature names

**Example**
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
print(selected_numeric)
```

---

## Notes

This API is designed to remain lightweight and reusable.  
Project-specific experimentation and result analysis are handled in the `anomaly.example.ipynb`.