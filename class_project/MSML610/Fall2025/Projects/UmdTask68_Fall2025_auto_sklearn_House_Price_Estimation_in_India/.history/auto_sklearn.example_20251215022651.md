## auto-sklearn Example Application (House Price Estimation in India)

### Intent
This document presents a **complete example application** that uses:

- **auto-sklearn’s native API** (`autosklearn.regression.AutoSklearnRegressor`) to automatically search/tune a regression ensemble.
- A lightweight **project wrapper/API layer** (`auto_sklearn_utils.py`) so the notebook stays clean and reusable logic lives in Python modules.

The corresponding runnable notebook is **`auto_sklearn_example.ipynb`**.

---

### What this example application does
Given a CSV of Indian housing listings with mixed feature types (numeric, categorical, and a delimited amenities field), the application:

- Loads the dataset.
- Builds a preprocessing pipeline (imputation + scaling + one-hot encoding + amenities expansion).
- Trains an **AutoML** regressor via auto-sklearn.
- Trains **baseline** models (Random Forest and XGBoost) for comparison.
- Evaluates all models with standard regression metrics (MAE/RMSE).
- (Optional) Saves the trained model for reuse.

---

### API layer used by the example
The notebook is intentionally thin; it should call the reusable API/wrapper functions below instead of embedding complex logic inline.

- **`auto_sklearn_utils.py`** (facade module):
  - Re-exports the project’s utility functions/classes so the notebook can simply import from one place.

- **Key wrapper utilities (imported via the facade):**
  - **`load_housing_data(data_path)`** (`utils_data_io.py`): reads the CSV and normalizes blank strings to `NaN`.
  - **`get_column_groups(X)`** (`utils_feature_engineering.py`): defines feature groups (numeric/binary/categorical/text).
  - **`encode_binary_columns(X, binary_cols)`** (`utils_feature_engineering.py`): maps `Yes/No` → `1/0`.
  - **`AmenitiesEncoder`** (`utils_transformers.py`): expands the `Amenities` text field into multi-hot binary columns.
  - **`create_preprocessor(column_groups)`** (`utils_preprocessing.py`): builds the `ColumnTransformer` preprocessing pipeline.
  - **`prepare_data(...)`** (`utils_preprocessing.py`): the main “application-ready” data-prep entry point that loads, splits, encodes, and returns processed train/test matrices.

**Design choice:** the example treats **data preparation as a first-class API** (`prepare_data`) because it is the most error-prone and reusable part of the workflow.

---

### End-to-end workflow (conceptual)
This is the logic executed end-to-end in the example notebook.

1) **Configuration & reproducibility**
- Pick a data path (`data/raw/india_housing_prices.csv`).
- Fix a random seed.
- Choose runtime limits (e.g., number of rows to use for a quick demo, auto-sklearn time budget).

2) **Load + prepare features** (wrapper layer)
- Call `prepare_data(...)` to get:
  - `X_train_processed`, `X_test_processed`
  - `y_train`, `y_test`
  - `preprocessor` (fitted transformer)
  - `feature_names`

3) **Train AutoML model** (native auto-sklearn API)
- Instantiate `AutoSklearnRegressor(...)` with a clear time budget and seed.
- Fit on training data.

4) **Train baselines**
- RandomForestRegressor
- XGBRegressor

5) **Evaluate & compare**
- Compute MAE and RMSE on the same held-out test split.
- Present a concise comparison table.

6) **(Optional) Persist model**
- Save the fitted auto-sklearn model with `joblib`.

---

### What outputs to expect (clean and commented)
The example notebook should produce **small, readable outputs** such as:

- **Dataset summary**: number of rows/columns and basic target description.
- **Prepared data summary**:
  - train/test shapes
  - number of engineered features
- **Training confirmation**:
  - a short “fit completed” message
  - optionally: best validation score line from `sprint_statistics()`
- **Evaluation table**:
  - rows = models (Auto-sklearn, Random Forest, XGBoost)
  - columns = MAE, RMSE
- **(Optional) one simple plot**:
  - residual distribution or predicted-vs-actual scatter

**Design choice:** avoid printing large arrays, full feature-name dumps, or full `automl.show_models()` output; keep notebook outputs interpretable.

---

### How to run (Docker)
auto-sklearn is **Linux-only** in practice on many setups, so the recommended run path is Docker.

1) **Build the image**

```bash
docker build -t india-housing-prices .
```

2) **Run JupyterLab**

```bash
docker run -p 8888:8888 -v $(pwd):/app india-housing-prices
```

3) **Open the notebook**
- In the Jupyter UI, open **`auto_sklearn_example.ipynb`**.
- Run **Kernel → Restart & Run All**.

**Expected terminal output:** a Jupyter server start message that includes a local URL/token.

---

### Notes on key design decisions
- **Wrapper-first notebook design**: notebooks call `prepare_data(...)` and related helpers from modules, keeping cells short and maintainable.
- **Feature engineering is explicit**: binary `Yes/No` features are mapped deterministically, and `Amenities` is expanded via a custom transformer.
- **Resource-aware AutoML**: the example is intended to run with a bounded time budget and (optionally) a reduced training subset for demonstration speed.
- **Reproducibility**: fixed random seeds are used for the train/test split and model training.
