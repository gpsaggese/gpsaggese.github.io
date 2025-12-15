
# AzuaHousing.Example

This notebook (`AzuaHousing.Example.ipynb`) is the **consumer example** for the trained model. It demonstrates how a downstream user would:

- Load the trained pipeline from `artifacts/model.joblib`
- Load metadata from `artifacts/metrics.json`
- Load and clean the dataset
- Run inference on a held-out split
- Compute evaluation metrics on the test set
- Inspect best/worst prediction errors
- Visualize results
- Save a small prediction sample to CSV

## Prerequisites

You must run the training notebook first:

- `AzuaHousing.API.ipynb` should produce `artifacts/model.joblib` and `artifacts/metrics.json`

Install dependencies (outside notebook):

```bash
pip install -r requirements.txt
````

## Inputs

Artifacts expected:

* `artifacts/model.joblib`
* `artifacts/metrics.json`

Dataset expected:

* `data/melb_data.csv` (or set `DATA_PATH`)

The notebook will stop early with a clear error if the artifacts are missing.

## Notebook flow

### 1) Verify artifacts exist

Checks for the required files under `artifacts/`. If missing, instructs you to run the API notebook.

### 2) Load model + metadata

* Loads the fitted sklearn `Pipeline` from `model.joblib`
* Loads training metadata from `metrics.json` (selected model name, CV RMSE, CV R²)

### 3) Load data

Uses the shared loader in `azua_utils.py` to ensure consistent preprocessing expectations.

### 4) Compute stats (raw)

Reports:

* shape, missingness, target distribution

### 5) Clean data

Applies the same conservative cleaning approach used in the training workflow to keep the evaluation comparable.

### 6) Compute stats (cleaned)

Verifies the data quality after cleaning.

### 7) Inference + evaluation

* Creates a held-out split
* Runs `model.predict(X_test)`
* Computes standard metrics:

  * RMSE
  * MAE
  * R²

This provides an end-to-end sanity check that the saved model can be used reliably after loading.

### 8) Error analysis (best/worst cases)

Creates an evaluation table with:

* `y_true`
* `y_pred`
* absolute error

Shows:

* smallest-error rows (best predictions)
* largest-error rows (worst predictions)

### 9) Visualize results

Produces:

* Predicted vs Actual scatter plot (with diagonal reference line)
* Residual histogram (Actual − Predicted)

### 10) Save a prediction sample

Writes:

* `artifacts/predictions_sample.csv`

This is useful for debugging and reporting.

### 11) Single-row prediction example

Shows how to score a single input row by passing a one-row dataframe into the pipeline.

### 12) Interactive prediction (manual input)

This notebook supports an interactive-style workflow using **manual input**. You provide a Python dictionary of feature values, convert it into a one-row `DataFrame`, and pass it to the saved pipeline for inference.

#### How it works
1) Edit the `example_input` dictionary in the notebook with your feature values.  
2) The notebook constructs a one-row feature matrix `X_one` that matches the columns expected by the trained pipeline.  
3) The notebook calls `model.predict(X_one)` and prints the predicted house price.

#### Notes
- You do **not** need to fill every feature. Missing values can be set to `np.nan`; the pipeline imputers will handle them.
- For categorical fields, provide strings (e.g., `Suburb`, `Type`, `Method`).
- The prediction uses the model stored in `artifacts/model.joblib`, so you must run `AzuaHousing.API.ipynb` first.


## Outputs

This notebook does not retrain. It reads artifacts and produces evaluation outputs and plots. It also writes:

* `artifacts/predictions_sample.csv`

## Recommended usage order

1. Run training notebook:

* `AzuaHousing.API.ipynb`

2. Run consumer example:

* `AzuaHousing.Example.ipynb`

If your goal is to integrate this model into an application, the Example notebook is the correct reference for artifact loading and inference.

