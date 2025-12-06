

## 📄 `econml.API.md`

````markdown
# EconML NHANES API – Project 3 (MSML610)

## 1. Overview

This module defines a small, opinionated API layer on top of the
[EconML](https://github.com/py-why/EconML) library for estimating
the causal effect of a health intervention on patient outcomes using
NHANES 2021–2023 data.

In our project, the **intervention** is:

> Any dietary supplement use (vs no supplements),

and the main outcomes are:

- `sbp_mean`: average systolic blood pressure (mmHg)
- `fasting_glucose_mg_dl`: fasting plasma glucose (mg/dL)

The API hides all of the data loading, merging, and model setup behind
a few high-level functions so that a user can run:

- one function to estimate treatment effects on SBP
- one function to estimate treatment effects on fasting glucose
- one function to get a traditional OLS baseline

---

## 2. Repository layout (relevant to this API)

Key files for this API layer:

- `econml_utils.py`  
  Low-level utilities: load cleaned NHANES CSVs, build the merged
  analysis table, and extract outcome / treatment / covariates.

- `econml.API.py`  
  High-level API functions that users will import and call:
  - `run_sbp_supplement_experiment`
  - `run_glucose_supplement_experiment`
  - `run_ols_for_outcome`

- `data/`  
  Directory containing cleaned numeric NHANES files, e.g.:
  - `BPXO_L_meaningful.csv`
  - `BMX_L_meaningful.csv`
  - `TCHOL_L_meaningful.csv`
  - `HDL_L_meaningful.csv`
  - `TRIGLY_L_meaningful.csv`
  - `GLU_L_meaningful.csv`
  - `HSCRP_L_meaningful.csv`
  - `DSQTOT_L_meaningful.csv`
  - `DEMO_L_meaningful.csv`

These CSVs are produced by separate data-preparation notebooks and are
treated as inputs to the API.

---

## 3. Native API (EconML)

Under the hood, we rely on EconML’s `DRLearner`:

```python
from econml.dr import DRLearner
from sklearn.linear_model import LogisticRegression
````

We use:

* **Binary treatment**: `treatment_supplement` (1 = any supplement, 0 = none)
* **Outcomes**:

  * `sbp_mean`
  * `fasting_glucose_mg_dl`
* **Covariates** (baseline confounders):

```text
age_years
sex
body_mass_index_kg_m2
weight_kg
waist_circumference_cm
total_cholesterol_mg_dl
direct_hdl_cholesterol_mg_dl
LBXTLG
fasting_glucose_mg_dl
hs_c_reactive_protein_mg_l
```

The API layer wraps DRLearner so users don’t need to construct
propensity models or handle missing data themselves.

---

## 4. Utils layer: `econml_utils.py`

### 4.1. Data loading

All raw NHANES component files are loaded via small helper functions,
e.g.:

```python
def load_bpxo_meaningful() -> pd.DataFrame:
    csv_path = DATA_DIR / "BPXO_L_meaningful.csv"
    return pd.read_csv(csv_path, na_values=[".", " "])
```

There are similar `load_*` functions for:

* BMX (body measures)
* TCHOL (total cholesterol)
* HDL
* TRIGLY (triglycerides & LDL)
* GLU (fasting glucose)
* HSCRP (hs-CRP)
* DSQTOT (supplements totals)
* DEMO (demographics & weights)

### 4.2. Blood pressure outcomes

```python
def build_bp_outcomes() -> pd.DataFrame:
    """Return df with respondent_sequence_number, sbp_mean, dbp_mean."""
    df = load_bpxo_meaningful()
    sbp_cols = [
        "systolic_1st_oscillometric_reading",
        "systolic_2nd_oscillometric_reading",
        "systolic_3rd_oscillometric_reading",
    ]
    dbp_cols = [
        "diastolic_1st_oscillometric_reading",
        "diastolic_2nd_oscillometric_reading",
        "diastolic_3rd_oscillometric_reading",
    ]
    df["sbp_mean"] = df[sbp_cols].mean(axis=1)
    df["dbp_mean"] = df[dbp_cols].mean(axis=1)
    return df[["respondent_sequence_number", "sbp_mean", "dbp_mean"]]
```

### 4.3. Building the merged analysis table

```python
def build_analysis_df() -> pd.DataFrame:
    """
    Merge all NHANES components into a single wide table:
      - outcomes: sbp_mean, dbp_mean, fasting_glucose_mg_dl
      - anthropometrics: BMI, weight, waist
      - labs: cholesterol, HDL, triglycerides, glucose, hs-CRP
      - treatment_supplement (1/0)
      - demographics: age_years, sex
    """
    # Implementation merges on respondent_sequence_number
    ...
```

### 4.4. Extract Y, T, X

```python
def get_y_t_x(df: pd.DataFrame,
              outcome_col: str,
              treatment_col: str = "treatment_supplement"):
    """
    Given the full analysis df, return:
      - y: outcome Series
      - t: treatment Series
      - X: covariate DataFrame
      - covariate_cols: list of covariate column names
    """
    covariate_cols = [
        "age_years",
        "sex",
        "body_mass_index_kg_m2",
        "weight_kg",
        "waist_circumference_cm",
        "total_cholesterol_mg_dl",
        "direct_hdl_cholesterol_mg_dl",
        "LBXTLG",
        "fasting_glucose_mg_dl",
        "hs_c_reactive_protein_mg_l",
    ]
    y = df[outcome_col]
    t = df[treatment_col]
    X = df[covariate_cols]
    return y, t, X, covariate_cols
```

---

## 5. API layer: `econml.API.py`

This file is what users import to run experiments.

### 5.1. `run_sbp_supplement_experiment`

```python
def run_sbp_supplement_experiment(random_state: int = 42) -> dict:
    """
    Estimate effect of any dietary supplement use on systolic BP (sbp_mean).

    Returns:
      {
        "ate_sbp": float,
        "covariates": List[str],
        "cate_df": pd.DataFrame,   # rows with tau_hat_sbp, age_bin, bmi_bin
        "age_effects": pd.Series,  # mean tau_hat_sbp by age_bin
        "bmi_effects": pd.Series,  # mean tau_hat_sbp by bmi_bin
      }
    """
    ...
```

* Uses `build_analysis_df` and `get_y_t_x` internally
* Drops rows with missing values
* Fits DRLearner with `LogisticRegression` as the treatment model
* Computes:

  * overall ATE on `sbp_mean`
  * individual CATEs (`tau_hat_sbp`)
  * average effects by age and BMI quartiles

### 5.2. `run_glucose_supplement_experiment`

```python
def run_glucose_supplement_experiment(random_state: int = 42) -> dict:
    """
    Estimate effect of supplement use on fasting_glucose_mg_dl.

    Returns:
      {
        "ate_glucose": float,
        "covariates": List[str],
        "cate_df": pd.DataFrame,   # rows with tau_hat_glucose
        "age_effects": pd.Series,
        "bmi_effects": pd.Series,
      }
    """
    ...
```

Same treatment and covariates as the SBP function; only the outcome changes.

### 5.3. `run_ols_for_outcome`

```python
def run_ols_for_outcome(outcome_col: str,
                        treatment_col: str = "treatment_supplement") -> dict:
    """
    Traditional linear regression baseline:
        outcome ~ treatment + covariates

    Returns:
      {
        "treatment_coef": float,
        "n": int,
        "covariate_names": List[str],
      }
    """
    ...
```

* Uses the same merged dataset and covariates
* Builds a design matrix `[treatment_supplement, X]`
* Fits `sklearn.linear_model.LinearRegression`
* Returns the treatment coefficient as a baseline estimate.

---

## 6. Minimal usage example

In Python or a notebook:

```python
from pathlib import Path
import importlib.util

# Load API module
api_path = Path.cwd() / "econml.API.py"
spec = importlib.util.spec_from_file_location("econml_api", api_path)
econml_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(econml_api)

# Run SBP experiment
sbp_results = econml_api.run_sbp_supplement_experiment()
print("ATE on SBP:", sbp_results["ate_sbp"])

# Run glucose experiment
glucose_results = econml_api.run_glucose_supplement_experiment()
print("ATE on fasting glucose:", glucose_results["ate_glucose"])

# Compare with OLS
ols_sbp = econml_api.run_ols_for_outcome("sbp_mean")
print("OLS SBP treatment coef:", ols_sbp["treatment_coef"])
```

This is exactly what `econml.API.ipynb` demonstrates, with a small number
of clean cells.