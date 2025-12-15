
# EconML API — NHANES Supplement “Intervention” Effects

**MSML610 Project:** TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes

This document explains the **internal project API** (your reusable code layer), implemented in:

- `econml_utils.py` → builds the analysis dataset + prepares model inputs  
- `econml.API.py`   → runs EconML DRLearner experiments + OLS baseline

If a beginner opens this repo, they should be able to answer:

What is the treatment?  
What are the outcomes?  
What do the API functions do?  
What do they return?  
How do I reuse them safely?

---

## 0. Quick glossary

**Treatment (T):** the “intervention” indicator (0/1).  
In this project: *any dietary supplement use*.

**Outcome (Y):** what we care about changing (e.g., SBP or fasting glucose).

**Covariates (X):** baseline characteristics we control for (BMI, lipids, inflammation, etc.).

**ATE:** Average Treatment Effect  
> average effect of treatment for the whole sample.

**CATE:** Conditional Average Treatment Effect (individualized effect)  
> per-person estimated effect; stored as `tau_hat_*` in the returned dataframe.

---

## 1. Project structure 

This repo follows the “API + Example + Utils” pattern:

- **`econml_utils.py`**
  - Loads cleaned NHANES meaningful CSVs
  - Builds one merged dataframe (one row per respondent)
  - Defines outcomes and treatment
  - Provides `get_y_t_x()` to extract `(Y, T, X)` safely

- **`econml.API.py`**
  - Wraps the native EconML API (`econml.dr.DRLearner`)
  - Exposes clean public functions that return plain Python objects

- **`econml.API.ipynb`**
  - A compact “developer demo” notebook showing how to call the API and what it returns

- **`econml.example.ipynb`**
  - The full end-to-end tutorial/story (results + plots + interpretation)


---

## 2. Data expectations

### 2.1 Where the data lives

All cleaned NHANES files are expected under:

- `data/`

Your API reads the **meaningful** files, for example:

- `BPXO_L_meaningful.csv`   (blood pressure readings)
- `BMX_L_meaningful.csv`    (BMI, weight, waist, etc.)
- `TCHOL_L_meaningful.csv`  (total cholesterol)
- `HDL_L_meaningful.csv`    (HDL cholesterol)
- `TRIGLY_L_meaningful.csv` (triglycerides; includes `LBXTLG`)
- `GLU_L_meaningful.csv`    (fasting glucose)
- `HSCRP_L_meaningful.csv`  (hs-CRP)
- `DSQTOT_L_meaningful.csv` (supplements; includes `any_dietary_supplements_taken`)
- `DEMO_L_meaningful.csv`   (demographics)

### 2.2 Person identifier (ID)

All files must share a person-level ID column.  
The project standardizes this to:

- `respondent_sequence_number` (NHANES typically calls this `SEQN`)

---

## 3. Utilities layer (`econml_utils.py`)

### 3.1 `build_analysis_df(join="inner", clean=True)`

```python
from econml_utils import build_analysis_df
analysis_df = build_analysis_df()
````

**Purpose:**
Build a single merged analysis dataframe (one row per respondent) containing:

* **Outcomes**

  * `sbp_mean` (mean systolic BP from oscillometric readings)
  * `dbp_mean` (mean diastolic BP)
  * `fasting_glucose_mg_dl` (fasting plasma glucose)

* **Treatment**

  * `treatment_supplement` (binary 0/1)

**How treatment is defined**

* Source column: `any_dietary_supplements_taken` (from `DSQTOT`)
* Mapping:

  * `1 → treatment_supplement = 1` (Yes)
  * `2 → treatment_supplement = 0` (No)
  * anything else → missing

**Notes**

* `join="inner"` (default) produces a more complete-case dataset.
* `clean=True` applies a few lightweight plausibility checks (e.g., impossible ranges → missing).

**Return**

* `pd.DataFrame` with many columns (your run is ~3996 rows × ~110 columns before outcome filtering).

---

### 3.2 `get_y_t_x(df, outcome_col, treatment_col="treatment_supplement", dropna=True)`

```python
from econml_utils import get_y_t_x

y, t, X, covariates = get_y_t_x(
    analysis_df,
    outcome_col="sbp_mean",
    treatment_col="treatment_supplement",
)
```

**Purpose:**
Extract aligned, model-ready **(Y, T, X)** for a specific outcome.

**Important behavior (matches your notebook outputs):**

* Returns `y`, `t`, and `X` aligned to the same rows
* Drops rows with missing values in Y/T/X if `dropna=True`
* So lengths can be smaller than `len(analysis_df)` (example: ~2638 for SBP)

**Covariates used (core baseline set):**
The function uses a “core” list and keeps only the columns that actually exist in your dataframe, typically:

* `body_mass_index_kg_m2`
* `weight_kg`
* `waist_circumference_cm`
* `total_cholesterol_mg_dl`
* `direct_hdl_cholesterol_mg_dl`
* `LBXTLG`
* `hs_c_reactive_protein_mg_l`
* sometimes `fasting_glucose_mg_dl` (depends on outcome)

**Leakage prevention (very important):**

* If outcome is `fasting_glucose_mg_dl`, then glucose is **not** used as a covariate.
* If outcome is SBP, glucose may be included as a baseline covariate.

**Return values**

* `y`: outcome series
* `t`: treatment series (0/1)
* `X`: covariate dataframe (numeric inputs)
* `covariates`: list of covariate names used in `X`

---

## 4. EconML API layer (`econml.API.py`)

This file wraps the causal modeling so notebooks don’t have to touch raw EconML code.

### 4.1 What native EconML API is used?

We use:

* `econml.dr.DRLearner`

with:

* outcome model: `sklearn.linear_model.LinearRegression`
* propensity model: `sklearn.linear_model.LogisticRegression`

The wrapper computes:

* **ATE** (average treatment effect)
* **CATE** (per-person effect) saved as a column `tau_hat_*`
* **Heterogeneity summaries** by BMI quartiles (and by age quartiles if an age column is available)

---

### 4.2 Important note about importing `econml.API.py`

Because the filename contains a dot (`econml.API.py`), standard Python imports can be confusing on some systems.

Your `econml.API.ipynb` uses the safe approach: **dynamic import via `importlib`**.
Follow that notebook’s pattern if a normal `import` fails.

---

### 4.3 `run_sbp_supplement_experiment(random_state=42, n_bootstrap=200)`

```python
sbp_results = econml_api.run_sbp_supplement_experiment(random_state=42)
```

**What it runs**

* Outcome: `sbp_mean`
* Treatment: `treatment_supplement`
* Covariates: from `get_y_t_x`

**What it returns (matches your output keys)**

```python
{
  "ate_sbp": float,
  "ate_ci_low": float,
  "ate_ci_high": float,
  "n_obs": int,
  "covariates": list[str],
  "cate_df": pd.DataFrame,
  "tau_col": str,
  "age_effects": pd.Series | None,
  "bmi_effects": pd.Series | None,
  "model": object
}
```

**What the fields mean**

* `ate_sbp`: average effect of supplement use on SBP
* `ate_ci_low`, `ate_ci_high`: bootstrap 95% CI for ATE
* `n_obs`: observations used after filtering
* `cate_df`: cleaned modeling dataframe + CATE column
* `tau_col`: name of the CATE column (for SBP: `tau_hat_sbp_mean`)
* `age_effects`: mean CATE by age quartile (can be `None`)
* `bmi_effects`: mean CATE by BMI quartile
* `model`: fitted DRLearner (mostly for debugging)

---

### 4.4 `run_glucose_supplement_experiment(random_state=42, n_bootstrap=200)`

```python
glucose_results = econml_api.run_glucose_supplement_experiment(random_state=42)
```

Same as SBP, but:

* Outcome: `fasting_glucose_mg_dl`
* Keys: `ate_glucose`, and `tau_hat_fasting_glucose_mg_dl` inside `cate_df`

Return structure:

```python
{
  "ate_glucose": float,
  "ate_ci_low": float,
  "ate_ci_high": float,
  "n_obs": int,
  "covariates": list[str],
  "cate_df": pd.DataFrame,
  "tau_col": str,
  "age_effects": pd.Series | None,
  "bmi_effects": pd.Series | None,
  "model": object
}
```

---

## 5. Traditional baseline: OLS (`run_ols_for_outcome`)

This provides a simple comparison against standard regression.

### 5.1 `run_ols_for_outcome(outcome_col, treatment_col="treatment_supplement")`

```python
ols_sbp = econml_api.run_ols_for_outcome("sbp_mean")
ols_glu = econml_api.run_ols_for_outcome("fasting_glucose_mg_dl")
```

**Model**
A standard linear regression:

Y = β0 + β1·T + β2·X1 + … + βp·Xp + ε

Interpretation:

* `β1` = OLS estimate of treatment effect after linear adjustment

**Return (matches your output keys)**

```python
{
  "outcome": str,
  "treatment_coef": float,
  "treatment_ci_low": float,
  "treatment_ci_high": float,
  "covariates": list[str],
  "n_obs": int,
  "method": str
}
```

Notes:

* Uses `statsmodels` with **HC3 robust standard errors**.

---

## 6. Recommended notebook usage pattern

1. Build dataset

```python
analysis_df = build_analysis_df()
```

2. (Optional) extract Y/T/X for sanity checks

```python
y, t, X, covariates = get_y_t_x(analysis_df, outcome_col="sbp_mean")
```

3. Run EconML experiments

```python
sbp_results = run_sbp_supplement_experiment(random_state=42)
glucose_results = run_glucose_supplement_experiment(random_state=42)
```

4. Run OLS baseline

```python
ols_sbp = run_ols_for_outcome("sbp_mean")
ols_glu = run_ols_for_outcome("fasting_glucose_mg_dl")
```

---

## 7. Summary (what this API gives you)

This project is intentionally split into two clean layers:

* `econml_utils.py` = data assembly + consistent model inputs
* `econml.API.py`   = causal modeling + uncertainty + a traditional baseline

So your notebooks stay readable:

“call function → print results → plot → interpret”
instead of mixing messy preprocessing + modeling inline.

````
