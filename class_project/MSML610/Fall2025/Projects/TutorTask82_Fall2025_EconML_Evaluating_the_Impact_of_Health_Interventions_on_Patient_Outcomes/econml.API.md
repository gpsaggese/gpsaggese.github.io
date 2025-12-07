
# EconML API for NHANES Supplement Effects

MSML610 Project:  
**TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes**

This document describes the **programming interface** exposed by:

- `econml_utils.py`
- `econml.API.py`

The goal of this API layer is to provide **simple, high-level functions** that:

1. Build an analysis-ready NHANES dataset.
2. Extract outcome (`Y`), treatment (`T`), and covariates (`X`).
3. Run **EconML DRLearner** experiments for:
   - Mean systolic blood pressure (`sbp_mean`)
   - Fasting glucose (`fasting_glucose_mg_dl`)
4. Provide a **baseline OLS comparison** for any chosen outcome.

The notebooks (`econml.API.ipynb` and `econml.example.ipynb`) use only these functions, not raw EconML calls.


---

## 1. Data and Utilities

### 1.1. Data location and format

All cleaned NHANES CSVs are expected in the project’s `data/` folder:

- `BPXO_L_meaningful.csv` – blood pressure readings  
- `BMX_L_meaningful.csv` – body measures (BMI, weight, waist, etc.)  
- `TCHOL_L_meaningful.csv` – total cholesterol  
- `HDL_L_meaningful.csv` – direct HDL cholesterol  
- `TRIGLY_L_meaningful.csv` – triglycerides (`LBXTLG`)  
- `GLU_L_meaningful.csv` – fasting plasma glucose  
- `HSCRP_L_meaningful.csv` – high-sensitivity CRP  
- `DSQTOT_L_meaningful.csv` – dietary supplements (includes `any_dietary_supplements_taken`)  
- `DEMO_L_meaningful.csv` – demographics (`age_years`, `sex`, etc.)

All these files share a common person-level identifier:

- `respondent_sequence_number`  
  (renamed from the original NHANES `SEQN` in the data prep step)


### 1.2. `econml_utils.build_analysis_df`

```python
from econml_utils import build_analysis_df

df = build_analysis_df()
````

**Purpose**

Build a single **merged analysis dataframe** that contains:

* Outcomes:

  * `sbp_mean` – mean systolic BP from 3 oscillometric readings
  * `dbp_mean` – mean diastolic BP
  * `fasting_glucose_mg_dl` – fasting plasma glucose
* Treatment:

  * `treatment_supplement` – binary:

    * `1` = any dietary supplement use
    * `0` = no dietary supplement use
* Covariates:

  * `age_years`
  * `sex`
  * `body_mass_index_kg_m2`
  * `weight_kg`
  * `waist_circumference_cm`
  * `total_cholesterol_mg_dl`
  * `direct_hdl_cholesterol_mg_dl`
  * `LBXTLG` (triglycerides)
  * `hs_c_reactive_protein_mg_l`

**Key logic**

1. Loads each `*_meaningful.csv` from `data/`.
2. Normalizes the ID column to `respondent_sequence_number`.
3. Computes `sbp_mean` and `dbp_mean` from the BPXO readings.
4. Inner-joins all components on `respondent_sequence_number`.
5. Constructs `treatment_supplement` from `any_dietary_supplements_taken`:

   * `1 → 1`
   * `2 → 0`
   * other values → `NaN`

### 1.3. `econml_utils.get_y_t_x`

```python
from econml_utils import get_y_t_x

y, t, X, covariate_cols = get_y_t_x(
    df,
    outcome_col="sbp_mean",
    treatment_col="treatment_supplement",
)
```

**Purpose**

Given the merged analysis dataframe from `build_analysis_df`, this helper extracts:

* `y` – outcome series
* `t` – treatment indicator (0/1)
* `X` – covariate matrix
* `covariate_cols` – list of covariate column names

**Signature**

```python
def get_y_t_x(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = "treatment_supplement",
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, List[str]]:
    ...
```

**Covariate set**

The function uses a **fixed candidate list** and keeps only those present:

* `age_years`
* `sex`
* `body_mass_index_kg_m2`
* `weight_kg`
* `waist_circumference_cm`
* `total_cholesterol_mg_dl`
* `direct_hdl_cholesterol_mg_dl`
* `LBXTLG`
* `fasting_glucose_mg_dl`
* `hs_c_reactive_protein_mg_l`

---

## 2. DRLearner API (EconML)

All EconML functionality is wrapped inside `econml.API.py`.

We use:

* `econml.dr.DRLearner`
* `sklearn.linear_model.LogisticRegression` for the propensity model
* `sklearn.linear_model.LinearRegression` for the outcome model

The main internal helper is `_fit_drl_for_outcome`, and you typically do **not** call it directly from notebooks. Instead, you use the two public functions:

* `run_sbp_supplement_experiment`
* `run_glucose_supplement_experiment`

### 2.1. `_fit_drl_for_outcome` (internal)

```python
from econml.API import _fit_drl_for_outcome  # usually not called directly
```

**Signature**

```python
def _fit_drl_for_outcome(
    outcome_col: str,
    treatment_col: str = "treatment_supplement",
    random_state: int = 42,
) -> Dict[str, Any]:
    ...
```

**Steps**

1. Calls `build_analysis_df()` to construct the merged NHANES dataset.

2. Calls `get_y_t_x(...)` to get `(Y, T, X, covariates)`.

3. Drops rows with any missing data in `Y`, `T`, or `X`.

4. Fits a `DRLearner`:

   ```python
   dr = DRLearner(
       model_regression=LinearRegression(),
       model_propensity=LogisticRegression(max_iter=2000, solver="lbfgs"),
       random_state=random_state,
   )
   dr.fit(Y=y_clean, T=t_clean, X=X_clean)
   ```

5. Computes:

   * ATE using `dr.ate(X_clean)`
   * Individual CATEs using `dr.effect(X_clean)`

6. Stores the CATEs in a copy of the cleaned dataframe as a new column:

   ```python
   tau_col = f"tau_hat_{outcome_col}"
   analysis_df_clean[tau_col] = tau_hat
   ```

7. Creates heterogeneity bins:

   * `age_bin` – quartiles of `age_years`
   * `bmi_bin` – quartiles of `body_mass_index_kg_m2`

8. Computes mean CATE per bin:

   * `age_effects = mean(tau_hat) by age_bin`
   * `bmi_effects = mean(tau_hat) by bmi_bin`

**Return value**

```python
{
    "ate": float,
    "covariates": list[str],
    "cate_df": pd.DataFrame,    # includes tau_hat_* column
    "tau_col": str,
    "age_effects": pd.Series or None,
    "bmi_effects": pd.Series or None,
}
```

### 2.2. `run_sbp_supplement_experiment`

```python
from econml.API import run_sbp_supplement_experiment

results = run_sbp_supplement_experiment(random_state=42)
```

**Signature**

```python
def run_sbp_supplement_experiment(random_state: int = 42) -> Dict[str, Any]:
    ...
```

**Purpose**

Runs the DRLearner pipeline with:

* Outcome: `sbp_mean`
* Treatment: `treatment_supplement`

Internally calls `_fit_drl_for_outcome(outcome_col="sbp_mean", ...)`.

**Return value**

```python
{
    "ate_sbp": float,           # ATE of supplement use on sbp_mean
    "covariates": list[str],    # covariate names in X
    "cate_df": pd.DataFrame,    # cleaned df with tau_hat_sbp_mean
    "tau_col": str,             # "tau_hat_sbp_mean"
    "age_effects": pd.Series or None,  # mean CATE by age_bin
    "bmi_effects": pd.Series or None,  # mean CATE by bmi_bin
}
```

This is the main entry point for the **primary outcome** in the tutorial notebook.

### 2.3. `run_glucose_supplement_experiment`

```python
from econml.API import run_glucose_supplement_experiment

results = run_glucose_supplement_experiment(random_state=42)
```

**Signature**

```python
def run_glucose_supplement_experiment(random_state: int = 42) -> Dict[str, Any]:
    ...
```

**Purpose**

Runs the same DRLearner pipeline with:

* Outcome: `fasting_glucose_mg_dl`
* Treatment: `treatment_supplement`

Internally calls `_fit_drl_for_outcome(outcome_col="fasting_glucose_mg_dl", ...)`.

**Return value**

```python
{
    "ate_glucose": float,       # ATE of supplement use on fasting_glucose_mg_dl
    "covariates": list[str],
    "cate_df": pd.DataFrame,    # cleaned df with tau_hat_fasting_glucose_mg_dl
    "tau_col": str,             # "tau_hat_fasting_glucose_mg_dl"
    "age_effects": pd.Series or None,
    "bmi_effects": pd.Series or None,
}
```

This is used for the **secondary outcome** in the example notebook.

---

## 3. OLS Baseline API

### 3.1. `run_ols_for_outcome`

```python
from econml.API import run_ols_for_outcome

ols_sbp = run_ols_for_outcome("sbp_mean")
ols_glu = run_ols_for_outcome("fasting_glucose_mg_dl")
```

**Signature**

```python
def run_ols_for_outcome(
    outcome_col: str,
    treatment_col: str = "treatment_supplement",
) -> Dict[str, Any]:
    ...
```

**Purpose**

Provides a **traditional OLS baseline** to compare against EconML:

[
Y = \beta_0 + \beta_1 \cdot \text{treatment} + \beta_2 X_1 + \dots + \beta_p X_p + \varepsilon
]

We interpret `β₁` (the coefficient on the treatment variable) as the simple regression-based estimate of the treatment effect, controlling linearly for covariates.

**Steps**

1. Calls `build_analysis_df()`.

2. Calls `get_y_t_x(df, outcome_col, treatment_col)` to get `(Y, T, X, covariates)`.

3. Drops rows with missing values.

4. Builds a design matrix:

   ```python
   T_matrix = t_clean.to_numpy().reshape(-1, 1)
   X_ols = np.column_stack([T_matrix, X_clean.to_numpy()])
   ```

5. Fits:

   ```python
   ols = LinearRegression()
   ols.fit(X_ols, y_clean.to_numpy())
   ```

6. Extracts the **first coefficient** as the treatment effect:

   ```python
   treatment_coef = float(ols.coef_[0])
   ```

**Return value**

```python
{
    "outcome": str,             # outcome_col
    "treatment_coef": float,    # OLS coefficient on treatment
    "covariates": list[str],    # same covariates as EconML
    "n_obs": int,               # number of observations used
}
```

In the example notebook, we place the DRLearner ATE and this OLS treatment coefficient side-by-side in a small comparison table.

---

## 4. How the API is used in notebooks

### 4.1. `econml.API.ipynb`

This notebook:

* Demonstrates how to import and call the API functions:

  * `build_analysis_df`, `get_y_t_x`
  * `run_sbp_supplement_experiment`
  * `run_glucose_supplement_experiment`
  * `run_ols_for_outcome`
* Shows the structure of the returned dictionaries.
* Explains the meaning of:

  * ATE (`ate_sbp`, `ate_glucose`)
  * CATEs (`tau_hat_*`)
  * Heterogeneity summaries (`age_effects`, `bmi_effects`)

### 4.2. `econml.example.ipynb`

This is the **main tutorial**:

* Introduces the intervention (“any dietary supplement use”).
* Uses `build_analysis_df()` to explore the merged NHANES dataset.
* Uses `get_y_t_x()` to explain `Y`, `T`, and `X`.
* Calls:

  * `run_sbp_supplement_experiment()` for the primary outcome
  * `run_glucose_supplement_experiment()` for the secondary outcome
* Visualizes heterogeneity by age and BMI.
* Calls `run_ols_for_outcome()` for both outcomes.
* Compares EconML DRLearner ATE and OLS coefficients to close the loop with the assignment tasks.

---

## 5. Summary

* The **utils layer** (`econml_utils.py`) handles **data preparation** and construction of `Y`, `T`, and `X`.
* The **API layer** (`econml.API.py`) wraps **EconML DRLearner** and **OLS** into three clean functions:

  * `run_sbp_supplement_experiment`
  * `run_glucose_supplement_experiment`
  * `run_ols_for_outcome`
* Notebooks work only through this API, which keeps the project:

  * Easy to understand,
  * Easy to reproduce,
  * Consistent with the MSML610 / Causify-style “API + example” pattern.

```
```
