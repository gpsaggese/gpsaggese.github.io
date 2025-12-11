# EconML API for NHANES Supplement Effects

MSML610 – Fall 2025
**Project:** `TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes`

---

## 0. Overview and Design Goals

This project studies the **causal effect of dietary supplement use** on key cardiometabolic outcomes using **NHANES 2021–2023** data and the **EconML** library.

The codebase is organized to follow the MSML610 “API + Example + Utils” tutorial pattern:

* `econml_utils.py`
  – Handles **data loading**, **cleaning**, **merging**, and **construction of Y/T/X**.

* `econml.API.py`
  – Exposes a **clean, high-level API** that:

  * Builds an analysis-ready dataset.
  * Defines treatment and control groups.
  * Fits **EconML DRLearner** models for causal effect estimation.
  * Computes **heterogeneous treatment effects** by age and BMI.
  * Provides a **traditional OLS baseline** for comparison.

* `econml.API.ipynb`
  – A compact notebook that **demonstrates the API itself** (how to call the functions and interpret outputs).

* `econml.example.ipynb`
  – A full **end-to-end example** (our actual project) that:

  * Introduces the clinical question.
  * Uses the API to run experiments.
  * Visualizes results.
  * Compares EconML vs. OLS.

This document focuses only on the **programming interface** of `econml_utils.py` and `econml.API.py` (notebook usage is described only at a high level).

---

## 1. Data and Utilities (`econml_utils.py`)

### 1.1. Input Data: Cleaned NHANES CSVs

All cleaned NHANES files live in the project’s `data/` folder:

* `BPXO_L_meaningful.csv`
  – Oscillometric blood pressure readings (3 SBP/DBP measurements).

* `BMX_L_meaningful.csv`
  – Anthropometrics: BMI, weight, waist, etc.

* `TCHOL_L_meaningful.csv`
  – Total cholesterol.

* `HDL_L_meaningful.csv`
  – Direct HDL cholesterol.

* `TRIGLY_L_meaningful.csv`
  – Triglycerides (`LBXTLG`) and derived LDL measures.

* `GLU_L_meaningful.csv`
  – Fasting plasma glucose.

* `HSCRP_L_meaningful.csv`
  – High-sensitivity C-reactive protein (hs-CRP).

* `DSQTOT_L_meaningful.csv`
  – 24-hour dietary intake and supplement summary, including:

  * `any_dietary_supplements_taken` (used to define treatment).

* `DEMO_L_meaningful.csv`
  – Demographics and survey design variables, including:

  * `age_in_years_at_screening` (renamed to `age_years`).
  * `gender` (recoded as `sex`).
  * Survey weights and design variables (not directly used in this tutorial’s models, but present in the data).

All files share a **common respondent ID**:

* `respondent_sequence_number`
  (this is the renamed version of original NHANES `SEQN`).

---

### 1.2. `build_analysis_df()`

```python
from econml_utils import build_analysis_df

df = build_analysis_df()
```

#### Purpose

Construct a **single, analysis-ready dataframe** that contains:

* **Outcomes**

  * `sbp_mean` – mean systolic BP from up to 3 oscillometric readings.
  * `dbp_mean` – mean diastolic BP.
  * `fasting_glucose_mg_dl` – fasting plasma glucose.

* **Treatment variable**

  * `treatment_supplement` – binary indicator for **any dietary supplement use**.

* **Covariates (baseline health & risk factors)**

  * `age_years`
  * `sex` (1 = Male, 2 = Female in the raw DEMO file; encoded numerically)
  * `body_mass_index_kg_m2`
  * `weight_kg`
  * `waist_circumference_cm`
  * `total_cholesterol_mg_dl`
  * `direct_hdl_cholesterol_mg_dl`
  * `LBXTLG` (triglycerides)
  * `hs_c_reactive_protein_mg_l`

This dataframe is the **single source of truth** for all later modeling functions.

#### Key Steps (Implementation Logic)

1. **Load each component dataset**
   Each `*_meaningful.csv` is read from the `data/` directory with `respondent_sequence_number` as the join key.

2. **Compute mean blood pressure** from BPXO:

   * `sbp_mean` is computed as the row-wise mean of:

     * `systolic_1st_oscillometric_reading`
     * `systolic_2nd_oscillometric_reading`
     * `systolic_3rd_oscillometric_reading`
   * `dbp_mean` is computed similarly using the diastolic readings.
   * Rows with entirely missing readings for SBP/DBP will have `NaN` means.

3. **Merge all tables**
   All component dataframes are **inner-joined** on `respondent_sequence_number` to ensure that the resulting sample has complete data across the selected domains (BP, labs, demographics, diet/supplements).

4. **Construct the treatment variable**
   The NHANES variable `any_dietary_supplements_taken` is recoded into a binary treatment:

   * `any_dietary_supplements_taken == 1` → `treatment_supplement = 1`
   * `any_dietary_supplements_taken == 2` → `treatment_supplement = 0`
   * Other codes or missing values → `NaN` (these rows are later dropped before modeling).

5. **Expose a clean analysis dataset**
   The final dataframe includes:

   * ID: `respondent_sequence_number`
   * Outcomes: `sbp_mean`, `dbp_mean`, `fasting_glucose_mg_dl`
   * Treatment: `treatment_supplement`
   * Covariates: age, sex, BMI, weight, waist, lipids, hs-CRP.

#### Return type

```python
pd.DataFrame
```

---

### 1.3. `get_y_t_x()`

```python
from econml_utils import get_y_t_x

y, t, X, covariate_cols = get_y_t_x(
    df,
    outcome_col="sbp_mean",
    treatment_col="treatment_supplement",
)
```

#### Signature

```python
def get_y_t_x(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = "treatment_supplement",
) -> tuple[pd.Series, pd.Series, pd.DataFrame, list[str]]:
    ...
```

#### Purpose

Given the merged analysis dataframe returned by `build_analysis_df()`, this function extracts:

* `y` – outcome vector (e.g. `sbp_mean`).
* `t` – treatment indicator (0/1) for supplement use.
* `X` – matrix of baseline covariates.
* `covariate_cols` – ordered list of covariate names used in `X`.

This ensures that the **EconML** models and the **OLS baseline** use the **same inputs** for fair comparison.

#### Covariate Set

`get_y_t_x()` selects a fixed, clinically reasonable covariate set (only those present in `df` are kept):

* `age_years`
* `sex`
* `body_mass_index_kg_m2`
* `weight_kg`
* `waist_circumference_cm`
* `total_cholesterol_mg_dl`
* `direct_hdl_cholesterol_mg_dl`
* `LBXTLG` (triglycerides)
* `fasting_glucose_mg_dl` (used as a covariate when the outcome is NOT glucose)
* `hs_c_reactive_protein_mg_l`

The function:

1. Validates that `outcome_col` and `treatment_col` exist.
2. Filters the candidate covariates to those actually present in `df`.
3. Returns `y`, `t`, `X` and the list of covariate names.

---

## 2. EconML DRLearner API (`econml.API.py`)

The EconML logic is wrapped in a small set of functions, so that notebooks can stay clean and “tutorial-like.”

At a high level, each experiment follows these steps:

1. Build the merged dataset (`build_analysis_df()`).
2. Extract `(Y, T, X)` (`get_y_t_x()`).
3. Drop rows with any missing values in `Y`, `T`, or `X`.
4. Fit a **Doubly Robust Learner (DRLearner)**.
5. Compute:

   * Overall **Average Treatment Effect (ATE)**.
   * **Conditional Average Treatment Effects (CATEs)** per individual.
   * Mean CATE by **age quartiles** and **BMI quartiles**.

We also provide an OLS baseline for comparison.

---

### 2.1. Internal Helper: `_fit_drl_for_outcome` (notebook users normally don’t call this)

```python
from econml.API import _fit_drl_for_outcome  # internal helper, not the main public API
```

#### Signature

```python
def _fit_drl_for_outcome(
    outcome_col: str,
    treatment_col: str = "treatment_supplement",
    random_state: int = 42,
) -> dict:
    ...
```

#### Purpose

Encapsulate **all shared DRLearner logic** so that:

* `run_sbp_supplement_experiment(...)` and
* `run_glucose_supplement_experiment(...)`

can be implemented as thin wrappers that pass the appropriate outcome.

#### Detailed Steps

1. **Build the merged dataset**

   ```python
   analysis_df = build_analysis_df()
   ```

2. **Extract Y, T, and X**

   ```python
   y, t, X, covariate_cols = get_y_t_x(
       analysis_df,
       outcome_col=outcome_col,
       treatment_col=treatment_col,
   )
   ```

3. **Drop missing observations**

   ```python
   mask = (~y.isna()) & (~t.isna()) & (~X.isna().any(axis=1))
   y_clean = y[mask]
   t_clean = t[mask]
   X_clean = X[mask]
   ```

   This simple strategy (complete-case analysis) keeps the tutorial focused on causal methods rather than missing-data techniques.

4. **Fit the DRLearner**

   We use:

   * `LinearRegression()` as the outcome model (`E[Y | X, T]`).
   * `LogisticRegression(max_iter=2000, solver="lbfgs")` as the propensity model (`P(T=1 | X)`).

   ```python
   from econml.dr import DRLearner
   from sklearn.linear_model import LinearRegression, LogisticRegression

   dr = DRLearner(
       model_regression=LinearRegression(),
       model_propensity=LogisticRegression(max_iter=2000, solver="lbfgs"),
       random_state=random_state,
   )
   dr.fit(Y=y_clean, T=t_clean, X=X_clean)
   ```

5. **Compute ATE and CATE**

   * **ATE**: average treatment effect among the cleaned sample

     ```python
     ate = float(dr.ate(X_clean))
     ```

   * **CATE** (individual-level effect for each row):

     ```python
     tau_hat = dr.effect(X_clean)
     tau_col = f"tau_hat_{outcome_col}"
     ```

6. **Attach CATEs and build heterogeneity summaries**

   ```python
   analysis_df_clean = analysis_df.loc[mask].copy()
   analysis_df_clean[tau_col] = tau_hat
   ```

   * **Age bins** (quartiles of `age_years`):

     ```python
     analysis_df_clean["age_bin"] = pd.qcut(
         analysis_df_clean["age_years"],
         4,
         labels=["Q1 (youngest)", "Q2", "Q3", "Q4 (oldest)"],
     )
     ```

   * **BMI bins** (quartiles of `body_mass_index_kg_m2`):

     ```python
     analysis_df_clean["bmi_bin"] = pd.qcut(
         analysis_df_clean["body_mass_index_kg_m2"],
         4,
         labels=["Q1 (leanest)", "Q2", "Q3", "Q4 (highest BMI)"],
     )
     ```

   * **Mean effect by bin**:

     ```python
     age_effects = (
         analysis_df_clean.groupby("age_bin")[tau_col]
         .mean()
         .sort_index()
     )

     bmi_effects = (
         analysis_df_clean.groupby("bmi_bin")[tau_col]
         .mean()
         .sort_index()
     )
     ```

7. **Return value**

   ```python
   {
       "ate": ate,
       "covariates": covariate_cols,
       "cate_df": analysis_df_clean,   # includes tau_col, age_bin, bmi_bin
       "tau_col": tau_col,
       "age_effects": age_effects,
       "bmi_effects": bmi_effects,
   }
   ```

---

### 2.2. `run_sbp_supplement_experiment(...)`

```python
from econml.API import run_sbp_supplement_experiment

results_sbp = run_sbp_supplement_experiment(random_state=42)
```

#### Signature

```python
def run_sbp_supplement_experiment(random_state: int = 42) -> dict:
    ...
```

#### Scientific Question

> **What is the causal effect of “any dietary supplement use” (`treatment_supplement`) on mean systolic blood pressure (`sbp_mean`)?**

This is the **primary outcome** in the project and maps directly to the assignment’s:

* **Data Preparation**
* **Defining treatment/control groups**
* **Causal effect estimation**
* **Heterogeneity analysis**

#### Internals

This function is a thin wrapper around `_fit_drl_for_outcome`:

```python
core = _fit_drl_for_outcome(
    outcome_col="sbp_mean",
    treatment_col="treatment_supplement",
    random_state=random_state,
)
```

It then renames the ATE key to make the dictionary more self-documenting:

```python
results = {
    "ate_sbp": core["ate"],
    "covariates": core["covariates"],
    "cate_df": core["cate_df"],
    "tau_col": core["tau_col"],           # "tau_hat_sbp_mean"
    "age_effects": core["age_effects"],
    "bmi_effects": core["bmi_effects"],
}
```

#### Return Value

```python
{
    "ate_sbp": float,         # Average treatment effect on sbp_mean
    "covariates": list[str],  # Covariate names used in X
    "cate_df": pd.DataFrame,  # Cleaned data with tau_hat_sbp_mean, age_bin, bmi_bin
    "tau_col": str,           # Name of the CATE column ("tau_hat_sbp_mean")
    "age_effects": pd.Series, # Mean effect per age quartile
    "bmi_effects": pd.Series, # Mean effect per BMI quartile
}
```

These fields are used in `econml.example.ipynb` to:

* Report the **ATE** (supplement vs no supplement on SBP).
* Plot **heterogeneous effects** by age and BMI.
* Discuss which subgroups benefit more/less.

---

### 2.3. `run_glucose_supplement_experiment(...)`

```python
from econml.API import run_glucose_supplement_experiment

results_glu = run_glucose_supplement_experiment(random_state=42)
```

#### Signature

```python
def run_glucose_supplement_experiment(random_state: int = 42) -> dict:
    ...
```

#### Scientific Question

> **What is the causal effect of “any dietary supplement use” on fasting glucose (`fasting_glucose_mg_dl`)?**

This provides a **secondary cardiometabolic outcome** and demonstrates reuse of the same API for a different `Y`.

#### Internals

Exactly the same pipeline as for SBP, but with:

* `outcome_col = "fasting_glucose_mg_dl"`

```python
core = _fit_drl_for_outcome(
    outcome_col="fasting_glucose_mg_dl",
    treatment_col="treatment_supplement",
    random_state=random_state,
)
```

Return dictionary:

```python
results = {
    "ate_glucose": core["ate"],
    "covariates": core["covariates"],
    "cate_df": core["cate_df"],           # includes tau_hat_fasting_glucose_mg_dl
    "tau_col": core["tau_col"],           # "tau_hat_fasting_glucose_mg_dl"
    "age_effects": core["age_effects"],
    "bmi_effects": core["bmi_effects"],
}
```

#### Return Value

```python
{
    "ate_glucose": float,
    "covariates": list[str],
    "cate_df": pd.DataFrame,
    "tau_col": str,
    "age_effects": pd.Series,
    "bmi_effects": pd.Series,
}
```

In the example notebook, we:

* Compare the **sign and magnitude** of the ATE for SBP vs glucose.
* Check whether heterogeneity patterns (by age/BMI) are consistent across outcomes.

---

## 3. OLS Baseline (`run_ols_for_outcome`)

To address the assignment’s **“Bonus: compare with traditional methods”** requirement, we provide a simple OLS baseline API.

### 3.1. `run_ols_for_outcome(...)`

```python
from econml.API import run_ols_for_outcome

ols_sbp = run_ols_for_outcome("sbp_mean")
ols_glu = run_ols_for_outcome("fasting_glucose_mg_dl")
```

#### Signature

```python
def run_ols_for_outcome(
    outcome_col: str,
    treatment_col: str = "treatment_supplement",
) -> dict:
    ...
```

#### Model

We fit a standard linear regression:

[
Y = \beta_0 + \beta_1 \cdot \text{treatment_supplement} + \beta_2 X_1 + \dots + \beta_p X_p + \varepsilon
]

* `Y` = outcome (e.g. `sbp_mean` or `fasting_glucose_mg_dl`)
* `treatment_supplement` = 1 if any supplement, 0 if none
* `X` = same covariates returned by `get_y_t_x()`

We interpret the **coefficient on the treatment variable (`β₁`)** as:

> “OLS estimate of the treatment effect, assuming a **linear model** and no strong violations (e.g. unmeasured confounding, nonlinearity).”

This allows us to directly compare:

* `DRLearner ATE` vs `OLS treatment coefficient` for the same outcome and covariates.

#### Steps

1. Build the analysis dataframe:

   ```python
   df = build_analysis_df()
   ```

2. Extract `y`, `t`, `X`:

   ```python
   y, t, X, covariate_cols = get_y_t_x(
       df,
       outcome_col=outcome_col,
       treatment_col=treatment_col,
   )
   ```

3. Drop missing values (same mask strategy as DRLearner).

4. Construct the design matrix with treatment as the **first column**:

   ```python
   T_matrix = t_clean.to_numpy().reshape(-1, 1)
   X_ols = np.column_stack([T_matrix, X_clean.to_numpy()])
   ```

5. Fit linear regression:

   ```python
   from sklearn.linear_model import LinearRegression

   ols = LinearRegression()
   ols.fit(X_ols, y_clean.to_numpy())
   treatment_coef = float(ols.coef_[0])
   ```

#### Return Value

```python
{
    "outcome": outcome_col,     # e.g., "sbp_mean"
    "treatment_coef": float,    # OLS coefficient on treatment_supplement
    "covariates": list[str],    # Covariates in the model
    "n_obs": int,               # Number of observations used after dropping NaNs
}
```

In `econml.example.ipynb` we build a small comparison table like:

| Outcome                 | DRLearner ATE | OLS treatment coef |
| ----------------------- | ------------- | ------------------ |
| `sbp_mean`              | …             | …                  |
| `fasting_glucose_mg_dl` | …             | …                  |

and discuss similarities/differences.

---

## 4. How the API Is Used in the Notebooks

### 4.1. `econml.API.ipynb` – API Demonstration

This notebook focuses on the **tool API itself** (not the full project story):

Typical flow:

1. **Import and inspect**:

   ```python
   from econml_utils import build_analysis_df, get_y_t_x
   from econml.API import (
       run_sbp_supplement_experiment,
       run_glucose_supplement_experiment,
       run_ols_for_outcome,
   )
   ```

2. **Show merged dataset structure** using `build_analysis_df()`.

3. **Show Y/T/X shapes** and covariate names via `get_y_t_x()`.

4. **Run DRLearner experiments** and print ATE + heterogeneity summaries:

   ```python
   sbp_results = run_sbp_supplement_experiment()
   glu_results = run_glucose_supplement_experiment()
   ```

5. **Run OLS baselines**:

   ```python
   ols_sbp = run_ols_for_outcome("sbp_mean")
   ols_glu = run_ols_for_outcome("fasting_glucose_mg_dl")
   ```

This notebook serves as a **mini-reference** for how to program against the API.

---

### 4.2. `econml.example.ipynb` – Full Project Example

This notebook is the **project tutorial** the professor will read:

1. **Motivation & clinical question**

   * Cardiometabolic risk and supplements.
   * NHANES as a large, representative dataset.

2. **Data preparation and treatment definition** (calls into `econml_utils`):

   * Build dataset via `build_analysis_df()`.
   * Explain `treatment_supplement` (treatment vs control groups).
   * Summarize baseline covariates.

3. **Causal effect estimation with EconML**:

   * Call `run_sbp_supplement_experiment()`.
   * Report `ate_sbp`.
   * Explain **DRLearner** intuition (doubly robust, machine-learning based).

4. **Heterogeneity analysis**:

   * Plot `age_effects` and `bmi_effects`.
   * Discuss which subgroups appear to benefit more/less.

5. **Secondary outcome**:

   * Call `run_glucose_supplement_experiment()`.
   * Compare ATE for SBP vs glucose.

6. **Comparison with traditional OLS**:

   * Call `run_ols_for_outcome("sbp_mean")` and `run_ols_for_outcome("fasting_glucose_mg_dl")`.
   * Build a simple comparison table.
   * Interpret differences and discuss why machine-learning based causal methods (DRLearner) can be preferable.

7. **Conclusion**:

   * Summarize the **estimated treatment effects**.
   * Highlight how the **API layer** made it easy to:

     * Swap outcomes.
     * Reuse covariate definitions.
     * Compare DRLearner vs OLS.

---

## 5. Summary of the API Surface

For quick reference, here is the full API surface you use from notebooks:

### From `econml_utils.py`

| Function                                    | Description                                                   |
| ------------------------------------------- | ------------------------------------------------------------- |
| `build_analysis_df()`                       | Load, merge, and engineer NHANES features into one dataframe. |
| `get_y_t_x(df, outcome_col, treatment_col)` | Extract Y, T, X and covariate list for modeling.              |

### From `econml.API.py`

| Function                                 | Outcome                 | Purpose                                         |
| ---------------------------------------- | ----------------------- | ----------------------------------------------- |
| `_fit_drl_for_outcome(...)`              | Generic (internal)      | Shared DRLearner pipeline for any outcome.      |
| `run_sbp_supplement_experiment(...)`     | `sbp_mean`              | Primary causal experiment (SBP vs supplements). |
| `run_glucose_supplement_experiment(...)` | `fasting_glucose_mg_dl` | Secondary experiment (glucose vs supplements).  |
| `run_ols_for_outcome(outcome_col, ...)`  | any of the above        | OLS baseline for comparison with EconML.        |

This separation between **utils**, **API**, and **example** keeps the project:

* Aligned with the **MSML610 tutorial template**,
* Easy to read and review,
* And directly connected to the assignment tasks:

  * Data preparation,
  * Treatment group definition,
  * Causal effect estimation with EconML,
  * Heterogeneity analysis,
  * Comparison with traditional methods.
