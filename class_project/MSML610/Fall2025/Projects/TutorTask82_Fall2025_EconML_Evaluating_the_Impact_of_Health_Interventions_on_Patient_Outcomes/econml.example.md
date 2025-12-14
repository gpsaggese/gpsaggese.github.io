# Evaluating the Impact of Dietary Supplement Use on Health Outcomes with EconML (NHANES 2021–2023)

MSML610 Project  
**TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes**

---

## 1. Overview

This tutorial walks through an **end-to-end causal inference workflow** using **NHANES 2021–2023** data and **EconML**.

**Main question**

> Does taking **any dietary supplement** affect:
> - **Mean systolic blood pressure** (`sbp_mean`)  
> - **Fasting glucose** (`fasting_glucose_mg_dl`)?

We treat this as a causal effect estimation problem:

- **Treatment (T):** `treatment_supplement` (1 = any supplement use, 0 = none)
- **Outcomes (Y):** `sbp_mean`, `fasting_glucose_mg_dl`
- **Covariates (X):** baseline health and lab markers (BMI, waist, lipids, hs-CRP, etc.)

We estimate:

- **ATE** (Average Treatment Effect)
- **CATE** (individual-level conditional effects) + simple heterogeneity summaries by BMI quartile
- A **baseline OLS** regression for comparison

The notebook version of this tutorial is `econml.example.ipynb`.  
This file explains the story and shows the core code used there.

---

## 2. Quickstart (how to run)

From the **project root**, run the example notebook (`econml.example.ipynb`).  
It relies only on the project’s internal API:

- `econml_utils.py`
- `econml.API.py`

> Note: The upstream library is also named `econml`. To avoid import confusion, the notebooks load `econml.API.py` **by file path** (dynamic import).

---

## 3. Data and Setup

### 3.1 NHANES components used

We start from cleaned NHANES files stored in `data/`:

- Blood pressure: `BPXO_L_meaningful.csv`
- Body measures: `BMX_L_meaningful.csv` (BMI, weight, waist)
- Lipids: `TCHOL_L_meaningful.csv`, `HDL_L_meaningful.csv`, `TRIGLY_L_meaningful.csv`
- Fasting glucose: `GLU_L_meaningful.csv`
- Inflammation: `HSCRP_L_meaningful.csv`
- Supplements: `DSQTOT_L_meaningful.csv`
- Demographics: `DEMO_L_meaningful.csv`

All files share a person-level ID:

- `respondent_sequence_number` (derived from NHANES `SEQN` in data prep)

---

### 3.2 Imports used in the example notebook

```python
import pathlib
import sys
import importlib.util

import pandas as pd
import matplotlib.pyplot as plt

from econml_utils import build_analysis_df, get_y_t_x
```

### 3.3 Load the project API (`econml.API.py`) safely

```python
project_dir = pathlib.Path.cwd()
api_path = project_dir / "econml.API.py"

spec = importlib.util.spec_from_file_location("econml_project_api", api_path)
econml_api = importlib.util.module_from_spec(spec)
sys.modules["econml_project_api"] = econml_api
spec.loader.exec_module(econml_api)
```

From here on, we call:

- `econml_api.run_sbp_supplement_experiment(...)`
- `econml_api.run_glucose_supplement_experiment(...)`
- `econml_api.run_ols_for_outcome(...)`

---

## 4. Build the merged analysis dataset

### 4.1 Create `analysis_df`

```python
analysis_df = build_analysis_df()
print(analysis_df.shape)
analysis_df.head()
```

**What this gives us**

- Each row is one NHANES participant
- Key columns include:
  - Outcomes: `sbp_mean`, `fasting_glucose_mg_dl`
  - Treatment: `treatment_supplement`
  - Covariates: BMI/waist, lipids, hs-CRP, etc.
  - Demographics: `age_in_years_at_screening`, `gender`, etc.

---

## 5. Define Y, T, and X

### 5.1 Extract Y/T/X for SBP with `get_y_t_x`

```python
y_sbp, t_supp, X_sbp, covariates = get_y_t_x(
    analysis_df,
    outcome_col="sbp_mean",
    treatment_col="treatment_supplement",
)

print(len(y_sbp), X_sbp.shape)
print(covariates)
```

**Interpretation**

- `y_sbp` is the outcome vector
- `t_supp` is treatment (0/1)
- `X_sbp` is the covariate matrix (baseline confounders)

This creates a clean interface so the modeling code stays consistent across outcomes.

---

## 6. EconML DRLearner: SBP as outcome

### 6.1 Run the SBP causal experiment

```python
sbp_results = econml_api.run_sbp_supplement_experiment(random_state=42)

sbp_results["ate_sbp"], sbp_results["ate_ci_low"], sbp_results["ate_ci_high"], sbp_results["n_obs"]
```

**Example run (your results)**

- **ATE (SBP):** ~ **-0.076 mmHg**
- **95% CI:** ~ **[-0.280, 0.128]**
- **n_obs:** **2638**

This suggests the average effect on SBP is **very small** and the CI includes 0.

### 6.2 Inspect individual CATEs

```python
cate_df = sbp_results["cate_df"]
tau_col = sbp_results["tau_col"]

cate_df[[ "respondent_sequence_number", "sbp_mean", "treatment_supplement", tau_col ]].head()
cate_df[tau_col].describe()
```

Interpretation: the model produces **individual-level effect estimates** (`tau_hat_sbp_mean`), which we can summarize and visualize.

### 6.3 BMI heterogeneity summary (SBP)

```python
sbp_results["bmi_effects"]
```

This reports mean CATE by BMI quartile (Q1 = leanest → Q4 = highest BMI).  
In your run, SBP effects vary a bit by BMI bin, but overall the ATE remains near 0.

> Note: `age_effects` is `None` in your run because the merged dataset uses
`age_in_years_at_screening` (not `age_years`), so the helper does not compute age bins.

---

## 7. EconML DRLearner: fasting glucose as outcome

### 7.1 Run the glucose causal experiment

```python
glucose_results = econml_api.run_glucose_supplement_experiment(random_state=42)

glucose_results["ate_glucose"], glucose_results["ate_ci_low"], glucose_results["ate_ci_high"], glucose_results["n_obs"]
```

**Example run (your results)**

- **ATE (glucose):** ~ **-1.887 mg/dL**
- **95% CI:** ~ **[-2.004, -1.762]**
- **n_obs:** **2674**

This is a clearer negative effect (CI does not cross 0), meaning supplement users are estimated to have **slightly lower fasting glucose on average**, after adjustment.

### 7.2 BMI heterogeneity summary (glucose)

```python
glucose_results["bmi_effects"]
```

In your run, the glucose effect becomes **more negative** as BMI increases (Q4 most negative).  
This is exactly what the heterogeneity summary is meant to surface.

---

## 8. Baseline comparison: OLS vs EconML

### 8.1 Run OLS for each outcome

```python
ols_sbp = econml_api.run_ols_for_outcome("sbp_mean")
ols_glu = econml_api.run_ols_for_outcome("fasting_glucose_mg_dl")

ols_sbp, ols_glu
```

These return the treatment coefficient and a robust CI (HC3).

### 8.2 Side-by-side comparison table

```python
comparison_df = pd.DataFrame([
    {
        "outcome": "sbp_mean",
        "econml_ate": sbp_results["ate_sbp"],
        "econml_ci_low": sbp_results["ate_ci_low"],
        "econml_ci_high": sbp_results["ate_ci_high"],
        "ols_treatment_coef": ols_sbp["treatment_coef"],
        "ols_ci_low": ols_sbp["treatment_ci_low"],
        "ols_ci_high": ols_sbp["treatment_ci_high"],
        "n_obs": sbp_results["n_obs"],
    },
    {
        "outcome": "fasting_glucose_mg_dl",
        "econml_ate": glucose_results["ate_glucose"],
        "econml_ci_low": glucose_results["ate_ci_low"],
        "econml_ci_high": glucose_results["ate_ci_high"],
        "ols_treatment_coef": ols_glu["treatment_coef"],
        "ols_ci_low": ols_glu["treatment_ci_low"],
        "ols_ci_high": ols_glu["treatment_ci_high"],
        "n_obs": glucose_results["n_obs"],
    },
])

comparison_df
```

**Interpretation (based on your run)**

- For **SBP**, EconML ATE is near 0 (slightly negative), while OLS is positive → the sign differs.
- For **glucose**, both EconML and OLS are negative → they agree in direction (EconML is stronger).

This is a good “teaching moment”: causal ML and linear regression can differ when assumptions differ or effects are small.

---

## 9. Summary

What this example demonstrates:

1. **Build** a clean merged NHANES dataset (`build_analysis_df`)
2. **Extract** consistent modeling inputs (`get_y_t_x`)
3. **Estimate causal effects** using EconML DRLearner (ATE + CATEs)
4. **Check heterogeneity** (here, BMI quartiles)
5. **Compare to a baseline OLS** model

The key design goal is that the notebooks stay clean: they call only the project API functions, and the modeling logic lives in `econml.API.py`.

