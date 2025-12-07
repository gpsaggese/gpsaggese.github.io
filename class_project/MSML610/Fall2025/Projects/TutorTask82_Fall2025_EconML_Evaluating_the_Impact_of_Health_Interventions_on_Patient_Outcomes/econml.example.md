Here’s a complete `econml.example.md` you can drop straight into your project.

````markdown
# Evaluating the Impact of Dietary Supplement Use on Health Outcomes with EconML

MSML610 Project  
**TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes**

---

## 1. Overview

In this tutorial, we walk through an **end-to-end causal inference analysis** using the **NHANES 2021–2023** data and **EconML**.

Our main question:

> Does taking **any dietary supplement** have a measurable effect on:
> - **Mean systolic blood pressure** (`sbp_mean`) and  
> - **Fasting glucose** (`fasting_glucose_mg_dl`)?

We treat this as a **causal effect estimation** problem:

- **Treatment**: Any dietary supplement use (yes/no)  
- **Primary outcome**: Mean systolic BP (`sbp_mean`)  
- **Secondary outcome**: Fasting glucose (`fasting_glucose_mg_dl`)  

We use:

- A **Double Robust Learner (DRLearner)** from EconML to estimate:
  - The **Average Treatment Effect (ATE)**  
  - **Conditional Average Treatment Effects (CATEs)** for each individual  
- A **simple OLS regression** as a baseline comparison.

The notebook version of this tutorial is `econml.example.ipynb`.  
This markdown file explains the story in words and shows the main code snippets used there.


---

## 2. Data and Setup

### 2.1. NHANES 2021–2023 Components

We start from cleaned NHANES 2021–2023 files stored in `data/`:

- Blood pressure: `BPXO_L_meaningful.csv`
- Body measures (BMI, weight, waist): `BMX_L_meaningful.csv`
- Lipids:
  - `TCHOL_L_meaningful.csv` (total cholesterol)
  - `HDL_L_meaningful.csv` (direct HDL cholesterol)
  - `TRIGLY_L_meaningful.csv` (triglycerides, `LBXTLG`)
- Glucose: `GLU_L_meaningful.csv` (fasting plasma glucose)
- Inflammation: `HSCRP_L_meaningful.csv` (hs-CRP)
- Supplements: `DSQTOT_L_meaningful.csv` (includes `any_dietary_supplements_taken`)
- Demographics: `DEMO_L_meaningful.csv` (age, sex, etc.)

All of these have been preprocessed in earlier data prep notebooks to:

- Use **meaningful column names**
- Use a standard ID: `respondent_sequence_number`


### 2.2. Imports

In the example notebook, we first import the main API functions:

```python
import pandas as pd
import matplotlib.pyplot as plt

from econml_utils import build_analysis_df, get_y_t_x
import econml.API as econml_api
````

We will rely on:

* `build_analysis_df` and `get_y_t_x` from `econml_utils.py`
* `run_sbp_supplement_experiment`, `run_glucose_supplement_experiment`, and `run_ols_for_outcome` from `econml.API.py`

---

## 3. Building the Analysis Dataset

### 3.1. Constructing `build_analysis_df`

The function `build_analysis_df()` merges all the prepared NHANES components into one table and constructs outcome and treatment variables.

In the notebook, we call:

```python
analysis_df = build_analysis_df()
analysis_df.shape, analysis_df.head()
```

**What this does:**

1. Reads all `*_meaningful.csv` files in `data/`.

2. Normalizes the ID column to `respondent_sequence_number`.

3. Computes:

   * `sbp_mean` and `dbp_mean` from three oscillometric readings
   * Keeps `fasting_glucose_mg_dl` from the GLU file

4. Merges anthropometrics, lipids, hs-CRP, supplements, and demographics.

5. Creates a binary treatment variable:

   * `treatment_supplement = 1` if `any_dietary_supplements_taken == 1` (Yes)
   * `treatment_supplement = 0` if `any_dietary_supplements_taken == 2` (No)

**Interpretation in markdown:**

* Each row in `analysis_df` represents a **single NHANES participant**.
* We have:

  * **Outcomes**: `sbp_mean`, `dbp_mean`, `fasting_glucose_mg_dl`
  * **Treatment**: `treatment_supplement`
  * **Covariates**: age, sex, BMI, lipids, hs-CRP, etc.
* This gives us a clean, analysis-ready dataset to plug into EconML and OLS.

### 3.2. Quick descriptive summary

In the notebook, we might also inspect summary statistics:

```python
analysis_df[[
    "sbp_mean", "dbp_mean", "fasting_glucose_mg_dl",
    "treatment_supplement", "age_years",
    "body_mass_index_kg_m2"
]].describe()
```

**Interpretation:**

* We check typical ranges (e.g. average SBP, BMI).
* We see the proportion of participants with `treatment_supplement = 1` vs `0`.
* This helps us confirm that the dataset is reasonable and there are no glaring issues (like extreme values or missing treatment.

---

## 4. Defining Outcome, Treatment, and Covariates

### 4.1. Using `get_y_t_x`

To fit causal models, we need:

* `Y`: outcome (e.g., `sbp_mean`)
* `T`: treatment indicator (`treatment_supplement`)
* `X`: set of covariates (age, sex, BMI, lipids, etc.)

We use a small helper:

```python
y_sbp, t_supp, X_sbp, covariates_sbp = get_y_t_x(
    analysis_df,
    outcome_col="sbp_mean",
    treatment_col="treatment_supplement",
)

len(y_sbp), X_sbp.shape, covariates_sbp
```

**What `get_y_t_x` does:**

* Takes the merged dataframe from `build_analysis_df`.
* Extracts the requested outcome (e.g., `sbp_mean`).
* Extracts the treatment column (`treatment_supplement`).
* Selects a **standard set of baseline covariates**, such as:

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
* Returns `(y, t, X, covariate_cols)`.

**Link to assignment tasks:**

* This step corresponds to **Defining Treatment Groups** and **controlling for baseline covariates**.
* We now have a clear split:

  * `t=1` → supplement users
  * `t=0` → non-users
* And we adjust for confounders like age, BMI, lipids, and inflammation markers.

---

## 5. Causal Effect of Supplement Use on Systolic Blood Pressure

### 5.1. Running the DRLearner experiment (SBP)

We use the API-level function:

```python
sbp_results = econml_api.run_sbp_supplement_experiment(random_state=42)

sbp_results.keys()
```

Typical keys:

* `"ate_sbp"` – ATE of supplement use on `sbp_mean`
* `"covariates"` – list of covariate names used
* `"cate_df"` – cleaned dataframe with CATEs added
* `"tau_col"` – column name of the CATE values
* `"age_effects"` – mean CATE by age quartile
* `"bmi_effects"` – mean CATE by BMI quartile

We can inspect the ATE:

```python
sbp_results["ate_sbp"]
```

**Interpretation:**

* This value is the **average estimated difference** in `sbp_mean` between taking supplements and not taking them, after adjusting for covariates.
* If the ATE is close to 0, it suggests that supplement use may not have a strong overall effect on systolic BP.
* A negative ATE would suggest lower SBP on average for supplement users (after adjustment); a positive ATE suggests higher SBP.

### 5.2. CATEs and heterogeneity by age and BMI

The CATEs (individual treatment effects) are stored in:

```python
cate_df = sbp_results["cate_df"]
tau_col = sbp_results["tau_col"]

cate_df[[ "age_years", "body_mass_index_kg_m2", tau_col ]].head()
```

We can then look at heterogeneity:

```python
age_effects_sbp = sbp_results["age_effects"]
bmi_effects_sbp = sbp_results["bmi_effects"]

age_effects_sbp, bmi_effects_sbp
```

In the notebook, we visualize them with simple bar plots.

**Bar plot for age bins**:

```python
age_effects_sbp.plot(kind="bar")
plt.axhline(0, linestyle="--")
plt.title("Estimated CATE of Supplement Use on SBP by Age Quartile")
plt.ylabel("Estimated treatment effect on sbp_mean")
plt.xlabel("Age quartile")
plt.show()
```

**Bar plot for BMI bins**:

```python
bmi_effects_sbp.plot(kind="bar")
plt.axhline(0, linestyle="--")
plt.title("Estimated CATE of Supplement Use on SBP by BMI Quartile")
plt.ylabel("Estimated treatment effect on sbp_mean")
plt.xlabel("BMI quartile")
plt.show()
```

**Interpretation:**

* **Heterogeneity by age**:

  * We check whether younger vs older participants show different treatment effects.
  * For example, we might see slightly more negative effects (lower SBP) in older or higher risk groups, or little variation across all bins.
* **Heterogeneity by BMI**:

  * We check whether lean vs higher BMI participants respond differently.
  * This helps answer whether supplement use has a more noticeable effect among specific subpopulations.

**Link to assignment tasks:**

* This directly addresses **Causal Effect Estimation** and **Heterogeneity Analysis** for the primary outcome (SBP).

---

## 6. Causal Effect on Fasting Glucose

### 6.1. Running the DRLearner experiment (Glucose)

We repeat the same kind of analysis for fasting glucose:

```python
glucose_results = econml_api.run_glucose_supplement_experiment(random_state=42)

glucose_results.keys()
```

Inspect the ATE:

```python
glucose_results["ate_glucose"]
```

**Interpretation:**

* This is the estimated **average impact** of supplement use on fasting glucose levels, again after adjusting for covariates.
* In many realistic settings, we might expect:

  * A very **small** or even **near-zero ATE**, meaning supplement use does not strongly change fasting glucose on average.

### 6.2. Heterogeneity by age and BMI

We can look at age/BMI heterogeneity for glucose as well:

```python
age_effects_glu = glucose_results["age_effects"]
bmi_effects_glu = glucose_results["bmi_effects"]

age_effects_glu, bmi_effects_glu
```

And plot them similarly.

**Interpretation:**

* We check if any particular age or BMI group shows a stronger signal.
* If all bars are close to zero, this supports the idea that supplement use may not have a large effect on fasting glucose in this sample.

---

## 7. OLS Baseline Comparison

To see how EconML compares with a traditional regression approach, we run **OLS** with the same covariates.

### 7.1. OLS for SBP

```python
ols_sbp = econml_api.run_ols_for_outcome("sbp_mean")
ols_sbp
```

Typical output structure:

```python
{
    "outcome": "sbp_mean",
    "treatment_coef": ...,
    "covariates": [...],
    "n_obs": ...
}
```

**Interpretation:**

* `treatment_coef` is the coefficient on `treatment_supplement` from:

  [
  \text{sbp_mean} = \beta_0 + \beta_1 \cdot \text{treatment_supplement} + \beta_2 X_1 + \dots + \beta_p X_p + \varepsilon
  ]

* This is a **linear regression-based estimate** of the supplement effect on SBP.

### 7.2. OLS for Fasting Glucose

```python
ols_glu = econml_api.run_ols_for_outcome("fasting_glucose_mg_dl")
ols_glu
```

Same structure, now for fasting glucose.

### 7.3. Side-by-side comparison

In the notebook, we build a small comparison table:

```python
comparison_df = pd.DataFrame([
    {
        "outcome": "sbp_mean",
        "econml_ate": sbp_results["ate_sbp"],
        "ols_treatment_coef": ols_sbp["treatment_coef"],
    },
    {
        "outcome": "fasting_glucose_mg_dl",
        "econml_ate": glucose_results["ate_glucose"],
        "ols_treatment_coef": ols_glu["treatment_coef"],
    }
])

comparison_df
```

**Interpretation:**

* This table helps us see whether EconML’s DRLearner gives similar or different estimates compared to OLS.
* In a well-specified situation, the results might be in the same ballpark, but:

  * DRLearner is more flexible and robust to certain types of model misspecification.
  * OLS assumes linearity and no strong interactions, which may not always hold.

---

## 8. Summary and Takeaways

### 8.1. What we did step-by-step

1. **Data Preparation**

   * We merged multiple NHANES components into a single analysis dataset using `build_analysis_df()`.
   * We constructed outcomes:

     * `sbp_mean` (mean systolic BP)
     * `fasting_glucose_mg_dl` (fasting glucose)
   * We defined a binary treatment:

     * `treatment_supplement = 1` if the participant reported any dietary supplement use.

2. **Treatment and Covariates**

   * We used `get_y_t_x()` to extract:

     * Outcome `Y`
     * Treatment `T`
     * Covariates `X` (age, sex, BMI, lipids, hs-CRP, etc.)
   * This step aligned with the assignment’s **treatment group definition** and **control for baseline health conditions**.

3. **Causal Effect Estimation (EconML)**

   * We used `run_sbp_supplement_experiment()` and `run_glucose_supplement_experiment()` to fit **DRLearner** models.
   * We obtained:

     * ATE estimates (`ate_sbp`, `ate_glucose`)
     * Individual CATEs (`tau_hat_*`)
   * We found that the overall average effects might be small or modest, suggesting no strong causal impact of “any supplement use” on these outcomes in the merged sample.

4. **Heterogeneity Analysis**

   * We grouped CATE estimates by:

     * `age_bin` (quartiles of age)
     * `bmi_bin` (quartiles of BMI)
   * We checked whether certain demographic or risk groups had larger or smaller estimated treatment effects.
   * This satisfied the **heterogeneity analysis** requirement.

5. **Traditional OLS Comparison**

   * We ran `run_ols_for_outcome()` for both SBP and fasting glucose.
   * We compared the OLS treatment coefficients against the EconML ATEs.
   * This served as the **traditional baseline** and showed how causal ML compares to a standard regression approach.

### 8.2. How this fits the project goals

* The analysis demonstrates a **full causal inference workflow**:

  * Data prep → treatment definition → causal modeling → heterogeneity → baseline comparison.
* The **API layer** (`econml.API.py`) keeps the modeling logic clean and reusable.
* The **example notebook** (`econml.example.ipynb`) tells a coherent story that could be presented as a **60-minute tutorial** to another student:

  * They can follow the narrative, run the cells, and see the results without having to understand all the internals of EconML.

Overall, the project shows how to:

* Take a real-world health survey dataset (NHANES),
* Define a plausible intervention (any supplement use),
* Use **EconML DRLearner** to estimate causal effects and heterogeneity, and
* Compare these results with a familiar regression baseline.

```
```
