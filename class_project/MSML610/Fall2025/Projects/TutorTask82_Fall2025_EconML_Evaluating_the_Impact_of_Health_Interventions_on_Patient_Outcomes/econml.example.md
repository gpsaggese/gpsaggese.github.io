

## 📄 `econml.example.md`

# EconML + NHANES Example – Evaluating Health Interventions (MSML610 Project)

## 1. Tutorial goal

This example notebook shows an end-to-end workflow for estimating the
causal effect of a **health intervention** on patient outcomes using:

- **Dataset:** NHANES 2021–2023
- **Technology:** [EconML](https://github.com/py-why/EconML) (Double Robust Learner)
- **Intervention:** Any dietary supplement use (yes/no)
- **Outcomes:**
  - average systolic BP (`sbp_mean`)
  - fasting glucose (`fasting_glucose_mg_dl`)

The goal is that a student can go from **cleaned NHANES CSVs** to
**treatment effect estimates and heterogeneity analysis** in about
60 minutes.

---

## 2. Prerequisites & setup

We assume:

- The repository has been cloned:
  - `class_project/MSML610/Fall2025/Projects/TutorTask82_...`
- A Python environment is created and required packages installed:
  - `econml`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, etc.
- Cleaned NHANES CSVs are available under `data/`:
  - `BPXO_L_meaningful.csv` (oscillometric blood pressure)
  - `BMX_L_meaningful.csv` (body measures)
  - `TCHOL_L_meaningful.csv` (total cholesterol)
  - `HDL_L_meaningful.csv` (HDL cholesterol)
  - `TRIGLY_L_meaningful.csv` (triglycerides & LDL)
  - `GLU_L_meaningful.csv` (fasting glucose)
  - `HSCRP_L_meaningful.csv` (hs-CRP)
  - `DSQTOT_L_meaningful.csv` (supplements)
  - `DEMO_L_meaningful.csv` (demographics & weights)

The cleaned CSVs are produced by two upstream notebooks:

- `MSML610_DataPreparation_Karthik.ipynb`
- `Data_Preparation_Sri.ipynb`

These notebooks are not part of the tutorial itself, but they document
the data cleaning workflow.

---

## 3. Step 1 – Build the merged NHANES analysis table

In `econml.example.ipynb`, we start by importing the utilities and
building the merged dataset:

```python
from econml_utils import build_analysis_df

analysis_df = build_analysis_df()
analysis_df.shape, analysis_df.head()
````

This call:

* Computes `sbp_mean` and `dbp_mean` from three oscillometric readings.
* Merges anthropometrics (BMI, weight, waist).
* Merges key labs (cholesterol, HDL, triglycerides, glucose, hs-CRP).
* Adds demographics (`age_years`, `sex`).
* Adds `treatment_supplement`:

  * 1 if `any_dietary_supplements_taken == 1`
  * 0 otherwise.

The resulting table has:

* ~7,800 participants
* One row per respondent
* All fields needed for causal analysis.

---

## 4. Step 2 – Define treatment, outcome, and covariates

We define the **intervention** as:

> Any dietary supplement use (treatment_supplement = 1)

and the **control** group as:

> No dietary supplement use (treatment_supplement = 0)

We use the helper function `get_y_t_x` to extract:

* `Y`: outcome (e.g., `sbp_mean`)
* `T`: treatment (`treatment_supplement`)
* `X`: baseline covariates (age, sex, BMI, lipids, glucose, hs-CRP)

Example:

```python
from econml_utils import get_y_t_x

y, t, X, covariate_cols = get_y_t_x(
    analysis_df,
    outcome_col="sbp_mean",
    treatment_col="treatment_supplement",
)

len(y), len(t), X.shape, covariate_cols
```

This step corresponds to the assignment’s requirement to **define
treatment groups** and **control for baseline conditions**.

---

## 5. Step 3 – Run EconML DRLearner for SBP

Instead of manually wiring DRLearner each time, we use the high-level
API function in `econml.API.py`:

```python
import importlib.util
from pathlib import Path

# Load econml.API.py as a module
api_path = Path.cwd() / "econml.API.py"
spec = importlib.util.spec_from_file_location("econml_api", api_path)
econml_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(econml_api)

# Run SBP experiment
sbp_results = econml_api.run_sbp_supplement_experiment()
sbp_results["ate_sbp"], sbp_results["age_effects"].round(2), sbp_results["bmi_effects"].round(2)
```

Internally, this function:

1. Rebuilds the merged dataset via `build_analysis_df`.
2. Extracts (`Y`, `T`, `X`) for `sbp_mean`.
3. Drops rows with missing data.
4. Fits a **DRLearner** with logistic regression propensity model.
5. Returns:

   * `ate_sbp`: overall ATE on systolic BP
   * `cate_df`: dataframe with `tau_hat_sbp` (individual CATEs)
   * `age_effects`: mean CATE by age quartiles
   * `bmi_effects`: mean CATE by BMI quartiles

**Typical results (rounded):**

* **ATE on SBP:** about −2.0 mmHg
  → Supplement users have ~2 mmHg lower SBP on average after adjustment.

* **Age heterogeneity:**

  * Q1 (youngest): small effect
  * Q4 (oldest): strongest effect (~−3.9 mmHg)

* **BMI heterogeneity:**

  * All BMI quartiles show similar effects (~−2 mmHg).

---

## 6. Step 4 – Heterogeneity analysis (age, BMI, sex)

Using `sbp_results["cate_df"]`, we can study **who benefits most**.

Example: SBP effect by age quartile.

```python
import matplotlib.pyplot as plt

age_effects = sbp_results["age_effects"].sort_index()

plt.figure(figsize=(5, 3.5))
age_effects.plot(kind="bar")
plt.axhline(0, linestyle="--", linewidth=1)
plt.ylabel("Estimated effect on SBP (mmHg)")
plt.xlabel("age_bin")
plt.title("Average supplement effect on SBP by age quartile")
plt.tight_layout()
plt.show()
```

The plot shows:

* All bars below zero → supplement use is associated with lower SBP.
* Stronger effects for older age groups.

Example: SBP effect by sex.

```python
cate_df = sbp_results["cate_df"].copy()

sex_map = {1: "Male", 2: "Female"}
cate_df["sex_label"] = cate_df["sex"].map(sex_map)

sex_effects = cate_df.groupby("sex_label")["tau_hat_sbp"].mean()
sex_effects
```

Observed pattern:

* Female: stronger BP-lowering effect (~−2.6 mmHg)
* Male: smaller effect (~−1.3 mmHg)

This directly addresses the assignment requirement for
**heterogeneity analysis** across demographics and health profiles.

---

## 7. Step 5 – Second outcome: fasting glucose

We repeat the analysis for fasting glucose using the second API function:

```python
glucose_results = econml_api.run_glucose_supplement_experiment()

glucose_results["ate_glucose"], \
glucose_results["age_effects"].round(2), \
glucose_results["bmi_effects"].round(2)
```

Findings:

* **ATE on fasting glucose:** essentially 0
* **Age/BMI heterogeneity:** all quartile effects also near 0

Interpretation:

> In this dataset, supplement use is **not** associated with a
> meaningful change in fasting glucose, even after adjusting for
> confounders. This contrasts with the modest but consistent effect
> observed for systolic blood pressure.

---

## 8. Step 6 – Bonus: compare EconML with OLS

To satisfy the “Bonus Ideas” requirement, we compare EconML’s ATEs
with traditional linear regression:

```python
ols_sbp = econml_api.run_ols_for_outcome("sbp_mean")
ols_glu = econml_api.run_ols_for_outcome("fasting_glucose_mg_dl")

comparison = pd.DataFrame(
    {
        "outcome": ["sbp_mean", "fasting_glucose_mg_dl"],
        "econml_ate": [sbp_results["ate_sbp"], glucose_results["ate_glucose"]],
        "ols_treatment_coef": [ols_sbp["treatment_coef"], ols_glu["treatment_coef"]],
    }
)

comparison.round(3)
```

Typical results:

* **SBP:**

  * EconML ATE ≈ −2.02 mmHg
  * OLS coef ≈ −1.98 mmHg

* **Fasting glucose:**

  * EconML ATE ≈ 0.00 mg/dL
  * OLS coef ≈ 0.00 mg/dL

Conclusion:

* For **average effects**, EconML and OLS agree closely.
* The added value of EconML is in:

  * **Double robustness**
  * **Individual CATEs**
  * Rich heterogeneity summaries (age, BMI, sex)

---

## 9. Step 7 – Summary of what the reader learns

By the end of `econml.example.ipynb`, a reader can:

1. Understand how to **build a merged analysis dataset** from multiple
   NHANES components using `econml_utils.py`.
2. Define a clear **treatment** (any supplements) and **control** group.
3. Use `econml.API.py` to:

   * Estimate the treatment effect on SBP with `run_sbp_supplement_experiment`.
   * Estimate the treatment effect on fasting glucose with
     `run_glucose_supplement_experiment`.
4. Analyze **heterogeneous treatment effects** by:

   * Age quartiles
   * BMI quartiles
   * Sex
5. Compare EconML’s treatment effect estimates with a **traditional OLS
   baseline** via `run_ols_for_outcome`.

This makes the tutorial a complete, hands-on introduction to using
EconML for causal inference on real-world health data (NHANES),
matching the project’s goal of helping a classmate ramp up in
~60 minutes.

```
