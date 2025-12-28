# EconML Example: Analyzing the Effects of Education Programs on Student Performance

## Problem Statement
We analyze the impact of a student education program on final course grades using the UCI Student Performance dataset. The main goal is to estimate:
- The overall impact of participating in the program on the population's final grades (ATE),
- How this effect varies across different demographic and socioeconomic groups (CATEs).

## Dataset Summary
We use the **Student Performance** dataset (from UCI ML Repository + Portuguese secondary school, math or Portuguese course).  
The dataset includes:
- Student demographics (sex, age, family structure),
- Family background and parental education,
- Study related and behavioral attributes,
- Final grades (G1, G2, G3) where G1 and G2 are 1st and 2nd period grades, and G3 is the final grade issued at the end of the 3rd period.

## Planned Analysis Workflow
1. Load and clean the student performance data.
2. Define treatment and outcome variables.
3. Fit EconML estimators using the wrapper functions in `econml_utils.py`.
4. Estimate ATE and CATEs.
5. Visualize heterogeneity across demographic groups.
6. Discuss robustness and limitations.

The complete implementation is in `econml.example.ipynb`.

## Data Model and Design Decisions

### Defining Causal Components (Y, T, W, X)
The strength of EconML lies in explicitly defining the components of the causal model based on observed data:
-   **Outcome (Y):** The final grade in the subject, G3. This is the variable we aim to affect.

-   **Treatment (T):** A binary variable representing participation in the educational support program (derived from variables like `extra_educational_support`).

-   **Control Variables (W):** A comprehensive set of variables used to *control* for confounding factors and satisfy the unconfoundedness assumption. This included student context (e.g., `famsize`, `Pstatus`, parents' jobs) and school-related variables (e.g., `school_support`, `studytime`).

-   **Heterogeneity Variables (X):** The specific student characteristics used to model the variation in the treatment effect τ(X). For this project, a small, impactful set was selected, likely including key metrics like the number of previous **failures** and **age**, as these are strong proxies for a student's at-risk status.

**Causal Question:** What is the causal effect, τ, of receiving educational support on a student's final grade (G3), and how does this effect τ(X) vary across observable student characteristics X?

### Wrapper-based pattern

```
df_raw = load_student_data(source="ucimlrepo")
df = clean_student_data(df_raw)
config = make_default_config()            # defines Y, T, X, W
config.estimator_type = "linear_dml"      # or "causal_forest", "sparse_linear_dml"
model = fit_econml_estimator(df, config)
ate_result = estimate_ate(model, df, config)
cate_by_sex = estimate_cate_by_subgroup(model, df, config, subgroup_col="sex")
```
This allows the **analysis notebooks** to switch between estimators by only changing `config.estimator_type`, without rewriting the fitting and post-processing logic.

### Estimator Selection: Double Machine Learning (DML)

For this project we work with **observational** student data rather than a randomized experiment. The educational support program (`schoolsup`) was not randomly assigned, and many observed variables (family background, prior performance, study behavior) can influence **both** program participation and final grades. At the same time, the dataset is fairly rich, with 30+ features that can serve as potential confounders.

Given this setup, we use **Double Machine Learning (DML)** estimators from EconML for: 

- **Observational data under unconfoundedness**: DML assumes that, conditional on a rich set of observed covariates, treatment assignment is “as good as random.” This fits our setting where demographics, family variables and prior grades are all measured.
- **High-dimensional controls**: DML can flexibly model the relationship between covariates and both treatment and outcome using machine learning (e.g., forests, regularized regression) and then estimate treatment effects on residualized targets.
- **Heterogeneous treatment effects (CATE)**: all chosen estimators directly target CATEs, which matches our goal of finding which student demographics benefit the most from the program.

1. **LinearDML (primary / baseline)**
   - Models the CATE as a **linear function** of student characteristics.
   - Very **interpretable**: coefficients tell us how the program effect changes with features like parent education, study time, or absences.
   - Supports inference: standard errors and confidence intervals for the estimated effects.
   - Well-suited as a first pass for explaining results to non-technical stakeholders (e.g., educators and policy makers).

2. **CausalForestDML (flexible non-linear alternative)**
   - Uses a **forest-based** final stage to model complex, non-linear heterogeneity in treatment effects.
   - Can automatically discover subgroups with different responses to the program, without specifying an explicit functional form.
   - Useful to check whether the linear assumptions of LinearDML are overly restrictive on this dataset.

Both estimators share the same high-level DML workflow:

1. **Nuisance modeling**: use ML models to predict treatment and outcome from covariates.
2. **Orthogonalization / residualization**: subtract these predictions to get “residual” treatment and outcome.
3. **Effect estimation**: fit a final-stage model on residuals to estimate causal effects that are robust to small errors in the nuisance models.

## Results

### Figure 1. Unadjusted grades by treatment group (schoolsup)

<img width="590" height="380" alt="figure1" src="https://github.com/user-attachments/assets/62026cc2-c910-46ef-9e00-abec31a71ef9" />

This boxplot compares the distribution of final grades `G3` between:
- `schoolsup = 0` (no extra support)
- `schoolsup = 1` (extra support)

Visually, the **median** and overall distribution for the extra-support group (`schoolsup = 1`) appear slightly **lower** than the no-support group. If we stopped here, we might incorrectly conclude that the program “hurts” performance.

However, this plot reflects an **unadjusted observational comparison**. In real settings, extra support is often **targeted** toward students who are already at academic risk. That means treated students may differ systematically from controls (selection / confounding). This motivates using EconML’s DML estimators to adjust for confounders and estimate a causal effect rather than a raw association.

### Figure 2. Estimated Average Treatment Effect (ATE) of school support on final grade

<img width="680" height="487" alt="figure2" src="https://github.com/user-attachments/assets/3f4fed9a-f22d-4d74-b754-c9e56807338f" />

We estimate the **average causal effect** of extra support (`schoolsup`) on `G3` using two Double ML estimators:

- **LinearDML:** ATE = **0.015**
  - 95% CI: **[-0.364, 0.395]**
- **CausalForestDML:** ATE = **0.038**
  - 95% CI: **[-1.007, 1.083]**

**Interpretation (grade scale):** `G3` is on a 0–20 scale. An ATE of **0.015 – 0.038** corresponds to far less than **0.1 grade point**, which is practically negligible.

**Interpretation (percent scale):**
- As a fraction of the full 0–20 grade scale, the estimated average effect is approximately:
  - LinearDML: **0.015 / 20 ≈ 0.08%**
  - CausalForestDML: **0.038 / 20 ≈ 0.19%**

**Conclusion:** Both estimators are centered very close to zero and their confidence intervals include 0, meaning we do **not** find strong evidence that the support program has a large average effect on final grades in this dataset. The wide CI (especially for the forest) also suggests limited precision, which is plausible given that the treated group is relatively small.

### Figure 3. Heterogeneous effects by mother’s education (Medu)

<img width="697" height="386" alt="figure3" src="https://github.com/user-attachments/assets/7615158b-9bc0-43b0-ae5b-3306e4577a4b" />

To identify **which demographics benefit most**, we examine heterogeneity by mother’s education (`Medu`, 0 – 4), a common proxy for socioeconomic background.

Both estimators show a clear **gradient**:
- Students with **lower maternal education** (Medu = 0–1) show **more positive** estimated effects.
- Students with **higher maternal education** (Medu = 4) show **near-zero or negative** effects.

Using the subgroup summary (CausalForestDML), the estimated mean CATEs are approximately:
- Medu = 0: **+0.388** (n = 6)
- Medu = 1: **+0.310** (n = 143)
- Medu = 2: **+0.047** (n = 186)
- Medu = 3: **+0.073** (n = 139)
- Medu = 4: **–0.235** (n = 175)

**Interpretation:** This pattern suggests the program may be **most beneficial for students from lower-education household backgrounds**, consistent with a policy intuition: students with less academic support at home may gain slightly more from school provided support.

**Caution:** The Medu = 0 subgroup has **very small sample size (n=6)**, so its estimate is especially noisy. The most reliable positive signal comes from Medu = 1 (n=143), which still shows a meaningful positive estimated effect relative to other groups.

### Figure 4. Heterogeneous effects by higher-education plans (higher)

<img width="697" height="386" alt="figure4" src="https://github.com/user-attachments/assets/9b529e86-9885-4010-a119-a5ffd5bea785" />

We also examine heterogeneity by students’ plans for higher education (`higher`, 0 = no, 1 = yes). This variable is useful because it can reflect motivation, academic trajectory, and baseline risk.

The subgroup CATEs suggest a strong split:
- **higher = 0 (no plans for higher education):** positive effect
- **higher = 1 (plans for higher education):** near zero effect

From the subgroup summary (CausalForestDML):
- higher = 0: **+0.413** (n = 69)
- higher = 1: **–0.007** (n = 580)

**Interpretation:** Students who **do not plan to pursue higher education** appear to benefit more from extra support (on the order of ~0.4 grade points on average). For students already on a higher-education track, the estimated effect is essentially zero.

This result directly supports the project goal of “optimizing for identifying who benefits most,” because it points to a concrete group that may be prioritized if resources are limited, while still acknowledging that estimates are model based and depend on the observational assumptions.

### Table 1. Ranked subgroup benefits (CATE ranking)

<img width="368" height="358" alt="table1" src="https://github.com/user-attachments/assets/33680f98-fc9a-48fb-a4d9-e1c5d1f99960" />

To explicitly answer “which demographics benefit most,” we aggregate subgroup CATE summaries across multiple features (`higher`, `Medu`, `address`, `sex`) and **rank subgroups by estimated mean CATE** (highest to lowest).

Top-ranked subgroups include:
1. `higher = 0` -> **+0.413** (n = 69)  
2. `Medu = 1` -> **+0.310** (n = 143)  
3. `address = 0` (rural) -> **+0.143** (n = 197)  
4. `sex = 1` -> **+0.064** (n = 266)  

(An even higher CATE appears for `Medu = 0` at +0.388, but it has **n = 6**, so it is not as reliable.)

**What this means:** The most practically actionable “benefit signals” come from subgroups that combine:
- **positive mean CATE**, and
- **reasonable subgroup size**.

By that standard, the strongest candidates for “who benefits most” are:
- students with **no higher-education plans** (`higher = 0`),
- students with **lower maternal education** (`Medu = 1`), and
- **rural** students (`address = 0`).

**Uncertainty note:** The subgroup CATE standard deviations (≈0.36–0.45) are larger than many mean effects, indicating substantial individual-level variation. These subgroup rankings should be interpreted as **suggestive prioritization directions**, not definitive targeting rules.

**Objective answer:** On average, extra school support shows a near-zero ATE on final grades, but heterogeneity analysis suggests the program may be most beneficial for students with **no higher-education plans**, those from **lower maternal education** backgrounds, and **rural** students, i.e., groups plausibly at higher academic risk.
