# EconML API Documentation

## 1. Overview

This project uses the [EconML](https://www.pywhy.org/EconML/) library to estimate causal treatment effects from observational data. EconML is designed for **heterogeneous treatment effect** estimation using modern machine learning methods. 

The file `econml_utils.py` provides a small, project-specific wrapper on top of EconML. The goals of this wrapper are to:

- Standardize how we configure and fit EconML estimators,
- Reduce boilerplate in the notebooks,
- Make it easier to compare different Double Machine Learning (DML) estimators.

The companion notebook `econml.API.ipynb` walks through the native EconML API and then demonstrates this wrapper in action.

## 2. Core causal concepts

To use EconML, we must specify four key components:

- **Outcome (Y)** – the variable whose causal response we want to measure (here: final grade `G3` in the Student Performance dataset).
- **Treatment (T)** – a binary indicator of whether a unit received the intervention (school support program, `schoolsup`).
- **Features (X)** – covariates that may drive **heterogeneity** in the treatment effect (e.g., demographics, parental education, study habits).
- **Controls (W)** – additional covariates used primarily to adjust for confounding (e.g., prior grades `G1`, `G2`, and other support flags).

Double Machine Learning estimators in EconML assume **unconfoundedness**: once we condition on a rich set of observed covariates (X, W), treatment assignment is “as good as random.”

## 3. Estimator choices for the Student Performance dataset

For this project we work with **observational** student data rather than a randomized experiment. The educational support program (`schoolsup`) was not randomly assigned, and many observed variables (family background, prior performance, study behavior) can influence **both** program participation and final grades. At the same time, the dataset is fairly rich, with 30+ features that can serve as potential confounders.

Given this setup, we use **Double Machine Learning (DML)** estimators from EconML, which are specifically designed for:

- **Observational data under unconfoundedness**: DML assumes that, conditional on a rich set of observed covariates, treatment assignment is “as good as random.” This fits our setting where demographics, family variables and prior grades are all measured.
- **High-dimensional controls**: DML can flexibly model the relationship between covariates and both treatment and outcome using machine learning (e.g., forests, regularized regression) and then estimate treatment effects on residualized targets.
- **Heterogeneous treatment effects (CATE)**: all chosen estimators directly target CATEs, which matches our goal of finding which student demographics benefit the most from the program.

We focus on three DML estimators:

1. **LinearDML (primary / baseline)**
   - Models the CATE as a **linear function** of student characteristics.
   - Very **interpretable**: coefficients tell us how the program effect changes with features like parent education, study time, or absences.
   - Supports inference: standard errors and confidence intervals for the estimated effects.
   - Well-suited as a first pass for explaining results to non-technical stakeholders (e.g., educators and policy makers).

2. **CausalForestDML (flexible non-linear alternative)**
   - Uses a **forest-based** final stage to model complex, non-linear heterogeneity in treatment effects.
   - Can automatically discover subgroups with different responses to the program, without specifying an explicit functional form.
   - Useful to check whether the linear assumptions of LinearDML are overly restrictive on this dataset.

3. **SparseLinearDML (feature-selection-focused)**
   - Specialized for **sparse linear CATE**: assumes only a subset of features actually moderate the treatment effect.
   - Uses L1-type regularization to perform automatic feature selection in the CATE model.
   - Helps answer: *Which demographic or academic variables truly matter for treatment effect heterogeneity?*

All three estimators share the same high-level DML workflow:

1. **Nuisance modeling**: use ML models to predict treatment and outcome from covariates.
2. **Orthogonalization / residualization**: subtract these predictions to get “residual” treatment and outcome.
3. **Effect estimation**: fit a final-stage model on residuals to estimate causal effects that are robust to small errors in the nuisance models.

Because they all follow the same EconML API (`fit`, `effect`, `ate`, etc.), the project’s wrapper functions (`build_econml_estimator`, `fit_econml_estimator`,`estimate_ate`, `estimate_cate_by_subgroup`) can switch between estimators with a single configuration flag.


## 4. API surface in `econml_utils.py`

### 4.1 `EconMLEducationConfig`

```
python
@dataclass
class EconMLEducationConfig:
    outcome_col: str
    treatment_col: str
    x_cols: List[str]
    w_cols: Optional[List[str]] = None
    estimator_type: str = "linear_dml"
```
Configuration object capturing all project-specific modeling choices:

-   `outcome_col`: name of the outcome column (e.g., `"G3"`).
-   `treatment_col`: name of the treatment column (e.g., `"schoolsup"`).
-   `x_cols`: list of feature columns used for CATE heterogeneity.
-   `w_cols`: optional list of control columns used for confounding adjustment.
-   `estimator_type`: which EconML estimator to use:
    -   `"linear_dml"`, `"causal_forest"`, or `"sparse_linear_dml"`.
A helper function, `make_default_config()`, returns a reasonable default configuration for the Student Performance dataset.

### 4.2 Model construction and fitting

-   `build_econml_estimator(config)`: Creates an unfit EconML estimator instance based on `config.estimator_type`.
-   `fit_econml_estimator(df, config, estimator=None)`: Splits the DataFrame into `(Y, T, X, W)` and calls `.fit(...)` on the estimator. If `estimator` is `None`, it calls `build_econml_estimator` internally.

### 4.3 Effect estimation helpers

-   `estimate_ate(model, df, config)`: Returns a dictionary with the ATE estimate and 95% confidence interval, based on the model's `.ate(...)` and `.ate_interval(...)` methods.
-   `estimate_cate_by_subgroup(model, df, config, subgroup_col)`: Computes individual-level treatment effects via `.effect(X)` and then aggregates them by `subgroup_col`, returning group-level CATE summaries.

### 4.4 Data-related utilities

-   `load_student_data(source="ucimlrepo", local_path=None)`: Loads the Student Performance dataset either via the `ucimlrepo` package (dataset ID 320) or from a local CSV file. [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance)
-   `clean_student_data(df)`: Applies light cleaning and encoding (e.g., mapping `yes`/`no` to 0/1).
-   `summarize_treatment(df, config)`: Returns a small table summarizing counts and average outcomes by treatment status.

## 5. Typical usage patterns (as used in the notebook)

### 5.1 Native EconML pattern

```
est = LinearDML(discrete_treatment=True, random_state=42)
est.fit(Y, T, X=X, W=W)
ate = est.ate(X=X)
```

### 5.2 Wrapper-based pattern

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

## 6. Design decisions and limitations

-   The API is intentionally **minimal** and tailored to this project, as a result, it does not expose every EconML feature.
-   All estimators are configured with `discrete_treatment=True` to reflect the binary nature of the education program indicator.
-   The validity of the causal conclusions still depends on the usual DML assumptions, especially that we have measured all major confounders.
-   Additional extensions (e.g., custom nuisance models, cross-fitting options, alternative estimators like DR Learners) could be added in future work if more flexibility is needed.