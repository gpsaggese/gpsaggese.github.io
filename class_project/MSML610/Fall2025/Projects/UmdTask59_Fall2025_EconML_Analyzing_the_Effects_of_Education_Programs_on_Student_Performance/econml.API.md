# EconML API Documentation

## Overview

This project uses the [EconML](https://www.pywhy.org/EconML/) library to estimate causal treatment effects from observational data. EconML is designed for **heterogeneous treatment effect** estimation using modern machine learning methods. 

The file `econml_utils.py` provides a small, project-specific wrapper on top of EconML. The goals of this wrapper are to:

- Standardize how we configure and fit EconML estimators,
- Reduce boilerplate in the notebooks,
- Make it easier to compare different Double Machine Learning (DML) estimators.

The companion notebook `econml.API.ipynb` walks through the native EconML API.

## Core causal concepts

To use EconML, we must specify four key components:

- **Outcome (Y)** – the variable whose causal response we want to measure
- **Treatment (T)** – a binary indicator of whether a unit received the intervention
- **Features (X)** – covariates that may drive **heterogeneity** in the treatment effect
- **Controls (W)** – additional covariates used primarily to adjust for confounding

Double Machine Learning estimators in EconML assume **unconfoundedness**: once we condition on a rich set of observed covariates (X, W), treatment assignment is “as good as random.”

## API surface in `econml_utils.py`

### `EconMLEducationConfig`

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
Configuration object capturing modeling choices:

-   `outcome_col`: name of the outcome column 
-   `treatment_col`: name of the treatment column
-   `x_cols`: list of feature columns used for CATE heterogeneity.
-   `w_cols`: optional list of control columns used for confounding adjustment.
-   `estimator_type`: which EconML estimator to use:
    -   `"linear_dml"`, `"causal_forest"`
A helper function, `make_default_config()`, returns a reasonable default configuration for the Student Performance dataset.

### Model construction and fitting

-   `build_econml_estimator(config)`: Creates an unfit EconML estimator instance based on `config.estimator_type`.
-   `fit_econml_estimator(df, config, estimator=None)`: Splits the DataFrame into `(Y, T, X, W)` and calls `.fit(...)` on the estimator. If `estimator` is `None`, it calls `build_econml_estimator` internally.

### Effect estimation helpers

-   `estimate_ate(model, df, config)`: Returns a dictionary with the ATE estimate and 95% confidence interval, based on the model's `.ate(...)` and `.ate_interval(...)` methods.
-   `estimate_cate_by_subgroup(model, df, config, subgroup_col)`: Computes individual-level treatment effects via `.effect(X)` and then aggregates them by `subgroup_col`, returning group-level CATE summaries.

## Design decisions and limitations
-   The API is intentionally **minimal** and tailored to this project, as a result, it does not expose every EconML feature.
-   All estimators are configured with `discrete_treatment=True` to reflect the binary nature of the education program indicator.
-   The validity of the causal conclusions still depends on the usual DML assumptions, especially that we have measured all major confounders.
-   Additional extensions (e.g., custom nuisance models, cross-fitting options, alternative estimators like DR Learners) could be added in future work if more flexibility is needed.