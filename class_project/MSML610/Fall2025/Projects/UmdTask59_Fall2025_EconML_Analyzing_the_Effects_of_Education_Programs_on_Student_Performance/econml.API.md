# EconML API Documentation

## 1. Overview

This document describes the API surface provided by `econml_utils.py` for working with EconML estimators in the context of student performance analysis.

The goal is to offer a small wrapper over EconML that:
- Standardizes how we configure and fit estimators,
- Provides helper functions for estimating average and heterogeneous treatment effects,
- Keeps the main notebooks readable and focused on the analysis logic.

## 2. Core Concepts

- **Outcome (Y)**: Target variable we want to improve (such as final grade).
- **Treatment (T)**: Binary variable indicating participation in an educational program.
- **Features (X)**: Student and family characteristics used to model treatment heterogeneity.
- **Controls (W)**: Additional covariates used for adjustment of confounding.

## 3. API Surface

The following objects and functions will be implemented in `econml_utils.py`:

- `EconMLEducationConfig` – configuration dataclass for specifying outcome, treatment, feature, and control columns, as well as the estimator type.
- `build_econml_estimator(config)` – creates an EconML estimator based on the config.
- `fit_econml_estimator(df, config)` – fits the estimator using data from a pandas DataFrame.
- `estimate_ate(model, df, config)` – estimates the Average Treatment Effect (ATE).
- `estimate_cate_by_subgroup(model, df, config, subgroup_col)` – computes subgroup level Conditional Average Treatment Effects (CATEs).

Further details and examples of usage are shown in `econml.API.ipynb`.

## 4. Estimator choices for the Student Performance dataset

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