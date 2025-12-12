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