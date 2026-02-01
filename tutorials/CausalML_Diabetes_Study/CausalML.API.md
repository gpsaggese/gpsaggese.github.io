<!-- toc -->

- [Summary](#summary)
- [CausalML API Documentation](#causalml-api-documentation)
  * [Installation and Docker Setup](#installation-and-docker-setup)
    + [1. Build the Image](#1-build-the-image)
    + [2. Run the Container](#2-run-the-container)
    + [3. Access the Project](#3-access-the-project)
  * [Overview](#overview)
  * [Architecture](#architecture)
  * [Class Reference: `CausalNavigator`](#class-reference-causalnavigator)
    + [Initialization](#initialization)
    + [Methods](#methods)
      - [`check_overlap(X, T)`](#check_overlapx-t)
      - [`fit_estimate(X, T, Y)`](#fit_estimatex-t-y)
      - [`get_cate_df(df_original)`](#get_cate_dfdf_original)
      - [`plot_heterogeneity(df_with_cate, col, bins=5)`](#plot_heterogeneitydf_with_cate-col-bins5)
      - [`run_placebo_test(X, T, Y, n_simulations=10)`](#run_placebo_testx-t-y-n_simulations10)
      - [`run_sensitivity_analysis(X, T, Y)`](#run_sensitivity_analysisx-t-y)
      - [`compare_estimators(X, T, Y)`](#compare_estimatorsx-t-y)
  * [Helper Functions](#helper-functions)
      - [`load_cdc_data(filepath)`](#load_cdc_datafilepath)
      - [`preprocess_for_causal(df, ...)`](#preprocess_for_causaldf-)

<!-- tocstop -->

# Summary

This document provides API documentation for the `CausalML` module, a high-level
wrapper around the `causalml` library designed to simplify causal inference
workflows for observational studies. It includes the `CausalNavigator` class for
heterogeneous treatment effect estimation, diagnostic methods for assumption
validation, and helper functions for data preprocessing and visualization.

# CausalML API Documentation

## Installation and Docker Setup

To ensure reproducibility, this project is containerized. Follow these steps to
build and run the analysis.

### 1. Build the Image

Run this command in the project root (where the `Dockerfile` is located):

```bash
docker build -t causalml_project .
```

### 2. Run the Container

Start the `Jupyter` environment with volume mounting (to save your notebook
changes):

```bash
# Mac/Linux/WSL
docker run -p 8888:8888 -v "$(pwd)":/app causalml_project
```

### 3. Access the Project

- Click the `http://127.0.0.1:8888...` link in your terminal to open
  `JupyterLab`
- Open `CausalML.API.ipynb` to test the tool
- Open `CausalML.example.ipynb` to see the full Diabetes analysis

## Overview

The `CausalML` module provides a high-level wrapper around the `causalml`
library. It is designed to simplify the workflow of Causal Inference for
observational studies, specifically focusing on Heterogeneous Treatment Effects
(HTE).

While `causalml` provides powerful meta-learners (S/T/X-Learners), the native
API requires significant boilerplate code for data preprocessing, assumption
checking, and visualization. This API standardizes that workflow into a single
`CausalNavigator` class.

## Architecture

The core component is the `CausalNavigator` class, which orchestrates the
following pipeline:

1.  **Diagnostic Layer**: Verifies causal assumptions (specifically Common
    Support) before estimation
2.  **Estimation Layer**: Wraps `causalml.inference.meta` classes
    (`BaseXRegressor`, etc.) and injects `XGBoost` as the standard base learner
3.  **Interpretation Layer**: Provides built-in methods to visualize
    heterogeneity, abstracting away `matplotlib` complexity

## Class Reference: `CausalNavigator`

### Initialization

```python
navigator = CausalNavigator(
    learner_type='X',       # Options: 'S', 'T', 'X'
    control_name='Control',
    treatment_name='Treatment'
)
```

### Methods

#### `check_overlap(X, T)`

**Purpose**: Validates the Positivity/Overlap assumption

- **Inputs**: Covariate matrix `X`, Treatment vector `T`
- **Output**: A density plot of Propensity Scores
- **Why use this**: If the distributions do not overlap, causal estimation is
  invalid. This method enforces safety before modeling

#### `fit_estimate(X, T, Y)`

**Purpose**: Trains the meta-learner and estimates CATE (Conditional Average
Treatment Effect)

- **Inputs**: Covariates `X`, Treatment `T`, Outcome `Y`
- **Output**: `numpy.array` of CATE values for each observation
- **Design Choice**: I use `XGBoost` as the base learner because it handles
  non-linearities in the response surface effectively, which is crucial for the
  X-Learner

#### `get_cate_df(df_original)`

**Purpose**: Helper to merge the estimated effects back into the original
dataframe

- **Inputs**: Original dataframe
- **Output**: Dataframe with a new column `cate`

#### `plot_heterogeneity(df_with_cate, col, bins=5)`

**Purpose**: Visualizes how the treatment effect varies across a specific
feature

- **Inputs**: Dataframe with CATE, column name, optional bins
- **Output**: A bar chart showing CATE by group with confidence intervals

#### `run_placebo_test(X, T, Y, n_simulations=10)`

**Purpose**: Robustness check (Refutation)

- **Logic**: Randomly shuffles the treatment array to break any true causal
  link, then re-trains the model
- **Success Criteria**: The "Placebo ATE" should cluster around 0. The "Actual
  ATE" should be far outside this distribution
- **Interpretation**: If the Actual Effect falls inside the Placebo
  distribution, the result is statistically indistinguishable from noise

#### `run_sensitivity_analysis(X, T, Y)`

**Purpose**: Quantifies the stability of the causal estimate

- **Logic**: Iteratively removes one covariate at a time (e.g., removing 'Age',
  then 'Income') and re-calculates the Average Treatment Effect (ATE)
- **Output**: A horizontal bar chart showing the ATE for each scenario compared
  to the baseline
- **Interpretation**:
  - **Stable**: Bars cluster near the baseline (red line)
  - **Sensitive**: A bar shifts significantly (or crosses zero), indicating that
    specific variable drives the result

#### `compare_estimators(X, T, Y)`

**Purpose**: Advanced model selection ("Horse Race")

- **Methodology**: Splits data into Train (70%) and Test (30%). Trains S, T, X,
  R, and DR learners on the training set
- **Metric**: Generates a **Cumulative Gain Chart (Uplift Curve)** on the test
  set
- **Why this metric**: Since ground truth CATE is impossible to observe, I
  cannot use RMSE. The Gain Chart measures how well a model sorts individuals
  from "High Responder" to "Low Responder"
- **Visual**: Produces a plot where the highest curve represents the
  best-performing model for targeting interventions
- **Outputs**:
  - **Uplift Curve Plot**: Visual comparison of model performance
  - **Qini/AUUC Score Table**: Numerical ranking of models (Area Under Uplift
    Curve)

## Helper Functions

#### `load_cdc_data(filepath)`

**Purpose**: Robustly loads the `CDC` dataset from a local directory

- **Inputs**: `filepath` (str) - The relative path to the `.csv` file (e.g.,
  `data/unprocessed/file.csv`)
- **Behavior**: Checks for file existence, removes duplicates, and casts all
  columns to `float` to ensure compatibility with `XGBoost`
- **Output**: A cleaned `pandas.DataFrame`

#### `preprocess_for_causal(df, ...)`

**Purpose**: splits the dataframe into the three required components for
`CausalML`:

1.  **X (Covariates)**: The features used to control for confounding
2.  **T (Treatment)**: The binary intervention vector
3.  **Y (Outcome)**: The target variable
