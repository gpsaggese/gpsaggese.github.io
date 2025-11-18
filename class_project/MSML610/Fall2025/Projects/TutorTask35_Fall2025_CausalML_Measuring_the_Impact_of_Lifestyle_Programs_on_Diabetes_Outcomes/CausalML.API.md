# CausalML API Documentation

## Overview
The `causal_impact` module provides a high-level wrapper around the `causalml` library. It is designed to simplify the workflow of Causal Inference for observational studies, specifically focusing on Heterogeneous Treatment Effects (HTE).

While `causalml` provides powerful meta-learners (S/T/X-Learners), the native API requires significant boilerplate code for data preprocessing, assumption checking, and visualization. This API standardizes that workflow into a single `CausalNavigator` class.

## Architecture
The core component is the `CausalNavigator` class, which orchestrates the following pipeline:
1.  **Diagnostic Layer:** Verifies causal assumptions (specifically *Common Support*) before estimation.
2.  **Estimation Layer:** Wraps `causalml.inference.meta` classes (`BaseXRegressor`, etc.) and injects XGBoost as the standard base learner.
3.  **Interpretation Layer:** Provides built-in methods to visualize heterogeneity, abstracting away `matplotlib` complexity.

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
**Purpose:** Validates the Positivity/Overlap assumption.
- **Inputs:** Covariate matrix `X`, Treatment vector `T`.
- **Output:** A density plot of Propensity Scores.
- **Why use this:** If the distributions do not overlap, causal estimation is invalid. This method enforces safety before modeling.

#### `fit_estimate(X, T, Y)`
**Purpose:** Trains the meta-learner and estimates CATE (Conditional Average Treatment Effect).
- **Inputs:** Covariates `X`, Treatment `T`, Outcome `Y`.
- **Output:** `numpy.array` of CATE values for each observation.
- **Design Choice:** We use XGBoost as the base learner because it handles non-linearities in the response surface effectively, which is crucial for the X-Learner.

#### `get_cate_df(df_original)`
**Purpose:** Helper to merge the estimated effects back into the original dataframe.
- **Inputs:** Original dataframe.
- **Output:** Dataframe with a new column `cate`.

#### `plot_heterogeneity(df_with_cate, col, bins=5)`
**Purpose:** Visualizes how the treatment effect varies across a specific feature.
- **Inputs:** Dataframe with CATE, column name, optional bins.
- **Output:** A bar chart showing CATE by group with confidence intervals.

## Helper Functions

#### `load_cdc_data(filepath)`
**Purpose:** Robustly loads the CDC dataset from a local directory.
- **Inputs:** `filepath` (str) - The relative path to the `.csv` file (e.g., `data/unprocessed/file.csv`).
- **Behavior:** Checks for file existence, removes duplicates, and casts all columns to `float` to ensure compatibility with `XGBoost`.
- **Output:** A cleaned `pandas.DataFrame`.


#### `preprocess_for_causal(df, ...)`
**Purpose:** splits the dataframe into the three required components for CausalML:
1.  **X (Covariates):** The features used to control for confounding.
2.  **T (Treatment):** The binary intervention vector.
3.  **Y (Outcome):** The target variable.