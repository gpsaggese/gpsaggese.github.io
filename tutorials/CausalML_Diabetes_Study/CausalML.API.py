# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Enable auto-reloading so edits in utils.py update immediately
# %load_ext autoreload
# %autoreload 2

import os
import numpy as np
import warnings

from utils import CausalNavigator, load_cdc_data, preprocess_for_causal

warnings.filterwarnings("ignore")

# %%
# https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv
filename = "diabetes_binary_health_indicators_BRFSS2015.csv"
DATA_PATH = os.path.join("data", "unprocessed", filename)
try:
    df_raw = load_cdc_data(DATA_PATH)
    # We use a subset of columns for the API demo
    df_clean, X, T, Y = preprocess_for_causal(
        df_raw,
        treatment_col="PhysActivity",
        outcome_col="Diabetes_binary",
        covariate_cols=[
            "HighBP",
            "HighChol",
            "Age",
            "Income",
            "Sex",
            "GenHlth",
            "BMI",
        ],
    )
    # Subsample 10k rows for speed
    sample_indices = np.random.choice(X.index, size=10000, replace=False)
    X_demo, T_demo, Y_demo = (
        X.loc[sample_indices],
        T.loc[sample_indices],
        Y.loc[sample_indices],
    )
    print(f"API Demo Data Loaded. Shape: {X_demo.shape}")
    display(X_demo.head())

except Exception as e:
    print(f"Error: {e}")

# %%
# Initialize the CausalNavigator
navigator = CausalNavigator(
    learner_type="X", control_name="Sedentary", treatment_name="Active"
)

# %%
# Check if there is "Common Support" between the treated and control groups.
navigator.check_overlap(X_demo, T_demo)

# %%
# Estimate Effects
cate_estimates = navigator.fit_estimate(X_demo, T_demo, Y_demo)
print(f"\nAverage Treatment Effect (ATE): {cate_estimates.mean():.4f}")

# %%
# Visualize Results
df_results = navigator.get_cate_df(X_demo)
navigator.plot_heterogeneity(df_results, col="Age")

# %%
# Robustness Check: Placebo Test
# We shuffle T to see if the model finds an effect where none exists.
navigator.run_placebo_test(X_demo, T_demo, Y_demo, n_simulations=3)

# %%
# Sensitivity Analysis
# Check how stable the result is when removing one covariate at a time.
navigator.run_sensitivity_analysis(X_demo, T_demo, Y_demo)

# %%
# Estimator Comparison ("Horse Race")
# Compare X-Learner against S, T, R, and DR Learners using Uplift Curves.
navigator.compare_estimators(X_demo, T_demo, Y_demo)
