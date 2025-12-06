"""
High-level API for MSML610 Project 3: EconML experiments on NHANES.

Exposes functions like:
  - run_sbp_supplement_experiment
  - run_glucose_supplement_experiment
  - run_ols_for_outcome  (baseline linear regression)

that coordinate data preparation, model fitting, and summary outputs.
"""

import numpy as np
import pandas as pd

from econml.dr import DRLearner
from sklearn.linear_model import LogisticRegression, LinearRegression

from econml_utils import build_analysis_df, get_y_t_x


def run_sbp_supplement_experiment(random_state: int = 42) -> dict:
    """
    Project 3 main experiment:
    Effect of any dietary supplement use on systolic blood pressure (sbp_mean).

    Pipeline
    --------
    1. Build merged NHANES analysis dataframe via `build_analysis_df()`:
         - Blood pressure outcomes (sbp_mean, dbp_mean)
         - Anthropometrics (BMI, weight, waist)
         - Lab values (cholesterol, HDL, triglycerides, glucose, hs-CRP)
         - Treatment indicator: treatment_supplement (any supplements taken)
         - Demographics (age_years, sex)
    2. Extract outcome (Y), treatment (T) and covariates (X) using
       `get_y_t_x(...)` with:
         - outcome_col   = "sbp_mean"
         - treatment_col = "treatment_supplement"
    3. Drop rows with missing values in Y, T or any X covariate.
    4. Fit a DRLearner to estimate the causal effect of supplement use on sbp_mean.
    5. Compute and return:
         - Overall ATE on sbp_mean.
         - Individual-level CATEs (tau_hat_sbp).
         - Average effects by age quartiles and BMI quartiles.
    """
    # 1. Build full analysis dataframe (merged NHANES tables)
    analysis_df = build_analysis_df()

    # Sanity check for key columns used later
    required_cols = [
        "sbp_mean",
        "treatment_supplement",
        "age_years",
        "body_mass_index_kg_m2",
    ]
    missing = [c for c in required_cols if c not in analysis_df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns in analysis_df: {missing}. "
            "Check econml_utils.build_analysis_df."
        )

    # 2. Get outcome, treatment and covariates explicitly
    y, t, X, covariate_cols = get_y_t_x(
        analysis_df,
        outcome_col="sbp_mean",
        treatment_col="treatment_supplement",
    )

    # 3. Drop rows with any missing data in Y, T or X
    mask = (~y.isna()) & (~t.isna()) & (~X.isna().any(axis=1))
    y_clean = y[mask]
    t_clean = t[mask]
    X_clean = X[mask]

    # 4. Fit DRLearner.
    #    Use LogisticRegression as the propensity model (treatment model).
    dr = DRLearner(
        model_propensity=LogisticRegression(max_iter=2000, solver="lbfgs"),
        random_state=random_state,
    )
    dr.fit(y_clean, t_clean, X=X_clean)

    # Overall average treatment effect on sbp_mean
    ate = float(dr.ate(X_clean))

    # Individual-level CATEs tau_hat(X)
    cate = dr.effect(X_clean)

    # Attach CATEs back to a clean copy of analysis_df
    analysis_df_clean = analysis_df.loc[mask].copy()
    analysis_df_clean["tau_hat_sbp"] = cate

    # 5. Heterogeneity summaries: age and BMI quartiles

    # Age bins
    analysis_df_clean["age_bin"] = pd.qcut(
        analysis_df_clean["age_years"],
        4,
        labels=["Q1 (youngest)", "Q2", "Q3", "Q4 (oldest)"],
    )

    # BMI bins
    analysis_df_clean["bmi_bin"] = pd.qcut(
        analysis_df_clean["body_mass_index_kg_m2"],
        4,
        labels=["Q1 (leanest)", "Q2", "Q3", "Q4 (highest BMI)"],
    )

    age_effects = (
        analysis_df_clean.groupby("age_bin")["tau_hat_sbp"]
        .mean()
        .sort_index()
    )

    bmi_effects = (
        analysis_df_clean.groupby("bmi_bin")["tau_hat_sbp"]
        .mean()
        .sort_index()
    )

    results = {
        "ate_sbp": ate,
        "covariates": covariate_cols,
        "cate_df": analysis_df_clean,
        "age_effects": age_effects,
        "bmi_effects": bmi_effects,
    }
    return results


def run_glucose_supplement_experiment(random_state: int = 42) -> dict:
    """
    Secondary experiment:
    Effect of any dietary supplement use on fasting glucose (fasting_glucose_mg_dl).

    This reuses the same treatment and covariates as the SBP experiment,
    but changes the outcome to fasting_glucose_mg_dl.

    Pipeline
    --------
    1. Build merged NHANES analysis dataframe via `build_analysis_df()`.
    2. Extract outcome (Y), treatment (T) and covariates (X) using
       `get_y_t_x(...)` with:
         - outcome_col   = "fasting_glucose_mg_dl"
         - treatment_col = "treatment_supplement"
    3. Drop rows with missing values in Y, T or any X covariate.
    4. Fit a DRLearner to estimate the causal effect of supplement use on glucose.
    5. Compute and return:
         - Overall ATE on fasting_glucose_mg_dl.
         - Individual-level CATEs (tau_hat_glucose).
         - Average effects by age quartiles and BMI quartiles.
    """
    # 1. Build full analysis dataframe (merged NHANES tables)
    analysis_df = build_analysis_df()

    # Sanity check for key columns used later
    required_cols = [
        "fasting_glucose_mg_dl",
        "treatment_supplement",
        "age_years",
        "body_mass_index_kg_m2",
    ]
    missing = [c for c in required_cols if c not in analysis_df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns in analysis_df: {missing}. "
            "Check econml_utils.build_analysis_df."
        )

    # 2. Get outcome, treatment and covariates explicitly
    y, t, X, covariate_cols = get_y_t_x(
        analysis_df,
        outcome_col="fasting_glucose_mg_dl",
        treatment_col="treatment_supplement",
    )

    # 3. Drop rows with any missing data in Y, T or X
    mask = (~y.isna()) & (~t.isna()) & (~X.isna().any(axis=1))
    y_clean = y[mask]
    t_clean = t[mask]
    X_clean = X[mask]

    # 4. Fit DRLearner with logistic regression as propensity model
    dr = DRLearner(
        model_propensity=LogisticRegression(max_iter=2000, solver="lbfgs"),
        random_state=random_state,
    )
    dr.fit(y_clean, t_clean, X=X_clean)

    # Overall average treatment effect on fasting glucose
    ate = float(dr.ate(X_clean))

    # Individual-level CATEs
    cate = dr.effect(X_clean)

    # Attach CATEs back to a clean copy of analysis_df
    analysis_df_clean = analysis_df.loc[mask].copy()
    analysis_df_clean["tau_hat_glucose"] = cate

    # Age and BMI bins
    analysis_df_clean["age_bin"] = pd.qcut(
        analysis_df_clean["age_years"],
        4,
        labels=["Q1 (youngest)", "Q2", "Q3", "Q4 (oldest)"],
    )
    analysis_df_clean["bmi_bin"] = pd.qcut(
        analysis_df_clean["body_mass_index_kg_m2"],
        4,
        labels=["Q1 (leanest)", "Q2", "Q3", "Q4 (highest BMI)"],
    )

    age_effects = (
        analysis_df_clean.groupby("age_bin")["tau_hat_glucose"]
        .mean()
        .sort_index()
    )

    bmi_effects = (
        analysis_df_clean.groupby("bmi_bin")["tau_hat_glucose"]
        .mean()
        .sort_index()
    )

    results = {
        "ate_glucose": ate,
        "covariates": covariate_cols,
        "cate_df": analysis_df_clean,
        "age_effects": age_effects,
        "bmi_effects": bmi_effects,
    }
    return results


def run_ols_for_outcome(
    outcome_col: str,
    treatment_col: str = "treatment_supplement",
) -> dict:
    """
    Baseline model: ordinary least squares (OLS) regression.

    We fit a simple linear regression of:

        outcome ~ treatment + all covariates X

    using the same analysis dataframe and covariates as the EconML
    experiments. The coefficient on `treatment` is a naive estimate of
    the treatment effect that we can compare to the DRLearner ATE.

    Parameters
    ----------
    outcome_col : str
        Name of the outcome column in analysis_df (e.g., "sbp_mean").
    treatment_col : str, optional
        Name of the treatment column, default "treatment_supplement".

    Returns
    -------
    results : dict
        {
          "outcome": outcome_col,
          "treatment_coef": float,
          "n": int,
          "model": LinearRegression,
        }
    """
    # Build dataset and extract Y, T, X
    analysis_df = build_analysis_df()
    y, t, X, covariate_cols = get_y_t_x(
        analysis_df,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
    )

    # Drop rows with missing data
    mask = (~y.isna()) & (~t.isna()) & (~X.isna().any(axis=1))
    y_clean = y[mask]
    t_clean = t[mask]
    X_clean = X[mask]

    # Design matrix: first column is treatment, then all covariates
    X_ols = pd.concat(
        [
            t_clean.rename(treatment_col).astype(float),
            X_clean.astype(float),
        ],
        axis=1,
    )

    ols = LinearRegression()
    ols.fit(X_ols, y_clean)

    # First coefficient corresponds to the treatment column
    treatment_coef = float(ols.coef_[0])

    return {
        "outcome": outcome_col,
        "treatment_coef": treatment_coef,
        "n": int(len(y_clean)),
        "model": ols,
    }
