"""
High-level API for MSML610 Project:
    TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes

This module provides a small, opinionated interface on top of EconML
for our NHANES 2021–2023 causal inference experiments.

Public functions:

    - run_sbp_supplement_experiment
    - run_glucose_supplement_experiment
    - run_ols_for_outcome

Each function:

    * Calls `build_analysis_df` from `econml_utils` to construct the
      merged NHANES analysis dataset.
    * Uses `get_y_t_x` to extract outcome (Y), treatment (T), and
      covariates (X).
    * Drops rows with missing values before fitting models.
    * Returns a dictionary with clean, easy-to-use results that the
      notebooks can consume.
"""

from typing import Dict, Any

import numpy as np
import pandas as pd

from econml.dr import DRLearner
from sklearn.linear_model import LogisticRegression, LinearRegression

from econml_utils import build_analysis_df, get_y_t_x


# ---------------------------------------------------------------------
# Internal helper: DRLearner for a single outcome
# ---------------------------------------------------------------------


def _fit_drl_for_outcome(
    outcome_col: str,
    treatment_col: str = "treatment_supplement",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Fit a DRLearner for a given outcome using supplement use as treatment.

    Parameters
    ----------
    outcome_col : str
        Name of the outcome column (e.g., "sbp_mean" or "fasting_glucose_mg_dl").
    treatment_col : str, default "treatment_supplement"
        Binary treatment indicator column.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    dict
        {
            "ate": float,
            "covariates": list[str],
            "cate_df": pd.DataFrame,
            "tau_col": str,
            "age_effects": pd.Series or None,
            "bmi_effects": pd.Series or None,
        }
    """
    # Build merged NHANES dataset
    analysis_df = build_analysis_df()

    # Extract outcome, treatment, and covariates
    y, t, X, covariate_cols = get_y_t_x(
        analysis_df, outcome_col=outcome_col, treatment_col=treatment_col
    )

    # Drop rows with any missing data in Y/T/X
    combined = pd.concat([y, t, X], axis=1)
    mask = combined.notna().all(axis=1)

    y_clean = y.loc[mask]
    t_clean = t.loc[mask]
    X_clean = X.loc[mask]

    # Mirror the cleaned rows back on the full analysis dataframe
    analysis_df_clean = analysis_df.loc[mask].copy()

    if len(y_clean) == 0:
        raise ValueError(
            "After dropping missing values, there are no rows left to fit the model. "
            "Please check the data preparation steps."
        )

    # Set up and fit DRLearner
    dr = DRLearner(
        model_regression=LinearRegression(),
        model_propensity=LogisticRegression(max_iter=2000, solver="lbfgs"),
        random_state=random_state,
    )

    dr.fit(
        Y=y_clean.to_numpy(),
        T=t_clean.to_numpy(),
        X=X_clean.to_numpy(),
    )

    # Average treatment effect (ATE) and individual CATEs
    ate = float(dr.ate(X_clean.to_numpy()))
    tau_hat = dr.effect(X_clean.to_numpy()).ravel()

    tau_col = f"tau_hat_{outcome_col}"
    analysis_df_clean[tau_col] = tau_hat

    # ------------------------------------------------------------------
    # Heterogeneity summaries: age and BMI quartiles
    # ------------------------------------------------------------------
    age_effects = None
    bmi_effects = None

    if "age_years" in analysis_df_clean.columns:
        analysis_df_clean["age_bin"] = pd.qcut(
            analysis_df_clean["age_years"],
            q=4,
            labels=["Q1 (youngest)", "Q2", "Q3", "Q4 (oldest)"],
            duplicates="drop",
        )
        age_effects = (
            analysis_df_clean.groupby("age_bin")[tau_col]
            .mean()
            .sort_index()
        )

    if "body_mass_index_kg_m2" in analysis_df_clean.columns:
        analysis_df_clean["bmi_bin"] = pd.qcut(
            analysis_df_clean["body_mass_index_kg_m2"],
            q=4,
            labels=["Q1 (leanest)", "Q2", "Q3", "Q4 (highest BMI)"],
            duplicates="drop",
        )
        bmi_effects = (
            analysis_df_clean.groupby("bmi_bin")[tau_col]
            .mean()
            .sort_index()
        )

    return {
        "ate": ate,
        "covariates": covariate_cols,
        "cate_df": analysis_df_clean,
        "tau_col": tau_col,
        "age_effects": age_effects,
        "bmi_effects": bmi_effects,
    }


# ---------------------------------------------------------------------
# Public API: DRLearner experiments for SBP and glucose
# ---------------------------------------------------------------------


def run_sbp_supplement_experiment(random_state: int = 42) -> Dict[str, Any]:
    """
    Run the DRLearner experiment with outcome = mean systolic BP (sbp_mean).

    This is the main entry point used in the notebooks for the SBP outcome.

    Returns
    -------
    dict
        {
            "ate_sbp": float,
            "covariates": list[str],
            "cate_df": pd.DataFrame,
            "tau_col": str,
            "age_effects": pd.Series or None,
            "bmi_effects": pd.Series or None,
        }
    """
    results = _fit_drl_for_outcome(
        outcome_col="sbp_mean",
        treatment_col="treatment_supplement",
        random_state=random_state,
    )

    return {
        "ate_sbp": results["ate"],
        "covariates": results["covariates"],
        "cate_df": results["cate_df"],
        "tau_col": results["tau_col"],
        "age_effects": results["age_effects"],
        "bmi_effects": results["bmi_effects"],
    }


def run_glucose_supplement_experiment(random_state: int = 42) -> Dict[str, Any]:
    """
    Run the DRLearner experiment with outcome = fasting_glucose_mg_dl.

    This is the main entry point used in the notebooks for the glucose
    outcome.

    Returns
    -------
    dict
        {
            "ate_glucose": float,
            "covariates": list[str],
            "cate_df": pd.DataFrame,
            "tau_col": str,
            "age_effects": pd.Series or None,
            "bmi_effects": pd.Series or None,
        }
    """
    results = _fit_drl_for_outcome(
        outcome_col="fasting_glucose_mg_dl",
        treatment_col="treatment_supplement",
        random_state=random_state,
    )

    return {
        "ate_glucose": results["ate"],
        "covariates": results["covariates"],
        "cate_df": results["cate_df"],
        "tau_col": results["tau_col"],
        "age_effects": results["age_effects"],
        "bmi_effects": results["bmi_effects"],
    }


# ---------------------------------------------------------------------
# Public API: OLS baseline for a single outcome
# ---------------------------------------------------------------------


def run_ols_for_outcome(
    outcome_col: str,
    treatment_col: str = "treatment_supplement",
) -> Dict[str, Any]:
    """
    Simple OLS comparison:

        Y ~ treatment + covariates

    We interpret the coefficient of the treatment variable as the
    "traditional" estimate of the treatment effect, controlling
    linearly for the covariates.

    Parameters
    ----------
    outcome_col : str
        Outcome column name (e.g., "sbp_mean").
    treatment_col : str, default "treatment_supplement"
        Binary treatment indicator column.

    Returns
    -------
    dict
        {
            "outcome": str,
            "treatment_coef": float,
            "covariates": list[str],
            "n_obs": int,
        }
    """
    analysis_df = build_analysis_df()

    # Use the same helper as for EconML to get Y/T/X
    y, t, X, covariate_cols = get_y_t_x(
        analysis_df, outcome_col=outcome_col, treatment_col=treatment_col
    )

    # Drop rows with any missing data
    combined = pd.concat([y, t, X], axis=1)
    mask = combined.notna().all(axis=1)

    y_clean = y.loc[mask]
    t_clean = t.loc[mask]
    X_clean = X.loc[mask]

    if len(y_clean) == 0:
        raise ValueError(
            "After dropping missing values, there are no rows left to fit the OLS model. "
            "Please check the data preparation steps."
        )

    # Build design matrix: [treatment, covariates]
    T_matrix = t_clean.to_numpy().reshape(-1, 1)
    X_ols = np.column_stack([T_matrix, X_clean.to_numpy()])

    ols = LinearRegression()
    ols.fit(X_ols, y_clean.to_numpy())

    # First coefficient corresponds to the treatment effect
    treatment_coef = float(ols.coef_[0])

    return {
        "outcome": outcome_col,
        "treatment_coef": treatment_coef,
        "covariates": covariate_cols,
        "n_obs": int(len(y_clean)),
    }
