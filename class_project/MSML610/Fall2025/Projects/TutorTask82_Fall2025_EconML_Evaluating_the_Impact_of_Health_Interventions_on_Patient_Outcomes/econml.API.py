"""
econml.API.py

Project-level API wrapper for EconML DRLearner experiments on NHANES (2021–2023).

Internal API for the MSML610 project:
TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes

Public entry points:
- run_sbp_supplement_experiment()
- run_glucose_supplement_experiment()
- run_ols_for_outcome()

Design goals:
- Keep notebooks clean (notebooks call these functions, not raw EconML).
- Return plain Python objects / DataFrames so reuse is safe and obvious.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from econml.dr import DRLearner

from econml_utils import build_analysis_df, get_y_t_x


# -----------------------------
# Small internal helpers
# -----------------------------

def _pick_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name from `candidates` that exists in `df`, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _make_quartile_bins(
    s: pd.Series,
    labels: Tuple[str, str, str, str],
) -> Optional[pd.Series]:
    """
    Make quartile bins with stable labels.

    Robust to:
    - non-numeric input
    - many duplicate values
    - qcut dropping bins (duplicates="drop")

    Returns:
      categorical Series or None if binning isn't possible.
    """
    try:
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().sum() < 10:
            return None

        # qcut without labels first (more robust), then rename categories
        binned = pd.qcut(s_num, q=4, duplicates="drop")

        if not pd.api.types.is_categorical_dtype(binned):
            return None

        k = len(binned.cat.categories)
        if k == 0:
            return None

        new_labels = list(labels)[:k]
        binned = binned.cat.rename_categories(new_labels)
        return binned
    except Exception:
        return None


def _ensure_binary_treatment(t: np.ndarray) -> None:
    """
    Validate treatment is binary {0,1} (or very close, float).
    Raises a clear error if unexpected codes appear.
    """
    uniq = np.unique(t[~np.isnan(t)])
    uniq_rounded = np.unique(np.round(uniq, 6))
    ok = set(uniq_rounded.tolist()).issubset({0.0, 1.0})
    if not ok:
        raise ValueError(
            f"Treatment is not binary 0/1. Found unique values: {uniq_rounded.tolist()}. "
            "Fix: check treatment encoding in econml_utils.build_analysis_df()."
        )


def _bootstrap_ate_ci(
    y: np.ndarray,
    t: np.ndarray,
    X: np.ndarray,
    random_state: int,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Nonparametric bootstrap CI for ATE.
    Refit DRLearner on bootstrap samples and compute ATE each time.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    ates: List[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        seed = int(rng.integers(0, 2**31 - 1))

        dr = DRLearner(
            model_regression=LinearRegression(),
            model_propensity=LogisticRegression(max_iter=2000, solver="lbfgs"),
            random_state=seed,
        )
        dr.fit(Y=y[idx], T=t[idx], X=X[idx])
        ates.append(float(dr.ate(X[idx])))

    low = float(np.quantile(ates, alpha / 2))
    high = float(np.quantile(ates, 1 - alpha / 2))
    return low, high


# -----------------------------
# Core DRLearner internal runner
# -----------------------------

def _fit_drl_for_outcome(
    outcome_col: str,
    treatment_col: str = "treatment_supplement",
    random_state: int = 42,
    n_bootstrap: int = 200,
    include_demographics: bool = False,
    join: str = "inner",
) -> Dict[str, Any]:
    """
    Fit DRLearner for a given outcome and return a reusable result bundle.

    Returns a dict with:
      - ate, ate_ci_low, ate_ci_high
      - n_obs
      - covariates
      - cate_df (cleaned df with individual tau_hat_* column)
      - tau_col
      - age_effects (Series or None)
      - bmi_effects (Series or None)
      - model (fitted DRLearner)
    """
    df = build_analysis_df(join=join)

    y, t, X, covariate_cols = get_y_t_x(
        df,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        include_demographics=include_demographics,
    )

    # Safety: never allow the outcome itself to leak into X
    if outcome_col in covariate_cols:
        covariate_cols = [c for c in covariate_cols if c != outcome_col]
        X = X[covariate_cols].copy()

    # Clean rows with any missingness in (Y, T, X)
    combined = pd.concat([y.rename("Y"), t.rename("T"), X], axis=1).dropna()
    y_clean = combined["Y"].to_numpy(dtype=float)
    t_clean = combined["T"].to_numpy(dtype=float)
    X_clean = combined[covariate_cols].to_numpy(dtype=float)

    _ensure_binary_treatment(t_clean)

    # Fit DRLearner
    dr = DRLearner(
        model_regression=LinearRegression(),
        model_propensity=LogisticRegression(max_iter=2000, solver="lbfgs"),
        random_state=random_state,
    )
    dr.fit(Y=y_clean, T=t_clean, X=X_clean)

    ate = float(dr.ate(X_clean))

    ate_ci_low, ate_ci_high = _bootstrap_ate_ci(
        y=y_clean,
        t=t_clean,
        X=X_clean,
        random_state=random_state,
        n_bootstrap=n_bootstrap,
        alpha=0.05,
    )

    # Individual effects
    tau_hat = dr.effect(X_clean).reshape(-1)
    tau_col = f"tau_hat_{outcome_col}"

    cate_df = df.loc[combined.index].copy()
    cate_df[tau_col] = tau_hat

    # Heterogeneity bins (best-effort)
    age_col = _pick_first_existing_col(
        cate_df,
        candidates=[
            "age_years",
            "age_in_years_at_screening",
            "age",
            "ridageyr",
            "respondent_age_years",
        ],
    )
    bmi_col = _pick_first_existing_col(
        cate_df,
        candidates=[
            "body_mass_index_kg_m2",
            "bmi",
            "bmx_bmi",
        ],
    )

    age_effects = None
    if age_col is not None:
        age_bin = _make_quartile_bins(
            cate_df[age_col],
            labels=("Q1 (youngest)", "Q2", "Q3", "Q4 (oldest)"),
        )
        if age_bin is not None:
            cate_df["age_bin"] = age_bin
            age_effects = cate_df.groupby("age_bin", observed=True)[tau_col].mean()

    bmi_effects = None
    if bmi_col is not None:
        bmi_bin = _make_quartile_bins(
            cate_df[bmi_col],
            labels=("Q1 (leanest)", "Q2", "Q3", "Q4 (highest BMI)"),
        )
        if bmi_bin is not None:
            cate_df["bmi_bin"] = bmi_bin
            bmi_effects = cate_df.groupby("bmi_bin", observed=True)[tau_col].mean()

    return {
        "ate": ate,
        "ate_ci_low": ate_ci_low,
        "ate_ci_high": ate_ci_high,
        "n_obs": int(len(y_clean)),
        "covariates": covariate_cols,
        "cate_df": cate_df,
        "tau_col": tau_col,
        "age_effects": age_effects,
        "bmi_effects": bmi_effects,
        "model": dr,
    }


# -----------------------------
# Public API functions
# -----------------------------

def run_sbp_supplement_experiment(
    random_state: int = 42,
    n_bootstrap: int = 200,
    include_demographics: bool = False,
    join: str = "inner",
) -> Dict[str, Any]:
    """
    DRLearner experiment:
      Outcome  : sbp_mean
      Treatment: treatment_supplement (0/1)
    """
    res = _fit_drl_for_outcome(
        outcome_col="sbp_mean",
        treatment_col="treatment_supplement",
        random_state=random_state,
        n_bootstrap=n_bootstrap,
        include_demographics=include_demographics,
        join=join,
    )
    return {
        "ate_sbp": res["ate"],
        "ate_ci_low": res["ate_ci_low"],
        "ate_ci_high": res["ate_ci_high"],
        "n_obs": res["n_obs"],
        "covariates": res["covariates"],
        "cate_df": res["cate_df"],
        "tau_col": res["tau_col"],
        "age_effects": res["age_effects"],
        "bmi_effects": res["bmi_effects"],
        "model": res["model"],
    }


def run_glucose_supplement_experiment(
    random_state: int = 42,
    n_bootstrap: int = 200,
    include_demographics: bool = False,
    join: str = "inner",
) -> Dict[str, Any]:
    """
    DRLearner experiment:
      Outcome  : fasting_glucose_mg_dl
      Treatment: treatment_supplement (0/1)
    """
    res = _fit_drl_for_outcome(
        outcome_col="fasting_glucose_mg_dl",
        treatment_col="treatment_supplement",
        random_state=random_state,
        n_bootstrap=n_bootstrap,
        include_demographics=include_demographics,
        join=join,
    )
    return {
        "ate_glucose": res["ate"],
        "ate_ci_low": res["ate_ci_low"],
        "ate_ci_high": res["ate_ci_high"],
        "n_obs": res["n_obs"],
        "covariates": res["covariates"],
        "cate_df": res["cate_df"],
        "tau_col": res["tau_col"],
        "age_effects": res["age_effects"],
        "bmi_effects": res["bmi_effects"],
        "model": res["model"],
    }


def run_ols_for_outcome(
    outcome_col: str,
    treatment_col: str = "treatment_supplement",
    include_demographics: bool = False,
    join: str = "inner",
) -> Dict[str, Any]:
    """
    OLS baseline comparison using statsmodels with robust (HC3) standard errors.

    Returns:
      outcome, treatment_coef, treatment_ci_low, treatment_ci_high,
      covariates, n_obs, method
    """
    try:
        import statsmodels.api as sm
    except ImportError as e:
        raise ImportError(
            "statsmodels is required for run_ols_for_outcome(). "
            "Add it to requirements.txt (e.g., statsmodels>=0.14)."
        ) from e

    df = build_analysis_df(join=join)
    y, t, X, covariate_cols = get_y_t_x(
        df,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        include_demographics=include_demographics,
    )

    # Prevent outcome leakage
    if outcome_col in covariate_cols:
        covariate_cols = [c for c in covariate_cols if c != outcome_col]
        X = X[covariate_cols].copy()

    combined = pd.concat([y.rename("Y"), t.rename("T"), X], axis=1).dropna()
    y_clean = combined["Y"].astype(float)
    t_clean = combined["T"].astype(float)
    X_clean = combined[covariate_cols].astype(float)

    X_ols = pd.concat([t_clean.rename(treatment_col), X_clean], axis=1)
    X_ols = sm.add_constant(X_ols, has_constant="add")

    model = sm.OLS(y_clean, X_ols).fit(cov_type="HC3")

    treatment_coef = float(model.params[treatment_col])
    ci = model.conf_int().loc[treatment_col]
    ci_low, ci_high = float(ci[0]), float(ci[1])

    return {
        "outcome": outcome_col,
        "treatment_coef": treatment_coef,
        "treatment_ci_low": ci_low,
        "treatment_ci_high": ci_high,
        "covariates": covariate_cols,
        "n_obs": int(len(y_clean)),
        "method": "OLS (statsmodels HC3)",
    }
