"""Utility helpers for MSML610 class project.

This module wraps common operations needed for the causal inference
analysis in the project notebooks, specifically using EconML's
Double Machine Learning (DML) framework on the **Student Performance**
dataset.

The module provides helpers for:

* **Data Management**:
    * Fetching and applying light cleaning to the **Student Performance**
      dataset from the UCI ML Repository.
    * Splitting a pandas.DataFrame into the required (Y, T, X, W) blocks
      for EconML estimators.
    * Defining a configuration object (`EconMLEducationConfig`) to specify
      the outcome (Y), treatment (T), heterogeneity features (X), and
      optional controls (W).
    * Simple summaries of the treatment vs. control groups.

* **Causal Estimation**:
    * Building an unfitted DML estimator (LinearDML, CausalForestDML,
      SparseLinearDML) based on the configuration.
    * Fitting the chosen DML estimator with a binary treatment model.
    * Estimating the **Average Treatment Effect (ATE)** and its confidence interval.
    * Estimating the **Conditional Average Treatment Effect (CATE)** by
      specified subgroups.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from econml.dml import LinearDML, CausalForestDML, SparseLinearDML


# ---------------------------------------------------------------------
# Configuration object
# ---------------------------------------------------------------------

@dataclass
class EconMLEducationConfig:
    """Configuration for running EconML on student performance data.

    Attributes
    ----------
    outcome_col : str
        Name of the outcome (Y) column, e.g. ``"G3"`` for the final grade.
    treatment_col : str
        Name of the treatment (T) column, e.g. ``"schoolsup"`` indicating
        whether the student received extra school support.
    x_cols : list[str]
        Feature columns used to model **treatment effect heterogeneity** (X).
    w_cols : list[str] | None, optional
        Optional control columns (W) used to adjust for confounding.
        If ``None``, no controls are used.
    estimator_type : str
        String identifier for which EconML estimator to build, e.g.
        ``"linear_dml"`` or ``"causal_forest"``. The interpretation of this
        field will be handled inside :func:`build_econml_estimator`.
    """

    outcome_col: str
    treatment_col: str
    x_cols: List[str]
    w_cols: Optional[List[str]] = None
    estimator_type: str = "linear_dml"

# ---------------------------------------------------------------------
# Data loading / cleaning
# ---------------------------------------------------------------------

def load_student_data(
    source: str = "ucimlrepo",
    local_path: Optional[str] = None,
) -> pd.DataFrame:
    
    """Load the Student Performance dataset as a single DataFrame.

    Parameters
    ----------
    source : {"ucimlrepo", "csv"}, default="ucimlrepo"
        - ``"ucimlrepo"``: download the dataset on the fly using the
          :mod:`ucimlrepo` package and dataset ID 320.
        - ``"csv"``: read from a locally available CSV file (e.g.,
          the original UCI ``student-mat.csv`` or a merged version).
    local_path : str, optional
        Path to a local CSV file when ``source="csv"`` is used.

    Returns
    -------
    pd.DataFrame
        DataFrame containing both features and targets. When using
        ``ucimlrepo``, this is simply ``pd.concat([X, y], axis=1)`` where
        ``X`` are the features and ``y`` are the targets from the package.

    Notes
    -----
    The project notebooks will call this helper so that users can
    reproduce the analysis without manually downloading the data.
    If ``ucimlrepo`` is not installed or the environment has no
    internet access, you can switch to ``source="csv"`` and point
    ``local_path`` to a checked in copy of the dataset.
    """

    if source == "ucimlrepo":
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise ImportError(
                "ucimlrepo is not installed. Install it with "
                "`pip install ucimlrepo` or use source='csv'."
            ) from exc

        # ID 320 corresponds to the UCI Student Performance dataset.
        student_performance = fetch_ucirepo(id=320)
        X = student_performance.data.features
        y = student_performance.data.targets

        # Combine features and targets into a single DataFrame.
        df = pd.concat([X, y], axis=1)
        return df

    if source == "csv":
        if local_path is None:
            raise ValueError(
                "When source='csv', you must provide local_path to a CSV file."
            )
        # The original UCI CSV files use ';' as the separator.
        df = pd.read_csv(local_path, sep=";")
        return df

    raise ValueError("source must be either 'ucimlrepo' or 'csv'")


def clean_student_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning and type conversions.

    This function deliberately keeps the cleaning very light:
    the raw dataset has no missing values, so we mainly convert
    binary yes/no flags into 0/1 and a few categorical fields into
    binary indicators.

    The returned DataFrame is suitable for passing into helper
    functions that construct (Y, T, X, W) arrays for EconML.

    Parameters
    ----------
    df : pd.DataFrame
        Raw student data as loaded by :func:`load_student_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned student data with consistent numeric encodings
        for the most important binary variables.
    """

    df = df.copy()

    # Map yes/no style flags to 0/1.
    binary_yn_cols = [
        "schoolsup",
        "famsup",
        "paid",
        "activities",
        "nursery",
        "higher",
        "internet",
        "romantic",
    ]
    for col in binary_yn_cols:
        if col in df.columns:
            df[col] = df[col].map({"no": 0, "yes": 1}).astype("int64")

    # Map a few key categorical variables to binary indicators.
    mapping_specs = {
        "sex": {"F": 0, "M": 1},
        "address": {"R": 0, "U": 1},
        "Pstatus": {"A": 0, "T": 1},
        "school": {"MS": 0, "GP": 1},
    }
    for col, mapping in mapping_specs.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype("int64")

    return df


# ---------------------------------------------------------------------
# Helper functions for (Y, T, X, W)
# ---------------------------------------------------------------------


def make_default_config() -> EconMLEducationConfig:
    """Create a default configuration for the education program analysis.

    This configuration reflects the choices described in the project
    writeup:

    * Outcome (Y): final grade (G3)
    * Treatment (T): extra school support (schoolsup)
    * Features (X): a mix of demographics and study habits
    * Controls (W): prior grades and a few additional flags

    The choices are not unique, but they provide a reasonable starting
    point for estimating the causal effect of school support on grades.
    """

    outcome_col = "G3"
    treatment_col = "schoolsup"

    # Columns that may drive heterogeneity in the treatment effect.
    x_cols = [
        "sex",        # 0 = female, 1 = male
        "age",
        "Medu",       # mother's education
        "Fedu",       # father's education
        "studytime",
        "failures",
        "higher",     # plans for higher education (0/1)
        "internet",   # internet access (0/1)
        "absences",
    ]

    # Additional controls for confounding (such as previous grades).
    w_cols = [
        "G1",
        "G2",
        "famsup",
        "paid",
    ]

    return EconMLEducationConfig(
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        x_cols=x_cols,
        w_cols=w_cols,
        estimator_type="linear_dml",
    )


def split_y_t_x_w(
    df: pd.DataFrame,
    config: EconMLEducationConfig,
) -> Dict[str, Any]:
    """Split a DataFrame into Y, T, X, W arrays according to the config.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned student data.
    config : EconMLEducationConfig
        Configuration specifying which columns to use.

    Returns
    -------
    dict
        Dictionary with the following keys:

        * ``"y"``: 1D numpy array of shape (n_samples,)
        * ``"t"``: 1D numpy array of shape (n_samples,)
        * ``"X"``: 2D numpy array of shape (n_samples, n_features_X)
        * ``"W"``: 2D numpy array or ``None`` if no controls are used.
    """

    y = df[config.outcome_col].to_numpy()
    t = df[config.treatment_col].to_numpy()
    X = df[config.x_cols].to_numpy()

    if config.w_cols:
        W = df[config.w_cols].to_numpy()
    else:
        W = None

    return {"y": y, "t": t, "X": X, "W": W}


def summarize_treatment(
    df: pd.DataFrame,
    config: EconMLEducationConfig,
) -> pd.DataFrame:
    """Compute simple summary statistics by treatment group.

    For the EDA section in the example notebook:
    it shows how many students are treated vs. control and their
    average outcomes.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned student data.
    config : EconMLEducationConfig
        Configuration specifying outcome and treatment columns.

    Returns
    -------
    pd.DataFrame
        Summary with one row per treatment value.
    """

    group = df.groupby(config.treatment_col)[config.outcome_col]
    summary = group.agg(
        count="count",
        mean_outcome="mean",
        std_outcome="std",
    ).reset_index()

    # For readability, rename the treatment column to "treatment".
    summary = summary.rename(columns={config.treatment_col: "treatment"})
    return summary

def build_econml_estimator(config: EconMLEducationConfig) -> Any:
    """Build an EconML Double Machine Learning estimator.

    This function centralizes which EconML estimator we use for this project.
    It maps a simple string in ``config.estimator_type`` to a concrete DML
    estimator class and configures a treatment model with enough iterations
    to avoid L-BFGS convergence warnings.

    See the API docs for a detailed discussion of why these estimators are
    appropriate for observational data with rich covariates.

    All of these estimators:
    - Implement Double Machine Learning under the **unconfoundedness**
      assumption (all confounders are measured).
    - Support **binary treatments** via ``discrete_treatment=True``.
    - Use ML models internally to learn the nuisance functions
      (treatment and outcome models) and then estimate treatment effects
      on residualized data.

    Parameters
    ----------
    config : EconMLEducationConfig
        Configuration specifying the estimator type.

    Returns
    -------
    Any
        An unfit EconML estimator instance.
    """
    est_type = config.estimator_type.lower()

    # Logistic regression model for the binary treatment (schoolsup).
    # Explicitly increased max_iter so that the lbfgs solver has room to converge.
    logit_t = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )
    if est_type == "linear_dml":
        estimator = LinearDML(
            model_t=logit_t,
            discrete_treatment=True,
            random_state=42,
        )
    elif est_type == "causal_forest":
        estimator = CausalForestDML(
            model_t=logit_t,
            discrete_treatment=True,
            random_state=42,
        )
    elif est_type == "sparse_linear_dml":
        estimator = SparseLinearDML(
            model_t=logit_t,
            discrete_treatment=True,
            random_state=42,
        )

    else:
        raise ValueError(
            f"Unknown estimator_type '{config.estimator_type}'. "
            "Use 'linear_dml', 'causal_forest', or 'sparse_linear_dml'."
        )

    return estimator

def fit_econml_estimator(
    df: pd.DataFrame,
    config: EconMLEducationConfig,
    estimator: Optional[Any] = None,
) -> Any:
    """Fit an EconML estimator using the given DataFrame and configuration.

    This helper:
    1. Builds an estimator (if one is not provided),
    2. Uses :func:`split_y_t_x_w` to construct (Y, T, X, W),
    3. Calls ``estimator.fit(Y, T, X=X, W=W)`` as per the EconML API.

    Parameters
    ----------
    df : pandas.DataFrame
        Student data containing outcome, treatment, heterogeneity features,
        and (optionally) controls.
    config : EconMLEducationConfig
        Configuration defining Y, T, X, W, and estimator type.
    estimator : Any, optional
        Pre-constructed EconML estimator. If ``None``, a new estimator
        is created via :func:`build_econml_estimator`.

    Returns
    -------
    Any
        A fitted EconML estimator.
    """
    if estimator is None:
        estimator = build_econml_estimator(config)

    arrays = split_y_t_x_w(df, config)
    y, t, X, W = arrays["y"], arrays["t"], arrays["X"], arrays["W"]

    estimator.fit(y, t, X=X, W=W)
    return estimator

def estimate_ate(
    model: Any,
    df: pd.DataFrame,
    config: EconMLEducationConfig,
) -> Dict[str, float]:
    """Estimate the Average Treatment Effect (ATE) and its confidence interval.

    This helper calls the EconML model's ``ate`` and ``ate_interval`` methods
    and normalizes the result into a simple dictionary.

    Parameters
    ----------
    model : Any
        A fitted EconML estimator (LinearDML, SparseLinearDML or CausalForestDML).
    df : pandas.DataFrame
        Data used for querying the ATE (usually the evaluation or full sample).
    config : EconMLEducationConfig
        Configuration defining which columns form X.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"ate"``: point estimate of the ATE
        - ``"ate_ci_lower"``: lower bound of 95% CI
        - ``"ate_ci_upper"``: upper bound of 95% CI
    """
    X = df[config.x_cols].to_numpy()

    # `ate` can return a scalar or a 1D array; convert safely to a float.
    ate_value = model.ate(X=X)
    ate_value = float(np.asarray(ate_value).ravel()[0])

    ci_low, ci_high = model.ate_interval(X=X)
    ci_low = float(np.asarray(ci_low).ravel()[0])
    ci_high = float(np.asarray(ci_high).ravel()[0])

    return {
        "ate": ate_value,
        "ate_ci_lower": ci_low,
        "ate_ci_upper": ci_high,
    }

def estimate_cate_by_subgroup(
    model: Any,
    df: pd.DataFrame,
    config: EconMLEducationConfig,
    subgroup_col: str,
) -> pd.DataFrame:
    """Estimate subgroup-level Conditional Average Treatment Effects (CATEs).

    Steps:
    1. Use the fitted EconML model to compute **individual-level**
       treatment effects via ``model.effect(X)``.
    2. Attach those effects back to the DataFrame.
    3. Group by ``subgroup_col`` (e.g., ``"sex"`` or ``"Medu"``) and
       compute mean, standard deviation, and count.

    Parameters
    ----------
    model : Any
        A fitted EconML estimator.
    df : pandas.DataFrame
        Data containing the subgroup column and heterogeneity features X.
    config : EconMLEducationConfig
        Configuration specifying which columns form X.
    subgroup_col : str
        Column name to group by when aggregating individual treatment effects.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per subgroup and columns:

        - ``subgroup``: subgroup label (e.g., 'F' vs 'M' or education level)
        - ``cate_mean``: mean estimated treatment effect for that subgroup
        - ``cate_std``: standard deviation of individual effects
        - ``n``: number of students in the subgroup
    """
    df = df.copy()
    X = df[config.x_cols].to_numpy()

    # Individual-level treatment effect estimates (CATE_i).
    te = model.effect(X=X)
    df["_individual_te"] = np.asarray(te).ravel()

    grouped = (
        df.groupby(subgroup_col)["_individual_te"]
        .agg(cate_mean="mean", cate_std="std", n="count")
        .reset_index()
        .rename(columns={subgroup_col: "subgroup"})
    )

    return grouped