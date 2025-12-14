"""Utility helpers for MSML610 class project.

This module wraps a few common operations needed in the notebooks:

* Fetching the **Student Performance** dataset from the UCI ML Repository
* Applying light cleaning and type conversions
* Defining a configuration object for EconML estimators
* Splitting a pandas.DataFrame into (Y, T, X, W) blocks
* Simple summaries of the treatment vs. control groups

The actual EconML model construction and fitting will be implemented
in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


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

    # Additional controls for confounding (e.g., previous grades).
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


# ---------------------------------------------------------------------
# Placeholders for later phases (EconML estimators)
# ---------------------------------------------------------------------


def build_econml_estimator(config: EconMLEducationConfig) -> Any:
    """Build an EconML estimator based on the provided configuration.

    This is a placeholder that will be implemented in a later phase.
    """
    raise NotImplementedError("build_econml_estimator is not implemented yet.")


def fit_econml_estimator(df: pd.DataFrame, config: EconMLEducationConfig) -> Any:
    """Fit an EconML estimator using the given DataFrame and configuration.

    This function will split the data into (Y, T, X, W) and call
    ``.fit`` on the appropriate EconML estimator.
    """
    raise NotImplementedError("fit_econml_estimator is not implemented yet.")


def estimate_ate(model: Any, df: pd.DataFrame, config: EconMLEducationConfig) -> float:
    """Estimate the Average Treatment Effect (ATE) using a fitted model.

    The implementation will call ``model.ate`` (or an equivalent
    method) and return a scalar effect estimate.
    """
    raise NotImplementedError("estimate_ate is not implemented yet.")


def estimate_cate_by_subgroup(
    model: Any,
    df: pd.DataFrame,
    config: EconMLEducationConfig,
    subgroup_col: str,
) -> pd.DataFrame:
    """Estimate subgroup-level CATEs by averaging individual effects.

    The implementation will:

    1. Use the EconML model to compute individual-level treatment
       effects for each student.
    2. Group those effects by the provided ``subgroup_col`` (e.g.,
       "sex" or "Medu").
    3. Return a small DataFrame of subgroup averages and counts.
    """
    raise NotImplementedError("estimate_cate_by_subgroup is not implemented yet.")