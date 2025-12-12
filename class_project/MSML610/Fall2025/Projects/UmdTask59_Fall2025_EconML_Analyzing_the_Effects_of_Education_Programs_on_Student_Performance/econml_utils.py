"""
Utility functions and small wrapper API around EconML estimators for:

"EconML: Analyzing the Effects of Education Programs on Student Performance"

This module is responsible for:
- Loading and preparing the student performance dataset,
- Defining a configuration object for EconML estimators,
- Building, fitting, and querying EconML treatment-effect models.
"""

from dataclasses import dataclass
from typing import List, Optional, Any

import pandas as pd

# TODO: import EconML estimators and any sklearn models once the environment is ready.
# from econml.dml import LinearDML
# from econml.dml import CausalForestDML
# from sklearn.linear_model import ElasticNet
# from sklearn.ensemble import RandomForestRegressor


@dataclass
class EconMLEducationConfig:
    """
    Configuration for setting up an EconML estimator on the student performance data.

    Attributes
    ----------
    outcome_col : str - Name of the outcome (Y) column, e.g. "G3".
    treatment_col : str - Name of the treatment (T) column, e.g. "schoolsup".
    x_cols : List[str] - Feature columns used to model treatment heterogeneity (X).
    w_cols : Optional[List[str]] - Optional control columns (W) used to adjust for confounding.
    estimator_type : str - Which EconML estimator to build, e.g. "linear_dml" or "causal_forest".
    """
    outcome_col: str
    treatment_col: str
    x_cols: List[str]
    w_cols: Optional[List[str]] = None
    estimator_type: str = "linear_dml"


def load_student_data(path: str) -> pd.DataFrame:
    """
    Load the student performance dataset from a CSV file.

    Parameters
    ----------
    path : str - Path to the CSV file (e.g., 'data/student-mat.csv').

    Returns
    -------
    pd.DataFrame
        Loaded student performance data.
    """
    df = pd.read_csv(path, sep=";")
    return df


def clean_student_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning and type conversions to the student dataset.

    This function is intentionally minimal for now and will be expanded in Phase 2.

    Parameters
    ----------
    df : pd.DataFrame - Raw student data.

    Returns
    -------
    pd.DataFrame
        Cleaned student data.
    """
    
    # TODO: implement any cleaning logic needed (e.g., handle missing values).
    
    return df


def build_econml_estimator(config: EconMLEducationConfig) -> Any:
    """
    Build an EconML estimator based on the provided configuration.

    This is a placeholder that will be implemented in a later phase.

    Parameters
    ----------
    config : EconMLEducationConfig - Configuration specifying which estimator to use.

    Returns
    -------
    Any
        An unfit EconML estimator instance.
        
    """
    # TODO: create and return a specific EconML estimator instance.
    
    raise NotImplementedError("build_econml_estimator is not implemented yet.")


def fit_econml_estimator(df: pd.DataFrame, config: EconMLEducationConfig) -> Any:
    """
    Fit an EconML estimator using the given DataFrame and configuration.

    Parameters
    ----------
    df : pd.DataFrame - Student data containing outcome, treatment, and feature columns.
    config : EconMLEducationConfig - Configuration defining Y, T, X, W, and estimator type.

    Returns
    -------
    Any
        A fitted EconML estimator.
    """
    
    # TODO: implement splitting df into Y, T, X, W and calling .fit on the estimator.
    
    raise NotImplementedError("fit_econml_estimator is not implemented yet.")


def estimate_ate(model: Any, df: pd.DataFrame, config: EconMLEducationConfig) -> float:
    """
    Estimate the Average Treatment Effect (ATE) using a fitted EconML model.

    Parameters
    ----------
    model : Any - A fitted EconML estimator.
    df : pd.DataFrame - DataFrame used for querying the treatment effects.
    config : EconMLEducationConfig - Configuration defining the feature columns.

    Returns
    -------
    float
        Estimated ATE.
    """
    
    # TODO: implement a call to model.ate or equivalent.
    
    raise NotImplementedError("estimate_ate is not implemented yet.")


def estimate_cate_by_subgroup(
    model: Any,
    df: pd.DataFrame,
    config: EconMLEducationConfig,
    subgroup_col: str,
) -> pd.DataFrame:
    """
    Estimate subgroup-level CATEs by averaging individual effects within each subgroup.

    Parameters
    ----------
    model : Any - A fitted EconML estimator.
    df : pd.DataFrame
    config : EconMLEducationConfig - Configuration defining X and W.
    subgroup_col : str - Column to group by when aggregating CATEs (e.g., 'sex' or 'Medu').

    Returns
    -------
    pd.DataFrame
        DataFrame with subgroup values and corresponding average CATEs.
    """
    
    # TODO: implement CATE prediction and aggregation by subgroup.
    
    raise NotImplementedError("estimate_cate_by_subgroup is not implemented yet.")