"""
lime_attrition_utils.py

Reusable utilities for the "Employee Attrition Prediction with LIME" project.

This module is used by:
- lime_attrition.API.ipynb      -> tool's programming interface (API layer)
- lime_attrition.example.ipynb  -> full attrition analysis (example layer)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import warnings
import re

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
)

from lime.lime_tabular import LimeTabularExplainer





# ---------------------------------------------------------------------
# 0. INTERNAL HELPERS
# ---------------------------------------------------------------------
def _to_dense(x: Any) -> np.ndarray:
    """Convert sparse-like matrices to a dense numpy array (LIME-friendly)."""
    # scipy.sparse matrices expose .toarray(); sklearn sometimes returns np.matrix
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x)
    return x

def _ensure_2d(x: Any) -> np.ndarray:
    """Ensure x is a 2D numpy array."""
    x = _to_dense(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x

def _ensure_1d_row(x: Any) -> np.ndarray:
    """Ensure x is a 1D feature row."""
    x = _to_dense(x)
    return x.ravel()

# ---------------------------------------------------------------------
# 0A. TOOL-ONLY LIME HELPERS 
# ---------------------------------------------------------------------
def lime_predict_proba_fn(model: Any):
    """Return a `predict_fn` compatible with LIME for a probabilistic classifier.

    Parameters
    ----------
    model : Any
        Any object exposing `predict_proba(X)` that returns shape (n_samples, n_classes).

    Returns
    -------
    callable
        A function `predict_fn(X)` returning class probabilities as a numpy array.
    """
    def predict_fn(x: Any) -> np.ndarray:
        x2 = _ensure_2d(x)
        probs = model.predict_proba(x2)
        return np.asarray(probs)
    return predict_fn


def lime_build_tabular_explainer(
    training_data: Any,
    feature_names: List[str],
    class_names: List[str],
    discretize_continuous: bool = True,
    mode: str = "classification",
    **kwargs: Any,
) -> LimeTabularExplainer:
    """Create a `LimeTabularExplainer` for tabular classification.

    This is a thin convenience wrapper around the native LIME constructor.

    Notes
    -----
    - `training_data` must be in the same feature space as the `predict_fn` you pass to
      `explainer.explain_instance(...)`.
    """
    training_data = _to_dense(training_data)
    return LimeTabularExplainer(
        training_data=training_data,
        feature_names=feature_names,
        class_names=class_names,
        mode=mode,
        discretize_continuous=discretize_continuous,
        **kwargs,
    )


def lime_explain_instance(
    explainer: LimeTabularExplainer,
    data_row: Any,
    predict_fn: Any,
    num_features: int = 10,
    num_samples: int = 5000,
    **kwargs: Any,
):
    """Explain a single instance using native LIME `explainer.explain_instance`.

    Parameters
    ----------
    explainer : LimeTabularExplainer
        Fitted explainer.
    data_row : Any
        1D feature vector in the explainer's feature space.
    predict_fn : callable
        Function returning probabilities of shape (n_samples, n_classes).
    num_features : int
        Number of top features in the explanation.
    num_samples : int
        Number of perturbed samples LIME generates.

    Returns
    -------
    lime.explanation.Explanation
        Native LIME explanation object.
    """
    row = _ensure_1d_row(data_row)
    return explainer.explain_instance(
        data_row=row,
        predict_fn=predict_fn,
        num_features=num_features,
        num_samples=num_samples,
        **kwargs,
    )


def lime_explanation_to_df(explanation: Any, top_k: Optional[int] = None) -> pd.DataFrame:
    """Convert a native LIME `Explanation` into a compact table.

    Returns a DataFrame with columns: feature, weight, abs_weight.
    """
    pairs = explanation.as_list()
    if top_k is not None:
        pairs = pairs[: int(top_k)]
    df = pd.DataFrame(pairs, columns=["feature", "weight"])
    df["abs_weight"] = df["weight"].abs()
    return df


class LimeTabularWrapper:
    """A lightweight, generic wrapper around the native LIME tabular API.

    This wrapper is *tool-only*. It simply
    standardizes a few common steps and returns a pandas DataFrame for readability.
    """

    def __init__(self, explainer: LimeTabularExplainer, predict_fn: Any, class_names: List[str]):
        self.explainer = explainer
        self.predict_fn = predict_fn
        self.class_names = class_names

    def explain(self, data_row: Any, top_k: int = 10, num_samples: int = 5000) -> pd.DataFrame:
        exp = lime_explain_instance(
            explainer=self.explainer,
            data_row=data_row,
            predict_fn=self.predict_fn,
            num_features=top_k,
            num_samples=num_samples,
        )
        df = lime_explanation_to_df(exp, top_k=top_k)
        return df

# ---------------------------------------------------------------------
# 1. CONFIG DATA CLASSES
# ---------------------------------------------------------------------

@dataclass
class AttritionDataConfig:
    """
    Configuration for handling the IBM HR Attrition dataset.

    Attributes
    ----------
    target_column : str
        Name of the attrition label column.
    id_columns : List[str]
        Columns to drop because they are IDs or constants.
    test_size : float
        Proportion of data to reserve for testing.
    random_state : int
        Random seed for reproducible splits.
    """
    target_column: str = "Attrition"
    id_columns: List[str] | None = None
    test_size: float = 0.2
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.id_columns is None:
            # Common choices for the IBM dataset
            self.id_columns = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]


@dataclass
class ModelConfig:
    """
    Configuration for training attrition models.

    Notes:
    - We keep a small set of shared hyperparameters (lr, n_estimators, max_depth)
      and model-specific blocks (random forest).
    """
    use_xgboost: bool = True
    use_lightgbm: bool = True
    use_random_forest: bool = True

    # Shared-ish boosting knobs
    learning_rate: float = 0.05
    n_estimators: int = 300
    max_depth: int = 3
    random_state: int = 42

    # Random Forest knobs
    rf_n_estimators: int = 500
    rf_max_depth: int | None = None
    rf_min_samples_leaf: int = 1
    rf_max_features: str | int | float = "sqrt"
    rf_class_weight: str | dict | None = "balanced"

    # Parallelism where supported
    n_jobs: int = -1

@dataclass
class LimeConfig:
    """
    Configuration for LIME explanations. This is also the default number
    """
    num_features: int = 10
    num_samples: int = 5000


@dataclass
class ModelArtifacts:
    """
    Container for all key pieces produced by the pipeline.
    """
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    trained_models: Dict[str, Any]   # e.g. {"xgboost": model, "lightgbm": model}
    metrics: Dict[str, Dict[str, float]]  # e.g. {"xgboost": {"accuracy": 0.9, ...}, ...}


# ---------------------------------------------------------------------
# 2. DATA LOADING & CLEANING
# ---------------------------------------------------------------------

def load_raw_attrition_data(csv_path: str) -> pd.DataFrame:
    """
    Load the IBM HR Employee Attrition dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame as read from disk.
    """
    df = pd.read_csv(csv_path)
    return df


def clean_attrition_data(df: pd.DataFrame, config: AttritionDataConfig) -> pd.DataFrame:
    """
    Basic cleaning and formatting for the attrition dataset.

    What we do here (typical for the IBM dataset):
    - Drop ID / constant columns.
    - Standardize the target to 0/1 (0 = No attrition, 1 = Yes attrition).
    - Drop rows with missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame.
    config : AttritionDataConfig
        Settings for target and ID/constant columns.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame, with target column numeric.
    """
    df = df.copy()

    # Drop ID / constant columns if they exist
    for col in config.id_columns:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Convert target to 0/1
    if df[config.target_column].dtype == "object":
        df[config.target_column] = df[config.target_column].map({"Yes": 1, "No": 0})

    # Simple missing-value handling for now
    df = df.dropna(axis=0)

    return df


# ---------------------------------------------------------------------
# 3. FEATURE PREPARATION
# ---------------------------------------------------------------------

def split_features_target(
    df: pd.DataFrame,
    config: AttritionDataConfig
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the cleaned DataFrame into features X and target y.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned attrition DataFrame.
    config : AttritionDataConfig
        Contains target column name.

    Returns
    -------
    X : pd.DataFrame
        Feature columns.
    y : pd.Series
        Target vector (0/1 attrition).
    """
    y = df[config.target_column]
    X = df.drop(columns=[config.target_column])
    return X, y


def train_test_split_attrition(
    X: pd.DataFrame,
    y: pd.Series,
    config: AttritionDataConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train-test split for attrition data.

    Uses stratification to maintain similar attrition rates in train/test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing pipeline that:
    - One-hot encodes categorical variables (dense output for LIME compatibility).
    - Standardizes numeric variables.

    Notes
    -----
    LIME expects a dense numeric matrix for `training_data` and `data_row`.
    To avoid sparse-matrix edge cases, we force OneHotEncoder to produce dense output
    and also set ColumnTransformer to return dense output.
    """
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # sklearn compatibility: sparse_output (new) vs sparse (old)
    try:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return preprocessor

# ---------------------------------------------------------------------
# 3b. SIMPLE EDA UTILITIES
# ---------------------------------------------------------------------

def compute_attrition_rate(df: pd.DataFrame, config: AttritionDataConfig) -> float:
    """
    Compute overall attrition rate (proportion of employees who left).

    Returns
    -------
    float
        Fraction of rows with Attrition = 1.
    """
    return df[config.target_column].mean()


def categorical_attrition_table(
    df: pd.DataFrame,
    column: str,
    config: AttritionDataConfig,
) -> pd.DataFrame:
    """
    For a categorical column (e.g., OverTime, JobRole), compute:
    - Count of employees in each category
    - Attrition rate in each category

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (with Attrition as 0/1).
    column : str
        Name of the categorical feature to analyze.
    config : AttritionDataConfig
        Provides the target column name.

    Returns
    -------
    pd.DataFrame
        Summary table indexed by category values.
    """
    grouped = (
        df.groupby(column)[config.target_column]
        .agg(["count", "mean"])
        .rename(columns={"count": "n_employees", "mean": "attrition_rate"})
        .sort_values("attrition_rate", ascending=False)
    )
    return grouped


def numeric_summary_by_attrition(
    df: pd.DataFrame,
    column: str,
    config: AttritionDataConfig,
) -> pd.DataFrame:
    """
    For a numeric column (e.g., Age, MonthlyIncome), summarize
    its distribution by attrition status.

    Returns
    -------
    pd.DataFrame
        Table with summary stats for attrition=0 vs attrition=1.
    """
    summary = (
        df
        .groupby(config.target_column)[column]
        .describe()
        .rename(index={0: "stayed", 1: "left"})
    )
    return summary


# ---------------------------------------------------------------------
# 4. MODEL TRAINING
# ---------------------------------------------------------------------

def train_attrition_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    model_config: ModelConfig,
) -> Dict[str, Any]:
    """
    Train one or more attrition models and return dict[name -> fitted Pipeline].

    Important implementation detail:
    - We CLONE the preprocessor for each pipeline to avoid shared mutable state.
    """
    trained: Dict[str, Any] = {}

    def _fit_pipeline(model: Any) -> Pipeline:
        pipe = Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                ("model", model),
            ]
        )
        return pipe.fit(X_train, y_train)

    # 1) Gradient Boosting (sklearn)
    from sklearn.ensemble import GradientBoostingClassifier

    gb_clf = GradientBoostingClassifier(
        learning_rate=model_config.learning_rate,
        n_estimators=model_config.n_estimators,
        max_depth=model_config.max_depth,
        random_state=model_config.random_state,
    )
    trained["gradient_boosting"] = _fit_pipeline(gb_clf)

    # 2) Random Forest (sklearn)
    if model_config.use_random_forest:
        from sklearn.ensemble import RandomForestClassifier

        rf_clf = RandomForestClassifier(
            n_estimators=model_config.rf_n_estimators,
            max_depth=model_config.rf_max_depth,
            min_samples_leaf=model_config.rf_min_samples_leaf,
            max_features=model_config.rf_max_features,
            class_weight=model_config.rf_class_weight,
            random_state=model_config.random_state,
            n_jobs=model_config.n_jobs,
        )
        trained["random_forest"] = _fit_pipeline(rf_clf)

    # 3) XGBoost
    if model_config.use_xgboost:
        try:
            import xgboost as xgb
        except ImportError:
            warnings.warn(
                "xgboost is not installed; skipping XGBoost model training.",
                RuntimeWarning,
            )
        else:
            xgb_clf = xgb.XGBClassifier(
                n_estimators=model_config.n_estimators,
                learning_rate=model_config.learning_rate,
                max_depth=model_config.max_depth,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=model_config.random_state,
                n_jobs=model_config.n_jobs,
            )
            trained["xgboost"] = _fit_pipeline(xgb_clf)

    # 4) LightGBM
    if model_config.use_lightgbm:
        try:
            import lightgbm as lgb
        except ImportError:
            warnings.warn(
                "lightgbm is not installed; skipping LightGBM model training.",
                RuntimeWarning,
            )
        else:
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=model_config.n_estimators,
                learning_rate=model_config.learning_rate,
                max_depth=model_config.max_depth,
                objective="binary",
                random_state=model_config.random_state,
                n_jobs=model_config.n_jobs,
                verbosity=-1, 
            )
            trained["lightgbm"] = _fit_pipeline(lgb_clf)

    return trained




def evaluate_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """
    Compute evaluation metrics for each model.

    Metrics:
      - accuracy
      - precision (positive class = attrition=1)
      - recall (positive class = attrition=1)
      - f1
      - roc_auc
      - pr_auc (average precision)  <-- very useful for imbalanced data
    """
    metrics: Dict[str, Dict[str, float]] = {}

    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "pr_auc": float(average_precision_score(y_test, y_proba)),
        }

    return metrics



# ---------------------------------------------------------------------
# 5. LIME EXPLANATIONS
# ---------------------------------------------------------------------


def build_lime_explainer(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    class_names: List[str],
) -> Any:
    """
    Build a LIME Tabular explainer in the *same numeric feature space* used by the model.

    Requirements
    ------------
    - `preprocessor` must be a *fitted* ColumnTransformer (e.g., from a trained pipeline).
    - We transform X_train into numeric space and pass a dense matrix to LIME.
    """
    # Guardrail: callers sometimes pass an unfitted preprocessor by accident.
    if not hasattr(preprocessor, "transformers_"):
        raise ValueError(
            "`preprocessor` must be fitted before calling build_lime_explainer(). "
            "Pass pipeline.named_steps['preprocess'] from a trained pipeline."
        )

    X_train_transformed = _to_dense(preprocessor.transform(X_train))

    # Human-readable feature names when available
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]

    explainer = LimeTabularExplainer(
        training_data=X_train_transformed,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
    )

    return explainer


def explain_single_employee(
    explainer: Any,
    model_pipeline: Any,
    raw_row: pd.Series,
    preprocessor: ColumnTransformer,
    lime_config: LimeConfig,
) -> Any:
    """
    Use LIME to explain a single employee's attrition prediction.

    Assumptions
    -----------
    - `model_pipeline` is a fitted sklearn Pipeline with steps:
        ("preprocess", ColumnTransformer), ("model", classifier)
    - `preprocessor` is the same *fitted* ColumnTransformer used in the pipeline.

    Implementation details
    ----------------------
    LIME operates in the numeric feature space. We therefore:
      1) transform the raw row using the fitted preprocessor,
      2) define a predict_fn that takes numeric-space arrays and calls the classifier,
      3) call explainer.explain_instance() on a 1D dense data_row.
    """
    # 1) Put the row in a 2D DataFrame (shape 1 x n_features)
    row_df = raw_row.to_frame().T

    # 2) Transform to numeric feature space (force dense for LIME)
    row_transformed = _to_dense(preprocessor.transform(row_df))
    data_row = _ensure_1d_row(row_transformed[0])

    # 3) Define prediction function that works directly on transformed data
    clf = model_pipeline.named_steps["model"]

    def predict_fn(z: Any) -> np.ndarray:
        z2 = _ensure_2d(z)
        probs = clf.predict_proba(z2)
        return np.asarray(probs)

    # 4) Explain this single instance
    explanation = explainer.explain_instance(
        data_row=data_row,
        predict_fn=predict_fn,
        num_features=lime_config.num_features,
        num_samples=lime_config.num_samples,
    )

    return explanation


def batch_lime_explanations(
    explainer: Any,
    model_pipeline: Any,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    preprocessor: ColumnTransformer,
    lime_config: LimeConfig,
    top_n: int = 10,
    top_k_features: int = 5,
) -> pd.DataFrame:
    """
    Generate LIME explanations for the top-N highest-risk employees.

    Parameters
    ----------
    explainer : LimeTabularExplainer
        Fitted LIME explainer.
    model_pipeline : Any
        Trained sklearn Pipeline (e.g. models["xgboost"]).
    X : pd.DataFrame
        Feature matrix for the population of interest
        (e.g., X_test or the full workforce).
    y : pd.Series or None
        True labels (0/1) if available; can be None in real deployment.
    preprocessor : ColumnTransformer
        Fitted preprocessor used in the pipeline.
    lime_config : LimeConfig
        LIME settings (num_features, num_samples).
    top_n : int, default=10
        Number of highest-risk employees to explain.
    top_k_features : int, default=5
        Number of top LIME features to include in the summary.

    Returns
    -------
    pd.DataFrame
        Table with one row per high-risk employee, containing:
        - rank
        - row_index (index in X)
        - predicted_leave_prob
        - actual_attrition (if y is provided)
        - top_factors (string summary of top features)
    """
    # 1. Predict probabilities of leaving for all rows in X
    leave_proba = model_pipeline.predict_proba(X)[:, 1]

    # 2. Sort indices by highest predicted risk
    order = np.argsort(-leave_proba)
    top_n = min(top_n, len(order))
    top_indices = order[:top_n]

    rows: List[Dict[str, Any]] = []

    for rank, idx in enumerate(top_indices, start=1):
        raw_row = X.iloc[idx]

        # 3. Get LIME explanation for this row
        explanation = explain_single_employee(
            explainer=explainer,
            model_pipeline=model_pipeline,
            raw_row=raw_row,
            preprocessor=preprocessor,
            lime_config=lime_config,
        )

        # 4. Take top-k features and format them nicely
        contribs = explanation.as_list()[:top_k_features]
        factor_strings = [
            f"{feat} ({weight:+.2f})" for feat, weight in contribs
        ]
        factors_summary = "; ".join(factor_strings)

        row_dict: Dict[str, Any] = {
            "rank": rank,
            "row_index": int(X.index[idx]),
            "predicted_leave_prob": float(leave_proba[idx]),
            "top_factors": factors_summary,
        }

        if y is not None:
            row_dict["actual_attrition"] = int(y.iloc[idx])

        rows.append(row_dict)

    result_df = pd.DataFrame(rows).sort_values(
        "predicted_leave_prob", ascending=False
    )

    return result_df

def lime_explanation_to_long_df(
    explanation: Any,
    row_index: int,
    predicted_leave_prob: float | None = None,
    actual_attrition: int | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    """
    Convert a single LIME Explanation into a long-form DataFrame.

    Output columns:
      row_index, feature, weight, abs_weight, direction, predicted_leave_prob, actual_attrition
    """
    pairs = explanation.as_list()
    if top_k is not None:
        pairs = pairs[:top_k]

    rows: List[Dict[str, Any]] = []
    for feat, w in pairs:
        w = float(w)
        rows.append(
            {
                "row_index": int(row_index),
                "feature": str(feat),
                "feature_hr": translate_lime_feature(str(feat)),
                "weight": w,
                "abs_weight": abs(w),
                "direction": "push_leave" if w > 0 else "push_stay",
                "predicted_leave_prob": None if predicted_leave_prob is None else float(predicted_leave_prob),
                "actual_attrition": None if actual_attrition is None else int(actual_attrition),
            }
        )
    return pd.DataFrame(rows)



_LIME_CAT_RE = re.compile(r"^cat__([^_]+)_(.+?)\s*(<=|<|>=|>)\s*([-\d.]+)$")
_LIME_NUM_RE = re.compile(r"^num__([^ ]+)\s*(<=|<|>=|>)\s*([-\d.]+)(?:\s+and\s+([-\d.]+)\s*<\s+[^ ]+)?$")

def translate_lime_feature(feature_str: str) -> str:
    """
    Translate encoded LIME feature strings into human-readable text.

    Examples:
      - 'cat__OverTime_No <= 0.00'  -> 'OverTime ≠ No' (binary case implies likely Yes)
      - 'cat__BusinessTravel_Travel_Frequently > 0.00' -> 'BusinessTravel = Travel_Frequently'
      - 'num__JobSatisfaction <= -0.65' -> 'JobSatisfaction is low (standardized)'
    """
    s = feature_str.strip()

    # Categorical one-hot style: cat__<col>_<level> <= 0.00 or > 0.00
    m = _LIME_CAT_RE.match(s)
    if m:
        col, level, op, thresh = m.groups()
        thresh = float(thresh)

        # For 0/1 one-hot outputs, LIME often uses <=0.00 or >0.00
        if op in ("<=", "<") and thresh <= 0.0:
            # means the one-hot is 0 (level NOT present)
            return f"{col} ≠ {level}"
        if op in (">", ">=") and thresh >= 0.0:
            # means the one-hot is 1 (level present)
            return f"{col} = {level}"

        # fallback generic
        return f"{col} {op} {thresh:g} for level {level}"

    # Numeric scaled feature: num__<col> <= value
    # Note: values are in STANDARDIZED (z-score) space because of StandardScaler.
    m2 = _LIME_NUM_RE.match(s)
    if m2:
        col, op, v1, v2 = m2.groups()
        v1 = float(v1)

        # If LIME gives a range like "-1.05 < num__X <= 0.37", it might not match perfectly,
        # so we keep a generic version when that happens.
        if v2 is not None:
            return f"{col} is in a mid range (standardized)"

        # Simple heuristics in standardized space
        if op in ("<=", "<") and v1 <= -0.5:
            return f"{col} is low"
        if op in (">", ">=") and v1 >= 0.5:
            return f"{col} is high"
        return f"{col} is around average"

    # Fallback: return original if we couldn't parse
    return feature_str


def batch_lime_explanations_long(
    explainer: Any,
    model_pipeline: Any,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    preprocessor: ColumnTransformer,
    lime_config: LimeConfig,
    top_n: int = 50,
    top_k_features: int = 10,
) -> pd.DataFrame:
    """
    Produce a long-form table of LIME weights for the top-N highest-risk employees.
    """
    leave_proba = model_pipeline.predict_proba(X)[:, 1]
    order = np.argsort(-leave_proba)
    top_n = min(top_n, len(order))
    top_indices = order[:top_n]

    all_rows: List[pd.DataFrame] = []
    for idx in top_indices:
        raw_row = X.iloc[idx]

        exp = explain_single_employee(
            explainer=explainer,
            model_pipeline=model_pipeline,
            raw_row=raw_row,
            preprocessor=preprocessor,
            lime_config=lime_config,
        )

        row_index = int(X.index[idx])
        actual = None if y is None else int(y.iloc[idx])
        df_one = lime_explanation_to_long_df(
            explanation=exp,
            row_index=row_index,
            predicted_leave_prob=float(leave_proba[idx]),
            actual_attrition=actual,
            top_k=top_k_features,
        )
        all_rows.append(df_one)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def aggregate_lime_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate many local explanations into a global-ish summary.

    Returns a DataFrame with:
      feature, count, mean_weight, mean_abs_weight, pct_push_leave
    """
    if long_df.empty:
        return pd.DataFrame(columns=["feature", "count", "mean_weight", "mean_abs_weight", "pct_push_leave"])

    agg = (
        long_df.groupby("feature_hr")
        .agg(
            count=("feature", "size"),
            mean_weight=("weight", "mean"),
            mean_abs_weight=("abs_weight", "mean"),
            pct_push_leave=("direction", lambda s: float((s == "push_leave").mean())),
        )
        .sort_values("mean_abs_weight", ascending=False)
        .reset_index()
    )
    return agg


def plot_lime_aggregate_bar(
    agg_df: pd.DataFrame,
    top_n: int = 15,
    sort_by: str = "mean_abs_weight",
    title: str = "Top LIME drivers (aggregated)",
):
    """
    Simple matplotlib horizontal bar chart of aggregated LIME importance.
    """
    import matplotlib.pyplot as plt

    if agg_df.empty:
        raise ValueError("agg_df is empty—run batch_lime_explanations_long() first.")

    show = agg_df.sort_values(sort_by, ascending=False).head(top_n).copy()
    show = show.iloc[::-1]  # reverse for nicer top-to-bottom plot

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(show["feature"], show[sort_by])
    ax.set_title(title)
    ax.set_xlabel(sort_by)
    ax.set_ylabel("feature")
    plt.tight_layout()
    return fig
