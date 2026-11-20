"""
Utility functions for the AutoGluon Tabular tutorial notebooks.

- `AutoGluon.API.ipynb` uses the synthetic data generator and the plotting
  helpers
- `AutoGluon.example.ipynb` uses the attrition data loader, the baseline, the
  decision-threshold helpers, and the SHAP wrapper
- The notebooks keep the AutoGluon calls visible and call these functions for
  data generation, metrics, and plots

Import as:

import tutorials.AutoGluon.AutoGluon_utils as tauauuti
"""

import hashlib
import logging
import os
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn.compose
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


def silence_noisy_loggers() -> None:
    """
    Hide log messages that do not help the reader of the notebooks.

    - The image does not install `torch`, so AutoGluon logs "Failed to import
      torch" each time it counts the GPUs, even if no model needs a GPU
    - SHAP's `KernelExplainer` logs its internal weights at the `INFO` level
    """
    logging.getLogger("autogluon.common.utils.resource_utils").setLevel(
        logging.CRITICAL
    )
    logging.getLogger("shap").setLevel(logging.WARNING)


# #############################################################################
# Synthetic customer data
# #############################################################################


def make_synthetic_customers(n_rows: int, *, seed: int = 42) -> pd.DataFrame:
    """
    Generate a customer table whose label depends on known features.

    The data has the mix of column types found in real tables, so that the
    AutoGluon feature pipeline has something to do:
    - Integer and float columns, with 5% missing values in `monthly_charges`
    - String categorical columns (`contract`, `region`)
    - A datetime column (`signup_date`)
    - A constant column (`country`) that carries no information

    There are three possible labels, one per problem type:
    - `churned` ("yes" / "no"): binary classification
      - It depends on `contract`, `support_calls`, `tenure_months`, and
        `monthly_charges`
      - It does not depend on `region`, `signup_date`, or `country`
    - `contract`: multiclass classification (3 classes)
    - `lifetime_value`: regression

    :param n_rows: number of rows to generate
    :param seed: seed of the random generator
    :return: dataframe with one row per customer
    """
    hdbg.dassert_lt(0, n_rows)
    rng = np.random.default_rng(seed)
    # Sample the features.
    contract = rng.choice(
        ["month-to-month", "one-year", "two-year"],
        size=n_rows,
        p=[0.5, 0.3, 0.2],
    )
    tenure_months = rng.integers(1, 73, size=n_rows)
    monthly_charges = np.round(rng.normal(70, 20, size=n_rows).clip(20, 150), 2)
    support_calls = rng.poisson(2, size=n_rows)
    region = rng.choice(["north", "south", "east", "west"], size=n_rows)
    signup_date = pd.Timestamp("2020-01-01") + pd.to_timedelta(
        rng.integers(0, 4 * 365, size=n_rows), unit="D"
    )
    # Build the churn probability from a known logistic model.
    contract_effect = pd.Series(contract).map(
        {"month-to-month": 1.2, "one-year": -0.3, "two-year": -1.5}
    )
    logit = (
        -1.0
        + contract_effect.to_numpy()
        + 0.6 * (support_calls - 2)
        - 0.04 * (tenure_months - 36)
        + 0.02 * (monthly_charges - 70)
    )
    churn_proba = 1.0 / (1.0 + np.exp(-logit))
    churned = np.where(rng.random(n_rows) < churn_proba, "yes", "no")
    # Build a regression target from tenure and charges.
    lifetime_value = np.round(
        tenure_months * monthly_charges * rng.normal(1.0, 0.1, size=n_rows), 2
    )
    df = pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "contract": contract,
            "support_calls": support_calls,
            "region": region,
            "signup_date": signup_date,
            "country": "US",
            "churned": churned,
            "lifetime_value": lifetime_value,
        }
    )
    # Remove 5% of the charges to show the handling of missing values.
    missing_mask = rng.random(n_rows) < 0.05
    df.loc[missing_mask, "monthly_charges"] = np.nan
    return df


def split_train_test(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    stratify_col: Optional[str] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe in a train and a test set.

    :param df: data to split
    :param test_size: fraction of the rows in the test set
    :param stratify_col: column to stratify on, e.g., the label of a
        classification problem
    :param seed: seed of the random split
    :return: train and test dataframes, with the index reset
    """
    hdbg.dassert_lt(0.0, test_size)
    hdbg.dassert_lt(test_size, 1.0)
    stratify = None
    if stratify_col is not None:
        hdbg.dassert_in(stratify_col, df.columns)
        stratify = df[stratify_col]
    train_df, test_df = sklearn.model_selection.train_test_split(
        df, test_size=test_size, random_state=seed, stratify=stratify
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df


# #############################################################################
# Plots of the AutoGluon outputs
# #############################################################################


def plot_leaderboard(
    leaderboard_df: pd.DataFrame,
    *,
    score_col: str = "score_test",
    title: str = "Leaderboard",
) -> None:
    """
    Plot the score of each model of a leaderboard as horizontal bars.

    :param leaderboard_df: output of `TabularPredictor.leaderboard()`
    :param score_col: column with the score to plot, e.g., `score_test` or
        `score_val`
    :param title: title of the plot
    """
    hdbg.dassert_in("model", leaderboard_df.columns)
    hdbg.dassert_in(score_col, leaderboard_df.columns)
    df = leaderboard_df.sort_values(score_col)
    # Highlight the ensembles, since they combine the other models.
    colors = [
        "darkorange" if "WeightedEnsemble" in model else "steelblue"
        for model in df["model"]
    ]
    _, ax = plt.subplots(figsize=(8, 0.35 * len(df) + 1))
    ax.barh(df["model"], df[score_col], color=colors)
    # Zoom on the range of the scores, so that small differences are visible.
    score_min = df[score_col].min()
    score_max = df[score_col].max()
    margin = max(0.05 * abs(score_max - score_min), 1e-3)
    ax.set_xlim(score_min - margin, score_max + margin)
    ax.set_xlabel(score_col)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    *,
    top_n: int = 15,
    title: str = "Permutation feature importance",
) -> None:
    """
    Plot permutation feature importances with their standard deviation.

    :param importance_df: output of `TabularPredictor.feature_importance()`
    :param top_n: number of most important features to plot
    :param title: title of the plot
    """
    hdbg.dassert_in("importance", importance_df.columns)
    hdbg.dassert_in("stddev", importance_df.columns)
    df = importance_df.head(top_n).iloc[::-1]
    _, ax = plt.subplots(figsize=(8, 0.35 * len(df) + 1))
    ax.barh(df.index, df["importance"], xerr=df["stddev"], color="steelblue")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Drop of the eval metric when the feature is shuffled")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# #############################################################################
# Employee attrition data
# #############################################################################


# IBM HR Analytics Employee Attrition dataset, pinned to a commit of the IBM
# repo (Apache-2.0) so that the file cannot change under the notebook.
# - It is the same file as the Kaggle dataset in the project description, but
#   it does not need a Kaggle account
_ATTRITION_URL = (
    "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/"
    "608d844df52c10fdfdadc72031b5fd25dbc3bce2/data/emp_attrition.csv"
)
_ATTRITION_SHA256 = (
    "a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659"
)


def _compute_sha256(file_path: str) -> str:
    """
    Compute the SHA-256 digest of a file.

    :param file_path: path to the file
    :return: hex digest of the file content
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def download_attrition_data(dst_dir: str) -> str:
    """
    Download the IBM attrition CSV, unless it is already in `dst_dir`.

    :param dst_dir: directory where to save the file, e.g., `data`
    :return: path to the CSV file
    """
    os.makedirs(dst_dir, exist_ok=True)
    file_path = os.path.join(dst_dir, "emp_attrition.csv")
    if os.path.exists(file_path):
        _LOG.info("Using the cached file '%s'", file_path)
    else:
        _LOG.info("Downloading '%s' to '%s'", _ATTRITION_URL, file_path)
        urllib.request.urlretrieve(_ATTRITION_URL, file_path)
    # Check that the file is the expected one.
    digest = _compute_sha256(file_path)
    hdbg.dassert_eq(
        digest,
        _ATTRITION_SHA256,
        "Unexpected content in '%s': delete it and run again",
        file_path,
    )
    return file_path


def plot_rate_by_category(
    df: pd.DataFrame,
    col: str,
    label: str,
    *,
    positive_class: str,
) -> None:
    """
    Plot the rate of the positive class for each value of a column.

    :param df: data with the column and the label
    :param col: categorical column to group by
    :param label: name of the label column
    :param positive_class: value of the label counted as positive
    """
    hdbg.dassert_in(col, df.columns)
    hdbg.dassert_in(label, df.columns)
    rate = (
        (df[label] == positive_class)
        .groupby(df[col])
        .mean()
        .sort_values(ascending=False)
    )
    overall_rate = (df[label] == positive_class).mean()
    _, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(rate.index.astype(str), rate.to_numpy(), color="steelblue")
    ax.axhline(
        overall_rate, color="darkorange", linestyle="--", label="Overall rate"
    )
    ax.set_ylabel(f"Rate of {label}={positive_class}")
    ax.set_xlabel(col)
    ax.set_title(f"{label} rate by {col}")
    ax.legend()
    plt.tight_layout()
    plt.show()


# #############################################################################
# Logistic regression baseline
# #############################################################################


def train_logistic_baseline(
    train_df: pd.DataFrame, label: str
) -> sklearn.pipeline.Pipeline:
    """
    Train a logistic regression with one-hot encoding and scaling.

    This is the model a data scientist writes by hand in a few minutes, and the
    reference that AutoGluon has to beat.

    :param train_df: training data with the features and the label
    :param label: name of the label column
    :return: fitted scikit-learn pipeline
    """
    hdbg.dassert_in(label, train_df.columns)
    features_df = train_df.drop(columns=[label])
    categorical_cols = features_df.select_dtypes(include="object").columns
    numerical_cols = features_df.select_dtypes(exclude="object").columns
    preprocessor = sklearn.compose.ColumnTransformer(
        [
            (
                "categorical",
                sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore"),
                list(categorical_cols),
            ),
            (
                "numerical",
                sklearn.preprocessing.StandardScaler(),
                list(numerical_cols),
            ),
        ]
    )
    model = sklearn.pipeline.Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                sklearn.linear_model.LogisticRegression(max_iter=1000),
            ),
        ]
    )
    model.fit(features_df, train_df[label])
    return model


# #############################################################################
# Metrics and decision thresholds
# #############################################################################


def compute_binary_metrics(
    y_true: pd.Series,
    proba: np.ndarray,
    *,
    positive_class: str,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute the classification metrics of a probability at a threshold.

    :param y_true: true labels
    :param proba: predicted probability of the positive class
    :param positive_class: value of the label counted as positive
    :param threshold: a row is predicted positive when `proba >= threshold`
    :return: accuracy, precision, recall, F1, and ROC-AUC
    """
    hdbg.dassert_eq(len(y_true), len(proba))
    is_positive = (np.asarray(y_true) == positive_class).astype(int)
    is_pred_positive = (np.asarray(proba) >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "accuracy": sklearn.metrics.accuracy_score(
            is_positive, is_pred_positive
        ),
        "precision": sklearn.metrics.precision_score(
            is_positive, is_pred_positive, zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(is_positive, is_pred_positive),
        "f1": sklearn.metrics.f1_score(is_positive, is_pred_positive),
        "roc_auc": sklearn.metrics.roc_auc_score(is_positive, proba),
    }
    return metrics


def bootstrap_roc_auc(
    y_true: pd.Series,
    proba: np.ndarray,
    *,
    positive_class: str,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Estimate a 95% confidence interval of the ROC-AUC with the bootstrap.

    A small test set gives a noisy score: the interval shows how much the score
    could change with another test set of the same size.

    :param y_true: true labels
    :param proba: predicted probability of the positive class
    :param positive_class: value of the label counted as positive
    :param n_resamples: number of bootstrap resamples
    :param seed: seed of the random generator
    :return: ROC-AUC on the full test set and the 2.5% and 97.5% quantiles
    """
    is_positive = (np.asarray(y_true) == positive_class).astype(int)
    proba = np.asarray(proba)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(is_positive), size=len(is_positive))
        # Skip the rare resamples with a single class, where ROC-AUC is undefined.
        if is_positive[idx].min() == is_positive[idx].max():
            continue
        scores.append(
            sklearn.metrics.roc_auc_score(is_positive[idx], proba[idx])
        )
    result = {
        "roc_auc": sklearn.metrics.roc_auc_score(is_positive, proba),
        "ci_low": float(np.quantile(scores, 0.025)),
        "ci_high": float(np.quantile(scores, 0.975)),
    }
    return result


def compute_expected_cost(
    y_true: pd.Series,
    proba: np.ndarray,
    *,
    positive_class: str,
    threshold: float,
    fn_cost: float,
    fp_cost: float,
) -> float:
    """
    Compute the average cost per row of the decisions at a threshold.

    :param y_true: true labels
    :param proba: predicted probability of the positive class
    :param positive_class: value of the label counted as positive
    :param threshold: a row is predicted positive when `proba >= threshold`
    :param fn_cost: cost of a false negative, e.g., an employee who leaves
        without being flagged
    :param fp_cost: cost of a false positive, e.g., a retention action for an
        employee who would have stayed
    :return: average cost per row
    """
    is_positive = np.asarray(y_true) == positive_class
    is_pred_positive = np.asarray(proba) >= threshold
    n_fn = np.sum(is_positive & ~is_pred_positive)
    n_fp = np.sum(~is_positive & is_pred_positive)
    cost = (fn_cost * n_fn + fp_cost * n_fp) / len(is_positive)
    return float(cost)


def compute_threshold_curve(
    y_true: pd.Series,
    proba: np.ndarray,
    *,
    positive_class: str,
    fn_cost: float,
    fp_cost: float,
    n_thresholds: int = 99,
) -> pd.DataFrame:
    """
    Compute precision, recall, F1, and cost on a grid of thresholds.

    :param y_true: true labels
    :param proba: predicted probability of the positive class
    :param positive_class: value of the label counted as positive
    :param fn_cost: cost of a false negative
    :param fp_cost: cost of a false positive
    :param n_thresholds: number of thresholds in (0, 1)
    :return: one row per threshold
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    rows = []
    for threshold in thresholds:
        metrics = compute_binary_metrics(
            y_true, proba, positive_class=positive_class, threshold=threshold
        )
        metrics["cost"] = compute_expected_cost(
            y_true,
            proba,
            positive_class=positive_class,
            threshold=threshold,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        rows.append(metrics)
    curve_df = pd.DataFrame(rows)
    return curve_df


def plot_threshold_curve(
    curve_df: pd.DataFrame, *, marked_thresholds: Dict[str, float]
) -> None:
    """
    Plot precision, recall, F1, and cost as a function of the threshold.

    :param curve_df: output of `compute_threshold_curve()`
    :param marked_thresholds: thresholds to mark with a vertical line, by name
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))
    # Plot the classification metrics.
    for col in ["precision", "recall", "f1"]:
        ax1.plot(curve_df["threshold"], curve_df[col], label=col)
    ax1.set_xlabel("Decision threshold")
    ax1.set_title("Classification metrics")
    # Plot the cost.
    ax2.plot(curve_df["threshold"], curve_df["cost"], color="black")
    ax2.set_xlabel("Decision threshold")
    ax2.set_title("Average cost per employee")
    # Mark the thresholds on both plots.
    colors = sns.color_palette("Set2", len(marked_thresholds))
    for (name, threshold), color in zip(marked_thresholds.items(), colors):
        for ax in (ax1, ax2):
            ax.axvline(threshold, color=color, linestyle="--", label=name)
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(
    y_true: pd.Series,
    proba: np.ndarray,
    *,
    positive_class: str,
    thresholds: Dict[str, float],
) -> None:
    """
    Plot one confusion matrix per decision threshold.

    :param y_true: true labels
    :param proba: predicted probability of the positive class
    :param positive_class: value of the label counted as positive
    :param thresholds: thresholds to compare, by name
    """
    is_positive = (np.asarray(y_true) == positive_class).astype(int)
    _, axes = plt.subplots(
        1, len(thresholds), figsize=(4 * len(thresholds), 3.2)
    )
    axes = np.atleast_1d(axes)
    for ax, (name, threshold) in zip(axes, thresholds.items()):
        is_pred_positive = (np.asarray(proba) >= threshold).astype(int)
        cm = sklearn.metrics.confusion_matrix(is_positive, is_pred_positive)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["stay", "leave"],
            yticklabels=["stay", "leave"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} (threshold={threshold:.2f})")
    plt.tight_layout()
    plt.show()


# #############################################################################
# SHAP explanations
# #############################################################################


def _make_proba_fn(
    predictor: object, feature_names: List[str], positive_class: str
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrap a `TabularPredictor` into a function from arrays to probabilities.

    SHAP's `KernelExplainer` perturbs rows as numpy arrays, while AutoGluon
    expects a dataframe with the original column names.

    :param predictor: fitted `TabularPredictor`
    :param feature_names: names of the columns, in the order of the arrays
    :param positive_class: class whose probability is returned
    :return: function mapping an array of rows to the positive probability
    """

    def _predict_positive_proba(rows: np.ndarray) -> np.ndarray:
        rows_df = pd.DataFrame(rows, columns=feature_names)
        # Restore the dtypes that the perturbation turned into `object`.
        rows_df = rows_df.infer_objects()
        proba_df = predictor.predict_proba(rows_df)
        return proba_df[positive_class].to_numpy()

    return _predict_positive_proba


def explain_with_kernel_shap(
    predictor: object,
    background_df: pd.DataFrame,
    rows_df: pd.DataFrame,
    *,
    positive_class: str,
    n_samples: int = 300,
) -> pd.DataFrame:
    """
    Explain the positive probability of some rows with Kernel SHAP.

    Kernel SHAP is model-agnostic, so it explains the whole AutoGluon ensemble
    and not a single model inside it.

    :param predictor: fitted `TabularPredictor`
    :param background_df: rows that define the "average" prediction, without
        the label
    :param rows_df: rows to explain, with the same columns as `background_df`
    :param positive_class: class whose probability is explained
    :param n_samples: number of perturbed rows evaluated per explained row
    :return: one row per explained row, one column per feature, with the
        contribution of each feature to the probability
    """
    hdbg.dassert_eq(list(background_df.columns), list(rows_df.columns))
    feature_names = list(rows_df.columns)
    proba_fn = _make_proba_fn(predictor, feature_names, positive_class)
    explainer = shap.KernelExplainer(proba_fn, background_df.to_numpy())
    shap_values = explainer.shap_values(
        rows_df.to_numpy(), nsamples=n_samples, silent=True
    )
    shap_df = pd.DataFrame(
        shap_values, columns=feature_names, index=rows_df.index
    )
    # Store the baseline, so that `baseline + sum(contributions) = proba`.
    shap_df.attrs["expected_value"] = float(explainer.expected_value)
    return shap_df


def plot_shap_contributions(
    shap_row: pd.Series,
    feature_row: pd.Series,
    *,
    expected_value: float,
    top_n: int = 10,
    title: str = "",
) -> None:
    """
    Plot the largest SHAP contributions to the prediction of one row.

    :param shap_row: contribution of each feature to the prediction
    :param feature_row: value of each feature for the same row
    :param expected_value: average prediction on the background data
    :param top_n: number of contributions to plot
    :param title: title of the plot
    """
    top = shap_row.reindex(shap_row.abs().sort_values().index)[-top_n:]
    labels = [f"{name} = {feature_row[name]}" for name in top.index]
    colors = ["indianred" if value > 0 else "steelblue" for value in top]
    _, ax = plt.subplots(figsize=(8, 0.4 * len(top) + 1))
    ax.barh(labels, top.to_numpy(), color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    proba = expected_value + shap_row.sum()
    ax.set_xlabel(
        "Contribution to the predicted probability "
        f"(baseline={expected_value:.2f}, prediction={proba:.2f})"
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_mean_abs_shap(
    shap_df: pd.DataFrame,
    *,
    top_n: int = 15,
    title: str = "Mean absolute SHAP value",
) -> None:
    """
    Plot the average size of the SHAP contribution of each feature.

    :param shap_df: output of `explain_with_kernel_shap()` on many rows
    :param top_n: number of features to plot
    :param title: title of the plot
    """
    mean_abs = shap_df.abs().mean().sort_values().iloc[-top_n:]
    _, ax = plt.subplots(figsize=(8, 0.35 * len(mean_abs) + 1))
    ax.barh(mean_abs.index, mean_abs.to_numpy(), color="steelblue")
    ax.set_xlabel("Average |contribution| to the predicted probability")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# #############################################################################
# Risk summaries
# #############################################################################


def summarize_risk_by(
    df: pd.DataFrame,
    proba: np.ndarray,
    col: str,
    label: str,
    *,
    positive_class: str,
) -> pd.DataFrame:
    """
    Compare predicted and actual attrition for each value of a column.

    :param df: data with the column and the label
    :param proba: predicted probability of the positive class, one per row
    :param col: column to group by, e.g., `Department`
    :param label: name of the label column
    :param positive_class: value of the label counted as positive
    :return: one row per value of `col`, sorted by predicted risk
    """
    hdbg.dassert_eq(len(df), len(proba))
    summary_df = (
        pd.DataFrame(
            {
                col: df[col].to_numpy(),
                "predicted_risk": np.asarray(proba),
                "actual_rate": (df[label] == positive_class).to_numpy(),
            }
        )
        .groupby(col)
        .agg(
            n_employees=("predicted_risk", "size"),
            predicted_risk=("predicted_risk", "mean"),
            actual_rate=("actual_rate", "mean"),
        )
        .sort_values("predicted_risk", ascending=False)
        .round(3)
    )
    return summary_df
