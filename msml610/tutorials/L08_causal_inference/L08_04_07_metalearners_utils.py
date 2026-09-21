"""
Utility functions for metalearners tutorial (L08_04_07).

Import as:

import msml610.tutorials.L08_causal_inference.L08_04_07_metalearners_utils as mtlcil00mu
"""

import logging
from typing import List, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm
import sklearn.linear_model

import fklearn.causal.validation.curves
import fklearn.causal.validation.auc

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(
        notebook_log, utils_log=_LOG, set_all_loggers_to_print=True
    )
    # Silence the verbose `lightgbm` logger.
    logging.getLogger("lightgbm").setLevel(logging.ERROR)


warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


# Plot styling constants.
MARKER = ["o", "s"]  # Circle for T=0, Square for T=1.
COLOR = ["#FF6B6B", "#4ECDC4"]  # Red for T=1, Teal for T=0.


# #############################################################################
# Cell 2: T-Learner with Synthetic Data
# #############################################################################


def _g_kernel(x: np.ndarray, *, c: float = 0, s: float = 0.05) -> np.ndarray:
    """
    Gaussian kernel function.

    Computes a Gaussian kernel centered at c with scale s.

    :param x: Input values.
    :param c: Center of the kernel.
    :param s: Scale parameter (smaller s = sharper kernel).
    :return: Kernel values.
    """
    return np.exp((-((x - c) ** 2)) / s)


def generate_synthetic_treatment_data(
    n0: int,
    n1: int,
    *,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Generate synthetic treatment/control data.

    Creates data with:
    - Control group (T=0): n0 samples from a Gaussian kernel-based model.
    - Treated group (T=1): n1 samples with a shifted mean (+1).

    :param n0: Number of control samples.
    :param n1: Number of treated samples.
    :param seed: Random seed for reproducibility.
    :return: DataFrame with columns 'x', 'y', 't'.

    Example:
        ```
               x         y  t
        0 -0.977578  0.271234  0
        1 -0.968858  0.289456  0
        2 -0.965465  0.303221  0
        3 -0.960224  0.315678  0
        4 -0.958736  0.321890  0
        ```
    """
    np.random.seed(seed)
    # Control group follows a Gaussian kernel-based model.
    x0 = np.random.uniform(-1, 1, n0)
    y0 = np.random.normal(0.3 * _g_kernel(x0), 0.1, n0)
    # Treated group has the same functional form but shifted up by 1.
    x1 = np.random.uniform(-1, 1, n1)
    y1 = np.random.normal(0.3 * _g_kernel(x1), 0.1, n1) + 1
    # Combine groups into single dataframe.
    df = pd.concat(
        [
            pd.DataFrame(dict(x=x0, y=y0, t=0)),
            pd.DataFrame(dict(x=x1, y=y1, t=1)),
        ]
    )
    # Sort for consistent visualization and analysis.
    df = df.sort_values(by="x")
    return df


def fit_tlearner_models(
    df: pd.DataFrame,
    min_child_samples: int = 25,
) -> Tuple[
    lightgbm.LGBMRegressor, lightgbm.LGBMRegressor, np.ndarray, np.ndarray
]:
    """
    Fit outcome models for each treatment group.

    Train separate `lightgbm.LGBMRegressor` models for control (T=0) and
    treated (T=1) groups to estimate conditional outcome expectations.

    :param df: DataFrame with columns 'x', 'y', 't'.
    :param min_child_samples: LightGBM min_child_samples parameter.
    :return: Tuple of (m0, m1, m0_predictions, m1_predictions) where
             m0 and m1 are fitted regressors and predictions are on the
             full dataset.
    """
    # Control group outcome model.
    x0 = np.asarray(df.query("t==0")["x"]).reshape(-1, 1)
    y0 = np.asarray(df.query("t==0")["y"])
    m0 = lightgbm.LGBMRegressor(
        min_child_samples=min_child_samples, verbosity=-1
    )
    m0.fit(x0, y0)
    # Treated group outcome model.
    x1 = np.asarray(df.query("t==1")["x"]).reshape(-1, 1)
    y1 = np.asarray(df.query("t==1")["y"])
    m1 = lightgbm.LGBMRegressor(
        min_child_samples=min_child_samples, verbosity=-1
    )
    m1.fit(x1, y1)
    # Generate predictions across the full dataset.
    X_full = np.asarray(df[["x"]])
    m0_hat = m0.predict(X_full)
    m1_hat = m1.predict(X_full)
    return m0, m1, m0_hat, m1_hat


def plot_tlearner_treatment_effect_analysis(
    df: pd.DataFrame,
    m0: lightgbm.LGBMRegressor,
    m1: lightgbm.LGBMRegressor,
    m0_hat: np.ndarray,
    m1_hat: np.ndarray,
) -> None:
    """
    Plot outcome models and treatment effect heterogeneity.

    Visualizes:
    - Top subplot: Scatter plots of control/treated outcomes and fitted models.
    - Bottom subplot: Estimated heterogeneous treatment effects.

    :param df: DataFrame with columns 'x', 'y', 't'.
    :param m0: Fitted outcome model for control group.
    :param m1: Fitted outcome model for treated group.
    :param m0_hat: Predictions from m0 on full dataset.
    :param m1_hat: Predictions from m1 on full dataset.
    """
    _ = plt.subplots(2, 1, figsize=(8, 6))
    ax1, ax2 = plt.gcf().axes[:2]
    # Extract and sort x values for consistent plotting across both subplots.
    x_vals = np.asarray(df["x"])
    x_sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[x_sort_idx]
    # Top subplot: Scatter plots of control (T=0) and treated (T=1) outcomes.
    x0 = np.asarray(df.query("t==0")["x"])
    y0 = np.asarray(df.query("t==0")["y"])
    ax1.scatter(
        x0,
        y0,
        alpha=0.5,
        label="T=0",
        marker=MARKER[0],
        color=COLOR[1],
    )
    x1 = np.asarray(df.query("t==1")["x"])
    y1 = np.asarray(df.query("t==1")["y"])
    ax1.scatter(
        x1,
        y1,
        alpha=0.7,
        label="T=1",
        marker=MARKER[1],
        color=COLOR[0],
    )
    # Overlay fitted outcome models on the scatter plots.
    m0_hat_sorted = m0_hat[x_sort_idx]
    m1_hat_sorted = m1_hat[x_sort_idx]
    ax1.plot(
        x_sorted,
        m0_hat_sorted,
        color="black",
        linestyle="solid",
        label=r"$\hat{\mu}_0$",
    )
    ax1.plot(
        x_sorted,
        m1_hat_sorted,
        color="black",
        linestyle="--",
        label=r"$\hat{\mu}_1$",
    )
    ax1.set_ylabel("Y", fontsize=12)
    ax1.set_xlabel("X", fontsize=12)
    ax1.legend(fontsize=14)
    # Bottom subplot: Heterogeneous treatment effects for each group.
    # For control units: estimated effect = treated prediction - actual outcome.
    x0_full = np.asarray(df.query("t==0")[["x"]])
    y0_full = np.asarray(df.query("t==0")["y"])
    tau_0 = m1.predict(x0_full) - y0_full
    ax2.scatter(
        x0,
        tau_0,
        label=r"$\hat{\tau}_0$",
        alpha=0.5,
        marker=MARKER[0],
        color=COLOR[1],
    )
    # For treated units: estimated effect = actual outcome - control prediction.
    x1_full = np.asarray(df.query("t==1")[["x"]])
    y1_full = np.asarray(df.query("t==1")["y"])
    tau_1 = y1_full - m0.predict(x1_full)
    ax2.scatter(
        x1,
        tau_1,
        label=r"$\hat{\tau}_1$",
        alpha=0.7,
        marker=MARKER[1],
        color=COLOR[0],
    )
    # Plot CATE: difference in predicted outcomes between treatment groups.
    X_full = np.asarray(df[["x"]])
    cate = m1.predict(X_full) - m0.predict(X_full)
    ax2.plot(
        x_sorted,
        cate[x_sort_idx],
        label=r"$\hat{CATE}$",
        color="black",
    )
    ax2.set_ylabel("Estimated Effect", fontsize=12)
    ax2.set_xlabel("X", fontsize=12)
    ax2.legend(fontsize=14)
    plt.tight_layout()


# #############################################################################
# Cell 3: X-Learner
# #############################################################################


def calculate_xlearner_heterogeneous_treatment_effects(
    df: pd.DataFrame,
    m0: lightgbm.LGBMRegressor,
    m1: lightgbm.LGBMRegressor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate heterogeneous treatment effects for X-Learner.

    Computes the estimated treatment effect for control and treated units:
    - For control units: tau_0 = m1_prediction - actual_outcome
    - For treated units: tau_1 = actual_outcome - m0_prediction

    :param df: DataFrame with columns 'x', 'y', 't'.
    :param m0: Fitted outcome model for control group.
    :param m1: Fitted outcome model for treated group.
    :return: Tuple of (tau_0, tau_1) arrays.
    """
    x0_full = np.asarray(df.query("t==0")[["x"]])
    y0_full = np.asarray(df.query("t==0")["y"])
    tau_0 = m1.predict(x0_full) - y0_full
    #
    x1_full = np.asarray(df.query("t==1")[["x"]])
    y1_full = np.asarray(df.query("t==1")["y"])
    tau_1 = y1_full - m0.predict(x1_full)
    return tau_0, tau_1


def fit_xlearner_models(
    df: pd.DataFrame,
    tau_0: np.ndarray,
    tau_1: np.ndarray,
    min_child_samples: int = 25,
) -> Tuple[
    lightgbm.LGBMRegressor, lightgbm.LGBMRegressor, np.ndarray, np.ndarray
]:
    """
    Fit X-Learner models on heterogeneous treatment effects.

    Trains separate models to predict the heterogeneous treatment effects
    for each treatment group.

    :param df: DataFrame with columns 'x', 't'.
    :param tau_0: Heterogeneous treatment effects for control units.
    :param tau_1: Heterogeneous treatment effects for treated units.
    :param min_child_samples: LightGBM min_child_samples parameter.
    :return: Tuple of (mu_tau0, mu_tau1, mu_tau0_hat, mu_tau1_hat).
    """
    # Control group effect model.
    x0 = np.asarray(df.query("t==0")[["x"]])
    mu_tau0 = lightgbm.LGBMRegressor(
        min_child_samples=min_child_samples, verbosity=-1
    )
    mu_tau0.fit(x0, tau_0)
    mu_tau0_hat = mu_tau0.predict(x0)
    # Treated group effect model.
    x1 = np.asarray(df.query("t==1")[["x"]])
    mu_tau1 = lightgbm.LGBMRegressor(
        min_child_samples=min_child_samples, verbosity=-1
    )
    mu_tau1.fit(x1, tau_1)
    mu_tau1_hat = mu_tau1.predict(x1)
    return mu_tau0, mu_tau1, mu_tau0_hat, mu_tau1_hat


def plot_xlearner_effect_estimates(
    df: pd.DataFrame,
    tau_0: np.ndarray,
    tau_1: np.ndarray,
    mu_tau0_hat: np.ndarray,
    mu_tau1_hat: np.ndarray,
) -> None:
    """
    Plot X-Learner heterogeneous treatment effect estimates.

    Visualizes the estimated heterogeneous effects and fitted models.

    :param df: DataFrame with columns 'x', 't'.
    :param tau_0: Heterogeneous effects for control units.
    :param tau_1: Heterogeneous effects for treated units.
    :param mu_tau0_hat: Fitted effect predictions for control group.
    :param mu_tau1_hat: Fitted effect predictions for treated group.
    """
    plt.figure(figsize=(8, 3))
    # Control group effects and fitted model.
    x0 = np.asarray(df.query("t==0")[["x"]])
    plt.scatter(
        x0,
        tau_0,
        label=r"$\hat{\tau}_0$",
        alpha=0.5,
        marker=MARKER[0],
        color=COLOR[1],
    )
    plt.plot(
        x0,
        mu_tau0_hat,
        color="black",
        linestyle="solid",
        label=r"$\hat{\mu}_{\tau_0}$",
    )
    # Treated group effects and fitted model.
    x1 = np.asarray(df.query("t==1")[["x"]])
    plt.scatter(
        x1,
        tau_1,
        label=r"$\hat{\tau}_1$",
        alpha=0.8,
        marker=MARKER[1],
        color=COLOR[0],
    )
    plt.plot(
        x1,
        mu_tau1_hat,
        color="black",
        linestyle="dashed",
        label=r"$\hat{\mu}_{\tau_1}$",
    )
    plt.ylabel("Estimated Effect")
    plt.xlabel("X")
    plt.legend(fontsize=14)


def plot_xlearner_with_propensity_scores(
    df: pd.DataFrame,
    mu_tau0: lightgbm.LGBMRegressor,
    mu_tau1: lightgbm.LGBMRegressor,
    tau_0: np.ndarray,
    tau_1: np.ndarray,
) -> None:
    """
    Plot X-Learner CATE with propensity score weighting.

    Visualizes the conditional average treatment effect (CATE) computed
    as a propensity-score-weighted average of the treatment effect
    estimates.

    :param df: DataFrame with columns 'x', 't'.
    :param mu_tau0: X-Learner model for control group effects.
    :param mu_tau1: X-Learner model for treated group effects.
    :param tau_0: Heterogeneous effects for control units.
    :param tau_1: Heterogeneous effects for treated units.
    """
    plt.figure(figsize=(8, 3))
    # Fit propensity score model.
    ps_model = sklearn.linear_model.LogisticRegression(penalty=None)
    ps_model.fit(df[["x"]], df["t"])
    ps = ps_model.predict_proba(df[["x"]])[:, 1]
    # Control group effects with propensity-based weighting.
    x0 = np.asarray(df.query("t==0")[["x"]])
    ps_0 = ps[df["t"] == 0]
    s_0 = 100 * ps_0
    plt.scatter(
        x0,
        tau_0,
        label=r"$\hat{\tau}_0$",
        alpha=0.5,
        s=s_0,
        marker=MARKER[0],
        color=COLOR[1],
    )
    # Treated group effects with propensity-based weighting.
    x1 = np.asarray(df.query("t==1")[["x"]])
    ps_1 = ps[df["t"] == 1]
    s_1 = 100 * (1 - ps_1)
    plt.scatter(
        x1,
        tau_1,
        label=r"$\hat{\tau}_1$",
        alpha=0.5,
        s=s_1,
        marker=MARKER[1],
        color=COLOR[0],
    )
    # Compute and plot CATE as propensity-score-weighted average.
    X_full = np.asarray(df[["x"]])
    # Weight treatment effect estimates by propensity scores:
    # - untreated units weighted by PS (towards tau1 estimates)
    # - treated units weighted by (1-PS) (towards tau0 estimates)
    # to balance estimates toward the target.
    cate = (1 - ps) * mu_tau1.predict(X_full) + ps * mu_tau0.predict(X_full)
    plt.plot(df[["x"]], cate, label="x-learner", color="black")
    plt.ylabel("Estimated Effect")
    plt.xlabel("X")
    plt.legend(fontsize=14)


# #############################################################################
# Cell 4: X-Learner with Real Data and Propensity Score Weighting
# #############################################################################


def fit_propensity_score_and_weighted_outcome_models(
    train: pd.DataFrame,
    X: List[str],
    T: str,
    y: str,
    *,
    seed: int = 123,
) -> Tuple[
    sklearn.linear_model.LogisticRegression,
    lightgbm.LGBMRegressor,
    lightgbm.LGBMRegressor,
]:
    """
    Fit propensity score and weighted first-stage outcome models for X-Learner.

    Fits a propensity score model to estimate treatment probability, then fits
    separate outcome models for control and treated groups using inverse
    probability weighting.

    :param train: Training DataFrame with features X, treatment T, and outcome y.
    :param X: List of feature column names.
    :param T: Treatment column name.
    :param y: Outcome column name.
    :param seed: Random seed for reproducibility.
    :return: Tuple of (ps_model, m0, m1) where ps_model is the fitted propensity
             score model and m0, m1 are the weighted outcome models.
    """
    np.random.seed(seed)
    ps_model = sklearn.linear_model.LogisticRegression(penalty=None)
    ps_model.fit(train[X], train[T])
    # Outcome models for control group.
    m0 = lightgbm.LGBMRegressor()
    train_t0 = train.query(f"{T}==0")
    w_t0 = 1 / ps_model.predict_proba(train_t0[X])[:, 0]
    m0.fit(
        train_t0[X].values,
        train_t0[y].values,
        sample_weight=w_t0,
    )
    # Outcome models for treated group.
    m1 = lightgbm.LGBMRegressor()
    train_t1 = train.query(f"{T}==1")
    w_t1 = 1 / ps_model.predict_proba(train_t1[X])[:, 1]
    m1.fit(
        train_t1[X].values,
        train_t1[y].values,
        sample_weight=w_t1,
    )
    return ps_model, m0, m1


def fit_xlearner_second_stage_models(
    train: pd.DataFrame,
    X: List[str],
    T: str,
    y: str,
    m0: lightgbm.LGBMRegressor,
    m1: lightgbm.LGBMRegressor,
    *,
    seed: int = 123,
) -> Tuple[lightgbm.LGBMRegressor, lightgbm.LGBMRegressor]:
    """
    Fit second-stage X-Learner models on residual treatment effects.

    Computes residual treatment effects (tau_hat) for each group and fits
    separate models to predict these effects.

    :param train: Training DataFrame with features X, treatment T, and outcome y.
    :param X: List of feature column names.
    :param T: Treatment column name.
    :param y: Outcome column name.
    :param m0: Fitted outcome model for control group.
    :param m1: Fitted outcome model for treated group.
    :param seed: Random seed for reproducibility.
    :return: Tuple of (m_tau_0, m_tau_1) models for predicting treatment effects.
    """
    np.random.seed(seed)
    # Second-stage model for control group effects.
    m_tau_0 = lightgbm.LGBMRegressor()
    train_t0 = train.query(f"{T}==0")
    tau_hat_0 = m1.predict(train_t0[X].values) - train_t0[y].values
    m_tau_0.fit(train_t0[X].values, tau_hat_0)
    # Second-stage model for treated group effects.
    m_tau_1 = lightgbm.LGBMRegressor()
    train_t1 = train.query(f"{T}==1")
    tau_hat_1 = train_t1[y].values - m0.predict(train_t1[X].values)
    m_tau_1.fit(train_t1[X].values, tau_hat_1)
    return m_tau_0, m_tau_1


def estimate_xlearner_cate(
    test: pd.DataFrame,
    X: List[str],
    ps_model: sklearn.linear_model.LogisticRegression,
    m_tau_0: lightgbm.LGBMRegressor,
    m_tau_1: lightgbm.LGBMRegressor,
) -> pd.DataFrame:
    """
    Estimate CATE using propensity-score-weighted X-Learner effects.

    Combines the second-stage treatment effect models with propensity score
    weighting to produce final CATE estimates.

    :param test: Test DataFrame with features X.
    :param X: List of feature column names.
    :param ps_model: Fitted propensity score model.
    :param m_tau_0: Fitted effect model for control group.
    :param m_tau_1: Fitted effect model for treated group.
    :return: DataFrame with CATE predictions in 'cate' column.
    """
    # Estimate propensity scores for the test set.
    ps_test = ps_model.predict_proba(test[X])[:, 1]
    # Compute treatment effect predictions for each learner.
    tau_0_pred = m_tau_0.predict(test[X].values)
    tau_1_pred = m_tau_1.predict(test[X].values)
    # Compute CATE as propensity-weighted average of control and treatment effects.
    cate = ps_test * tau_0_pred + (1 - ps_test) * tau_1_pred
    # Add CATE estimates to test set.
    return test.assign(cate=cate)


def plot_gain_curve_analysis(
    cate_test: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    title: str = "Model",
) -> float:
    """
    Plot and calculate gain curve analysis for CATE evaluation.

    Computes the relative cumulative gain curve and AUC metric, displaying
    the results in a plot.

    :param cate_test: DataFrame with treatment, outcome, and CATE predictions.
    :param treatment_col: Name of the treatment column.
    :param outcome_col: Name of the outcome column.
    :param title: Title for the plot.
    :return: AUC value for the gain curve.
    """
    gain_curve_test = (
        fklearn.causal.validation.curves.relative_cumulative_gain_curve(
            cate_test, treatment_col, outcome_col, prediction="cate"
        )
    )
    auc = fklearn.causal.validation.auc.area_under_the_relative_cumulative_gain_curve(
        cate_test, treatment_col, outcome_col, prediction="cate"
    )
    # Plot gain curve and AUC baseline.
    plt.figure(figsize=(8, 3))
    plt.plot(gain_curve_test, color="C0", label=f"AUC: {auc:.2f}")
    plt.hlines(0, 0, 100, linestyle="--", color="black", label="Baseline")
    plt.legend()
    plt.title(title)
    return auc


# #############################################################################
# Cell 5: S-Learner
# #############################################################################


def fit_slearner_model(
    train: pd.DataFrame,
    X: List[str],
    T: str,
    y: str,
    *,
    seed: int = 123,
) -> lightgbm.LGBMRegressor:
    """
    Fit an S-Learner model on the training data.

    Trains a single regressor on the combined feature set (X) and treatment (T)
    to predict the outcome (y).

    :param train: Training DataFrame with features X, treatment T, and outcome y.
    :param X: List of feature column names.
    :param T: Treatment column name.
    :param y: Outcome column name.
    :param seed: Random seed for reproducibility.
    :return: Fitted lightgbm.LGBMRegressor model.
    """
    hdbg.dassert_in(T, train.columns)
    hdbg.dassert_in(y, train.columns)
    for col in X:
        hdbg.dassert_in(col, train.columns)
    np.random.seed(seed)
    s_learner = lightgbm.LGBMRegressor()
    s_learner.fit(train[X + [T]], train[y])
    return s_learner


def generate_slearner_counterfactual_predictions(
    test: pd.DataFrame,
    X: List[str],
    T: str,
    y: str,
    s_learner: lightgbm.LGBMRegressor,
    treatment_values: np.ndarray,
) -> pd.DataFrame:
    """
    Generate counterfactual predictions for S-Learner across treatment values.

    Creates a dataset where each row in test is expanded to have predictions
    under multiple treatment values.

    :param test: Test DataFrame with features X.
    :param X: List of feature column names.
    :param T: Treatment column name.
    :param y: Outcome column name.
    :param s_learner: Fitted S-Learner model.
    :param treatment_values: Array of treatment values to predict for.
    :return: DataFrame with counterfactual predictions appended.
    """
    # Create a grid of treatment values for counterfactual predictions.
    t_grid = pd.DataFrame(dict(key=1, **{T: treatment_values}))
    y_hat_col = f"{y}_hat"
    # Remove original treatment values to create a clean base for counterfactuals.
    test_cf = test.drop(columns=[T])
    # Add merge key to enable cross-product with treatment grid.
    test_cf = test_cf.assign(key=1)
    # Create counterfactual dataset by merging each test row with all treatment
    # values.
    test_cf = test_cf.merge(t_grid)
    # Generate predictions for each (features, treatment) combination.
    test_cf = test_cf.assign(
        **{y_hat_col: lambda d: s_learner.predict(d[X + [T]])}
    )
    return test_cf


def _calculate_linear_effect(df: pd.DataFrame, y: str, t: str) -> float:
    """
    Calculate linear treatment effect as a slope coefficient.

    Computes the linear regression slope: Cov(y, t) / Var(t), which
    represents the expected change in y per unit change in t.

    :param df: DataFrame with columns y and t.
    :param y: Outcome column name.
    :param t: Treatment column name.
    :return: Linear effect coefficient (regression slope).
    """
    return np.cov(df[y], df[t])[0, 1] / df[t].var()


def estimate_slearner_cate(
    test_cf: pd.DataFrame,
    test: pd.DataFrame,
    T: str,
    y: str,
) -> pd.DataFrame:
    """
    Estimate CATE for S-Learner using linear treatment effects.

    Groups the counterfactual predictions by grouping keys and computes
    the linear treatment effect within each group as the slope of predicted
    outcomes against treatment T.

    :param test_cf: Counterfactual DataFrame with predictions and group keys.
    :param test: Original test DataFrame.
    :param T: Treatment column name.
    :param y: Outcome column name (used to compute y_hat column name).
    :return: DataFrame with CATE predictions.
    """
    y_hat_col = f"{y}_hat"
    # Identify columns to group by (exclude treatment, prediction, and merge key).
    groupby_cols = list(test_cf.columns.difference([T, y_hat_col, "key"]))
    # Group counterfactual predictions by feature combinations.
    grouped = test_cf.groupby(groupby_cols)
    # Compute linear treatment effect (slope) within each group.
    cate_values = grouped.apply(
        lambda df: _calculate_linear_effect(df, y_hat_col, T),
        include_groups=False,
    )
    # Set series name to indicate CATE column.
    cate_values.name = "cate"
    cate = cate_values
    # Join CATE estimates back to the original test set.
    result = test.set_index(groupby_cols).join(cate).reset_index()
    return result


# #############################################################################
# Cell 6: Double ML / R-Learner
# #############################################################################


def fit_rlearner_models(
    train: pd.DataFrame,
    X: List[str],
    T: str,
    y: str,
    *,
    seed: int = 123,
) -> Tuple[lightgbm.LGBMRegressor, lightgbm.LGBMRegressor]:
    """
    Fit debiasing and denoising models for R-Learner.

    Trains two models for residualization:
    - Debiasing model: predicts treatment T from features X
    - Denoising model: predicts outcome y from features X

    :param train: Training DataFrame with features X, treatment T, and outcome y.
    :param X: List of feature column names.
    :param T: Treatment column name.
    :param y: Outcome column name.
    :param seed: Random seed for reproducibility.
    :return: Tuple of (debias_m, denoise_m) models.
    """
    np.random.seed(seed)
    debias_m = lightgbm.LGBMRegressor()
    denoise_m = lightgbm.LGBMRegressor()
    return debias_m, denoise_m


def _calculate_rlearner_residuals(
    train: pd.DataFrame,
    X: List[str],
    T: str,
    y: str,
    debias_m: lightgbm.LGBMRegressor,
    denoise_m: lightgbm.LGBMRegressor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate residuals for R-Learner using cross-validated predictions.

    Computes:
    - Treatment residuals: T - predicted_T
    - Outcome residuals: y - predicted_y

    :param train: Training DataFrame.
    :param X: List of feature column names.
    :param T: Treatment column name.
    :param y: Outcome column name.
    :param debias_m: Model for treatment prediction.
    :param denoise_m: Model for outcome prediction.
    :return: Tuple of (treatment_residuals, outcome_residuals).
    """
    from sklearn.model_selection import cross_val_predict

    t_res = train[T] - cross_val_predict(debias_m, train[X], train[T], cv=5)
    y_res = train[y] - cross_val_predict(denoise_m, train[X], train[y], cv=5)
    return t_res, y_res


def fit_rlearner_cate_model(
    train: pd.DataFrame,
    X: List[str],
    T: str,
    y: str,
    debias_m: lightgbm.LGBMRegressor,
    denoise_m: lightgbm.LGBMRegressor,
    *,
    seed: int = 123,
) -> Tuple[lightgbm.LGBMRegressor, np.ndarray, np.ndarray]:
    """
    Fit R-Learner CATE model using residuals.

    Computes treatment and outcome residuals, then fits a CATE model
    on residual-scaled outcomes weighted by treatment residuals.

    :param train: Training DataFrame with features X, treatment T, and outcome y.
    :param X: List of feature column names.
    :param T: Treatment column name.
    :param y: Outcome column name.
    :param debias_m: Model for treatment prediction.
    :param denoise_m: Model for outcome prediction.
    :param seed: Random seed for reproducibility.
    :return: Tuple of (cate_model, treatment_residuals, outcome_residuals).
    """
    np.random.seed(seed)
    # Calculate residuals.
    t_res, y_res = _calculate_rlearner_residuals(
        train, X, T, y, debias_m, denoise_m
    )
    # Compute weighted outcome and weights for CATE fitting.
    y_star = y_res / t_res
    w = t_res**2
    # Fit CATE model with weighted regression.
    cate_model = lightgbm.LGBMRegressor()
    cate_model.fit(train[X], y_star, sample_weight=w)
    return cate_model, t_res, y_res


def estimate_rlearner_cate(
    test: pd.DataFrame,
    X: List[str],
    cate_model: lightgbm.LGBMRegressor,
) -> pd.DataFrame:
    """
    Estimate CATE for R-Learner on test data.

    Generates CATE predictions from the fitted R-Learner model.

    :param test: Test DataFrame with features X.
    :param X: List of feature column names.
    :param cate_model: Fitted R-Learner CATE model.
    :return: DataFrame with CATE predictions in 'cate' column.
    """
    return test.assign(cate=cate_model.predict(test[X]))
