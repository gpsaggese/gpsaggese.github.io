"""
Utility functions for the propensity score tutorial (L08_04_05).

Import as:

import msml610.tutorials.L08_causal_inference.L08_04_05_propensity_score_utils as mtl0psu
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
import sklearn.linear_model
import sklearn.neighbors
import patsy

# Standard figure sizes
_FIGSIZE_SINGLE = (8, 3)
_FIGSIZE_PER_SUBPLOT = (4, 3)


# #############################################################################
# Cell 1: Engagement Score Regression Analysis.
# #############################################################################


def plot_engagement_vs_intervention(
    data: pd.DataFrame,
    *,
    figsize: Tuple[int, int] = _FIGSIZE_SINGLE,
    jitter_strength: float = 0.03,
) -> None:
    """
    Plot regression of engagement_score against intervention treatment.

    Visualizes the relationship between intervention (treatment) and
    engagement_score with a scatter plot and regression line. Horizontal
    jitter is applied to reveal overlapping points.

    :param data: DataFrame with columns 'intervention' and 'engagement_score'
    :param figsize: Figure size as (width, height)
    :param jitter_strength: Amount of horizontal jitter to add
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Add jitter to intervention values.
    jitter = np.random.normal(0, jitter_strength, size=len(data))
    x_jittered = np.array(data["intervention"]) + jitter
    # Scatter plot for control group (intervention=0).
    df_control = data[data["intervention"] == 0]
    control_indices = df_control.index
    ax.scatter(
        x=x_jittered[control_indices],
        y=df_control["engagement_score"],
        alpha=0.4,
        s=50,
        color="blue",
        label="Control",
    )
    # Scatter plot for treated group (intervention=1).
    df_treated = data[data["intervention"] == 1]
    treated_indices = df_treated.index
    ax.scatter(
        x=x_jittered[treated_indices],
        y=df_treated["engagement_score"],
        alpha=0.4,
        s=50,
        color="red",
        label="Treated",
    )
    # Fit and plot regression line.
    X = np.array(data["intervention"]).reshape(-1, 1)
    y = np.array(data["engagement_score"])
    model = sklearn.linear_model.LinearRegression().fit(X, y)
    x_range = np.array([0, 1]).reshape(-1, 1)
    y_pred = model.predict(x_range)
    intercept = model.intercept_
    coef = model.coef_[0]
    label = f"Regression: y={intercept:.3f}+{coef:.3f}*x"
    ax.plot(
        [0, 1],
        y_pred,
        color="black",
        linewidth=2,
        linestyle="--",
        label=label,
    )
    ax.set_xlabel("Intervention", fontsize=11)
    ax.set_ylabel("Engagement Score", fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Control", "Treated"], fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_engagement_density_by_intervention(
    data: pd.DataFrame,
    *,
    figsize: Tuple[int, int] = _FIGSIZE_SINGLE,
    jitter_strength: float = 0.02,
) -> None:
    """
    Plot density of engagement_score by intervention group with mean and variance.

    Shows the distribution of engagement_score separately for control and
    treated groups using kernel density estimation curves. Mean values are shown
    as solid lines and ±1 standard deviation bounds as dotted lines. Scatter
    points with jitter show the underlying data.

    :param data: DataFrame with columns 'intervention' and 'engagement_score'
    :param figsize: Figure size as (width, height)
    :param jitter_strength: Amount of jitter for scatter points
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Plot density for control group.
    df_control = data[data["intervention"] == 0]
    sns.kdeplot(
        data=df_control,
        x="engagement_score",
        ax=ax,
        color="blue",
        linewidth=2,
        label=f"Control (n={len(df_control)})",
    )
    # Add scatter with jitter for control.
    control_jitter = np.random.normal(0, jitter_strength, size=len(df_control))
    ax.scatter(
        x=df_control["engagement_score"],
        y=control_jitter * 0.02,
        alpha=0.2,
        s=30,
        color="blue",
    )
    # Add mean line for control.
    control_mean = float(df_control["engagement_score"].mean())
    ax.axvline(
        control_mean,
        color="blue",
        linestyle="-",
        linewidth=2,
        alpha=0.7,
    )
    # Plot density for treated group.
    df_treated = data[data["intervention"] == 1]
    sns.kdeplot(
        data=df_treated,
        x="engagement_score",
        ax=ax,
        color="red",
        linewidth=2,
        label=f"Treated (n={len(df_treated)})",
    )
    # Add scatter with jitter for treated.
    treated_jitter = np.random.normal(0, jitter_strength, size=len(df_treated))
    ax.scatter(
        x=df_treated["engagement_score"],
        y=treated_jitter * 0.02,
        alpha=0.2,
        s=30,
        color="red",
    )
    # Add mean line for treated.
    treated_mean = float(df_treated["engagement_score"].mean())
    ax.axvline(
        treated_mean,
        color="red",
        linestyle="-",
        linewidth=2,
        alpha=0.7,
    )
    ax.set_xlabel("Engagement Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_engagement_vs_intervention_by_department(
    data: pd.DataFrame,
    *,
    figsize_per_plot: Tuple[int, int] = _FIGSIZE_PER_SUBPLOT,
    max_departments: int = 6,
    jitter_strength: float = 0.03,
) -> None:
    """
    Plot regression of engagement_score vs intervention, stratified by department.

    Creates multiple subplots showing the relationship between intervention and
    engagement_score for each department, useful for understanding heterogeneous
    treatment effects across departments. Horizontal jitter is applied to reveal
    overlapping points.

    :param data: DataFrame with columns 'intervention', 'engagement_score',
                 'departament_id' (or 'department_id')
    :param figsize_per_plot: Size per subplot as (width, height)
    :param max_departments: Maximum number of departments to plot
    :param jitter_strength: Amount of horizontal jitter to add
    """
    # Handle both spelling variants.
    dept_col = (
        "departament_id" if "departament_id" in data.columns else "department_id"
    )
    departments = sorted(data[dept_col].unique())[:max_departments]
    n_depts = len(departments)
    # Calculate grid dimensions.
    n_cols = 3
    n_rows = (n_depts + n_cols - 1) // n_cols
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes.flatten() if n_depts > 1 else [axes]
    for idx, dept_id in enumerate(departments):
        ax = axes_flat[idx]
        dept_data = data[data[dept_col] == dept_id].reset_index(drop=True)
        # Add jitter to intervention values.
        jitter = np.random.normal(0, jitter_strength, size=len(dept_data))
        x_jittered = np.array(dept_data["intervention"]) + jitter
        # Plot control group.
        df_control = dept_data[dept_data["intervention"] == 0]
        control_pos = df_control.index.tolist()
        ax.scatter(
            x=x_jittered[control_pos],
            y=df_control["engagement_score"].values,
            alpha=0.4,
            s=40,
            color="blue",
            label="Control",
        )
        # Plot treated group.
        df_treated = dept_data[dept_data["intervention"] == 1]
        treated_pos = df_treated.index.tolist()
        ax.scatter(
            x=x_jittered[treated_pos],
            y=df_treated["engagement_score"].values,
            alpha=0.4,
            s=40,
            color="red",
            label="Treated",
        )
        # Add mean lines.
        if len(df_control) > 0:
            control_mean = float(df_control["engagement_score"].mean())
            ax.axhline(
                y=control_mean,
                xmin=0,
                xmax=0.45,
                color="blue",
                linestyle="-",
                linewidth=2,
                alpha=0.7,
            )
        if len(df_treated) > 0:
            treated_mean = float(df_treated["engagement_score"].mean())
            ax.axhline(
                y=treated_mean,
                xmin=0.55,
                xmax=1,
                color="red",
                linestyle="-",
                linewidth=2,
                alpha=0.7,
            )
        ax.set_xlabel("Intervention", fontsize=11)
        ax.set_ylabel("Engagement Score", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Control", "Treated"], fontsize=9)
        ax.set_title(f"Department {dept_id}", fontsize=12)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=10, loc="best")
    # Hide unused subplots.
    for idx in range(n_depts, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    plt.tight_layout()


def plot_all_correlations_to_intervention(
    data: pd.DataFrame,
    *,
    numeric_cols: Optional[List[str]] = None,
    figsize: Tuple[int, int] = _FIGSIZE_SINGLE,
) -> None:
    """
    Regress all numeric columns on intervention and plot treatment effects.

    Fits linear regressions with intervention as the predictor and each numeric
    column as the outcome. Displays coefficients and confidence intervals,
    showing the treatment effect on each variable.

    :param data: DataFrame with 'intervention' column and numeric columns
    :param numeric_cols: List of numeric column names to include. If None,
                         uses all numeric columns except 'intervention'
    :param figsize: Figure size as (width, height)
    """
    if numeric_cols is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if "intervention" in numeric_cols:
            numeric_cols.remove("intervention")
    # Fit regressions for each column.
    results = []
    X = np.array(data["intervention"]).reshape(-1, 1)
    for col in numeric_cols:
        y = np.array(data[col])
        # Skip columns with NaN values.
        valid_mask = ~(np.isnan(X.flatten()) | np.isnan(y))
        if valid_mask.sum() < 2:
            continue
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        model = sklearn.linear_model.LinearRegression().fit(X_valid, y_valid)
        coef = model.coef_[0]
        results.append({"variable": col, "coefficient": coef})
    results_df = pd.DataFrame(results).sort_values("coefficient")
    # Plot coefficients.
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["blue" if x < 0 else "red" for x in results_df["coefficient"]]
    ax.barh(
        results_df["variable"],
        results_df["coefficient"],
        color=colors,
        alpha=0.7,
    )
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Treatment Effect Coefficient", fontsize=11)
    ax.set_ylabel("Variable", fontsize=11)
    ax.set_title("Regression of All Variables on Intervention", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()


def plot_all_variables_vs_intervention(
    data: pd.DataFrame,
    *,
    numeric_cols: Optional[List[str]] = None,
    figsize_per_plot: Tuple[int, int] = _FIGSIZE_PER_SUBPLOT,
) -> None:
    """
    Plot scatter with jitter and means for each numeric variable vs intervention.

    Creates a grid of subplots, one for each numeric column, showing scatter
    points colored by intervention (blue=control, red=treated) with horizontal
    jitter and mean lines for each group.

    :param data: DataFrame with 'intervention' column and numeric columns
    :param numeric_cols: List of numeric column names to include. If None,
                         uses all numeric columns except 'intervention'
    :param figsize_per_plot: Size per subplot as (width, height)
    """
    if numeric_cols is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if "intervention" in numeric_cols:
            numeric_cols.remove("intervention")
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes.flatten() if len(numeric_cols) > 1 else [axes]
    # Add jitter to intervention values once for all subplots.
    jitter = np.random.normal(0, 0.03, size=len(data))
    x_jittered = np.array(data["intervention"]) + jitter
    for idx, col in enumerate(numeric_cols):
        ax = axes_flat[idx]
        # Get control and treated data.
        df_control = data[data["intervention"] == 0]
        df_treated = data[data["intervention"] == 1]
        control_indices = df_control.index
        treated_indices = df_treated.index
        # Plot control group.
        ax.scatter(
            x=x_jittered[control_indices],
            y=df_control[col],
            alpha=0.4,
            s=50,
            color="blue",
            label="Control",
        )
        # Plot treated group.
        ax.scatter(
            x=x_jittered[treated_indices],
            y=df_treated[col],
            alpha=0.4,
            s=50,
            color="red",
            label="Treated",
        )
        # Add mean lines.
        control_mean = df_control[col].mean()
        treated_mean = df_treated[col].mean()
        ax.axhline(
            y=control_mean,
            xmin=0,
            xmax=0.45,
            color="blue",
            linestyle="-",
            linewidth=2,
            alpha=0.7,
        )
        ax.axhline(
            y=treated_mean,
            xmin=0.55,
            xmax=1,
            color="red",
            linestyle="-",
            linewidth=2,
            alpha=0.7,
        )
        ax.set_xlabel("Intervention", fontsize=10)
        ax.set_ylabel(col, fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Control", "Treated"], fontsize=9)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9, loc="best")
    # Hide unused subplots.
    for idx in range(len(numeric_cols), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    plt.suptitle("All Variables vs Intervention", fontsize=14, y=0.995)
    plt.tight_layout()


def plot_all_variables_density_by_intervention(
    data: pd.DataFrame,
    *,
    numeric_cols: Optional[List[str]] = None,
    figsize_per_plot: Tuple[int, int] = _FIGSIZE_PER_SUBPLOT,
) -> None:
    """
    Plot density curves for each numeric variable by intervention group.

    Creates a grid of subplots, one for each numeric column, showing probability
    density functions for control (blue) and treated (red) groups with mean
    lines marking central tendency.

    :param data: DataFrame with 'intervention' column and numeric columns
    :param numeric_cols: List of numeric column names to include. If None,
                         uses all numeric columns except 'intervention'
    :param figsize_per_plot: Size per subplot as (width, height)
    """
    if numeric_cols is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if "intervention" in numeric_cols:
            numeric_cols.remove("intervention")
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes.flatten() if len(numeric_cols) > 1 else [axes]
    for idx, col in enumerate(numeric_cols):
        ax = axes_flat[idx]
        # Get control and treated data.
        df_control = data[data["intervention"] == 0]
        df_treated = data[data["intervention"] == 1]
        # Plot density for control group.
        sns.kdeplot(
            data=df_control,
            x=col,
            ax=ax,
            color="blue",
            linewidth=2,
            label=f"Control (n={len(df_control)})",
            fill=False,
        )
        # Plot density for treated group.
        sns.kdeplot(
            data=df_treated,
            x=col,
            ax=ax,
            color="red",
            linewidth=2,
            label=f"Treated (n={len(df_treated)})",
            fill=False,
        )
        # Add mean lines.
        control_mean = float(df_control[col].mean())
        treated_mean = float(df_treated[col].mean())
        ax.axvline(
            control_mean,
            color="blue",
            linestyle="-",
            linewidth=2,
            alpha=0.7,
        )
        ax.axvline(
            treated_mean,
            color="red",
            linestyle="-",
            linewidth=2,
            alpha=0.7,
        )
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9, loc="best")
    # Hide unused subplots.
    for idx in range(len(numeric_cols), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    plt.suptitle(
        "Distribution of All Variables by Intervention", fontsize=14, y=0.995
    )
    plt.tight_layout()


# #############################################################################
# Cell 2: Propensity Score Matching Methods.
# #############################################################################


def propensity_score_matching(
    data: pd.DataFrame,
    *,
    treatment_col: str = "intervention",
    ps_col: str = "propensity_score",
    outcome_col: str = "engagement_score",
) -> pd.DataFrame:
    """
    Perform 1-nearest neighbor propensity score matching.

    For each treated unit, finds the nearest control unit (and vice versa)
    based on propensity score distance. Uses KNeighborsRegressor with
    n_neighbors=1 to fit outcome models for each group.

    :param data: DataFrame with treatment, propensity score, and outcome columns
    :param treatment_col: Name of binary treatment column
    :param ps_col: Name of propensity score column
    :param outcome_col: Name of outcome column
    :return: DataFrame with original data plus 'match' column with matched outcomes
    """

    # Separate treated and control groups.
    treated = data.query(f"{treatment_col}==1")
    untreated = data.query(f"{treatment_col}==0")
    # Fit KNN regressors on outcomes for each group.
    # Control model predicts outcomes for treated units.
    knn_control = sklearn.neighbors.KNeighborsRegressor(n_neighbors=1).fit(
        untreated[[ps_col]], untreated[outcome_col]
    )
    # Treated model predicts outcomes for control units.
    knn_treated = sklearn.neighbors.KNeighborsRegressor(n_neighbors=1).fit(
        treated[[ps_col]], treated[outcome_col]
    )
    # Get matched outcomes for treated: what they would have gotten as control.
    treated_with_match = treated.assign(
        match=knn_control.predict(treated[[ps_col]])
    )
    # Get matched outcomes for control: what they would have gotten as treated.
    untreated_with_match = untreated.assign(
        match=knn_treated.predict(untreated[[ps_col]])
    )
    # Combine and return.
    return pd.concat([treated_with_match, untreated_with_match])


def calculate_psm_ate(
    data: pd.DataFrame,
    *,
    treatment_col: str = "intervention",
    outcome_col: str = "engagement_score",
) -> float:
    """
    Calculate Average Treatment Effect from propensity score matching.

    Computes ATE as the average difference between observed and matched outcomes,
    weighted by treatment assignment:
    - For treated: (Y_obs - Y_matched)
    - For control: (Y_matched - Y_obs)

    :param data: DataFrame from propensity_score_matching with 'match' column
    :param treatment_col: Name of binary treatment column
    :param outcome_col: Name of outcome column
    :return: Estimated average treatment effect
    """
    treatment = data[treatment_col].values
    outcome = data[outcome_col].values
    matched = data["match"].values
    # Compute difference: observed - matched outcome.
    diff = outcome - matched
    # Weight by treatment: treated units contribute positively, control negatively.
    ate = np.mean(treatment * diff - (1 - treatment) * diff)
    return float(ate)


def plot_iptw(
    data: pd.DataFrame,
    *,
    ps_col: str = "propensity_score",
    outcome_col: str = "engagement_score",
    treatment_col: str = "intervention",
    figsize: Tuple[int, int] = _FIGSIZE_SINGLE,
) -> None:
    """
    Plot Inverse Probability of Treatment Weighting (IPTW) results.

    Visualizes the relationship between propensity score and outcome, with
    point sizes representing IPTW weights. Blue dots represent control units,
    red dots represent treated units.

    :param data: DataFrame with treatment, propensity score, and outcome columns
    :param ps_col: Name of propensity score column
    :param outcome_col: Name of outcome column
    :param treatment_col: Name of binary treatment column
    :param figsize: Figure size as (width, height)
    """
    # Calculate IPTW weights.
    iptw_data = data.assign(
        weight=(
            data[treatment_col] / data[ps_col]
            + (1 - data[treatment_col]) / (1 - data[ps_col])
        ),
        ps_rounded=data[ps_col].round(2),
    )
    # Aggregate by propensity score and treatment.
    grouped = (
        iptw_data.groupby(["ps_rounded", treatment_col])[["weight", outcome_col]]
        .mean()
        .reset_index()
    )
    # Plot with blue for control, red for treated.
    fig, ax = plt.subplots(figsize=figsize)
    for t, color in [(0, "blue"), (1, "red")]:
        subset = grouped.query(f"{treatment_col}=={t}")
        ax.scatter(
            data=subset,
            x="ps_rounded",
            y=outcome_col,
            s=subset["weight"] * 100,
            color=color,
            alpha=0.6,
            label="Control" if t == 0 else "Treated",
            edgecolors="black",
            linewidth=0.5,
        )
    ax.set_xlabel("Propensity Score", fontsize=11)
    ax.set_ylabel("Engagement Score", fontsize=11)
    ax.set_title("Inverse Probability of Treatment Weighting", fontsize=12)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def estimate_ate_iptw(
    data: pd.DataFrame,
    *,
    ps_col: str = "propensity_score",
    outcome_col: str = "engagement_score",
    treatment_col: str = "intervention",
) -> Tuple[float, float, float]:
    """
    Estimate Average Treatment Effect using Inverse Probability weighting.

    Computes IPTW weights as 1/PS for treated and 1/(1-PS) for control,
    then calculates weighted averages of outcomes in each group.

    :param data: DataFrame with treatment, propensity score, and outcome columns
    :param ps_col: Name of propensity score column
    :param outcome_col: Name of outcome column
    :param treatment_col: Name of binary treatment column
    :return: Tuple of (weighted_E_Y1, weighted_E_Y0, ATE)
    """
    # Separate treated and control groups.
    treated = data.query(f"{treatment_col}==1")
    control = data.query(f"{treatment_col}==0")
    # Compute IPTW weights: 1/PS for treated, 1/(1-PS) for control.
    weight_treated = 1 / treated[ps_col]
    weight_control = 1 / (1 - control[ps_col])
    # Compute weighted averages of outcomes.
    # E[Y|T=1] and E[Y|T=0] in the pseudo-population.
    weighted_e_y1 = np.sum(treated[outcome_col] * weight_treated) / len(data)
    weighted_e_y0 = np.sum(control[outcome_col] * weight_control) / len(data)
    # ATE is the difference.
    ate = weighted_e_y1 - weighted_e_y0
    return weighted_e_y1, weighted_e_y0, ate


def estimate_ate_with_ps(
    data: pd.DataFrame,
    ps_formula: str,
    *,
    treatment_col: str = "intervention",
    outcome_col: str = "engagement_score",
) -> float:
    """
    Estimate ATE using IPW estimator with logistic regression propensity score.

    Fits a logistic regression to estimate propensity scores from the provided
    formula, then computes the IPW estimator:
    ATE = E[(T - PS) / (PS * (1 - PS)) * Y]

    :param data: DataFrame with treatment and outcome columns
    :param ps_formula: Patsy formula for propensity score model
    :param treatment_col: Name of binary treatment column
    :param outcome_col: Name of outcome column
    :return: Estimated average treatment effect
    """
    # Create design matrix from formula.
    X = patsy.dmatrix(ps_formula, data)
    # Fit logistic regression to estimate propensity scores.
    ps_model = sklearn.linear_model.LogisticRegression(max_iter=1000).fit(
        X, data[treatment_col]
    )
    ps = ps_model.predict_proba(X)[:, 1]
    # Compute IPW estimator: E[(T - PS) / (PS * (1 - PS)) * Y].
    return np.mean(
        (data[treatment_col] - ps) / (ps * (1 - ps)) * data[outcome_col]
    )


def estimate_ate_stabilized_weights(
    data: pd.DataFrame,
    *,
    ps_col: str = "propensity_score",
    outcome_col: str = "engagement_score",
    treatment_col: str = "intervention",
) -> float:
    """
    Estimate ATE using stabilized propensity weights.

    Computes stabilized weights: PS_weight = P(T=1) / PS for treated,
    and (1 - P(T=1)) / (1 - PS) for control. This reduces the influence
    of extreme propensity scores compared to standard IPTW.

    :param data: DataFrame with treatment, propensity score, and outcome columns
    :param ps_col: Name of propensity score column
    :param outcome_col: Name of outcome column
    :param treatment_col: Name of binary treatment column
    :return: Estimated average treatment effect
    """
    # Compute marginal probability of treatment.
    p_treatment = data[treatment_col].mean()
    # Separate treated and control groups.
    treated = data.query(f"{treatment_col}==1")
    control = data.query(f"{treatment_col}==0")
    # Compute stabilized weights.
    # Stabilized weight for treated: P(T=1) / PS.
    weight_treated_stable = p_treatment / treated[ps_col]
    # Stabilized weight for control: (1 - P(T=1)) / (1 - PS).
    weight_control_stable = (1 - p_treatment) / (1 - control[ps_col])
    # Compute weighted averages.
    n_treated = len(treated)
    n_control = len(control)
    weighted_mean_y1 = (
        np.sum(treated[outcome_col] * weight_treated_stable) / n_treated
    )
    weighted_mean_y0 = (
        np.sum(control[outcome_col] * weight_control_stable) / n_control
    )
    # ATE is the difference.
    ate = weighted_mean_y1 - weighted_mean_y0
    return ate


def plot_propensity_distributions(
    data: pd.DataFrame,
    *,
    ps_col: str = "propensity_score",
    treatment_col: str = "intervention",
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """
    Plot propensity score distributions before and after weighting.

    Creates side-by-side histograms showing the unweighted propensity
    distribution and the propensity distribution after applying stabilized
    weights. This illustrates how weighting improves covariate balance.

    :param data: DataFrame with treatment, propensity score columns, and optional weights
    :param ps_col: Name of propensity score column
    :param treatment_col: Name of binary treatment column
    :param figsize: Figure size as (width, height)
    """
    # Compute marginal probability of treatment and stabilized weights.
    p_treatment = data[treatment_col].mean()
    treated = data.query(f"{treatment_col}==1")
    control = data.query(f"{treatment_col}==0")
    weight_treated = p_treatment / treated[ps_col]
    weight_control = (1 - p_treatment) / (1 - control[ps_col])
    # Create side-by-side subplots.
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, sharex=True, sharey=True
    )
    # Plot 1: Unweighted propensity score distributions.
    sns.histplot(
        control[ps_col],
        stat="probability",
        label="Control",
        color="C0",
        bins=30,
        ax=ax1,
        alpha=0.5,
    )
    sns.histplot(
        treated[ps_col],
        stat="probability",
        label="Treated",
        color="C2",
        alpha=0.5,
        bins=30,
        ax=ax1,
    )
    ax1.set_title("Unweighted Propensity Distribution")
    ax1.legend()
    # Plot 2: Weighted propensity score distributions.
    sns.histplot(
        control.assign(w=weight_control),
        x=ps_col,
        stat="probability",
        color="C0",
        weights="w",
        label="Control",
        bins=30,
        ax=ax2,
        alpha=0.5,
    )
    sns.histplot(
        treated.assign(w=weight_treated),
        x=ps_col,
        stat="probability",
        color="C2",
        weights="w",
        label="Treated",
        bins=30,
        alpha=0.5,
        ax=ax2,
    )
    ax2.set_title("Weighted Propensity Distribution (Stabilized Weights)")
    ax2.legend()
    plt.tight_layout()


# #############################################################################
# Cell 3: Bootstrap Methods for Confidence Intervals.
# #############################################################################


def bootstrap(
    data: pd.DataFrame,
    est_fn: Callable[[pd.DataFrame], float],
    *,
    rounds: int = 200,
    seed: int = 123,
    pcts: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Bootstrap helper function for estimating confidence intervals.

    Resamples data with replacement and applies estimation function to each
    sample to estimate parameter distribution.

    :param data: DataFrame to resample
    :param est_fn: Estimation function to apply to each sample
    :param rounds: Number of bootstrap resamples
    :param seed: Random seed for reproducibility
    :param pcts: Percentiles to compute (default: [2.5, 97.5])
    :return: Percentile values from bootstrap distribution
    """
    if pcts is None:
        pcts = [2.5, 97.5]
    np.random.seed(seed)
    stats = joblib.Parallel(n_jobs=4)(
        joblib.delayed(est_fn)(data.sample(frac=1, replace=True))
        for _ in range(rounds)
    )
    return np.percentile(np.array(list(stats)), pcts)


def estimate_confidence_interval_bootstrap(
    data: pd.DataFrame,
    est_fn: Callable[[pd.DataFrame], float],
    *,
    rounds: int = 200,
    seed: int = 123,
    n_jobs: int = 4,
    pcts: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Estimate confidence interval using bootstrap resampling.

    Resamples data with replacement and applies estimation function to each
    sample to estimate parameter distribution. Computes percentiles from
    the bootstrap distribution to form a confidence interval.

    :param data: DataFrame to resample
    :param est_fn: Estimation function to apply to each sample
    :param rounds: Number of bootstrap resamples
    :param seed: Random seed for reproducibility
    :param n_jobs: Number of parallel jobs for computation
    :param pcts: Percentiles to compute (default: [2.5, 97.5])
    :return: Array of percentile values forming confidence interval
    """
    if pcts is None:
        pcts = [2.5, 97.5]
    np.random.seed(seed)
    stats = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(est_fn)(data.sample(frac=1, replace=True))
        for _ in range(rounds)
    )
    return np.percentile(np.array(list(stats)), pcts)
