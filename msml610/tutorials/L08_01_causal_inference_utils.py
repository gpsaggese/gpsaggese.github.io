"""
Utility functions for causal inference tutorial (L08_01).

Import as:

import msml610.tutorials.L08_01_causal_inference_utils as mtl0cireout
"""

import os

import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression


# #############################################################################
# Data loading functions.
# #############################################################################


def load_xmas_sales_data(data_dir: str) -> pd.DataFrame:
    """
    Load and prepare Christmas sales data.

    :param data_dir: Directory containing xmas_sales.csv
    :return: DataFrame with xmas sales data
    """
    data = pd.read_csv(os.path.join(data_dir, "xmas_sales.csv"))
    data["is_on_sale"] = data["is_on_sale"].astype(float)
    return data


# #############################################################################
# Plotting functions.
# #############################################################################


def plot_xmas_sales_boxplot(data: pd.DataFrame, *, figsize: tuple = (10, 5)) -> mfigure.Figure:
    """
    Create a boxplot of weekly sales by treatment status.

    :param data: DataFrame with columns 'is_on_sale', 'weekly_amount_sold'
    :param figsize: Figure size as (width, height)
    :return: matplotlib figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.boxplot(
        y="weekly_amount_sold",
        x="is_on_sale",
        data=data,
        ax=ax
    )
    ax.set_xlabel("is_on_sale", fontsize=20)
    ax.set_ylabel("weekly_amount_sold", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=18)
    return fig


def plot_sales_bias_analysis(data: pd.DataFrame, marker: str = "o") -> mfigure.Figure:
    """
    Plot sales bias analysis showing treated vs control groups with regression lines.

    :param data: DataFrame with columns 'is_on_sale', 'avg_week_sales', 'weekly_amount_sold'
    :param marker: Marker style for plotting
    :return: matplotlib figure object
    """
    plt.rc("font", size=20)
    fig = plt.figure()

    df_treated = data.query("is_on_sale==1")
    sns.regplot(
        data=df_treated,
        ci=None,
        x="avg_week_sales",
        y="weekly_amount_sold",
        scatter=False,
        line_kws={"color": "red", "linewidth": 1, "linestyle": "--", "marker": "."},
    )
    plt.scatter(
        x=df_treated["avg_week_sales"],
        y=df_treated["weekly_amount_sold"],
        label="treated (cut prices)",
        color="red",
        alpha=0.1,
        marker=marker,
    )

    df_control = data.query("is_on_sale==0")
    sns.regplot(
        data=df_control,
        ci=None,
        x="avg_week_sales",
        y="weekly_amount_sold",
        scatter=False,
        line_kws={"color": "blue", "linewidth": 1, "linestyle": "--", "marker": "."},
    )

    plt.scatter(
        x=df_control["avg_week_sales"],
        y=df_control["weekly_amount_sold"],
        label="control (not cut prices)",
        color="blue",
        alpha=0.1,
        marker=marker,
    )
    plt.legend(fontsize="14")
    return fig


def plot_single_vs_separate_trends():
    """
    Plot comparison of single trend line vs separate trend lines for groups.

    :return: matplotlib figure object
    """
    # Set the style.
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    # Create synthetic data.
    np.random.seed(42)
    # Generate data for large businesses.
    large_discount = np.array([0.1, 0.2, 0.3])
    large_amount = np.array([8, 12, 16]) + 10
    # Generate data for small businesses.
    small_discount = np.array([0.05, 0.15, 0.25])
    small_amount = np.array([4, 8, 12])
    # Set up the figure with two subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Single trend line.
    ax1.set_title("Single Trend Line Model")
    ax1.set_xlabel("Price discount")
    ax1.set_ylabel("Amount sold")
    # Scatter plots.
    ax1.scatter(
        large_discount,
        large_amount,
        s=120,
        color="lightblue",
        edgecolor="black",
        alpha=0.7,
        zorder=2,
    )
    ax1.scatter(
        small_discount,
        small_amount,
        s=120,
        color="#1f77b4",
        edgecolor="black",
        alpha=0.7,
        zorder=2,
    )
    # Combine data for single trend line.
    all_discount = np.concatenate([large_discount, small_discount])
    all_amount = np.concatenate([large_amount, small_amount])
    # Calculate and plot the trend line.
    z = np.polyfit(all_discount, all_amount, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 0.35, 100)
    ax1.plot(x_line, p(x_line), "k-", linewidth=2, zorder=1)

    # Plot 2: Separate trend lines.
    ax2.set_title("Separate Trend Lines Model")
    ax2.set_xlabel("Price discount")
    ax2.set_ylabel("Amount sold")
    # Scatter plots (same as plot 1).
    ax2.scatter(
        large_discount,
        large_amount,
        s=120,
        color="lightblue",
        edgecolor="black",
        alpha=0.7,
        zorder=2,
    )
    ax2.scatter(
        small_discount,
        small_amount,
        s=120,
        color="#1f77b4",
        edgecolor="black",
        alpha=0.7,
        zorder=2,
    )
    # Calculate and plot separate trend lines.
    z_large = np.polyfit(large_discount, large_amount, 1)
    p_large = np.poly1d(z_large)
    ax2.plot(x_line, p_large(x_line), "k-", linewidth=2, zorder=1)
    z_small = np.polyfit(small_discount, small_amount, 1)
    p_small = np.poly1d(z_small)
    ax2.plot(x_line, p_small(x_line), "k-", linewidth=2, zorder=1)

    # Remove top and right spines.
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 0.35)
        ax.set_ylim(0, 35)
        # Remove tick marks.
        ax.tick_params(axis="both", which="both", length=0)
        # Remove numerical ticks.
        ax.set_xticks([])
        ax.set_yticks([])
    # Create legend.
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="lightblue",
            markeredgecolor="black",
            markersize=12,
            label="Large business",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1f77b4",
            markeredgecolor="black",
            markersize=12,
            label="Small business",
        ),
    ]
    fig.legend(
        handles=legend_elements, loc="upper right", bbox_to_anchor=(0.99, 0.99)
    )
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    return fig


def plot_simpsons_paradox():
    """
    Plot Simpson's paradox showing how overall trend can differ from group trends.

    :return: matplotlib figure object
    """
    # Generate synthetic data.
    x1 = np.array([1, 2, 3, 4, 5])
    y1 = 2 * x1 + 2  # Positive trend.
    x2 = np.array([7, 8, 9, 10, 11])
    y2 = 2 * x2 - 20  # Positive trend but different intercept.
    # Combine the data.
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    group = ["blue"] * len(x1) + ["red"] * len(x2)
    # Create a DataFrame.
    df = pd.DataFrame({"x": x, "y": y, "group": group})
    # Fit separate linear models for each group.
    model_blue = LinearRegression().fit(x1.reshape(-1, 1), y1)
    model_red = LinearRegression().fit(x2.reshape(-1, 1), y2)
    # Fit an overall model ignoring group distinction.
    model_all = LinearRegression().fit(x.reshape(-1, 1), y)
    # Create a plot.
    fig = plt.figure(figsize=(7, 5))
    # Plot data points.
    plt.scatter(
        df[df["group"] == "blue"]["x"],
        df[df["group"] == "blue"]["y"],
        edgecolor="blue",
        facecolor="none",
        s=100,
        label="Group Blue",
    )
    plt.scatter(
        df[df["group"] == "red"]["x"],
        df[df["group"] == "red"]["y"],
        edgecolor="red",
        facecolor="none",
        s=100,
        label="Group Red",
    )
    # Plot group regression lines.
    plt.plot(x1, model_blue.predict(x1.reshape(-1, 1)), color="blue", linewidth=2)
    plt.plot(x2, model_red.predict(x2.reshape(-1, 1)), color="red", linewidth=2)
    # Plot overall regression line.
    x_range = np.linspace(min(x), max(x), 100)
    plt.plot(
        x_range,
        model_all.predict(x_range.reshape(-1, 1)),
        "k--",
        linewidth=2,
        label="Overall Trend",
    )
    # Labels and legend.
    plt.xlabel("x")
    plt.ylabel("y")
    return fig
