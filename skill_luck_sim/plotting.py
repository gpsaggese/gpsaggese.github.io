"""
Plot the sweep results.

Import as:

import skill_luck_sim.plotting as slsiplot
"""

import logging

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_LOG = logging.getLogger(__name__)

# One blue hue stepped light to dark, for ordered values like K or S. Checked
# with a palette validator for monotone lightness and contrast on white.
_ORDINAL_BLUES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
_INK = "#2b2b2b"
_MUTED = "#8a8a85"
_GRID = "#e6e6e3"


# #############################################################################
# Helpers
# #############################################################################


def _style_axis(ax: plt.Axes, n_tasks: list[int]) -> None:
    """
    Make the grid and spines recessive and use a log x axis for N.

    Ticks go only at the N values of the grid, without minor ticks, so that
    labels do not collide.
    """
    ax.set_xscale("log")
    ax.set_xticks(n_tasks)
    ax.set_xticklabels([str(n) for n in n_tasks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_INK, labelsize=8)


def _plot_lines_by(
    ax: plt.Axes,
    df: pd.DataFrame,
    line_col: str,
    y_col: str,
    line_values: list[int],
    label_prefix: str,
) -> None:
    """
    Draw one line per value of `line_col`, y against N, with 95% error bars.
    """
    for color, value in zip(_ORDINAL_BLUES, line_values):
        df_line = df[df[line_col] == value].sort_values("n_tasks")
        ax.errorbar(
            df_line["n_tasks"],
            df_line[y_col],
            yerr=1.96 * df_line[f"{y_col}_se"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            capsize=0,
            label=f"{label_prefix}{value}",
        )


# #############################################################################
# Figures
# #############################################################################


def plot_curse(df: pd.DataFrame, file_name: str) -> None:
    """
    Plot the winner's curse and the fresh-task error against N.

    One column per S, one line per K. The top row is the observed score of
    the winner minus its true Ybar. The bottom row is the same for a
    re-evaluation of the winner on fresh tasks.

    :param df: curse sweep, one row per (K, N, S)
    :param file_name: output PNG path
    """
    seeds = sorted(df["n_seeds"].unique())
    configs = sorted(df["n_configs"].unique())
    fig, axes = plt.subplots(
        2, len(seeds), figsize=(3.2 * len(seeds), 6), sharex=True, sharey=True
    )
    rows = [
        ("curse", "Winner's score minus true rate"),
        ("fresh_task_error", "Fresh-task score minus true rate"),
    ]
    for i, (y_col, y_label) in enumerate(rows):
        for j, n_seeds in enumerate(seeds):
            ax = axes[i, j]
            _style_axis(ax, sorted(df["n_tasks"].unique()))
            ax.axhline(0, color=_MUTED, linewidth=1, linestyle="--")
            df_panel = df[df["n_seeds"] == n_seeds]
            _plot_lines_by(ax, df_panel, "n_configs", y_col, configs, "K = ")
            if i == 0:
                ax.set_title(f"S = {n_seeds} seeds", color=_INK, fontsize=10)
            if j == 0:
                ax.set_ylabel(y_label, color=_INK, fontsize=9)
            if i == 1:
                ax.set_xlabel("N tasks", color=_INK, fontsize=9)
    axes[0, 0].legend(
        frameon=False, fontsize=8, title="configs", title_fontsize=8
    )
    fig.tight_layout()
    fig.savefig(file_name, dpi=150, facecolor="white")
    plt.close(fig)
    _LOG.info("Saved '%s'", file_name)


def plot_prob_best(df: pd.DataFrame, file_name: str) -> None:
    """
    Plot the probability that the observed winner is the truly best config.

    :param df: curse sweep, one row per (K, N, S)
    :param file_name: output PNG path
    """
    seeds = sorted(df["n_seeds"].unique())
    configs = sorted(df["n_configs"].unique())
    fig, axes = plt.subplots(
        1, len(seeds), figsize=(3.2 * len(seeds), 3.2), sharey=True
    )
    for j, n_seeds in enumerate(seeds):
        ax = axes[j]
        _style_axis(ax, sorted(df["n_tasks"].unique()))
        df_panel = df[df["n_seeds"] == n_seeds]
        _plot_lines_by(
            ax, df_panel, "n_configs", "is_truly_best", configs, "K = "
        )
        ax.set_ylim(0, 1)
        ax.set_title(f"S = {n_seeds} seeds", color=_INK, fontsize=10)
        ax.set_xlabel("N tasks", color=_INK, fontsize=9)
    axes[0].set_ylabel("P(winner is truly best)", color=_INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, title="configs", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(file_name, dpi=150, facecolor="white")
    plt.close(fig)
    _LOG.info("Saved '%s'", file_name)


def plot_coverage(df: pd.DataFrame, file_name: str) -> None:
    """
    Plot bootstrap CI coverage of the true Ybar against N.

    One panel per resampling scheme, one line per S, nominal 95% dashed.

    :param df: coverage sweep, one row per (method, N, S)
    :param file_name: output PNG path
    """
    methods = [
        ("task", "Resample tasks"),
        ("task_seed", "Resample tasks, then seeds"),
    ]
    seeds = sorted(df["n_seeds"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (method, title) in zip(axes, methods):
        _style_axis(ax, sorted(df["n_tasks"].unique()))
        ax.axhline(0.95, color=_MUTED, linewidth=1, linestyle="--")
        df_panel = df[df["method"] == method]
        _plot_lines_by(ax, df_panel, "n_seeds", "coverage", seeds, "S = ")
        ax.set_title(title, color=_INK, fontsize=10)
        ax.set_xlabel("N tasks", color=_INK, fontsize=9)
    axes[0].set_ylabel("Coverage of true rate (95% CI)", color=_INK, fontsize=9)
    axes[1].legend(
        frameon=False,
        fontsize=8,
        title="seeds",
        title_fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    fig.tight_layout()
    fig.savefig(file_name, dpi=150, facecolor="white")
    plt.close(fig)
    _LOG.info("Saved '%s'", file_name)
