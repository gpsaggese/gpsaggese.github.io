"""
Analyze and plot the results of the causal success simulation.

The simulation and its config live in `causal_success_utils.py`. This file only
analyzes the output of `run_experiment()` for `causal_success_example.ipynb`, and
its functions are named and ordered after the cells of that notebook.

Import as:

import research.Causal_Analysis_of_Agent_Skill_And_Luck.causal_success_analysis_utils as rcaoasalcsau
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import econml.dml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import sklearn.ensemble

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import research.Causal_Analysis_of_Agent_Skill_And_Luck.causal_success_utils as rcaoasalcsu

_LOG = logging.getLogger(__name__)

# Talent columns of `get_results_dataframe()`: the confounders of the causal
# analysis.
_TALENT_COLS = ["talent_intensity", "talent_iq", "talent_networking"]


# #############################################################################
# Cell 1.1: The config of the baseline experiment
# #############################################################################


def cell1_1_config_to_df(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Flatten a nested config into a table with one row per parameter.

    :param config: config of an experiment, e.g., from `get_default_config()`
    :return: table with the columns `group`, `parameter`, and `value`
        ```
        group       parameter  value
        experiment  seed       42
        population  n_agents   100
        ```
    """
    _LOG.debug("config.keys()=%s", list(config.keys()))
    rows: List[Dict[str, str]] = []
    for group, values in config.items():
        if isinstance(values, dict):
            # A group of parameters, e.g., `population`.
            for name, value in values.items():
                rows.append(
                    {"group": group, "parameter": name, "value": str(value)}
                )
        else:
            # A top-level parameter, e.g., `seed`.
            rows.append(
                {"group": "experiment", "parameter": group, "value": str(values)}
            )
    df = pd.DataFrame(rows)
    _LOG.debug("df.shape=%s", df.shape)
    return df


def cell1_1_get_setup_txt(config: Dict[str, Any]) -> str:
    """
    Describe the set-up of an experiment in one line, for plot titles.

    :param config: config of an experiment
    :return: set-up of the experiment, e.g., "100 agents, 80 periods, seed 42"
    """
    _LOG.debug("config.keys()=%s", list(config.keys()))
    setup_txt = (
        f"{config['population']['n_agents']} agents, "
        f"{config['events']['n_periods']} periods, seed {config['seed']}"
    )
    _LOG.debug("return=%s", setup_txt)
    return setup_txt


# #############################################################################
# Cell 1.3: The talents of the agents
# #############################################################################


def cell1_3_plot_talent_distributions(
    df: pd.DataFrame,
    setup_txt: str,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the distribution of each talent, on the same axes.

    :param df: output of `get_results_dataframe()`
    :param setup_txt: set-up of the experiment, shown in the title
    :param figsize: figure size
        - Default: `plt.rcParams["figure.figsize"]`
    """
    _LOG.debug(hprint.to_str("df.shape setup_txt figsize"))
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
    # Use the same bins on [0, 1] for all the talents, since they share the
    # same range.
    bins = np.linspace(0.0, 1.0, 21)
    labels = ["Intensity", "IQ", "Networking"]
    for ax, col, label in zip(axes, _TALENT_COLS, labels):
        values = df[col]
        sns.histplot(x=values, bins=bins, ax=ax)
        ax.axvline(
            values.mean(),
            color="red",
            linestyle="--",
            label=f"mean {values.mean():.3f}",
        )
        ax.set_xlabel(label)
        ax.set_title(f"{label} (std {values.std():.3f})")
        ax.legend()
    fig.suptitle(f"Talents of the agents ({setup_txt})")
    fig.tight_layout()
    plt.show()


# #############################################################################
# Cell 2.1: The distribution of the final capital
# #############################################################################


def cell2_1_plot_capital_distribution(
    df: pd.DataFrame,
    stats: Dict[str, float],
    setup_txt: str,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the final capital, its log, and its Lorenz curve.

    :param df: output of `get_results_dataframe()`
    :param stats: output of `generate_summary_statistics()` for the same agents
    :param setup_txt: set-up of the experiment, shown in the title
    :param figsize: figure size
        - Default: `plt.rcParams["figure.figsize"]`
    """
    _LOG.debug(hprint.to_str("df.shape setup_txt figsize"))
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    capital = df["capital"].to_numpy(dtype=float)
    log_capital = np.log(capital)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    # Plot the capital with its mean and median.
    sns.histplot(x=capital, bins=20, color="orange", ax=axes[0])
    axes[0].axvline(
        np.mean(capital),
        color="red",
        linestyle="--",
        label=f"mean {np.mean(capital):.2f}",
    )
    axes[0].axvline(
        np.median(capital),
        color="black",
        linestyle=":",
        label=f"median {np.median(capital):.2f}",
    )
    axes[0].set_xlabel("Final capital")
    axes[0].set_title(
        f"Capital (Gini {stats['gini_coefficient']:.3f}, "
        f"skewness {scipy.stats.skew(capital):.2f})"
    )
    axes[0].legend()
    # Plot the log of the capital, the outcome of the causal analysis.
    sns.histplot(x=log_capital, bins=20, ax=axes[1])
    axes[1].set_xlabel("log(capital)")
    axes[1].set_title(
        f"log(capital) (skewness {scipy.stats.skew(log_capital):.2f})"
    )
    # Plot the Lorenz curve: the share of capital held by the poorest agents.
    capital_sorted = np.sort(capital)
    cum_share = np.concatenate([[0.0], np.cumsum(capital_sorted)])
    cum_share = cum_share / cum_share[-1]
    pop_share = np.linspace(0.0, 1.0, len(cum_share))
    axes[2].plot([0, 1], [0, 1], "k:", label="equality")
    axes[2].plot(pop_share, cum_share, "b-", label="simulation")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Share of agents, poorest first")
    axes[2].set_ylabel("Share of capital")
    axes[2].set_title(
        f"Lorenz (top 10% {stats['top_10_pct_share']:.1%}, "
        f"bottom 50% {stats['bottom_50_pct_share']:.1%})"
    )
    axes[2].legend()
    fig.suptitle(f"Final capital ({setup_txt})")
    fig.tight_layout()
    plt.show()


# #############################################################################
# Cell 3.2: Correlations with log-capital
# #############################################################################


def cell3_2_compute_log_capital_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation of each talent and event count with
    log-capital.

    The interval is the 95% confidence interval of the correlation (Fisher
    transformation), so it describes the sampling noise of one population.

    :param df: output of `get_results_dataframe()`
    :return: one row per predictor, indexed by its column name, with the
        columns `kind` (`talent` or `events`), `r`, `ci_lower`, `ci_upper`, and
        `p_value`
    """
    _LOG.debug("df.shape=%s", df.shape)
    hdbg.dassert_lt(3, len(df), "A correlation interval needs more than 3 rows")
    predictors = [
        ("talent", "talent_norm"),
        ("talent", "talent_intensity"),
        ("talent", "talent_iq"),
        ("talent", "talent_networking"),
        ("events", "lucky_events"),
        ("events", "unlucky_events"),
        ("events", "net_events"),
    ]
    log_capital = np.log(df["capital"].to_numpy(dtype=float))
    rows: List[Dict[str, Any]] = []
    for kind, col in predictors:
        res = scipy.stats.pearsonr(df[col].to_numpy(dtype=float), log_capital)
        ci = res.confidence_interval(confidence_level=0.95)
        rows.append(
            {
                "predictor": col,
                "kind": kind,
                "r": float(res.statistic),
                "ci_lower": float(ci.low),
                "ci_upper": float(ci.high),
                "p_value": float(res.pvalue),
            }
        )
    corr_df = pd.DataFrame(rows).set_index("predictor")
    _LOG.debug("corr_df.shape=%s", corr_df.shape)
    return corr_df


# #############################################################################
# Cell 3.3: Talent and luck against log-capital
# #############################################################################


def cell3_3_plot_talent_and_luck_vs_log_capital(
    df: pd.DataFrame,
    corr_df: pd.DataFrame,
    setup_txt: str,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot log-capital against the talent norm and against the lucky events.

    :param df: output of `get_results_dataframe()`
    :param corr_df: output of `cell3_2_compute_log_capital_correlations()` for
        the same agents, used for the correlations in the titles
    :param setup_txt: set-up of the experiment, shown in the title
    :param figsize: figure size
        - Default: `plt.rcParams["figure.figsize"]`
    """
    _LOG.debug(hprint.to_str("df.shape setup_txt figsize"))
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    plot_df = df.assign(log_capital=np.log(df["capital"]))
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    panels = [
        ("talent_norm", "Talent norm", axes[0]),
        ("lucky_events", "Realized beneficial events", axes[1]),
    ]
    for col, label, ax in panels:
        sns.regplot(
            data=plot_df,
            x=col,
            y="log_capital",
            ci=None,
            scatter_kws={"alpha": 0.6},
            line_kws={"color": "red", "linestyle": "--"},
            ax=ax,
        )
        row = corr_df.loc[col]
        ax.set_xlabel(label)
        ax.set_ylabel("log(capital)")
        ax.set_title(
            f"{label}\nr = {row['r']:.2f}, "
            f"95% CI [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]"
        )
    fig.suptitle(f"Talent and luck against log-capital ({setup_txt})")
    fig.tight_layout()
    plt.show()


# #############################################################################
# Cell 3.4: Ranks of the agents
# #############################################################################


def cell3_4_add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the talent rank and the capital rank of each agent.

    Rank 1 is the most talented agent (highest `talent_norm`) and the richest
    agent (highest `capital`), so a lower rank is better and the average rank of
    the population is (N + 1) / 2.

    :param df: output of `get_results_dataframe()`
    :return: copy of `df` with the columns `talent_rank` and `capital_rank`
    """
    _LOG.debug("df.shape=%s", df.shape)
    df_ranked = df.copy()
    df_ranked["talent_rank"] = df_ranked["talent_norm"].rank(ascending=False)
    df_ranked["capital_rank"] = df_ranked["capital"].rank(ascending=False)
    _LOG.debug("df_ranked.shape=%s", df_ranked.shape)
    return df_ranked


# #############################################################################
# Cell 3.5: Top performers
# #############################################################################


def cell3_5_summarize_top_performers(
    df_ranked: pd.DataFrame, n_top: int
) -> pd.DataFrame:
    """
    Compare the richest agents with the others and with the population.

    :param df_ranked: output of `cell3_4_add_ranks()`
    :param n_top: number of richest agents in the top group
    :return: one row per group (`top`, `others`, `all`) with the number of
        agents, the mean and median talent rank, the mean capital, and the mean
        event counts
    """
    _LOG.debug(hprint.to_str("df_ranked.shape n_top"))
    hdbg.dassert_lt(0, n_top, "The top group needs at least one agent")
    hdbg.dassert_lt(n_top, len(df_ranked), "The other group needs an agent")
    is_top = df_ranked["capital_rank"] <= n_top
    groups = {
        "top": df_ranked[is_top],
        "others": df_ranked[~is_top],
        "all": df_ranked,
    }
    rows: List[Dict[str, Any]] = []
    for name, group in groups.items():
        rows.append(
            {
                "group": name,
                "n_agents": len(group),
                "mean_talent_rank": float(group["talent_rank"].mean()),
                "median_talent_rank": float(group["talent_rank"].median()),
                "mean_capital": float(group["capital"].mean()),
                "mean_lucky_events": float(group["lucky_events"].mean()),
                "mean_unlucky_events": float(group["unlucky_events"].mean()),
                "mean_net_events": float(group["net_events"].mean()),
            }
        )
    summary_df = pd.DataFrame(rows).set_index("group")
    _LOG.debug("summary_df.shape=%s", summary_df.shape)
    return summary_df


# #############################################################################
# Cell 4.2: Log-capital change of each realized event
# #############################################################################


def cell4_2_compute_log_change_per_event(
    agents: List[rcaoasalcsu.Agent],
) -> pd.DataFrame:
    """
    Measure the observed change in log-capital of each realized event.

    Every applied event multiplies the capital, so it appends one entry to
    `capital_history`, and its change in log-capital is the log of the ratio
    of two consecutive entries. The kind of an event is read from the sign of
    its change:
    - `beneficial`: a positive change, from a realized direct event or a
      spillover
    - `detrimental`: a negative change, from an unlucky event
      - The capital floor can make the change smaller than `log(1 - impact)`
    - `unchanged`: a zero change, whose kind cannot be read from the history
      - E.g., an unlucky event on an agent already at the capital floor, or an
        event with a zero impact in a config that allows it

    :param agents: agents after the simulation
    :return: one row per kind (`beneficial`, `detrimental`, `unchanged`) with
        the number of events and the mean, std, min, and max of the
        log-capital change
        - The statistics are NaN for a kind with no event
    """
    _LOG.debug("len(agents)=%s", len(agents))
    log_changes = np.concatenate(
        [np.diff(np.log(np.asarray(a.capital_history))) for a in agents]
    )
    gains = log_changes[log_changes > 0]
    losses = log_changes[log_changes < 0]
    unchanged = log_changes[log_changes == 0]
    # Each applied event adds one entry to the history. A lucky event never
    # lowers the capital and an unlucky one never raises it, so the gains and
    # the losses are bounded by the counts of each kind.
    n_lucky = sum(a.lucky_events for a in agents)
    n_unlucky = sum(a.unlucky_events for a in agents)
    hdbg.dassert_eq(
        len(log_changes), n_lucky + n_unlucky, "One history entry per event"
    )
    hdbg.dassert_lte(len(gains), n_lucky, "Only lucky events raise capital")
    hdbg.dassert_lte(len(losses), n_unlucky, "Only unlucky events lower capital")
    rows: List[Dict[str, Any]] = []
    kinds = [
        ("beneficial", gains),
        ("detrimental", losses),
        ("unchanged", unchanged),
    ]
    for kind, values in kinds:
        n_events = len(values)
        if n_events > 0:
            stats = [
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
            ]
        else:
            # No event of this kind: the statistics are undefined.
            stats = [np.nan] * 4
        rows.append(
            {
                "event_kind": kind,
                "n_events": n_events,
                "mean_log_change": float(stats[0]),
                "std_log_change": float(stats[1]),
                "min_log_change": float(stats[2]),
                "max_log_change": float(stats[3]),
            }
        )
    log_change_df = pd.DataFrame(rows).set_index("event_kind")
    _LOG.debug("log_change_df.shape=%s", log_change_df.shape)
    return log_change_df


# #############################################################################
# Causal estimation helpers
# #############################################################################


def _get_causal_inputs(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the outcome, the treatment, and the confounders.

    :param df: output of `get_results_dataframe()`
    :return: tuple `(y, d, x)`
        - `y`: log of the final capital of each agent
        - `d`: realized beneficial events of each agent, including spillovers
        - `x`: talents of each agent, one column per talent
    """
    _LOG.debug("df.shape=%s", df.shape)
    capital = df["capital"].to_numpy(dtype=float)
    # The simulator floors the capital at a positive value, so the log is
    # defined without the `+ 1` shift.
    hdbg.dassert(np.all(capital > 0), "Capital must be positive to take its log")
    y = np.log(capital)
    d = df["lucky_events"].to_numpy(dtype=float)
    x = df[_TALENT_COLS].to_numpy(dtype=float)
    _LOG.debug(hprint.to_str("y.shape d.shape x.shape"))
    return y, d, x


def _get_nuisance_model(seed: int) -> sklearn.ensemble.RandomForestRegressor:
    """
    Build the model of the outcome and of the treatment given the talents.

    :param seed: seed of the random forest
    :return: unfitted random forest regressor
    """
    _LOG.debug(hprint.to_str("seed"))
    model = sklearn.ensemble.RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=seed
    )
    _LOG.debug("return=%s", model)
    return model


# #############################################################################
# Cell 4.3: Double machine learning
# #############################################################################


def cell4_3_estimate_dml_effect(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Estimate the average effect of one realized beneficial event on log-capital.

    The estimator is `econml.dml.LinearDML` with a constant effect: the talents
    are controls (`W`), and there are no effect modifiers (`X=None`). The
    standard error and the 95% interval come from the estimator's inference,
    not from the spread of fitted per-agent effects.

    :param df: output of `get_results_dataframe()`
    :param seed: seed of the cross-fitting and of the nuisance models
    :return: one row with the effect, its standard error, its 95% interval,
        its p-value, and the effect as a percent change of capital
    """
    _LOG.debug(hprint.to_str("df.shape seed"))
    y, d, x = _get_causal_inputs(df)
    dml = econml.dml.LinearDML(
        model_y=_get_nuisance_model(seed),
        model_t=_get_nuisance_model(seed),
        discrete_treatment=False,
        random_state=seed,
    )
    dml.fit(y, d, X=None, W=x)
    # Read the raw inference values: `summary_frame()` rounds them.
    inference = dml.const_marginal_effect_inference()
    effect = float(np.ravel(inference.point_estimate)[0])
    ci_lower, ci_upper = inference.conf_int(alpha=0.05)
    dml_df = pd.DataFrame(
        [
            {
                "estimator": "LinearDML",
                "effect": effect,
                "stderr": float(np.ravel(inference.stderr)[0]),
                "ci_lower": float(np.ravel(ci_lower)[0]),
                "ci_upper": float(np.ravel(ci_upper)[0]),
                "p_value": float(np.ravel(inference.pvalue())[0]),
                # A change of `effect` in log-capital multiplies the capital by
                # `exp(effect)`.
                "pct_change": float(np.expm1(effect)),
            }
        ]
    ).set_index("estimator")
    _LOG.debug("dml_df.shape=%s", dml_df.shape)
    return dml_df


# #############################################################################
# Cell 4.4: Causal forest
# #############################################################################


def cell4_4_estimate_cate(
    df: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate the effect of one realized beneficial event for each agent.

    The estimator is `econml.dml.CausalForestDML`, with the talents as effect
    modifiers (`X`).

    :param df: output of `get_results_dataframe()`
    :param seed: seed of the forest and of the nuisance models
    :return: tuple `(cate_df, summary_df)`
        - `cate_df`: one row per agent, with its `id`, its talents, its
          conditional effect `cate`, and its 95% interval
        - `summary_df`: one row with the average effect over the agents, its
          standard error and 95% interval, the std of the per-agent effects,
          the mean width of their intervals, and the share of agents whose
          interval excludes the average effect
          - These describe how much the fitted effects vary relative to their
            uncertainty: they are not a formal test of heterogeneity
    """
    _LOG.debug(hprint.to_str("df.shape seed"))
    y, d, x = _get_causal_inputs(df)
    forest = econml.dml.CausalForestDML(
        model_y=_get_nuisance_model(seed),
        model_t=_get_nuisance_model(seed),
        discrete_treatment=False,
        n_estimators=100,
        max_depth=10,
        random_state=seed,
    )
    forest.fit(y, d, X=x)
    # Per-agent effects and their intervals.
    cate = forest.effect(x)
    cate_lower, cate_upper = forest.effect_interval(x, alpha=0.05)
    cate_df = df[["id"] + _TALENT_COLS].copy()
    cate_df["cate"] = cate
    cate_df["ci_lower"] = cate_lower
    cate_df["ci_upper"] = cate_upper
    # Average effect over the agents, with its own interval.
    population = forest.ate_inference(x)
    mean_lower, mean_upper = population.conf_int_mean(alpha=0.05)
    mean_cate = float(population.mean_point)
    excludes_mean = (cate_upper < mean_cate) | (cate_lower > mean_cate)
    summary_df = pd.DataFrame(
        [
            {
                "estimator": "CausalForestDML",
                "mean_cate": mean_cate,
                "stderr_mean": float(population.stderr_mean),
                "ci_lower_mean": float(mean_lower),
                "ci_upper_mean": float(mean_upper),
                "std_cate": float(np.std(cate)),
                "mean_ci_width": float(np.mean(cate_upper - cate_lower)),
                "share_ci_excludes_mean": float(np.mean(excludes_mean)),
            }
        ]
    ).set_index("estimator")
    _LOG.debug(hprint.to_str("cate_df.shape summary_df.shape"))
    return cate_df, summary_df


# #############################################################################
# Cell 4.5: Conditional effects
# #############################################################################


def cell4_5_plot_cate(
    cate_df: pd.DataFrame,
    dml_df: pd.DataFrame,
    setup_txt: str,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the per-agent effects with their intervals, sorted and against IQ.

    The error bars are precomputed intervals, so the plot uses `matplotlib`
    `errorbar()`, which `seaborn` cannot draw directly.

    :param cate_df: first output of `cell4_4_estimate_cate()`
    :param dml_df: output of `cell4_3_estimate_dml_effect()`, drawn as a
        reference band
    :param setup_txt: set-up of the experiment, shown in the title
    :param figsize: figure size
        - Default: `plt.rcParams["figure.figsize"]`
    """
    _LOG.debug(hprint.to_str("cate_df.shape setup_txt figsize"))
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    dml_row = dml_df.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    panels = [
        (
            cate_df.sort_values("cate").reset_index(drop=True),
            None,
            "Agents sorted by effect",
            axes[0],
        ),
        (cate_df, "talent_iq", "IQ", axes[1]),
    ]
    for plot_df, x_col, x_label, ax in panels:
        x_values = plot_df.index if x_col is None else plot_df[x_col]
        err = [
            plot_df["cate"] - plot_df["ci_lower"],
            plot_df["ci_upper"] - plot_df["cate"],
        ]
        ax.errorbar(
            x_values,
            plot_df["cate"],
            yerr=err,
            fmt="o",
            markersize=3,
            alpha=0.6,
            label="causal forest, 95% CI",
        )
        # Draw the DML effect and its interval as the reference.
        ax.axhspan(
            dml_row["ci_lower"], dml_row["ci_upper"], color="gray", alpha=0.2
        )
        ax.axhline(
            dml_row["effect"],
            color="black",
            linestyle="--",
            label=f"DML {dml_row['effect']:.3f}, 95% CI",
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Effect on log(capital)")
        ax.legend()
    axes[0].set_title("Per-agent effects with their intervals")
    axes[1].set_title("Per-agent effects against IQ")
    fig.suptitle(f"Effect of one realized beneficial event ({setup_txt})")
    fig.tight_layout()
    plt.show()


# #############################################################################
# Cell 5.2: Policy comparison
# #############################################################################


def _get_starting_capitals(agents: List[rcaoasalcsu.Agent]) -> np.ndarray:
    """
    Get the capital each agent actually started the simulation with.

    The configured `initial_capital` is not always the starting capital:
    - The simulator raises an `initial_capital` below `capital_floor` to the
      floor
    - A policy adds its allocation before the first event
    Both are in the first entry of the capital history.

    :param agents: agents after the simulation
    :return: starting capital of each agent, in the order of `agents`
    """
    _LOG.debug("len(agents)=%s", len(agents))
    starting_capitals = np.array(
        [a.capital_history[0] for a in agents], dtype=float
    )
    _LOG.debug("starting_capitals.shape=%s", starting_capitals.shape)
    return starting_capitals


def cell5_2_run_policy_comparison(
    config: Dict[str, Any], cate_values: np.ndarray
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Run the baseline config once per policy, and summarize each run.

    Each run is `run_experiment()` on a copy of `config` that changes only the
    policy name, so all the runs share the population, the seed, and the
    budget `config["policy"]["resource_amount"]`.

    The total initial capital is the sum of the actual starting capitals, see
    `_get_starting_capitals()`, and not `n_agents * initial_capital + budget`:
    the two differ when `initial_capital` is below `capital_floor`.

    :param config: baseline config, with no policy
    :param cate_values: per-agent effects used by the `cate_optimal` policy
    :return: tuple `(policy_df, policy_results)`
        - `policy_df`: one row per policy with the total initial and final
          capital, their ratio, the mean and median capital, the Gini, and the
          top 10% and bottom 50% shares
        - `policy_results`: output of `run_experiment()` of each policy
    """
    _LOG.debug("config['policy']=%s", config["policy"])
    hdbg.dassert_is(
        config["policy"]["name"], None, "The baseline config has no policy"
    )
    n_agents = config["population"]["n_agents"]
    hdbg.dassert_eq(len(cate_values), n_agents, "One CATE value per agent")
    rows: List[Dict[str, Any]] = []
    policy_results: Dict[str, Dict[str, Any]] = {}
    # Policies compared, in display order: `none` runs the baseline without any
    # allocation, and the others are the policies of `run_policy_simulation()`.
    POLICY_NAMES = (
        "none",
        "egalitarian",
        "meritocratic",
        "performance",
        "random",
        "cate_optimal",
    )
    for name in POLICY_NAMES:
        # Change only the policy, so that the runs differ only by the
        # allocation.
        policy_config = copy.deepcopy(config)
        if name != "none":
            policy_config["policy"]["name"] = name
        # Only the `cate_optimal` policy reads the CATE values.
        policy_cate = cate_values if name == "cate_optimal" else None
        results = rcaoasalcsu.run_experiment(
            policy_config, cate_values=policy_cate
        )
        stats = results["stats"]
        df = results["df"]
        # Use the capital the agents started with, after the floor and the
        # allocation.
        total_initial = float(np.sum(_get_starting_capitals(results["agents"])))
        total_final = float(df["capital"].sum())
        rows.append(
            {
                "policy": name,
                "total_initial_capital": total_initial,
                "total_final_capital": total_final,
                "final_to_initial_ratio": total_final / total_initial,
                "mean_capital": stats["mean_capital"],
                "median_capital": stats["median_capital"],
                "gini": stats["gini_coefficient"],
                "top_10_pct_share": stats["top_10_pct_share"],
                "bottom_50_pct_share": stats["bottom_50_pct_share"],
            }
        )
        policy_results[name] = results
    policy_df = pd.DataFrame(rows).set_index("policy")
    _LOG.debug("policy_df.shape=%s", policy_df.shape)
    return policy_df, policy_results


# #############################################################################
# Cell 5.3: Policies rescale each agent's luck while the floor never binds
# #############################################################################


def _count_agents_at_floor(agents: List[rcaoasalcsu.Agent]) -> int:
    """
    Count the agents whose capital reached the floor after an event.

    The simulator sets the capital to `max(capital_floor, capital * (1 -
    impact))`, so a bound floor leaves the capital exactly at the floor.

    :param agents: agents after the simulation
    :return: number of agents with a capital at or below the floor after at
        least one event
    """
    _LOG.debug("len(agents)=%s", len(agents))
    n_at_floor = sum(
        bool(np.any(np.asarray(a.capital_history[1:]) <= a.capital_floor))
        for a in agents
    )
    _LOG.debug("n_at_floor=%s", n_at_floor)
    return n_at_floor


def cell5_3_compute_implied_initial_capital(
    policy_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Compute the initial capital that each policy run implies for each agent.

    Without a policy, agent `i` starts with `S_i` and ends with `S_i * M_i`,
    where `M_i` is the product of the multipliers of its events. If a policy
    run has the same events, the agent ends with `C_i0 * M_i`, so its starting
    capital is `C_i0 = final capital / M_i`.

    `S_i` is the actual starting capital of the run without policy, see
    `_get_starting_capitals()`, and not the configured `initial_capital`: the
    simulator raises an `initial_capital` below `capital_floor` to the floor.
    Starting at the floor does not break the rescaling, since the start is
    not an event.

    The rescaling holds only if the capital floor never binds on an event:
    - A bound floor replaces the multiplier of an event with a value that
      depends on the capital, so the same events give different `M_i`
    - E.g., an agent that starts poorer hits the floor sooner
    - So the implied values are NaN for a policy whose run, or the run without
      policy, has an agent at the floor after an event

    :param policy_results: second output of `cell5_2_run_policy_comparison()`
    :return: one row per policy with:
        - `same_event_counts`: whether every agent has the same lucky and
          unlucky event counts as without a policy
        - `n_agents_at_floor`: number of agents of the run that reached the
          capital floor after an event
        - `rescaling_holds`: whether the same events and no bound floor, in
          this run and in the run without policy, make the inference valid
        - `sum_actual_initial`: sum of the actual starting capitals of the run
        - The min, mean, max, and sum of the implied initial capital, NaN when
          `rescaling_holds` is False
    """
    _LOG.debug("policy_results.keys()=%s", list(policy_results.keys()))
    hdbg.dassert_in("none", policy_results, "The run without policy is needed")
    baseline_df = policy_results["none"]["df"]
    baseline_agents = policy_results["none"]["agents"]
    baseline_at_floor = _count_agents_at_floor(baseline_agents)
    baseline_start = _get_starting_capitals(baseline_agents)
    _LOG.debug("baseline_start[:5]=%s", baseline_start[:5])
    multipliers = baseline_df["capital"].to_numpy(dtype=float) / baseline_start
    rows: List[Dict[str, Any]] = []
    for name, results in policy_results.items():
        df = results["df"]
        hdbg.dassert_eq(
            df["id"].tolist(), baseline_df["id"].tolist(), "Same agents order"
        )
        same_counts = bool(
            df["lucky_events"].equals(baseline_df["lucky_events"])
            and df["unlucky_events"].equals(baseline_df["unlucky_events"])
        )
        n_at_floor = _count_agents_at_floor(results["agents"])
        rescaling_holds = bool(
            same_counts and n_at_floor == 0 and baseline_at_floor == 0
        )
        if rescaling_holds:
            implied = df["capital"].to_numpy(dtype=float) / multipliers
            implied_stats = [
                np.min(implied),
                np.mean(implied),
                np.max(implied),
                np.sum(implied),
            ]
        else:
            # The final capital is not `C_i0 * M_i`: do not infer `C_i0`.
            implied_stats = [np.nan] * 4
        rows.append(
            {
                "policy": name,
                "same_event_counts": same_counts,
                "n_agents_at_floor": n_at_floor,
                "rescaling_holds": rescaling_holds,
                "sum_actual_initial": float(
                    np.sum(_get_starting_capitals(results["agents"]))
                ),
                "min_implied_initial": float(implied_stats[0]),
                "mean_implied_initial": float(implied_stats[1]),
                "max_implied_initial": float(implied_stats[2]),
                "sum_implied_initial": float(implied_stats[3]),
            }
        )
    implied_df = pd.DataFrame(rows).set_index("policy")
    _LOG.debug("implied_df.shape=%s", implied_df.shape)
    return implied_df


# #############################################################################
# Cell 6.1: Baseline metrics over seeds
# #############################################################################


def _compute_comparison_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute the metrics that the paper reports, for one simulation run.

    :param results: output of `run_experiment()`
    :return: metric name to value, e.g., `{"gini": 0.217, ...}`
    """
    _LOG.debug("results['df'].shape=%s", results["df"].shape)
    df = results["df"]
    stats = results["stats"]
    log_capital = np.log(df["capital"].to_numpy(dtype=float))
    n_top = max(1, len(df) // 10)
    metrics = {
        # Mean of the sample std of the 3 talents.
        "talent_std": float(df[_TALENT_COLS].std().mean()),
        "gini": stats["gini_coefficient"],
        "top_10_pct_share": stats["top_10_pct_share"],
        "bottom_50_pct_share": stats["bottom_50_pct_share"],
        "mean_lucky_events": stats["mean_lucky_events"],
        "top_decile_mean_lucky_events": float(
            df.nlargest(n_top, "capital")["lucky_events"].mean()
        ),
        "r_talent_norm_log_capital": float(
            np.corrcoef(df["talent_norm"], log_capital)[0, 1]
        ),
        "r_lucky_events_log_capital": float(
            np.corrcoef(df["lucky_events"], log_capital)[0, 1]
        ),
        "r_net_events_log_capital": float(
            np.corrcoef(df["net_events"], log_capital)[0, 1]
        ),
    }
    _LOG.debug("metrics.keys()=%s", list(metrics.keys()))
    return metrics


def cell6_1_run_seed_sweep(
    config: Dict[str, Any], seeds: List[int]
) -> pd.DataFrame:
    """
    Run the baseline config once per seed and compute the paper's metrics.

    Only `config["seed"]` changes between the runs.

    :param config: baseline config, with no policy
    :param seeds: seeds to run
    :return: one row per seed, indexed by the seed, with one column per metric
        of `_compute_comparison_metrics()`
    """
    _LOG.debug(hprint.to_str("seeds"))
    hdbg.dassert_is(
        config["policy"]["name"], None, "The baseline config has no policy"
    )
    hdbg.dassert_lt(0, len(seeds), "At least one seed is needed")
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_config = copy.deepcopy(config)
        seed_config["seed"] = seed
        results = rcaoasalcsu.run_experiment(seed_config)
        row: Dict[str, Any] = {"seed": seed}
        row.update(_compute_comparison_metrics(results))
        rows.append(row)
    sweep_df = pd.DataFrame(rows).set_index("seed")
    _LOG.debug("sweep_df.shape=%s", sweep_df.shape)
    return sweep_df


# #############################################################################
# Cell 6.2: Comparison with the paper
# #############################################################################


def cell6_2_compare_with_paper(
    baseline_results: Dict[str, Any],
    sweep_df: pd.DataFrame,
    dml_effect: float,
    cate_std: float,
) -> pd.DataFrame:
    """
    Put the paper's values next to the baseline run and the seed average.

    The DML effect and the CATE std are estimated on the baseline run only, so
    their seed columns are NaN.

    :param baseline_results: output of `run_experiment()` for the baseline
    :param sweep_df: output of `cell6_1_run_seed_sweep()`
    :param dml_effect: effect of `cell4_3_estimate_dml_effect()` on the baseline
    :param cate_std: `std_cate` of `cell4_4_estimate_cate()` on the baseline
    :return: one row per metric with the columns `paper`, `paper_basis`,
        `baseline`, `seeds_mean`, and `seeds_std`
    """
    _LOG.debug(hprint.to_str("sweep_df.shape dml_effect cate_std"))
    baseline = _compute_comparison_metrics(baseline_results)
    baseline["dml_effect"] = dml_effect
    baseline["cate_std"] = cate_std
    rows: List[Dict[str, Any]] = []
    # Values reported in
    # `papers/Causal_Analysis_of_Agent_Skill_And_Luck/03_experimental_evaluation.tex`,
    # with how the paper says each one was computed.
    PAPER_VALUES = {
        # The set-up of the paper draws each talent with std 0.15, while its
        # results report 0.11 without saying which talent measure it is.
        "talent_std": (0.11, "results section; set-up says 0.15"),
        "gini": (0.38, "average over runs, number of runs not stated"),
        "top_10_pct_share": (0.28, "not stated"),
        "bottom_50_pct_share": (0.25, "not stated"),
        "mean_lucky_events": (4.8, "not stated"),
        "top_decile_mean_lucky_events": (8.3, "not stated"),
        "r_talent_norm_log_capital": (0.12, "not stated"),
        "r_lucky_events_log_capital": (0.78, "not stated"),
        "r_net_events_log_capital": (0.82, "not stated"),
        "dml_effect": (0.124, "not stated"),
        "cate_std": (0.03, "not stated"),
    }
    for metric, (paper_value, paper_basis) in PAPER_VALUES.items():
        if metric in sweep_df.columns:
            seeds_mean = float(sweep_df[metric].mean())
            seeds_std = float(sweep_df[metric].std())
        else:
            seeds_mean = np.nan
            seeds_std = np.nan
        rows.append(
            {
                "metric": metric,
                "paper": paper_value,
                "paper_basis": paper_basis,
                "baseline": baseline[metric],
                "seeds_mean": seeds_mean,
                "seeds_std": seeds_std,
            }
        )
    comparison_df = pd.DataFrame(rows).set_index("metric")
    _LOG.debug("comparison_df.shape=%s", comparison_df.shape)
    return comparison_df


# #############################################################################
# Cell 6.3: Seed variability
# #############################################################################


def cell6_3_plot_seed_sweep(
    sweep_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    setup_txt: str,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the Gini and the luck correlation over seeds, against the paper.

    :param sweep_df: output of `cell6_1_run_seed_sweep()`
    :param comparison_df: output of `cell6_2_compare_with_paper()`
    :param setup_txt: set-up of the experiment, shown in the title
    :param figsize: figure size
        - Default: `plt.rcParams["figure.figsize"]`
    """
    _LOG.debug(hprint.to_str("sweep_df.shape setup_txt figsize"))
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    panels = [
        ("gini", "Gini of final capital"),
        ("r_lucky_events_log_capital", "r(lucky events, log(capital))"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, (metric, label) in zip(axes, panels):
        row = comparison_df.loc[metric]
        sns.histplot(x=sweep_df[metric], bins=15, ax=ax)
        ax.axvline(
            row["seeds_mean"],
            color="black",
            linestyle="--",
            label="mean over seeds",
        )
        ax.axvline(
            row["baseline"], color="blue", linestyle="-", label="baseline"
        )
        ax.axvline(row["paper"], color="red", linestyle=":", label="paper")
        ax.set_xlabel(label)
        # Write the values in the title, so that one legend serves both panels.
        ax.set_title(
            f"{label}\nmean {row['seeds_mean']:.3f}, "
            f"baseline {row['baseline']:.3f}, paper {row['paper']:.3f}"
        )
    # Put one legend below the panels, so that it does not hide the bars.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(
        f"Seed variability over {len(sweep_df)} seeds, baseline: {setup_txt}"
    )
    fig.tight_layout(rect=(0.0, 0.1, 1.0, 1.0))
    plt.show()
