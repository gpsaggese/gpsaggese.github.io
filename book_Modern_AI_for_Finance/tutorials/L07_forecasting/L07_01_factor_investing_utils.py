"""
Utility functions for the factor investing notebook.

Simulate a universe of securities whose returns load on factor premia with a
known mean, and run on it the two procedures used in factor research:
- Fama-MacBeth: one cross-sectional regression per period, then the mean and
  standard deviation of the estimates
- Fama-French: quantile portfolios, a long-short portfolio, and time-series
  regressions

Import as:

import L07_01_factor_investing_utils as fainvuti
"""

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import ipywidgets
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import statsmodels.api as sm
from IPython.display import clear_output, display

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import helpers.hprint as hprint
import helpers.htutorial as htutori

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Simulation of the universe
# #############################################################################

# Volatility of the per-period factor premium around its mean `mu`.
_TAU = 0.5
# AR(1) persistence of the characteristics of each security.
_PHI = 0.98
# Mean and volatility of the common market return, in percent per period.
_MARKET_MEAN = 0.5
_MARKET_VOL = 4.0
# Colors for true (theoretical) and estimated (empirical) quantities.
_TRUE_COLOR = "darkorange"
_EST_COLOR = "navy"


@dataclasses.dataclass
class Universe:
    """
    Simulated panel of securities, characteristics, and returns.

    Arrays are indexed by period `t` first, then by security `i`:
    - `chars[t, i, k]`: characteristic `k` of security `i`, known at the start
      of period `t` (i.e., observed at the end of period `t - 1`)
    - `control[t, i]`: control variable, known at the start of period `t`
    - `market[t]`: common market return in period `t`
    - `premia[t, k]`: realized premium of factor `k` in period `t`
    - `noise[t, i]`: idiosyncratic return of security `i` in period `t`
    - `returns[t, i]`: total return of security `i` in period `t`
    """

    chars: np.ndarray
    control: np.ndarray
    market: np.ndarray
    premia: np.ndarray
    noise: np.ndarray
    returns: np.ndarray


def simulate_universe(
    n_securities: int,
    n_periods: int,
    mus: List[float],
    sigma: float,
    seed: int,
    *,
    gamma: float = 0.0,
    rho: float = 0.0,
    rho_factors: float = 0.0,
) -> Universe:
    """
    Simulate returns that load on known factor premia.

    The return of security `i` in period `t` is:
    `r[t, i] = m[t] + sum_k lambda[t, k] x[t, i, k] + gamma z[t, i] + eps[t, i]`
    where:
    - `x` and `z` are persistent and standardized in each period
    - `lambda[t, k] ~ N(mus[k], _TAU^2)` is the premium of factor `k`
    - `m[t] ~ N(_MARKET_MEAN, _MARKET_VOL^2)` is the market return
    - `eps[t, i] ~ N(0, sigma^2)` is the idiosyncratic noise

    Characteristics, premia, and noise come from independent random streams,
    so changing `sigma` or `n_securities` keeps the same premium path.

    :param n_securities: number of securities `N`
    :param n_periods: number of periods `T`
    :param mus: mean premium of each factor, in percent per period
    :param sigma: idiosyncratic volatility, in percent per period
    :param seed: random seed
    :param gamma: effect of the control variable on returns
    :param rho: correlation between the first characteristic and the control
    :param rho_factors: correlation between the first two characteristics
    :return: simulated universe
    """
    _LOG.debug(
        hprint.to_str(
            "n_securities n_periods mus sigma seed gamma rho rho_factors"
        )
    )
    hdbg.dassert_lte(1, len(mus), "The model needs at least one factor")
    n_factors = len(mus)
    chars_rng, premia_rng, noise_rng = [
        np.random.default_rng(child)
        for child in np.random.SeedSequence(seed).spawn(3)
    ]
    # Correlation matrix of the variables `[x_1, ..., x_K, z]`.
    n_vars = n_factors + 1
    corr = np.eye(n_vars)
    corr[0, n_factors] = corr[n_factors, 0] = rho
    if n_factors >= 2:
        corr[0, 1] = corr[1, 0] = rho_factors
    chol = np.linalg.cholesky(corr)
    # Simulate persistent characteristics as AR(1) processes with correlated
    # shocks, starting from the stationary distribution so that no burn-in is
    # needed.
    shocks = chars_rng.standard_normal((n_periods, n_securities, n_vars))
    shocks = shocks @ chol.T
    latent = np.empty_like(shocks)
    latent[0] = shocks[0]
    shock_scale = np.sqrt(1 - _PHI**2)
    for t in range(1, n_periods):
        latent[t] = _PHI * latent[t - 1] + shock_scale * shocks[t]
    # Standardize each variable across securities in every period, as factor
    # researchers do with characteristics.
    latent = latent - latent.mean(axis=1, keepdims=True)
    latent = latent / latent.std(axis=1, keepdims=True)
    chars = latent[:, :, :n_factors]
    control = latent[:, :, n_factors]
    # Draw the premium path and the market path from their own stream.
    premia = np.asarray(mus, dtype=float) + _TAU * premia_rng.standard_normal(
        (n_periods, n_factors)
    )
    market = _MARKET_MEAN + _MARKET_VOL * premia_rng.standard_normal(n_periods)
    # Draw the idiosyncratic noise.
    noise = sigma * noise_rng.standard_normal((n_periods, n_securities))
    # Assemble the returns.
    systematic = np.einsum("tik,tk->ti", chars, premia) + gamma * control
    returns = market[:, None] + systematic + noise
    universe = Universe(
        chars=chars,
        control=control,
        market=market,
        premia=premia,
        noise=noise,
        returns=returns,
    )
    return universe


# #############################################################################
# Estimation procedures
# #############################################################################


def summarize_estimates(values: pd.Series) -> Dict[str, float]:
    """
    Summarize a series of per-period estimates as Fama-MacBeth does.

    :param values: one estimate per period
    :return: dict with `mean`, `std`, standard error `se = std / sqrt(T)`, and
        `t_stat = mean / se`
    """
    n_obs = len(values)
    hdbg.dassert_lte(2, n_obs, "A standard deviation needs two periods")
    mean = float(values.mean())
    std = float(values.std())
    se = std / np.sqrt(n_obs)
    res = {"mean": mean, "std": std, "se": se, "t_stat": mean / se}
    return res


def run_fama_macbeth(
    universe: Universe,
    *,
    use_control: bool = False,
) -> pd.DataFrame:
    """
    Run one least-squares cross-sectional regression per period.

    Each period regresses the returns of all securities on their
    characteristics (and optionally on the control) plus an intercept.

    :param universe: simulated universe
    :param use_control: whether to add the control variable `z` as regressor
    :return: one row per period with the slope on the first characteristic
        (`beta`), its standard error (`se`), and the intercept (`alpha`)
    """
    rows = []
    for t in range(universe.returns.shape[0]):
        regressors = universe.chars[t]
        if use_control:
            regressors = np.column_stack([regressors, universe.control[t]])
        res = sm.OLS(universe.returns[t], sm.add_constant(regressors)).fit()
        rows.append(
            {"alpha": res.params[0], "beta": res.params[1], "se": res.bse[1]}
        )
    df = pd.DataFrame(rows)
    df.index.name = "period"
    return df


def to_long_frame(universe: Universe) -> pd.DataFrame:
    """
    Convert a universe into one row per period and security.

    :param universe: simulated universe
    :return: dataframe with columns `period`, `security`, `x1`, ..., `z`,
        `ret`, and `excess` (return minus the universe mean of the period)
    """
    n_periods, n_securities, n_factors = universe.chars.shape
    data = {
        "period": np.repeat(np.arange(n_periods), n_securities),
        "security": np.tile(np.arange(n_securities), n_periods),
    }
    for k in range(n_factors):
        data[f"x{k + 1}"] = universe.chars[:, :, k].ravel()
    data["z"] = universe.control.ravel()
    data["ret"] = universe.returns.ravel()
    df = pd.DataFrame(data)
    # Subtracting the equal-weighted universe mean removes the common market
    # move, which hides the differences across securities.
    universe_mean = df.groupby("period")["ret"].transform("mean")
    df["excess"] = df["ret"] - universe_mean
    return df


def assign_quantiles(
    df: pd.DataFrame,
    col: str,
    by: List[str],
    q: int,
) -> pd.Series:
    """
    Assign each row to a quantile of `col` within the groups `by`.

    :param df: data with the column to rank
    :param col: column to rank, e.g., `x1`
    :param by: columns defining the groups ranked separately, e.g., `["period"]`
    :param q: number of quantiles
    :return: quantile label in `1, ..., q`, with `q` the highest values
    """
    pct = df.groupby(by)[col].rank(method="first", pct=True)
    quantile = np.ceil(pct * q).astype(int)
    return quantile


def long_short_weights(
    chars: np.ndarray,
    frac: float,
    scheme: str,
) -> np.ndarray:
    """
    Compute long-short portfolio weights from a ranking in each period.

    The top `frac` of securities forms the long leg and the bottom `frac`
    forms the short leg. The long leg sums to +1 and the short leg to -1.

    :param chars: characteristic with shape `(T, N)`
    :param frac: share of securities in each leg, in `(0, 0.5]`
    :param scheme:
        - "equal": equal weights within each leg
        - "rank": weights proportional to the distance of the rank from the
          median, so the most extreme securities weigh most
    :return: weights with shape `(T, N)`
    """
    hdbg.dassert_in(scheme, ["equal", "rank"], "Unknown weighting scheme")
    hdbg.dassert_lt(0.0, frac, "Each leg needs some securities")
    hdbg.dassert_lte(frac, 0.5, "The two legs cannot overlap")
    pct = pd.DataFrame(chars).rank(axis=1, pct=True).to_numpy()
    in_long = pct > 1 - frac
    in_short = pct <= frac
    if scheme == "equal":
        raw = in_long.astype(float) - in_short.astype(float)
    else:
        raw = (pct - 0.5) * (in_long | in_short)
    # Normalize each leg separately so that the portfolio is dollar neutral.
    long_part = np.where(raw > 0, raw, 0.0)
    short_part = np.where(raw < 0, raw, 0.0)
    weights = long_part / long_part.sum(axis=1, keepdims=True)
    weights = weights - short_part / short_part.sum(axis=1, keepdims=True)
    return weights


# #############################################################################
# Widget and panel helpers
# #############################################################################


def _build_seed_control(
    *,
    initial_value: int = 0,
) -> Tuple[Union[ipywidgets.FloatSlider, ipywidgets.IntSlider], ipywidgets.HBox]:
    """
    Build the `seed` control shared by every cell.

    :param initial_value: seed shown when the cell first renders
    :return: slider and its layout box
    """
    seed_slider, seed_box = htutori.build_widget_control(
        name="seed",
        description="Random seed",
        min_val=0,
        max_val=999,
        step=1,
        initial_value=initial_value,
        is_float=False,
    )
    return seed_slider, seed_box


def _display_interactive(
    boxes: List[ipywidgets.Widget],
    info_items: List[Tuple[str, str]],
    output: ipywidgets.Output,
) -> None:
    """
    Display controls and their descriptions on top, and the plots below.

    :param boxes: control boxes, in display order
    :param info_items: `(name, description)` pairs, one per control
    :param output: widget holding the figure
    """
    lines = [f"<b>{name}</b>: {desc}" for name, desc in info_items]
    info = ipywidgets.HTML(
        "<div style='background:#f5f5f5; padding:10px 14px; "
        "border-radius:4px; border-left:3px solid #4682b4; font-size:13px; "
        "line-height:1.8'>" + "<br>".join(lines) + "</div>"
    )
    top_row = ipywidgets.HBox([ipywidgets.VBox(boxes), info])
    display(ipywidgets.VBox([top_row, output]))


def _fill_comments_panel(ax: matplotlib.axes.Axes, text: str) -> None:
    """
    Turn an axis into the `Comments` panel with the current state.

    :param ax: axis to fill
    :param text: parameter values and statistics to show
    """
    ax.axis("off")
    ax.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    htutori.add_fitted_text_box(ax, text, max_fontsize=12, min_fontsize=8)


def _observe_all(widgets: List[ipywidgets.Widget], callback: Any) -> None:
    """
    Call `callback` whenever any widget changes value.

    :param widgets: widgets to observe
    :param callback: function receiving the change
    """
    for widget in widgets:
        widget.observe(callback, names="value")


# #############################################################################
# Cell 1.1: Cross-Section Versus Time-Series Variation
# #############################################################################


def cell1_1_cross_section_vs_time_series(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Compare the cross-section of average returns with the time series.

    :param figsize: figure size, defaulting to `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    n_periods = 120
    mu_slider, mu_box = htutori.build_widget_control(
        name="mu",
        description="True premium",
        min_val=0.0,
        max_val=1.0,
        step=0.1,
        initial_value=0.5,
        is_float=True,
    )
    sigma_slider, sigma_box = htutori.build_widget_control(
        name="sigma",
        description="Idiosyncratic volatility",
        min_val=1.0,
        max_val=10.0,
        step=1.0,
        initial_value=5.0,
        is_float=True,
    )
    n_slider, n_box = htutori.build_log_widget_control(
        name="log2(N)",
        description="Number of securities",
        min_exp=5,
        max_exp=10,
        initial_exp=8,
        base=2,
    )
    seed_slider, seed_box = _build_seed_control()
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the panels for the current widget values.

        :param change: widget change information (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            mu = mu_slider.value
            sigma = sigma_slider.value
            n_securities = 2**n_slider.value
            seed = seed_slider.value
            universe = simulate_universe(
                n_securities, n_periods, [mu], sigma, seed
            )
            # Average each security over time: the cross-section to explain.
            avg_char = universe.chars[:, :, 0].mean(axis=0)
            avg_ret = universe.returns.mean(axis=0)
            slope, intercept = np.polyfit(avg_char, avg_ret, 1)
            r_squared = np.corrcoef(avg_char, avg_ret)[0, 1] ** 2
            # Average each period over securities: the common time movement.
            period_mean = universe.returns.mean(axis=1)
            # Split the return variance into its three true components.
            systematic = universe.returns - universe.market[:, None]
            systematic = systematic - universe.noise
            variances = pd.Series(
                {
                    "common time effect": universe.market.var(),
                    "characteristic": systematic.var(),
                    "noise": universe.noise.var(),
                }
            )
            shares = variances / variances.sum()
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: cross-section of average returns.
            sns.scatterplot(
                x=avg_char, y=avg_ret, ax=ax1, s=15, alpha=0.6, color="steelblue"
            )
            grid = np.linspace(avg_char.min(), avg_char.max(), 50)
            ax1.plot(
                grid,
                universe.market.mean() + mu * grid,
                color=_TRUE_COLOR,
                linestyle=":",
                linewidth=2,
                label=f"true slope = {mu:.2f}",
            )
            ax1.plot(
                grid,
                intercept + slope * grid,
                color=_EST_COLOR,
                linewidth=2,
                label=f"fitted slope = {slope:.2f}",
            )
            ax1.set_xlabel("time-averaged characteristic")
            ax1.set_ylabel("time-averaged return (%)")
            ax1.set_title("Cross-section", fontsize=14, fontweight="bold")
            ax1.legend(fontsize=10)
            # Panel 2: time series of the cross-sectional mean return.
            ax2.plot(period_mean, color="gray", linewidth=1.5)
            ax2.axhline(period_mean.mean(), color=_EST_COLOR, linewidth=1.5)
            ax2.set_xlabel("period")
            ax2.set_ylabel("cross-sectional mean return (%)")
            ax2.set_title("Time series", fontsize=14, fontweight="bold")
            # Panel 3: stacked bar of variance shares.
            colors = ["gray", "seagreen", "lightsteelblue"]
            bottom = 0.0
            for (name, share), color in zip(shares.items(), colors):
                ax3.bar(0, share, bottom=bottom, color=color, label=name)
                # Put the label of a thin segment beside the bar, so that it
                # does not overlap the neighboring segments.
                is_thin = share < 0.05
                ax3.text(
                    0.45 if is_thin else 0,
                    bottom + share / 2,
                    f"{share:.1%}",
                    ha="left" if is_thin else "center",
                    va="center",
                )
                bottom += share
            ax3.set_xlim(-1, 1)
            ax3.set_xticks([])
            ax3.set_ylabel("share of return variance")
            ax3.set_title("Variance split", fontsize=14, fontweight="bold")
            ax3.legend(fontsize=10, loc="upper right")
            # Panel 4: comments.
            comment_text = (
                f"Parameters:\n"
                f"  mu: {mu:.2f}\n"
                f"  sigma: {sigma:.1f}\n"
                f"  N: {n_securities}\n"
                f"  T: {n_periods}\n"
                f"  seed: {seed}\n\n"
                f"Cross-section:\n"
                f"  fitted slope: {slope:.3f}\n"
                f"  R^2: {r_squared:.3f}\n\n"
                f"Variance shares:\n"
                + "\n".join(
                    f"  {name}: {share:.1%}" for name, share in shares.items()
                )
            )
            _fill_comments_panel(ax4, comment_text)
            plt.tight_layout()
            plt.show()

    _observe_all([mu_slider, sigma_slider, n_slider, seed_slider], update_plot)
    update_plot()
    info_items = [
        ("mu", "true premium per unit of characteristic, percent per period"),
        ("sigma", "idiosyncratic volatility, percent per period"),
        ("log2(N)", "number of securities N, on a log scale"),
        ("seed", "random seed, redraws the whole universe"),
    ]
    _display_interactive(
        [mu_box, sigma_box, n_box, seed_box], info_items, output
    )


# #############################################################################
# Cell 1.2: Anatomy of Returns, Factors, and Controls
# #############################################################################


def cell1_2_returns_factors_controls(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show the returns `Y`, factors `X`, and controls `Z` of one period.

    :param figsize: figure size, defaulting to `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    # Illustrative parameters: a low noise so that the pattern is visible.
    n_periods = 120
    all_mus = [1.0, 0.5, -0.5]
    sigma = 1.0
    gamma = 0.5
    rho = 0.3
    t_slider, t_box = htutori.build_widget_control(
        name="t",
        description="Period shown",
        min_val=1,
        max_val=n_periods,
        step=1,
        initial_value=1,
        is_float=False,
    )
    k_slider, k_box = htutori.build_widget_control(
        name="K",
        description="Number of factors",
        min_val=1,
        max_val=3,
        step=1,
        initial_value=2,
        is_float=False,
    )
    n_slider, n_box = htutori.build_widget_control(
        name="N",
        description="Number of securities shown",
        min_val=8,
        max_val=64,
        step=8,
        initial_value=32,
        is_float=False,
    )
    sort_checkbox = ipywidgets.Checkbox(
        value=False, description="sort: order rows by x1"
    )
    seed_slider, seed_box = _build_seed_control()
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the panels for the current widget values.

        :param change: widget change information (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            t = t_slider.value
            n_factors = k_slider.value
            n_securities = n_slider.value
            sort_rows = sort_checkbox.value
            seed = seed_slider.value
            mus = all_mus[:n_factors]
            universe = simulate_universe(
                n_securities, n_periods, mus, sigma, seed, gamma=gamma, rho=rho
            )
            # Extract the three matrices of period `t`.
            factor_cols = [f"x{k + 1}" for k in range(n_factors)]
            y_df = pd.DataFrame({"Y": universe.returns[t - 1]})
            x_df = pd.DataFrame(universe.chars[t - 1], columns=factor_cols)
            z_df = pd.DataFrame({"z": universe.control[t - 1]})
            if sort_rows:
                # Highest `x1` on top, so a premium shows as a vertical shade.
                order = np.argsort(-x_df["x1"].to_numpy())
                y_df = y_df.iloc[order]
                x_df = x_df.iloc[order]
                z_df = z_df.iloc[order]
            corrs = x_df.corrwith(y_df["Y"])
            _, (ax1, ax2, ax3, ax4) = plt.subplots(
                1,
                4,
                figsize=figsize,
                gridspec_kw={"width_ratios": [1, 1.5, 1, 2.5]},
            )
            # Panels 1-3: heat strips of `Y`, `X`, and `Z`.
            panels = [
                (ax1, y_df, "Y", y_df["Y"].mean(), "return (%)"),
                (ax2, x_df, "X", 0.0, "standardized value"),
                (ax3, z_df, "Z", 0.0, "standardized value"),
            ]
            for ax, df, title, center, label in panels:
                sns.heatmap(
                    df,
                    ax=ax,
                    cmap="coolwarm",
                    center=center,
                    yticklabels=False,
                    cbar_kws={"label": label},
                )
                ax.set_title(title, fontsize=14, fontweight="bold")
            ax1.set_ylabel("security")
            # Panel 4: comments.
            comment_text = (
                f"Parameters:\n"
                f"  t: {t}\n"
                f"  K: {n_factors}\n"
                f"  N: {n_securities}\n"
                f"  sort: {sort_rows}\n"
                f"  seed: {seed}\n\n"
                f"True model:\n"
                + "\n".join(
                    f"  mu_{k + 1}: {mu:+.2f}" for k, mu in enumerate(mus)
                )
                + f"\n  gamma: {gamma:.2f}\n  sigma: {sigma:.1f}\n\n"
                f"corr(X column, Y) in period t:\n"
                + "\n".join(f"  {col}: {val:+.2f}" for col, val in corrs.items())
            )
            _fill_comments_panel(ax4, comment_text)
            plt.tight_layout()
            plt.show()

    _observe_all(
        [t_slider, k_slider, n_slider, sort_checkbox, seed_slider], update_plot
    )
    update_plot()
    info_items = [
        ("t", "period shown, from 1 to T"),
        ("K", "number of factors, the columns of X"),
        ("N", "number of securities shown, the rows"),
        ("sort", "order the rows by the first characteristic x1"),
        ("seed", "random seed, redraws the whole universe"),
    ]
    _display_interactive(
        [t_box, k_box, n_box, sort_checkbox, seed_box], info_items, output
    )


# #############################################################################
# Cell 2.1: One-Period Cross-Sectional Regression
# #############################################################################


def cell2_1_one_period_regression(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Regress the returns of one period on the characteristic.

    :param figsize: figure size, defaulting to `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    n_periods = 120
    mu = 0.5
    t_slider, t_box = htutori.build_widget_control(
        name="t",
        description="Period regressed",
        min_val=1,
        max_val=n_periods,
        step=1,
        initial_value=1,
        is_float=False,
    )
    n_slider, n_box = htutori.build_log_widget_control(
        name="log2(N)",
        description="Number of securities",
        min_exp=5,
        max_exp=10,
        initial_exp=7,
        base=2,
    )
    sigma_slider, sigma_box = htutori.build_widget_control(
        name="sigma",
        description="Idiosyncratic volatility",
        min_val=1.0,
        max_val=10.0,
        step=1.0,
        initial_value=5.0,
        is_float=True,
    )
    seed_slider, seed_box = _build_seed_control()
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the panels for the current widget values.

        :param change: widget change information (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            t = t_slider.value
            n_securities = 2**n_slider.value
            sigma = sigma_slider.value
            seed = seed_slider.value
            universe = simulate_universe(
                n_securities, n_periods, [mu], sigma, seed
            )
            # Fit the cross-sectional regression of period `t`.
            x = universe.chars[t - 1, :, 0]
            y = universe.returns[t - 1]
            res = sm.OLS(y, sm.add_constant(x)).fit()
            alpha_hat, beta_hat = res.params
            beta_se = res.bse[1]
            ci_low, ci_high = res.conf_int()[1]
            true_lambda = universe.premia[t - 1, 0]
            true_market = universe.market[t - 1]
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: scatter with fitted, true, and reference lines.
            sns.scatterplot(x=x, y=y, ax=ax1, s=15, alpha=0.6, color="steelblue")
            grid = np.linspace(x.min(), x.max(), 50)
            ax1.plot(
                grid,
                true_market + mu * grid,
                color="gray",
                linestyle="--",
                linewidth=1.5,
                label=f"mean premium mu = {mu:.2f}",
            )
            ax1.plot(
                grid,
                true_market + true_lambda * grid,
                color=_TRUE_COLOR,
                linestyle=":",
                linewidth=2.5,
                label=f"true lambda_t = {true_lambda:.2f}",
            )
            ax1.plot(
                grid,
                alpha_hat + beta_hat * grid,
                color=_EST_COLOR,
                linewidth=2,
                label=f"fitted beta_t = {beta_hat:.2f}",
            )
            ax1.set_xlabel("characteristic x")
            ax1.set_ylabel("return (%)")
            ax1.set_title("Scatter", fontsize=14, fontweight="bold")
            ax1.legend(fontsize=9)
            # Panel 2: estimate with its interval against the truth.
            ax2.bar(0, beta_hat, color="lightsteelblue", width=0.5)
            ax2.errorbar(
                0,
                beta_hat,
                yerr=[[beta_hat - ci_low], [ci_high - beta_hat]],
                color=_EST_COLOR,
                capsize=10,
                linewidth=2,
                label="estimate, 95% interval",
            )
            ax2.scatter(
                0,
                true_lambda,
                color=_TRUE_COLOR,
                s=120,
                zorder=3,
                label="true lambda_t",
            )
            ax2.axhline(
                mu, color="gray", linestyle="--", label="mean premium mu"
            )
            ax2.axhline(0, color="black", linewidth=0.8)
            ax2.set_xlim(-1, 1)
            ax2.set_xticks([0])
            ax2.set_xticklabels(["beta_t"])
            ax2.set_ylabel("slope (% per unit of x)")
            ax2.set_title("Estimates", fontsize=14, fontweight="bold")
            ax2.legend(fontsize=9)
            # Panel 3: residuals.
            sns.histplot(res.resid, ax=ax3, bins=30, color="steelblue")
            ax3.set_xlabel("residual (%)")
            ax3.set_title("Residuals", fontsize=14, fontweight="bold")
            # Panel 4: comments.
            comment_text = (
                f"Parameters:\n"
                f"  t: {t}\n"
                f"  N: {n_securities}\n"
                f"  sigma: {sigma:.1f}\n"
                f"  seed: {seed}\n\n"
                f"Estimates:\n"
                f"  beta_t: {beta_hat:.3f}\n"
                f"  se: {beta_se:.3f}\n"
                f"  95% CI: [{ci_low:.2f}, {ci_high:.2f}]\n"
                f"  R^2: {res.rsquared:.3f}\n"
                f"  intercept: {alpha_hat:.2f}\n\n"
                f"Truth:\n"
                f"  lambda_t: {true_lambda:.3f}\n"
                f"  mu: {mu:.2f}\n"
                f"  market m_t: {true_market:.2f}"
            )
            _fill_comments_panel(ax4, comment_text)
            plt.tight_layout()
            plt.show()

    _observe_all([t_slider, n_slider, sigma_slider, seed_slider], update_plot)
    update_plot()
    info_items = [
        ("t", "period whose cross-section is regressed"),
        ("log2(N)", "number of securities N, on a log scale"),
        ("sigma", "idiosyncratic volatility, percent per period"),
        ("seed", "random seed, redraws the whole universe"),
    ]
    _display_interactive([t_box, n_box, sigma_box, seed_box], info_items, output)


# #############################################################################
# Cell 2.2: Averaging Betas Across Periods
# #############################################################################


def cell2_2_fama_macbeth(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Run the Fama-MacBeth procedure and test the mean premium.

    :param figsize: figure size, defaulting to `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    n_securities = 128
    t_slider, t_box = htutori.build_log_widget_control(
        name="log2(T)",
        description="Number of periods",
        min_exp=4,
        max_exp=9,
        initial_exp=7,
        base=2,
    )
    mu_slider, mu_box = htutori.build_widget_control(
        name="mu",
        description="True premium",
        min_val=0.0,
        max_val=1.0,
        step=0.1,
        initial_value=0.5,
        is_float=True,
    )
    sigma_slider, sigma_box = htutori.build_widget_control(
        name="sigma",
        description="Idiosyncratic volatility",
        min_val=1.0,
        max_val=10.0,
        step=1.0,
        initial_value=5.0,
        is_float=True,
    )
    seed_slider, seed_box = _build_seed_control()
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the panels for the current widget values.

        :param change: widget change information (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            n_periods = 2**t_slider.value
            mu = mu_slider.value
            sigma = sigma_slider.value
            seed = seed_slider.value
            universe = simulate_universe(
                n_securities, n_periods, [mu], sigma, seed
            )
            betas = run_fama_macbeth(universe)["beta"]
            stats = summarize_estimates(betas)
            # Running mean and its two-standard-error band.
            n_obs = np.arange(1, n_periods + 1)
            running_mean = betas.expanding().mean()
            running_se = betas.expanding(min_periods=2).std() / np.sqrt(n_obs)
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: path of the per-period estimates.
            ax1.plot(betas.index, betas, color="steelblue", linewidth=1)
            ax1.axhline(
                stats["mean"],
                color=_EST_COLOR,
                linewidth=2,
                label=f"sample mean = {stats['mean']:.2f}",
            )
            ax1.axhline(
                mu,
                color=_TRUE_COLOR,
                linestyle=":",
                linewidth=2.5,
                label=f"true mu = {mu:.2f}",
            )
            ax1.set_xlabel("period t")
            ax1.set_ylabel("beta_t (% per unit of x)")
            ax1.set_title("Beta path", fontsize=14, fontweight="bold")
            ax1.legend(fontsize=9)
            # Panel 2: distribution of the per-period estimates.
            sns.histplot(betas, ax=ax2, bins=25, color="steelblue")
            ax2.axvline(
                stats["mean"], color=_EST_COLOR, linewidth=2, label="mean"
            )
            for sign in [-1, 1]:
                ax2.axvline(
                    stats["mean"] + sign * stats["std"],
                    color=_EST_COLOR,
                    linestyle="--",
                    linewidth=1,
                    label="mean +/- std" if sign == 1 else None,
                )
            ax2.axvline(mu, color=_TRUE_COLOR, linestyle=":", linewidth=2.5)
            ax2.set_xlabel("beta_t")
            ax2.set_title("Histogram", fontsize=14, fontweight="bold")
            ax2.legend(fontsize=9)
            # Panel 3: running mean with a band of two standard errors.
            ax3.plot(n_obs, running_mean, color=_EST_COLOR, linewidth=2)
            ax3.fill_between(
                n_obs,
                running_mean - 2 * running_se,
                running_mean + 2 * running_se,
                color=_EST_COLOR,
                alpha=0.2,
                label="+/- 2 se",
            )
            ax3.axhline(mu, color=_TRUE_COLOR, linestyle=":", linewidth=2.5)
            ax3.axhline(0, color="black", linewidth=0.8)
            ax3.set_xlabel("periods averaged")
            ax3.set_ylabel("running mean of beta_t")
            ax3.set_title("Running mean", fontsize=14, fontweight="bold")
            ax3.legend(fontsize=9)
            # Panel 4: comments.
            comment_text = (
                f"Parameters:\n"
                f"  T: {n_periods}\n"
                f"  mu: {mu:.2f}\n"
                f"  sigma: {sigma:.1f}\n"
                f"  N: {n_securities}\n"
                f"  seed: {seed}\n\n"
                f"Fama-MacBeth:\n"
                f"  mean beta_t: {stats['mean']:.3f}\n"
                f"  std beta_t: {stats['std']:.3f}\n"
                f"  se = std / sqrt(T): {stats['se']:.3f}\n"
                f"  t-stat: {stats['t_stat']:.2f}\n"
                f"  |t| > 2: {abs(stats['t_stat']) > 2}"
            )
            _fill_comments_panel(ax4, comment_text)
            plt.tight_layout()
            plt.show()

    _observe_all([t_slider, mu_slider, sigma_slider, seed_slider], update_plot)
    update_plot()
    info_items = [
        ("log2(T)", "number of periods T, on a log scale"),
        ("mu", "true mean premium, percent per period"),
        ("sigma", "idiosyncratic volatility, percent per period"),
        ("seed", "random seed, redraws the whole universe"),
    ]
    _display_interactive(
        [t_box, mu_box, sigma_box, seed_box], info_items, output
    )


# #############################################################################
# Cell 2.3: Controls Remove Omitted-Variable Bias
# #############################################################################


def cell2_3_controls_and_bias(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Compare Fama-MacBeth estimates with and without the control variable.

    :param figsize: figure size, defaulting to `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    n_securities = 128
    n_periods = 120
    sigma = 5.0
    rho_slider, rho_box = htutori.build_widget_control(
        name="rho",
        description="Correlation of factor and control",
        min_val=0.0,
        max_val=0.9,
        step=0.1,
        initial_value=0.5,
        is_float=True,
    )
    gamma_slider, gamma_box = htutori.build_widget_control(
        name="gamma",
        description="Effect of the control",
        min_val=0.0,
        max_val=1.0,
        step=0.1,
        initial_value=0.5,
        is_float=True,
    )
    mu_slider, mu_box = htutori.build_widget_control(
        name="mu",
        description="True premium",
        min_val=0.0,
        max_val=1.0,
        step=0.1,
        initial_value=0.5,
        is_float=True,
    )
    seed_slider, seed_box = _build_seed_control()
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the panels for the current widget values.

        :param change: widget change information (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            rho = rho_slider.value
            gamma = gamma_slider.value
            mu = mu_slider.value
            seed = seed_slider.value
            universe = simulate_universe(
                n_securities, n_periods, [mu], sigma, seed, gamma=gamma, rho=rho
            )
            # Run the same procedure without and with the control.
            betas = pd.DataFrame(
                {
                    "without Z": run_fama_macbeth(universe)["beta"],
                    "with Z": run_fama_macbeth(universe, use_control=True)[
                        "beta"
                    ],
                }
            )
            stats = {col: summarize_estimates(betas[col]) for col in betas}
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: factor against control in the last period.
            x_last = universe.chars[-1, :, 0]
            z_last = universe.control[-1]
            sns.scatterplot(
                x=x_last, y=z_last, ax=ax1, s=15, alpha=0.6, color="steelblue"
            )
            ax1.set_xlabel("characteristic x")
            ax1.set_ylabel("control z")
            ax1.set_title("X versus Z", fontsize=14, fontweight="bold")
            # Panel 2: mean estimates with 95% intervals.
            colors = ["firebrick", "seagreen"]
            for pos, (col, color) in enumerate(zip(betas.columns, colors)):
                ax2.bar(
                    pos, stats[col]["mean"], color=color, alpha=0.5, width=0.6
                )
                ax2.errorbar(
                    pos,
                    stats[col]["mean"],
                    yerr=1.96 * stats[col]["se"],
                    color=color,
                    capsize=10,
                    linewidth=2,
                )
            ax2.axhline(
                mu,
                color=_TRUE_COLOR,
                linestyle=":",
                linewidth=2.5,
                label=f"true mu = {mu:.2f}",
            )
            ax2.set_xticks([0, 1])
            ax2.set_xticklabels(list(betas.columns))
            ax2.set_ylabel("mean beta_t, 95% interval")
            ax2.set_title("Estimates", fontsize=14, fontweight="bold")
            ax2.legend(fontsize=9)
            # Panel 3: per-period estimates of both specifications.
            for col, color in zip(betas.columns, colors):
                ax3.plot(
                    betas.index, betas[col], color=color, linewidth=1, label=col
                )
            ax3.axhline(mu, color=_TRUE_COLOR, linestyle=":", linewidth=2.5)
            ax3.set_xlabel("period t")
            ax3.set_ylabel("beta_t")
            ax3.set_title("Beta path", fontsize=14, fontweight="bold")
            ax3.legend(fontsize=9)
            # Panel 4: comments.
            comment_text = (
                f"Parameters:\n"
                f"  rho: {rho:.2f}\n"
                f"  gamma: {gamma:.2f}\n"
                f"  mu: {mu:.2f}\n"
                f"  seed: {seed}\n\n"
                + "\n".join(
                    f"{col}:\n"
                    f"  mean: {stats[col]['mean']:.3f}\n"
                    f"  bias: {stats[col]['mean'] - mu:+.3f}\n"
                    f"  se: {stats[col]['se']:.3f}\n"
                    f"  t-stat: {stats[col]['t_stat']:.2f}\n"
                    for col in betas.columns
                )
                + f"\ngamma * rho: {gamma * rho:.3f}"
            )
            _fill_comments_panel(ax4, comment_text)
            plt.tight_layout()
            plt.show()

    _observe_all([rho_slider, gamma_slider, mu_slider, seed_slider], update_plot)
    update_plot()
    info_items = [
        ("rho", "correlation between the factor x and the control z"),
        ("gamma", "effect of the control on returns, percent per period"),
        ("mu", "true mean premium, percent per period"),
        ("seed", "random seed, redraws the whole universe"),
    ]
    _display_interactive(
        [rho_box, gamma_box, mu_box, seed_box], info_items, output
    )


# #############################################################################
# Cell 3.1: Ranking Securities Into Quantile Portfolios
# #############################################################################


def cell3_1_quantile_portfolios(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Sort securities into quantile portfolios and compare their returns.

    :param figsize: figure size, defaulting to `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    n_securities = 256
    n_periods = 120
    q_slider, q_box = htutori.build_widget_control(
        name="q",
        description="Number of quantiles",
        min_val=2,
        max_val=10,
        step=1,
        initial_value=5,
        is_float=False,
    )
    mu_slider, mu_box = htutori.build_widget_control(
        name="mu",
        description="True premium",
        min_val=0.0,
        max_val=1.0,
        step=0.1,
        initial_value=0.5,
        is_float=True,
    )
    sigma_slider, sigma_box = htutori.build_widget_control(
        name="sigma",
        description="Idiosyncratic volatility",
        min_val=1.0,
        max_val=10.0,
        step=1.0,
        initial_value=5.0,
        is_float=True,
    )
    seed_slider, seed_box = _build_seed_control()
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the panels for the current widget values.

        :param change: widget change information (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            q = q_slider.value
            mu = mu_slider.value
            sigma = sigma_slider.value
            seed = seed_slider.value
            universe = simulate_universe(
                n_securities, n_periods, [mu], sigma, seed
            )
            # Rank on the characteristic known at the start of each period.
            df = to_long_frame(universe)
            df["quantile"] = assign_quantiles(df, "x1", ["period"], q)
            # Equal-weighted excess return of each quantile in each period.
            port = df.groupby(["period", "quantile"])["excess"].mean()
            port = port.unstack("quantile")
            avg = port.mean()
            se = port.std() / np.sqrt(n_periods)
            cumulative = port.cumsum()
            palette = sns.color_palette("viridis", q)
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: ranking of the last period, colored by quantile.
            last = df[df["period"] == n_periods - 1].sort_values("x1")
            ranks = np.arange(1, len(last) + 1)
            ax1.scatter(
                ranks,
                last["x1"],
                c=[palette[g - 1] for g in last["quantile"]],
                s=12,
            )
            ax1.set_xlabel("rank of security")
            ax1.set_ylabel("characteristic x")
            ax1.set_title("Ranking", fontsize=14, fontweight="bold")
            # Panel 2: average excess return of each quantile.
            ax2.bar(avg.index, avg, color=palette)
            ax2.errorbar(
                avg.index,
                avg,
                yerr=1.96 * se,
                fmt="none",
                color="black",
                capsize=4,
            )
            ax2.axhline(0, color="black", linewidth=0.8)
            ax2.set_xticks(list(avg.index))
            ax2.set_xlabel("quantile (1 = lowest x)")
            ax2.set_ylabel("average excess return (%)")
            ax2.set_title("Average return", fontsize=14, fontweight="bold")
            # Panel 3: cumulative excess return of each quantile.
            for g in cumulative.columns:
                ax3.plot(
                    cumulative.index,
                    cumulative[g],
                    color=palette[g - 1],
                    linewidth=1.5,
                    label=f"Q{g}",
                )
            ax3.axhline(0, color="black", linewidth=0.8)
            ax3.set_xlabel("period")
            ax3.set_ylabel("cumulative excess return (%)")
            ax3.set_title("Cumulative return", fontsize=14, fontweight="bold")
            ax3.legend(fontsize=8, ncol=2)
            # Panel 4: comments.
            spread = avg.iloc[-1] - avg.iloc[0]
            comment_text = (
                f"Parameters:\n"
                f"  q: {q}\n"
                f"  mu: {mu:.2f}\n"
                f"  sigma: {sigma:.1f}\n"
                f"  N: {n_securities}\n"
                f"  T: {n_periods}\n"
                f"  seed: {seed}\n\n"
                f"Securities per quantile: {n_securities / q:.1f}\n\n"
                f"Average excess return:\n"
                f"  top Q{q}: {avg.iloc[-1]:+.3f} (se {se.iloc[-1]:.3f})\n"
                f"  bottom Q1: {avg.iloc[0]:+.3f} (se {se.iloc[0]:.3f})\n"
                f"  spread: {spread:+.3f}"
            )
            _fill_comments_panel(ax4, comment_text)
            plt.tight_layout()
            plt.show()

    _observe_all([q_slider, mu_slider, sigma_slider, seed_slider], update_plot)
    update_plot()
    info_items = [
        ("q", "number of quantile portfolios"),
        ("mu", "true mean premium, percent per period"),
        ("sigma", "idiosyncratic volatility, percent per period"),
        ("seed", "random seed, redraws the whole universe"),
    ]
    _display_interactive(
        [q_box, mu_box, sigma_box, seed_box], info_items, output
    )


# #############################################################################
# Cell 3.2: Building the Long-Short Portfolio
# #############################################################################


def cell3_2_long_short_portfolio(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Build the long-short portfolio and show its return series.

    :param figsize: figure size, defaulting to `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    n_securities = 256
    n_periods = 120
    sigma = 5.0
    frac_slider, frac_box = htutori.build_widget_control(
        name="frac",
        description="Share of securities in each leg",
        min_val=0.1,
        max_val=0.5,
        step=0.05,
        initial_value=0.3,
        is_float=True,
    )
    scheme_dropdown = ipywidgets.Dropdown(
        options=["equal", "rank"],
        value="equal",
        description="scheme",
    )
    mu_slider, mu_box = htutori.build_widget_control(
        name="mu",
        description="True premium",
        min_val=0.0,
        max_val=1.0,
        step=0.1,
        initial_value=0.5,
        is_float=True,
    )
    seed_slider, seed_box = _build_seed_control()
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the panels for the current widget values.

        :param change: widget change information (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            frac = frac_slider.value
            scheme = scheme_dropdown.value
            mu = mu_slider.value
            seed = seed_slider.value
            universe = simulate_universe(
                n_securities, n_periods, [mu], sigma, seed
            )
            chars = universe.chars[:, :, 0]
            weights = long_short_weights(chars, frac, scheme)
            # The market cancels out since the weights sum to zero.
            ls_returns = pd.Series((weights * universe.returns).sum(axis=1))
            stats = summarize_estimates(ls_returns)
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: weights of the last period against the rank.
            order = np.argsort(chars[-1])
            w_sorted = weights[-1, order]
            ranks = np.arange(1, n_securities + 1)
            # Fill the area under each leg, which reads better than hundreds of
            # one-pixel bars.
            ax1.fill_between(
                ranks,
                0,
                np.where(w_sorted > 0, w_sorted, 0.0),
                step="mid",
                color="seagreen",
                label="long leg",
            )
            ax1.fill_between(
                ranks,
                0,
                np.where(w_sorted < 0, w_sorted, 0.0),
                step="mid",
                color="firebrick",
                label="short leg",
            )
            ax1.legend(fontsize=9)
            ax1.axhline(0, color="black", linewidth=0.8)
            ax1.set_xlabel("rank of security (1 = lowest x)")
            ax1.set_ylabel("portfolio weight")
            ax1.set_title("Weights", fontsize=14, fontweight="bold")
            # Panel 2: return in each period and cumulative return.
            ax2.bar(ls_returns.index, ls_returns, color="steelblue", alpha=0.5)
            ax2.axhline(0, color="black", linewidth=0.8)
            ax2.set_xlabel("period")
            ax2.set_ylabel("long-short return (%)")
            ax2_cum = ax2.twinx()
            ax2_cum.plot(
                ls_returns.index,
                ls_returns.cumsum(),
                color=_EST_COLOR,
                linewidth=2,
            )
            ax2_cum.set_ylabel("cumulative return (%)", color=_EST_COLOR)
            ax2.set_title("Long-short return", fontsize=14, fontweight="bold")
            # Panel 3: distribution of the returns.
            sns.histplot(ls_returns, ax=ax3, bins=25, color="steelblue")
            ax3.axvline(
                stats["mean"],
                color=_EST_COLOR,
                linewidth=2,
                label=f"mean = {stats['mean']:.2f}",
            )
            ax3.axvline(0, color="black", linestyle=":", linewidth=1.5)
            ax3.set_xlabel("long-short return (%)")
            ax3.set_title("Distribution", fontsize=14, fontweight="bold")
            ax3.legend(fontsize=9)
            # Panel 4: comments.
            leg_long = weights[-1][weights[-1] > 0].sum()
            leg_short = weights[-1][weights[-1] < 0].sum()
            comment_text = (
                f"Parameters:\n"
                f"  frac: {frac:.2f}\n"
                f"  scheme: {scheme}\n"
                f"  mu: {mu:.2f}\n"
                f"  N: {n_securities}\n"
                f"  T: {n_periods}\n"
                f"  seed: {seed}\n\n"
                f"Weights (last period):\n"
                f"  long leg: {leg_long:+.2f}\n"
                f"  short leg: {leg_short:+.2f}\n"
                f"  sum: {leg_long + leg_short:+.2f}\n\n"
                f"Long-short return:\n"
                f"  mean: {stats['mean']:.3f}\n"
                f"  std: {stats['std']:.3f}\n"
                f"  t-stat: {stats['t_stat']:.2f}"
            )
            _fill_comments_panel(ax4, comment_text)
            plt.tight_layout()
            plt.show()

    _observe_all(
        [frac_slider, scheme_dropdown, mu_slider, seed_slider], update_plot
    )
    update_plot()
    info_items = [
        ("frac", "share of securities in each leg"),
        ("scheme", "equal weights, or weights growing with the rank"),
        ("mu", "true mean premium, percent per period"),
        ("seed", "random seed, redraws the whole universe"),
    ]
    _display_interactive(
        [frac_box, scheme_dropdown, mu_box, seed_box], info_items, output
    )


# #############################################################################
# Cell 3.3: Time-Series Regression on Factor Returns
# #############################################################################


def cell3_3_time_series_regression(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Regress each quantile portfolio on the long-short factor returns.

    :param figsize: figure size, defaulting to `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    n_securities = 256
    n_periods = 120
    sigma = 5.0
    q = 5
    frac = 0.3
    quantile_dropdown = ipywidgets.Dropdown(
        options=[f"Q{g}" for g in range(1, q + 1)],
        value="Q1",
        description="quantile",
    )
    control_dropdown = ipywidgets.Dropdown(
        options=["none", "market"],
        value="none",
        description="control",
    )
    mu_slider, mu_box = htutori.build_widget_control(
        name="mu",
        description="True premium",
        min_val=0.0,
        max_val=1.0,
        step=0.1,
        initial_value=0.5,
        is_float=True,
    )
    seed_slider, seed_box = _build_seed_control()
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the panels for the current widget values.

        :param change: widget change information (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            selected = int(quantile_dropdown.value[1:])
            control = control_dropdown.value
            mu = mu_slider.value
            seed = seed_slider.value
            universe = simulate_universe(
                n_securities, n_periods, [mu], sigma, seed
            )
            # Build `Y`: the return series of each quantile portfolio.
            df = to_long_frame(universe)
            df["quantile"] = assign_quantiles(df, "x1", ["period"], q)
            port = df.groupby(["period", "quantile"])["ret"].mean()
            port = port.unstack("quantile")
            # Build `X`: the long-short factor return series.
            weights = long_short_weights(universe.chars[:, :, 0], frac, "equal")
            factor = (weights * universe.returns).sum(axis=1)
            # Build `Z`: the equal-weighted market return series.
            market = universe.returns.mean(axis=1)
            regressors = factor[:, None]
            if control == "market":
                regressors = np.column_stack([factor, market])
            # Fit one time-series regression per quantile portfolio.
            fits = {
                g: sm.OLS(port[g].to_numpy(), sm.add_constant(regressors)).fit()
                for g in port.columns
            }
            summary = pd.DataFrame(
                {
                    "beta": [fits[g].params[1] for g in port.columns],
                    "ci_half": [1.96 * fits[g].bse[1] for g in port.columns],
                    "mean_ret": port.mean().to_numpy(),
                },
                index=port.columns,
            )
            fit = fits[selected]
            palette = sns.color_palette("viridis", q)
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: selected portfolio against the factor.
            ax1.scatter(
                factor,
                port[selected],
                s=15,
                alpha=0.6,
                color=palette[selected - 1],
            )
            beta_hat = fit.params[1]
            grid = np.linspace(factor.min(), factor.max(), 50)
            line_level = port[selected].mean() - beta_hat * factor.mean()
            ax1.plot(
                grid,
                line_level + beta_hat * grid,
                color=_EST_COLOR,
                linewidth=2,
                label=f"slope beta = {beta_hat:.2f}",
            )
            ax1.set_xlabel("long-short factor return (%)")
            ax1.set_ylabel(f"Q{selected} portfolio return (%)")
            ax1.set_title("Scatter", fontsize=14, fontweight="bold")
            ax1.legend(fontsize=9)
            # Panel 2: beta of each portfolio with its interval.
            edgecolors = [
                "black" if g == selected else "none" for g in summary.index
            ]
            ax2.bar(
                summary.index,
                summary["beta"],
                color=palette,
                edgecolor=edgecolors,
                linewidth=2.5,
            )
            ax2.errorbar(
                summary.index,
                summary["beta"],
                yerr=summary["ci_half"],
                fmt="none",
                color="black",
                capsize=4,
            )
            ax2.axhline(0, color="black", linewidth=0.8)
            ax2.set_xticks(list(summary.index))
            ax2.set_xlabel("quantile (1 = lowest x)")
            ax2.set_ylabel("beta on the factor, 95% interval")
            ax2.set_title("Beta by quantile", fontsize=14, fontweight="bold")
            # Panel 3: average return against beta across portfolios.
            ax3.scatter(
                summary["beta"], summary["mean_ret"], c=palette, s=80, zorder=3
            )
            ax3.scatter(
                summary.loc[selected, "beta"],
                summary.loc[selected, "mean_ret"],
                s=250,
                facecolors="none",
                edgecolors="black",
                linewidths=2,
                zorder=4,
            )
            slope, intercept = np.polyfit(
                summary["beta"], summary["mean_ret"], 1
            )
            grid = np.linspace(summary["beta"].min(), summary["beta"].max(), 50)
            ax3.plot(
                grid,
                intercept + slope * grid,
                color=_EST_COLOR,
                linewidth=1.5,
                label=f"slope = {slope:.2f}",
            )
            ax3.set_xlabel("beta on the factor")
            ax3.set_ylabel("average return (%)")
            ax3.set_title(
                "Mean return versus beta", fontsize=14, fontweight="bold"
            )
            ax3.legend(fontsize=9)
            # Panel 4: comments.
            comment_text = (
                f"Parameters:\n"
                f"  quantile: Q{selected}\n"
                f"  control: {control}\n"
                f"  mu: {mu:.2f}\n"
                f"  q: {q}, frac: {frac:.2f}\n"
                f"  seed: {seed}\n\n"
                f"Regression of Q{selected}:\n"
                f"  beta: {beta_hat:.3f}\n"
                f"  t-stat: {fit.tvalues[1]:.2f}\n"
                f"  intercept: {fit.params[0]:.3f}\n"
                f"  R^2: {fit.rsquared:.3f}\n\n"
                f"Mean factor return: {factor.mean():.3f}"
            )
            _fill_comments_panel(ax4, comment_text)
            plt.tight_layout()
            plt.show()

    _observe_all(
        [quantile_dropdown, control_dropdown, mu_slider, seed_slider],
        update_plot,
    )
    update_plot()
    info_items = [
        ("quantile", "portfolio shown in Scatter and highlighted elsewhere"),
        ("control", "no control, or the market return as Z"),
        ("mu", "true mean premium, percent per period"),
        ("seed", "random seed, redraws the whole universe"),
    ]
    _display_interactive(
        [quantile_dropdown, control_dropdown, mu_box, seed_box],
        info_items,
        output,
    )


# #############################################################################
# Cell 3.4: Multifactor Sorts Partition Each Subset
# #############################################################################


def cell3_4_compute_double_sort(
    universe: Universe,
    q: int,
    sort: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort on two characteristics and average returns in each cell.

    :param universe: simulated universe with two factors
    :param q: number of quantiles per factor
    :param sort:
        - "sequential": partition by `x1`, then each subset again by `x2`
        - "independent": rank both factors on the full universe
    :return: two `q` by `q` grids, first factor on rows and second factor on
        columns:
        - mean excess return of each cell, averaged over periods
        - mean number of securities in each cell
    """
    hdbg.dassert_in(sort, ["sequential", "independent"], "Unknown sort")
    df = to_long_frame(universe)
    df["g1"] = assign_quantiles(df, "x1", ["period"], q)
    by_second = ["period", "g1"] if sort == "sequential" else ["period"]
    df["g2"] = assign_quantiles(df, "x2", by_second, q)
    full_index = pd.MultiIndex.from_product(
        [range(1, q + 1), range(1, q + 1)], names=["g1", "g2"]
    )
    # Mean excess return of each cell in each period, then over periods.
    cell_ret = df.groupby(["period", "g1", "g2"])["excess"].mean()
    mean_grid = cell_ret.groupby(["g1", "g2"]).mean().reindex(full_index)
    mean_grid = mean_grid.unstack("g2")
    # Count securities in each cell, with zero for empty cells.
    counts = df.groupby(["period", "g1", "g2"]).size()
    counts = counts.unstack(["g1", "g2"], fill_value=0)
    counts = counts.reindex(columns=full_index, fill_value=0).mean()
    count_grid = counts.unstack("g2")
    return mean_grid, count_grid


def cell3_4_multifactor_sort(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show double-sorted portfolios on two characteristics.

    :param figsize: figure size, defaulting to `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    n_securities = 1000
    n_periods = 60
    sigma = 5.0
    q_slider, q_box = htutori.build_widget_control(
        name="q",
        description="Quantiles per factor",
        min_val=2,
        max_val=5,
        step=1,
        initial_value=5,
        is_float=False,
    )
    sort_dropdown = ipywidgets.Dropdown(
        options=["independent", "sequential"],
        value="independent",
        description="sort",
    )
    rho_slider, rho_box = htutori.build_widget_control(
        name="rho",
        description="Correlation of the characteristics",
        min_val=0.0,
        max_val=0.9,
        step=0.1,
        initial_value=0.0,
        is_float=True,
    )
    mu1_slider, mu1_box = htutori.build_widget_control(
        name="mu1",
        description="Premium of the first factor",
        min_val=0.0,
        max_val=1.0,
        step=0.1,
        initial_value=0.5,
        is_float=True,
    )
    mu2_slider, mu2_box = htutori.build_widget_control(
        name="mu2",
        description="Premium of the second factor",
        min_val=0.0,
        max_val=1.0,
        step=0.1,
        initial_value=0.5,
        is_float=True,
    )
    seed_slider, seed_box = _build_seed_control()
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the panels for the current widget values.

        :param change: widget change information (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            q = q_slider.value
            sort = sort_dropdown.value
            rho = rho_slider.value
            mu1 = mu1_slider.value
            mu2 = mu2_slider.value
            seed = seed_slider.value
            universe = simulate_universe(
                n_securities,
                n_periods,
                [mu1, mu2],
                sigma,
                seed,
                rho_factors=rho,
            )
            mean_grid, count_grid = cell3_4_compute_double_sort(
                universe, q, sort
            )
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: mean excess return of each cell.
            sns.heatmap(
                mean_grid,
                ax=ax1,
                cmap="coolwarm",
                center=0,
                annot=True,
                fmt=".2f",
                cbar_kws={"label": "mean excess return (%)"},
            )
            ax1.set_xlabel("second factor quantile (x2)")
            ax1.set_ylabel("first factor quantile (x1)")
            ax1.set_title("Mean return grid", fontsize=14, fontweight="bold")
            # Panel 2: mean number of securities in each cell.
            sns.heatmap(
                count_grid,
                ax=ax2,
                cmap="Blues",
                annot=True,
                fmt=".0f",
                cbar_kws={"label": "securities per cell"},
            )
            ax2.set_xlabel("second factor quantile (x2)")
            ax2.set_ylabel("first factor quantile (x1)")
            ax2.set_title("Count grid", fontsize=14, fontweight="bold")
            # Panel 3: marginal means along each factor.
            by_first = mean_grid.mean(axis=1)
            by_second = mean_grid.mean(axis=0)
            ax3.plot(
                by_first.index,
                by_first,
                marker="o",
                color="firebrick",
                label="by x1 quantile",
            )
            ax3.plot(
                by_second.index,
                by_second,
                marker="s",
                color="seagreen",
                label="by x2 quantile",
            )
            ax3.axhline(0, color="black", linewidth=0.8)
            ax3.set_xticks(list(by_first.index))
            ax3.set_xlabel("quantile")
            ax3.set_ylabel("mean excess return (%)")
            ax3.set_title("Marginal means", fontsize=14, fontweight="bold")
            ax3.legend(fontsize=9)
            # Panel 4: comments.
            counts = count_grid.to_numpy()
            comment_text = (
                f"Parameters:\n"
                f"  q: {q}\n"
                f"  sort: {sort}\n"
                f"  rho: {rho:.2f}\n"
                f"  mu1: {mu1:.2f}\n"
                f"  mu2: {mu2:.2f}\n"
                f"  N: {n_securities}, T: {n_periods}\n"
                f"  seed: {seed}\n\n"
                f"Cell counts:\n"
                f"  smallest: {counts.min():.1f}\n"
                f"  largest: {counts.max():.1f}\n\n"
                f"Mean-return range:\n"
                f"  along x1: {by_first.max() - by_first.min():.3f}\n"
                f"  along x2: {by_second.max() - by_second.min():.3f}"
            )
            _fill_comments_panel(ax4, comment_text)
            plt.tight_layout()
            plt.show()

    _observe_all(
        [
            q_slider,
            sort_dropdown,
            rho_slider,
            mu1_slider,
            mu2_slider,
            seed_slider,
        ],
        update_plot,
    )
    update_plot()
    info_items = [
        ("q", "number of quantiles per factor"),
        (
            "sort",
            "sequential partitions each x1 subset again by x2, independent "
            "ranks both factors on the full universe",
        ),
        ("rho", "correlation between the two characteristics"),
        ("mu1", "premium of the first factor (value), percent per period"),
        ("mu2", "premium of the second factor (size), percent per period"),
        ("seed", "random seed, redraws the whole universe"),
    ]
    _display_interactive(
        [q_box, sort_dropdown, rho_box, mu1_box, mu2_box, seed_box],
        info_items,
        output,
    )


# #############################################################################
# Cell 4.1: Statistical Significance and False Discoveries
# #############################################################################


def cell4_1_simulate_studies(
    n_periods: int,
    n_trials: int,
    n_securities: int,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """
    Simulate the per-period Fama-MacBeth slopes of many studies, with `mu = 0`.

    With a characteristic standardized in each period and Gaussian noise, the
    least-squares slope of period `t` has the exact distribution
    `beta_t ~ N(lambda_t, sigma^2 / N)`, so it is drawn directly instead of
    running `T` regressions per study. Adding `mu` to every entry gives the
    slopes of studies with a premium of mean `mu`.

    :param n_periods: number of periods `T` per study
    :param n_trials: number of simulated studies
    :param n_securities: number of securities `N`
    :param sigma: idiosyncratic volatility, in percent per period
    :param seed: random seed
    :return: slopes with shape `(n_trials, n_periods)`
    """
    rng = np.random.default_rng(seed)
    premium_part = _TAU * rng.standard_normal((n_trials, n_periods))
    sampling_part = (sigma / np.sqrt(n_securities)) * rng.standard_normal(
        (n_trials, n_periods)
    )
    betas = premium_part + sampling_part
    return betas


def _t_stats(betas: np.ndarray, mu: float) -> np.ndarray:
    """
    Compute the Fama-MacBeth t-statistic of each study.

    :param betas: zero-mean slopes with shape `(n_trials, n_periods)`
    :param mu: mean premium added to every slope
    :return: one t-statistic per study
    """
    n_periods = betas.shape[1]
    se = betas.std(axis=1, ddof=1) / np.sqrt(n_periods)
    t_stats = (betas.mean(axis=1) + mu) / se
    return t_stats


def cell4_1_significance_and_false_discoveries(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show how often studies find a significant premium.

    :param figsize: figure size, defaulting to `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    n_securities = 128
    sigma = 5.0
    mu_grid = np.linspace(0.0, 1.0, 41)
    mu_slider, mu_box = htutori.build_widget_control(
        name="mu",
        description="True premium",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.0,
        is_float=True,
    )
    t_slider, t_box = htutori.build_log_widget_control(
        name="log2(T)",
        description="Number of periods",
        min_exp=4,
        max_exp=9,
        initial_exp=6,
        base=2,
    )
    trials_slider, trials_box = htutori.build_log_widget_control(
        name="log2(n_trials)",
        description="Number of studies",
        min_exp=5,
        max_exp=10,
        initial_exp=9,
        base=2,
    )
    seed_slider, seed_box = _build_seed_control()
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the panels for the current widget values.

        :param change: widget change information (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            mu = mu_slider.value
            n_periods = 2**t_slider.value
            n_trials = 2**trials_slider.value
            seed = seed_slider.value
            betas = cell4_1_simulate_studies(
                n_periods, n_trials, n_securities, sigma, seed
            )
            t_stats = _t_stats(betas, mu)
            rejected = np.abs(t_stats) > 2
            # Rejection rate on a grid of premia, reusing the same draws.
            rates = [np.mean(np.abs(_t_stats(betas, m)) > 2) for m in mu_grid]
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: distribution of the t-statistics.
            sns.histplot(t_stats, ax=ax1, bins=40, color="steelblue")
            x_low = min(-4.0, t_stats.min())
            x_high = max(4.0, t_stats.max())
            ax1.axvspan(x_low, -2, color="firebrick", alpha=0.1)
            ax1.axvspan(2, x_high, color="firebrick", alpha=0.1)
            for threshold in [-2, 2]:
                ax1.axvline(threshold, color="firebrick", linestyle="--")
            ax1.set_xlim(x_low, x_high)
            ax1.set_xlabel("Fama-MacBeth t-statistic")
            ax1.set_ylabel("number of studies")
            ax1.set_title("t-stat distribution", fontsize=14, fontweight="bold")
            # Panel 2: rejection rate against the true premium.
            ax2.plot(mu_grid, rates, color=_EST_COLOR, linewidth=2)
            ax2.scatter(
                [mu],
                [rejected.mean()],
                color="firebrick",
                s=120,
                zorder=3,
                label="current mu",
            )
            ax2.axhline(
                0.05,
                color=_TRUE_COLOR,
                linestyle=":",
                linewidth=2,
                label="5% level",
            )
            ax2.set_ylim(0, 1.05)
            ax2.set_xlabel("true premium mu")
            ax2.set_ylabel("share of studies with |t| > 2")
            ax2.set_title("Rejection rate", fontsize=14, fontweight="bold")
            ax2.legend(fontsize=9)
            # Panel 3: comments.
            comment_text = (
                f"Parameters:\n"
                f"  mu: {mu:.2f}\n"
                f"  T: {n_periods}\n"
                f"  n_trials: {n_trials}\n"
                f"  N: {n_securities}, sigma: {sigma:.1f}\n"
                f"  seed: {seed}\n\n"
                f"Studies:\n"
                f"  share |t| > 2: {rejected.mean():.1%}\n"
                f"  mean t-stat: {t_stats.mean():.2f}"
            )
            _fill_comments_panel(ax3, comment_text)
            plt.tight_layout()
            plt.show()

    _observe_all([mu_slider, t_slider, trials_slider, seed_slider], update_plot)
    update_plot()
    info_items = [
        ("mu", "true mean premium, percent per period"),
        ("log2(T)", "number of periods T per study, on a log scale"),
        ("log2(n_trials)", "number of simulated studies, on a log scale"),
        ("seed", "random seed for all the studies"),
    ]
    _display_interactive(
        [mu_box, t_box, trials_box, seed_box], info_items, output
    )


# #############################################################################
# Cell 4.2: Comparing the Two Procedures
# #############################################################################


def cell4_2_simulate_both_procedures(
    mu: float,
    n_periods: int,
    n_securities: int,
    frac: float,
    sigma: float,
    n_trials: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Simulate the per-period estimates of both procedures in many studies.

    In period `t`, given a standardized characteristic and Gaussian noise:
    - the Fama-MacBeth slope is `lambda_t + e_F`, with `var(e_F) = sigma^2 / N`
    - the equal-weighted long-short return is `lambda_t s + e_L`, where `s` is
      the gap in mean characteristic between the legs and
      `var(e_L) = 2 sigma^2 / (frac N)`
    - both errors come from the same noise, with `cov(e_F, e_L) = sigma^2 s / N`
    The long-short return is divided by `s` so that both estimates are in
    percent per unit of characteristic.

    :param mu: true mean premium
    :param n_periods: number of periods `T` per study
    :param n_securities: number of securities `N`
    :param frac: share of securities in each long-short leg
    :param sigma: idiosyncratic volatility, in percent per period
    :param n_trials: number of simulated studies
    :param seed: random seed
    :return: dict with arrays of shape `(n_trials, n_periods)`:
        - `fm`: Fama-MacBeth slopes
        - `ls`: rescaled long-short returns
    """
    rng = np.random.default_rng(seed)
    # Gap in mean characteristic between the top and the bottom `frac` of a
    # standard normal.
    exposure = 2 * scipy.stats.norm.pdf(scipy.stats.norm.ppf(1 - frac)) / frac
    premia = mu + _TAU * rng.standard_normal((n_trials, n_periods))
    # Draw the two correlated sampling errors.
    var_f = sigma**2 / n_securities
    var_l = 2 * sigma**2 / (frac * n_securities)
    cov_fl = sigma**2 * exposure / n_securities
    chol = np.linalg.cholesky(np.array([[var_f, cov_fl], [cov_fl, var_l]]))
    shocks = rng.standard_normal((n_trials, n_periods, 2)) @ chol.T
    fm = premia + shocks[:, :, 0]
    ls = (premia * exposure + shocks[:, :, 1]) / exposure
    res = {"fm": fm, "ls": ls}
    return res


def cell4_2_compare_procedures(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Compare the verdicts of the two procedures on the same studies.

    :param figsize: figure size, defaulting to `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    sigma = 5.0
    n_trials = 500
    mu_slider, mu_box = htutori.build_widget_control(
        name="mu",
        description="True premium",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.15,
        is_float=True,
    )
    t_slider, t_box = htutori.build_log_widget_control(
        name="log2(T)",
        description="Number of periods",
        min_exp=4,
        max_exp=9,
        initial_exp=6,
        base=2,
    )
    n_slider, n_box = htutori.build_log_widget_control(
        name="log2(N)",
        description="Number of securities",
        min_exp=5,
        max_exp=10,
        initial_exp=7,
        base=2,
    )
    frac_slider, frac_box = htutori.build_widget_control(
        name="frac",
        description="Share of securities in each leg",
        min_val=0.1,
        max_val=0.5,
        step=0.05,
        initial_value=0.3,
        is_float=True,
    )
    seed_slider, seed_box = _build_seed_control()
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the panels for the current widget values.

        :param change: widget change information (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            mu = mu_slider.value
            n_periods = 2**t_slider.value
            n_securities = 2**n_slider.value
            frac = frac_slider.value
            seed = seed_slider.value
            sims = cell4_2_simulate_both_procedures(
                mu, n_periods, n_securities, frac, sigma, n_trials, seed
            )
            # Summarize every study under both procedures.
            means = {key: val.mean(axis=1) for key, val in sims.items()}
            ses = {
                key: val.std(axis=1, ddof=1) / np.sqrt(n_periods)
                for key, val in sims.items()
            }
            t_fm = means["fm"] / ses["fm"]
            t_ls = means["ls"] / ses["ls"]
            sig_fm = np.abs(t_fm) > 2
            sig_ls = np.abs(t_ls) > 2
            verdicts = pd.Series(
                np.select(
                    [sig_fm & sig_ls, sig_fm & ~sig_ls, ~sig_fm & sig_ls],
                    ["both", "only FM", "only LS"],
                    default="neither",
                )
            )
            shares = verdicts.value_counts(normalize=True)
            shares = shares.reindex(
                ["both", "only FM", "only LS", "neither"], fill_value=0.0
            )
            corr = np.corrcoef(t_fm, t_ls)[0, 1]
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: estimates of the first study with 95% intervals.
            labels = ["Fama-MacBeth", "long-short / s"]
            colors = ["firebrick", "seagreen"]
            for pos, (key, color) in enumerate(zip(["fm", "ls"], colors)):
                ax1.errorbar(
                    pos,
                    means[key][0],
                    yerr=1.96 * ses[key][0],
                    fmt="o",
                    color=color,
                    capsize=10,
                    markersize=10,
                    linewidth=2,
                )
            ax1.axhline(
                mu,
                color=_TRUE_COLOR,
                linestyle=":",
                linewidth=2.5,
                label=f"true mu = {mu:.2f}",
            )
            ax1.axhline(0, color="black", linewidth=0.8)
            ax1.set_xlim(-0.6, 1.6)
            ax1.set_xticks([0, 1])
            ax1.set_xticklabels(labels)
            ax1.set_ylabel("premium per unit of x (%)")
            ax1.set_title("Estimates", fontsize=14, fontweight="bold")
            ax1.legend(fontsize=9)
            # Panel 2: t-statistics of every study under both procedures.
            ax2.scatter(t_fm, t_ls, s=10, alpha=0.5, color="steelblue")
            low = min(t_fm.min(), t_ls.min(), -3)
            high = max(t_fm.max(), t_ls.max(), 3)
            ax2.plot([low, high], [low, high], color="gray", linestyle="--")
            for threshold in [-2, 2]:
                ax2.axvline(threshold, color="firebrick", linewidth=0.8)
                ax2.axhline(threshold, color="firebrick", linewidth=0.8)
            ax2.set_xlabel("Fama-MacBeth t-statistic")
            ax2.set_ylabel("long-short t-statistic")
            ax2.set_title("t-stat scatter", fontsize=14, fontweight="bold")
            # Panel 3: share of studies by verdict.
            ax3.bar(
                shares.index,
                shares,
                color=["seagreen", "firebrick", "darkorange", "gray"],
            )
            ax3.set_ylim(0, 1)
            ax3.set_ylabel("share of studies")
            ax3.set_title("Verdicts", fontsize=14, fontweight="bold")
            # Panel 4: comments.
            disagree = shares["only FM"] + shares["only LS"]
            comment_text = (
                f"Parameters:\n"
                f"  mu: {mu:.2f}\n"
                f"  T: {n_periods}\n"
                f"  N: {n_securities}\n"
                f"  frac: {frac:.2f}\n"
                f"  seed: {seed}\n"
                f"  studies: {n_trials}\n\n"
                f"Across studies:\n"
                f"  corr(t_FM, t_LS): {corr:.3f}\n"
                f"  verdicts disagree: {disagree:.1%}\n"
                f"  both significant: {shares['both']:.1%}"
            )
            _fill_comments_panel(ax4, comment_text)
            plt.tight_layout()
            plt.show()

    _observe_all(
        [mu_slider, t_slider, n_slider, frac_slider, seed_slider], update_plot
    )
    update_plot()
    info_items = [
        ("mu", "true mean premium, percent per period"),
        ("log2(T)", "number of periods T per study, on a log scale"),
        ("log2(N)", "number of securities N, on a log scale"),
        ("frac", "share of securities in each long-short leg"),
        ("seed", "random seed for all the studies"),
    ]
    _display_interactive(
        [mu_box, t_box, n_box, frac_box, seed_box], info_items, output
    )
