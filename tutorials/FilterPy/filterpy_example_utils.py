"""
filterpy_example_utils.py

Utility functions for the FilterPy example notebook - financial applications.

- All widget and plotting logic lives here; the notebook only calls these
  functions.
- Covers financial applications of four Kalman filter variants:
  - Linear Kalman Filter (KF): stock price trend extraction
  - Extended Kalman Filter (EKF): pairs trading beta estimation
  - Unscented Kalman Filter (UKF): stochastic volatility estimation
  - Ensemble Kalman Filter (EnKF): portfolio risk scenario analysis

Import as:

import tutorials.FilterPy.filterpy_example_utils as utils
"""

import logging

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import (
    EnsembleKalmanFilter,
    ExtendedKalmanFilter,
    KalmanFilter,
    MerweScaledSigmaPoints,
    UnscentedKalmanFilter,
)
from IPython.display import display
from scipy.stats import norm

_LOG = logging.getLogger(__name__)

# #############################################################################
# Shared helpers
# #############################################################################


def _rmse(estimated: np.ndarray, true: np.ndarray) -> float:
    """
    Compute root-mean-square error.

    :param estimated: estimated values
    :param true: true values
    :return: RMSE scalar
    """
    return float(np.sqrt(np.mean((estimated - true) ** 2)))


def _simulate_price_trend(
    n_days: int = 100,
    seed: int = 42,
    drift: float = 0.05,
    obs_noise_std: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a latent price trend and noisy observed prices.

    The true trend is a linear drift with a sinusoidal modulation plus
    small cumulative process noise.  Observations are the true trend plus
    i.i.d. Gaussian noise with standard deviation obs_noise_std.

    :param n_days: number of trading days
    :param seed: random seed for reproducibility
    :param drift: daily linear drift in price
    :param obs_noise_std: observation noise standard deviation
    :return: (days, true_trend, noisy_prices) arrays of length n_days
    """
    rng = np.random.default_rng(seed)
    days = np.arange(n_days, dtype=float)
    # True latent trend: linear drift + sinusoidal variation.
    true_trend = (
        100.0
        + drift * days
        + 5.0 * np.sin(2.0 * np.pi * days / 50.0)
    )
    # Cumulative process noise makes it look like a realistic price path.
    process_noise = np.cumsum(rng.normal(0, 0.1, size=n_days))
    true_trend = true_trend + process_noise
    # Noisy observations.
    noisy_prices = true_trend + rng.normal(0, obs_noise_std, size=n_days)
    return days, true_trend, noisy_prices


# #############################################################################
# Cell 1: Introduction - Signal vs Noise in Financial Markets
# #############################################################################


def plot_intro_signal_vs_noise(n_days: int = 100, seed: int = 42) -> None:
    """
    Show the predict-update cycle and signal vs noise for financial data.

    Panel 1 overlays a noisy observed price series on the true latent
    trend to illustrate why filtering is needed.  Panel 2 draws the
    standard Kalman flow diagram annotated with financial meaning.

    :param n_days: number of trading days to simulate
    :param seed: random seed for the simulation
    """
    days, true_trend, noisy_prices = _simulate_price_trend(
        n_days=n_days, seed=seed, obs_noise_std=5.0
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # ------------------------------------------------------------------
    # Panel 1: Noisy prices vs true underlying value.
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.plot(
        days,
        noisy_prices,
        color="orange",
        lw=1.2,
        alpha=0.8,
        label="Observed Price (noisy)",
    )
    ax.plot(
        days,
        true_trend,
        "b-",
        lw=2.5,
        label="True Underlying Value",
    )
    ax.set_xlim(0, n_days - 1)
    ax.set_ylim(80, 140)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Price ($)")
    ax.set_title(
        "What We See (Noisy Prices) vs What We Want (True Value)"
    )
    ax.legend(fontsize=9)
    # ------------------------------------------------------------------
    # Panel 2: Kalman flow diagram with financial annotations.
    # ------------------------------------------------------------------
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Kalman Filter: Predict-Update Cycle in Finance")
    # Node boxes.
    box_style = dict(
        boxstyle="round,pad=0.5",
        facecolor="lightblue",
        edgecolor="navy",
        lw=1.5,
    )
    noise_style = dict(
        boxstyle="round,pad=0.4",
        facecolor="lightyellow",
        edgecolor="darkorange",
        lw=1.2,
    )
    nodes = [
        (1.5, 3.0, "True Market\nState x"),
        (5.0, 3.0, "Predicted\nState"),
        (8.5, 3.0, "Updated\nEstimate"),
    ]
    for xn, yn, label in nodes:
        ax.text(
            xn,
            yn,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=box_style,
        )
    # Noise inputs.
    noise_nodes = [
        (3.25, 4.8, "Process Noise Q\n(macro shocks)"),
        (6.75, 4.8, "Meas. Noise R\n(bid-ask spread)"),
    ]
    for xn, yn, label in noise_nodes:
        ax.text(
            xn,
            yn,
            label,
            ha="center",
            va="center",
            fontsize=8,
            bbox=noise_style,
        )
    # Observed price box.
    ax.text(
        6.75,
        1.2,
        "Observed Price z\n(noisy signal)",
        ha="center",
        va="center",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="lightsalmon",
            edgecolor="red",
            lw=1.2,
        ),
    )
    # Arrows between nodes.
    arrow_kw = dict(
        arrowstyle="->",
        color="navy",
        lw=1.5,
        connectionstyle="arc3,rad=0",
    )
    from matplotlib.patches import FancyArrowPatch  # noqa: PLC0415
    for (x1, y1, _), (x2, y2, _) in zip(nodes[:-1], nodes[1:]):
        ax.annotate(
            "",
            xy=(x2 - 0.7, y2),
            xytext=(x1 + 0.7, y1),
            arrowprops=arrow_kw,
        )
    # Noise arrows.
    ax.annotate(
        "",
        xy=(3.25, 3.3),
        xytext=(3.25, 4.4),
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2),
    )
    ax.annotate(
        "",
        xy=(6.75, 3.3),
        xytext=(6.75, 4.4),
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2),
    )
    # Observation arrow from bottom.
    ax.annotate(
        "",
        xy=(7.5, 2.7),
        xytext=(6.75, 1.6),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
    )
    # Kalman gain label on update arrow.
    ax.text(6.75, 3.0, "K", ha="center", va="center", fontsize=10,
            color="darkgreen", fontweight="bold")
    plt.tight_layout()
    plt.show()
    # ------------------------------------------------------------------
    # Summary table of Kalman matrix roles.
    # ------------------------------------------------------------------
    rows = [
        ["F", "Price dynamics model (random walk, mean reversion)"],
        ["H", "Mapping from hidden state to observed price"],
        ["Q", "Process noise (market volatility, model uncertainty)"],
        ["R", "Measurement noise (microstructure, bid-ask spread)"],
        ["P", "Current uncertainty about the hidden state"],
        ["K", "Kalman gain: how much to trust the new price"],
    ]
    fig2, ax2 = plt.subplots(figsize=(9, 2.4))
    ax2.axis("off")
    tbl = ax2.table(
        cellText=rows,
        colLabels=["Matrix", "Financial Meaning"],
        cellLoc="left",
        loc="center",
        colWidths=[0.08, 0.72],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    ax2.set_title(
        "Kalman Filter Matrices and Their Financial Roles",
        fontsize=11,
        pad=12,
    )
    plt.tight_layout()
    plt.show()


# #############################################################################
# Cell 2: Linear Kalman Filter - Extracting Price Trend from Noise
# #############################################################################


def _run_linear_kf_trend(
    R_val: float,
    Q_val: float,
    n_days: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a linear KF to extract a latent price trend from noisy observations.

    State is [price_level, price_drift]; only price_level is observed.

    :param R_val: measurement noise variance
    :param Q_val: process noise variance (white noise acceleration)
    :param n_days: number of trading days
    :param seed: random seed
    :return: (days, true_trend, noisy_prices, estimates, std_devs)
    """
    days, true_trend, noisy_prices = _simulate_price_trend(
        n_days=n_days, seed=seed, obs_noise_std=np.sqrt(R_val)
    )
    kf = KalmanFilter(dim_x=2, dim_z=1)
    # State: [price_level, price_drift].
    kf.x = np.array([[noisy_prices[0]], [0.0]])
    # Constant-velocity state transition.
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    # Observe price level only.
    kf.H = np.array([[1.0, 0.0]])
    # Initial covariance.
    kf.P = np.eye(2) * 500.0
    kf.R = np.array([[R_val]])
    kf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=Q_val)
    estimates = np.zeros(n_days)
    stds = np.zeros(n_days)
    for i, z in enumerate(noisy_prices):
        kf.predict()
        kf.update(np.array([[z]]))
        estimates[i] = kf.x[0, 0]
        stds[i] = np.sqrt(kf.P[0, 0])
    return days, true_trend, noisy_prices, estimates, stds


def plot_linear_kf_trend(R_val: float = 5.0, Q_val: float = 0.1) -> None:
    """
    Plot linear KF trend extraction from noisy stock prices.

    Shows true latent trend, noisy observed prices, KF estimate with
    1-sigma confidence band, and RMSE in the legend.

    :param R_val: measurement noise variance
    :param Q_val: process noise variance
    """
    days, true_trend, noisy_prices, estimates, stds = _run_linear_kf_trend(
        R_val, Q_val
    )
    rmse = _rmse(estimates, true_trend)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(days, true_trend, "b-", lw=2, label="True latent trend")
    ax.scatter(
        days,
        noisy_prices,
        c="orange",
        s=15,
        alpha=0.6,
        label="Noisy observed prices",
    )
    ax.plot(
        days,
        estimates,
        "g-",
        lw=2,
        label=f"KF estimate (RMSE={rmse:.2f})",
    )
    ax.fill_between(
        days,
        estimates - stds,
        estimates + stds,
        alpha=0.2,
        color="green",
        label="1-sigma band",
    )
    ax.set_xlim(0, 99)
    ax.set_ylim(80, 140)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Price ($)")
    ax.set_title(
        "Kalman Filter: Extracting Trend from Noisy Stock Prices"
        f"  (R={R_val:.1f}, Q={Q_val:.3f})"
    )
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.show()


def show_linear_kf_trend_interactive() -> None:
    """
    Display interactive widget for linear KF price trend extraction.

    Sliders control measurement noise R (microstructure noise) and
    process noise Q (market volatility).
    """
    r_slider = widgets.FloatSlider(
        value=5.0,
        min=0.5,
        max=50.0,
        step=0.5,
        description="Observation Noise R:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="520px"),
    )
    q_slider = widgets.FloatSlider(
        value=0.1,
        min=0.001,
        max=2.0,
        step=0.01,
        description="Process Noise Q:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="520px"),
    )
    out = widgets.interactive_output(
        plot_linear_kf_trend,
        {"R_val": r_slider, "Q_val": q_slider},
    )
    display(widgets.VBox([r_slider, q_slider, out]))


# #############################################################################
# Cell 3: Linear KF - Kalman Gain and Uncertainty Convergence
# #############################################################################


def _run_kf_uncertainty(
    R_val: float,
    P0_val: float,
    n_days: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run linear KF and collect uncertainty P and Kalman gain K over time.

    :param R_val: measurement noise variance
    :param P0_val: initial state covariance diagonal value
    :param n_days: number of trading days
    :return: (pos_variance, vel_variance, kalman_gain) arrays
    """
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[100.0], [0.0]])
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.P = np.eye(2) * P0_val
    kf.R = np.array([[R_val]])
    kf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=0.1)
    pos_var = np.zeros(n_days)
    vel_var = np.zeros(n_days)
    k_gain = np.zeros(n_days)
    rng = np.random.default_rng(42)
    for i in range(n_days):
        kf.predict()
        z = 100.0 + rng.normal(0, np.sqrt(R_val))
        kf.update(np.array([[z]]))
        pos_var[i] = kf.P[0, 0]
        vel_var[i] = kf.P[1, 1]
        k_gain[i] = kf.K[0, 0]
    return pos_var, vel_var, k_gain


def plot_kf_uncertainty(
    P0_val: float = 1000.0, R_val: float = 5.0
) -> None:
    """
    Plot covariance P and Kalman gain K convergence over trading days.

    :param P0_val: initial covariance diagonal value
    :param R_val: measurement noise variance
    """
    days = np.arange(100)
    pos_var, vel_var, k_gain = _run_kf_uncertainty(R_val, P0_val)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    # Panel 1: P convergence.
    ax = axes[0]
    ax.plot(days, pos_var, "b-", lw=2, label="Position variance P[0,0]")
    ax.plot(
        days, vel_var, "-", color="orange", lw=2,
        label="Velocity variance P[1,1]",
    )
    ax.set_xlim(0, 99)
    ax.set_ylim(0, 500)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Variance")
    ax.set_title("Uncertainty P Converges Over Time")
    ax.legend(fontsize=9)
    # Panel 2: K evolution.
    ax = axes[1]
    ax.plot(days, k_gain, "b-", lw=2, label="Kalman gain K[0,0]")
    ax.axhline(
        0.5,
        color="gray",
        linestyle="--",
        lw=1.5,
        label="Equal trust: model vs measurement",
    )
    ax.set_xlim(0, 99)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Kalman Gain")
    ax.set_title("Kalman Gain K: How Much to Trust New Prices")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def show_kf_uncertainty_interactive() -> None:
    """
    Display interactive widget for KF uncertainty and Kalman gain evolution.

    Logarithmic slider controls initial uncertainty P0; linear slider
    controls measurement noise R.
    """
    p0_slider = widgets.FloatLogSlider(
        value=1000.0,
        base=10,
        min=1,
        max=4,
        step=0.1,
        description="Initial Uncertainty P0:",
        style={"description_width": "200px"},
        layout=widgets.Layout(width="540px"),
    )
    r_slider = widgets.FloatSlider(
        value=5.0,
        min=0.5,
        max=50.0,
        step=0.5,
        description="Observation Noise R:",
        style={"description_width": "200px"},
        layout=widgets.Layout(width="540px"),
    )
    out = widgets.interactive_output(
        plot_kf_uncertainty,
        {"P0_val": p0_slider, "R_val": r_slider},
    )
    display(widgets.VBox([p0_slider, r_slider, out]))


# #############################################################################
# Cell 4: Extended Kalman Filter - Time-Varying Beta in Pairs Trading
# #############################################################################


def _simulate_pairs(
    n_days: int = 200,
    sigma_beta: float = 0.05,
    R_val: float = 1.0,
    seed: int = 42,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Simulate two cointegrated stocks with a time-varying hedge ratio.

    Stock B follows a Brownian motion; stock A is a multiple of stock B
    plus independent noise.  Beta evolves as a random walk with
    volatility sigma_beta.

    :param n_days: number of trading days
    :param sigma_beta: drift volatility for beta
    :param R_val: observation noise standard deviation for stock A
    :param seed: random seed
    :return: (days, stock_a, stock_b, true_beta)
    """
    rng = np.random.default_rng(seed)
    days = np.arange(n_days, dtype=float)
    # Stock B: Brownian motion around 100.
    stock_b = np.zeros(n_days)
    stock_b[0] = 100.0
    for t in range(1, n_days):
        stock_b[t] = stock_b[t - 1] + rng.normal(0, 1.0)
    # Time-varying beta: starts at 1, random walk.
    true_beta = np.zeros(n_days)
    true_beta[0] = 1.0
    for t in range(1, n_days):
        true_beta[t] = np.clip(
            true_beta[t - 1] + rng.normal(0, sigma_beta), 0.1, 5.0
        )
    # Stock A = beta * stock_b + noise.
    stock_a = true_beta * stock_b + rng.normal(0, np.sqrt(R_val), size=n_days)
    return days, stock_a, stock_b, true_beta


def _run_ekf_pairs(
    stock_a: np.ndarray,
    stock_b: np.ndarray,
    sigma_beta: float,
    R_val: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run EKF to track a time-varying log-beta between two stocks.

    State is [log_beta]; h(x) = exp(x[0]) * stock_b[t] is nonlinear.
    The Jacobian H = exp(x[0]) * stock_b[t].

    :param stock_a: observed stock A prices
    :param stock_b: observed stock B prices
    :param sigma_beta: beta drift volatility (process noise)
    :param R_val: measurement noise variance
    :return: (ekf_beta, ekf_beta_std) arrays
    """
    n = len(stock_a)
    ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1)
    # State: log_beta; initial estimate beta = 1 so log_beta = 0.
    ekf.x = np.array([[0.0]])
    ekf.F = np.array([[1.0]])
    ekf.P = np.array([[1.0]])
    ekf.R = np.array([[R_val]])
    ekf.Q = np.array([[sigma_beta ** 2]])
    ekf_beta = np.zeros(n)
    ekf_std = np.zeros(n)
    for t in range(n):
        sb = stock_b[t]

        def hx(x: np.ndarray, sb: float = sb) -> np.ndarray:
            return np.array([[np.exp(x[0, 0]) * sb]])

        def h_jacobian(x: np.ndarray, sb: float = sb) -> np.ndarray:
            return np.array([[np.exp(x[0, 0]) * sb]])

        ekf.predict()
        ekf.update(np.array([[stock_a[t]]]), h_jacobian, hx)
        ekf_beta[t] = np.exp(ekf.x[0, 0])
        ekf_std[t] = np.exp(ekf.x[0, 0]) * np.sqrt(ekf.P[0, 0])
    return ekf_beta, ekf_std


def _run_rolling_ols(
    stock_a: np.ndarray,
    stock_b: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """
    Compute rolling OLS beta between stock_a and stock_b.

    :param stock_a: dependent variable
    :param stock_b: independent variable
    :param window: rolling window size
    :return: rolling OLS beta array (NaN for first window - 1 points)
    """
    n = len(stock_a)
    beta = np.full(n, np.nan)
    for t in range(window - 1, n):
        x = stock_b[t - window + 1: t + 1]
        y = stock_a[t - window + 1: t + 1]
        # Simple OLS through origin: beta = sum(x*y) / sum(x^2).
        beta[t] = np.dot(x, y) / np.dot(x, x)
    return beta


def plot_ekf_pairs_trading(
    sigma_beta: float = 0.05, R_val: float = 1.0
) -> None:
    """
    Plot EKF pairs trading with time-varying beta tracking.

    Three panels: stock prices, true vs estimated beta, and the
    resulting spread with trading signal bands.

    :param sigma_beta: beta drift volatility
    :param R_val: price observation noise variance
    """
    n_days = 200
    days, stock_a, stock_b, true_beta = _simulate_pairs(
        n_days=n_days, sigma_beta=sigma_beta, R_val=R_val
    )
    ekf_beta, _ekf_std = _run_ekf_pairs(stock_a, stock_b, sigma_beta, R_val)
    rolling_beta = _run_rolling_ols(stock_a, stock_b)
    # Spread = stock_a - beta * stock_b.
    ekf_spread = stock_a - ekf_beta * stock_b
    ols_spread = stock_a - rolling_beta * stock_b
    # Trading signal bands: +/- 2 sigma of ekf spread.
    ekf_spread_std = float(np.nanstd(ekf_spread))
    rmse_ekf = _rmse(ekf_beta, true_beta)
    valid = ~np.isnan(rolling_beta)
    rmse_ols = _rmse(rolling_beta[valid], true_beta[valid])
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    # Panel 1: Stock prices.
    ax = axes[0]
    ax.plot(days, stock_a, "b-", lw=1.5, alpha=0.8, label="Stock A (e.g. XOM)")
    ax.plot(days, stock_b, "-", color="orange", lw=1.5, alpha=0.8, label="Stock B (e.g. CVX)")
    ax.set_xlim(0, n_days - 1)
    ax.set_ylim(80, 160)
    ax.set_ylabel("Price ($)")
    ax.set_title("Cointegrated Stock Pair (e.g. XOM vs CVX)")
    ax.legend(fontsize=9)
    # Panel 2: Beta estimation.
    ax = axes[1]
    ax.plot(days, true_beta, "b-", lw=2, label="True beta")
    ax.plot(days, ekf_beta, "g--", lw=1.8, label=f"EKF beta (RMSE={rmse_ekf:.3f})")
    ax.plot(
        days,
        rolling_beta,
        "r:",
        lw=1.8,
        label=f"Rolling OLS beta (RMSE={rmse_ols:.3f})",
    )
    ax.set_xlim(0, n_days - 1)
    ax.set_ylim(0.3, 2.5)
    ax.set_ylabel("Hedge Ratio beta")
    ax.set_title("True vs EKF Estimated Hedge Ratio beta")
    ax.legend(fontsize=9)
    # Panel 3: Spread.
    ax = axes[2]
    ax.plot(
        days, ekf_spread, "g-", lw=1.5, alpha=0.8, label="Spread (EKF beta)"
    )
    ax.plot(
        days, ols_spread, "r-", lw=1.2, alpha=0.6, label="Spread (OLS beta)"
    )
    ax.axhline(
        2 * ekf_spread_std, color="gray", linestyle="--", lw=1.2,
        label="+/- 2 sigma signal"
    )
    ax.axhline(-2 * ekf_spread_std, color="gray", linestyle="--", lw=1.2)
    ax.set_xlim(0, n_days - 1)
    ax.set_ylim(-15, 15)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Spread")
    ax.set_title("Pairs Trading Spread")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def show_ekf_pairs_trading_interactive() -> None:
    """
    Display interactive widget for EKF pairs trading beta estimation.

    Sliders control beta drift speed and observation noise.
    """
    sigma_slider = widgets.FloatSlider(
        value=0.05,
        min=0.001,
        max=0.2,
        step=0.005,
        description="Beta Drift sigma_beta:",
        style={"description_width": "200px"},
        layout=widgets.Layout(width="540px"),
    )
    r_slider = widgets.FloatSlider(
        value=1.0,
        min=0.1,
        max=10.0,
        step=0.1,
        description="Observation Noise R:",
        style={"description_width": "200px"},
        layout=widgets.Layout(width="540px"),
    )
    out = widgets.interactive_output(
        plot_ekf_pairs_trading,
        {"sigma_beta": sigma_slider, "R_val": r_slider},
    )
    display(widgets.VBox([sigma_slider, r_slider, out]))


# #############################################################################
# Cell 5: EKF - Linearization in Log-Return Space
# #############################################################################


def plot_ekf_linearization(
    x0: float = 1.0, sigma: float = 0.3
) -> None:
    """
    Show how EKF linearizes h(x)=log(x) and where the approximation breaks.

    Three panels: (1) nonlinear log function with tangent line at x0,
    (2) input Gaussian distribution, (3) true vs EKF output distribution
    from Monte Carlo.

    :param x0: current price estimate (operating point)
    :param sigma: input price uncertainty standard deviation
    """
    n_mc = 5000
    rng = np.random.default_rng(42)
    # Monte Carlo samples for input.
    x_samples = rng.normal(x0, sigma, size=n_mc)
    # Only positive values can be log-transformed.
    x_pos = x_samples[x_samples > 0.001]
    y_mc = np.log(x_pos)
    # EKF Gaussian approximation: mean = log(x0), var = (sigma/x0)^2.
    ekf_mean = np.log(x0)
    ekf_std = sigma / x0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # Panel 1: Nonlinear function h(x) = log(x) with tangent.
    ax = axes[0]
    x_range = np.linspace(0.1, 5, 200)
    ax.plot(x_range, np.log(x_range), "b-", lw=2, label="h(x) = log(x)")
    # Tangent line at x0: log(x0) + (1/x0)*(x - x0).
    tangent = np.log(x0) + (1.0 / x0) * (x_range - x0)
    ax.plot(x_range, tangent, "g--", lw=1.8, label=f"Jacobian at x0={x0:.1f}")
    ax.axvline(x0, color="gray", linestyle=":", lw=1.2)
    # Show divergence arrow.
    ax.annotate(
        "Error grows\naway from x0",
        xy=(3.5, np.log(3.5)),
        xytext=(2.5, -2.0),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=8,
        color="red",
    )
    ax.set_xlim(0.1, 5)
    ax.set_ylim(-3, 2)
    ax.set_xlabel("x (price)")
    ax.set_ylabel("log(x)")
    ax.set_title("EKF Linearizes log(x) at Current Estimate x0")
    ax.legend(fontsize=9)
    # Panel 2: Input Gaussian distribution.
    ax = axes[1]
    x_plot = np.linspace(max(0.05, x0 - 4 * sigma), x0 + 4 * sigma, 300)
    ax.fill_between(
        x_plot,
        norm.pdf(x_plot, x0, sigma),
        alpha=0.4,
        color="blue",
        label=f"p(x) = N({x0:.1f}, {sigma:.2f}^2)",
    )
    ax.plot(x_plot, norm.pdf(x_plot, x0, sigma), "b-", lw=1.5)
    ax.axvline(x0, color="navy", linestyle="--", lw=1.2, label=f"x0={x0:.1f}")
    ax.set_xlim(0.1, 5)
    ax.set_xlabel("x (price)")
    ax.set_ylabel("Density")
    ax.set_title("Input Distribution: Price Uncertainty")
    ax.legend(fontsize=9)
    # Panel 3: True vs EKF output distribution.
    ax = axes[2]
    y_range = np.linspace(-4, 2, 300)
    # True output distribution (Monte Carlo histogram).
    ax.hist(
        y_mc,
        bins=60,
        density=True,
        alpha=0.5,
        color="blue",
        label="True output (Monte Carlo)",
    )
    # EKF Gaussian approximation.
    ax.plot(
        y_range,
        norm.pdf(y_range, ekf_mean, ekf_std),
        "g--",
        lw=2,
        label=f"EKF approx N({ekf_mean:.2f}, {ekf_std:.2f}^2)",
    )
    ax.set_xlim(-4, 2)
    ax.set_xlabel("log(x)")
    ax.set_ylabel("Density")
    ax.set_title("True vs EKF Approximated Output Distribution")
    ax.legend(fontsize=9)
    # Comment annotation.
    bias = float(np.mean(y_mc)) - ekf_mean
    ax.text(
        0.02,
        0.97,
        f"EKF bias = {bias:.3f}\n(underestimates skewness)",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
    )
    plt.tight_layout()
    plt.show()


def show_ekf_linearization_interactive() -> None:
    """
    Display interactive widget for EKF linearization visualization.

    Sliders control operating point x0 and input uncertainty sigma.
    """
    x0_slider = widgets.FloatSlider(
        value=1.0,
        min=0.2,
        max=4.0,
        step=0.1,
        description="Price Estimate x0:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="520px"),
    )
    sigma_slider = widgets.FloatSlider(
        value=0.3,
        min=0.05,
        max=1.5,
        step=0.05,
        description="Price Uncertainty sigma:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="520px"),
    )
    out = widgets.interactive_output(
        plot_ekf_linearization,
        {"x0": x0_slider, "sigma": sigma_slider},
    )
    display(widgets.VBox([x0_slider, sigma_slider, out]))


# #############################################################################
# Cell 6: Unscented Kalman Filter - Stochastic Volatility Estimation
# #############################################################################


def _simulate_stochastic_vol(
    n_days: int = 200,
    kappa: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate log-returns from a simple stochastic volatility model.

    State: [log_price, log_vol]; log_vol follows a mean-reverting AR(1).

    :param n_days: number of trading days
    :param kappa: volatility-of-volatility parameter
    :param seed: random seed
    :return: (days, log_returns, true_vol) arrays
    """
    rng = np.random.default_rng(seed)
    days = np.arange(n_days, dtype=float)
    # Log-vol AR(1) process.
    log_vol = np.zeros(n_days)
    log_vol[0] = np.log(0.15)
    for t in range(1, n_days):
        # Mean reversion to log(0.15) with volatility kappa.
        log_vol[t] = (
            0.9 * log_vol[t - 1]
            + 0.1 * np.log(0.15)
            + kappa * rng.normal()
        )
    true_vol = np.exp(log_vol)
    log_returns = true_vol * rng.normal(size=n_days)
    return days, log_returns, true_vol


def _run_ukf_volatility(
    log_returns: np.ndarray,
    alpha_param: float,
    kappa_param: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run UKF to estimate latent volatility from observed log-returns.

    State: [log_vol]; measurement: |log_return| as volatility proxy.

    :param log_returns: observed log-return series
    :param alpha_param: sigma-point spread parameter alpha
    :param kappa_param: secondary scaling parameter kappa
    :return: (ukf_vol, ukf_vol_std) arrays
    """
    n = len(log_returns)
    # UKF for scalar state.
    points = MerweScaledSigmaPoints(
        n=1, alpha=alpha_param, beta=2.0, kappa=kappa_param
    )

    def fx(x: np.ndarray, dt: float) -> np.ndarray:
        # AR(1) update for log_vol.
        return np.array([0.9 * x[0] + 0.1 * np.log(0.15)])

    def hx(x: np.ndarray) -> np.ndarray:
        # Observation: expected absolute return = vol * sqrt(2/pi).
        return np.array([np.exp(x[0]) * np.sqrt(2.0 / np.pi)])

    ukf = UnscentedKalmanFilter(
        dim_x=1,
        dim_z=1,
        dt=1.0,
        fx=fx,
        hx=hx,
        points=points,
    )
    ukf.x = np.array([np.log(0.15)])
    ukf.P = np.eye(1) * 1.0
    ukf.R = np.array([[0.01]])
    ukf.Q = np.array([[0.01]])
    ukf_vol = np.zeros(n)
    ukf_std = np.zeros(n)
    for t in range(n):
        ukf.predict()
        ukf.update(np.array([np.abs(log_returns[t])]))
        ukf_vol[t] = np.exp(ukf.x[0])
        ukf_std[t] = ukf_vol[t] * np.sqrt(ukf.P[0, 0])
    return ukf_vol, ukf_std


def _run_ekf_volatility(
    log_returns: np.ndarray,
    kappa_param: float,
) -> np.ndarray:
    """
    Run EKF to estimate latent volatility from log-returns.

    :param log_returns: observed log-return series
    :param kappa_param: process noise level
    :return: ekf_vol array
    """
    n = len(log_returns)
    ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1)
    ekf.x = np.array([[np.log(0.15)]])
    ekf.F = np.array([[0.9]])
    ekf.P = np.array([[1.0]])
    ekf.R = np.array([[0.01]])
    ekf.Q = np.array([[kappa_param ** 2 + 0.001]])
    ekf_vol = np.zeros(n)
    for t in range(n):
        def hx(x: np.ndarray) -> np.ndarray:
            return np.array([[np.exp(x[0, 0]) * np.sqrt(2.0 / np.pi)]])

        def h_jacobian(x: np.ndarray) -> np.ndarray:
            return np.array([[np.exp(x[0, 0]) * np.sqrt(2.0 / np.pi)]])

        ekf.predict()
        ekf.update(np.array([[np.abs(log_returns[t])]]), h_jacobian, hx)
        ekf_vol[t] = np.exp(ekf.x[0, 0])
    return ekf_vol


def _run_garch_volatility(log_returns: np.ndarray) -> np.ndarray:
    """
    Estimate volatility using a simple GARCH(1,1) recursion.

    :param log_returns: observed log-return series
    :return: garch_vol array
    """
    n = len(log_returns)
    omega = 0.0001
    alpha_g = 0.1
    beta_g = 0.85
    h = np.zeros(n)
    h[0] = np.var(log_returns[:10]) if len(log_returns) >= 10 else 0.01
    for t in range(1, n):
        h[t] = omega + alpha_g * log_returns[t - 1] ** 2 + beta_g * h[t - 1]
    return np.sqrt(h)


def _compute_sigma_points_2d(
    mean: np.ndarray,
    cov: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Compute 2n+1 Merwe sigma points for a 2D state.

    :param mean: state mean vector (2,)
    :param cov: state covariance (2, 2)
    :param alpha: sigma point spread
    :return: sigma points array of shape (5, 2)
    """
    n = 2
    kappa = 0.0
    lam = alpha ** 2 * (n + kappa) - n
    scale = np.sqrt(n + lam)
    L = np.linalg.cholesky(cov)
    pts = np.zeros((2 * n + 1, 2))
    pts[0] = mean
    for i in range(n):
        pts[i + 1] = mean + scale * L[:, i]
        pts[n + i + 1] = mean - scale * L[:, i]
    return pts


def plot_ukf_volatility(
    alpha_param: float = 0.1, kappa_param: float = 0.1
) -> None:
    """
    Plot UKF stochastic volatility estimation with sigma points.

    Three panels: (1) sigma points in state space, (2) propagated sigma
    points, (3) volatility estimate time series vs EKF and GARCH.

    :param alpha_param: sigma point spread parameter alpha
    :param kappa_param: volatility-of-volatility process noise
    """
    n_days = 200
    days, log_returns, true_vol = _simulate_stochastic_vol(
        n_days=n_days, kappa=kappa_param
    )
    ukf_vol, _ukf_std = _run_ukf_volatility(log_returns, alpha_param, kappa_param)
    ekf_vol = _run_ekf_volatility(log_returns, kappa_param)
    garch_vol = _run_garch_volatility(log_returns)
    # Sigma points illustration around a representative 2D state.
    state_mean = np.array([np.log(0.15), np.log(0.15) * 0.9 + 0.1 * np.log(0.15)])
    state_cov = np.array([[0.5, 0.1], [0.1, 0.5]])
    rng = np.random.default_rng(0)
    mc_samples = rng.multivariate_normal(state_mean, state_cov, size=300)
    sigma_pts = _compute_sigma_points_2d(state_mean, state_cov, alpha_param)
    # Propagated sigma points via log-vol transition.
    def _prop(pt: np.ndarray) -> np.ndarray:
        return np.array([
            0.9 * pt[0] + 0.1 * np.log(0.15),
            0.9 * pt[1] + 0.1 * np.log(0.15),
        ])
    sigma_pts_prop = np.array([_prop(p) for p in sigma_pts])
    mc_prop = np.array([_prop(p) for p in mc_samples])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    # Panel 1: Sigma points in state space.
    ax = axes[0]
    ax.scatter(
        mc_samples[:, 0],
        mc_samples[:, 1],
        s=8,
        alpha=0.4,
        color="lightblue",
        label="Monte Carlo samples",
    )
    ax.scatter(
        sigma_pts[:, 0],
        sigma_pts[:, 1],
        s=100,
        color="red",
        marker="x",
        lw=2,
        label="Sigma points",
    )
    ax.scatter(state_mean[0], state_mean[1], s=80, color="navy", zorder=5)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("log-price")
    ax.set_ylabel("log-vol")
    ax.set_title("Sigma Points in [log-price, log-vol] State Space")
    ax.legend(fontsize=8)
    # Panel 2: Propagated sigma points.
    ax = axes[1]
    ax.scatter(
        mc_prop[:, 0],
        mc_prop[:, 1],
        s=8,
        alpha=0.4,
        color="lightblue",
        label="Monte Carlo propagated",
    )
    ax.scatter(
        sigma_pts_prop[:, 0],
        sigma_pts_prop[:, 1],
        s=100,
        color="red",
        marker="x",
        lw=2,
        label="Sigma pts propagated",
    )
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("log-price")
    ax.set_ylabel("log-vol")
    ax.set_title("Propagated Sigma Points vs True Distribution")
    ax.legend(fontsize=8)
    # Panel 3: Volatility time series.
    ax = axes[2]
    rmse_ukf = _rmse(ukf_vol, true_vol)
    rmse_ekf = _rmse(ekf_vol, true_vol)
    rmse_garch = _rmse(garch_vol, true_vol)
    ax.plot(days, true_vol, "b-", lw=2, label="True latent volatility")
    ax.plot(days, ukf_vol, "g--", lw=1.8, label=f"UKF vol (RMSE={rmse_ukf:.4f})")
    ax.plot(days, ekf_vol, "r:", lw=1.8, label=f"EKF vol (RMSE={rmse_ekf:.4f})")
    ax.plot(days, garch_vol, color="gray", linestyle="--", lw=1.4,
            label=f"GARCH(1,1) (RMSE={rmse_garch:.4f})")
    ax.set_xlim(0, n_days - 1)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Volatility")
    ax.set_title("Volatility Estimation: UKF vs EKF vs GARCH")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.show()


def show_ukf_volatility_interactive() -> None:
    """
    Display interactive widget for UKF stochastic volatility estimation.

    Sliders control sigma-point spread alpha and volatility-of-volatility
    kappa.
    """
    alpha_slider = widgets.FloatSlider(
        value=0.1,
        min=0.001,
        max=1.0,
        step=0.01,
        description="Sigma Point Spread alpha:",
        style={"description_width": "210px"},
        layout=widgets.Layout(width="560px"),
    )
    kappa_slider = widgets.FloatSlider(
        value=0.1,
        min=0.01,
        max=0.5,
        step=0.01,
        description="Vol of Volatility kappa:",
        style={"description_width": "210px"},
        layout=widgets.Layout(width="560px"),
    )
    out = widgets.interactive_output(
        plot_ukf_volatility,
        {"alpha_param": alpha_slider, "kappa_param": kappa_slider},
    )
    display(widgets.VBox([alpha_slider, kappa_slider, out]))


# #############################################################################
# Cell 7: UKF vs EKF - Volatility Estimation Under Market Stress
# #############################################################################


def _simulate_stress_returns(
    n_days: int = 200,
    spike_magnitude: float = 4.0,
    spike_duration: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Simulate log-returns with a volatility spike in the middle.

    :param n_days: total number of trading days
    :param spike_magnitude: factor by which volatility increases during stress
    :param spike_duration: number of days in the stress period
    :param seed: random seed
    :return: (days, log_returns, spike_start, spike_end)
    """
    rng = np.random.default_rng(seed)
    days = np.arange(n_days, dtype=float)
    base_vol = 0.01
    spike_start = n_days // 2 - spike_duration // 2
    spike_end = spike_start + spike_duration
    vol = np.full(n_days, base_vol)
    vol[spike_start:spike_end] = base_vol * spike_magnitude
    log_returns = vol * rng.normal(size=n_days)
    return days, log_returns, spike_start, spike_end


def _true_stress_vol(
    n_days: int,
    base_vol: float,
    spike_magnitude: float,
    spike_start: int,
    spike_end: int,
) -> np.ndarray:
    """
    Return the true underlying volatility profile used in stress simulation.

    :param n_days: total days
    :param base_vol: baseline volatility
    :param spike_magnitude: spike multiplier
    :param spike_start: start of spike
    :param spike_end: end of spike
    :return: true_vol array
    """
    true_vol = np.full(n_days, base_vol)
    true_vol[spike_start:spike_end] = base_vol * spike_magnitude
    return true_vol


def plot_ukf_vs_ekf_stress(
    spike_magnitude: float = 4.0, spike_duration: int = 10
) -> None:
    """
    Compare UKF and EKF on volatility estimation during market stress.

    Three panels: simulated returns, EKF volatility estimate, UKF
    volatility estimate.  Stress period is highlighted in all panels.

    :param spike_magnitude: factor by which volatility spikes
    :param spike_duration: number of days in the stress period
    """
    n_days = 200
    days, log_returns, spike_start, spike_end = _simulate_stress_returns(
        n_days=n_days,
        spike_magnitude=spike_magnitude,
        spike_duration=int(spike_duration),
    )
    base_vol = 0.01
    true_vol = _true_stress_vol(
        n_days, base_vol, spike_magnitude, spike_start, spike_end
    )
    alpha_param = 0.1
    kappa_param = 0.1
    ukf_vol, _ukf_std = _run_ukf_volatility(log_returns, alpha_param, kappa_param)
    ekf_vol = _run_ekf_volatility(log_returns, kappa_param)
    rmse_ukf = _rmse(ukf_vol, true_vol)
    rmse_ekf = _rmse(ekf_vol, true_vol)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    # Shared stress region shading helper.
    def _shade(ax: plt.Axes) -> None:
        ax.axvspan(
            spike_start, spike_end, alpha=0.15, color="red", label="Stress period"
        )
    # Panel 1: Returns.
    ax = axes[0]
    ax.plot(days, log_returns, "b-", lw=0.8, alpha=0.8, label="Log returns")
    _shade(ax)
    ax.set_xlim(0, n_days - 1)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Log Return")
    ax.set_title("Simulated Returns with Volatility Spike")
    ax.legend(fontsize=8)
    # Panel 2: EKF volatility.
    ax = axes[1]
    ax.plot(days, true_vol, "b-", lw=2, label="True volatility")
    ax.plot(days, ekf_vol, "g-", lw=1.8, label=f"EKF vol (RMSE={rmse_ekf:.4f})")
    _shade(ax)
    ax.annotate(
        "EKF falls\nbehind",
        xy=(spike_start + spike_duration // 4, base_vol * spike_magnitude * 0.6),
        xytext=(spike_start - 20, base_vol * spike_magnitude * 0.9),
        arrowprops=dict(arrowstyle="->", color="darkred"),
        fontsize=8,
        color="darkred",
    )
    ax.set_xlim(0, n_days - 1)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Volatility")
    ax.set_title("EKF Volatility Estimate")
    ax.legend(fontsize=8)
    # Panel 3: UKF volatility.
    ax = axes[2]
    ax.plot(days, true_vol, "b-", lw=2, label="True volatility")
    ax.plot(days, ukf_vol, "g-", lw=1.8, label=f"UKF vol (RMSE={rmse_ukf:.4f})")
    _shade(ax)
    ax.set_xlim(0, n_days - 1)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Volatility")
    ax.set_title("UKF Volatility Estimate")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    # Comment box.
    fig2, ax2 = plt.subplots(figsize=(10, 1.5))
    ax2.axis("off")
    comment = (
        "UKF tracks the volatility spike more accurately because sigma "
        "points capture\nthe nonlinear amplification during stress.  "
        "EKF linearization underestimates tail risk."
        f"  UKF RMSE={rmse_ukf:.4f}, EKF RMSE={rmse_ekf:.4f}."
    )
    ax2.text(
        0.5,
        0.5,
        comment,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="lightyellow",
            edgecolor="gray",
        ),
    )
    ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.show()


def show_ukf_vs_ekf_stress_interactive() -> None:
    """
    Display interactive widget for UKF vs EKF market stress comparison.

    Sliders control the spike magnitude and stress period duration.
    """
    magnitude_slider = widgets.FloatSlider(
        value=4.0,
        min=1.0,
        max=10.0,
        step=0.5,
        description="Volatility Spike Size:",
        style={"description_width": "200px"},
        layout=widgets.Layout(width="540px"),
    )
    duration_slider = widgets.IntSlider(
        value=10,
        min=5,
        max=40,
        step=1,
        description="Stress Duration (days):",
        style={"description_width": "200px"},
        layout=widgets.Layout(width="540px"),
    )
    out = widgets.interactive_output(
        plot_ukf_vs_ekf_stress,
        {
            "spike_magnitude": magnitude_slider,
            "spike_duration": duration_slider,
        },
    )
    display(widgets.VBox([magnitude_slider, duration_slider, out]))


# #############################################################################
# Cell 8: Ensemble Kalman Filter - Portfolio Risk Scenario Analysis
# #############################################################################


def _run_enkf_portfolio(
    n_ensemble: int = 100,
    Q_val: float = 0.1,
    n_days: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run EnKF on a portfolio price state and return ensemble trajectories.

    :param n_ensemble: number of ensemble members
    :param Q_val: process noise variance
    :param n_days: number of trading days
    :param seed: random seed
    :return: (days, true_price, ensemble_mean, ensemble_std, particles_final)
    """
    rng = np.random.default_rng(seed)
    days, true_trend, noisy_prices = _simulate_price_trend(
        n_days=n_days, seed=seed, obs_noise_std=2.0
    )

    def fx(x: np.ndarray, dt: float) -> np.ndarray:
        return x + rng.normal(0, np.sqrt(Q_val), size=x.shape)

    def hx(x: np.ndarray) -> np.ndarray:
        return x.copy()

    enkf = EnsembleKalmanFilter(
        x=np.array([noisy_prices[0]]),
        P=np.array([[100.0]]),
        dim_z=1,
        dt=1.0,
        N=n_ensemble,
        hx=hx,
        fx=fx,
    )
    enkf.R = np.array([[4.0]])
    enc_mean = np.zeros(n_days)
    enc_std = np.zeros(n_days)
    # Store all ensemble particles over time for scatter plot.
    all_particles = []
    for i, z in enumerate(noisy_prices):
        enkf.predict()
        enkf.update(np.array([z]))
        enc_mean[i] = np.mean(enkf.ensemble[:, 0])
        enc_std[i] = np.std(enkf.ensemble[:, 0])
        if i % 5 == 0:
            all_particles.append((i, enkf.ensemble[:, 0].copy()))
    # Final ensemble.
    particles_final = enkf.ensemble[:, 0].copy()
    return days, true_trend, enc_mean, enc_std, particles_final, all_particles


def plot_enkf_portfolio(
    n_ensemble: int = 100, Q_val: float = 0.1
) -> None:
    """
    Plot EnKF portfolio risk scenario analysis.

    Three panels: (1) ensemble particles scatter over time, (2) price
    estimate with risk bands, (3) final portfolio price distribution
    with Value at Risk.

    :param n_ensemble: number of ensemble particles
    :param Q_val: process noise variance
    """
    result = _run_enkf_portfolio(n_ensemble=n_ensemble, Q_val=Q_val)
    days, true_trend, enc_mean, enc_std, particles_final, all_particles = result
    var_5 = float(np.percentile(particles_final, 5))
    var_95 = float(np.percentile(particles_final, 95))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    # Panel 1: Ensemble particle scatter over time.
    ax = axes[0]
    n_snaps = len(all_particles)
    cmap = plt.cm.coolwarm  # noqa: PD003
    for k, (t_idx, pts) in enumerate(all_particles):
        color = cmap(k / max(n_snaps - 1, 1))
        ax.scatter(
            np.full(len(pts), t_idx),
            pts,
            s=3,
            alpha=0.3,
            color=color,
        )
    ax.plot(
        days[::5],
        enc_mean[::5],
        "w-",
        lw=2.5,
        label="Ensemble mean",
    )
    ax.set_xlim(0, 99)
    ax.set_ylim(80, 140)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Price ($)")
    ax.set_title("Ensemble Particles: Portfolio State Over Time")
    ax.legend(fontsize=9)
    # Panel 2: Price estimate with risk bands.
    ax = axes[1]
    ax.plot(days, true_trend, "b-", lw=2, label="True price")
    ax.plot(days, enc_mean, "g-", lw=1.8, label="Ensemble mean")
    ax.fill_between(
        days,
        enc_mean - enc_std,
        enc_mean + enc_std,
        alpha=0.3,
        color="green",
        label="1-sigma band",
    )
    ax.fill_between(
        days,
        enc_mean - 2 * enc_std,
        enc_mean + 2 * enc_std,
        alpha=0.15,
        color="orange",
        label="2-sigma band",
    )
    ax.annotate(
        "VaR region",
        xy=(80, float(enc_mean[80] - 2 * enc_std[80])),
        xytext=(60, 85),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=8,
        color="red",
    )
    ax.set_xlim(0, 99)
    ax.set_ylim(80, 140)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Price ($)")
    ax.set_title("Ensemble Price Estimate with Risk Bands")
    ax.legend(fontsize=8)
    # Panel 3: Final distribution with VaR.
    ax = axes[2]
    ax.hist(
        particles_final,
        bins=30,
        density=True,
        color="steelblue",
        alpha=0.6,
        label="Final particle distribution",
    )
    ax.axvline(
        var_5,
        color="red",
        linestyle="--",
        lw=1.8,
        label=f"5th pct VaR = {var_5:.1f}",
    )
    ax.axvline(
        var_95,
        color="navy",
        linestyle="--",
        lw=1.8,
        label=f"95th pct = {var_95:.1f}",
    )
    # Shade the left tail.
    x_tail = np.linspace(
        float(particles_final.min()), var_5, 100
    )
    ax.fill_between(
        x_tail,
        0,
        np.ones_like(x_tail) * 0.05,
        alpha=0.3,
        color="red",
        label="VaR tail",
    )
    ax.set_xlabel("Final Price ($)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Final Portfolio Price Distribution (5th pct VaR={var_5:.1f})"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def show_enkf_portfolio_interactive() -> None:
    """
    Display interactive widget for EnKF portfolio risk scenario analysis.

    Sliders control ensemble size N and process noise Q.
    """
    n_slider = widgets.IntSlider(
        value=100,
        min=10,
        max=500,
        step=10,
        description="Ensemble Size N:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="520px"),
    )
    q_slider = widgets.FloatSlider(
        value=0.1,
        min=0.001,
        max=1.0,
        step=0.01,
        description="Process Noise Q:",
        style={"description_width": "180px"},
        layout=widgets.Layout(width="520px"),
    )
    out = widgets.interactive_output(
        plot_enkf_portfolio,
        {"n_ensemble": n_slider, "Q_val": q_slider},
    )
    display(widgets.VBox([n_slider, q_slider, out]))


# #############################################################################
# Cell 9: All Four Filters - Financial State Estimation Comparison
# #############################################################################


def _run_all_filters_financial(
    n_days: int = 100, seed: int = 42
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Run all four filters on the same financial state estimation problem.

    :param n_days: number of trading days
    :param seed: random seed
    :return: dict mapping filter name to (estimates, stds)
    """
    days, true_trend, noisy_prices = _simulate_price_trend(
        n_days=n_days, seed=seed, obs_noise_std=5.0
    )
    results = {}
    # --- KF ---
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[noisy_prices[0]], [0.0]])
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.P = np.eye(2) * 500.0
    kf.R = np.array([[25.0]])
    kf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=0.1)
    kf_est = np.zeros(n_days)
    kf_std = np.zeros(n_days)
    for i, z in enumerate(noisy_prices):
        kf.predict()
        kf.update(np.array([[z]]))
        kf_est[i] = kf.x[0, 0]
        kf_std[i] = np.sqrt(kf.P[0, 0])
    results["KF"] = (kf_est, kf_std)
    # --- EKF ---
    ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
    ekf.x = np.array([[noisy_prices[0]], [0.0]])
    ekf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    ekf.P = np.eye(2) * 500.0
    ekf.R = np.array([[25.0]])
    ekf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=0.1)
    ekf_est = np.zeros(n_days)
    ekf_std = np.zeros(n_days)
    for i, z in enumerate(noisy_prices):
        def hx_ekf(x: np.ndarray) -> np.ndarray:
            return np.array([[x[0, 0]]])

        def hj_ekf(x: np.ndarray) -> np.ndarray:
            return np.array([[1.0, 0.0]])

        ekf.predict()
        ekf.update(np.array([[z]]), hj_ekf, hx_ekf)
        ekf_est[i] = ekf.x[0, 0]
        ekf_std[i] = np.sqrt(ekf.P[0, 0])
    results["EKF"] = (ekf_est, ekf_std)
    # --- UKF ---
    points = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2.0, kappa=0.0)

    def fx_ukf(x: np.ndarray, dt: float) -> np.ndarray:
        return np.array([x[0] + x[1], x[1]])

    def hx_ukf(x: np.ndarray) -> np.ndarray:
        return np.array([x[0]])

    ukf = UnscentedKalmanFilter(
        dim_x=2, dim_z=1, dt=1.0, fx=fx_ukf, hx=hx_ukf, points=points
    )
    ukf.x = np.array([noisy_prices[0], 0.0])
    ukf.P = np.eye(2) * 500.0
    ukf.R = np.array([[25.0]])
    ukf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=0.1)
    ukf_est = np.zeros(n_days)
    ukf_std = np.zeros(n_days)
    for i, z in enumerate(noisy_prices):
        ukf.predict()
        ukf.update(np.array([z]))
        ukf_est[i] = ukf.x[0]
        ukf_std[i] = np.sqrt(ukf.P[0, 0])
    results["UKF"] = (ukf_est, ukf_std)
    # --- EnKF ---
    rng = np.random.default_rng(seed)

    def fx_enkf(x: np.ndarray, dt: float) -> np.ndarray:
        return np.array([x[0] + x[1] + rng.normal(0, 0.05),
                         x[1] + rng.normal(0, 0.05)])

    def hx_enkf(x: np.ndarray) -> np.ndarray:
        return np.array([x[0]])

    enkf2 = EnsembleKalmanFilter(
        x=np.array([noisy_prices[0], 0.0]),
        P=np.eye(2) * 500.0,
        dim_z=1,
        dt=1.0,
        N=100,
        hx=hx_enkf,
        fx=fx_enkf,
    )
    enkf2.R = np.array([[25.0]])
    enkf_est = np.zeros(n_days)
    enkf_std = np.zeros(n_days)
    for i, z in enumerate(noisy_prices):
        enkf2.predict()
        enkf2.update(np.array([z]))
        enkf_est[i] = np.mean(enkf2.ensemble[:, 0])
        enkf_std[i] = np.std(enkf2.ensemble[:, 0])
    results["EnKF"] = (enkf_est, enkf_std)
    return days, true_trend, noisy_prices, results


def plot_all_filters_financial_comparison() -> None:
    """
    Compare all four Kalman filter variants on the same financial problem.

    Four panels (KF, EKF, UKF, EnKF) show price tracking with 1-sigma
    bands and RMSE.  A bar chart compares RMSE across filters, and a
    decision flowchart guides filter selection in practice.
    """
    days, true_trend, noisy_prices, results = _run_all_filters_financial()
    colors = {"KF": "green", "EKF": "darkorange", "UKF": "purple", "EnKF": "red"}
    filter_names = ["KF", "EKF", "UKF", "EnKF"]
    rmse_vals = {
        name: _rmse(results[name][0], true_trend) for name in filter_names
    }
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, name in zip(axes, filter_names):
        ests, stds = results[name]
        c = colors[name]
        ax.plot(days, true_trend, "b-", lw=1.8, label="True trend")
        ax.scatter(
            days, noisy_prices, s=8, alpha=0.4, color="orange",
            label="Observed"
        )
        ax.plot(
            days, ests, "-", color=c, lw=2,
            label=f"{name} (RMSE={rmse_vals[name]:.2f})",
        )
        ax.fill_between(
            days, ests - stds, ests + stds, alpha=0.2, color=c
        )
        ax.set_xlim(0, 99)
        ax.set_ylim(80, 140)
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("Price ($)")
        ax.set_title(name)
        ax.legend(fontsize=7)
    plt.suptitle(
        "All Four Kalman Filter Variants on Financial Price Estimation",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
    # RMSE bar chart.
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    bar_colors = [colors[n] for n in filter_names]
    bars = ax2.bar(filter_names, [rmse_vals[n] for n in filter_names],
                   color=bar_colors, alpha=0.8, edgecolor="black")
    for bar, name in zip(bars, filter_names):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rmse_vals[name]:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax2.set_ylabel("RMSE")
    ax2.set_title("RMSE Comparison: KF vs EKF vs UKF vs EnKF")
    plt.tight_layout()
    plt.show()
    # Decision flowchart as text diagram.
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.axis("off")
    flowchart = (
        "Is the model linear?\n"
        "  Yes -> use KF  (optimal, fastest)\n"
        "  No ->\n"
        "    Is nonlinearity mild?\n"
        "      Yes -> use EKF  (fast, requires Jacobians)\n"
        "      No ->\n"
        "        Is nonlinearity strong?\n"
        "          Yes -> use UKF  (better accuracy, no Jacobians, ~3x EKF cost)\n"
        "          Do you need the full distribution?\n"
        "            Yes -> use EnKF  (most flexible, scales with N)"
    )
    ax3.text(
        0.02,
        0.95,
        flowchart,
        ha="left",
        va="top",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="lightyellow",
            edgecolor="gray",
        ),
    )
    ax3.set_title(
        "Filter Selection Guide", fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()
