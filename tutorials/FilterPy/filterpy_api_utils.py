"""
filterpy_api_utils.py

Utility functions for the FilterPy API tutorial notebook.

- All widget and plotting logic lives here; the notebook only calls these
  functions.
- Covers: Linear Kalman Filter, Extended Kalman Filter (EKF), Unscented Kalman
  Filter (UKF), and Ensemble Kalman Filter (EnKF).

Import as:

import tutorials.FilterPy.filterpy_api_utils as utils
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
from matplotlib.patches import FancyArrowPatch

_LOG = logging.getLogger(__name__)

# #############################################################################
# Shared helpers
# #############################################################################


def _make_1d_signal(
    n_steps: int = 50, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a 1D true position signal and its noisy measurements.

    :param n_steps: number of time steps
    :param seed: random seed for reproducibility
    :return: (true_positions, time_array) arrays of shape (n_steps,)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=float)
    # True position: linear drift + sinusoidal component.
    true_pos = 0.4 * t + 3.0 * np.sin(0.3 * t)
    return true_pos, t


def _add_measurement_noise(
    true_signal: np.ndarray, r_std: float, seed: int = 42
) -> np.ndarray:
    """
    Add Gaussian measurement noise to a signal.

    :param true_signal: array of true values
    :param r_std: standard deviation of measurement noise
    :param seed: random seed
    :return: noisy measurements
    """
    rng = np.random.default_rng(seed)
    return true_signal + rng.normal(0, r_std, size=true_signal.shape)


def _rmse(estimated: np.ndarray, true: np.ndarray) -> float:
    """
    Compute root-mean-square error.

    :param estimated: estimated values
    :param true: true values
    :return: RMSE scalar
    """
    return float(np.sqrt(np.mean((estimated - true) ** 2)))


# #############################################################################
# Cell 1: Predict-Update Cycle Diagram
# #############################################################################


def plot_predict_update_diagram() -> None:
    """
    Draw a static diagram showing the Kalman filter predict-update cycle.

    Shows the flow: Prior State -> Predict -> Predicted State -> Update ->
    Posterior State, with process noise Q and measurement noise R feeding in.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    # Draw boxes.
    box_params = dict(
        boxstyle="round,pad=0.4",
        facecolor="lightblue",
        edgecolor="steelblue",
        linewidth=2,
    )
    boxes = {
        "prior": (1.0, 2.5, "Prior State\nx(t-1), P(t-1)"),
        "predict": (3.5, 2.5, "PREDICT\nx- = F x\nP- = F P F.T + Q"),
        "update": (6.5, 2.5, "UPDATE\nK = P- H.T inv(S)\nx = x- + K(z - Hx-)\nP = (I-KH)P-"),
        "posterior": (9.0, 2.5, "Posterior\nx(t), P(t)"),
    }
    for key, (cx, cy, label) in boxes.items():
        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=box_params,
        )
    # Draw arrows between boxes.
    arrow_style = dict(
        arrowstyle="->", color="steelblue", lw=2, mutation_scale=20
    )
    connections = [
        ((1.7, 2.5), (2.6, 2.5)),
        ((4.4, 2.5), (5.5, 2.5)),
        ((7.5, 2.5), (8.5, 2.5)),
    ]
    for start, end in connections:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=2),
        )
    # Draw noise inputs.
    ax.annotate(
        "",
        xy=(3.5, 3.4),
        xytext=(3.5, 4.5),
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=2),
    )
    ax.text(3.5, 4.7, "Process noise Q", ha="center", color="darkorange", fontsize=9)
    ax.annotate(
        "",
        xy=(6.5, 3.5),
        xytext=(6.5, 4.5),
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2),
    )
    ax.text(6.5, 4.7, "Measurement z\n(noise R)", ha="center", color="darkgreen", fontsize=9)
    ax.set_title(
        "Kalman Filter: Predict-Update Cycle",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def show_matrix_table() -> None:
    """
    Display a summary table of Kalman filter matrices and their roles.
    """
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    headers = ["Matrix", "Shape", "Description"]
    rows = [
        ["F", "(dim_x, dim_x)", "State transition: propagates state forward in time"],
        ["H", "(dim_z, dim_x)", "Measurement function: maps state to measurement space"],
        ["Q", "(dim_x, dim_x)", "Process noise covariance: model uncertainty"],
        ["R", "(dim_z, dim_z)", "Measurement noise covariance: sensor uncertainty"],
        ["P", "(dim_x, dim_x)", "State covariance: current estimate uncertainty"],
        ["x", "(dim_x, 1)", "State vector: current best estimate"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="left",
        loc="center",
        colWidths=[0.08, 0.25, 0.6],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    ax.set_title(
        "Kalman Filter Matrices", fontsize=13, fontweight="bold", pad=15
    )
    plt.tight_layout()
    plt.show()


# #############################################################################
# Cell 2: Linear Kalman Filter - 1D Tracking
# #############################################################################


def _run_linear_kf_1d(
    true_pos: np.ndarray,
    measurements: np.ndarray,
    r_var: float,
    q_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a 1D linear Kalman filter tracking position and velocity.

    :param true_pos: true position array
    :param measurements: noisy position measurements
    :param r_var: measurement noise variance
    :param q_var: process noise variance (white noise acceleration)
    :return: (estimates, std_devs) arrays of shape (n_steps,)
    """
    n = len(true_pos)
    kf = KalmanFilter(dim_x=2, dim_z=1)
    # State: [position, velocity].
    kf.x = np.array([[measurements[0]], [0.0]])
    # State transition: constant velocity.
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    # Measurement: observe position only.
    kf.H = np.array([[1.0, 0.0]])
    # Initial covariance.
    kf.P = np.eye(2) * 1000.0
    # Measurement noise.
    kf.R = np.array([[r_var]])
    # Process noise.
    kf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=q_var)
    estimates = np.zeros(n)
    stds = np.zeros(n)
    for i, z in enumerate(measurements):
        kf.predict()
        kf.update(np.array([[z]]))
        estimates[i] = kf.x[0, 0]
        stds[i] = np.sqrt(kf.P[0, 0])
    return estimates, stds


def plot_linear_kf_tracking(r_val: float = 5.0, q_val: float = 0.1) -> None:
    """
    Plot 1D Kalman filter tracking with interactive noise sliders.

    Shows true position, noisy measurements, KF estimate, and 1-sigma band.

    :param r_val: measurement noise variance R
    :param q_val: process noise variance Q
    """
    true_pos, t = _make_1d_signal()
    measurements = _add_measurement_noise(true_pos, r_std=np.sqrt(r_val))
    estimates, stds = _run_linear_kf_1d(true_pos, measurements, r_val, q_val)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, true_pos, "b-", lw=2, label="True position")
    ax.scatter(t, measurements, c="orange", s=20, alpha=0.7, label="Measurements")
    ax.plot(t, estimates, "g-", lw=2, label=f"KF estimate (RMSE={_rmse(estimates, true_pos):.2f})")
    ax.fill_between(
        t,
        estimates - stds,
        estimates + stds,
        alpha=0.2,
        color="green",
        label="1-sigma band",
    )
    ax.set_xlim(0, len(t) - 1)
    ax.set_ylim(-5, 35)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Position")
    ax.set_title(f"Linear Kalman Filter 1D Tracking  (R={r_val:.1f}, Q={q_val:.3f})")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.show()


def show_linear_kf_tracking_interactive() -> None:
    """
    Display interactive widget for 1D linear Kalman filter tracking.
    """
    r_slider = widgets.FloatSlider(
        value=5.0,
        min=0.1,
        max=50.0,
        step=0.5,
        description="R (meas. noise):",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="450px"),
    )
    q_slider = widgets.FloatSlider(
        value=0.1,
        min=0.001,
        max=1.0,
        step=0.01,
        description="Q (proc. noise):",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="450px"),
    )
    out = widgets.interactive_output(
        plot_linear_kf_tracking,
        {"r_val": r_slider, "q_val": q_slider},
    )
    display(widgets.VBox([r_slider, q_slider, out]))


# #############################################################################
# Cell 3: Linear KF - Uncertainty Evolution
# #############################################################################


def _run_linear_kf_uncertainty(
    n_steps: int,
    r_var: float,
    p0_var: float,
    q_var: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run linear KF and collect P diagonal elements and Kalman gain over time.

    :param n_steps: number of time steps
    :param r_var: measurement noise variance
    :param p0_var: initial state covariance diagonal value
    :param q_var: process noise variance
    :return: (pos_variance, vel_variance, kalman_gain) arrays of shape (n_steps,)
    """
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.0], [0.0]])
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.P = np.eye(2) * p0_var
    kf.R = np.array([[r_var]])
    kf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=q_var)
    pos_var = np.zeros(n_steps)
    vel_var = np.zeros(n_steps)
    k_gain = np.zeros(n_steps)
    rng = np.random.default_rng(42)
    for i in range(n_steps):
        kf.predict()
        z = rng.normal(0, np.sqrt(r_var))
        kf.update(np.array([[z]]))
        pos_var[i] = kf.P[0, 0]
        vel_var[i] = kf.P[1, 1]
        k_gain[i] = kf.K[0, 0]
    return pos_var, vel_var, k_gain


def plot_uncertainty_evolution(
    p0_val: float = 1000.0, r_val: float = 5.0
) -> None:
    """
    Plot covariance P and Kalman gain K convergence over time.

    :param p0_val: initial covariance diagonal value
    :param r_val: measurement noise variance
    """
    n = 50
    pos_var, vel_var, k_gain = _run_linear_kf_uncertainty(n, r_val, p0_val)
    t = np.arange(n)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # Panel 1: variance over time.
    ax1.plot(t, pos_var, "b-", lw=2, label="Pos variance P[0,0]")
    ax1.plot(t, vel_var, "r-", lw=2, label="Vel variance P[1,1]")
    ax1.set_xlim(0, n - 1)
    ax1.set_ylim(0, 200)
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Variance")
    ax1.set_title("State Covariance P Convergence")
    ax1.legend(fontsize=9)
    # Panel 2: Kalman gain over time.
    ax2.plot(t, k_gain, "g-", lw=2, label="K[0] (position gain)")
    ax2.set_xlim(0, n - 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Kalman Gain K")
    ax2.set_title("Kalman Gain Convergence")
    ax2.legend(fontsize=9)
    plt.suptitle(
        f"Uncertainty Evolution  (P0={p0_val:.0f}, R={r_val:.1f})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def show_uncertainty_evolution_interactive() -> None:
    """
    Display interactive widget for covariance and Kalman gain convergence.
    """
    p0_slider = widgets.FloatLogSlider(
        value=1000.0,
        base=10,
        min=0,
        max=4,
        step=0.25,
        description="P0 (init. cov.):",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="450px"),
    )
    r_slider = widgets.FloatSlider(
        value=5.0,
        min=0.1,
        max=50.0,
        step=0.5,
        description="R (meas. noise):",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="450px"),
    )
    out = widgets.interactive_output(
        plot_uncertainty_evolution,
        {"p0_val": p0_slider, "r_val": r_slider},
    )
    display(widgets.VBox([p0_slider, r_slider, out]))


# #############################################################################
# Cell 4: Extended Kalman Filter - Radar Tracking
# #############################################################################


def _polar_to_cartesian(r: float, bearing: float) -> tuple[float, float]:
    """
    Convert polar coordinates to Cartesian.

    :param r: range
    :param bearing: bearing in radians
    :return: (x, y) Cartesian coordinates
    """
    return r * np.cos(bearing), r * np.sin(bearing)


def _generate_circular_path(
    n_steps: int = 60, radius: float = 8.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a circular path in 2D Cartesian coordinates.

    :param n_steps: number of time steps
    :param radius: circle radius
    :return: (x_true, y_true) arrays of shape (n_steps,)
    """
    angles = np.linspace(0, 2 * np.pi * 0.9, n_steps)
    return radius * np.cos(angles), radius * np.sin(angles)


def _run_ekf_radar(
    x_true: np.ndarray,
    y_true: np.ndarray,
    sigma_r: float,
    sigma_b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run EKF to track 2D motion from noisy polar (range, bearing) measurements.

    State: [x, y, vx, vy]; Measurement: [range, bearing].

    :param x_true: true x positions
    :param y_true: true y positions
    :param sigma_r: std dev of range noise
    :param sigma_b: std dev of bearing noise in radians
    :return: (x_est, y_est) arrays of shape (n_steps,)
    """
    n = len(x_true)
    dt = 0.1
    ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
    # Initial state: start near truth.
    ekf.x = np.array([x_true[0], y_true[0], 0.0, 0.0])
    # State transition: constant velocity.
    ekf.F = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    # Process noise.
    q_block = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)
    ekf.Q = np.block(
        [[q_block, np.zeros((2, 2))], [np.zeros((2, 2)), q_block]]
    )
    # Measurement noise.
    ekf.R = np.diag([sigma_r**2, sigma_b**2])
    ekf.P = np.eye(4) * 10.0

    def hx(state: np.ndarray) -> np.ndarray:
        # Nonlinear measurement: range and bearing.
        px, py = state[0], state[1]
        rng = np.sqrt(px**2 + py**2)
        bearing = np.arctan2(py, px)
        return np.array([rng, bearing])

    def h_jacobian(state: np.ndarray) -> np.ndarray:
        # Jacobian of hx with respect to state.
        px, py = state[0], state[1]
        r2 = px**2 + py**2
        r = np.sqrt(r2)
        return np.array(
            [
                [px / r, py / r, 0, 0],
                [-py / r2, px / r2, 0, 0],
            ]
        )

    rng = np.random.default_rng(42)
    x_est = np.zeros(n)
    y_est = np.zeros(n)
    for i in range(n):
        # Generate noisy polar measurement.
        true_r = np.sqrt(x_true[i] ** 2 + y_true[i] ** 2)
        true_b = np.arctan2(y_true[i], x_true[i])
        z = np.array(
            [true_r + rng.normal(0, sigma_r), true_b + rng.normal(0, sigma_b)]
        )
        ekf.predict()
        ekf.update(z, h_jacobian, hx)
        x_est[i] = ekf.x[0]
        y_est[i] = ekf.x[1]
    return x_est, y_est


def plot_ekf_radar_tracking(
    sigma_r: float = 0.5, sigma_b_deg: float = 2.0
) -> None:
    """
    Plot EKF 2D radar tracking result.

    :param sigma_r: range measurement noise std dev
    :param sigma_b_deg: bearing measurement noise std dev in degrees
    """
    sigma_b = np.deg2rad(sigma_b_deg)
    x_true, y_true = _generate_circular_path()
    # Convert noisy polar measurements back to Cartesian for display.
    rng = np.random.default_rng(42)
    true_r = np.sqrt(x_true**2 + y_true**2)
    true_b = np.arctan2(y_true, x_true)
    z_r = true_r + rng.normal(0, sigma_r, size=len(true_r))
    z_b = true_b + rng.normal(0, sigma_b, size=len(true_b))
    meas_x = z_r * np.cos(z_b)
    meas_y = z_r * np.sin(z_b)
    x_est, y_est = _run_ekf_radar(x_true, y_true, sigma_r, sigma_b)
    pos_rmse = _rmse(
        np.stack([x_est, y_est], axis=1),
        np.stack([x_true, y_true], axis=1),
    )
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_true, y_true, "b-", lw=2, label="True path")
    ax.scatter(meas_x, meas_y, c="orange", s=25, alpha=0.7, label="Measurements (polar->Cartesian)")
    ax.plot(x_est, y_est, "g--", lw=2, label=f"EKF estimate (RMSE={pos_rmse:.2f})")
    ax.scatter([x_true[0]], [y_true[0]], c="black", s=80, zorder=5, label="Start")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect("equal")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title(
        f"EKF Radar Tracking  (sigma_r={sigma_r:.1f}, sigma_b={sigma_b_deg:.1f} deg)"
    )
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.show()


def show_ekf_radar_interactive() -> None:
    """
    Display interactive widget for EKF radar tracking.
    """
    r_slider = widgets.FloatSlider(
        value=0.5,
        min=0.1,
        max=5.0,
        step=0.1,
        description="sigma_r (range):",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="450px"),
    )
    b_slider = widgets.FloatSlider(
        value=2.0,
        min=0.5,
        max=10.0,
        step=0.5,
        description="sigma_b (bearing, deg):",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="460px"),
    )
    out = widgets.interactive_output(
        plot_ekf_radar_tracking,
        {"sigma_r": r_slider, "sigma_b_deg": b_slider},
    )
    display(widgets.VBox([r_slider, b_slider, out]))


# #############################################################################
# Cell 5: EKF - Jacobian Linearization Visualization
# #############################################################################


def _plot_linearization(x0: float, sigma: float) -> None:
    """
    Plot nonlinear function, Jacobian tangent, and propagated distributions.

    :param x0: operating point for linearization
    :param sigma: input Gaussian standard deviation
    """
    # Nonlinear function: y = atan(x).
    def h_func(x: np.ndarray) -> np.ndarray:
        return np.arctan(x)

    def h_jacobian(x: float) -> float:
        return 1.0 / (1.0 + x**2)

    x_range = np.linspace(-4, 4, 400)
    y_nonlinear = h_func(x_range)
    # Tangent line at x0.
    slope = h_jacobian(x0)
    y_tangent = h_func(np.array([x0]))[0] + slope * (x_range - x0)
    # Monte Carlo propagation for true distribution.
    n_mc = 5000
    rng = np.random.default_rng(42)
    x_samples = rng.normal(x0, sigma, n_mc)
    y_samples = h_func(x_samples)
    # EKF approximation: linearized mean and variance.
    ekf_mean = h_func(np.array([x0]))[0]
    ekf_std = np.abs(slope) * sigma
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    # Panel 1: nonlinear function and tangent.
    ax1.plot(x_range, y_nonlinear, "b-", lw=2, label="h(x) = atan(x)")
    ax1.plot(x_range, y_tangent, "r--", lw=2, label=f"Tangent at x0={x0:.2f}")
    ax1.axvline(x0, color="gray", linestyle=":", lw=1)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-np.pi / 2 - 0.2, np.pi / 2 + 0.2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("h(x)")
    ax1.set_title("Nonlinear Function & Tangent")
    ax1.legend(fontsize=8)
    # Panel 2: input distribution.
    x_pdf = np.linspace(x0 - 4 * sigma, x0 + 4 * sigma, 300)
    y_pdf = np.exp(-0.5 * ((x_pdf - x0) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax2.fill_between(x_pdf, y_pdf, alpha=0.4, color="blue", label=f"Input N({x0:.2f}, {sigma:.2f}^2)")
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(0, None)
    ax2.set_xlabel("x")
    ax2.set_ylabel("Density")
    ax2.set_title("Input Gaussian")
    ax2.legend(fontsize=8)
    # Panel 3: true vs EKF propagated output distribution.
    y_min = min(y_samples.min(), ekf_mean - 4 * ekf_std)
    y_max = max(y_samples.max(), ekf_mean + 4 * ekf_std)
    bins = np.linspace(-np.pi / 2, np.pi / 2, 50)
    ax3.hist(y_samples, bins=bins, density=True, alpha=0.5, color="blue", label="True (MC)")
    y_ekf_range = np.linspace(-np.pi / 2, np.pi / 2, 300)
    y_ekf_pdf = (
        np.exp(-0.5 * ((y_ekf_range - ekf_mean) / ekf_std) ** 2)
        / (ekf_std * np.sqrt(2 * np.pi))
    )
    ax3.plot(y_ekf_range, y_ekf_pdf, "r--", lw=2, label=f"EKF approx N({ekf_mean:.2f}, {ekf_std:.2f}^2)")
    ax3.set_xlim(-np.pi / 2, np.pi / 2)
    ax3.set_xlabel("h(x)")
    ax3.set_ylabel("Density")
    ax3.set_title("Output: True vs EKF Linearized")
    ax3.legend(fontsize=8)
    plt.suptitle(
        "Jacobian Linearization Visualization",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def show_linearization_interactive() -> None:
    """
    Display interactive widget for Jacobian linearization visualization.
    """
    x0_slider = widgets.FloatSlider(
        value=0.5,
        min=-3.0,
        max=3.0,
        step=0.1,
        description="x0 (op. point):",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="450px"),
    )
    sigma_slider = widgets.FloatSlider(
        value=0.5,
        min=0.1,
        max=2.0,
        step=0.1,
        description="sigma (input std):",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="450px"),
    )
    out = widgets.interactive_output(
        _plot_linearization,
        {"x0": x0_slider, "sigma": sigma_slider},
    )
    display(widgets.VBox([x0_slider, sigma_slider, out]))


# #############################################################################
# Cell 6: UKF - Sigma Points Intuition
# #############################################################################


def _nonlinear_transform_2d(
    pts: np.ndarray,
) -> np.ndarray:
    """
    Apply a nonlinear 2D transform: (x,y) -> (x + 0.5*sin(y), y + 0.5*sin(x)).

    :param pts: array of shape (N, 2)
    :return: transformed array of shape (N, 2)
    """
    out = np.zeros_like(pts)
    out[:, 0] = pts[:, 0] + 0.5 * np.sin(pts[:, 1])
    out[:, 1] = pts[:, 1] + 0.5 * np.sin(pts[:, 0])
    return out


def _plot_sigma_points(alpha: float, sigma_x: float) -> None:
    """
    Plot sigma points before and after nonlinear transformation.

    :param alpha: sigma point spread parameter
    :param sigma_x: input distribution standard deviation
    """
    n_state = 2
    # Create sigma point generator.
    sp = MerweScaledSigmaPoints(n=n_state, alpha=alpha, beta=2.0, kappa=0.0)
    mean_in = np.array([0.0, 0.0])
    cov_in = np.diag([sigma_x**2, sigma_x**2])
    sigmas = sp.sigma_points(mean_in, cov_in)
    # Monte Carlo samples for true propagation.
    n_mc = 2000
    rng = np.random.default_rng(42)
    mc_samples = rng.multivariate_normal(mean_in, cov_in, n_mc)
    mc_out = _nonlinear_transform_2d(mc_samples)
    sigmas_out = _nonlinear_transform_2d(sigmas)
    # Compute UKF output mean/cov from sigma points.
    wm = sp.Wm
    wc = sp.Wc
    ut_mean = np.dot(wm, sigmas_out)
    diff = sigmas_out - ut_mean
    ut_cov = sum(
        wc[i] * np.outer(diff[i], diff[i]) for i in range(len(wm))
    )
    mc_mean = mc_out.mean(axis=0)
    mc_cov = np.cov(mc_out.T)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    # Panel 1: input sigma points.
    mc_in_subset = mc_samples[:500]
    ax1.scatter(mc_in_subset[:, 0], mc_in_subset[:, 1], c="lightblue", s=5, alpha=0.5, label="Input samples")
    ax1.scatter(sigmas[:, 0], sigmas[:, 1], c="red", s=80, marker="x", zorder=5, linewidths=2, label="Sigma points")
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect("equal")
    ax1.set_title("Input: Sigma Points")
    ax1.legend(fontsize=8)
    # Panel 2: transformed sigma points vs MC.
    ax2.scatter(mc_out[:, 0], mc_out[:, 1], c="lightblue", s=5, alpha=0.3, label="MC output")
    ax2.scatter(sigmas_out[:, 0], sigmas_out[:, 1], c="red", s=80, marker="x", zorder=5, linewidths=2, label="Transformed sigma pts")
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_aspect("equal")
    ax2.set_title("Output: After Nonlinear Transform")
    ax2.legend(fontsize=8)
    # Panel 3: comments box.
    ax3.axis("off")
    comment = (
        f"Unscented Transform Results\n"
        f"(alpha={alpha:.3f}, sigma={sigma_x:.2f})\n\n"
        f"Input mean:  [{mean_in[0]:.2f}, {mean_in[1]:.2f}]\n\n"
        f"True MC output mean:\n  [{mc_mean[0]:.3f}, {mc_mean[1]:.3f}]\n\n"
        f"UKF output mean:\n  [{ut_mean[0]:.3f}, {ut_mean[1]:.3f}]\n\n"
        f"True MC output cov[0,0]: {mc_cov[0,0]:.3f}\n"
        f"UKF output cov[0,0]:     {ut_cov[0,0]:.3f}\n\n"
        f"Sigma points: {len(sigmas)} = 2n+1\n"
        f"MC samples: {n_mc}"
    )
    ax3.text(
        0.05,
        0.95,
        comment,
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    ax3.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    plt.suptitle("UKF Sigma Point Transform", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def show_sigma_points_interactive() -> None:
    """
    Display interactive widget for UKF sigma points visualization.
    """
    alpha_slider = widgets.FloatLogSlider(
        value=0.1,
        base=10,
        min=-3,
        max=0,
        step=0.25,
        description="alpha (spread):",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="450px"),
    )
    sigma_slider = widgets.FloatSlider(
        value=1.0,
        min=0.1,
        max=2.0,
        step=0.1,
        description="sigma_x (input std):",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="460px"),
    )
    out = widgets.interactive_output(
        _plot_sigma_points,
        {"alpha": alpha_slider, "sigma_x": sigma_slider},
    )
    display(widgets.VBox([alpha_slider, sigma_slider, out]))


# #############################################################################
# Cell 7: UKF vs EKF Comparison
# #############################################################################


def _generate_curved_path(
    n_steps: int = 60, curvature: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a curved 2D path controlled by a curvature parameter.

    :param n_steps: number of time steps
    :param curvature: controls how sharply the path curves (higher = more)
    :return: (x_true, y_true) arrays of shape (n_steps,)
    """
    t = np.linspace(0, 2 * np.pi * curvature, n_steps)
    radius = 8.0
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    return x, y


def _run_ukf_2d(
    x_true: np.ndarray,
    y_true: np.ndarray,
    r_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run UKF to track 2D position from noisy Cartesian measurements.

    :param x_true: true x positions
    :param y_true: true y positions
    :param r_var: measurement noise variance
    :return: (x_est, y_est) arrays of shape (n_steps,)
    """
    n = len(x_true)
    dt = 0.1
    points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2.0, kappa=0.0)

    def fx(state: np.ndarray, dt_: float) -> np.ndarray:
        # Constant velocity transition.
        F = np.array(
            [[1, 0, dt_, 0], [0, 1, 0, dt_], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        return F @ state

    def hx(state: np.ndarray) -> np.ndarray:
        # Measure position directly.
        return np.array([state[0], state[1]])

    ukf = UnscentedKalmanFilter(
        dim_x=4,
        dim_z=2,
        dt=dt,
        fx=fx,
        hx=hx,
        points=points,
    )
    ukf.x = np.array([x_true[0], y_true[0], 0.0, 0.0])
    ukf.P = np.eye(4) * 10.0
    q_block = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)
    ukf.Q = np.block([[q_block, np.zeros((2, 2))], [np.zeros((2, 2)), q_block]])
    ukf.R = np.eye(2) * r_var
    rng = np.random.default_rng(42)
    x_est = np.zeros(n)
    y_est = np.zeros(n)
    for i in range(n):
        noise = rng.normal(0, np.sqrt(r_var), 2)
        z = np.array([x_true[i] + noise[0], y_true[i] + noise[1]])
        ukf.predict()
        ukf.update(z)
        x_est[i] = ukf.x[0]
        y_est[i] = ukf.x[1]
    return x_est, y_est


def _run_ekf_2d(
    x_true: np.ndarray,
    y_true: np.ndarray,
    r_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run EKF to track 2D position from noisy Cartesian measurements.

    :param x_true: true x positions
    :param y_true: true y positions
    :param r_var: measurement noise variance
    :return: (x_est, y_est) arrays of shape (n_steps,)
    """
    n = len(x_true)
    dt = 0.1
    ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
    ekf.x = np.array([x_true[0], y_true[0], 0.0, 0.0])
    F = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=float,
    )
    ekf.F = F
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    ekf.H = H
    ekf.P = np.eye(4) * 10.0
    q_block = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)
    ekf.Q = np.block([[q_block, np.zeros((2, 2))], [np.zeros((2, 2)), q_block]])
    ekf.R = np.eye(2) * r_var

    def hx(state: np.ndarray) -> np.ndarray:
        return np.array([state[0], state[1]])

    def h_jac(state: np.ndarray) -> np.ndarray:
        return H

    rng = np.random.default_rng(42)
    x_est = np.zeros(n)
    y_est = np.zeros(n)
    for i in range(n):
        noise = rng.normal(0, np.sqrt(r_var), 2)
        z = np.array([x_true[i] + noise[0], y_true[i] + noise[1]])
        ekf.predict()
        ekf.update(z, h_jac, hx)
        x_est[i] = ekf.x[0]
        y_est[i] = ekf.x[1]
    return x_est, y_est


def plot_ekf_vs_ukf(curvature: float = 0.5, r_val: float = 1.0) -> None:
    """
    Plot side-by-side EKF vs UKF tracking comparison.

    :param curvature: controls path nonlinearity (higher = more curved)
    :param r_val: measurement noise variance
    """
    x_true, y_true = _generate_curved_path(curvature=curvature)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, np.sqrt(r_val), (len(x_true), 2))
    meas_x = x_true + noise[:, 0]
    meas_y = y_true + noise[:, 1]
    x_ekf, y_ekf = _run_ekf_2d(x_true, y_true, r_val)
    x_ukf, y_ukf = _run_ukf_2d(x_true, y_true, r_val)
    rmse_ekf = _rmse(
        np.stack([x_ekf, y_ekf], axis=1),
        np.stack([x_true, y_true], axis=1),
    )
    rmse_ukf = _rmse(
        np.stack([x_ukf, y_ukf], axis=1),
        np.stack([x_true, y_true], axis=1),
    )
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect("equal")
    ax1.plot(x_true, y_true, "b-", lw=2, label="True path")
    ax1.scatter(meas_x, meas_y, c="orange", s=15, alpha=0.7, label="Measurements")
    ax1.set_title("True Path + Measurements")
    ax1.legend(fontsize=8)
    ax2.plot(x_true, y_true, "b-", lw=1, alpha=0.4)
    ax2.plot(x_ekf, y_ekf, "g-", lw=2, label=f"EKF  RMSE={rmse_ekf:.3f}")
    ax2.set_title("EKF Estimate")
    ax2.legend(fontsize=9)
    ax3.plot(x_true, y_true, "b-", lw=1, alpha=0.4)
    ax3.plot(x_ukf, y_ukf, "r-", lw=2, label=f"UKF  RMSE={rmse_ukf:.3f}")
    ax3.set_title("UKF Estimate")
    ax3.legend(fontsize=9)
    plt.suptitle(
        f"EKF vs UKF  (curvature={curvature:.1f}, R={r_val:.1f})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def show_ekf_vs_ukf_interactive() -> None:
    """
    Display interactive widget for EKF vs UKF comparison.
    """
    curv_slider = widgets.FloatSlider(
        value=0.5,
        min=0.1,
        max=2.0,
        step=0.1,
        description="Curvature:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="420px"),
    )
    r_slider = widgets.FloatSlider(
        value=1.0,
        min=0.1,
        max=20.0,
        step=0.5,
        description="R (meas. noise):",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="450px"),
    )
    out = widgets.interactive_output(
        plot_ekf_vs_ukf,
        {"curvature": curv_slider, "r_val": r_slider},
    )
    display(widgets.VBox([curv_slider, r_slider, out]))


# #############################################################################
# Cell 8: Ensemble Kalman Filter
# #############################################################################


def _run_enkf_1d(
    true_pos: np.ndarray,
    measurements: np.ndarray,
    n_ensemble: int,
    q_var: float,
    r_var: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run 1D Ensemble Kalman Filter.

    :param true_pos: true position array
    :param measurements: noisy measurements
    :param n_ensemble: number of ensemble members
    :param q_var: process noise variance
    :param r_var: measurement noise variance
    :return: (ensemble_means, ensemble_stds, all_particles) where
             all_particles has shape (n_steps, n_ensemble)
    """
    n = len(true_pos)
    dt = 1.0

    def fx(state: np.ndarray, dt_: float) -> np.ndarray:
        # Constant velocity: position integrates velocity.
        return np.array([state[0] + state[1] * dt_, state[1]])

    def hx(state: np.ndarray) -> np.ndarray:
        return np.array([state[0]])

    enkf = EnsembleKalmanFilter(
        x=np.array([measurements[0], 0.0]),
        P=np.eye(2) * 10.0,
        dim_z=1,
        dt=dt,
        N=n_ensemble,
        hx=hx,
        fx=fx,
    )
    enkf.R = np.array([[r_var]])
    enkf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=q_var)
    means = np.zeros(n)
    stds = np.zeros(n)
    particles = np.zeros((n, n_ensemble))
    for i, z in enumerate(measurements):
        enkf.predict()
        enkf.update(np.array([z]))
        # Collect ensemble state (position component).
        pos_ensemble = enkf.sigmas[:, 0]
        particles[i] = pos_ensemble
        means[i] = pos_ensemble.mean()
        stds[i] = pos_ensemble.std()
    return means, stds, particles


def plot_enkf_visualization(
    n_ensemble: int = 100, q_val: float = 0.1
) -> None:
    """
    Plot EnKF ensemble particles and ensemble mean vs true state.

    :param n_ensemble: number of ensemble members
    :param q_val: process noise variance
    """
    true_pos, t = _make_1d_signal()
    r_var = 5.0
    measurements = _add_measurement_noise(true_pos, r_std=np.sqrt(r_var))
    means, stds, particles = _run_enkf_1d(
        true_pos, measurements, n_ensemble, q_val, r_var
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Panel 1: particle scatter colored by time.
    n_steps = len(t)
    for i in range(0, n_steps, max(1, n_steps // 20)):
        ax1.scatter(
            [t[i]] * n_ensemble,
            particles[i],
            c=[[i / n_steps, 0.2, 1 - i / n_steps]],
            s=3,
            alpha=0.3,
        )
    ax1.plot(t, means, "g-", lw=2, label="Ensemble mean")
    ax1.plot(t, true_pos, "b-", lw=1.5, alpha=0.7, label="True position")
    ax1.set_xlim(0, n_steps - 1)
    ax1.set_ylim(-5, 35)
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Position")
    ax1.set_title(f"Ensemble Particles (N={n_ensemble})")
    ax1.legend(fontsize=9)
    # Panel 2: mean with std band.
    ax2.plot(t, true_pos, "b-", lw=2, label="True position")
    ax2.scatter(t, measurements, c="orange", s=15, alpha=0.7, label="Measurements")
    ax2.plot(t, means, "g-", lw=2, label=f"EnKF mean (RMSE={_rmse(means, true_pos):.2f})")
    ax2.fill_between(
        t,
        means - stds,
        means + stds,
        alpha=0.25,
        color="green",
        label="Ensemble std",
    )
    ax2.set_xlim(0, n_steps - 1)
    ax2.set_ylim(-5, 35)
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Position")
    ax2.set_title("EnKF: Mean and Spread")
    ax2.legend(fontsize=9)
    plt.suptitle(
        f"Ensemble Kalman Filter  (N={n_ensemble}, Q={q_val:.3f})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def show_enkf_interactive() -> None:
    """
    Display interactive widget for EnKF ensemble visualization.
    """
    n_slider = widgets.IntSlider(
        value=100,
        min=10,
        max=500,
        step=10,
        description="N (ensemble size):",
        style={"description_width": "160px"},
        layout=widgets.Layout(width="460px"),
    )
    q_slider = widgets.FloatSlider(
        value=0.1,
        min=0.001,
        max=1.0,
        step=0.01,
        description="Q (proc. noise):",
        style={"description_width": "150px"},
        layout=widgets.Layout(width="450px"),
    )
    out = widgets.interactive_output(
        plot_enkf_visualization,
        {"n_ensemble": n_slider, "q_val": q_slider},
    )
    display(widgets.VBox([n_slider, q_slider, out]))


# #############################################################################
# Cell 9: All Four Filters Comparison
# #############################################################################


def plot_all_filters_comparison() -> None:
    """
    Compare all four filter types on the same 1D tracking problem.

    Shows KF, EKF, UKF, and EnKF side by side with RMSE values and a summary
    comments panel.
    """
    true_pos, t = _make_1d_signal()
    r_var = 5.0
    q_var = 0.1
    measurements = _add_measurement_noise(true_pos, r_std=np.sqrt(r_var))
    # Run all filters.
    kf_est, kf_std = _run_linear_kf_1d(true_pos, measurements, r_var, q_var)
    # EKF (linear problem so identical to KF - use polar wrapper).
    # For fair comparison use same linear EKF.
    ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
    ekf.x = np.array([measurements[0], 0.0])
    ekf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    ekf.H = H
    ekf.P = np.eye(2) * 1000.0
    ekf.R = np.array([[r_var]])
    ekf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=q_var)

    def hx_lin(state: np.ndarray) -> np.ndarray:
        return np.array([state[0]])

    def h_jac_lin(state: np.ndarray) -> np.ndarray:
        return H

    ekf_est = np.zeros(len(true_pos))
    ekf_std_arr = np.zeros(len(true_pos))
    for i, z in enumerate(measurements):
        ekf.predict()
        ekf.update(np.array([z]), h_jac_lin, hx_lin)
        ekf_est[i] = ekf.x[0]
        ekf_std_arr[i] = np.sqrt(ekf.P[0, 0])
    # UKF.
    points = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2.0, kappa=0.0)

    def fx_ukf(state: np.ndarray, dt_: float) -> np.ndarray:
        return np.array([state[0] + state[1] * dt_, state[1]])

    def hx_ukf(state: np.ndarray) -> np.ndarray:
        return np.array([state[0]])

    ukf = UnscentedKalmanFilter(
        dim_x=2, dim_z=1, dt=1.0, fx=fx_ukf, hx=hx_ukf, points=points
    )
    ukf.x = np.array([measurements[0], 0.0])
    ukf.P = np.eye(2) * 1000.0
    ukf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=q_var)
    ukf.R = np.array([[r_var]])
    ukf_est = np.zeros(len(true_pos))
    ukf_std_arr = np.zeros(len(true_pos))
    for i, z in enumerate(measurements):
        ukf.predict()
        ukf.update(np.array([z]))
        ukf_est[i] = ukf.x[0]
        ukf_std_arr[i] = np.sqrt(ukf.P[0, 0])
    # EnKF.
    enkf_means, enkf_stds, _ = _run_enkf_1d(
        true_pos, measurements, 200, q_var, r_var
    )
    rmse_kf = _rmse(kf_est, true_pos)
    rmse_ekf = _rmse(ekf_est, true_pos)
    rmse_ukf = _rmse(ukf_est, true_pos)
    rmse_enkf = _rmse(enkf_means, true_pos)
    n_steps = len(t)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    filter_data = [
        ("KF", kf_est, kf_std, rmse_kf, "green"),
        ("EKF", ekf_est, ekf_std_arr, rmse_ekf, "purple"),
        ("UKF", ukf_est, ukf_std_arr, rmse_ukf, "red"),
        ("EnKF", enkf_means, enkf_stds, rmse_enkf, "darkorange"),
    ]
    for ax, (name, est, std, rmse, color) in zip(axes, filter_data):
        ax.plot(t, true_pos, "b-", lw=2, alpha=0.8, label="True")
        ax.scatter(t, measurements, c="orange", s=10, alpha=0.5)
        ax.plot(t, est, "-", color=color, lw=2, label=f"{name}  RMSE={rmse:.2f}")
        ax.fill_between(
            t,
            est - std,
            est + std,
            alpha=0.2,
            color=color,
        )
        ax.set_xlim(0, n_steps - 1)
        ax.set_ylim(-5, 35)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Position" if ax == axes[0] else "")
        ax.set_title(name)
        ax.legend(fontsize=8, loc="upper left")
    plt.suptitle(
        "All Four Filters Compared on 1D Tracking (R=5, Q=0.1)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
    # Print summary table.
    print("Filter Comparison Summary:")
    print(f"{'Filter':<10} {'RMSE':>8}")
    print("-" * 20)
    for name, _, _, rmse, _ in filter_data:
        print(f"{name:<10} {rmse:>8.4f}")
    print(
        "\nNote: For linear-Gaussian problems KF is optimal."
        " EKF/UKF/EnKF converge to KF on linear problems."
    )
