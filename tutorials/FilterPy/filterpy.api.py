# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FilterPy API Tutorial
#
# This notebook provides a hands-on tutorial for the FilterPy library, covering
# four major Kalman filter variants used in Bayesian state estimation.
#
# Topics covered:
# - Linear Kalman Filter (KF): optimal estimator for linear-Gaussian systems
# - Extended Kalman Filter (EKF): linearizes nonlinear systems via Jacobians
# - Unscented Kalman Filter (UKF): uses sigma points to propagate distributions
# - Ensemble Kalman Filter (EnKF): Monte Carlo approach with particle ensembles
#
# References:
# - https://filterpy.readthedocs.io/en/latest/
# - Roger Labbe, "Kalman and Bayesian Filters in Python"
# - filterpy.api.md

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# %% [markdown]
# ## Imports

# %%
import os
import subprocess
import sys

# Find the git root and add necessary paths to sys.path.
_git_root = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True
).strip()
_helpers_root = os.path.join(_git_root, "helpers_root")
for _path in [_git_root, _helpers_root]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

# %%
import logging

import helpers.hdbg as hdbg
import helpers.hprint as hprint

# %% [markdown]
# ## Configuration

# %%
hdbg.init_logger(verbosity=logging.INFO)

_LOG = logging.getLogger(__name__)

hprint.config_notebook()

# %%
import tutorials.FilterPy.filterpy_api_utils as utils

# %% [markdown]
# ---
# ## Cell 1: Introduction - The Predict-Update Cycle
#
# - Purpose: Introduce Bayesian filtering and the core predict-update cycle
#   that all Kalman variants share.
# - All Kalman variants follow the same two-step loop:
#   - **Predict**: project the state forward using the dynamics model F, adding
#     process noise Q.
#   - **Update**: incorporate the new measurement z using the Kalman gain K,
#     which trades off model prediction vs measurement uncertainty.
# - Key matrices:
#   - F: state transition; H: measurement function
#   - Q: process noise; R: measurement noise; P: state covariance

# %%
# Show predict-update cycle diagram.
utils.plot_predict_update_diagram()

# %%
# Show matrix reference table.
utils.show_matrix_table()

# %% [markdown]
# ---
# ## Cell 2: Linear Kalman Filter - 1D Position and Velocity Tracking
#
# - Purpose: Demonstrate the `KalmanFilter` API for tracking a moving object
#   with noisy position-only measurements.
# - State: [position, velocity]; Measurement: [position].
# - State transition (constant velocity model):
#   `F = [[1, dt], [0, 1]]`; H = [[1, 0]]
# - Key insight: The R/Q ratio controls whether the filter trusts measurements
#   (low R) or the dynamic model (high Q relative to R).

# %%
# Display interactive 1D KF tracking with R and Q sliders.
utils.show_linear_kf_tracking_interactive()

# %% [markdown]
# ---
# ## Cell 3: Linear Kalman Filter - Uncertainty Evolution
#
# - Purpose: Show how the state covariance P and the Kalman gain K converge
#   over time regardless of the initial P0 value.
# - Key insight: P always converges to a steady-state value; K decreases as
#   the filter becomes more confident in its estimate.

# %%
# Display interactive covariance and Kalman gain convergence.
utils.show_uncertainty_evolution_interactive()

# %% [markdown]
# ---
# ## Cell 4: Extended Kalman Filter - Radar Tracking with Polar Measurements
#
# - Purpose: Demonstrate `ExtendedKalmanFilter` for a nonlinear measurement
#   function: range and bearing (polar) -> Cartesian state.
# - The EKF linearizes hx at each time step using its Jacobian H_jac.
# - API:
#   ```python
#   ekf.update(z, HJacobian_at, hx)
#   ```
# - Key insight: EKF works well for mildly nonlinear systems; the Jacobian
#   must be derived and coded manually.

# %%
# Display interactive EKF radar tracking with noise sliders.
utils.show_ekf_radar_interactive()

# %% [markdown]
# ---
# ## Cell 5: Extended Kalman Filter - Jacobian Linearization Visualization
#
# - Purpose: Visually show how linearization approximates the nonlinear
#   function h(x) = atan(x) and where the EKF approximation breaks down.
# - The EKF assumes the output distribution is Gaussian; the true propagated
#   distribution through a nonlinear function is generally non-Gaussian.
# - Key insight: Linearization error grows with input uncertainty and function
#   curvature; EKF can underestimate posterior variance.

# %%
# Display interactive Jacobian linearization visualization.
utils.show_linearization_interactive()

# %% [markdown]
# ---
# ## Cell 6: Unscented Kalman Filter - Sigma Points Intuition
#
# - Purpose: Introduce the Unscented Transform and sigma points as a way to
#   propagate distributions through nonlinear functions without Jacobians.
# - The UKF selects 2n+1 deterministic sigma points, propagates them through
#   the nonlinear function, then recovers the output mean and covariance.
# - API:
#   ```python
#   points = MerweScaledSigmaPoints(n=dim_x, alpha=0.1, beta=2., kappa=0)
#   ukf = UnscentedKalmanFilter(dim_x=..., dim_z=..., dt=...,
#                                fx=fx, hx=hx, points=points)
#   ```
# - Key insight: The UT captures higher-order statistics; alpha controls the
#   spread of sigma points around the mean.

# %%
# Display interactive sigma points visualization.
utils.show_sigma_points_interactive()

# %% [markdown]
# ---
# ## Cell 7: UKF vs EKF - Side-by-Side Tracking Comparison
#
# - Purpose: Compare UKF and EKF on the same nonlinear 2D tracking problem to
#   show when UKF outperforms EKF.
# - The `curvature` slider controls how strongly nonlinear the path is.
# - Key insight: UKF typically achieves lower RMSE for strongly nonlinear
#   systems; EKF may diverge under high curvature; both converge for linear
#   systems.

# %%
# Display interactive EKF vs UKF comparison.
utils.show_ekf_vs_ukf_interactive()

# %% [markdown]
# ---
# ## Cell 8: Ensemble Kalman Filter - Particle Ensemble Visualization
#
# - Purpose: Demonstrate `EnsembleKalmanFilter` API using Monte Carlo particles
#   to represent state uncertainty.
# - The ensemble spread represents the posterior distribution without assuming
#   Gaussianity.
# - API:
#   ```python
#   enkf = EnsembleKalmanFilter(x=x0, P=P, dim_z=1, dt=dt,
#                                N=n_ensemble, hx=hx, fx=fx)
#   enkf.predict()
#   enkf.update(z)
#   ```
# - Key insight: N=100+ particles usually gives stable estimates; computational
#   cost scales linearly with N.

# %%
# Display interactive EnKF ensemble particle visualization.
utils.show_enkf_interactive()

# %% [markdown]
# ---
# ## Cell 9: Filter Comparison - All Four Filters on One Problem
#
# - Purpose: Directly compare all four filter types on the same 1D tracking
#   problem and tabulate RMSE.
# - For a linear-Gaussian problem, KF is theoretically optimal; EKF, UKF, and
#   EnKF converge to the KF solution.
# - Guidance for choosing a filter:
#   - Linear system: use KF
#   - Mildly nonlinear: use EKF (fastest)
#   - Strongly nonlinear: use UKF (more accurate, no Jacobian needed)
#   - Very high-dimensional or non-Gaussian: use EnKF or particle filters

# %%
# Run and display all four filters side by side.
utils.plot_all_filters_comparison()
