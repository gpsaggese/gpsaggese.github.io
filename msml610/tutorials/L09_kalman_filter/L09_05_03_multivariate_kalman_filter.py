# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Multivariate Kalman filter
#
# - This notebook extends the Kalman filter to multivariate Gaussians, and
#   uses it to track a dog whose velocity is a hidden variable
# - The pedagogical arc:
#   - Multivariate Gaussians, covariance, and using correlations to improve
#     estimates
#   - Sum and product of bidimensional Gaussians
#   - Tracking a dog with hidden variables: designing the state, the system
#     model, the noise matrices, and the measurement function, then running
#     the filter
#   - The effect of adding a hidden variable, comparing 1D and 2D filters

# %%
# `filterpy` is an extra for this notebook, pinned to the version already
# in requirements.txt.
# !sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet filterpy==1.4.5)"

import filterpy

print("filterpy version: ", filterpy.__version__)

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import numpy as np

# %%
import helpers.hintrospection as hintros
import helpers.hio as hio
import helpers.hnotebook as hnotebook

import L09_05_03_multivariate_kalman_filter_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %%
dst_dir = "figures"
hio.create_dir(dst_dir, incremental=True)
# !cp msml610/tutorials/figures/*.png msml610/lectures_source/figures

# %% [markdown]
# # Part 1: Multivariate Gaussians

# %% [markdown]
# ## Cell 1.1: Correlated variables and covariance
#
# - A multivariate Gaussian is a normal distribution on multiple
#   dimensions
#   - The dimensions don't need to be necessarily spatial dimensions
#   - E.g., position, velocity, acceleration in 3 dimensions
#   - E.g., milk production and feed rate at a dairy
# - The components of a Gaussian are correlated
#   - E.g., height and weight of people are (positively) correlated
#   - E.g., outdoor temperature and home heating bills are (negatively)
#     correlated
#   - E.g., the weight of my dog and the price of coffee are uncorrelated
# - Correlation allows prediction
#   - E.g., height and weight are correlated. If you are much taller than
#     me, I can predict that you weigh more than me
#   - Noise in the measurements, uncertainty in the knowledge of the
#     system, intrinsic stochasticity make correlations (and
#     predictions) imperfect
# - Assuming linear correlation, the covariance between 2 vars is
#   defined as
#   $$cov(X, Y) = E[(X - \mu_X)(Y - \mu_Y)]$$

# %%
height = [60, 62, 63, 65, 65.1, 68, 69, 70, 72, 74]
weight = [95, 120, 127, 119, 151, 143, 173, 171, 180, 210]
utils.plot_correlated_data(
    height, weight, xlabel="Height (in)", ylabel="Weight (lbs)", equal=False
)
print("cov=\n", np.cov(height, weight))

# %% [markdown]
# ## Cell 1.2: The multivariate Gaussian PDF
#
# **Goal**
# - Evaluate the multivariate Gaussian PDF at a point, given a mean
#   vector and a covariance matrix
#
# **Implementation** `filterpy.stats.multivariate_gaussian(x, mu, P)`

# %%
from filterpy.stats import multivariate_gaussian

x = np.array([2.5, 7.3])
mu = np.array([2.0, 7.0])
P = np.array([[8.0, 0.0], [0.0, 3.0]])

print("multivariate_gaussian=", multivariate_gaussian(x, mu, P))

# %% [markdown]
# ## Cell 1.3: Visualizing a 2D covariance
#
# **Goal**
# - Visualize a 2D Gaussian's covariance both as a 3D sampled surface and
#   as a 2D covariance-ellipse-style matrix plot

# %%
utils.plot_3d_sampled_covariance(mu, P)

# %% [markdown]
# **Description**
# - Inputs
#   - `sigma_x`, `sigma_y`: standard deviations along each axis
#   - `rho`: correlation coefficient between the 2 axes
#
# - Panels
#   - `left`: the resulting covariance matrix as a heatmap
#   - `right`: the corresponding covariance ellipse

# %%
utils.cell_1_1_plot_covariance_matrix()

# %% [markdown]
# **Guided usage**
# - Set `rho=0`, with `sigma_x != sigma_y`
#   - Observe the ellipse is axis-aligned
# - Raise `rho` toward `1`
#   - Observe the ellipse tilts and elongates along the diagonal

# %% [markdown]
# **Implementation** `utils.plot_3d_sampled_covariance(mu, P)`,
# `utils.cell_1_1_plot_covariance_matrix()`

# %%
hintros.print_obj_info(utils.cell_1_1_plot_covariance_matrix)

# %% [markdown]
# ## Cell 1.4: Using correlations to improve estimates
#
# - Given an aircraft that we need to locate on a 2D map, ignoring the
#   altitude
# - We are tracking an aircraft with 2 radars
#   - Each radar provides the position as 2 coordinates, such as the
#     "range" (distance) and the "bearing" (angle) to a target
# - The areas on the X-Y diagrams represent where the plane is likely to
#   be
# - One radar measurement (let's assume it's the prior) is equally
#   distributed across the 2 axes (yellow)
# - The other radar measurement (let's assume it's the evidence) instead
#   is inaccurate along the range but precise along the bearing
#   estimates (green)
# - The posterior is given by the multiplication of the 2 Gaussians
#   (blue): its uncertainty is smaller than either of the 2 measurements

# %%
# Prior.
P0 = [[6, 0], [0, 6]]
filterpy.stats.plot_covariance_ellipse((10, 10), P0, fc="y", alpha=0.6)

# Evidence.
P1 = [[2, 1.9], [1.9, 2]]
filterpy.stats.plot_covariance_ellipse((10, 10), P1, fc="g", alpha=0.9)

# Posterior.
P2 = filterpy.stats.multivariate_multiply((10, 10), P0, (10, 10), P1)[1]
print("P2=", P2)
filterpy.stats.plot_covariance_ellipse((10, 10), P2, ec="k", fc="b")

# %% [markdown]
# - If the 2 measurements are like below, the resulting measurement is
#   much smaller
#   - We are "triangulating" the aircraft
#   - This is optimal when the radars are orthogonal

# %%
P3 = [[2, -1.9], [-1.9, 2.2]]
filterpy.stats.plot_covariance_ellipse((10, 10), P2, ec="k", fc="y", alpha=0.6)
filterpy.stats.plot_covariance_ellipse((10, 10), P3, ec="k", fc="g", alpha=0.6)

P4 = filterpy.stats.multivariate_multiply((10, 10), P2, (10, 10), P3)[1]
filterpy.stats.plot_covariance_ellipse((10, 10), P4, ec="k", fc="b")

# %% [markdown]
# # Part 2: Sum and Product of Bidimensional Gaussians

# %% [markdown]
# ## Cell 2.1: Sum of 2 2D Gaussians
#
# - If $X \sim N(0, \Sigma_1)$ and $Y \sim N(0, \Sigma_2)$ are
#   independent, then $X + Y \sim N(0, \Sigma_1 + \Sigma_2)$: the
#   covariances add
#   - Yellow: G1
#   - Green: G2
#   - Blue: G1 + G2
# - The sum is always larger (less certain) than either factor
#
# **Implementation** `utils.cell_1_2_plot_sum_of_gaussians()`

# %%
utils.cell_1_2_plot_sum_of_gaussians()

# %% [markdown]
# ## Cell 2.2: Product of 2 2D Gaussians
#
# - The product of 2 Gaussian PDFs is also a Gaussian (up to
#   normalization)
# - Given $G1 \sim N(0, \Sigma_1)$ and $G2 \sim N(0, \Sigma_2)$:
#   $\Sigma^{-1} = \Sigma_1^{-1} + \Sigma_2^{-1}$
#   - Yellow: G1
#   - Green: G2
#   - Blue: G1 * G2
# - The product is always smaller (more certain) than either factor
#
# **Implementation** `utils.cell_1_3_plot_product_of_gaussians()`

# %%
utils.cell_1_3_plot_product_of_gaussians()

# %% [markdown]
# # Part 3: Tracking a Dog with Hidden Variables
#
# - We go back to tracking a dog on a 1D track and use hidden variables
#   to improve our estimates
#   - The underlying ideas are the same as the previous chapters: we are
#     just multiplying and adding Gaussians
# - The dog moves approximately 1 meter per step
#   - At each step, the velocity varies according to the process
#     variance `process_var`
#   - After updating the position, a measurement is computed with an
#     assumed sensor variance `z_var`
#   - Time is discrete

# %% [markdown]
# ## Cell 3.1: Simulating the dog
#
# **Goal**
# - Simulate the dog's true trajectory and noisy sensor measurements,
#   used throughout the rest of Part 3
#
# **Implementation** `utils.compute_dog_data(z_var, process_var,
# count)`

# %%
z_var = 1.0
process_var = 0.1
count = 50
xs, zs = utils.compute_dog_data(z_var, process_var, count=count)
print("xs=", xs)
print("zs=", zs)

# %%
plt.figure(figsize=(8, 3))
_ = plt.plot(xs, label="True position (xs)")

# %% [markdown]
# ## Cell 3.2: Predict step: the state vector
#
# - The state vector $\mathbf{x}_t$ tracks both position and velocity:
#   $$
#   \mathbf{x}_t = \begin{bmatrix} x_t \\ \dot{x}_t \end{bmatrix}
#   $$
# - Position $x_t$ is observed by the sensor
# - Velocity $\dot{x}_t$ is a **hidden variable**: it is estimated by the
#   filter, not measured directly
# - Additional hidden variables (e.g., acceleration, jerk) can be added
#   at the cost of a larger state vector

# %%
dt = 1.0  # Time step (seconds).
# Initial state: position 0 m, velocity 1 m/s.
x0 = np.array([[0.0], [1.0]])
print("x0 (initial state) =\n", x0)

# %% [markdown]
# ## Cell 3.3: Designing the state covariance
#
# - The state covariance matrix $P$ encodes our uncertainty about the
#   state
# - We initialize it with large diagonal values to reflect ignorance at
#   startup:
#   $$
#   P = \begin{bmatrix} \sigma_x^2 & 0 \\ 0 & \sigma_{\dot{x}}^2
#   \end{bmatrix} = \begin{bmatrix} 500 & 0 \\ 0 & 49 \end{bmatrix}
#   $$
#   - $\sigma_x^2 = 500\,\mathrm{m}^2$: we do not know the dog's starting
#     position
#   - $\sigma_{\dot{x}}^2 = 49\,(\mathrm{m/s})^2$: top dog speed is
#     21 m/s, so $3\,\sigma_{\dot{x}} = 21 \;\Rightarrow\;
#     \sigma_{\dot{x}}^2 = 49$
#   - Off-diagonal terms are zero: position and velocity are initially
#     uncorrelated

# %%
# Initial state covariance: large uncertainty in position and velocity.
P = np.diag([500.0, 49.0])
print("P =\n", P)

# %% [markdown]
# ## Cell 3.4: Designing the system model
#
# - The state-transition matrix $F$ describes how the state evolves over
#   one time step under a constant-velocity assumption:
#   $$
#   x_{t+1} = x_t + \dot{x}_t\,\Delta t, \qquad \dot{x}_{t+1} = \dot{x}_t
#   $$
# - In matrix form $\mathbf{x}_{t+1} = F\,\mathbf{x}_t$:
#   $$
#   F = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}
#   $$
# - The constant-velocity assumption is approximate; the filter
#   tolerates deviations through the process noise $Q$

# %%
# State transition matrix: constant velocity model.
F = np.array([[1.0, dt], [0.0, 1.0]])
print("F =\n", F)

# %% [markdown]
# ## Cell 3.5: Predicting the system
#
# - Without a new measurement the filter propagates the state and
#   covariance forward (the **predict step**):
#   $$
#   \bar{\mathbf{x}}_t = F\,\mathbf{x}_{t-1}
#   $$
#   $$
#   \bar{P}_t = F\,P_{t-1}\,F^{\top} + Q
#   $$
# - The covariance $\bar{P}_t$ grows at every prediction step because the
#   future velocity is uncertain; adding measurements (update step)
#   shrinks it back

# %%
# One illustrative prediction step (no measurement yet, Q = 0 for clarity).
x_pred = F @ x0
P_pred = F @ P @ F.T
print("x_pred (one step) =\n", x_pred)
print("P_pred (one step) =\n", P_pred)

# %% [markdown]
# ## Cell 3.6: Designing the system noise
#
# - The dog's velocity is not perfectly constant; it is perturbed by
#   unmodeled forces (distraction, fatigue, wind)
# - We model the velocity perturbation as zero-mean Gaussian noise $w$:
#   $$
#   \dot{x}_t = \dot{x}_{t-1} + w, \quad w \sim \mathcal{N}(0,\,q)
#   $$
# - The process noise covariance $Q =
#   E\!\left[\mathbf{w}\,\mathbf{w}^T\right]$ captures this uncertainty
# - Position is not directly noisy (it inherits noise only through
#   velocity), so only the velocity variance is non-zero:
#   $$
#   Q = \begin{bmatrix} 0 & 0 \\ 0 & q \end{bmatrix}
#   $$

# %%
# Process noise covariance: only velocity is directly perturbed.
Q = np.array([[0.0, 0.0], [0.0, process_var]])
print("Q =\n", Q)

# %% [markdown]
# ## Cell 3.7: Designing the control function
#
# - A known control input $\mathbf{u}$ can shift the predicted state:
#   $$
#   \bar{\mathbf{x}}_t = F\,\mathbf{x}_{t-1} + B\,\mathbf{u}_t
#   $$
# - Examples of control inputs:
#   - Car: steering angle, throttle
#   - Dog: the owner's voice command, the sight of a squirrel
# - In this example there is no known control input, so $B = 0$

# %%
# No control input in this example.
B = np.zeros((2, 1))
u = np.zeros((1, 1))
print("B =\n", B)
print("u =\n", u)

# %% [markdown]
# ## Cell 3.8: Update step: the measurement function
#
# - The sensor measures only position, not velocity
# - The measurement $z_t$ is related to the full state $\mathbf{x}_t$ via
#   the measurement matrix $H$:
#   $$
#   z_t = H\,\mathbf{x}_t + v, \quad v \sim \mathcal{N}(0,\,R)
#   $$
# - For position-only observation:
#   $$
#   H = \begin{bmatrix} 1 & 0 \end{bmatrix}
#   $$
# - The innovation (residual) is the difference between the actual
#   measurement and the predicted measurement:
#   $$
#   \mathbf{y}_t = z_t - H\,\bar{\mathbf{x}}_t
#   $$

# %%
# Measurement matrix: H selects position from the state vector.
H = np.array([[1.0, 0.0]])
print("H =", H)

# %% [markdown]
# ## Cell 3.9: Designing the measurement noise matrix R
#
# - $R$ encodes the variance of the sensor noise:
#   $$
#   R = \begin{bmatrix} \sigma_z^2 \end{bmatrix}
#   $$
# - $R$ can be difficult to estimate in practice:
#   - Noise may not be Gaussian
#   - There can be a systematic bias in the sensor
#   - The error can be asymmetric (e.g., a temperature sensor is less
#     precise at high temperatures)

# %%
# Measurement noise covariance.
R = np.array([[z_var]])
print("R =", R)

# %% [markdown]
# ## Cell 3.10: Running the Kalman filter
#
# **Goal**
# - Run the full predict-update cycle on the simulated dog data, and
#   check that the estimate converges close to the true position
#
# **Implementation** `utils.run_dog_kalman_filter(zs, z_var,
# process_var)`
# - The filter alternates predict and update at every time step
# - After a few steps the estimate converges close to the true position
# - The uncertainty (shaded band) shrinks rapidly as measurements
#   accumulate

# %%
means, variances = utils.run_dog_kalman_filter(zs, z_var, process_var)
utils.plot_dog_tracking(xs, zs, means, variances)

# %% [markdown]
# ## Cell 3.11: Interactively exploring dog tracking
#
# **Goal**
# - Let students sweep the noise levels and see the filter adapt in real
#   time

# %% [markdown]
# **Description**
# - Inputs
#   - `seed`: random seed for reproducibility
#   - `z_var`: measurement noise variance
#   - `process_var`: process noise variance
#   - `count`: number of simulation steps
#
# - Panels
#   - `left`: true position, measurements, KF estimate, and +-1 std
#     dev band
#   - `Comments`: the current parameters, and the final estimate's
#     MSE against the true position

# %%
utils.cell_dog_tracking_interactive()

# %% [markdown]
# **Guided usage**
# - Raise `z_var` far above `process_var`
#   - Observe the KF estimate line smooths out, tracking its own
#     prediction more than the noisy measurements
# - Raise `process_var` far above `z_var`
#   - Observe the KF estimate follows the measurements closely

# %% [markdown]
# **Implementation** `utils.cell_dog_tracking_interactive()`
# - Increasing `z_var` makes the sensor noisier: the filter smooths more
#   aggressively and leans on its own prediction
# - Increasing `process_var` makes the dog more unpredictable: the
#   filter trusts measurements more and follows them closely
# - The blue band shows the +-1 sigma position uncertainty of the filter

# %%
hintros.print_obj_info(utils.cell_dog_tracking_interactive)

# %% [markdown]
# # Part 4: Effect of Hidden Variables

# %% [markdown]
# ## Cell 4.1: What changes when we add a hidden variable?
#
# - A **1D Kalman filter** tracks position only: $\mathbf{x}_t = [x_t]$
#   - The filter has no model of velocity; it assumes the dog is
#     stationary between measurements
#   - Each prediction step simply holds position constant:
#     $\bar{x}_t = x_{t-1}$
#   - The filter can only react to measurements, not anticipate movement
# - A **2D Kalman filter** tracks position and velocity:
#   $\mathbf{x}_t = [x_t, \dot{x}_t]^T$
#   - Velocity $\dot{x}_t$ is the **hidden variable**: never measured
#     directly, but inferred from successive position measurements
#   - The prediction step uses velocity to anticipate where the dog will
#     be: $\bar{x}_t = x_{t-1} + \dot{x}_{t-1}\,\Delta t$
#   - The covariance matrix develops off-diagonal terms that capture the
#     correlation between position and velocity errors
#
# **Why hidden variables help**
# - The 1D filter is essentially a recursive average: it is
#   well-calibrated only when the dog is stationary or very slow
# - The 2D filter learns the dog's speed implicitly; after a few steps
#   the estimated velocity is close to the true velocity and the
#   position predictions are much more accurate
# - The improvement is quantified by the Mean Squared Error (MSE) shown
#   below: the 2D filter consistently achieves a lower MSE than the 1D
#   filter
#
# **Experiment setup**
# - Same dog simulation: the dog moves ~1 m/step with Gaussian velocity
#   noise
# - Same measurements: position + Gaussian sensor noise
#   $\sigma^2 = z\_var$
# - 1D filter: $F = [[1]]$, $Q = [[process\_var]]$, $H = [[1]]$,
#   $R = [[z\_var]]$
# - 2D filter: $F = [[1, \Delta t], [0, 1]]$,
#   $Q = diag(0, process\_var)$, $H = [[1, 0]]$, $R = [[z\_var]]$
#
# **Implementation** `utils.plot_hidden_variable_comparison(...)`

# %%
# Static comparison with default parameters.
np.random.seed(42)
xs_ex, zs_ex = utils.compute_dog_data(z_var=1.0, process_var=0.1, count=50)
means_1d, var_1d = utils.run_dog_kalman_filter_1d(
    zs_ex, z_var=1.0, process_var=0.1
)
means_2d, var_2d = utils.run_dog_kalman_filter(zs_ex, z_var=1.0, process_var=0.1)
utils.plot_hidden_variable_comparison(
    xs_ex, zs_ex, means_1d, var_1d, means_2d, var_2d
)

# %% [markdown]
# - **Left panel (1D KF, position only)**:
#   - The filter lags behind the true trajectory because each prediction
#     step does not use velocity: it simply holds the previous position
#     estimate
#   - The uncertainty band (shaded area) is wide because the filter must
#     account for large unpredicted jumps in position
#   - MSE is typically higher
# - **Right panel (2D KF, position + hidden velocity)**:
#   - The filter rapidly learns the dog's velocity from the first few
#     measurements
#   - Subsequent predictions are accurate because the motion model
#     ($x_{t+1} = x_t + v_t$) anticipates where the dog will be
#   - The uncertainty band is narrower and centered on the true
#     trajectory
#   - MSE is consistently lower

# %% [markdown]
# ## Cell 4.2: Interactively comparing 1D vs 2D filters
#
# **Goal**
# - Let students vary the noise levels and see the MSE gap between the
#   1D and 2D filters change in real time

# %% [markdown]
# **Description**
# - Inputs
#   - `seed`: random seed for reproducibility
#   - `z_var`: measurement noise variance
#   - `process_var`: process noise variance
#   - `count`: number of simulation steps
#
# - Panels
#   - `left`: 1D filter (position only) tracking
#   - `middle`: 2D filter (position + hidden velocity) tracking
#   - `Comments`: the current parameters, both filters' MSE, and
#     which one wins

# %%
utils.cell_hidden_variable_comparison_interactive()

# %% [markdown]
# **Guided usage**
# - Raise `process_var` while keeping `z_var` low
#   - Observe the MSE gap between the 2 filters in Comments narrows: the
#     2D filter's advantage shrinks when the dog's own motion is erratic
# - Raise `z_var` while keeping `process_var` low
#   - Observe both MSEs grow, but the 2D filter's MSE stays lower: it
#     still benefits from anticipating motion even with noisy sensors

# %% [markdown]
# **Implementation** `utils.cell_hidden_variable_comparison_interactive()`
# - Increasing `z_var`: both filters degrade, but the 2D filter degrades
#   less because it uses its motion model to bridge noisy measurements
# - Increasing `process_var`: the dog's velocity changes more
#   erratically; the advantage of the 2D filter is reduced but still
#   present

# %%
hintros.print_obj_info(utils.cell_hidden_variable_comparison_interactive)
