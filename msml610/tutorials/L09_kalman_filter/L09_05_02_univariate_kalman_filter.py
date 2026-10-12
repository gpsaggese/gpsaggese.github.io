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
# # Univariate Kalman filter
#
# - This notebook builds a univariate Kalman filter from the sum and the
#   product of Gaussians, and uses it to track a dog moving in a hallway
#   with a noisy sensor
# - The pedagogical arc:
#   - Representing a Gaussian, and its sum and product
#   - The predict-update cycle, and simulating the dog and its sensor
#   - Running the Kalman filter, with a bad initial estimate, extreme noise,
#     and too much belief in the model
#   - An interactive exploration of the filter

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import numpy as np

# %%
import helpers.hintrospection as hintros
import helpers.hio as hio
import helpers.hnotebook as hnotebook

import L09_05_02_univariate_kalman_filter_utils as utils

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
# # Part 1: Sum and Product of Gaussians

# %% [markdown]
# ## Cell 1.1: Representing a Gaussian
#
# **Goal**
# - Introduce `utils.Gaussian(mean, var)`, the `(mean, var)` named tuple
#   used to represent a belief throughout this notebook

# %%
x = utils.Gaussian(3.4, 10.1)
print("x=", x)
print("x.mean=", x.mean)
print("x.var=", x.var)

# %% [markdown]
# ## Cell 1.2: Sum of Gaussians
#
# - Given 2 Gaussians $X$ and $Y$
#   $$X \sim Normal(\mu_1, \sigma_1^2)$$
#   $$Y \sim Normal(\mu_2, \sigma_2^2)$$
# - For correlated Gaussians with correlation coefficient $\rho$, the sum
#   $Z = X + Y$ is a Gaussian $Normal(\mu, \sigma^2)$ with:
#   $$\mu = \mu_1 + \mu_2$$
#   $$\sigma^2 = \sigma_1^2 + \sigma_2^2 + 2\rho\sigma_1\sigma_2$$
# - **Interpretation:**
#   - The mean is the sum of the means (by linearity)
#   - For independent Gaussians ($\rho = 0$), the variance is the sum of
#     variances (uncertainty increases)
#   - Positive correlation increases variance, negative correlation
#     decreases it
#
# **Implementation** `utils.gaussian_sum(x, y)`

# %%
# Sum two Gaussians.
x = utils.Gaussian(10, 0.2**2)
y = utils.Gaussian(15, 0.7**2)

z = utils.gaussian_sum(x, y)
print("z=", z)

# %%
ax = utils.plot_gaussian(x, label="x")
utils.plot_gaussian(y, ax=ax, label="y")
_ = utils.plot_gaussian(z, ax=ax, label="z")

# %% [markdown]
# **Description**
# - Inputs
#   - `mu1` ($\mu_1$): mean of $X$
#   - `sigma1` ($\sigma_1$): standard deviation of $X$
#   - `mu2` ($\mu_2$): mean of $Y$
#   - `sigma2` ($\sigma_2$): standard deviation of $Y$
#   - `rho` ($\rho$): correlation coefficient between $X$ and $Y$
#
# - Panels
#   - `left`: $X$ and $Y$ as filled PDFs, the analytical sum $Z$, and
#     the sampled numerical sum

# %%
utils.cell1_1_plot_gaussian_sum()

# %% [markdown]
# **Guided usage**
# - Set `rho=0`
#   - Observe $Z$'s variance equals $\sigma_1^2 + \sigma_2^2$ exactly
# - Raise `rho` toward `1`
#   - Observe $Z$'s variance grows beyond $\sigma_1^2 + \sigma_2^2$
# - Lower `rho` toward `-1`
#   - Observe $Z$'s variance shrinks below $\sigma_1^2 + \sigma_2^2$
# - In every case, observe $Z$'s mean stays $\mu_1 + \mu_2$

# %% [markdown]
# **Implementation** `utils.cell1_1_plot_gaussian_sum()`
# - Plots input Gaussians $X$ (blue) and $Y$ (yellow) as filled PDFs, the
#   analytical sum $Z$ (red line), and a numerical sum via sampling
#   (light coral histogram)

# %%
hintros.print_obj_info(utils.cell1_1_plot_gaussian_sum)

# %% [markdown]
# ## Cell 1.3: Product of Gaussians
#
# - Given 2 Gaussians $X$ and $Y$
#   $$X \sim Normal(\mu_X, \sigma_X^2)$$
#   $$Y \sim Normal(\mu_Y, \sigma_Y^2)$$
# - The product $Z = X \cdot Y$ (PDF multiplication) is a Gaussian
#   $Normal(\mu_Z, \sigma_Z^2)$ with:
#   $$\mu_Z = \frac{\mu_X \sigma_Y^2 + \mu_Y \sigma_X^2}{\sigma_X^2 +
#     \sigma_Y^2}$$
#   $$\sigma_Z^2 = \frac{\sigma_X^2 \sigma_Y^2}{\sigma_X^2 + \sigma_Y^2}$$
# - **Interpretation:**
#   - Reduces variance by incorporating more information
#   - If one Gaussian $X$ is narrower (more accurate), the result leans
#     towards $X$
#   - If the 2 Gaussians are similar (measures corroborate), the result
#     becomes more certain

# %% [markdown]
# **Gaussian products in terms of precision**
#
# - The precision of a Gaussian is
#   $$\tau = \frac{1}{\sigma^2}$$
# - The precision of the product is the sum of the precisions
#   $$\tau_Z = \tau_X + \tau_Y$$
#   $$\sigma_Z^2 = \frac{1}{\frac{1}{\sigma_X^2} + \frac{1}{\sigma_Y^2}}$$
# - The mean is the average of the means weighted by the precisions
#   $$\mu_Z = \sigma_Z^2 (\frac{\mu_X}{\sigma_X^2} +
#     \frac{\mu_Y}{\sigma_Y^2})$$
#
# - The mean is averaged towards the more certain Gaussian
# - The variance is smaller than both
#
# **Implementation** `utils.gaussian_multiply(x, y)`

# %%
# Product of two equal Gaussians.
x = utils.Gaussian(10, 1.0)

z = utils.gaussian_multiply(x, x)
print("z=", z)

# The result is more certain than both.

# %%
ax = utils.plot_gaussian(x, label="x")
utils.plot_gaussian(x, ax=ax, label="x")
_ = utils.plot_gaussian(z, ax=ax, label="z")

# %%
# Product of two different Gaussians.
x = utils.Gaussian(10, 0.2**2)
y = utils.Gaussian(15, 0.7**2)

z = utils.gaussian_multiply(x, y)
print("z=", z)

ax = utils.plot_gaussian(x, label="x")
utils.plot_gaussian(y, ax=ax, label="y")
_ = utils.plot_gaussian(z, ax=ax, label="z")

# %%
x = utils.Gaussian(10.2, 1)
y = utils.Gaussian(9.7, 1)

z = utils.gaussian_multiply(x, y)
print("z=", z)

ax = utils.plot_gaussian(x, label="x")
utils.plot_gaussian(y, ax=ax, label="y")
_ = utils.plot_gaussian(z, ax=ax, label="z")

# %% [markdown]
# **Description**
# - Inputs
#   - `mu1` ($\mu_1$): mean of $X$
#   - `sigma1` ($\sigma_1$): standard deviation of $X$
#   - `mu2` ($\mu_2$): mean of $Y$
#   - `sigma2` ($\sigma_2$): standard deviation of $Y$
#
# - Panels
#   - `left`: $X$ and $Y$ as filled PDFs, the analytical product $Z$,
#     and the sampled numerical product

# %%
utils.cell1_2_plot_gaussian_product()

# %% [markdown]
# **Guided usage**
# - Set `sigma1` much smaller than `sigma2`
#   - Observe $Z$'s mean sits close to `mu1`: the product pulls the mean
#     toward the more certain (narrower) Gaussian
# - Set `mu1` close to `mu2`, both with small sigmas
#   - Observe $Z$ becomes very sharp: when the inputs agree, the product
#     is highly certain
# - Compare $Z$'s width to both inputs
#   - Observe it is always narrower than either input

# %% [markdown]
# **Implementation** `utils.cell1_2_plot_gaussian_product()`
# - Plots input Gaussians $X$ (blue) and $Y$ (yellow) as filled PDFs, the
#   analytical product $Z$ (red line), and a numerical product via
#   importance sampling (light coral histogram)

# %%
hintros.print_obj_info(utils.cell1_2_plot_gaussian_product)

# %% [markdown]
# # Part 2: Tracking the Dog

# %% [markdown]
# ## Cell 2.1: The predict-update cycle
#
# - The intuition is the same as the discrete case
# - There is a cycle of prediction and updates
#   1. Predict: prior = x_est using system model
#   2. Update: posterior = likelihood * prior
# - Create prior (using current estimate and system model)
#   - `prior = predict(x, process_model)`
# - Create likelihood (using measurement)
#   - `likelihood = gaussian(z, sensor_var)`
# - Update belief using prior and likelihood
#   - `x = update(prior, likelihood)`
# - Sum adds uncertainty; multiplication reduces uncertainty
# - Let's assume that the dog moves in the hallway, back and forth (it's
#   not circular), and we have a sensor that measures the distance of the
#   dog from one extreme

# %% [markdown]
# We can use Newton's equation of motion to compute the position of the
# dog, based on current position and velocity
#
# $$\overline{x}_k = x_{k-1} + v_k \Delta_t$$
#
# - $x_{k-1}$ has uncertainty quantified by a Gaussian
# - $v_k$ has also uncertainty quantified by a Gaussian
#
# We can compute the sum of 2 Gaussians in terms of mean and uncertainty.
# It makes sense since we know that uncertainty becomes larger.
#
# The likelihood $z | x$ is the probability of measures given the current
# state.

# %% [markdown]
# ## Cell 2.2: Simulating the dog and its sensor
#
# **Goal**
# - Simulate a dog moving at constant velocity, with a noisy sensor, to
#   generate the data the filter will track in Cell 2.3
#
# **Implementation** `utils.DogSimulation(...)`

# %%
np.random.seed(13)

# Variance in the dog's movement.
process_var = 1.0
# Variance in the sensor.
sensor_var = 2.0

# Dog's initial position.
x = utils.Gaussian(0.0, 20.0**2)
velocity = 1.0
# Time step in seconds.
dt = 1.0
# Displacement to add to x (representing how to model the movement of the
# dog).
process_model = utils.Gaussian(velocity * dt, process_var)

# Simulate dog and get measurements.
dog = utils.DogSimulation(
    x0=x.mean,
    velocity=process_model.mean,
    measurement_var=sensor_var,
    process_var=process_model.var,
)

# Simulate dog and collect measurements and actual positions.
n_steps = 10
sim_data = [dog.move_and_sense() for _ in range(n_steps)]
zs = [m for m, _ in sim_data]
actual_positions = [pos for _, pos in sim_data]
print("zs=", zs)

# %% [markdown]
# ## Cell 2.3: Running the Kalman filter
#
# **Goal**
# - Run the predict-update cycle from Cell 2.1 on the simulated data, and
#   check that the posterior's uncertainty settles below the sensor's

# %%
# Perform Kalman filter on measurements.
kf_info = []
for z, actual_pos in zip(zs, actual_positions):
    prior = utils.predict(x, process_model)
    likelihood = utils.Gaussian(z, sensor_var)
    x = utils.update(prior, likelihood)
    kf_info.append(
        utils.KfInfo(
            prior=prior, measurement=z, actual_pos=actual_pos, posterior=x
        )
    )

print(utils.kf_info_to_df(kf_info))

# %% [markdown]
# - The uncertainty after prediction is larger than the uncertainty after
#   update (as usual)
# - The variance of the prior at time 0 is very large, but after we
#   measure, the variance of the measurement "dominates"
# - The posterior values are always between the measurement and the
#   prior
# - After a few cycles the posterior variance is around 1, which is
#   smaller than the sensor variance (~2): using a model + measurements
#   is better

# %%
# Plot Kalman filter results.
utils.plot_kf_info(kf_info, show_actual_pos="scatter")

# %% [markdown]
# ## Cell 2.4: Bad initial estimate
#
# **Goal**
# - Start the filter's belief far from the dog's true starting position,
#   and check that it still converges given enough measurements

# %%
seed = 42
process_var = 2.0
sensor_var = 2.0**2
# Belief about initial position.
initial_position = 400
initial_pos_var = 1.0
# Actual initial position.
actual_initial_pos = 0.0
n_steps = 100

kf_info = utils._run_dog_simulation(
    seed,
    process_var,
    sensor_var,
    initial_position,
    actual_initial_pos=actual_initial_pos,
    initial_pos_var=initial_pos_var,
    n_steps=n_steps,
)
utils.plot_kf_info(
    kf_info, show_prior="none", show_actual_pos="line", show_posterior="line"
)

# %%
# Belief about initial position.
initial_position = 400
initial_pos_var = 100.0
# Actual initial position.
actual_initial_pos = 0.0

kf_info = utils._run_dog_simulation(
    seed,
    process_var,
    sensor_var,
    initial_position,
    actual_initial_pos=actual_initial_pos,
    initial_pos_var=initial_pos_var,
    n_steps=n_steps,
)
utils.plot_kf_info(
    kf_info, show_prior="none", show_actual_pos="line", show_posterior="line"
)

# %% [markdown]
# ## Cell 2.5: Extreme amount of noise
#
# **Goal**
# - Push the sensor noise to an extreme, and check that the filter still
#   recovers the dog's position as long as the process model is trusted

# %%
seed = 42
process_var = 2.0
sensor_var = 300.0**2
initial_position = 0.0
n_steps = 1000

kf_info = utils._run_dog_simulation(
    seed, process_var, sensor_var, initial_position, n_steps=n_steps
)
utils.plot_kf_info(
    kf_info, show_prior="none", show_actual_pos="line", show_posterior="line"
)

# %% [markdown]
# - Even with extreme amounts of noise we recover the position of the
#   dog
# - This is because the process error is small (we can trust the model)

# %% [markdown]
# ## Cell 2.6: Too much belief in the model
#
# **Goal**
# - Have the dog accelerate, violating the constant-velocity process
#   model, and check that an over-confident filter fails to track it

# %%
seed = 42
process_var = 2.0
sensor_var = 300.0**2
acceleration = 0.04
initial_position = 0.0
n_steps = 300

kf_info = utils._run_dog_simulation(
    seed,
    process_var,
    sensor_var,
    initial_position,
    acceleration=acceleration,
    n_steps=n_steps,
)
utils.plot_kf_info(
    kf_info, show_prior="none", show_actual_pos="line", show_posterior="line"
)

# %% [markdown]
# - The filter is not able to follow the change of velocity of the dog

# %%
seed = 42
process_var = 2.0
sensor_var = 2.0**2
acceleration = 0.04
initial_position = 0.0
n_steps = 50

kf_info = utils._run_dog_simulation(
    seed,
    process_var,
    sensor_var,
    initial_position,
    acceleration=acceleration,
    n_steps=n_steps,
)
utils.plot_kf_info(
    kf_info,
    show_prior="none",
    show_actual_pos="line",
    show_posterior="line",
    show_measurements="none",
)

# %% [markdown]
# ## Cell 2.7: Interactively exploring the filter
#
# **Goal**
# - Let students sweep every parameter from Cells 2.4-2.6 at once, to
#   consolidate the intuition for how the filter balances the process
#   model against sensor measurements

# %% [markdown]
# **Description**
# - Inputs
#   - `process_var` ($\sigma_p^2$): variance in the dog's movement
#     model
#   - `sensor_var` ($\sigma_s^2$): variance in the sensor
#     measurements
#   - `initial_position` ($x_0$): belief about the dog's starting
#     position
#   - `actual_initial_pos`: the dog's true starting position (can
#     differ from the belief to simulate a wrong initial estimate)
#   - `initial_pos_var`: uncertainty (variance) in the initial
#     position belief; large values mean we are very uncertain about
#     where the dog starts
#   - `acceleration`: dog's acceleration ($m/s^2$); non-zero values
#     make the dog speed up over time, testing the filter's ability to
#     track
#   - `seed`: random seed for reproducibility
#
# - Panels
#   - `left`: prior, measurements, posterior, and posterior
#     uncertainty bands over time
#   - `Comments`: the current parameters, and the final posterior
#     mean/variance

# %%
utils.cell2_interactive_dog_simulation()

# %% [markdown]
# **Guided usage**
# - Lower `sensor_var` far below `process_var`
#   - Observe the posterior in Comments tracks the measurements closely:
#     the filter trusts the sensor
# - Lower `process_var` far below `sensor_var`
#   - Observe the posterior changes less: the motion model dominates
# - Let the simulation run its full 25 steps
#   - Observe the final posterior variance in Comments stabilizes below
#     both input variances
# - Is it better to have precise measurements (`sensor_var` <<
#   `process_var`) or vice versa?

# %% [markdown]
# **Implementation** `utils.cell2_interactive_dog_simulation()`
# - Prior (predict) as red up-triangles: prediction from motion model
# - Measurement as black circles: noisy sensor readings
# - Posterior (update) as green down-triangles: filtered estimate
# - Posterior uncertainty as shaded green bands (+-1, 2, 3 sigma)

# %%
hintros.print_obj_info(utils.cell2_interactive_dog_simulation)
