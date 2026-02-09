# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style.
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %%
import msml610_utils as ut
import L05_02_bias_variance_utils as utils

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# ## Cell 1: Approximation
#
# This cell demonstrates the concept of approximation - how well different models
# can fit a target function.
#
# **Purpose**: Visualize how a constant model (horizontal line) and a linear
# model (diagonal line) approximate a sinusoidal target function. This shows
# the difference in approximation capability between simple and more complex
# models.
#
# **Target Function**:
# - f(x) = sin(pi*x) for x in [-1, 1]
#
# **Models**:
# - **Constant Model (g_0)**: g_0(x) = b, where b is the mean of f(x)
# - **Linear Model (g_1)**: g_1(x) = a*x + b, where a and b are fitted using
#   least squares
#
# **Three plots**:
# 1. **True Function vs Constant Model**: Shows the target function (blue) and
#    the best constant approximation (green), with orange shading indicating
#    approximation error
# 2. **True Function vs Linear Model**: Shows the target function (blue) and
#    the best linear approximation (magenta), with orange shading indicating
#    approximation error
# 3. **Comments**: Displays the approximation errors and key observations
#
# **Key observations**:
# - The constant model has high approximation error - it cannot capture any
#   variation in the target function
# - The linear model has lower approximation error - it can capture the general
#   trend, though not the curvature
# - Lower approximation error means better fit, but doesn't always mean better
#   learning (as we'll see with bias-variance tradeoff)

# %%
# Display approximation comparison between constant and linear models.
utils.cell1_approximation()

# %% [markdown]
# ## Cell 2: Learning Once
#
# This cell demonstrates the difference between learning and approximation.
# Learning uses a limited training set, whereas approximation assumes full
# knowledge of the target function.
#
# **Purpose**: Show how models trained on a few random samples perform
# differently than models that approximate the full function. This illustrates
# the key difference between in-sample error (E_in) and out-of-sample error
# (E_out).
#
# **Parameters**:
# - `seed`: Random seed controlling which training points are sampled
# - `N_samples`: Number of random points in the training set (default: 2)
#
# **Target Function**:
# - f(x) = sin(pi*x) for x in [-1, 1]
#
# **Models**:
# - **Constant Model (g_0)**: Fitted to training data only
# - **Linear Model (g_1)**: Fitted to training data only
#
# **Three plots**:
# 1. **Constant Model**: Shows true function, fitted constant model, and
#    training points (red dots). Title displays E_in and E_out.
# 2. **Linear Model**: Shows true function, fitted linear model, and training
#    points (red dots). Title displays E_in and E_out.
# 3. **Comments**: Displays errors and key observations about learning vs
#    approximation.
#
# **Key observations**:
# - E_in measures how well the model fits the training data
# - E_out measures how well the model generalizes to the full function
# - With very few samples (e.g., N=2), a linear model can achieve E_in=0
#   (perfect fit on training data) but still have high E_out
# - This demonstrates that learning from limited data is fundamentally
#   different from approximation
# - Try different seeds to see how training set selection affects performance

# %%
# Display learning from N random samples with interactive controls.
utils.cell2_learning_once()

# %% [markdown]
# ## Cell 3: Learning (Bias-Variance Decomposition)
#
# This cell visualizes the bias-variance tradeoff by showing how models
# trained on different random training sets vary around the true function.
#
# **Purpose**: Demonstrate bias and variance decomposition by running multiple
# learning experiments with different training sets. Shows how model complexity
# affects both bias (systematic error) and variance (sensitivity to training
# data).
#
# **Parameters**:
# - `seed`: Random seed for reproducibility
# - `N_samples`: Number of training points per experiment (default: 2)
# - `N_experiments`: Number of different training sets to generate (default: 100)
#
# **Target Function**:
# - f(x) = sin(pi*x) for x in [-1, 1]
#
# **Visualization**:
# - For each experiment, sample N_samples random points and fit both models
# - Plot all fitted models with transparency (alpha=0.5) to show variation
# - Plot average model across all experiments as a dashed line
#
# **Three plots**:
# 1. **Constant Models**: Shows true function and all fitted constant models
#    (green lines with transparency). Dashed line shows average model.
# 2. **Linear Models**: Shows true function and all fitted linear models
#    (magenta lines with transparency). Dashed line shows average model.
# 3. **Comments**: Displays average errors and explanation of bias-variance
#    tradeoff.
#
# **Key observations**:
# - **Constant model (g_0)**: Low variance (all lines very similar), high bias
#   (far from true function). The model is too simple to capture the pattern.
# - **Linear model (g_1)**: Higher variance (lines spread out more), lower bias
#   (average model closer to true function). The model is more flexible but
#   sensitive to training data.
# - This illustrates the **BIAS-VARIANCE TRADEOFF**: simpler models have low
#   variance but high bias; more complex models have lower bias but higher
#   variance.
# - The total out-of-sample error decomposes as: E_out = bias^2 + variance + noise
# - Try increasing N_samples to see how more data reduces variance
# - Try increasing N_experiments to get more stable estimates of bias and variance

# %%
# Display bias-variance decomposition over multiple experiments.
utils.cell3_learning_bias_variance()

# %% [markdown]
# ## Cell 4: Learning Plots (Bias-Variance as Function of Training Set Size)
#
# This cell shows how bias, variance, and overall error change as we increase
# the number of training samples. This is a comprehensive view of the
# bias-variance tradeoff across different dataset sizes.
#
# **Purpose**: Visualize the bias-variance decomposition as a function of
# training set size (N_samples). Shows how more data affects both components
# of the learning error and demonstrates the classic bias-variance curves.
#
# **Parameters**:
# - `seed`: Random seed for reproducibility (fixed to ensure consistent comparison)
# - `N_experiments`: Number of experiments to average over for each N_samples value
# - `max_N_samples`: Maximum number of training samples to test
#
# **Error Decomposition**:
# For a deterministic target function (no noise):
# - **E_out = Bias² + Variance**
# - **Bias²**: Squared error between the average model (over all possible training sets) and the true function
# - **Variance**: Average squared deviation of individual models from the average model
# - **E_in**: In-sample error on training data
# - **E_out**: Out-of-sample error on the full function domain
#
# **Two plots**:
# 1. **Constant Model (g_0)**: Shows E_in, E_out, Bias², and Variance as functions of N_samples
# 2. **Linear Model (g_1)**: Shows E_in, E_out, Bias², and Variance as functions of N_samples
# 3. **Comments**: Displays the decomposition formula and key observations
#
# **Key observations**:
# - **Constant model**: Has very low variance (almost constant across N_samples) because it's insensitive to training data, but has high bias because it cannot capture the sinusoidal pattern
# - **Linear model**: Has higher variance (especially with few samples) because it's more flexible and sensitive to training data, but has lower bias because it can better approximate the target function
# - **As N_samples increases**: Variance decreases for both models (more data leads to more stable fits), while bias remains relatively constant (determined by model capacity)
# - **E_out decomposition**: You can verify that E_out ≈ Bias² + Variance by comparing the curves
# - **The bias-variance tradeoff**: Simpler models (constant) have low variance but high bias; more complex models (linear) have higher variance but lower bias
# - Try increasing N_experiments to get smoother, more stable curves
# - Try changing max_N_samples to see the trend over larger dataset sizes

# %%
# Display bias-variance decomposition as a function of N_samples.
utils.cell4_learning_plots()

# %% [markdown]
# ## Cell 5: Learning with Noise (Bias-Variance Decomposition)
#
# This cell extends Cell 3 by adding Gaussian noise to the training data,
# demonstrating how noise affects the bias-variance tradeoff.
#
# **Purpose**: Show how adding noise to the training data affects both the
# variance and out-of-sample error of learned models. Demonstrates that noise
# increases the difficulty of learning and adds a third term to the error
# decomposition.
#
# **Parameters**:
# - `seed`: Random seed for reproducibility
# - `N_samples`: Number of training points per experiment (default: 2)
# - `N_experiments`: Number of different training sets to generate (default: 100)
# - `noise_std`: Standard deviation of Gaussian noise added to training labels (default: 0.0)
#
# **Target Function**:
# - f(x) = sin(pi*x) for x in [-1, 1]
# - Training labels: y = f(x) + N(0, noise_std²)
#
# **Visualization**:
# - For each experiment, sample N_samples random points, add Gaussian noise, and fit both models
# - Plot all fitted models with transparency (alpha=0.5) to show variation
# - Plot average model across all experiments as a dashed line
#
# **Three plots**:
# 1. **Constant Models**: Shows true function and all fitted constant models
#    (green lines with transparency). Dashed line shows average model.
# 2. **Linear Models**: Shows true function and all fitted linear models
#    (magenta lines with transparency). Dashed line shows average model.
# 3. **Comments**: Displays average errors and explanation of noise effects.
#
# **Key observations**:
# - **With noise_std = 0**: Same behavior as Cell 3 (no noise case)
# - **With noise_std > 0**: Training data is corrupted by Gaussian noise
#   - Models try to fit the noisy observations instead of the true function
#   - This increases variance for both models (more sensitivity to data)
#   - E_out increases because models partially fit the noise
#   - The error decomposition becomes: **E_out = Bias² + Variance + σ²** (noise variance)
# - **Constant model**: Still has low variance, but noise increases E_out
# - **Linear model**: Variance increases significantly with noise (tries to fit noise)
# - Try increasing noise_std to see how noise affects the spread of fitted models
# - Try increasing N_samples to see how more data helps average out the noise

# %%
# Display bias-variance decomposition with noise over multiple experiments.
utils.cell5_learning_with_noise()

# %% [markdown]
# ## Cell 6: Learning Plots with Noise (Bias-Variance as Function of Training Set Size)
#
# This cell extends Cell 4 by adding Gaussian noise to the training data,
# showing how noise affects the bias-variance decomposition across different
# training set sizes.
#
# **Purpose**: Visualize how E_in, E_out, Bias², and Variance change as a
# function of training set size when training data is corrupted by Gaussian
# noise. Demonstrates how more data helps mitigate the effects of noise.
#
# **Parameters**:
# - `seed`: Random seed for reproducibility (fixed to ensure consistent comparison)
# - `N_experiments`: Number of experiments to average over for each N_samples value
# - `max_N_samples`: Maximum number of training samples to test
# - `noise_std`: Standard deviation of Gaussian noise added to training labels (default: 0.0)
#
# **Error Decomposition**:
# With noise:
# - **E_out = Bias² + Variance + σ²** (noise variance)
# - **Bias²**: Squared error between the average model and the true function
# - **Variance**: Average squared deviation of individual models from the average model
# - **σ²**: Noise variance (noise_std²) - irreducible error
# - **E_in**: In-sample error on training data
# - **E_out**: Out-of-sample error on the true function (without noise)
#
# **Two plots**:
# 1. **Constant Model (g_0)**: Shows E_in, E_out, Bias², and Variance as functions of N_samples
# 2. **Linear Model (g_1)**: Shows E_in, E_out, Bias², and Variance as functions of N_samples
# 3. **Comments**: Displays the decomposition formula and key observations
#
# **Key observations**:
# - **With noise_std = 0**: Same behavior as Cell 4 (deterministic case)
# - **With noise_std > 0**: Training data includes random noise
#   - Variance increases for both models compared to the no-noise case
#   - E_out increases by approximately σ² (the irreducible error from noise)
#   - As N_samples increases, variance decreases (more data averages out noise)
#   - Bias remains relatively constant (determined by model capacity, not noise)
# - **The noise term**: Represents the best possible error - even a perfect model cannot do better than σ² when learning from noisy data
# - **More data helps**: Increasing N_samples reduces the variance component but cannot reduce the noise component
# - Try setting noise_std = 0.1 or 0.2 to see the noise effect
# - Try increasing max_N_samples to see how variance continues to decrease with more data
# - Compare with Cell 4 (noise_std = 0) to see the additional error from noise

# %%
# Display bias-variance decomposition with noise as a function of N_samples.
utils.cell6_learning_plots_with_noise()
