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
import L05_02_overfitting_utils as utils

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# # Cell 1: True Target Function and Data Sampling
#
# **Goal**:
# - Visualize the true target function that we want to learn in a machine learning problem
# - Understand that in real-world scenarios, we don't have access to the complete target function $f(x)$
# - Demonstrate the process of sampling noisy observations from the true function
# - Show the train/test split: in-sample (training) data vs out-of-sample (test) data
# - Build intuition for the fundamental ML setup: learning from limited, noisy samples
#
# **Plots**:
# - Display four panels:
#   - _True Target Function_: Complete unknown function $f(x)$ we want to learn
#     - Blue solid line: Noiseless true function
#     - Blue transparent line: Noisy version (if $\epsilon > 0$)
#   - _In-Sample Data (80%)_: Green points representing training data
#     - These $n_{\text{train}}$ points are used to fit the model
#   - _Out-of-Sample Data (20%)_: Red points representing test data
#     - These $n_{\text{test}}$ points are used to evaluate generalization
#   - _Comments_: Text summary of parameters and key observations
#
# **Parameters**:
# - `Function`: Select the true target function from available options:
#   - Slow Sinusoid: $f(x) = \sin(0.5\pi x)$
#   - Fast Sinusoid: $f(x) = \sin(2\pi x)$
#   - Parabola: $f(x) = 2x^2 - 1$
#   - Constant: $f(x) = 0$
#   - Linear: $f(x) = x$
# - `epsilon` ($\epsilon$): Standard deviation of Gaussian noise added to observations (noise level)
# - `N (total samples)` ($N$): Total number of data points to sample from the function
# - `seed`: Random seed for reproducibility of sampling
#
# **Key observations**:
# - The complete curve represents the unknown target function $f(x)$ that we wish to learn
# - In practice, we only have access to a finite set of noisy samples: $(x_i, y_i)$ where $y_i = f(x_i) + \epsilon_i$
# - Data is split into:
#   - Training set (80%): Used to learn the model parameters
#   - Test set (20%): Used to evaluate how well the model generalizes
# - The fundamental ML challenge: Learn from green (training) points to predict well on red (test) points
# - Increasing $N$ provides more information, while increasing $\epsilon$ makes learning harder due to noise

# %%
# Display the true target function with interactive controls.
utils.cell1_plot_true_target_function()

# %% [markdown]
# ## Cell 2: Model Comparison - Constant vs Linear
#
# **Goal**:
# - Compare two hypothesis classes: constant model $h(x) = b$ vs linear model $h(x) = ax + b$
# - Understand the bias-variance tradeoff through concrete examples
# - Observe how model complexity affects approximation quality and stability
# - Visualize in-sample error ($E_{\text{in}}$) vs out-of-sample error ($E_{\text{out}}$)
# - Demonstrate how different models fit the same training data and generalize to test data
#
# **Note**: This cell uses the same configuration as Cell 1. All parameters (function type, $\epsilon$, $N$, seed) are synchronized with Cell 1. To change the setup, adjust the parameters in Cell 1.
#
# **Models**:
# - **Constant model**: $h(x) = b$
#   - Parameter $b$ is the mean of training $y$-values: $b = \frac{1}{n}\sum_{i=1}^{n} y_i$
#   - Hypothesis class: $\mathcal{H}_0 = \{h(x) = b : b \in \mathbb{R}\}$
#   - Characteristics: High bias (limited expressiveness), low variance (stable across datasets)
# - **Linear model**: $h(x) = ax + b$
#   - Parameters $a, b$ are fit using least squares regression
#   - Hypothesis class: $\mathcal{H}_1 = \{h(x) = ax + b : a, b \in \mathbb{R}\}$
#   - Characteristics: Lower bias (more expressive), higher variance (sensitive to training data)
#
# **Plots**:
# - Display four panels:
#   - _In-Sample Data_: Green training points with fitted model
#     - Shows model fit on training data
#     - Displays $E_{\text{in}}$ (training error)
#   - _Out-of-Sample Data_: Red test points with fitted model
#     - Shows model generalization on test data
#     - Displays $E_{\text{out}}$ (test error)
#   - _True Function vs Model_: Comparison between target function and learned model
#     - Blue line: True target function $f(x)$
#     - Model line: Fitted hypothesis $h(x)$
#     - Orange shaded area: Approximation error between $f(x)$ and $h(x)$
#   - _Comments_: Text summary showing learned parameters, errors, and observations
#
# **Parameters**:
# - `Model Type`: Dropdown to select between Constant and Linear models
# - `Resample and Relearn`: Button to generate new training data (increments seed) and refit the model
#
# **Key observations**:
# - **Constant model** ($\mathcal{H}_0$):
#   - HIGH BIAS: Poor approximation of complex target functions (large orange shaded area)
#   - LOW VARIANCE: Very stable - produces similar $h(x)$ across different training sets
#   - Simple model (1 parameter) cannot capture patterns in data
#   - $E_{\text{in}}$ and $E_{\text{out}}$ are typically close (low variance)
# - **Linear model** ($\mathcal{H}_1$):
#   - LOWER BIAS: Better approximation capability (smaller orange shaded area for linear targets)
#   - HIGHER VARIANCE: More sensitive to specific training points - $h(x)$ changes more with resampling
#   - More complex model (2 parameters) can capture linear trends
#   - $E_{\text{in}}$ and $E_{\text{out}}$ may differ more (higher variance)
# - Bias-variance tradeoff:
#   - Constant model: Underfits (high bias) but is stable (low variance)
#   - Linear model: Better fit but less stable (bias-variance tradeoff)
#   - Click "Resample and Relearn" multiple times to observe variance: how much does $h(x)$ change?
#   - Compare $E_{\text{in}}$ vs $E_{\text{out}}$ to assess generalization

# %%
# Display model learning with interactive controls.
utils.cell2_plot_model()

# %%
