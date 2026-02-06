# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lesson 05.1: Hoeffding Inequality Interactive Study
#
# **Course**: MSML610: Advanced Machine Learning
#
# **Instructor**: Dr. GP Saggese
#
# **Purpose**: Interactive exploration of the Hoeffding inequality through
# Bernoulli binomial sampling experiments.

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import seaborn as sns

import utils_Lesson05_Learning_Theory as utils

# Set plotting style.
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %%
import msml610_utils as ut

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# # Step 1: Building Intuition about Hoeffding Inequality
#
# We will build intuition about the Hoeffding Inequality progressively through
# multiple cells, each adding more understanding about how sample means relate
# to the true probability.

# %% [markdown]
# ## Cell 1: Basic Bernoulli Sampling Code
#
# Start with a simple code example showing how to create Bernoulli samples and
# compute basic statistics.

# %%
# Demonstrate basic Bernoulli sampling.
utils.demonstrate_bernoulli_sampling()
# This shows the code for generating samples and computing the empirical mean.

# %% [markdown]
# ## Cell 2: Samples Over Time and Empirical PDF
#
# Visualize the N samples from a Bernoulli distribution both as a sequence over
# time and as an empirical probability distribution function (PDF).
#
# **Parameters**:
# - `mu`: True probability of success (between 0 and 1)
# - `N`: Number of samples to draw
# - `seed`: Random seed for reproducibility

# %%
# Display N samples over time and their empirical PDF.
utils.sample_bernoulli_with_pdf()

# %% [markdown]
# ## Cell 3: PDF, Empirical Mean, and Statistics
#
# Examine the probability distribution of N samples, compute the empirical mean
# nu, and compare with the theoretical mean and variance of the Bernoulli
# distribution.
#
# **Key concepts**:
# - Empirical mean nu = average of the samples
# - Theoretical mean = mu
# - Theoretical variance = mu * (1 - mu)
# - Change the seed to generate new realizations

# %%
# Display PDF, empirical mean nu, and compare with theoretical statistics.
utils.create_interactive_widget_cell3(
    utils.plot_bernoulli_pdf_cell2,
    mu_init=0.6,
    N_init=100,
    seed_init=42,
)
# Changing the seed generates new realizations with different empirical values.

# %% [markdown]
# ## Cell 4: Distribution of Empirical Mean
#
# Examine what happens when we repeatedly sample N points many times. Each
# trial produces an empirical mean nu. This cell shows the distribution of nu
# over many trials and compares it with the expected distribution predicted by
# the Law of Large Numbers and Central Limit Theorem.
#
# **Key concepts**:
# - By the Law of Large Numbers, nu converges to mu as N increases
# - By the Central Limit Theorem, nu is approximately normally distributed:
#   nu ~ N(mu, sqrt(mu * (1-mu) / N))

# %%
# Display the distribution of empirical mean nu from repeated sampling.
utils.create_interactive_widget_cell4(
    utils.plot_empirical_mean_distribution_cell3,
    mu_init=0.6,
    N_init=100,
    n_samples_init=1000,
    seed_init=42,
)
# As N increases, the distribution becomes more concentrated around mu.
