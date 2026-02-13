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

import L05_01_02_bin_analogy_ml_utils as utils

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
# ## Cell 1: Visual Bin: Population of Marbles
#
# **Goal**:
# - Visualize the concept of an unknown population of marbles in a bin
# - Introduce the parameter $\mu$ as the true (unknown) proportion of red marbles
# - Build intuition for how we have a fixed population with a fixed proportion
#
# **Plots**:
# - Display a single plot:
#   - _Bin with Marbles_: A 2D grid showing red and green marbles arranged in the bin
#   - The visual distribution reflects the true proportion $\mu$
#
# **Parameters**:
# - `mu` ($\mu$): True proportion of red marbles in the population (0 to 1)
# - `seed`: Random seed controlling the spatial arrangement of marbles (for reproducibility)
#
# **Key observations**:
# - The bin represents a population with a fixed but "unknown" parameter $\mu$
# - In real-world scenarios, we don't know the true $\mu$ - we can only sample from the population
# - The spatial arrangement of marbles is random, but the overall proportion is always $\mu$
# - This sets up the fundamental problem: How do we estimate $\mu$ from samples?

# %%
utils.cell1_draw_bin_with_marbles_interactive()

# %% [markdown]
# ## Cell 2: Single Experiment: Is $\nu$ Close to $\mu$?
#
# **Goal**:
# - Demonstrate a single sampling experiment to estimate the population parameter
# - Show how we compute the sample proportion $\nu$ from $N$ random draws
# - Examine the relationship between $\nu$ (sample statistic) and $\mu$ (population parameter)
# - Understand that a single experiment gives us one estimate, which may or may not be close to $\mu$
#
# **Plots**:
# - Display two panels:
#   - _Population vs Sample_: Bar chart comparing $\mu$ (population) and $\nu$ (sample)
#     - Color-coded bars indicate how close $\nu$ is to $\mu$ (green = close, yellow = medium, red = far)
#   - _Interpretation_: Text box showing parameters, results, and assessment of the single experiment
#
# **Parameters**:
# - `mu` ($\mu$): True proportion of red marbles in the population (0 to 1)
# - `N` ($N$): Number of marbles to sample (sample size)
# - `seed`: Random seed for reproducibility of sampling
#
# **Key observations**:
# - A single experiment produces one sample proportion $\nu = \frac{1}{N}\sum_{i=1}^{N} x_i$
# - The error $|\nu - \mu|$ varies depending on the random sample drawn
# - Sometimes $\nu$ is close to $\mu$, sometimes it's not - we see natural sampling variability
# - Try different seeds to observe how $\nu$ changes across different random samples
# - This demonstrates why a single experiment is insufficient - we need to understand the distribution of $\nu$

# %%
utils.cell2_plot_single_experiment_interactive()

# %% [markdown]
# ## Cell 2.1: Limitations of Single Experiments
#
# - Single experiments don't tell the full story:
#   - We saw that $\nu$ can vary significantly across different random samples
#   - One experiment gives us a point estimate, but no information about reliability
#   - What we really need to know: $P(|\nu - \mu| > \epsilon)$ - the probability that our estimate is "far" from truth
# - Critical questions to consider:
#   - What if we got unlucky in our sample? How would we know?
#   - How confident can we be that $\nu \approx \mu$ based on a single observation?
#   - Does sample size $N$ matter? How much does it help?
#   - Can we quantify the uncertainty in our estimate?
# - The solution: "Let's repeat this many times..."
#   - By running many experiments, we can:
#     - Observe the distribution of $\nu$ values
#     - Estimate $P(|\nu - \mu| > \epsilon)$ empirically
#     - Understand how sample size $N$ affects the accuracy of $\nu$

# %% [markdown]
# ## Cell 3: Monte Carlo Simulation: Distribution of $\nu$
#
# **Goal**:
# - Run many repeated sampling experiments to understand the distribution of $\nu$
# - Empirically estimate $P(|\nu - \mu| > \epsilon)$ - the probability that $\nu$ is far from $\mu$
# - Observe how the distribution of $\nu$ concentrates around $\mu$ as we increase sample size $N$
# - Build intuition for the Law of Large Numbers and Central Limit Theorem
#
# **Plots**:
# - Display two panels:
#   - _Distribution of nu_: Histogram with KDE overlay showing the distribution of sample proportions across experiments
#     - Green dashed line: True parameter $\mu$
#     - Red shaded regions: Areas where $|\nu - \mu| > \epsilon$
#     - Statistics box: Mean, standard deviation, and empirical probability
#   - _Key Insights_: Text box summarizing parameters, results, and observations
#
# **Parameters**:
# - `mu` ($\mu$): True proportion of red marbles in the population (0 to 1)
# - `N` ($N$): Number of samples per experiment (sample size)
# - `n_experiments`: Number of repeated experiments to run
# - `eps` ($\epsilon$): Tolerance threshold for defining "far from $\mu$"
# - `seed`: Random seed for reproducibility
#
# **Key observations**:
# - The distribution of $\nu$ values clusters around the true $\mu$ - demonstrating unbiasedness
# - As $N$ increases, the distribution becomes tighter (smaller variance) - Law of Large Numbers
# - The empirical probability $P(|\nu - \mu| > \epsilon)$ decreases with larger $N$
# - The distribution approaches a normal shape - Central Limit Theorem
# - We can now quantify: "What fraction of experiments produce estimates within $\epsilon$ of $\mu$?"
# - This connects to confidence intervals and the precision of statistical estimates

# %%
utils.cell3_monte_carlo_simulation_interactive()

# %% [markdown]
# ## Cell 3.1: Observations from Monte Carlo Simulation
#
# - Distribution properties:
#   - The distribution of $\nu$ values is centered around the true $\mu$ - this shows that $\nu$ is an unbiased estimator
#   - The shape of the distribution approximates a normal (Gaussian) distribution - illustrating the Central Limit Theorem
#   - Most $\nu$ values fall close to $\mu$, with fewer extreme outliers
# - Effect of sample size $N$:
#   - As $N$ increases, the distribution becomes tighter (smaller variance: $\text{Var}(\nu) \propto \frac{1}{N}$)
#   - Larger $N$ means more precise estimates - the "spread" of possible $\nu$ values shrinks
#   - This is the Law of Large Numbers in action: $\nu \xrightarrow{P} \mu$ as $N \to \infty$
# - Probabilistic guarantees:
#   - The empirical probability $P(|\nu - \mu| > \epsilon)$ decreases with larger $N$
#   - We can quantify confidence: "With sample size $N$, approximately X% of experiments yield $\nu$ within $\epsilon$ of $\mu$"
#   - This connects to Hoeffding's inequality and other concentration bounds
# - Connection to machine learning:
#   - Replace "marbles in bin" with "data points" and "$\mu$" with "expected error"
#   - Sample proportion $\nu$ represents training error on $N$ samples
#   - True $\mu$ represents generalization error on the full population
#   - Key insight: With enough samples ($N$ large), training error $\nu$ is close to generalization error $\mu$
