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
# # Lesson 05.1: Hoeffding Inequality
#
# **Course**: MSML610: Advanced Machine Learning
#
# **Instructor**: Dr. GP Saggese

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import seaborn as sns

import utils_Lesson05_Learning_Theory_Hoeffding_Inequality as utils

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
# # Cell 1: Building Intuition about Hoeffding Inequality

# %% [markdown]
# ## Cell 1.1: Basic Bernoulli Sampling Code
#
# - Demonstrate basic Bernoulli sampling
# - Show the code for:
#   - Generating Bernoulli samples
#   - Computing the empirical mean $\nu$
#   - Comparing with the true mean $\mu$

# %%
# Demonstrate basic Bernoulli sampling.
utils.cell1_basic_bernoulli_sampling()

# %% [markdown]
# ## Cell 1.2: Samples Over Time and Empirical PDF
#
# - Visualize $N$ samples from a Bernoulli distribution:
#   - As a sequence over time
#   - As an empirical probability distribution function (PDF)
#
# **Parameters**:
# - `mu` ($\mu$): True probability of success (between 0 and 1)
# - `N` ($N$): Number of samples to draw
# - `seed`: Random seed for reproducibility

# %%
# Display N samples over time and their empirical PDF.
utils.cell2_samples_over_time_and_pdf()

# %% [markdown]
# ## Cell 1.3: Distribution of Empirical Mean
#
# - Examine what happens when we repeatedly sample $N$ points many times
# - Each trial produces an empirical mean $\nu$
# - This cell:
#   - Shows the distribution of $\nu$ over many trials
#   - Compares it with the expected distribution predicted by:
#     - Law of Large Numbers
#     - Central Limit Theorem
#
# **Parameters**:
# - `mu` ($\mu$): True probability of success (between 0 and 1)
# - `N` ($N$): Number of samples drawn in each trial
# - `n_samples`: Number of trials to repeat the experiment (how many times we compute $\nu$)
# - `seed`: Random seed for reproducibility
#
# **Key concepts**:
# - By the Law of Large Numbers: $\nu$ converges to $\mu$ as $N$ increases
# - By the Central Limit Theorem: $\nu$ is approximately normally distributed:
#   - $\nu \sim \mathcal{N}\left(\mu, \sqrt{\frac{\mu(1-\mu)}{N}}\right)$

# %%
# Display the distribution of empirical mean nu from repeated sampling.
utils.cell3_distribution_empirical_mean()

# %% [markdown]
# # Cell 2: Hoeffding Inequality: Theoretical Bounds
#
# - The Hoeffding inequality provides a concentration bound
# - It quantifies how quickly the sample mean converges to the true mean as $N$
#   increases

# %% [markdown]
# ## Cell 2.1: Hoeffding Inequality Statement
#
# - For $N$ independent Bernoulli random variables $X_1, \ldots, X_N$ with
#   probability $\mu$
# - Let $\nu = \frac{1}{N} \sum_{i=1}^{N} X_i$ be the sample mean
#
# - The Hoeffding inequality states:
#
# $$P(|\nu - \mu| \geq \epsilon) \leq 2 \exp(-2N\epsilon^2)$$
#
# - Where:
#   - $\nu$ is the sample mean (empirical probability)
#   - $\mu$ is the true probability
#   - $\epsilon > 0$ is the deviation threshold
#   - $N$ is the number of samples
#
# **Key insights**:
# - The bound decreases exponentially with $N$
# - The bound is independent of $\mu$ (distribution-free)
# - Larger $\epsilon$ requires larger $N$ for the same confidence
# - The factor of 2 accounts for both tails:
#   - $\nu > \mu + \epsilon$
#   - $\nu < \mu - \epsilon$

# %% [markdown]
# ## Cell 2.2: Interactive Hoeffding Inequality Demonstration
#
# - This interactive visualization demonstrates the Hoeffding inequality across
#   multiple probability distributions
# - The Hoeffding inequality is distribution-free:
#   - It applies to any bounded random variable in [0, 1]
#   - Regardless of its specific distribution
#
# - The visualization shows four plots:
#   - **Underlying Distribution**: The PDF/PMF of the selected distribution
#     showing the shape of the random variable $X$
#   - **Distribution of Sample Mean**: Histogram of sample means $\nu$ from
#     repeated sampling, with tail areas highlighted in red
#   - **Bound vs Empirical**: Comparison of theoretical Hoeffding bound vs
#     empirical probability
#   - **Comments**: Parameters and interpretation
#
# - Note: The bound is capped at 1.0 since probabilities cannot exceed 1
#
# **Distribution options**:
# - **Bernoulli**: Binary outcomes (0 or 1), parameter $\mu$ is success
#   probability
# - **Uniform [0, 1]**: Continuous uniform distribution ($\mu$ parameter ignored)
# - **Binomial (scaled)**: Binomial(10, $\mu$) scaled to [0, 1]
# - **Truncated Gaussian**: Normal($\mu$, 0.2) truncated to [0, 1]
# - **Truncated Exponential**: Exponential with mean near $\mu$, truncated to
#   [0, 1]
#
# **Parameters**:
# - `Distribution`: Select the probability distribution
# - `mu` ($\mu$): Distribution parameter (interpretation varies by distribution)
# - `N` ($N$): Number of samples per trial (larger $N$ = tighter concentration)
# - `epsilon` ($\epsilon$): Deviation threshold (smaller $\epsilon$ = stricter
#   bound)
# - `seed`: Random seed for reproducibility
#
# **Key insight**:
# - The Hoeffding bound works for ALL these distributions
# - Without knowing which one is being used
# - This is the power of distribution-free bounds
#
# **Experiments to try**:
# - Compare Bernoulli vs Uniform:
#   - Both satisfy the bound despite different shapes
# - Increase $N$:
#   - See how all distributions concentrate around their mean
# - Try Truncated Gaussian with different $\mu$ values:
#   - The bound still holds even though the distribution shape changes
#     dramatically near boundaries
# - Compare bound tightness:
#   - Some distributions give tighter empirical probabilities than others
#   - But the bound always holds

# %%
# Demonstrate the Hoeffding inequality with multiple distributions.
utils.cell4_hoeffding_inequality_demo()

# %% [markdown]
# ## Cell 2.3: Empirical Probability vs Hoeffding Bound
#
# - This visualization shows how both the theoretical Hoeffding bound and the
#   empirical probability change
# - We vary one parameter while holding the other fixed
# - This helps understand:
#   - **Exponential decay**: Both quantities decrease exponentially
#   - **Bound validity**: The empirical probability is always below the bound
#   - **Bound tightness**: How close the empirical probability is to the bound
#   - **Parameter trade-offs**: The relationship between $N$ and $\epsilon$
#
# **Two scanning modes**:
#
# - **Scan $N$ (fix $\epsilon$)**:
#   - Shows how increasing sample size $N$ improves concentration
#   - For a fixed deviation threshold $\epsilon$
#   - Both bound and empirical probability decrease exponentially with $N$
#   - Demonstrates why we need relatively few samples for good concentration
#   - Useful for determining required sample size for target confidence
#
# - **Scan $\epsilon$ (fix $N$)**:
#   - Shows how the probability of large deviations decreases
#   - As we increase the tolerance $\epsilon$
#   - Both quantities decrease as $\epsilon$ increases
#   - Larger $\epsilon$ means more tolerance, so deviation probability drops
#   - Useful for understanding achievable precision for given sample size
#
# **Interactive controls**:
# - `Distribution`: Select probability distribution
# - `Scan variable`: Choose to scan $N$ or $\epsilon$
# - `mu` ($\mu$): Distribution parameter
# - `fixed_N`: $N$ value used when scanning $\epsilon$
# - `fixed_epsilon`: $\epsilon$ value used when scanning $N$
# - `seed`: Random seed for reproducibility
#
# **Key observation**:
# - The empirical probability (blue line) is always at or below the theoretical
#   bound (red line)
# - This confirms the Hoeffding inequality
# - The gap between them shows how conservative the bound is

# %%
# Visualize how bound and empirical probability change with N or epsilon.
utils.cell5_empirical_vs_bound()

# %% [markdown]
# ## Cell 2.4: Hoeffding Bound as a Function of $N$ and $\epsilon$
#
# - The Hoeffding bound formula is:
#
# $$\text{Bound} = 2 \exp(-2N\epsilon^2)$$
#
# - This interactive visualization shows how the bound changes as we vary $N$
#   and $\epsilon$
# - Understanding this relationship is crucial for:
#   - Choosing appropriate sample sizes $N$ for a desired confidence level
#   - Understanding the trade-off between deviation tolerance ($\epsilon$) and
#     sample requirements
#   - Seeing the exponential decay in both $N$ and $\epsilon^2$
#
# **View modes**:
# - **Heatmap**:
#   - Shows the bound value as a color map across all $(N, \epsilon)$
#     combinations
# - **Fix $N$, vary $\epsilon$**:
#   - See how increasing tolerance (larger $\epsilon$) affects the bound
#   - For a fixed sample size
# - **Fix $\epsilon$, vary $N$**:
#   - See how increasing sample size improves the bound
#   - For a fixed deviation threshold
# - **Contour plot**:
#   - Shows curves of constant probability
#   - Useful for finding $(N, \epsilon)$ pairs that achieve the same confidence
#
# **Key observations**:
# - The bound is exponentially sensitive to both $N$ and $\epsilon$
# - To halve $\epsilon$ while maintaining the same bound, you need to quadruple
#   $N$
# - For practical confidence levels (e.g., 0.05):
#   - The required $N$ grows quadratically with $1/\epsilon$

# %%
# Explore the Hoeffding bound as a function of N and epsilon.
utils.cell6_bound_surface_heatmap()

# %% [markdown]
# ## Cell 2.5: 3D Surface Visualization of Hoeffding Bound
#
# - This cell provides a three-dimensional surface plot of the Hoeffding bound
# - Offering a different perspective on how the bound varies with $N$ and
#   $\epsilon$
#
# - The 3D surface makes it easier to:
#   - Visualize the exponential decay in both dimensions simultaneously
#   - See the steepest descent directions
#   - Understand the "valley" structure where the bound is smallest
#   - Rotate the view to examine the surface from different angles
#
# **Interactive controls**:
# - `N_max`, `epsilon_max`: Control the range of the surface
# - `elevation`: Viewing angle from above (0=horizontal, 90=top-down)
# - `azimuth`: Rotation angle around the vertical axis
# - `Use log scale for Z-axis`: Toggle logarithmic scale for better visibility
#   of small bound values
#
# **Suggested experiments**:
# - Start with default view to see the overall shape
# - Rotate using azimuth slider (0 to 360 degrees) to view from different sides
# - Change elevation to see the surface from different heights
# - Enable log scale to better see the structure at small bound values
# - Compare with the heatmap view above to build intuition

# %%
# Visualize the Hoeffding bound as a 3D surface.
utils.cell7_bound_3d_surface()
