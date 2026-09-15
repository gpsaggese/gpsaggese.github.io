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
# # Hoeffding Inequality

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import seaborn as sns

import helpers.hintrospection as hintros
import L05_01_01_hoeffding_inequality_utils as utils

# Set plotting style.
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %%
import helpers.htutorial as ut

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# # Part 1: Building Intuition about Hoeffding Inequality

# %% [markdown]
# ## Cell 1.1: Basic Bernoulli sampling code
#
# **Goal**:
# - Walk through the mechanics of Bernoulli sampling in plain code, before
#   any interactive widget: draw samples, compute the empirical mean $\nu$,
#   and compare it against the true mean $\mu$
#
# **Implementation**: `cell1_1_basic_bernoulli_sampling(mu=0.6, N=10,
# seed=42)`
# - Draws `N` Bernoulli(`mu`) samples with `_generate_bernoulli_samples()`
# - Prints the raw samples, the count of successes and failures, and the
#   empirical mean $\nu$ next to the true mean $\mu$

# %%
# Demonstrate basic Bernoulli sampling.
utils.cell1_1_basic_bernoulli_sampling()

# %% [markdown]
# ## Cell 1.2: Samples over time and empirical PDF
#
# **Goal**:
# - Visualize $N$ samples from a Bernoulli distribution as a sequence over
#   time and as an empirical probability distribution function (PDF), side
#   by side
#
# **Implementation**: `cell1_2_samples_over_time_and_pdf()`
# - Draws `N` Bernoulli(`mu`) samples with `_generate_bernoulli_samples()`
# - Plots the samples over time, then the same samples as an empirical PDF
#   next to the theoretical Bernoulli PDF, in `_plot_bernoulli_sample2()`

# %%
hintros.print_obj_info(utils.cell1_2_samples_over_time_and_pdf)

# %% [markdown]
# **Usage**
# - Inputs
#   - **`mu`**: true probability of success, 0.1-0.9
#   - **`N`**: number of samples drawn, 10-500
#   - **`seed`**: random seed for the draw
#
# - Panels
#   - **`Bernoulli samples over time`**: each sample plotted by index,
#     colored by outcome, with the true `mu` marked as a dashed line
#   - **`Empirical PDF`**: bar comparison of the empirical outcome
#     frequencies against the theoretical Bernoulli PDF
#   - **`Comments`**: current `mu`, `N`, `seed`, and the counts of
#     successes/failures

# %%
# Display N samples over time and their empirical PDF.
utils.cell1_2_samples_over_time_and_pdf()

# %% [markdown]
# **Guided usage**
# - Raise `N` from 10 toward 500, leaving `mu` fixed
#   - Observe the empirical PDF bars converge toward the theoretical
#     Bernoulli PDF
# - Change `seed` a few times at a small `N`
#   - Observe the empirical PDF swing further from the theoretical one than
#     it does at large `N`

# %% [markdown]
# ## Cell 1.3: Distribution of empirical mean
#
# **Goal**:
# - Show the distribution of the empirical mean $\nu$ over many repeated
#   trials, and compare it with the Central Limit Theorem prediction
#
# **Implementation**: `cell1_3_distribution_empirical_mean()`
# - Repeats sampling `n_samples` times in `_plot_bernoulli_sample4()`, each
#   trial drawing `N` Bernoulli(`mu`) samples and recording its own $\nu$
# - Overlays the empirical histogram of $\nu$ with the
#   $\mathcal{N}\left(\mu, \sqrt{\mu(1-\mu)/N}\right)$ density predicted by
#   the Central Limit Theorem

# %%
hintros.print_obj_info(utils.cell1_3_distribution_empirical_mean)

# %% [markdown]
# **Usage**
# - Inputs
#   - **`mu`**: true probability of success, 0.1-0.9
#   - **`N`**: samples per trial, 10-500
#   - **`log10(n_samples)`**: number of trials, log-scaled from 100 to
#     10000
#   - **`seed`**: random seed for the trials
#
# - Panels
#   - **`Distribution of empirical mean nu`**: histogram of $\nu$ across
#     trials, with the CLT-predicted normal density overlaid and the true
#     `mu` marked
#   - **`Comments`**: current parameters, empirical mean/std of $\nu$, and
#     the CLT-predicted mean/std

# %%
# Display the distribution of empirical mean nu from repeated sampling.
utils.cell1_3_distribution_empirical_mean()

# %% [markdown]
# **Guided usage**
# - Raise `N` from 10 toward 500, leaving `n_samples` fixed
#   - Observe the histogram narrow around `mu`, matching the shrinking
#     $\sqrt{\mu(1-\mu)/N}$ predicted spread
# - Raise `log10(n_samples)` toward its maximum
#   - Observe the histogram fill in and match the CLT curve more closely,
#     since more trials means a smoother empirical distribution
# - By the Law of Large Numbers, $\nu$ converges to $\mu$ as $N$ increases,
#   which the previous experiment already shows as the narrowing histogram

# %% [markdown]
# # Part 2: Hoeffding Inequality: Theoretical Bounds
#
# - The Hoeffding inequality provides a concentration bound
#     - It quantifies how quickly the sample mean converges to the true mean as $N$
#       increases

# %% [markdown]
# ## Cell 2.1: Hoeffding inequality statement
#
# **Goal**:
# - State the Hoeffding inequality precisely, and name every symbol in it,
#   so the interactive cells that follow can be read against a fixed
#   reference
#
# - For $N$ independent Bernoulli random variables $X_1, \ldots, X_N$ with
#   probability $\mu$
# - Let $\nu = \frac{1}{N} \sum_{i=1}^{N} X_i$ be the sample mean
# - The Hoeffding inequality states:
#
# $$P(|\nu - \mu| \geq \epsilon) \leq 2 \exp(-2N\epsilon^2)$$
#
# - Where:
#   - $\nu$ is the sample mean (empirical probability)
#   - $\mu$ is the true probability
#   - $\epsilon > 0$ is the deviation threshold
#   - $N$ is the number of samples
# - The bound decreases exponentially with $N$
# - The bound is independent of $\mu$ (distribution-free)
# - Larger $\epsilon$ requires larger $N$ for the same confidence
# - The factor of 2 accounts for both tails:
#   - $\nu > \mu + \epsilon$
#   - $\nu < \mu - \epsilon$

# %% [markdown]
# ## Cell 2.2: Interactive Hoeffding inequality demonstration
#
# **Goal**:
# - Demonstrate that the Hoeffding inequality is distribution-free: watch it
#   hold across five different bounded distributions on $[0, 1]$, not only
#   the Bernoulli one
#
# **Implementation**: `cell2_2_hoeffding_inequality_demo()`
# - Draws `N` samples from the chosen `Distribution` with
#   `_generate_samples_from_distribution()`, repeats it to build the
#   empirical distribution of $\nu$
# - Computes the Hoeffding bound $2\exp(-2N\epsilon^2)$, capped at 1.0, and
#   the empirical tail probability $P(|\nu - \text{mean}| \geq \epsilon)$,
#   then plots both in `_plot_hoeffding_inequality_demo()`

# %%
hintros.print_obj_info(utils.cell2_2_hoeffding_inequality_demo)

# %% [markdown]
# **Usage**
# - Inputs
#   - **`Distribution`**: Bernoulli, Uniform, Binomial, Truncated Gaussian,
#     or Truncated Exponential
#   - **`mu`**: distribution parameter, interpretation depends on
#     `Distribution`
#   - **`N`** (log scale): number of samples per trial, powers of 2 from 8
#     to 1024
#   - **`epsilon`**: deviation threshold
#   - **`seed`**: random seed
#
# - Panels
#   - **`<Distribution> distribution`**: the PDF/PMF of the selected
#     distribution
#   - **`Distribution of sample mean`**: histogram of $\nu$ across trials,
#     with the tail beyond `epsilon` shaded red
#   - **`Bound vs empirical`**: bar comparison of the Hoeffding bound
#     against the measured tail probability
#   - **`Comments`**: current distribution, parameters, and both
#     probabilities

# %%
# Demonstrate the Hoeffding inequality with multiple distributions.
utils.cell2_2_hoeffding_inequality_demo()

# %% [markdown]
# **Guided usage**
# - Switch `Distribution` through all five options, leaving `mu`, `N`,
#   `epsilon` fixed
#   - Observe the empirical probability stays at or below the Hoeffding
#     bound every time, even though the underlying shape changes
#     completely
# - Raise `N` from small to large
#   - Observe both the bound and the empirical probability shrink, and the
#     histogram narrow around the true mean

# %% [markdown]
# ## Cell 2.3: Empirical probability vs Hoeffding bound
#
# **Goal**:
# - Compare the Hoeffding bound against the empirical tail probability as a
#   single parameter, $N$ or $\epsilon$, sweeps across its range
#
# **Implementation**: `cell2_3_empirical_vs_bound()`
# - Sweeps `Scan variable` (`N` or `epsilon`) over a fixed range with the
#   other one held at `fixed_N`/`fixed_epsilon`, in
#   `_plot_hoeffding_inequality_demo2()`
# - At each sweep point, computes the theoretical bound and estimates the
#   empirical tail probability over `n_trials` resamples

# %%
hintros.print_obj_info(utils.cell2_3_empirical_vs_bound)

# %% [markdown]
# **Usage**
# - Inputs
#   - **`Distribution`**: distribution the samples are drawn from
#   - **`Scan variable`**: sweep `N` (fix epsilon) or sweep `epsilon` (fix
#     N)
#   - **`mu`**: distribution parameter
#   - **`fixed_N`**: `N` used while scanning `epsilon`
#   - **`fixed_epsilon`**: `epsilon` used while scanning `N`
#   - **`seed`**: random seed for the resamples
#   - **`Use log scale for y-axis`**: toggle a log y-axis on the scan plot
#
# - Panels
#   - **`Hoeffding bound vs empirical`**: bound and empirical probability
#     plotted against the scanned variable
#   - **`Comments`**: current scan variable, fixed value, and distribution

# %%
# Visualize how bound and empirical probability change with N or epsilon.
utils.cell2_3_empirical_vs_bound()

# %% [markdown]
# **Guided usage**
# - Scan `N` with `epsilon` fixed
#   - Observe both curves decay exponentially, and the empirical curve stay
#     at or below the bound at every point
# - Switch to scanning `epsilon` with `N` fixed
#   - Observe the same exponential-decay shape, now driven by $\epsilon^2$
#     instead of $N$

# %% [markdown]
# ## Cell 2.4: Hoeffding bound as a function of $N$ and $\epsilon$
#
# **Goal**:
# - Explore the two-parameter shape of the Hoeffding bound
#   $2\exp(-2N\epsilon^2)$ as a function of both $N$ and $\epsilon$ at once
#
# **Implementation**: `cell2_4_bound_surface_heatmap()`
# - Evaluates the bound over a grid of `N` and `epsilon` values up to
#   `N_max`/`epsilon_max`, in `_plot_hoeffding_bound_surface()`
# - Renders the grid as a heatmap, a contour plot, or a 1D slice at a fixed
#   `N` or `epsilon`, depending on `View mode`

# %%
hintros.print_obj_info(utils.cell2_4_bound_surface_heatmap)

# %% [markdown]
# **Usage**
# - Inputs
#   - **`View mode`**: heatmap, fix `N` and vary `epsilon`, fix `epsilon`
#     and vary `N`, or contour plot
#   - **`N_max`**: upper end of the `N` range plotted
#   - **`epsilon_max`**: upper end of the `epsilon` range plotted
#   - **`fixed_N`**: `N` held fixed in `Fix N, vary epsilon` mode
#   - **`fixed_epsilon`**: `epsilon` held fixed in `Fix epsilon, vary N`
#     mode
#
# - Panels
#   - **`Bound surface`**: heatmap, contour, or line view of the bound,
#     depending on `View mode`
#   - **`Comments`**: current view mode and range

# %%
# Explore the Hoeffding bound as a function of N and epsilon.
utils.cell2_4_bound_surface_heatmap()

# %% [markdown]
# **Guided usage**
# - Switch `View mode` from `Heatmap` to `Contour plot`
#   - Observe the same bound surface, now read off as curves of constant
#     probability instead of color
# - Raise `fixed_N` in `Fix N, vary epsilon` mode
#   - Observe the bound-vs-epsilon curve drop faster, since a larger `N`
#     makes the bound more sensitive to `epsilon`

# %% [markdown]
# ## Cell 2.5: 3D surface visualization of Hoeffding bound
#
# **Goal**:
# - View the Hoeffding bound as a 3D surface over $N$ and $\epsilon$, to see
#   the exponential decay in both dimensions at once
#
# **Implementation**: `cell2_5_bound_3d_surface()`
# - Evaluates the bound over the same $(N, \epsilon)$ grid as Cell 2.4, in
#   `_plot_hoeffding_bound_3d()`
# - Renders it as a 3D surface, with the viewing angle set by
#   `elevation`/`azimuth`, optionally with a log-scaled $Z$-axis

# %%
hintros.print_obj_info(utils.cell2_5_bound_3d_surface)

# %% [markdown]
# **Usage**
# - Inputs
#   - **`N_max`**: upper end of the `N` range plotted
#   - **`epsilon_max`**: upper end of the `epsilon` range plotted
#   - **`elevation`**: viewing angle from above, 0 (horizontal) to 90
#     (top-down)
#   - **`azimuth`**: rotation angle around the vertical axis, 0-360
#   - **`Use log scale for Z-axis`**: toggle a log-scaled bound axis
#
# - Panels
#   - **`Bound surface (3D)`**: the bound plotted as a surface over $N$ and
#     $\epsilon$
#   - **`Comments`**: current range and viewing angle

# %%
# Visualize the Hoeffding bound as a 3D surface.
utils.cell2_5_bound_3d_surface()

# %% [markdown]
# **Guided usage**
# - Rotate `azimuth` through a full sweep from 0 to 360
#   - Observe the same "valley" shape from every side: the bound is
#     smallest at large `N` and large `epsilon`
# - Turn on `Use log scale for Z-axis`
#   - Observe the surface's flat-looking tail resolve into visible
#     structure, since the bound spans several orders of magnitude
