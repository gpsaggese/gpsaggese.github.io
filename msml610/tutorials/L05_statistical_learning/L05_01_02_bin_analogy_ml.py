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
# # Bin Analogy ML
#
# - This notebook uses the bin of red and green marbles, the lecture's own
#   analogy, to connect drawing a sample from a population to learning from
#   data, without a larger running project
# - The pedagogical arc:
#   - The bin as an unknown population, with true fraction $\mu$ of red marbles
#   - A single experiment: is the sample fraction $\nu$ close to $\mu$?
#   - Monte Carlo simulation of the distribution of $\nu$

# %%
# %load_ext autoreload
# %autoreload 2

import logging


# %%
import helpers.hintrospection as hintros
import helpers.hnotebook as hnotebook

import L05_01_02_bin_analogy_ml_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# # Part 1: Bin Analogy for Machine Learning

# %% [markdown]
# ## Cell 1.1: Visual bin: population of marbles
#
# **Goal**
# - Visualize an unknown population of marbles in a bin, and introduce $\mu$
#   as the population's true, unknown proportion of red marbles

# %% [markdown]
# **Description**
# - Inputs
#   - `seed`: random seed for the spatial arrangement of marbles
#   - `mu`: true proportion of red marbles in the population, 0-1
#
# - Panels
#   - `Bin with marbles`: the full population, red and green marbles
#     laid out on a grid, with `mu` and the red count in the title

# %%
utils.cell1_draw_bin_with_marbles_interactive()

# %% [markdown]
# **Guided usage**
# - Change `seed` several times, leaving `mu` fixed
#   - Observe the marbles reshuffle into a new arrangement, but the total
#     red count stays exactly $\mu \times$ total marbles every time
# - Raise `mu` from 0 toward 1
#   - Observe the bin fill up with more red marbles, since $\mu$ is the
#     fixed population proportion, not something observed by sampling
# - In real-world scenarios, $\mu$ is never known directly, only estimated
#   from samples: the rest of this notebook builds that estimation problem

# %% [markdown]
# **Implementation** `cell1_draw_bin_with_marbles_interactive()`
# - Places a grid of marbles, colors a `mu` fraction of them red with
#   `_draw_bin_with_marbles()`, and shuffles the colors with `seed` so the
#   arrangement looks random while the proportion stays fixed

# %%
hintros.print_obj_info(utils.cell1_draw_bin_with_marbles_interactive)

# %% [markdown]
# ## Cell 1.2: Single experiment: is $\nu$ close to $\mu$?
#
# **Goal**
# - Draw one sample of `N` marbles from the bin, compute the sample
#   proportion $\nu$, and see how close a single estimate lands to $\mu$

# %% [markdown]
# **Description**
# - Inputs
#   - `seed`: random seed for the sample drawn
#   - `mu`: true proportion of red marbles in the population, 0-1
#   - `N`: number of marbles sampled, 10-1000
#
# - Panels
#   - `Population vs sample`: bar chart comparing $\mu$ against $\nu$,
#     colored by how close the two are
#   - `Interpretation`: current parameters, the sampled $\nu$, and the
#     error $|\nu - \mu|$

# %%
utils.cell2_plot_single_experiment_interactive()

# %% [markdown]
# **Guided usage**
# - Change `seed` several times, leaving `mu` and `N` fixed
#   - Observe $\nu$ land in a different place each time, and the bar color
#     flip between green, yellow, and red: one experiment is not reliable
#     on its own
# - Raise `N` from 10 toward 1000, leaving `seed` fixed
#   - Observe $\nu$ settle closer to $\mu$ and the bar turn green more
#     often
# - A single experiment only gives a point estimate, never a sense of how
#   reliable that estimate is
#   - What is really needed is $P(|\nu - \mu| > \epsilon)$: the probability
#     that an estimate this far off would happen at all
#   - Repeating the experiment many times, next, is how that probability
#     gets measured

# %% [markdown]
# **Implementation** `cell2_plot_single_experiment_interactive()`
# - Draws `N` Bernoulli(`mu`) samples with `_plot_single_experiment()`, and
#   computes $\nu = \frac{1}{N}\sum_{i=1}^{N} x_i$
# - Colors the comparison bar chart green, yellow, or red by how far $\nu$
#   landed from $\mu$

# %%
hintros.print_obj_info(utils.cell2_plot_single_experiment_interactive)

# %% [markdown]
# ## Cell 1.3: Monte Carlo simulation: distribution of $\nu$
#
# **Goal**
# - Repeat the sampling experiment many times to see the full distribution
#   of $\nu$, and empirically estimate $P(|\nu - \mu| > \epsilon)$

# %% [markdown]
# **Description**
# - Inputs
#   - `seed`: random seed for the experiments
#   - `mu`: true proportion of red marbles in the population, 0-1
#   - `N`: samples per experiment, 10-500
#   - `n_experiments`: number of repeated experiments, 100-10000
#   - `eps`: tolerance $\epsilon$ defining "far from $\mu$"
#
# - Panels
#   - `Distribution of nu`: histogram plus KDE of $\nu$ across
#     experiments, true `mu` marked, and the $|\nu - \mu| > \epsilon$ tails
#     shaded red
#   - `Key insights`: current parameters, $\text{mean}(\nu)$,
#     $\text{std}(\nu)$, and $P(|\nu - \mu| > \epsilon)$

# %%
utils.cell3_monte_carlo_simulation_interactive()

# %% [markdown]
# **Guided usage**
# - Raise `N` from 10 toward 500, leaving `n_experiments` fixed
#   - Observe the histogram narrow around `mu` and
#     $P(|\nu - \mu| > \epsilon)$ drop: this is the Law of Large Numbers,
#     $\nu \xrightarrow{P} \mu$ as $N \to \infty$
# - Raise `n_experiments` toward 10000
#   - Observe the histogram fill in and its shape approach the smooth
#     normal curve predicted by the Central Limit Theorem
# - Replace "marbles in a bin" with "data points", $\mu$ with
#   generalization error, and $\nu$ with training error on `N` samples
#   - The same result then reads: with enough training samples, training
#     error is close to generalization error, the connection this bin
#     analogy sets up for the rest of the course

# %% [markdown]
# **Implementation** `cell3_monte_carlo_simulation_interactive()`
# - Runs `n_experiments` independent trials of `N` Bernoulli(`mu`) samples
#   each in `_plot_monte_carlo_simulation()`, recording $\nu$ per trial
# - Overlays a KDE on the $\nu$ histogram, shades the region where
#   $|\nu - \mu| > \epsilon$, and reports the fraction of trials landing
#   there

# %%
hintros.print_obj_info(utils.cell3_monte_carlo_simulation_interactive)
