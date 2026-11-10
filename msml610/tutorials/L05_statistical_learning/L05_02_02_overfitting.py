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
# # Overfitting
#
# - This notebook shows overfitting on a noisy sample of a known target
#   function, by comparing models of different complexity on the same data
# - The pedagogical arc:
#   - The true target function, noisy observations, and the train/test split
#   - Model comparison: constant vs linear fit to the same sample

# %%
# %load_ext autoreload
# %autoreload 2

import logging


# %%
import helpers.hintrospection as hintros
import helpers.hnotebook as hnotebook

import L05_02_02_overfitting_utils as utils

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
# # Part 1: Overfitting: Data, Models, and Generalization

# %% [markdown]
# ## Cell 1.1: True target function and data sampling
#
# **Goal**
# - Visualize an unknown target function $f(x)$, sample noisy observations
#   from it, and split them into training and test data, the basic setup
#   every learning problem in this notebook starts from

# %% [markdown]
# **Description**
# - Inputs
#   - `seed`: random seed for the sampled points and the train/test
#     split
#   - `Function`: target function $f$: slow sinusoid, fast sinusoid,
#     parabola, constant, or linear
#   - `epsilon`: standard deviation of the observation noise, 0-1
#   - `N (total samples)` (log scale): total points sampled, 4-1024
#
# - Panels
#   - `True target function`: the noiseless $f(x)$, plus a noisy
#     overlay when `epsilon` > 0
#   - `In-sample data (80%)`: the training points
#   - `Out-of-sample data (20%)`: the test points
#   - `Comments`: current parameters and the train/test split sizes

# %%
# Display the true target function with interactive controls.
utils.cell1_plot_true_target_function()

# %% [markdown]
# **Guided usage**
# - Switch `Function` from `Slow Sinusoid` to `Fast Sinusoid`, leaving
#   everything else fixed
#   - Observe the same `N` points now trace a much busier curve, since the
#     function itself oscillates faster
# - Raise `epsilon` from 0 toward 1
#   - Observe the training and test points scatter further from the true
#     curve, foreshadowing why $E_{in}$ alone will not tell the full story
#     in Cell 1.2

# %% [markdown]
# **Implementation** `cell1_plot_true_target_function()`
# - Samples `N` points from the chosen `Function`, adds Gaussian noise of
#   scale `epsilon`, and splits them 80/20 into training and test sets
# - Stores the split in shared state so Cell 1.2 fits models against the
#   same data

# %%
hintros.print_obj_info(utils.cell1_plot_true_target_function)

# %% [markdown]
# ## Cell 1.2: Model comparison: constant vs linear
#
# **Goal**
# - Fit a constant or linear model to Cell 1.1's training split, and
#   compare in-sample error $E_{in}$ against out-of-sample error $E_{out}$

# %% [markdown]
# **Description**
# - Inputs
#   - `Model Type`: `Constant` ($h(x) = b$) or `Linear`
#     ($h(x) = ax + b$)
#   - `Resample and Relearn`: draws a new training/test split and
#     refits
#
# - Panels
#   - `In-sample data and model`: training points, the fitted $h(x)$,
#     and $E_{in}$
#   - `Out-of-sample data and model`: test points, the same $h(x)$,
#     and $E_{out}$
#   - `True function vs model`: $f(x)$ against $h(x)$, with the
#     approximation error shaded
#   - `Comments`: learned parameters, $E_{in}$, and $E_{out}$

# %%
# Display model learning with interactive controls.
utils.cell2_plot_model()

# %% [markdown]
# **Guided usage**
# - Click `Resample and Relearn` several times on `Constant`
#   - Observe $h(x)$ barely moves between draws: **low variance**, but
#     the shaded error against $f$ stays large: **high bias**
# - Switch to `Linear` and repeat
#   - Observe $h(x)$ shift more between draws (**higher variance**),
#     while the shaded error against $f$ shrinks (**lower bias**): the
#     same bias-variance tradeoff as before, now seen on real train/test
#     splits
# - Run Cell 1.1 again with a new `Function` or `seed` before revisiting
#   this cell: it always fits against whatever split Cell 1.1 last
#   produced

# %% [markdown]
# **Implementation** `cell2_plot_model()`
# - Fits $h(x) = b$ (hypothesis class $\mathcal{H}_0$) or
#   $h(x) = ax + b$ (hypothesis class $\mathcal{H}_1$) to the training
#   data from Cell 1.1's shared state, depending on `Model Type`
# - `Resample and Relearn` draws a fresh training/test split from the same
#   `Function`/`epsilon`/`N` and refits

# %%
hintros.print_obj_info(utils.cell2_plot_model)
