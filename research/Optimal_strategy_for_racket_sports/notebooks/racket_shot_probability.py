# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Serve and Return: Probability That the Return Lands In
#
# - This notebook follows one serve and one return on a tennis or pickleball court,
#   and computes the probability $P_{in}$ that the return lands in player 1's half
# - The pedagogical arc:
#   - The two courts and their shared coordinate frame
#   - Player 1 serves to $(x_1, y_1)$, then player 2 returns toward $(x_2, y_2)$
#     with a Gaussian error
#   - The closed-form $P_{in}$ next to a Monte Carlo estimate from the same model

# %%
# %load_ext autoreload
# %autoreload 2

import logging

# Outcome: edited modules reload on the next run, and `logging` is imported.

# %%
import helpers.hnotebook as hnotebook
from research.Optimal_strategy_for_racket_sports import racket_params
from research.Optimal_strategy_for_racket_sports.notebooks import (
    racket_shot_probability_utils as utils,
)

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# Outcome: the utils are loaded as `utils`, and their logs print here.

# %%
# Pick the sport for the widget: `racket_params.TENNIS` or
# `racket_params.PICKLEBALL`.
SPORT = racket_params.TENNIS
print("SPORT=", SPORT.name)
# Outcome: `SPORT= Tennis`, the court the widget below draws.

# %% [markdown]
# # Part 0: The Courts

# %% [markdown]
# ## Cell 0.1: Tennis and Pickleball Side by Side
#
# - Both courts use the same frame, in meters:
#   - Origin at the center of the net
#   - $x$ runs across the court, $y$ along it
#   - Player 1 (the server) is on the $y < 0$ side, player 2 on the $y > 0$ side
# - Lines:
#   - Solid: the singles court boundary, and the net at $y = 0$
#   - Dashed: the service lines and the center service line
#   - Dotted: the kitchen (non-volley zone) lines, pickleball only

# %%
# Draw both courts side by side in the shared frame.
utils.cell0_1_plot_courts()
# Outcome: the pickleball court is smaller, has kitchen lines near the net, and
# its center line starts at the kitchen line.

# %% [markdown]
# # Part 1: Serve and Return

# %% [markdown]
# ## Cell 1.1: Probability That the Return Lands In
#
# **Goal**
# - See how the aim point $(x_2, y_2)$ and the spread $(\sigma_x, \sigma_y)$ of a
#   return set the probability $P_{in}$ that it lands in player 1's half
# - Check the closed form against a Monte Carlo count from the same model

# %% [markdown]
# **Description**
# - Inputs
#   - `x1`, `y1`: where the serve lands on player 2's side; player 2 moves there
#   - `x2`, `y2`: where player 2 aims the return on player 1's side
#   - `std_x2`, `std_y2`: spread $\sigma_x$, $\sigma_y$ of the return, sideways and
#     in depth, in meters
#   - `seed`: random seed of the Monte Carlo landing points
#
# - Panels
#   - `Court`: the serve (blue arrow), the return (orange arrow), the target
#     service box (shaded), the 1 and 2 sigma ellipses, and 1000 simulated
#     landing points
#   - `Comments`: whether the serve is in, $P_{in}$ from the formula, and the
#     Monte Carlo estimate $\hat{P}$ with its standard error

# %%
# Build the serve and return widget for `SPORT`.
_ = utils.cell1_1_plot_shot_widget(SPORT)
# Outcome: sliders on top, then the court and the comments; with the default
# sliders the return lands in with probability about 0.93.

# %% [markdown]
# **Reading the output**
# - Each dot is one simulated return
#   - Green if it lands in player 1's half, red if not
# - The closed form multiplies a sideways part and a depth part:
#   $$P_{in} = P_x \cdot P_y$$
#   $$P_x = \Phi((W/2 - x_2) / \sigma_x) - \Phi((-W/2 - x_2) / \sigma_x)$$
#   $$P_y = \Phi((0 - y_2) / \sigma_y) - \Phi((-L/2 - y_2) / \sigma_y)$$
#   - $\Phi$ is the standard normal CDF, $W$ the court width, $L$ its length
# - $\hat{P}$ is the fraction of green dots
#   - It should be within about 2 standard errors of $P_{in}$
# - The ellipses are 1 and 2 standard deviations away on each axis
#   - In 2D they hold about 39% and 86% of the returns, not 68% and 95% as in 1D
# - The serve is checked against the diagonal service box: `x1 <= 0` is the deuce
#   box

# %% [markdown]
# **Guided usage**
# - Drag `y2` down to the baseline
#   - Observe red dots appear behind the baseline, and $P_{in}$ drop toward 0.5
#     as the aim reaches the line
# - Raise `std_x2` with the aim near a sideline
#   - Observe the ellipses widen and more dots land wide
# - Change `seed`
#   - Observe the dots and $\hat{P}$ move a little, while $P_{in}$ stays the same
# - Move `x1` and `y1`
#   - Observe the arrows move, while $P_{in}$ stays the same

# %% [markdown]
# **Implementation** `cell1_1_plot_shot_widget(sport)`
# - Builds 7 sliders with `htutori.build_widget_control()`
# - On every slider change:
#   - Finds the target service box with `racket_params.get_service_box_region()`,
#     and checks the serve with `CourtRegion.contains()`
#   - Computes $P_{in}$ with `compute_p_in()` on `get_striker_half_region()`
#   - Draws 1000 landing points with `sample_landings()`, and counts the ones in
#   - Draws the court with `draw_court()` plus the center service line, and the
#     comments with `htutori.add_fitted_text_box()`

# %% [markdown]
# ## Model Assumptions and Limits
#
# - Sideways and depth errors are independent
#   - The ellipses stay aligned with the court axes, not with the direction of
#     the shot
# - The net is ignored
#   - A short return only counts as out if it lands on the wrong side ($y > 0$)
#   - A return that would hit the net is not modeled
# - The serve point does not change $P_{in}$
#   - Player 2 moves to $(x_1, y_1)$ instantly
#   - The return's spread does not depend on where it is hit from
# - The serve itself is exact
#   - It has no spread, and is only checked against the service box
# - The target service box follows the sign of `x1`: `x1 <= 0` is the deuce box
#   - So the serve can never land in the wrong box, and the `x1` slider stops at
#     the sidelines, so it can never land wide
#   - The serve check can only fail on depth: past the service line (tennis), or
#     in the kitchen or on its line (pickleball)
