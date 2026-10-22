# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
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
#   - `side`: deuce or ad, the side player 1 serves from; it sets where they stand
#     and which service box is legal
#   - `x1`, `y1`: where the serve lands on player 2's side; player 2 moves there
#   - `x2`, `y2`: where player 2 aims the return on player 1's side
#   - `std_x2`, `std_y2`: spread $\sigma_x$, $\sigma_y$ of the return, sideways and
#     in depth, in meters
#   - `seed`: random seed of the Monte Carlo landing points
#
# - Panels
#   - `Court`: the serve (blue arrow) and the legal service box (shaded)
#     - If the serve is in: the return (orange arrow), the 1 and 2 sigma
#       ellipses, and 1000 simulated landing points
#     - If it is a fault: a red cross where it lands and a "Fault" label
#   - `Comments`: whether the serve is in, and if not why; if it is in, $P_{in}$
#     from the formula and the Monte Carlo estimate $\hat{P}$ with its standard
#     error; and the legend of the court

# %%
# Build the serve and return widget for `SPORT`.
_ = utils.cell1_1_plot_shot_widget(SPORT)
# Outcome: the side toggle and the sliders on top, then the court and the
# comments; by default the deuce serve is in, and the return lands in with
# probability about 0.93.

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
# - The serve is in only if it lands in the shaded box, diagonal from the server
#   - A deuce server stands right of the center mark ($x > 0$) and serves into the
#     box at $x \le 0$; ad is the mirror image
#   - Otherwise the court and the comments say "Fault" and why (wide, wrong box,
#     long, kitchen), and there is no return to compute

# %% [markdown]
# **Guided usage**
# - Drag `y2` down to the baseline
#   - Observe red dots appear behind the baseline, and $P_{in}$ drop toward 0.5
#     as the aim reaches the line
# - Raise `std_x2` with the aim near a sideline
#   - Observe the ellipses widen and more dots land wide
# - Change `seed`
#   - Observe the dots and $\hat{P}$ move a little, while $P_{in}$ stays the same
# - Move `x1` and `y1` inside the shaded box
#   - Observe the arrows move, while $P_{in}$ stays the same
# - Switch `side` to ad, leaving `x1` where it is
#   - Observe the server move left of the center mark and the shaded box move
#     right: the serve is now a fault into the wrong box, and the return disappears

# %% [markdown]
# **Implementation** `cell1_1_plot_shot_widget(sport)`
# - Builds a `side` toggle, and 7 sliders with `htutori.build_widget_control()`
# - On every control change:
#   - Rounds the slider values to 6 decimals with `_round_slider_value()`, so
#     the `+` / `-` buttons cannot drift past a line
#   - Places the server with `_get_server_xy()`, and checks the serve with
#     `_check_serve()` for the chosen side: only the diagonal box is legal, a
#     serve on the kitchen line is out, and the box stops at the baseline
#   - If the serve is a fault, draws it with `_draw_fault()` and stops there: no
#     return, no dots, no $P_{in}$
#   - Otherwise computes $P_{in}$ with `compute_p_in()` on
#     `get_striker_half_region()`
#   - Draws 1000 landing points with `sample_landings()`, and counts the ones in
#   - Draws the court with `draw_court()` plus the center service line
#   - Fills the comments panel with `htutori.add_fitted_text_box()`, and puts the
#     legend below the comments, so it never covers the court

# %% [markdown]
# ## Model Assumptions and Limits
#
# - Sideways and depth errors are independent
#   - The ellipses stay aligned with the court axes, not with the direction of
#     the shot
# - The net is ignored
#   - A short return only counts as out if it lands on the wrong side ($y > 0$)
#   - A return that would hit the net is not modeled
# - Where an in serve lands does not change $P_{in}$
#   - Player 2 moves to $(x_1, y_1)$ instantly
#   - The return's spread does not depend on where it is hit from
# - The serve itself is exact
#   - It has no spread, and is only checked against the service box
# - The serve follows the service rules
#   - The side (deuce or ad) sets the server's stance and the legal box: only the
#     box diagonal from the server is in
#   - Every line is in, except the pickleball kitchen line; the box stops at the
#     baseline
#   - The `x1` and `y1` sliders cover player 2's half, so a serve can be long, in
#     the wrong box, or in the kitchen; a wide serve (past the sideline) cannot be
#     set with the sliders
# - The server's stance, 0.5 m beside the center mark and 0.3 m behind the
#   baseline, is a typical one, not a rule value
