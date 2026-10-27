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
#   - The probability that the return is a winner, for every aim point of player
#     1's half, and where to aim to make it highest

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
# Pick the sport for the widgets: `racket_params.TENNIS` or
# `racket_params.PICKLEBALL`.
SPORT = racket_params.TENNIS
print("SPORT=", SPORT.name)
# Outcome: `SPORT= Tennis`, the court the widgets below draw.

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

# %% [markdown]
# # Part 2: Where to Aim the Return for a Winner
#
# - Player 2 now wants a winner: a return that lands in, and that player 1 cannot
#   reach before it lands
# - After the serve, player 1 stands at the center of their baseline, $(0, -L/2)$:
#   they react for $t_r$ seconds, then run in a straight line at the speed $v_p$
# - Every aim point $(x_2, y_2)$ of a grid over player 1's half gets its
#   probability of a winner, $P(winner)$, counted as one event on the same Monte
#   Carlo landing points
# - **Limitation: in this model, hitting as fast as possible always looks best;
#   see [#723](https://github.com/gpsaggese/gpsaggese.github.io/issues/723)**
#   - The spread `std_x2`, `std_y2` does not grow with `ball_speed`, so a faster
#     ball only takes time away from player 1, at no cost in accuracy
#   - Real players miss more when they hit harder, so the best `ball_speed` cannot
#     be read from this notebook

# %% [markdown]
# ## Cell 2.1: Probability That the Return Is a Winner
#
# **Goal**
# - See where player 2 should aim the return to make $P(winner)$ highest, given
#   the serve, the spread of the return, the ball speed, and how fast player 1
#   reacts and runs
# - See when a winner depends on landing in, and when on landing out of player 1's
#   reach

# %% [markdown]
# **Description**
# - Inputs
#   - `side`: deuce or ad, the side player 1 serves from, as in Cell 1.1
#   - `x1`, `y1`: where the serve lands; player 2 returns from there
#   - `std_x2`, `std_y2`: spread $\sigma_x$, $\sigma_y$ of the return, sideways and
#     in depth, in meters
#   - `ball_speed`: speed of the return along the ground, in m/s, from 5 m/s up to
#     the sport's fastest ball
#   - `t_r`: reaction time of player 1, in seconds
#   - `v_p`: running speed of player 1, in m/s; 0 means that player 1 does not move
#   - `seed`: random seed of the Monte Carlo landing points
#
# - Panels
#   - `Court`: $P(winner)$ of every aim point of player 1's half, on a fixed color
#     scale from 0 to 1, with the best region (red outline), the best target (red
#     star), player 1 (blue), and player 2 (orange)
#     - If the serve is a fault: the fault as in Cell 1.1, and no map
#   - `Comments`: whether the serve is in; the best target, its $P(winner)$ with
#     the standard error, and its $P(in)$ and $P(reached)$; the size of the best
#     region; the settings string; and the legend of the court
#   - `Settings line`: under the figure, the settings string again, as text that
#     can be selected and copied

# %%
# Build the winner probability widget for `SPORT`.
_ = utils.cell2_1_plot_winner_widget(SPORT)
# Outcome: the side toggle and the sliders on top, then the map and the comments;
# by default the best target is in the middle of player 1's half.

# %% [markdown]
# **Reading the output**
# - Each cell of the map is one aim point $(x_2, y_2)$, the center of the cell
#   - Its color is $P(winner)$: the fraction of its 5000 landing points that land
#     in and that player 1 does not reach
# - Player 1 reaches a landing point if they can run there before the ball lands:
#   $$d \le v_p \cdot \max(0, T - t_r)$$
#   - $d$ is the distance from player 1 to the landing point
#   - $T$ is the flight time: the distance from $(x_1, y_1)$ to the landing point,
#     divided by `ball_speed`
# - The best region holds every cell within 2 standard errors of the highest
#   $P(winner)$, and the star is the cell of the region closest to its middle
#   - Many cells can tie: if every landing point is a winner, $P(winner) = 1$ and
#     its standard error is 0, so the comments give the count of winners instead,
#     e.g., "5000 of 5000 samples"
# - $P(in)$ and $P(reached)$ come from the same landing points as $P(winner)$, to
#   explain it; they are not multiplied
# - All cells use the same random draws, so two cells differ because of where they
#   aim, not because of sampling noise
# - The settings string lists every control, then the grid size and the samples
#   per target: copy it from the line under the figure into the Gdoc, next to
#   the figure, and set the controls to it to get the same numbers

# %% [markdown]
# **What we see**
# - With the defaults the best return is the middle of player 1's half, not a spot
#   near the lines
#   - Player 1 covers only about 0.6 m in tennis: a return landing next to them
#     flies about 15 m in 0.61 s at 25 m/s, so they run $1.5 \cdot (0.61 - 0.2)$,
#     about 0.6 m (1.1 m in pickleball)
#   - So $P(winner)$ is almost $P(in)$, except in a dip around player 1, and the
#     middle keeps the most returns in

# %% [markdown]
# **Guided usage**
# - Raise `v_p` from 1.5 to 6 m/s
#   - Observe the dark dip around player 1 grow: in tennis the lowest cell falls
#     from about 0.5 to under 0.1, while the best target stays near the middle of
#     the half; in pickleball the best target moves just past the kitchen line
# - Lower `ball_speed` to 5 m/s
#   - Observe player 1 reach far more returns, about 1 in 5 on average instead of
#     under 1 in 20, and $P(winner)$ fall around player 1
# - Raise `ball_speed` to its maximum
#   - Observe no cell go down: in this model hitting as fast as possible always
#     looks best (see #723)
# - Raise `std_x2` and `std_y2` to 2 m
#   - Observe the map fade, and the best $P(winner)$ fall with its $P(in)$: from
#     1.00 to 0.95 in tennis, and from 0.996 to 0.76 in pickleball
# - Change `seed`
#   - Observe each cell move by less than 3 standard errors, and the star move to
#     another cell of the best region
# - Switch `side` to ad, leaving `x1` where it is
#   - Observe the fault of Cell 1.1: the map disappears, and the comments say why

# %% [markdown]
# **Implementation** `cell2_1_plot_winner_widget(sport)`
# - Builds the `side` toggle and 8 sliders with `htutori.build_widget_control()`,
#   with the serve controls of Cell 1.1
# - On every control change:
#   - Rounds the slider values with `_round_slider_value()`, and writes the
#     settings string with `_build_settings_string()`, for the comments and the
#     line under the figure
#   - Checks the serve with `_check_serve()`; for a fault, draws it with
#     `_draw_fault()` and stops there: no map
#   - Otherwise computes the map with `compute_winner_heatmap()`, which calls
#     `compute_winner_probability()` for each cell center of
#     `racket_scoring.make_target_grid()`, with the same seed for every cell
#   - For each target, `compute_winner_probability()` draws the landing points
#     with `sample_landings()`, and checks each one for 2 things:
#     - In: `get_striker_half_region()`
#     - Reached: `compute_flight_time()` and `compute_is_reached()`, with player 1
#       at `get_player1_xy()`
#   - Draws the map with `plot_winner_heatmap()`, which outlines the best region
#     of `find_best_region()` and puts the star at `find_best_target()`
#   - Fills the comments panel with `htutori.add_fitted_text_box()`, and puts the
#     legend below the comments, so it never covers the court

# %% [markdown]
# ## Winner Model Assumptions and Limits
#
# - Player 1 stands at the center of their baseline, $(0, -L/2)$, when the return
#   is hit
#   - Not at their serve stance, and not where a real player would move to
# - $t_r$ and $v_p$ are constant
#   - Player 1 reacts for $t_r$, then runs in a straight line at the speed $v_p$,
#     with no acceleration
# - Player 1 must reach the landing point by the time the ball lands
#   - The bounce is ignored as a 3D effect: in a real rally player 1 can also play
#     the ball after the bounce, which gives them more time
#   - The ball flies in a straight line at the constant `ball_speed`: no arc, no
#     slowdown
# - The spread of the return does not depend on the ball speed
#   - In this model, hitting as fast as possible always looks best; see
#     [#723](https://github.com/gpsaggese/gpsaggese.github.io/issues/723)
# - The net is ignored, as in Part 1
#   - A return that would hit the net is not modeled
# - Sideways and depth errors are independent, as in Part 1
# - The map is a grid of 10 x 15 cells for tennis and 9 x 10 for pickleball, with
#   5000 landing points per cell
