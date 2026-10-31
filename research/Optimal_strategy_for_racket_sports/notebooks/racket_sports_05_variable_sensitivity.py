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
# # Which Variables Matter: Sensitivity of the Winner Probability
#
# - This notebook keeps the Milestone 2 model with all its variables and asks which
#   of them change the probability of a winner, for tennis and pickleball (#694,
#   #789)
# - The pedagogical arc:
#   - One setting at a time, around the defaults and around a fast player 1
#   - The corners of the settings, and the best serve for player 1
#   - The speed ratio $v_p / v_b$ with no reaction time, and what a realistic
#     reaction time changes
#   - A summary of which variables matter and which do not

# %%
# %load_ext autoreload
# %autoreload 2

import logging

# Outcome: edited modules reload on the next run, and `logging` is imported.

# %%
import helpers.hnotebook as hnotebook
from research.Optimal_strategy_for_racket_sports.notebooks import (
    racket_sports_05_variable_sensitivity_utils as utils,
)

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print

# Outcome: the utils are loaded as `utils`, and their logs print here.

# %% [markdown]
# # Part 0: How Much to Run

# %% [markdown]
# ## Cell 0.1: Reduced or Full Runs
#
# - The reduced runs take a few minutes, on one core
#   - Fewer sweep values, one serve side, a 1 m serve grid, and 2000 samples per
#     target
#   - The cells of each map are the same as in the full runs, so a reduced result
#     compares cell by cell with a full one
# - The full runs take about 10 minutes, and give the numbers of the paper
#   - The default grid and 5000 samples per target of Cell 2.1 of notebook 01
#   - Seed 1, so each map repeats exactly from its settings string
# - The numbers quoted in the markdown of this notebook are from the full runs

# %%
# Set to `True` for the full runs used in the paper.
RUN_FULL_SIMS = False
config = utils.get_experiment_config(RUN_FULL_SIMS)
print("RUN_FULL_SIMS=", RUN_FULL_SIMS)
print("config=", config)
# Outcome: the reduced configuration, unless `RUN_FULL_SIMS` is changed.

# %% [markdown]
# # Part 1: One Setting at a Time

# %% [markdown]
# ## Cell 1.1: Sweeps Around the Defaults
#
# **Goal**
# - See how far each setting moves the winner probability when the others stay at
#   their defaults

# %% [markdown]
# **Description**
# - Settings
#   - `ball_speed`: from 5 m/s to the sport's fastest ball
#   - `v_p`: speed of player 1, from 0 to 6 m/s
#   - `t_r`: reaction time of player 1, from 0 to 1 s
#   - `std_x2`, `std_y2`: spread of the return, from 0.1 to 3 m
# - Panels, one row per setting
#   - `highest P(winner)`: the highest cell of the map, with a 2 SE band
#   - `best region`: the fraction of cells within 2 SE of the highest
#   - `best region centroid y`: where the region sits; more negative is deeper
# - Every map of a sweep uses the same seed, so a change between 2 values comes
#   from the setting and not from sampling noise

# %%
# Run the sweeps around the defaults and around a fast player 1.
sweeps = utils.run_all_sweeps(config)
print("sweeps.shape=", sweeps.shape)
# Outcome: one row per map, with its settings string in the `settings` column.

# %%
# Plot the sweeps around the defaults.
_ = utils.plot_sweeps(sweeps, "sweep_defaults")
# Outcome: only `std_x2` and `std_y2` pull the highest `P(winner)` far from 1.

# %% [markdown]
# **Reading the output**
# - Around the defaults the highest $P(winner)$ is close to 1, whatever the speeds
#   - Almost every return that lands in is a winner, so being in is what counts
# - The best region shrinks and moves toward the net as `v_p` rises
# - Saturation: the highest $P(winner)$ is 1.000 for many tennis settings
#   - It cannot rank those settings, so read the mean over the half as well

# %% [markdown]
# ## Cell 1.2: Sweeps Around a Fast Player 1
#
# **Goal**
# - See whether the settings that do not matter at the defaults start to matter
#   when player 1 is fast

# %% [markdown]
# **Description**
# - The same sweeps and panels as Cell 1.1, with `v_p` = 6 m/s, the slider maximum
#   - It is a fast player 1, not a measured value: the paper's references give 0.55
#     to 1.9 m/s around the split step, and no peak speed

# %%
# Plot the sweeps around a fast player 1.
_ = utils.plot_sweeps(sweeps, "sweep_fast_p1")
# Outcome: the ball speed now matters a lot in pickleball, and a little in tennis.

# %% [markdown]
# **Reading the output**
# - A slow ball is what lets a fast player 1 get to the returns
#   - Pickleball at 5 m/s falls to 0.463, tennis to 0.967
# - Reaction time matters more than at the defaults
#   - The distance lost while reacting is $v_p t_r$, so 1.2 m at 6 m/s

# %% [markdown]
# ## Cell 1.3: Maps Along the `v_p` Sweep
#
# **Goal**
# - See where the reach of player 1 removes winners on the court, as `v_p` rises

# %% [markdown]
# **Description**
# - One map per value of `v_p`, cropped to player 1's half, all else at the defaults
#   - The outline is the best region, and the star its middle

# %%
# Plot the maps along the `v_p` sweep, for each sport.
for sport_name in config.sports:
    settings = utils.get_default_settings(sport_name, n_samples=config.n_samples)
    v_ps = [0.0, 1.5, 3.0, 4.0, 6.0]
    _ = utils.plot_montage(
        [dict(settings, v_p=v_p) for v_p in v_ps],
        [f"v_p = {v_p:g}" for v_p in v_ps],
        f"{sport_name}: v_p sweep, other settings at the defaults",
    )
# Outcome: the dip around player 1 grows, and the best region moves toward the net.

# %% [markdown]
# ## Cell 1.4: Maps Along the Ball Speed Sweep, Fast Player 1
#
# **Goal**
# - See how a slower ball hands the court to a fast player 1

# %% [markdown]
# **Description**
# - One map per ball speed, with `v_p` = 6 m/s and all else at the defaults

# %%
# Plot the maps along the ball speed sweep, for each sport.
for sport_name in config.sports:
    settings = utils.get_default_settings(sport_name, n_samples=config.n_samples)
    settings["v_p"] = utils.V_P_MAX_MPS
    values = utils.get_sweep_values(sport_name)["ball_speed"]
    balls = [values[0], values[2], values[4], values[-3], values[-1]]
    _ = utils.plot_montage(
        [dict(settings, ball_speed=ball) for ball in balls],
        [f"ball = {ball:g} m/s" for ball in balls],
        f"{sport_name}: ball speed sweep, v_p = 6 m/s, other settings at the"
        " defaults",
    )
# Outcome: at 5 m/s in pickleball the best target moves up to the net.

# %% [markdown]
# # Part 2: Corners and the Best Serve

# %% [markdown]
# ## Cell 2.1: The Corners of the Settings
#
# **Goal**
# - Check GP's idea on #694: there are regions of the settings where the values do
#   not matter, and the central serve should be the best one

# %% [markdown]
# **Description**
# - Corners: 9 serves x 3 ball speeds x 3 values of `v_p`, per side and per sport
#   - Serves: wide, body, and T, each short, mid, and deep
#   - Ball speed: 5 m/s, the default, and the sport's fastest ball
#   - `v_p`: 0, 1.5, and 6 m/s
# - The score of a serve is the highest $P(winner)$ that player 2 can get after it,
#   so the best serve for player 1 has the lowest score
# - Each cell of the table shows the score and, below it, the mean over the half
#   - A box marks a serve lower than the highest of its row by more than 2 standard
#     errors of the difference; a row with no box is a tie within noise

# %%
# Run the corners.
corners = utils.run_corners(config)
print("corners.shape=", corners.shape)
# Outcome: one row per serve, corner, side, and sport.

# %%
# Plot the serve table of tennis.
_ = utils.plot_serve_table(corners, "Tennis")
# Outcome: the only row with boxes is the slow ball with a fast player 1.

# %%
# Plot the serve table of pickleball.
_ = utils.plot_serve_table(corners, "Pickleball")
# Outcome: the same row has the largest effect, by far.

# %% [markdown]
# **Reading the output**
# - Most corners do not matter: the highest $P(winner)$ is at least 0.99 in 146 of
#   162 tennis corner runs, and in 108 of 162 pickleball ones
# - The exception is a slow ball (5 m/s) with a fast player 1 (6 m/s)
#   - Tennis falls to between 0.816 and 0.991, depending on the serve
#   - Pickleball falls to between 0.047 and 0.705
# - With `v_p` = 6 m/s and the default ball, pickleball still falls to 0.904
# - Depth matters more than width: the short serves leave 0.68 to 0.70 in
#   pickleball at the corner above, the deep ones 0.05 to 0.09

# %% [markdown]
# ## Cell 2.2: The Best Serve per Corner
#
# **Goal**
# - Read the best serve of each corner, and how many serves tie with it

# %% [markdown]
# **Description**
# - `best_serve`: the serve with the lowest score
# - `max_p_winner`: its score
# - `highest_over_serves`: the score of the worst serve of the corner
# - `n_tied`: serves within 2 standard errors of the best one, itself included

# %%
# Find the best serve of each corner, and show the corners where serves differ.
best_serves = utils.find_best_serves(corners)
n_serves = corners["serve_name"].nunique()
display(best_serves[best_serves["n_tied"] < n_serves].round(3))
# Outcome: outside these corners all the serves tie, so no serve is best.

# %% [markdown]
# ## Cell 2.3: The Serve Map
#
# **Goal**
# - Look between the named serves: the score of every legal deuce serve

# %% [markdown]
# **Description**
# - A grid of serve landing points, 0.5 m apart in the full runs, 1 m in the reduced
#   - The reduced runs draw only the corner of a fast player 1 and a slow ball
# - One color scale from 0 to 1 in all the panels, so a flat panel looks flat
# - The triangle marks the lowest score, the best serve for player 1

# %%
# Run the serve maps and plot them.
serve_map = utils.run_serve_map(config)
_ = utils.plot_serve_maps(serve_map)
# Outcome: with a slow ball and a fast player 1, the score falls with depth.

# %% [markdown]
# **Reading the output**
# - With a slow ball and a fast player 1, the lowest score is deep and near the T
#   - Tennis: 0.816 at (-0.3, 6.1)
#   - Pickleball: 0.047 at (-0.3, 6.4)
# - GP's expectation holds where it can be seen, but depth moves the score much more
#   than width
#   - A deep serve pushes player 2 back: the return flies farther, so player 1 has
#     more time to run

# %% [markdown]
# # Part 3: The Speed Ratio and the Reaction Time

# %% [markdown]
# ## Cell 3.1: Scale Both Speeds, With No Reaction Time
#
# **Goal**
# - Check that with $t_r = 0$ the map depends on the two speeds only through the
#   ratio $r = v_p / v_b$ (#789)

# %% [markdown]
# **Description**
# - Player 1 reaches a landing point at distance $dist$, with flight distance $d$, if
#   $$dist \le v_p \max(0, d / v_b - t_r) = \max(0, r d - v_p t_r)$$
# - The landing points do not depend on the speeds, so with $t_r = 0$ only $r$ is
#   left in the rule
# - The check multiplies both speeds by $k$, keeps the seed, and compares every cell
#   of the 2 maps
#   - `max_diff`: the largest difference over `p_winner`, `p_winner_se`, `p_in`, and
#     `p_reached`
#   - `mean_p_reached`: how much of the court player 1 reaches, so the rule is active

# %%
# Scale both speeds on 5 settings per sport, all with `t_r = 0`.
ratio_checks = utils.run_ratio_checks(config)
display(ratio_checks[["sport", "setting", "k", "max_diff", "mean_p_reached"]])
# Outcome: `max_diff` is 0 on every row, with the reach rule active.

# %% [markdown]
# ## Cell 3.2: Random Settings and Factors
#
# **Goal**
# - Try to break the invariance with settings that nobody picked by hand

# %% [markdown]
# **Description**
# - Random sport, serve, spread, speeds, and seed, and a random factor from 0.1 to 10
#   - Small maps: 4 x 5 cells and 2000 samples
# - A factor that is not a power of 2 could flip a sample that sits within one
#   rounding error of the reach boundary, so a difference above `1e-12` is counted

# %%
# Run the check on random settings.
random_checks = utils.run_random_ratio_checks(config)
n_above = int((random_checks["max_diff"] > 1e-12).sum())
print("len(random_checks)=", len(random_checks))
print("n_above=", n_above)
print("random_checks['max_diff'].max()=", random_checks["max_diff"].max())
# Outcome: no setting has a difference above `1e-12`.

# %% [markdown]
# ## Cell 3.3: One Curve per Ratio
#
# **Goal**
# - See the invariance as 3 speed pairs with the same ratio falling on one curve, and
#   see it break when $t_r > 0$

# %% [markdown]
# **Description**
# - `v_p` = 1.5, 3, and 6 m/s, each with $v_b = v_p / r$, for a grid of ratios
# - `t_r` = 0, 0.2, and 0.25 s, with the default serve and spread
# - The table gives the largest spread across the 3 speed pairs of a ratio
#   - `max_cell_diff`: the largest difference of one cell

# %%
# Run the speed grid.
grid, p_winner_maps = utils.run_speed_grid(config)
print("grid.shape=", grid.shape)
# Outcome: one row per map, and the `p_winner` of every cell in `p_winner_maps`.

# %%
# Measure the spread across the speed pairs of each ratio.
display(utils.summarize_ratio_collapse(grid, p_winner_maps).round(4))
# Outcome: all 0 at `t_r = 0`, and not at `t_r > 0`.

# %%
# Plot the speed pairs of each ratio, for tennis.
_ = utils.plot_ratio_collapse(grid, "Tennis")
# Outcome: the top row is one curve, and the bottom row splits at high ratios.

# %%
# Plot the speed pairs of each ratio, for pickleball.
_ = utils.plot_ratio_collapse(grid, "Pickleball")
# Outcome: the same, with a wider split.

# %% [markdown]
# **Reading the output**
# - Proposition: with $t_r = 0$, $P(winner)$ of every target depends on $v_p$ and
#   $v_b$ only through $v_p / v_b$
#   - The sketch of the proof is the rule above, and it is in the paper's appendix
# - With $t_r > 0$ there is a second quantity: the reaction distance $v_p t_r$, the
#   reach player 1 loses while reacting
#   - The full grid has curves up to 12.62 points apart at $t_r = 0.2$ s

# %% [markdown]
# ## Cell 3.4: Reaction Time, None vs Realistic
#
# **Goal**
# - Answer GP's question on #789: does a realistic reaction time change the results?

# %% [markdown]
# **Description**
# - Each map of the speed grid against the same map with $t_r = 0$
#   - Realistic values, from the paper's references: 190 to 250 ms of simple visual
#     reaction time, and `v_p` = 1.5 m/s within the 0.55 to 1.9 m/s around the split
#     step
# - The table keeps the ball speeds of the widget, from 5 m/s to the fastest ball,
#   and gives the largest change for each sport, `v_p`, and `t_r`
#   - `best_points`, `mean_points`: the highest $P(winner)$ and the mean over the
#     half, in percentage points
#   - `region_cells`, `centroid_m`: the best region
#   - `max_cell_diff`: the largest change of one cell

# %%
# Compare with no reaction time, within the ball speeds of the widget.
comparison = utils.compare_reaction_time(grid, p_winner_maps)
display(utils.summarize_reaction_time(comparison).round(3))
# Outcome: small at `v_p` = 1.5 m/s, and much larger at 6 m/s.

# %%
# Plot the reaction times at the realistic player speed.
_ = utils.plot_reaction_time(grid, 1.5)
# Outcome: the 3 curves are almost one, apart from the slowest balls.

# %% [markdown]
# **Reading the output**
# - At $t_r$ = 0.2 s and `v_p` = 1.5 m/s the reaction time does not change where to
#   aim
#   - The best region moves by 0.26 m in tennis and 0.45 m in pickleball, under a
#     cell
#   - The highest $P(winner)$ moves by 0.00 points in tennis and 0.82 in pickleball
#   - The mean over the half rises by up to 2.34 and 3.95 points, for slow balls
# - It matters more for a fast player 1, since the lost distance is $v_p t_r$
#   - At 6 m/s: up to 17.52 points on the highest $P(winner)$ in pickleball
# - A reaction time always helps player 2: it can only shrink the reach of player 1

# %% [markdown]
# # Part 4: Which Variables Matter

# %% [markdown]
# ## Cell 4.1: The Effect of Each Setting
#
# **Goal**
# - Put the sweeps of Part 1 side by side, and rank the settings

# %% [markdown]
# **Description**
# - One row per experiment, sport, and setting
#   - `best_low`, `best_high`, `best_range_points`: the highest $P(winner)$ over the
#     sweep
#   - `mean_low`, `mean_high`, `mean_range_points`: the mean over the half
# - Within an experiment and a sport, the setting that moves the highest $P(winner)$
#   most comes first

# %%
# Rank the settings by how much they move the highest `P(winner)`.
sensitivity = utils.summarize_sensitivity(sweeps)
display(sensitivity.round(3))
# Outcome: around the defaults the two spreads come first, in both sports.

# %% [markdown]
# ## Cell 4.2: Summary
#
# - Measured by the highest $P(winner)$, the target player 2 should aim at:
#   - **The error spread matters most**
#     - With a spread of 3 m the highest $P(winner)$ falls to 0.827 in tennis and
#       0.695 in pickleball for `std_x2`, and to 0.955 and 0.727 for `std_y2`
#   - **Player speed and ball speed matter mainly in one corner**: a slow ball with
#     a fast player 1, where deep serves help player 1
#     - Pickleball is the exception outside that corner: the highest $P(winner)$
#       goes down to 0.904 at `v_p` = 6 m/s with the default ball
#   - **Reaction time changes little at realistic values**, and stays in the model
#     for completeness
#     - With $t_r = 0$ the speeds enter only through $v_p / v_b$
# - The mean $P(winner)$ over the half tells another story, for a target picked at
#   random rather than the best one
#   - It moves from 0.51 to 0.83 with `v_p` in pickleball, so the speeds decide how
#     much a poor aim costs
# - Caveats
#   - Saturation: the highest $P(winner)$ is 1.000 for many tennis settings, so it
#     cannot rank them
#   - The spread does not depend on the ball speed yet (#723), so a faster ball
#     always looks better
#   - `v_p` = 6 m/s and spreads of 3 m are slider ends, not measured values
# - Next: a closed form for these relations, with symbolic regression (#790)

# %% [markdown]
# ## Cell 4.3: Save the Data of the Paper's Figures
#
# - The full runs write 2 small CSV files next to the paper's figures
#   - `figures/make_figures.py` of the paper draws Figures 3 and 4 from them
# - The reduced runs write nothing, so they never change the paper's data

# %%
# Save the data of the paper's figures, for the full runs only.
if RUN_FULL_SIMS:
    paths = utils.save_paper_figure_data(
        sweeps, serve_map, utils.PAPER_FIGURES_DIR
    )
    print("paths=", paths)
# Outcome: nothing in the reduced runs.
