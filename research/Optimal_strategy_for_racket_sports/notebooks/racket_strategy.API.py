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
# # `racket_strategy` API
#
# A guided exploration of the full `racket_strategy` package, from
# `research/Optimal_strategy_for_racket_sports/`, one layer at a time:
# - **Core abstraction**: a shot is a `(theta, v0)` launch; a grid of
#   candidate targets is scored by `P_in(c) * (1 - R(c))`, the probability
#   the shot both lands in and beats the opponent to the ball
# - **Use case**: given a sport, a striker, and a returner position, find
#   the best target on the court to aim at
# - **Learning path**: sport parameters -> trajectory feasibility -> error
#   propagation -> target grid -> in-bounds probability -> reachability and
#   score -> serve
#
# This notebook covers `PR1`-`PR3` and `PR5` of `plan.racket_strategy.md`.
# The zero-sum placement game (`PR4`) is out of scope here.

# %% [markdown]
# ## Imports and Setup

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
import helpers.hdbg as hdbg
import helpers.hintrospection as hintros
import helpers.hnotebook as hnotebook
from research.Optimal_strategy_for_racket_sports import racket_params
from research.Optimal_strategy_for_racket_sports import racket_scoring
from research.Optimal_strategy_for_racket_sports import (
    racket_strategy_utils as utils,
)
from research.Optimal_strategy_for_racket_sports import racket_trajectory

hdbg.init_logger(verbosity=logging.INFO)
_LOG = logging.getLogger(__name__)
hnotebook.config_notebook()

try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# ## Library Overview
#
# - **What problem it solves**: turns the paper's shot-placement scoring
#   rule into tested, reusable functions: given a striker, a returner, and
#   a sport, which target on the court is safest and hardest to reach
# - **Key abstraction**: a candidate target cell `c`, scored by
#   `S(c) = P_in(c) * (1 - R(c))`
# - **Mental model**:
#
# | Object | Description | Comments |
# |--------|-------------|----------|
# | `SportParams`, `PlayerParams` | Static inputs | `TENNIS`, `PICKLEBALL`, `DEFAULT_PLAYER` |
# | `ShotSituation` | Striker, position, legal region | From `make_rally_situation()` / `make_serve_situation()` |
# | `make_target_grid()` | Candidate cells | `cell_id, x_m, y_m` plus bounds |
# | `estimate_launch_table()` | Opponent-independent Monte Carlo | One row per `(cell_id, theta_deg)` |
# | `score_targets()` | Reachability and score, one returner | One row per `cell_id` |
# | `select_best_cell()` | Argmax cell | The recommended target |
#
# - **Key classes/functions**:
#   - `racket_params.SportParams`, `racket_params.PlayerParams`: static
#     inputs, covered in `racket_params.py`
#   - `racket_trajectory.get_feasible_launches()`,
#     `racket_trajectory.simulate_landings()`: closed-form physics and
#     Monte Carlo error propagation, covered in `racket_trajectory.py`
#   - `racket_scoring.estimate_launch_table()`,
#     `racket_scoring.score_targets()`: the grid scoring pipeline, covered
#     in `racket_scoring.py`

# %% [markdown]
# # Part 1: Sport Parameters

# %% [markdown]
# ## Cell 1.1: `SportParams`: Table I as a table

# %%
hintros.print_obj_info(racket_params.SportParams)

# %%
# Build a Table I comparison DataFrame directly from the two presets.
sports_df = pd.DataFrame(
    {
        "Tennis": {
            "length_m": racket_params.TENNIS.court.length_m,
            "width_m": racket_params.TENNIS.court.width_m,
            "net_height_center_m": (
                racket_params.TENNIS.court.net_height_center_m
            ),
            "max_ball_speed_mps": racket_params.TENNIS.max_ball_speed_mps,
        },
        "Pickleball": {
            "length_m": racket_params.PICKLEBALL.court.length_m,
            "width_m": racket_params.PICKLEBALL.court.width_m,
            "net_height_center_m": (
                racket_params.PICKLEBALL.court.net_height_center_m
            ),
            "max_ball_speed_mps": racket_params.PICKLEBALL.max_ball_speed_mps,
        },
    }
)
display(sports_df)
# Outcome: tennis has the larger court and higher net, but pickleball's
# max ball speed is far lower.

# %% [markdown]
# ## Cell 1.2: `draw_court()`: the two courts, side by side

# %%
_, (ax_tennis, ax_pball) = plt.subplots(1, 2, figsize=(10, 6))
utils.draw_court(ax_tennis, racket_params.TENNIS.court)
ax_tennis.set_title("Tennis")
utils.draw_court(ax_pball, racket_params.PICKLEBALL.court)
ax_pball.set_title("Pickleball")
plt.tight_layout()
plt.show()
# Outcome: the pickleball court is visibly smaller, with non-volley zone
# lines near the net that tennis does not have.

# %% [markdown]
# # Part 2: Trajectory Feasibility

# %% [markdown]
# ## Cell 2.1: Reproducing Figure 1
#
# - `theta=8 deg`, `d_target=20 m`, `h0=1 m`: the paper's worked example

# %%
striker = racket_params.DEFAULT_PLAYER
striker_xy = (0.0, -12.0)
target_xy = (0.0, 8.0)
theta_deg = 8.0
d_target = np.sqrt(
    (target_xy[0] - striker_xy[0]) ** 2 + (target_xy[1] - striker_xy[1]) ** 2
)
v0_fig1 = racket_trajectory.solve_launch_speed(
    np.array([d_target]), np.array([np.radians(theta_deg)]), 1.0
)[0]
flight_time_fig1 = racket_trajectory.get_flight_time(
    np.array([v0_fig1]), np.array([np.radians(theta_deg)]), 1.0
)[0]
print("v0 (m/s):", v0_fig1)
print("flight_time (s):", flight_time_fig1)
# Outcome: v0 ~= 22.91 m/s, flight_time ~= 0.88 s, matching the paper.

# %% [markdown]
# ## Cell 2.2: `plot_trajectory_fan()`: nominal plus perturbed trajectories

# %%
hintros.print_obj_info(utils.plot_trajectory_fan)

# %%
_, ax = plt.subplots()
striker_fig1 = racket_params.PlayerParams(
    reaction_time_s=0.2,
    move_speed_mps=1.5,
    contact_height_m=1.0,
    error=racket_params.DEFAULT_ERROR,
)
utils.plot_trajectory_fan(
    racket_params.TENNIS, striker_fig1, striker_xy, target_xy, theta_deg, ax=ax
)
plt.show()
# Outcome: the nominal arc clears the net near its peak; the dashed fan
# shows how +/- angle and speed error spread the landing point.

# %% [markdown]
# # Part 3: Feasible Launch Set

# %% [markdown]
# ## Cell 3.1: `v0`, flight time, and net clearance versus `theta`

# %%
theta_grid_rad = np.radians(np.arange(1.0, 45.0, 1.0))
launches = racket_trajectory.get_feasible_launches(
    striker_xy,
    target_xy,
    1.0,
    racket_params.TENNIS,
    theta_grid_rad=theta_grid_rad,
)
launches_df = pd.DataFrame(
    {
        "theta_deg": np.degrees(launches.theta_rad),
        "v0_mps": launches.v0_mps,
        "flight_time_s": launches.flight_time_s,
        "net_clearance_m": launches.net_clearance_m,
    }
)
_, (ax_v0, ax_tf, ax_clear) = plt.subplots(1, 3, figsize=(15, 4))
ax_v0.plot(launches_df["theta_deg"], launches_df["v0_mps"], marker="o")
ax_v0.set_xlabel("theta (deg)")
ax_v0.set_ylabel("v0 (m/s)")
ax_tf.plot(launches_df["theta_deg"], launches_df["flight_time_s"], marker="o")
ax_tf.set_xlabel("theta (deg)")
ax_tf.set_ylabel("flight time (s)")
ax_clear.plot(
    launches_df["theta_deg"], launches_df["net_clearance_m"], marker="o"
)
ax_clear.set_xlabel("theta (deg)")
ax_clear.set_ylabel("net clearance (m)")
plt.tight_layout()
plt.show()
# Outcome: flatter angles need higher speed but clear the net by less;
# steeper angles need less speed, clear by more, but take longer to land.

# %% [markdown]
# # Part 4: Error Propagation

# %% [markdown]
# ## Cell 4.1: Landing scatter for 3 targets

# %%
rng = np.random.default_rng(0)
errors = racket_trajectory.sample_shot_errors(
    racket_params.DEFAULT_ERROR, 200, rng
)
demo_targets = [(-2.0, 6.0), (0.0, 8.0), (3.0, 5.0)]
_, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, demo_target in zip(axes, demo_targets):
    demo_launches = racket_trajectory.get_feasible_launches(
        striker_xy,
        demo_target,
        1.0,
        racket_params.TENNIS,
        theta_grid_rad=theta_grid_rad,
    )
    demo_landings = racket_trajectory.simulate_landings(
        striker_xy, demo_target, 1.0, racket_params.TENNIS, demo_launches, errors
    )
    ax.scatter(
        demo_landings.x_m.flatten(), demo_landings.y_m.flatten(), alpha=0.3, s=8
    )
    ax.scatter([demo_target[0]], [demo_target[1]], color="red", marker="x", s=80)
    utils.draw_court(ax, racket_params.TENNIS.court)
    ax.set_title(f"Target {demo_target}")
plt.tight_layout()
plt.show()
# Outcome: each cloud of landing points is centered on its target, spread
# by the shared execution error model.

# %% [markdown]
# # Part 5: Grid and In-Bounds Probability

# %% [markdown]
# ## Cell 5.1: `make_target_grid()` and `estimate_launch_table()`

# %%
hintros.print_obj_info(racket_scoring.make_target_grid)

# %%
rally_situation = racket_scoring.make_rally_situation(
    racket_params.TENNIS, racket_params.DEFAULT_PLAYER, striker_xy
)
targets = racket_scoring.make_target_grid(rally_situation.legal_region, 12, 12)
config = racket_scoring.ScoringConfig(n_samples=1000, seed=1)
launch_table = racket_scoring.estimate_launch_table(
    racket_params.TENNIS, rally_situation, targets, config
)
display(launch_table.head())
print("num_rows:", len(launch_table))
# Outcome: one row per feasible (cell, angle) pair, opponent-independent.

# %% [markdown]
# ## Cell 5.2: `plot_court_heatmap()`: best `P_in` per cell

# %%
p_in_by_cell = (
    launch_table.groupby("cell_id")
    .agg(x_m=("x_m", "first"), y_m=("y_m", "first"), p_in=("p_in", "max"))
    .reset_index()
)
_, ax = plt.subplots(figsize=(6, 8))
utils.plot_court_heatmap(p_in_by_cell, "p_in", racket_params.TENNIS, ax=ax)
plt.show()
# Outcome: P_in is highest near the center of the court and drops toward
# the sidelines and baseline, where a small aim error is more likely to
# land the ball out.

# %% [markdown]
# # Part 6: Reachability and Score

# %% [markdown]
# ## Cell 6.1: `score_targets()`: one fixed returner position

# %%
returner = racket_params.DEFAULT_PLAYER
returner_xy = (0.0, 6.0)
scores = racket_scoring.score_targets(launch_table, returner, returner_xy)
best_cell = racket_scoring.select_best_cell(scores)
print("best_cell:\n", best_cell)
# Outcome: the argmax cell is safe (high P_in) and just outside the
# returner's reach envelope, mirroring the paper's Table II intuition.

# %% [markdown]
# ## Cell 6.2: Score and `1 - R` heatmaps

# %%
scores_plot = scores.copy()
scores_plot["winner_prob"] = 1 - scores_plot["reachable"].astype(float)
markers = {"Striker": striker_xy, "Returner": returner_xy}
_, (ax_winner, ax_score) = plt.subplots(1, 2, figsize=(12, 6))
utils.plot_court_heatmap(
    scores_plot,
    "winner_prob",
    racket_params.TENNIS,
    markers=markers,
    ax=ax_winner,
)
utils.plot_court_heatmap(
    scores_plot, "score", racket_params.TENNIS, markers=markers, ax=ax_score
)
plt.tight_layout()
plt.show()
# Outcome: `1 - R` is a sharp step around the returner's reach radius;
# `score` is that step multiplied by the smoother `P_in` surface.

# %% [markdown]
# **Goal**:
# - Explore how the score surface shifts as the returner recovers to a
#   different position, without re-running the Monte Carlo sampling step
#
# **Implementation**: `build_score_widget(sport, launch_table, returner)`
# - Re-runs only `racket_scoring.score_targets()` on every slider move,
#   reusing the cached `launch_table` from Cell 5.1
#
# **Usage**
# - Inputs
#   - **`returner_x`**, **`returner_y`**: the returner's court position
#   - **`t_r`**: the returner's reaction time
#   - **`v_p`**: the returner's maximum movement speed
#
# - Panels
#   - **`P_in heatmap`**: in-bounds probability, unaffected by the sliders
#   - **`Score heatmap`**: `S(c)`, redrawn on every slider move

# %%
_ = utils.build_score_widget(racket_params.TENNIS, launch_table, returner)

# %% [markdown]
# **Guided usage**
# - Drag `returner_y` from near the net toward the baseline
#   - Observe the low-score region (the returner's reach envelope) grow,
#     since a larger `T_f` at that position sums to a longer trip
# - Raise `v_p` toward its maximum
#   - Observe more of the court fall inside the reachable (score-zero)
#     region

# %% [markdown]
# # Part 7: Serve

# %% [markdown]
# ## Cell 7.1: `make_serve_situation()`: the deuce service box

# %%
hintros.print_obj_info(racket_scoring.make_serve_situation)

# %%
serve_situation = racket_scoring.make_serve_situation(
    racket_params.TENNIS, racket_params.SERVE_PLAYER_TENNIS, "deuce"
)
serve_targets = racket_scoring.make_target_grid(
    serve_situation.legal_region, 6, 6
)
serve_config = racket_scoring.ScoringConfig(n_samples=500, seed=2)
serve_launch_table = racket_scoring.estimate_launch_table(
    racket_params.TENNIS, serve_situation, serve_targets, serve_config
)
serve_p_in_by_cell = (
    serve_launch_table.groupby("cell_id")
    .agg(x_m=("x_m", "first"), y_m=("y_m", "first"), p_in=("p_in", "max"))
    .reset_index()
)
_, ax = plt.subplots(figsize=(6, 8))
utils.plot_court_heatmap(serve_p_in_by_cell, "p_in", racket_params.TENNIS, ax=ax)
plt.show()
# Outcome: P_in is computed only over the deuce service box; cells outside
# it never appear in the grid.

# %% [markdown]
# **Interactive exploration**
# - What happens to the serve `P_in` heatmap for `"ad"` instead of
#   `"deuce"`? Is it a mirror image?
# - How does `SERVE_PLAYER_PICKLEBALL`'s lower, underhand contact height
#   change the shape of the feasible launch set?

# %% [markdown]
# ## Summary: The Mental Model
#
# - Every function reads its physical constants from a `SportParams`; no
#   module hardcodes a court dimension or speed limit
# - `estimate_launch_table()` is the one expensive, opponent-independent
#   step; `score_targets()` is cheap and can be re-run for many returner
#   positions against one cached table
# - The composite score `S(c) = P_in(c) * (1 - R(c))` is zero the moment a
#   cell is reachable, regardless of how safe the shot is, which is what
#   drives the argmax away from both the safest and the most extreme cell
