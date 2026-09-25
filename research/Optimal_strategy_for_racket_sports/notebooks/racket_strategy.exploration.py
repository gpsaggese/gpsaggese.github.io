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
# # `racket_strategy` Exploration
#
# An interactive exploration of shot placement: for a chosen sport, where
# can the striker (A) safely place the ball, and how likely is the
# returner (B) to miss it (a "winner")?
# - **What this answers**: given a ball speed, a shot dispersion, a
#   returner move speed, and both players' positions, which cells on the
#   court are safe (`P_in`), which are winners (`1 - R`), and which are
#   both (`S`)
# - **What this does not add**: no new scoring math. It is a presentation
#   layer over `racket_scoring.py` (`PR3`), reusing `draw_court()` and
#   `plot_court_heatmap()` from `racket_strategy_utils.py` (`PR5`)
#
# This notebook covers `PR6` of `plan.racket_strategy.md`.

# %% [markdown]
# ## Imports and Setup

# %%
# %load_ext autoreload
# %autoreload 2

import logging

# %%
import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebook
from research.Optimal_strategy_for_racket_sports import racket_params
from research.Optimal_strategy_for_racket_sports import racket_scoring
from research.Optimal_strategy_for_racket_sports import (
    racket_strategy_utils as utils,
)

hdbg.init_logger(verbosity=logging.INFO)
_LOG = logging.getLogger(__name__)
hnotebook.config_notebook()

# %% [markdown]
# # Part 1: Setup

# %% [markdown]
# ## Cell 1.1: Pick a sport
#
# - Switch `SPORT` to `racket_params.PICKLEBALL` to explore the other
#   regime

# %%
SPORT = racket_params.TENNIS
print("sport:", SPORT.name)
print("max_ball_speed_mps:", SPORT.max_ball_speed_mps)

# %% [markdown]
# # Part 2: Controls

# %% [markdown]
# **Goal**:
# - Explore how the shot-placement scoring surface responds to the shot
#   and player parameters the paper leaves as inputs: ball speed, shot
#   dispersion, returner move speed, and both players' positions
#
# **Implementation**: `build_exploration_widget(sport, config)`
# - Builds a `ShotSituation` at the chosen striker position via
#   `racket_scoring.make_rally_situation()`
# - Re-samples an 8x8 target grid via
#   `racket_scoring.estimate_launch_table()` on every "Run" click, the
#   costly Monte Carlo step, not on every slider drag
# - Re-scores against the chosen returner position via
#   `racket_scoring.score_targets()`
#
# **Usage**
# - Inputs
#   - **`max_ball_speed_mps`**: overrides the sport's speed limit
#   - **`error_scale`**: multiplies all three of `DEFAULT_ERROR`'s sigmas
#   - **`move_speed_mps`**: the returner's maximum movement speed
#   - **`striker_x`**, **`striker_y`**: the striker's court position
#   - **`returner_x`**, **`returner_y`**: the returner's court position
#   - **`Run`**: recomputes the Monte Carlo sample and redraws
#
# - Panels
#   - **`P_in heatmap`**: in-bounds probability, independent of the
#     returner's position
#   - **`1 - R heatmap`**: winner probability for the striker, a step
#     function around the returner's reach envelope
#   - **`Score heatmap`**: `S(c) = P_in(c) * (1 - R(c))`, the combined
#     "probability B misses the shot"

# %%
config = racket_scoring.ScoringConfig(n_samples=500, seed=1)
_ = utils.build_exploration_widget(SPORT, config)

# %% [markdown]
# # Part 3: Reading the Output
#
# - Each run draws 3 heatmaps over the target grid, with the striker and
#   returner positions marked by stars
# - `P_in` alone never changes with the returner sliders: it depends only
#   on the striker, the ball speed, and the error scale
# - `1 - R` is a sharp step: 1 outside the returner's reach envelope
#   (`v_p * (T_f - t_r)`), 0 inside it
# - `S` is the product of the two: the cell the striker should actually
#   aim at is the argmax of this last panel

# %% [markdown]
# # Part 4: Worked Scenarios

# %% [markdown]
# ## Scenario 1: A fast ball against a slow mover
#
# - Raise `max_ball_speed_mps` toward its maximum and lower `move_speed_mps`
#   toward 0.5 m/s, leaving positions at their defaults, then click `Run`
# - **Takeaway**: a fast, short-flight-time shot shrinks the returner's
#   reach envelope to almost nothing, so `S` looks nearly identical to
#   `P_in`: the placement decision becomes accuracy-dominated, the same
#   qualitative regime as the paper's pickleball net-exchange example

# %% [markdown]
# ## Scenario 2: A wide error scale
#
# - Raise `error_scale` toward 3.0, leaving everything else at its
#   defaults, then click `Run`
# - **Takeaway**: `P_in` drops everywhere, most sharply near the sidelines
#   and baseline, since a wider shot dispersion pushes more of the landing
#   cloud out of bounds; the argmax cell drifts toward the center of the
#   court

# %% [markdown]
# ## Scenario 3: Returner recovers toward the striker's favorite cell
#
# - Move `returner_x`, `returner_y` toward wherever `Scenario 1`'s argmax
#   cell was, then click `Run`
# - **Takeaway**: the score at that cell drops toward 0, since it is now
#   inside the returner's reach envelope; this is exactly the
#   exploitability the paper's Section V-A argues for, and motivates
#   mixing across cells rather than always aiming at the single argmax

# %% [markdown]
# ## Summary: The Mental Model
#
# - This notebook adds no new scoring math: every heatmap is
#   `racket_scoring.score_targets()`'s output, drawn by
#   `racket_strategy_utils.plot_court_heatmap()`
# - `estimate_launch_table()` is the one control-dependent expensive step,
#   which is why the panel is click-to-run rather than live sliders
# - A single `error_scale` cannot say which of the three underlying sigmas
#   (angle, speed, azimuth) drives a given change; only their combined
#   effect on `P_in` is visible here
