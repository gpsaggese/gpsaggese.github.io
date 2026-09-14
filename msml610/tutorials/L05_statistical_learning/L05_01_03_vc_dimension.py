# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # VC Dimension

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style.
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %%
import helpers.hintrospection as hintros
import helpers.htutorial as ut
import L05_01_03_vc_dimension_utils as utils

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# # Part 1: Dichotomies and the Perceptron Break Point

# %% [markdown]
# ## Cell 1.1: Dichotomy explorer: 2D perceptron with 3 points
#
# **Goal**:
# - Explore how a 2D perceptron (a line through the plane) can label 3
#   points in different ways, and see how many of the $2^3 = 8$ possible
#   dichotomies are actually reachable
#
# **Implementation**: `cell1_dichotomy_explorer_3points()`
# - Classifies each of the 3 points by which side of the line it falls on,
#   in `_draw_dichotomy_3points()`, for the line set by `angle` and
#   `offset`
# - Redraws the points and line whenever `Point Config`, `angle`, or
#   `offset` changes

# %%
hintros.print_obj_info(utils.cell1_dichotomy_explorer_3points)

# %% [markdown]
# **Usage**
# - Inputs
#   - **`Point Config`**: `collinear1`/`collinear2` (points on a line) or
#     `triangle1`/`triangle2`/`triangle3` (points in a triangle)
#   - **`angle`**: angle of the line's normal, 0-360 degrees
#   - **`offset`**: distance of the line from the origin, -1.5 to 1.5
#
# - Panels
#   - **`2D perceptron, 3 points`**: the 3 labeled points and the current
#     separating line
#   - **`Comments`**: current `angle`, `offset`, and each point's
#     classification

# %%
# Explore how a 2D perceptron can classify 3 points in different ways.
utils.cell1_dichotomy_explorer_3points()

# %% [markdown]
# **Guided usage**
# - Sweep `angle` through a full turn at a fixed `offset`, on `triangle1`
#   - Observe every rotation flips which points fall on which side, and by
#     combining `angle` with `offset` all 8 dichotomies can be reached
# - Switch `Point Config` to `collinear1`
#   - Observe some labelings become harder or impossible to reach with a
#     single straight line, since collinear points constrain which
#     sub-splits a line can produce

# %% [markdown]
# ## Cell 1.2: Dichotomy explorer: 2D perceptron with 3 points, target assignment
#
# **Goal**:
# - Fix a target labeling of the same 3 points, and either search by hand
#   or let `Find Solution` compute a line that realizes it
#
# **Implementation**: `cell2_dichotomy_explorer_3points_target()`
# - Colors the 3 points by the chosen `Target` assignment, in
#   `_draw_dichotomy_3points_with_target()`
# - `Find Solution` calls `_find_solution()` to solve for an
#   `angle`/`offset` pair that reproduces the target exactly, and writes it
#   back to the sliders

# %%
hintros.print_obj_info(utils.cell2_dichotomy_explorer_3points_target)

# %% [markdown]
# **Usage**
# - Inputs
#   - **`Point Config`**: same 5 point layouts as Cell 1.1
#   - **`Target`**: one of the 8 possible +1/-1 assignments to A, B, C
#   - **`angle`**: angle of the line's normal, 0-360 degrees
#   - **`offset`**: distance of the line from the origin, -1.5 to 1.5
#   - **`Find Solution`**: sets `angle`/`offset` to a line matching
#     `Target`
#
# - Panels
#   - **`2D perceptron, 3 points`**: the 3 points colored by `Target`, the
#     current line, and whether it currently matches
#   - **`Comments`**: current `Target`, `angle`, `offset`, and the match
#     status

# %%
# Discover that 3 points can be classified in 2^3 = 8 different ways.
utils.cell2_dichotomy_explorer_3points_target()

# %% [markdown]
# **Guided usage**
# - Pick each of the 8 `Target` values in turn on `triangle1`, then click
#   `Find Solution`
#   - Observe a matching line is found for every one of them: all 8
#     dichotomies are realizable in general position
# - Switch to `collinear2` and repeat
#   - Observe `Find Solution` still succeeds, but the matching lines
#     cluster into fewer distinct angles, since collinear points give a
#     line less to work with

# %% [markdown]
# ## Cell 1.3: Dichotomy explorer: 2D perceptron with 4 points
#
# **Goal**:
# - Show the limitation that motivates the VC dimension: with 4 points, a
#   2D perceptron cannot reach all $2^4 = 16$ possible labelings
#
# **Implementation**: `cell3_dichotomy_explorer_4points()`
# - Classifies the 4 points by the current line in
#   `_draw_dichotomy_4points()`
# - Flags the dichotomies that `_get_impossible_dichotomies_4points()`
#   marks as unreachable for the current `Point Config`

# %%
hintros.print_obj_info(utils.cell3_dichotomy_explorer_4points)

# %% [markdown]
# **Usage**
# - Inputs
#   - **`Point Config`**: `square`, `circle`, `line`, or `diamond`
#   - **`angle`**: angle of the line's normal, 0-360 degrees
#   - **`offset`**: distance of the line from the origin, -1.5 to 1.5
#
# - Panels
#   - **`2D perceptron, 4 points`**: the 4 labeled points and the current
#     line, with the dichotomy's reachability noted
#   - **`Comments`**: current parameters and the labeling produced

# %%
# Explore how 4 points reveal the break point for 2D perceptrons.
utils.cell3_dichotomy_explorer_4points()

# %% [markdown]
# **Guided usage**
# - On `square`, try to make opposite corners share a color and the other
#   pair share the other color (the XOR pattern)
#   - Observe no `angle`/`offset` combination ever produces it: this is
#     the break point, $k = 4$, for the 2D perceptron
# - Sweep `angle` and `offset` broadly on `square`
#   - Observe only 14 of the 16 possible labelings ever appear, matching
#     $m_H(4) = 14 < 2^4$

# %% [markdown]
# # Part 2: Growth Functions Across Hypothesis Classes

# %% [markdown]
# ## Cell 2.1: Dichotomy explorer: positive rays
#
# **Goal**:
# - Explore the simplest hypothesis class, a single threshold on a line,
#   and see why its growth function is only linear in $N$
#
# **Implementation**: `cell4_dichotomy_explorer_positive_rays()`
# - Places `N` points evenly on a line and labels each +1 if it is at or
#   past `threshold`, else -1, in `_draw_positive_rays()`

# %%
hintros.print_obj_info(utils.cell4_dichotomy_explorer_positive_rays)

# %% [markdown]
# **Usage**
# - Inputs
#   - **`N`**: number of points on the line, 1-10
#   - **`threshold`**: position of the threshold, -1.5 to 1.5
#
# - Panels
#   - **`Positive rays`**: the `N` points on a line, colored by the
#     current labeling, with the threshold marked
#   - **`Comments`**: current `N`, `threshold`, $m_H(N)$, and $2^N$ side
#     by side

# %%
# Explore positive rays with linear growth function.
utils.cell4_dichotomy_explorer_positive_rays()

# %% [markdown]
# **Guided usage**
# - Sweep `threshold` from left to right of all `N` points
#   - Observe exactly $N + 1$ distinct labelings appear: before the first
#     point, between each consecutive pair, and after the last
# - Raise `N` while re-sweeping `threshold`
#   - Observe $m_H(N) = N + 1$ grows linearly, while $2^N$ next to it grows
#     far faster

# %% [markdown]
# ## Cell 2.2: Dichotomy explorer: positive intervals
#
# **Goal**:
# - Explore a slightly richer hypothesis class, an interval on a line, and
#   see why its growth function is quadratic instead of linear
#
# **Implementation**: `cell5_dichotomy_explorer_positive_intervals()`
# - Places `N` points evenly on a line and labels each +1 if it falls
#   between `left` and `right`, else -1, in `_draw_positive_intervals()`

# %%
hintros.print_obj_info(utils.cell5_dichotomy_explorer_positive_intervals)

# %% [markdown]
# **Usage**
# - Inputs
#   - **`N`**: number of points on the line, 1-8
#   - **`left`**: left boundary of the interval, -1.5 to 1.5
#   - **`right`**: right boundary of the interval, -1.5 to 1.5
#
# - Panels
#   - **`Positive intervals`**: the `N` points on a line, colored by the
#     current labeling, with the interval `[left, right]` shaded
#   - **`Comments`**: current `N`, `left`, `right`, $m_H(N)$, and $2^N$

# %%
# Explore positive intervals with quadratic growth function.
utils.cell5_dichotomy_explorer_positive_intervals()

# %% [markdown]
# **Guided usage**
# - Slide `left` and `right` together across all `N` points
#   - Observe every contiguous run of points can be made +1, but a
#     labeling with two separate +1 runs never appears
# - Raise `N` while re-sweeping the interval
#   - Observe $m_H(N) \approx N^2/2 + N + 1$ grows faster than the
#     positive rays' linear count, but still far slower than $2^N$

# %% [markdown]
# ## Cell 2.3: Dichotomy explorer: convex sets
#
# **Goal**:
# - Explore a hypothesis class with no break point at all: convex sets over
#   points on a circle can realize every possible labeling
#
# **Implementation**: `cell6_dichotomy_explorer_convex_sets()`
# - Places `N` points on a circle, and `seed` picks a random target
#   labeling via `_get_target_classification_convex()`
# - `Find Solution` calls `_find_convex_hull_for_target()` to select the
#   points whose convex hull reproduces that target, drawn in
#   `_draw_convex_sets()`

# %%
hintros.print_obj_info(utils.cell6_dichotomy_explorer_convex_sets)

# %% [markdown]
# **Usage**
# - Inputs
#   - **`N`**: number of points on the circle, 3-8
#   - **`seed`**: random seed for the target labeling
#   - **`Find Solution`**: selects the point subset whose convex hull
#     matches the current target
#
# - Panels
#   - **`Convex sets, target dichotomy`**: the `N` points, fill color for
#     the target labeling and border color for the hull's current
#     labeling, titled `MATCH!` once they agree
#   - **`Comments`**: current `N`, target labeling, current labeling, and
#     match status

# %%
# Explore convex sets with exponential growth function.
utils.cell6_dichotomy_explorer_convex_sets()

# %% [markdown]
# **Guided usage**
# - Change `seed` a few times at a fixed `N`, clicking `Find Solution` each
#   time
#   - Observe a matching convex hull is found for every random target: no
#     labeling is ever out of reach
# - Raise `N` toward 8 and repeat
#   - Observe $m_H(N) = 2^N$ still holds at every size, since there is no
#     break point to cap it, unlike every hypothesis class explored
#     earlier in this notebook
