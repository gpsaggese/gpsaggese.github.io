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
# # Exact Inference in Bayesian Networks
#
# - This notebook teaches how to compute exact posteriors $P(X \mid \mathbf{e})$
#   in a discrete Bayesian network
# - Concepts are built on the burglary-alarm network and the query
#   $P(Burglary \mid JohnCalls, MaryCalls)$
# - The pedagogical arc:
#   - Inference by enumeration (brute force)
#   - Variable elimination (caching)
#   - Irrelevant-variable pruning
#   - Complexity and limits

# %%
# %load_ext autoreload
# %autoreload 2

import logging


# %%
import helpers.hintrospection as hintros
import helpers.hnotebook as hnotebook

import L06_01_exact_inference_utils as utils

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
# # Part 1: Setting Up the Inference Problem

# %% [markdown]
# ## Cell 1.1: The burglary alarm network and its CPTs
#
# **Goal**
# - Ground every later inference step in one concrete network: the
#   burglary-alarm story and its five conditional probability tables
# - The five variables are:
#   - Root causes: $Burglary$, $Earthquake$ (blue)
#   - The $Alarm$ (orange)
#   - The calls: $JohnCalls$, $MaryCalls$ (purple)
# - The joint factorizes into five small CPTs:
#   $$P(B,E,A,J,M) = P(B)\,P(E)\,P(A \mid B,E)\,P(J \mid A)\,P(M \mid A)$$
#
# **Implementation** `cell1_1_show_network_and_cpts()`
# - Draws the DAG colored by role, prints the priors $P(B)$/$P(E)$, the
#   $P(Alarm \mid B, E)$ table, and the two call CPTs

# %%
# Display the DAG and its five conditional probability tables.
utils.cell1_1_show_network_and_cpts()

# TODO(ai_gp): The prob for P(Alarm | Burglary, Earthquake) do not match
# the slides in L06.1
# TODO(ai_gp): The colors of the nodes are different than the slides

# %% [markdown]
# - The joint over five variables factorizes into five small CPTs
# - Calls depend on the world only through $Alarm$: they are conditionally
#   independent of $Burglary$ and $Earthquake$ given $Alarm$
# - Storing 5 small CPTs (10 independent numbers) is far cheaper than the full
#   joint ($2^5 - 1 = 31$ numbers)
# - Inference never touches the full joint directly: it works with the factored
#   form, a product of the small CPTs attached to each node

# %% [markdown]
# ## Cell 1.2: Query, evidence, and hidden variables
#
# **Goal**
# - Build intuition for which variables must be summed out, by letting
#   students pick the query and mark any subset of nodes as evidence

# %% [markdown]
# **Description**
# - Inputs
#   - `Query X`: which node is being asked about
#   - `observe <node>`: one checkbox plus True/False value per node,
#     marking it as evidence
#
# - Panels
#   - `Roles for the current query`: the DAG recolored by role: query
#     (blue), evidence (red), hidden (grey)
#   - `Comments`: the query restated in math, and the hidden-term
#     count

# %%
# Assign each node a role and recolor the DAG for the current query.
utils.cell1_2_query_roles_widget()

# %% [markdown]
# **Guided usage**
# - Mark every node except `Query X` as evidence
#   - Observe no node stays hidden, and the term count in Comments drops
#     to $2^0 = 1$: nothing needs to be summed out
# - Uncheck every evidence box, leaving only `Query X` set
#   - Observe every other node turns hidden, and the term count climbs to
#     $2^4$: the posterior we want, $P(X \mid \mathbf{e})$, still has to
#     marginalize all of them away, they cannot simply be dropped

# %% [markdown]
# **Implementation** `cell1_2_query_roles_widget()`
# - Recolors the DAG by role, query/evidence/hidden, for the current
#   choice of `Query X` and observed nodes, via `_read_query()`
# - Restates the query in math and counts the $2^{|\mathbf{Y}|}$ hidden
#   terms in the Comments panel

# %%
hintros.print_obj_info(utils.cell1_2_query_roles_widget)

# %% [markdown]
# # Part 2: Inference by Enumeration

# %% [markdown]
# ## Cell 2.1: From conditional to joint via normalization
#
# **Goal**
# - Derive why a conditional query reduces to summing the joint and
#   rescaling by a normalization constant $\alpha$

# %% [markdown]
# **Description**
# - Inputs
#   - `show normalization`: toggle between the unnormalized joint
#     $P(B, j, m)$ and the normalized posterior $P(B \mid j, m)$
#
# - Panels
#   - `Derivation`: the three equation lines from $P(X \mid e)$ down
#     to the alarm-network sum
#   - `Unnormalized joint`/`Normalized posterior`: bars for $B$=T/F in
#     whichever view is toggled on
#   - `Comments`: the unnormalized values, $\alpha$, and the
#     resulting posterior

# %%
# Toggle between the unnormalized joint and the normalized posterior.
utils.cell2_1_normalization_widget()

# %% [markdown]
# **Guided usage**
# - Toggle `show normalization` off, then on
#   - Observe the bars change scale (they no longer sum to 1) but keep
#     the same relative height: rescaling by $\alpha$ never changes which
#     value is more likely, only the units
# - Read $\alpha$ off the Comments panel
#   - Observe $\alpha = 1 / \sum_x P(x, e)$: exactly the constant that
#     makes the posterior sum to 1, computed without ever touching
#     $P(e)$ directly

# %% [markdown]
# **Implementation** `cell2_1_normalization_widget()`
# - Fixes the canonical query $P(Burglary \mid JohnCalls, MaryCalls)$ and
#   computes both the unnormalized joint and the normalized posterior with
#   `_enumerate_posterior()`
# - `show normalization` switches the middle panel between the two views

# %%
hintros.print_obj_info(utils.cell2_1_normalization_widget)

# %% [markdown]
# ## Cell 2.2: Computing the posterior by enumeration
#
# **Goal**
# - Compute the exact posterior by hand, summing CPT products over the
#   hidden variables, and confirm it against the `pgmpy` engine

# %% [markdown]
# **Description**
# - Inputs
#   - `Query X`: which node is being asked about
#   - `observe <node>`: one checkbox plus True/False value per node,
#     marking it as evidence
#
# - Panels
#   - `Sum for X=T`/`Sum for X=F`: every hidden assignment and its
#     CPT product, for each query value
#   - `Posterior`: the resulting bars, with the `pgmpy` reference
#     overlaid in dashed red
#   - `Comments`: hidden-variable count, row count summed, and
#     whether the two engines agree

# %%
# Compute the posterior by enumeration and validate against pgmpy.
utils.cell2_2_enumeration_widget()

# %% [markdown]
# **Guided usage**
# - Mark one more node as evidence, then another
#   - Observe the row count in Comments halve each time: one fewer
#     hidden variable means one fewer factor of 2 in $2^{|\mathbf{Y}|}$
# - Compare the enumeration bars against the dashed `pgmpy` reference for
#   several different queries
#   - Observe they always coincide: hand enumeration and the library
#     engine compute the same posterior, e.g. the famous result
#     $P(Burglary \mid j,m) \approx 0.284$

# %% [markdown]
# **Implementation** `cell2_2_enumeration_widget()`
# - Enumerates every hidden assignment and its CPT product with
#   `_enumerate_posterior()`, for the current `Query X`/evidence
# - Cross-checks the result against `pgmpy`'s `VariableElimination`
#   reference

# %%
hintros.print_obj_info(utils.cell2_2_enumeration_widget)

# %% [markdown]
# ## Cell 2.3: Visualizing the enumeration tree
#
# **Goal**
# - Expose the enumeration computation as a tree, to see the repeated
#   subexpressions that motivate variable elimination

# %% [markdown]
# **Description**
# - Inputs
#   - `Sum order`: which hidden variable branches first, `Earthquake`
#     or `Alarm`
#   - `highlight repeated work`: color leaves by their shared factor
#
# - Panels
#   - `Enumeration evaluation tree`: the branch tree, leaves labeled
#     by their CPT product
#   - `Comments`: the operation count and why the repeats are wasteful

# %%
# Draw the enumeration tree and highlight the repeated subexpressions.
utils.cell2_3_enumeration_tree_widget()

# %% [markdown]
# **Guided usage**
# - Turn on `highlight repeated work`
#   - Observe leaves under different first-level branches share the same
#     color whenever `Alarm` agrees: the identical $P(j \mid a)P(m \mid a)$
#     factor is recomputed in more than one branch
# - Switch `Sum order` between the two options
#   - Observe the tree's shape changes, but the same repeated-factor
#     pattern persists either way: caching it once, instead of
#     recomputing it per branch, is the one idea behind variable
#     elimination

# %% [markdown]
# **Implementation** `cell2_3_enumeration_tree_widget()`
# - Draws the two-level branch tree over the hidden `Earthquake`/`Alarm`
#   values, for the fixed query $P(Burglary \mid j, m)$
# - Colors each leaf's shared $P(j \mid a)P(m \mid a)$ factor by the value
#   of `Alarm` on that branch, when highlighting is on

# %%
hintros.print_obj_info(utils.cell2_3_enumeration_tree_widget)

# %% [markdown]
# # Part 3: Variable Elimination

# %% [markdown]
# ## Cell 3.1: Factors as the unit of computation
#
# **Goal**
# - Introduce the factor, a table over a subset of variables, as the
#   single data structure variable elimination manipulates

# %% [markdown]
# **Description**
# - Inputs
#   - `Factor`: which CPT to view as a factor, labeled by its scope
#   - `Sum out`: which variable in that factor's scope to marginalize
#     away
#
# - Panels
#   - `Before`: the selected factor's table and scope
#   - `After summing out <var>`: the same factor once that variable
#     is marginalized out
#   - `Comments`: how many rows the table shrank by, and the two
#     operations inference needs

# %%
# Explore factors and the summing-out operation.
utils.cell3_1_factor_operations_widget()

# %% [markdown]
# **Guided usage**
# - Pick a factor with a scope of 3 variables, then sum out one of them
#   - Observe the row count drop from 8 to 4: summing out one variable
#     halves the table, one dimension at a time
# - Switch `Factor` to a CPT with only 1 variable in scope
#   - Observe `Sum out` leaves a `(scalar)` factor: summing out every
#     remaining variable collapses the table to a single number

# %% [markdown]
# **Implementation** `cell3_1_factor_operations_widget()`
# - Renders the chosen CPT as a `Factor` table, then marginalizes out
#   `Sum out` with `pgmpy`'s `factor.marginalize()`

# %%
hintros.print_obj_info(utils.cell3_1_factor_operations_widget)

# %% [markdown]
# ## Cell 3.2: Variable elimination step by step
#
# **Goal**
# - Walk through variable elimination on the alarm query, one elimination
#   at a time, and compare its operation count against enumeration

# %% [markdown]
# **Description**
# - Inputs
#   - `Order`: which hidden variable, `Earthquake` or `Alarm`, is
#     eliminated first
#   - `elimination step`: how many variables have been eliminated so
#     far, 0-2
#
# - Panels
#   - `Active factors (step N)`: the current factor scopes
#   - `New factor`: the factor created at this step, or "no
#     elimination yet" at step 0
#   - `Posterior and op count`: the running posterior next to the
#     enumeration-vs-VE operation comparison
#   - `Comments`: order, step, and both operation counts

# %%
# Step through variable elimination one variable at a time.
utils.cell3_2_variable_elimination_widget()

# %% [markdown]
# **Guided usage**
# - Advance `elimination step` from 0 to 2
#   - Observe the active-factor list shrink by one each step, while the
#     posterior bars never change: the final answer is fixed from the
#     start, only the work to reach it is staged
# - Compare the VE operation total in Comments against the enumeration
#   count
#   - Observe VE finishes with noticeably fewer operations, by reusing
#     each cached factor instead of recomputing it per branch

# %% [markdown]
# **Implementation** `cell3_2_variable_elimination_widget()`
# - Advances `step` through `_variable_elimination_steps()`, which
#   eliminates hidden variables in the chosen `Order` and caches each new
#   factor
# - Tallies multiplications and additions used so far, against the fixed
#   enumeration operation count

# %%
hintros.print_obj_info(utils.cell3_2_variable_elimination_widget)

# %% [markdown]
# ## Cell 3.3: Pruning irrelevant variables
#
# **Goal**
# - Show that a variable which is not an ancestor of the query or evidence
#   can be deleted before any computation, for free

# %% [markdown]
# **Description**
# - Inputs
#   - `Query X`: which node is being asked about, over the extended
#     network
#   - `observe <node>`: one checkbox plus True/False value per node
#   - `prune irrelevant variables`: toggle between the full and
#     pruned network
#
# - Panels
#   - `Relevant (green) vs irrelevant (grey)`: the DAG shaded by
#     relevance to the current query/evidence
#   - `Full vs pruned posterior`: bars for both networks, side by
#     side
#   - `Comments`: the irrelevant nodes and whether the two
#     posteriors match

# %%
# Shade nodes by relevance and compare full vs pruned posteriors.
utils.cell3_3_pruning_widget()

# %% [markdown]
# **Guided usage**
# - Leave `Neighbor` unobserved and toggle `prune irrelevant variables`
#   on and off
#   - Observe the posterior bars stay identical either way, while
#     Comments lists `Neighbor` as irrelevant: an unobserved effect node
#     with no bearing on the query contributes nothing
# - Mark `Neighbor` as evidence
#   - Observe it turns green (relevant) and pruning stops removing it: an
#     observed node is never irrelevant, however far it sits from the
#     query

# %% [markdown]
# **Implementation** `cell3_3_pruning_widget()`
# - Builds an extended network with an extra `Neighbor` leaf, then shades
#   nodes by whether they are the query, evidence, or an ancestor of
#   either
# - `prune irrelevant variables` compares the posterior on the full
#   network against the posterior on the ancestor-only subgraph

# %%
hintros.print_obj_info(utils.cell3_3_pruning_widget)

# %% [markdown]
# # Part 4: Complexity and Limits

# %% [markdown]
# ## Cell 4.1: Complexity: polytrees vs general graphs
#
# **Goal**
# - Make concrete that exact inference is cheap on tree-shaped networks
#   and blows up on densely connected ones

# %% [markdown]
# **Description**
# - Inputs
#   - `n`: number of nodes in both example graphs, 3-10
#   - `Structure`: which curve the marked operating point sits on,
#     `polytree` or `fully connected`
#   - `Order quality`: `good` or `bad`, scaling the cost constant
#
# - Panels
#   - `Polytree (n=N)`: a chain network of the chosen size
#   - `Dense network (n=N)`: a fully connected network of the same
#     size
#   - `Cost vs network size`: the two cost curves on a log y-axis,
#     with the current operating point marked
#   - `Comments`: the operating-point cost and the role of order
#     quality

# %%
# Compare exact-inference cost on polytrees vs dense networks.
utils.cell4_1_complexity_widget()

# %% [markdown]
# **Guided usage**
# - Raise `n` from 3 toward 10 with `Structure` set to `fully connected`
#   - Observe the marked point climb the exponential curve, while the
#     same raise under `polytree` barely moves the point on the linear
#     curve
# - Switch `Order quality` from `good` to `bad`
#   - Observe the operating-point cost jump higher at the same `n`: a
#     worse elimination order inflates the constant in front of both
#     curves

# %% [markdown]
# **Implementation** `cell4_1_complexity_widget()`
# - Draws a chain (`polytree`) and a fully connected graph of the same
#   `n`, then plots $O(n)$ vs $O(2^n)$ operation-count curves with
#   `Order quality` scaling the constant

# %%
hintros.print_obj_info(utils.cell4_1_complexity_widget)

# %% [markdown]
# ## Cell 4.2: When exact inference breaks down
#
# **Goal**
# - Summarize the three regimes where exact inference is the right tool,
#   or is not, and motivate the approximate methods covered next

# %% [markdown]
# **Description**
# - Inputs
#   - `Scenario`: `Small discrete polytree`, `Large dense discrete
#     network`, or `Continuous variables`
#
# - Panels
#   - `Regimes for exact inference`: the three-regime comparison
#     table, with the current scenario's row highlighted
#   - `Continuous case`: a small DAG sketch showing
#     $\sum_y \to \int dy$
#   - `Comments`: the recommendation banner, `EXACT` or
#     `APPROXIMATE`, and why

# %%
# Select a regime and read off the exact-vs-approximate recommendation.
utils.cell4_2_breakdown_widget()

# %% [markdown]
# **Guided usage**
# - Step through all three `Scenario` options
#   - Observe only `Small discrete polytree` recommends `EXACT`; both
#     other scenarios recommend `APPROXIMATE`, for two different reasons:
#     an exponential cost blowup, or a sum that is not even a finite sum
# - Compare this recommendation table against Cell 4.1's cost curves
#   - Observe the `Large dense discrete network` row is exactly the
#     regime where Cell 4.1's exponential curve makes exact inference
#     impractical

# %% [markdown]
# **Implementation** `cell4_2_breakdown_widget()`
# - Looks up the chosen `Scenario` in a fixed regime table and highlights
#   its row and recommendation

# %%
hintros.print_obj_info(utils.cell4_2_breakdown_widget)
