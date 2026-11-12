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
# # The 4x3 grid world with gymnasium
#
# - This notebook mirrors the `L12_01_gridworld_4x3` notebook cell-by-cell
# - Uses a `gymnasium.Env` subclass for the same 4x3 grid world
# - The environment exposes `P[s][a]` (FrozenLake convention) for planning, and
#   `step()` for model-free learning
# - The pedagogical arc:
#   - Build the env
#   - Solve it with full knowledge
#   - Learn without a model, as in the from-scratch version
# - The only difference from the from-scratch version is the API:
#   - States are integer IDs (0-10) instead of `(col, row)` tuples
#   - Actions are integer IDs (0-3) instead of strings
#   - `gymnasium` calls `reset()` and `step()` instead of direct model access

# %%
# %load_ext autoreload
# %autoreload 2

import logging


# %%
import helpers.hintrospection as hintros
import helpers.hnotebook as hnotebook

import L12_02_gridworld_4x3_gymnasium_utils as utils

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
# # Part 1: Building the Grid World Environment (Gymnasium)

# %% [markdown]
# ## Cell 1.1: The 4x3 grid and its states
#
# **Goal**
# - Same layout as the from-scratch version: 4x3 grid with START, +1, -1,
#   WALL
# - This time, states are `gym.spaces.Discrete(11)` and actions are
#   `gym.spaces.Discrete(4)`
# - The grid world is fully observable (the agent sees its state ID) but
#   stochastic (actions do not always succeed)
#
# **Implementation** `utils.cell1_1_show_grid()`

# %%
# Draw the grid and print the gymnasium observation / action spaces.
utils.cell1_1_show_grid()

# %% [markdown]
# - The environment has 11 reachable states (12 cells minus 1 wall)
# - Two states are terminal: reaching either ends the episode
# - State IDs 0-10 map to the same `(col, row)` cells

# %% [markdown]
# ## Cell 1.2: Stochastic action model
#
# **Goal**
# - See the stochastic slip model through the gymnasium `env.P[s][a]`
#   interface
# - Understand how the intended action and perpendicular slips share
#   probability mass

# %% [markdown]
# **Description**
# - Inputs
#   - `action`: which action to inspect
#   - `p_intended`: probability the intended action succeeds
#
# - Panels
#   - `Grid`: highlighted state, with arrow thickness encoding outcome
#     probability
#   - `Comments`: current parameter values and computed outcome
#     probabilities

# %%
# Show how an intended action spreads probability mass through env.P.
utils.cell1_2_stochastic_action()

# %% [markdown]
# **Guided usage**
# - Raise `p_intended` toward `1.0`
#   - Observe the intended-direction arrow thickens while the 2
#     perpendicular arrows thin out: the intended action and
#     perpendicular slips share probability mass

# %% [markdown]
# **Implementation** `utils.cell1_2_stochastic_action()`

# %%
hintros.print_obj_info(utils.cell1_2_stochastic_action)

# %% [markdown]
# ## Cell 1.3: Transition model from env.P
#
# **Goal**
# - See how `env.P[s][a]` exposes the transition model as `(prob, s',
#   reward, terminated)` tuples
# - Understand that planning algorithms read this dict while Q-learning
#   uses `step()`

# %% [markdown]
# **Description**
# - Inputs
#   - `state`: which state to inspect
#   - `action`: which action to inspect
#
# - Panels
#   - `Transition heatmap`: probability distribution over next states
#     for the selected (state, action)
#   - `Transition table`: each outcome with probability, reward, and
#     terminal flag

# %%
# Display the explicit transition row from env.P for a chosen (state, action).
utils.cell1_3_transition_table()

# %% [markdown]
# **Guided usage**
# - Select a state next to a terminal, with the action pointing into it
#   - Observe the outcome row's `terminated` flag is set for the terminal
#     next state, and its reward matches the terminal's value

# %% [markdown]
# **Implementation** `utils.cell1_3_transition_table()`

# %%
hintros.print_obj_info(utils.cell1_3_transition_table)

# %% [markdown]
# ## Cell 1.4: Rewards and episode returns
#
# **Goal**
# - Define the reward structure and connect per-step rewards to
#   discounted return
# - See a sample trajectory rolled out via `env.step()`

# %% [markdown]
# **Description**
# - Inputs
#   - `r_step`: living reward per step
#   - `gamma`: discount factor
#   - `seed`: trajectory seed
#
# - Panels
#   - `Rewards and trajectory`: per-cell living rewards, with a
#     sample path from START
#   - `Comments`: the discounted-return computation and step-by-step
#     rewards

# %%
# Show per-cell rewards and the discounted return of a sample trajectory.
utils.cell1_4_rewards_and_returns()

# %% [markdown]
# **Guided usage**
# - Resample with a different `seed`
#   - Observe `env.step()` produces a different path each time, and the
#     Comments panel's discounted return changes with it

# %% [markdown]
# **Implementation** `utils.cell1_4_rewards_and_returns()`

# %%
hintros.print_obj_info(utils.cell1_4_rewards_and_returns)

# %% [markdown]
# # Part 2: Solving the MDP with Value Iteration

# %% [markdown]
# ## Cell 2.1: The Bellman equation for one state
#
# **Goal**
# - Build intuition for the Bellman update on a single state using
#   integer state and action IDs
# - See how the Bellman update reads `env.P[s][a]` to compute Q-values

# %% [markdown]
# **Description**
# - Inputs
#   - `state`: which state to inspect the action values for
#   - `gamma`: discount factor
#
# - Panels
#   - the Q-value of each action at the selected state

# %%
# Show the value of each action at one state under converged utilities.
utils.cell2_1_bellman_one_state()

# %% [markdown]
# **Guided usage**
# - Compare the Q-values across actions at any state
#   - Observe the utility of a state is the value of its best action: the
#     max is what makes the system nonlinear, requiring iteration

# %% [markdown]
# **Implementation** `utils.cell2_1_bellman_one_state()`

# %%
hintros.print_obj_info(utils.cell2_1_bellman_one_state)

# %% [markdown]
# ## Cell 2.2: Value iteration converging over sweeps
#
# **Goal**
# - Watch value iteration converge on the gymnasium `env.P` model
# - See utility information propagate backward from the terminals

# %% [markdown]
# **Description**
# - Inputs
#   - `iteration`: value iteration sweep to display
#   - `gamma`: discount factor
#   - `r_step`: living reward
#
# - Panels
#   - the grid's utilities at the selected sweep

# %%
# Step through value iteration sweeps and watch utilities converge.
utils.cell2_2_value_iteration()

# %% [markdown]
# **Guided usage**
# - Step `iteration` forward from 0
#   - Observe value propagates backward from the terminals, one ring per
#     sweep, and the change per sweep shrinks geometrically

# %% [markdown]
# **Implementation** `utils.cell2_2_value_iteration()`

# %%
hintros.print_obj_info(utils.cell2_2_value_iteration)

# %% [markdown]
# ## Cell 2.3: Extracting the optimal policy
#
# **Goal**
# - Turn converged utilities into an actionable policy
# - Take the greedy action in every cell

# %% [markdown]
# **Description**
# - Inputs
#   - `r_step`: living reward
#
# - Panels
#   - the grid with an arrow per cell pointing toward the greedy action

# %%
# Show the greedy policy extracted from converged utilities.
utils.cell2_3_extract_policy()

# %% [markdown]
# **Guided usage**
# - Make `r_step` a large negative number
#   - Observe expensive steps push the agent toward the short risky path
# - Make `r_step` close to 0
#   - Observe cheap steps let the agent take the long safe route instead

# %% [markdown]
# **Implementation** `utils.cell2_3_extract_policy()`

# %%
hintros.print_obj_info(utils.cell2_3_extract_policy)

# %% [markdown]
# # Part 3: Solving the MDP with Policy Iteration

# %% [markdown]
# ## Cell 3.1: Policy evaluation for a fixed policy
#
# **Goal**
# - Compute the utility of a fixed policy by solving
#   $(I - \gamma P) U = b$
# - Read transition probabilities from `env.P[s][a]`

# %% [markdown]
# **Description**
# - Inputs
#   - `policy`: which fixed policy to evaluate
#   - `gamma`: discount factor
#
# - Panels
#   - the utility of every state under the selected fixed policy

# %%
# Evaluate a fixed policy by solving the linear Bellman system.
utils.cell3_1_policy_evaluation()

# %% [markdown]
# **Guided usage**
# - Select a deliberately bad `policy`
#   - Observe it yields low utilities, especially near the `-1`
#     terminal: evaluation only answers "how good is this policy"

# %% [markdown]
# **Implementation** `utils.cell3_1_policy_evaluation()`

# %%
hintros.print_obj_info(utils.cell3_1_policy_evaluation)

# %% [markdown]
# ## Cell 3.2: Policy improvement and iteration to optimality
#
# **Goal**
# - Alternate evaluation and improvement until the policy stops changing
# - Watch convergence from a deliberately poor policy to the optimal one

# %% [markdown]
# **Description**
# - Inputs
#   - `iteration`: evaluate/improve round to display
#
# - Panels
#   - the policy arrows at the selected round

# %%
# Step through policy iteration rounds and watch arrows flip.
utils.cell3_2_policy_iteration()

# %% [markdown]
# **Guided usage**
# - Step `iteration` forward from 0
#   - Observe policy iteration typically converges in fewer rounds than
#     value iteration sweeps; each round is more expensive (it solves a
#     linear system), but the total wall-clock can still be lower

# %% [markdown]
# **Implementation** `utils.cell3_2_policy_iteration()`

# %%
hintros.print_obj_info(utils.cell3_2_policy_iteration)

# %% [markdown]
# ## Cell 3.3: Value iteration vs policy iteration
#
# **Goal**
# - Contrast the two exact methods and their convergence behavior
# - Understand the tradeoff between many cheap sweeps and few expensive
#   rounds

# %% [markdown]
# **Description**
# - Inputs
#   - `gamma`: discount factor
#
# - Panels
#   - convergence curves for both methods

# %%
# Compare convergence of the two exact methods.
utils.cell3_3_compare_solvers()

# %% [markdown]
# **Guided usage**
# - Raise `gamma` toward 1
#   - Observe value iteration needs many more sweeps, while policy
#     iteration is relatively unaffected by gamma

# %% [markdown]
# **Implementation** `utils.cell3_3_compare_solvers()`

# %%
hintros.print_obj_info(utils.cell3_3_compare_solvers)

# %% [markdown]
# # Part 4: Learning Without a Model (Q-Learning)

# %% [markdown]
# ## Cell 4.1: Why reinforcement learning is harder than planning
#
# **Goal**
# - Contrast planning (reading `env.P[s][a]`) with learning (calling
#   `env.step()`)
# - Understand why the same optimal policy takes a harder route in RL
# - The agent calls `env.step()` and never reads `env.P[s][a]`
# - It must learn the value of actions purely from the experience tuples
#   it gets back from each step
#
# **Implementation** `utils.cell4_1_planning_vs_learning()`

# %%
# Contrast planning (reads env.P) with learning (calls env.step()).
utils.cell4_1_planning_vs_learning()

# %% [markdown]
# - The transition model is hidden behind the gymnasium API
# - The agent discovers the world by interacting with it

# %% [markdown]
# ## Cell 4.2: The Q-learning update rule
#
# **Goal**
# - Introduce the single update that powers Q-learning
# - Show how one `env.step()` call produces the TD update tuple

# %% [markdown]
# **Description**
# - Inputs
#   - `alpha`: learning rate
#   - `gamma`: discount factor
#
# - Panels
#   - the TD update applied to one `env.step()` tuple

# %%
# Show how a single env.step() tuple nudges a Q-value.
utils.cell4_2_q_update_rule()

# %% [markdown]
# **Guided usage**
# - Set `alpha` near 0 vs near 1
#   - Observe the TD error measures surprise, the gap between old
#     expectation and observed outcome, and no transition probabilities
#     are needed to compute it

# %% [markdown]
# **Implementation** `utils.cell4_2_q_update_rule()`

# %%
hintros.print_obj_info(utils.cell4_2_q_update_rule)

# %% [markdown]
# ## Cell 4.3: Exploration vs exploitation with epsilon-greedy
#
# **Goal**
# - Show why the agent must sometimes act randomly
# - Compare state coverage at low and high exploration rates

# %% [markdown]
# **Description**
# - Inputs
#   - `epsilon`: exploration probability
#   - `log(n_episodes)`: training episodes
#   - `seed`: random seed
#
# - Panels
#   - a visit heatmap of which states the agent has explored

# %%
# Compare state coverage under low vs high epsilon using q_learning().
utils.cell4_3_exploration()

# %% [markdown]
# **Guided usage**
# - Compare `epsilon` low vs high
#   - Observe the visit heatmap reveals exactly which states the agent
#     has explored: a balance between exploration and exploitation is
#     essential

# %% [markdown]
# **Implementation** `utils.cell4_3_exploration()`

# %%
hintros.print_obj_info(utils.cell4_3_exploration)

# %% [markdown]
# ## Cell 4.4: Watching Q-learning learn the optimal policy
#
# **Goal**
# - Run full Q-learning and watch the learned policy emerge
# - Compare it to the policy value iteration found with full knowledge

# %% [markdown]
# **Description**
# - Inputs
#   - `log(n_episodes)`: training episodes
#   - `alpha`: learning rate
#   - `epsilon`: exploration probability
#   - `seed`: random seed
#
# - Panels
#   - the learned policy arrows, and the learning curve of returns over
#     episodes

# %%
# Train Q-learning via env.step() and compare to the planning optimum.
utils.cell4_4_q_learning_converges()

# %% [markdown]
# **Guided usage**
# - Raise `log(n_episodes)` to its max
#   - Observe the learned arrows converge to the value iteration arrows
#     as episodes grow, and returns rise and flatten as the Q-table
#     stabilises: the same optimal behaviour emerges from pure
#     experience, no model required

# %% [markdown]
# **Implementation** `utils.cell4_4_q_learning_converges()`

# %%
hintros.print_obj_info(utils.cell4_4_q_learning_converges)

# %% [markdown]
# # Summary: The Mental Model
#
# - A `gymnasium.Env` subclass is an MDP: its `P[s][a]` encodes the
#   transition model and its `step()` lets the agent interact without
#   reading the model
# - When the model is known (planning), `value_iteration()` and
#   `policy_iteration()` read `env.P` and compute the optimal policy
#   exactly
# - When the model is unknown (learning), `q_learning()` calls
#   `env.step()` and discovers the optimal policy from experience tuples
# - All three methods converge to the same optimal policy on the same
#   environment: the difference is whether you plan with the model or
#   learn without it
# - The gymnasium `GridWorldEnv` produces exactly the same optimal
#   utilities and policies as the from-scratch `GridWorld` class
