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
# # Monte Carlo Tree Search (MCTS) Engine API
#
# A guided tour of the game-agnostic MCTS engine in `mcts_utils.py`, using
# the phases and notation of MCTS (Selection, Expansion, Simulation,
# Backpropagation; $N(s)$, $N(s, a)$, $Q(s, a)$):
# - `MCTSNode`: one node of the search tree (visit count, value, children)
# - `run_mcts()`: selection -> expansion -> rollout -> backpropagation, repeated
# - `play_game()` / `evaluate_win_rate()`: compose player functions into games
#   and games into statistics
#
# The `Game` interface itself (`game.py`) and the two games this notebook
# searches, `TicTacToe` and `ConnectFour` (`game_examples.py`), are covered
# in full in `game.01.API.ipynb`; see that notebook first for the API these
# demos build on.
#
# Parts 2-5 use `TicTacToe`, kept small so the tree diagrams stay readable.
#
# Part 6 swaps in `ConnectFour` to show that the engine itself never
# changes: same `Game` interface, same `run_mcts()`, only a bigger board and
# a gravity rule. Since a 6x7 board is too big to read as text, Part 6 adds
# board pictures (`plot_connect_four_board()`) everywhere `TicTacToe` used
# plain text.
#
# See `README.md` for a description of every file in this directory.

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebook

_LOG = logging.getLogger(__name__)

hdbg.init_logger(verbosity=logging.INFO)
hnotebook.config_notebook()

# %%
# The base image runs `apt-get update && ... && rm -rf /var/lib/apt/lists/*`
# at build time, which deletes the package-list cache; without a fresh
# `apt-get update` here, `apt-get install` below fails to find `graphviz`
# (silently, from the notebook's point of view) and only the Python bindings
# get installed, leaving the `dot` binary missing.
# !apt-get update --quiet
# !apt-get install --quiet --yes --no-install-recommends graphviz
# !/bin/bash -c "(source /venv/bin/activate; pip install --quiet graphviz)"

# %%
import helpers.hintrospection as hintros
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.game_API_utils as rimtsaazgau
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.game_examples as rimtsaazge
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.mcts_API_utils as rimtsaazmau
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.mcts_utils as rimtsaazmu

# %% [markdown]
# # Part 1: Library Overview
#
# ## What problem does `mcts_utils.py` solve?
#
# - Pick a good move in a two-player game without a hand-written evaluation
#   function, by running many random-rollout simulations from each candidate
#   move and leveraging the law of large numbers
# - Balance trying new moves (exploration) against refining promising ones
#   (exploitation) via the UCT formula
# - Stay game-agnostic: the search code only calls the `Game` methods (see
#   `game.01.API.ipynb`), so a new game plugs in without touching
#   `mcts_utils.py`
#
# ## Mental model
#
# | Object | Description | Type / Comments |
# |--------|-------------|------------------|
# | `Game` | Game-rules interface | see `game.01.API.ipynb` |
# | `MCTSNode` | One tree node | tracks `state`, `children`, `visit_count`, `value_sum` |
# | `build_mcts_tree(game, state)` | Search primitive | returns the built root `MCTSNode` for inspection |
# | `run_mcts(game, state)` | Search entry point | returns the most-visited `Move` at the root |
# | `random_player(game, state)` | Uniform-random player | matches the player-function signature |
# | `make_mcts_player(...)` | MCTS player factory | returns a player function backed by `run_mcts()` |
# | `play_game(game, p1, p2)` | Play one game | alternates two player functions to a terminal state |
# | `evaluate_win_rate(...)` | Play many games | returns win/draw/loss rates |
# | `plot_win_rate_results(...)` | Visualize `evaluate_win_rate()` | bar chart |
#
# ## Cell 1.1: Construct the game to search
#
# See `game.01.API.ipynb` Part 2 for the full `Game` API tour; here we only
# need a `TicTacToe` instance and its initial state to search from below.

# %%
game = rimtsaazge.TicTacToe()
state = game.get_initial_state()
print(f"Type: {type(game)}")
print(game.render(state))

# %% [markdown]
# # Part 2: `MCTSNode`
#
# **Mental model**:
# - each `MCTSNode` wraps one `State` plus the running statistics (`visit_count`,
#   `value_sum`) the UCT formula needs
# - `run_mcts()` grows a tree of these nodes rooted at the state being searched.
#
# ## Cell 2.1: Construct a node directly

# %%
hintros.print_obj_info(rimtsaazmu.MCTSNode)

# %%
root = rimtsaazmu.MCTSNode(
    game.get_initial_state(), untried_moves=game.get_legal_moves(state)
)
print(f"Type: {type(root)}")
print("visit_count:", root.visit_count)
print("is_fully_expanded:", root.is_fully_expanded)
print("mean_value (no visits yet):", root.mean_value)

# %% [markdown]
# ## Cell 2.2: A node's stats update as simulations backpropagate through it
#
# - `run_mcts()` mutates a tree of nodes like `root` in place
# - a node visited once with a win backpropagated has `visit_count == 1` and `mean_value == 1.0`.

# %%
root.visit_count += 1
root.value_sum += 1.0
print("visit_count:", root.visit_count)
print("mean_value:", root.mean_value)

# %%
print(root)

# %% [markdown]
# # Part 3: `run_mcts()`
#
# **Mental model**:
# - `run_mcts()` runs `num_simulations` rounds of
#   select/expand/rollout/backpropagate from `state`
# - then returns the root child with the highest visit count.
#
# ## Cell 3.1: Find an immediate winning move
#
# X already has two in a row; playing the third cell wins immediately.

# %%
demo_state = (1, 1, 0, -1, -1, 0, 0, 0, 0)
print(game.render(demo_state))

# %%
move = rimtsaazmu.run_mcts(game, demo_state, num_simulations=200)
print("MCTS move:", move)

# %%
# TODO(ai_gp): Print state

# %% [markdown]
# ## Cell 3.2: Visualizing the search tree
#
# **Goal**:
# - See the actual search tree `run_mcts()` builds from `demo_state`: the
#   root and its immediate children, annotated with visit count `N` and mean
#   value `Q`
# - Control the two knobs from the UCT formula
#   ($Q/N + C \sqrt{\ln N(s) / N(s, a)}$) and watch how the visit counts
#   respond:
#   - `num_simulations`: the search budget
#   - `C`: the exploration constant (theory suggests $C = \sqrt{2}$ for
#     rewards in $[0, 1]$)
# - Increasing `num_simulations` is literally "how MCTS nodes are updated":
#   each simulation runs one more selection/expansion/rollout/backpropagation
#   pass and adds to the `visit_count` / `value_sum` of every node it touches

# %%
rimtsaazmau.cell3_2_build_tree_widget(game, demo_state)

# %% [markdown]
# ## Cell 3.3: The exploration/simulation-count knobs

# %%
print("EXPLORATION_CONSTANT (the `C` in UCT):", rimtsaazmu.EXPLORATION_CONSTANT)
print("DEFAULT_NUM_SIMULATIONS:", rimtsaazmu.DEFAULT_NUM_SIMULATIONS)

# %%
hintros.print_obj_info(rimtsaazmu.run_mcts)

# %% [markdown]
# # Part 4: Composing players and games
#
# **Mental model**: a "player" is any `(game, state) -> Move` function;
# `play_game()` alternates two of them until the board is terminal, so a
# uniformly-random player and an MCTS-backed player are interchangeable.
#
# ## Cell 4.1: `random_player` and `make_mcts_player()`

# %%
mcts_player = rimtsaazmu.make_mcts_player(num_simulations=100)
print("random_player move:", rimtsaazmu.random_player(game, state))
print("mcts_player move:", mcts_player(game, state))

# %% [markdown]
# ## Cell 4.2: `play_game()` runs one full game

# %%
winner, history = rimtsaazmu.play_game(
    game, mcts_player, rimtsaazmu.random_player
)
print("winner:", winner)
print("number of states visited:", len(history))
print(game.render(history[-1]))

# %%
hintros.print_obj_info(rimtsaazmu.play_game)

# %% [markdown]
# # Part 5: Evaluation API
#
# **Mental model**: `evaluate_win_rate()` calls `play_game()` many times and
# reports outcome rates; `plot_win_rate_results()` turns that dict into a bar
# chart.
#
# ## Cell 5.1: Win rate over 100 games

# %%
results = rimtsaazmu.evaluate_win_rate(
    game, mcts_player, rimtsaazmu.random_player, num_games=100
)
print(results)

# %% [markdown]
# ## Cell 5.2: Plot the results

# %%
rimtsaazmu.plot_win_rate_results(results)

# %% [markdown]
# # Part 6: Connect Four
#
# **Mental model**: none of Parts 2-5 mentioned `TicTacToe` by name; the
# same `Game` interface, `run_mcts()`, and `play_game()` work unchanged
# against `ConnectFour`. What changes is only the board: 42 cells instead
# of 9, and a move is a column (the disc drops to the lowest empty row:
# gravity), not a free choice of cell. See `game.01.API.ipynb` Part 3 for the
# full `ConnectFour` API tour, including an interactive board.
#
# ## Cell 6.1: A tactical position to search from
#
# **Goal**:
# - Build intuition for how MCTS handles a position with a concrete threat,
#   for Connect Four
#
# - Red already has three discs in a row on the bottom row, anchored against
#   the left edge (columns 0-2)
# - Yellow's 3 discs are stacked out of the way in column 6
# - It is red's turn next
# - Column 3 is the only cell that completes a four-in-a-row, and it wins immediately for red

# %%
connect_four = rimtsaazge.ConnectFour()
_cf_empty_row = (0, 0, 0, 0, 0, 0, 0)
_cf_row3 = (0, 0, 0, 0, 0, 0, -1)
_cf_row4 = (0, 0, 0, 0, 0, 0, -1)
_cf_row5 = (1, 1, 1, 0, 0, 0, -1)
cf_demo_state = _cf_empty_row * 3 + _cf_row3 + _cf_row4 + _cf_row5
print(connect_four.render(cf_demo_state))
rimtsaazgau.plot_connect_four_board(cf_demo_state)

# %% [markdown]
# ## Cell 6.2: Visualizing the search tree on Connect Four
#
# **Goal**:
# - See the search tree `build_mcts_tree()` grows from `cf_demo_state`: the
#   board picture ties each child's move (a column) back to where the disc
#   would land, exactly like Cell 3.2 did for tic-tac-toe with text alone
# - With the exploration constant $C$ from the UCT formula
#   ($Q(s, a) + C \sqrt{\ln N(s) / N(s, a)}$) and the search budget
#   `num_simulations` set high enough, the robust child (highest $N(s, a)$)
#   should be column 3, the immediate win
# - Watch how few simulations it takes before column 3 pulls ahead of the
#   other 6 columns

# %%
rimtsaazmau.cell6_2_build_connect_four_tree_widget(connect_four, cf_demo_state)

# %% [markdown]
# **Key observations**:
# - Column 3 (the winning move) accumulates far more visits $N(s, a)$ than
#   the other 6 columns once the search budget is large enough
# - A small `num_simulations` can still miss it: with only 7 legal moves at
#   the root, every column is tried once before any is revisited (the
#   $N(s, a) = 0$ case in the UCT formula), so a tiny budget barely gets
#   past that first sweep
# - Raising `C` spreads visits more evenly across all 7 columns; lowering
#   it concentrates them on whichever column looked best early, right or
#   wrong

# %% [markdown]
# ## Cell 6.3: Evaluating MCTS on Connect Four
#
# **Goal**:
# - Reuse Part 5's `evaluate_win_rate()` / `plot_win_rate_results()`
#   unchanged, this time against `ConnectFour`, to see the same engine
#   generalize its evaluation, not just its search, to a bigger game

# %%
cf_mcts_player = rimtsaazmu.make_mcts_player(num_simulations=150)
cf_results = rimtsaazmu.evaluate_win_rate(
    connect_four, cf_mcts_player, rimtsaazmu.random_player, num_games=50
)
print(cf_results)

# %%
rimtsaazmu.plot_win_rate_results(cf_results)

# %% [markdown]
# ## Summary: The Mental Model
#
# - `Game` is the only contract `mcts_utils.py` depends on (see
#   `game.01.API.ipynb`): any class implementing its 6 methods can be searched
#   by `run_mcts()`, as shown by `TicTacToe` (Parts 2-5) and `ConnectFour`
#   (Part 6) sharing every line of search code
# - MCTS repeats 4 phases per simulation: Selection (descend by UCT),
#   Expansion (add one child), Simulation (random rollout to a terminal
#   state), Backpropagation (push the outcome back up, flipping sign each
#   ply)
# - The UCT formula $Q(s, a) + C \sqrt{\ln N(s) / N(s, a)}$ balances
#   exploiting the best-known child against exploring an under-visited one;
#   an unvisited child ($N(s, a) = 0$) is always tried first
# - The visit distribution at the root, not just its arg max, is the real
#   output of the search: `run_mcts()` returns the robust child, the one
#   with the highest $N(s, a)$, not the one with the highest $Q(s, a)$
