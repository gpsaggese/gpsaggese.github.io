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
# # Monte Carlo Tree Search for tic-tac-toe
#
# Milestone 1 of the [MCTS and AlphaZero project](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/research/ideas/draft.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.md): pure MCTS on tic-tac-toe, no neural network involved yet.
#
# - The game-agnostic MCTS engine lives in `mcts_utils.py`
# - Concrete games (`TicTacToe`, `ConnectFour`) live in `game_examples.py`
# - This notebook only imports from them, runs a few games, and reports the
#   results
# - See `README.md` for a description of every file in this directory

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
# !/bin/bash -c "(source /venv/bin/activate; pip install --quiet tqdm)"

# %%
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.mcts_utils as rimtsaazmu
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.game_examples as rimtsaazge


# %% [markdown]
# # Part 2: The game
#
# `TicTacToe` exposes a small, game-agnostic interface: legal moves, applying a
# move, checking for a winner, reporting whose turn it is, and rendering the
# board. Any other two-player game (Connect Four, say) can implement the same
# six methods and reuse the search code below unchanged; `game_examples.py`
# already includes both.
#
# ## Cell 2.1: An empty board


# %%
game = rimtsaazge.TicTacToe()
state = game.get_initial_state()
print(game.render(state))


# %% [markdown]
# # Part 3: MCTS on a single position
#
# ## Cell 3.1: MCTS finds the winning move
#
# X already has two in a row; playing the third cell wins immediately. A couple
# hundred simulations should be enough for MCTS to find it: every rollout that
# passes through that move backpropagates a win, so its visit count quickly
# pulls ahead of the other candidates.

# %%
demo_state = (1, 1, 0, -1, -1, 0, 0, 0, 0)
print(game.render(demo_state))

move = rimtsaazmu.run_mcts(game, demo_state, num_simulations=200)
print("MCTS move:", move)

# %% [markdown]
# # Part 4: Full games
#
# `play_game()` alternates two player functions until the board is terminal
# and, with `verbose=True`, prints the board after every move. Two matchups:
# MCTS against a random player, and MCTS against itself.
#
# ## Cell 4.1: MCTS vs random

# %%
mcts_player = rimtsaazmu.make_mcts_player(num_simulations=200)

winner, _ = rimtsaazmu.play_game(
    game, mcts_player, rimtsaazmu.random_player, verbose=True
)
print("\nwinner:", winner)

# %% [markdown]
# ## Cell 4.2: MCTS vs MCTS
#
# No randomness on either side this time.

# %%
winner, _ = rimtsaazmu.play_game(game, mcts_player, mcts_player, verbose=True)
print("\nwinner:", winner)

# %% [markdown]
# **Key observations**:
# - Tic-tac-toe is a solved game: with correct play it is always a draw
# - MCTS vs MCTS reflects that most of the time
# - MCTS vs random usually ends in a win for MCTS, since a uniformly random
#   opponent occasionally leaves a line open

# %% [markdown]
# # Part 5: Evaluation
#
# ## Cell 5.1: Win rate over 300 games
#
# A single game does not say much about whether MCTS is actually better than
# chance. Playing several hundred games against a random opponent and tracking
# the outcome rate gives a more reliable picture.

# %%
results = rimtsaazmu.evaluate_win_rate(
    game, mcts_player, rimtsaazmu.random_player, num_games=300
)
print(results)
rimtsaazmu.plot_win_rate_results(results)

# %% [markdown]
# **Key observations**:
# - MCTS wins the large majority of games and essentially never loses
# - The rare non-wins are draws: the worst outcome a random opponent can force
#   against correct play, never a loss
