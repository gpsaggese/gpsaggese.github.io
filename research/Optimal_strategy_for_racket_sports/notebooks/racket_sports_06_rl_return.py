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
# # Learning Where to Return a Serve: a One-Shot RL Env
#
# - This notebook turns the Milestone 2 model into a Gymnasium env, where player 2
#   learns where to aim the return of a serve from a reward of 1 for a winner, in
#   tennis and pickleball (#713)
# - The pedagogical arc:
#   - The env: one point, one shot, a contextual bandit
#   - Brute force: the winner probability of every target, through the env
#   - Milestone 2 recomputed on the same grid, and compared with the brute force
#   - PPO learns the target from the reward only
#   - The targets of PPO against both maps, with a strict pass rule

# %%
# Install the CPU-only `torch` first: about 200 MB and 1 to 3 minutes the first
# time. From PyPI, `stable-baselines3` would pull the CUDA build, about 1.9 GB.
# !pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
# !pip install -q gymnasium==1.3.0 stable-baselines3==2.9.2

import gymnasium
import stable_baselines3
import torch

print("torch version: ", torch.__version__)
print("gymnasium version: ", gymnasium.__version__)
print("stable-baselines3 version: ", stable_baselines3.__version__)

# %%
# %load_ext autoreload
# %autoreload 2

import functools
import inspect
import logging

import gymnasium as gym
import numpy as np
import pandas as pd
import stable_baselines3 as sb3
from gymnasium.utils.env_checker import check_env

# %%
import helpers.hintrospection as hintros
import helpers.hnotebook as hnotebook

from research.Optimal_strategy_for_racket_sports import racket_params
from research.Optimal_strategy_for_racket_sports.notebooks import (
    racket_sports_06_rl_return_utils as utils,
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

# %% [markdown]
# # Part 0: How Much to Run

# %% [markdown]
# ## Cell 0.1: Reduced or Full Run
#
# - The reduced run takes about 10 minutes on a CPU
#   - The middle serve of the deuce box, 1000 points per target, and one PPO run of
#     50k steps with seed 0
# - The full run takes about 1 hour and gives the results of #713
#   - 9 serves on a 3 x 3 grid over the deuce box, 1000 points per target, and 3
#     PPO runs of 75k steps with the fresh seeds 3, 4, and 5
# - Both use the Milestone 2 defaults, its grid, and a fresh Milestone 2 map with
#   5000 samples per target and seed 2 (Milestone 2 used seed 1)

# %%
# Set to `True` for the full run.
RUN_FULL_SIMS = False
config = utils.get_experiment_config(RUN_FULL_SIMS)
print("RUN_FULL_SIMS=", RUN_FULL_SIMS)
print("config=", config)
sports = [racket_params.TENNIS, racket_params.PICKLEBALL]

# %% [markdown]
# # Part 1: The Env

# %% [markdown]
# ## Cell 1.1: Read the Code of the Env
#
# **Goal**
# - See the 4 pieces of the env: the 2 spaces, `reset()`, and `step()`
# - See that `step()` is one call of the Milestone 2 function
#   `compute_winner_probability()` with 1 sample, so the env has no model of its own

# %%
# Print the code of the env, which lives in the utils.
print(inspect.getsource(utils.ReturnEnv))

# %% [markdown]
# **Guided usage**
# - Find the line that sets the reward in `step()`
#   - Observe that the reward is the `p_winner` of a 1-sample call: 1 for a winner,
#     in and not reached, else 0
# - Find where `reset()` draws the serve
#   - Observe that every draw comes from `self.np_random`, so a seed fixes the point

# %% [markdown]
# ## Cell 1.2: Create the Env for Both Sports
#
# **Goal**
# - Build one env per sport with the Milestone 2 defaults, and read the task from
#   the spaces

# %%
# Build one env per sport, with the Milestone 2 grid and ball speed.
envs = {sport.name: utils.build_env(sport) for sport in sports}
for name, env in envs.items():
    print(f"{name}: observation_space=", env.observation_space)
    print(
        f"{name}: action_space=", env.action_space, "grid_size=", env.grid_size
    )
    print(f"{name}: serve_box=", env.serve_box)
# Outcome: `Discrete(150)` and `Discrete(90)`, the grids of Milestone 2.

# %% [markdown]
# **Guided usage**
# - Read the action spaces
#   - Observe 150 targets in tennis and 90 in pickleball: the grids of Milestone 2
# - Read the bounds of the observation space
#   - Observe 3 points, player 1, player 2, and the serve, each bounded by the court

# %% [markdown]
# ## Cell 1.3: Play One Point
#
# **Goal**
# - Reset with a fixed serve and aim at 2 targets: the middle of player 1's half,
#   and the cell in front of player 1
# - See the outcome in `info`, and that one shot ends the point

# %%
# Aim 2 returns from the middle serve of the tennis deuce box, with the same seed.
env = envs["Tennis"]
serve_xy = utils.get_eval_serves(env, 1, 1)[0]
middle_xy = (0.0, -env.sport.court.length_m / 4)
targets = {
    "middle": utils.get_nearest_action(env, middle_xy),
    "near player 1": utils.get_nearest_action(env, env.player1_xy),
}
seed = 0
obs, _ = env.reset(seed=seed, options={"serve_xy": serve_xy})
print("obs=", obs)
for label, action in targets.items():
    env.reset(seed=seed, options={"serve_xy": serve_xy})
    _, reward, terminated, truncated, info = env.step(action)
    print(f"{label}: action={action} reward={reward} info={info}")
    print(f"  terminated={terminated} truncated={truncated}")
# Outcome: with seed 0, a winner in the middle, and a return in front of player 1
# that they reach.

# %% [markdown]
# **Guided usage**
# - Change `seed`
#   - Observe the middle target stay a winner for every seed, while the target in
#     front of player 1 wins about half the time: about a third of its returns
#     land out past the baseline, and a fifth are reached
# - Read `terminated` and `truncated`
#   - Observe that one shot ends the point as terminated, not truncated

# %% [markdown]
# ## Cell 1.4: Check the Env Against the Gymnasium API
#
# **Goal**
# - Run the checks that Gymnasium ships for custom envs, before any training: the
#   spaces, the types that `reset()` and `step()` return, and the seeding

# %%
# Check each env; the env has no render modes, so skip the render check.
for name, env in envs.items():
    check_env(env, skip_render_check=True)
    print(f"{name}: check_env passed")
# Outcome: both envs pass.

# %% [markdown]
# **Guided usage**
# - Replace `self.np_random` with `np.random.default_rng()` in `step()` in a copy
#   of the env
#   - Observe that `check_env()` may still pass, since it compares one step, while
#     the unit test that compares 50 seeded points fails

# %% [markdown]
# # Part 2: Brute Force Through the Env

# %% [markdown]
# ## Cell 2.1: Score Every Target of Each Serve
#
# **Goal**
# - Estimate the winner probability of every target with the env only: points
#   played through `reset()` and `step()`, with the same reset seeds for every
#   target (common random numbers)

# %% [markdown]
# **Description**
# - Inputs
#   - `config.n_serves_per_side`: 1 gives the middle serve, 3 gives 9 serves
#   - `config.n_samples`: points per target
#
# - Panels
#   - `Top targets`: the 3 targets with the highest winner probability of the
#     middle serve, per sport

# %%
# Score every target of every serve through the env, for each sport.
serves = {}
env_maps = {}
for name, env in envs.items():
    n = config.n_serves_per_side
    serves[name] = utils.get_eval_serves(env, n, n)
    env_maps[name] = [
        utils.compute_env_heatmap(env, serve, config.n_samples, config.env_seed)
        for serve in serves[name]
    ]
# The middle serve: the only one of the reduced run, the 5th of 9 in the full run.
middle = config.n_serves_per_side**2 // 2
columns = ["cell_id", "x_m", "y_m", "p_winner", "p_winner_se"]
for name in envs:
    print(f"{name}: middle serve {serves[name][middle]}")
    display(env_maps[name][middle].nlargest(3, "p_winner")[columns])

# %% [markdown]
# **Guided usage**
# - Read the top targets of tennis
#   - Observe several targets at the same winner probability: a plateau
# - Set `RUN_FULL_SIMS = True` in Cell 0.1 and rerun
#   - Observe 9 serves per sport instead of the middle one, in about 1 hour

# %% [markdown]
# **Implementation** `utils.compute_env_heatmap(env, serve_xy, n_samples, seed)`
# - Plays `n_samples` points per target, each a `reset()` with the serve and the
#   seed `seed + k` and a `step()` aimed at the target
# - Averages the rewards per target into `p_winner`, with the standard error of a
#   fraction, in the columns of `compute_winner_heatmap()`

# %%
hintros.print_obj_info(utils.compute_env_heatmap)

# %% [markdown]
# ## Cell 2.2: Map of the Brute Force and Its Best Region
#
# **Goal**
# - See where the winners are: the brute force map of the middle serve on the court

# %% [markdown]
# **Description**
# - Panels
#   - `Brute force`: one panel per sport, each cell colored by its winner
#     probability from red (0) to green (1), the best region outlined (the cells
#     within 2 standard errors of the highest), its middle as a star, and the 2
#     players as dots

# %%
# Plot the brute force map of the middle serve, for each sport.
for sport in sports:
    name = sport.name
    _ = utils.plot_heatmaps(
        {"Brute force": env_maps[name][middle]},
        sport.court,
        serves[name][middle],
        figsize=(6.0, 7.0),
    )

# %% [markdown]
# **Guided usage**
# - Compare the 2 sports
#   - Observe a large best region in tennis and a small one in pickleball

# %% [markdown]
# **Implementation** `utils.plot_heatmaps(heatmaps, court, serve_xy, targets_xy=(),
# figsize=None)`
# - Draws one panel per map with `plot_winner_heatmap()` of notebook 01
# - Marks `targets_xy` on every panel, and crops each panel to player 1's half

# %%
hintros.print_obj_info(utils.plot_heatmaps)

# %% [markdown]
# # Part 3: Milestone 2 Recomputed

# %% [markdown]
# ## Cell 3.1: Recompute the Milestone 2 Map on the Same Grid
#
# **Goal**
# - Compute the Milestone 2 map now, with the settings and the grid of the env and
#   a new seed, instead of reading a saved result

# %%
# Recompute the Milestone 2 map of every serve, for each sport.
m2_maps = {
    name: [
        utils.compute_m2_heatmap(
            envs[name], serve, config.m2_n_samples, config.m2_seed
        )
        for serve in serves[name]
    ]
    for name in envs
}
for name in envs:
    print(f"{name}: middle serve {serves[name][middle]}")
    display(m2_maps[name][middle].nlargest(3, "p_winner")[columns])

# %% [markdown]
# **Guided usage**
# - Compare with Cell 2.1
#   - Observe top targets with a winner probability at or close to 1 in both maps,
#     on different cells: both maps have a plateau of near ties

# %% [markdown]
# **Implementation** `utils.compute_m2_heatmap(env, serve_xy, n_samples, seed)`
# - Calls `compute_winner_heatmap()` of notebook 01 with the spread, the speeds,
#   the court, and the grid of the env

# %%
hintros.print_obj_info(utils.compute_m2_heatmap)

# %% [markdown]
# ## Cell 3.2: Compare the Brute Force With Milestone 2
#
# **Goal**
# - Check that the env reproduces Milestone 2: the highest cell of each map is in
#   the best region of the other, and the 2 maps rank the cells the same way

# %% [markdown]
# **Description**
# - Panels
#   - `Comparison`: one row per sport and serve: the containment both ways, the
#     Spearman rank correlation over the cells, and the size of each best region
#   - `Maps`: the brute force and Milestone 2 maps of the middle serve, per sport

# %%
# Compare the 2 maps of every serve.
rows = []
for name in envs:
    for serve, env_map, m2_map in zip(
        serves[name], env_maps[name], m2_maps[name]
    ):
        rows.append(
            {
                "sport": name,
                "serve": serve,
                **utils.compare_heatmaps(env_map, m2_map),
            }
        )
comparison = pd.DataFrame(rows)
display(comparison)

# %%
# Plot both maps of the middle serve, for each sport.
for sport in sports:
    name = sport.name
    _ = utils.plot_heatmaps(
        {
            "Brute force": env_maps[name][middle],
            "Milestone 2": m2_maps[name][middle],
        },
        sport.court,
        serves[name][middle],
        figsize=(12.0, 7.0),
    )

# %% [markdown]
# **Guided usage**
# - Read the `spearman` column
#   - Observe a rank correlation above 0.95, the threshold of #713
# - Compare the 2 region sizes
#   - Observe a larger region for the brute force, which has fewer samples per cell
# - Read `env_best_in_m2_region` and `m2_best_in_env_region`
#   - Observe `True` for both: the highest cell of each map is in the best region
#     of the other
#   - With 200 points per target, many cells tie at a winner probability of 1, and
#     the first of the tie fell outside the narrow region of Milestone 2 (#713,
#     Phase 3), so both runs use 1000 points

# %% [markdown]
# **Implementation** `utils.compare_heatmaps(env_heatmap, m2_heatmap)`
# - Finds the best region of each map with `find_best_region()` of notebook 01
# - Checks the highest cell of each map against the region of the other, and
#   computes the Spearman correlation of the 2 maps

# %%
hintros.print_obj_info(utils.compare_heatmaps)

# %% [markdown]
# # Part 4: Train a PPO Agent

# %% [markdown]
# ## Cell 4.1: Create the Agents on Rescaled Observations
#
# **Goal**
# - Create one `PPO` agent per sport and seed, with the default `MlpPolicy`, on its
#   own env
# - Rescale the observation from the court to $[-1, 1]$ for the agent, while the
#   env stays in meters
#   - SB3 feeds a `Box` observation to the network as it is, and player 1 at
#     $y = -11.885$ m is a constant input that saturated a quarter to a third of
#     the first layer (#713, Phase 2)

# %%
# Create one agent per sport and seed, on rescaled observations.
models = {}
for sport in sports:
    for seed in config.seeds:
        train_env = gym.wrappers.RescaleObservation(
            utils.build_env(sport), np.float32(-1.0), np.float32(1.0)
        )
        models[(sport.name, seed)] = sb3.PPO(
            "MlpPolicy", train_env, seed=seed, device="cpu", verbose=0
        )
# One wrapper per sport rescales the observations of the evaluation.
rescalers = {
    sport.name: gym.wrappers.RescaleObservation(
        utils.build_env(sport), np.float32(-1.0), np.float32(1.0)
    )
    for sport in sports
}
obs = envs["Tennis"].reset(options={"serve_xy": serves["Tennis"][middle]})[0]
print("obs in meters=", obs)
print("obs rescaled=", rescalers["Tennis"].observation(obs))
print("policy=", models[("Tennis", config.seeds[0])].policy)

# %% [markdown]
# **Guided usage**
# - Read the 2 observations
#   - Observe player 1 at $-11.885$ m become $-1$, and the serve fall in $[-1, 1]$
# - Read the size of the action head of the policy
#   - Observe one output per target cell

# %% [markdown]
# ## Cell 4.2: Train for a Fixed Budget
#
# **Goal**
# - Train each agent on random serves for the budget of the config: 50k steps in
#   the reduced run, 75k in the full run, before PPO starts to drift on a reward
#   that is almost always 1 (#713, Phase 2)

# %%
# Train each agent in place.
for (name, seed), model in models.items():
    model.learn(total_timesteps=config.total_timesteps)
    rewards = [info["r"] for info in model.ep_info_buffer]
    print(
        f"{name}, seed {seed}: {model.num_timesteps} steps, mean reward of the"
        f" last {len(rewards)} points {np.mean(rewards):.3f}"
    )

# %% [markdown]
# **Guided usage**
# - Read the mean reward of the last points
#   - Observe a value close to the best winner probability of Part 3

# %% [markdown]
# ## Cell 4.3: Score the Agents on the Serves
#
# **Goal**
# - For each agent and serve: the greedy cell, whether it is in the best region of
#   each map, its regret, and the expected regret over every cell the policy plays

# %% [markdown]
# **Description**
# - Panels
#   - `Results`: one row per sport, seed, and serve; the regrets are in points of
#     winner probability against the Milestone 2 map
#     - `regret_points`: the highest winner probability minus that of the greedy
#       cell
#     - `expected_regret_points`: the same for the policy, which plays each cell
#       with its probability
#     - `random_regret_points`: the same for a uniformly random target

# %%
# Map an observation in meters to the action probabilities of `model`, with the
# observation rescaled as in training.
def get_probs(
    model: sb3.PPO, rescaler: gym.ObservationWrapper, obs: np.ndarray
) -> np.ndarray:
    obs_tensor, _ = model.policy.obs_to_tensor(rescaler.observation(obs))
    probs = model.policy.get_distribution(obs_tensor).distribution.probs
    probs_np: np.ndarray = probs.detach().numpy()[0]
    return probs_np


# Score each agent on the serves of its sport.
results = []
for (name, seed), model in models.items():
    get_model_probs = functools.partial(get_probs, model, rescalers[name])
    result = utils.evaluate_policy(
        envs[name], get_model_probs, serves[name], env_maps[name], m2_maps[name]
    )
    results.append(result.assign(sport=name, seed=seed))
results = pd.concat(results, ignore_index=True)
display(results.round(4))

# %% [markdown]
# **Guided usage**
# - Compare `regret_points` with `random_regret_points`
#   - Observe a regret of a fraction of a point, against 10 to 20 points for a
#     random target
# - Read `top_prob`
#   - Observe that the policy is still spread: its greedy cell gets a fraction of
#     the probability, so `expected_regret_points` is the fairer summary

# %% [markdown]
# **Implementation** `utils.evaluate_policy(env, get_probs, serves, env_heatmaps,
# m2_heatmaps)`
# - Resets the env with each serve and gets the action probabilities of the policy
# - Takes the most likely action as the greedy cell, and scores it with
#   `score_target()`
# - Weighs the Milestone 2 map with the probabilities for the expected regret

# %%
hintros.print_obj_info(utils.evaluate_policy)

# %% [markdown]
# ## Cell 4.4: Apply the Pass Rule
#
# **Goal**
# - Apply the rule of #713, written before the full run: an agent passes if the
#   greedy cell of every serve is in the best region of the fresh Milestone 2 map
# - Also report, not as a pass condition, the max regrets and a pass with a
#   tolerance of 1 point

# %%
# Summarize each run, one per sport and seed.
summary = utils.summarize_pass_rule(results)
display(summary.round(4))
print("strict pass for every run=", bool(summary["strict_pass"].all()))
print(
    "runs passing with a 1-point tolerance=",
    f"{int(summary['pass_1_point'].sum())} of {len(summary)}",
)

# %% [markdown]
# **Guided usage**
# - Read `strict_pass` and `pass_1_point`
#   - Observe whether a miss of the strict rule is a near miss, within 1 point

# %% [markdown]
# **Implementation** `utils.summarize_pass_rule(results)`
# - Groups the results by sport and seed, and counts the serves whose greedy cell
#   is in the Milestone 2 region
# - Passes a run if every serve is in, and with the tolerance if every greedy
#   regret is at most 1 point

# %%
hintros.print_obj_info(utils.summarize_pass_rule)

# %% [markdown]
# # Part 5: Compare Everything

# %% [markdown]
# ## Cell 5.1: Maps With the Targets of PPO
#
# **Goal**
# - See the greedy cells of the agents on both maps of the middle serve

# %% [markdown]
# **Description**
# - Panels
#   - `Brute force` and `Milestone 2`: as in Cell 3.2, with the greedy cell of each
#     seed as a purple cross

# %%
# Plot both maps of the middle serve with the greedy cells, for each sport.
for sport in sports:
    name = sport.name
    middle_serve = serves[name][middle]
    targets_xy = tuple(
        (row.x_m, row.y_m)
        for row in results.itertuples()
        if row.sport == name and (row.x1, row.y1) == middle_serve
    )
    _ = utils.plot_heatmaps(
        {
            "Brute force": env_maps[name][middle],
            "Milestone 2": m2_maps[name][middle],
        },
        sport.court,
        middle_serve,
        targets_xy=targets_xy,
        figsize=(12.0, 7.0),
    )

# %% [markdown]
# **Guided usage**
# - Find the purple crosses on the pickleball maps
#   - Observe them inside the black outline of the best region

# %% [markdown]
# ## Conclusion
#
# - The env scores every target from its own points, and PPO learns a target from
#   the reward alone; neither reads a Milestone 2 result
# - Part 3 compares the env with a fresh Milestone 2 map, and Cell 4.4 applies the
#   pass rule to PPO
# - Results of the full run: to be written after Phase 4 of #713
# - Next: cut the search space with closed-form results, the ad side, rallies,
#   then self-play with `PettingZoo`
