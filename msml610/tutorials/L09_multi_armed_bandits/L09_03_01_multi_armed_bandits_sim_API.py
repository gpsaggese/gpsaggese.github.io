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
# # Multi-Armed Bandit Simulation Classes API
#
# An exploration of the simulation classes used in the Multi-Armed
# Bandits lesson (`L09_03_multi_armed_bandits_sim.py`):
# - **Environment**: `MultiArmedBandit`, a K-armed casino with hidden reward
#   means
# - **Policy**: `Strategy` and its concrete subclasses, deciding which arm to
#   pull next
# - **Orchestration**: `BanditExperiment`, `BanditSimulation`, and
#   `BanditEnsemble`, running one, many, or many-times-many trials

# %% [markdown]
# ## Imports and Setup

# %%
# %load_ext autoreload
# %autoreload 2

import logging
import warnings

warnings.filterwarnings("ignore")

# %%
import L09_03_multi_armed_bandits_sim as sim
import L09_03_multi_armed_bandits_utils as utils

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import helpers.hintrospection as hintros

hdbg.init_logger(verbosity=logging.INFO)
_LOG = logging.getLogger(__name__)
hnotebo.config_notebook()

try:
    from IPython.display import display, Markdown
except ImportError:
    display = print  # type: ignore
    Markdown = lambda x: x

# %% [markdown]
# ## Library Overview
#
# - **What problem it solves**:
#   - Simulates the exploration/exploitation tradeoff: a gambler with $N$
#     coins facing $K$ slot machines with unknown, fixed payout means
#   - Lets different action-selection policies be run against the same
#     environment and compared statistically
# - **Key abstraction**:
#   - An environment (`MultiArmedBandit`) is pulled by a policy (`Strategy`)
#   - A single run is a `BanditExperiment`
#   - Many runs (varying seed) form a `BanditSimulation`
#   - Many simulations (varying the hidden means) form a `BanditEnsemble`
# - **Mental model**:
#
# | Object | Description | Comments |
# |--------|-------------|----------|
# | `MultiArmedBandit(...)` | Environment with $K$ machines | Holds hidden `mu_values`, tracks pulls/rewards |
# | `Strategy` | Abstract policy interface | `.select_machine(bandit) -> int` |
# | `ExplorationStrategy` | Pure random policy | Concrete `Strategy` |
# | `ExploitationStrategy` | Pure greedy policy | Concrete `Strategy` |
# | `EpsilonGreedyStrategy` | Explore w.p. $\epsilon$, else exploit | Concrete `Strategy` |
# | `BanditExperiment(...)` | One bandit + one strategy | `.run() -> (rewards, cumulative, total)` |
# | `BanditSimulation(...)` | Many experiments, fixed `mu_values` | `.run_trials(...)`, `.epsilon_sweep(...)` |
# | `BanditEnsemble(...)` | Many simulations, random `mu_values` | `.run_ensemble(...)`, `.compare_strategies_ensemble(...)` |
#
# - **Key classes**:
#   - `MultiArmedBandit`: the environment
#   - `Strategy` (and `ExplorationStrategy`, `ExploitationStrategy`,
#     `EpsilonGreedyStrategy`): the policies
#   - `BanditExperiment`, `BanditSimulation`, `BanditEnsemble`: increasing
#     levels of repetition, for statistics instead of a single anecdote

# %% [markdown]
# # Part 1: The Environment: `MultiArmedBandit`

# %% [markdown]
# ## Cell 1.1: Mental Model and Smallest Construction
#
# **Goal**:
# - Understand what state a `MultiArmedBandit` holds
# - Construct the smallest possible bandit
#
# | Member | Description | Signature |
# |--------|-------------|-----------|
# | `MultiArmedBandit(*, k_machines, mu_values, seed, width=0.3)` | Constructor | Rewards are uniform in `[mu_i - width, mu_i + width]`, clipped to `[-1, 1]` |
# | `.pull(machine_idx)` | Pull one machine | Returns a reward, updates statistics |
# | `.get_empirical_means()` | Observed means so far | `List[float]`, one per machine |
# | `.reset(seed=None)` | Clear statistics | Keeps `mu_values` |

# %% [markdown]
# ## Cell 1.2: Inspect the Object

# %%
hintros.print_obj_info(sim.MultiArmedBandit)

# %%
# Smallest possible bandit: 3 machines with distinct hidden means.
bandit = sim.MultiArmedBandit(k_machines=3, mu_values=[-0.2, 0.0, 0.5], seed=42)
print("type(bandit)=", type(bandit))
print("bandit.k_machines=", bandit.k_machines)
print("bandit.mu_values=", bandit.mu_values)

# %% [markdown]
# ## Cell 1.3: Pulling Machines
#
# **Goal**:
# - See that `.pull()` returns a random reward and mutates internal state

# %%
# Pull machine 2 (the best one, mu=0.5) three times.
for _ in range(3):
    reward = bandit.pull(2)
    print("reward=", reward)
print("bandit.machine_pulls=", bandit.machine_pulls)

# %% [markdown]
# ## Cell 1.4: Empirical Means and Reset
#
# **Goal**:
# - Compare the true hidden means to what has been observed so far
# - Confirm that `.reset()` clears statistics but keeps `mu_values`

# %%
print("empirical_means=", bandit.get_empirical_means())
bandit.reset()
print("after reset, machine_pulls=", bandit.machine_pulls)
print("mu_values still=", bandit.mu_values)

# %% [markdown]
# # Part 2: `Strategy` Hierarchy

# %% [markdown]
# ## Cell 2.1: `Strategy` is Abstract
#
# **Goal**:
# - Confirm `Strategy` cannot be instantiated directly
#
# | Member | Description | Signature |
# |--------|-------------|-----------|
# | `Strategy` (`abc.ABC`) | Common policy interface | Abstract base class |
# | `.select_machine(bandit)` | Choose next machine | Abstract, `-> int` |
# | `.reset()` | Clear internal state | Concrete no-op default |

# %%
hintros.print_obj_info(sim.Strategy)

try:
    sim.Strategy()
except TypeError as e:
    print("TypeError=", e)

# %% [markdown]
# ## Cell 2.2: `ExplorationStrategy`: Pure Random
#
# **Goal**:
# - See that exploration ignores the bandit's statistics entirely

# %%
hintros.print_obj_info(sim.ExplorationStrategy)

# Instantiate strategy.
exploration = sim.ExplorationStrategy(seed=0)
bandit.reset()
picks = [exploration.select_machine(bandit) for _ in range(10)]
print("random picks=", picks)

# %% [markdown]
# ## Cell 2.3: `ExploitationStrategy`: Pure Greedy
#
# **Goal**:
# - See the mandatory warm-up (one pull per machine)
# - See that afterward, the current best empirical machine is always picked

# %%
hintros.print_obj_info(sim.ExploitationStrategy)

# %%
exploitation = sim.ExploitationStrategy()
bandit.reset()
picks = []
for _ in range(6):
    machine_idx = exploitation.select_machine(bandit)
    bandit.pull(machine_idx)
    picks.append(machine_idx)
print("picks=", picks, "(first 3 are the warm-up pulls 0, 1, 2)")

# %% [markdown]
# ## Cell 2.4: `EpsilonGreedyStrategy`: Balanced
#
# **Goal**:
# - See that `epsilon` controls how often a random (exploratory) pull happens

# %%
hintros.print_obj_info(sim.EpsilonGreedyStrategy)

# %%
epsilon_greedy = sim.EpsilonGreedyStrategy(epsilon=0.5, seed=0)
bandit.reset()
picks = []
for _ in range(20):
    machine_idx = epsilon_greedy.select_machine(bandit)
    bandit.pull(machine_idx)
    picks.append(machine_idx)
print("picks=", picks)

# %% [markdown]
# # Part 3: A Single Run: `BanditExperiment`

# %% [markdown]
# ## Cell 3.1: Mental Model and Construction
#
# **Goal**:
# - Wrap a bandit and a strategy into a single, runnable experiment
#
# | Member | Description | Signature |
# |--------|-------------|-----------|
# | `BanditExperiment(*, bandit, strategy, n_coins)` | Constructor | `n_coins` is the number of pulls to play |
# | `.run()` | Play `n_coins` pulls | Returns `(rewards, cumulative_rewards, final_total)` |

# %%
hintros.print_obj_info(sim.MultiArmedBandit)

# %%
bandit = sim.MultiArmedBandit(k_machines=3, mu_values=[-0.2, 0.0, 0.5], seed=42)
strategy = sim.EpsilonGreedyStrategy(epsilon=0.2, seed=1)
experiment = sim.BanditExperiment(bandit=bandit, strategy=strategy, n_coins=20)
print("type(experiment)=", type(experiment))

# %% [markdown]
# ## Cell 3.2: Running the Experiment
#
# **Goal**:
# - Observe the three outputs of `.run()`: per-pull rewards, running total,
#   and the final total

# %%
rewards, cumulative_rewards, final_total = experiment.run()
print("rewards[:5]=", rewards[:5])
print("cumulative_rewards[:5]=", cumulative_rewards[:5])
print("final_total=", final_total)

# %% [markdown]
# # Part 4: Many Runs: `BanditSimulation`

# %% [markdown]
# ## Cell 4.1: Mental Model and Construction
#
# **Goal**:
# - Fix the environment (`k_machines`, `mu_values`, `n_coins`) once, then run
#   many independent experiments against it
#
# | Member | Description | Signature |
# |--------|-------------|-----------|
# | `BanditSimulation(*, k_machines, mu_values, n_coins, base_seed=0)` | Constructor | Holds fixed environment parameters |
# | `.run_trials(*, strategy_class, strategy_params, n_trials)` | Repeat `n_trials` experiments | New bandit + strategy seed per trial |
# | `.epsilon_sweep(*, n_trials, epsilon_values=None)` | Compare across $\epsilon$ | Also runs pure exploration/exploitation baselines |

# %%
hintros.print_obj_info(sim.BanditSimulation)

# %%
simulation = sim.BanditSimulation(
    k_machines=3, mu_values=[-0.2, 0.0, 0.5], n_coins=50, base_seed=0
)
print("type(simulation)=", type(simulation))

# %% [markdown]
# ## Cell 4.2: `.run_trials()`: Statistics Over Many Experiments
#
# **Goal**:
# - See that `.run_trials()` returns aggregated statistics, not a single
#   anecdote
#
# **Non-obvious behavior**: if `"seed"` is a key in `strategy_params`,
# `.run_trials()` overwrites its value per trial (`bandit_seed + 1000`), so
# each trial's strategy gets its own seed. The key must be present (even with
# a placeholder value) for a strategy that requires `seed`.

# %%
trial_results = simulation.run_trials(
    strategy_class=sim.EpsilonGreedyStrategy,
    # `seed` is a placeholder: `.run_trials()` overwrites it per trial.
    strategy_params={"epsilon": 0.2, "seed": 0},
    n_trials=10,
)
print("keys=", list(trial_results.keys()))
print("mean_final=", trial_results["mean_final"])
print("std_final=", trial_results["std_final"])

# %% [markdown]
# ## Cell 4.3: `.epsilon_sweep()`: Comparing Policies
#
# **Goal**:
# - Compare pure exploration, pure exploitation, and epsilon-greedy across a
#   range of $\epsilon$ values
#
# _Left panel_: mean final reward for each $\epsilon$, with exploration and
# exploitation shown as horizontal reference lines
#
# _Right panel_: cumulative reward over time for the best $\epsilon$ found

# %%
sweep_results = simulation.epsilon_sweep(
    n_trials=10, epsilon_values=[0.0, 0.25, 0.5, 0.75, 1.0]
)
utils.plot_epsilon_sweep(sweep_results=sweep_results, n_coins=50)

# %% [markdown]
# **Key observations**:
# - Pure exploitation is fragile: an unlucky warm-up can lock it onto a
#   suboptimal machine
# - Pure exploration wastes coins on known-bad machines forever
# - A moderate $\epsilon$ (neither 0 nor 1) usually wins

# %% [markdown]
# # Part 5: Many Simulations: `BanditEnsemble`

# %% [markdown]
# ## Cell 5.1: Mental Model and Construction
#
# **Goal**:
# - Go one level higher: average results over many *random* hidden-mean
#   configurations, not just one fixed `mu_values`
#
# | Member | Description | Signature |
# |--------|-------------|-----------|
# | `BanditEnsemble(*, k_machines, n_coins, mu_range=(-0.5, 0.5), base_seed=0)` | Constructor | `mu_values` are drawn randomly per configuration |
# | `.run_ensemble(*, strategy_class, strategy_params, n_trials, n_mu_configs)` | Repeat `.run_trials()` per config | Returns cross-configuration statistics |
# | `.compare_strategies_ensemble(*, n_trials, n_mu_configs, epsilon=0.1)` | Run all 3 policies | Returns one result dict per policy |
# | `.plot_ensemble_comparison(*, ensemble_results, epsilon=0.1)` | Bar chart | Visualizes `.compare_strategies_ensemble()` output |

# %%
hintros.print_obj_info(sim.BanditEnsemble)

# %%
ensemble = sim.BanditEnsemble(k_machines=3, n_coins=50, base_seed=0)
print("type(ensemble)=", type(ensemble))

# %% [markdown]
# ## Cell 5.2: `.run_ensemble()`: One Policy, Many Random Worlds

# %%
ensemble_one_policy = ensemble.run_ensemble(
    strategy_class=sim.EpsilonGreedyStrategy,
    # `seed` is a placeholder: `.run_ensemble()` overwrites it per trial.
    strategy_params={"epsilon": 0.2, "seed": 0},
    n_trials=5,
    n_mu_configs=4,
)
print("overall_mean=", ensemble_one_policy["overall_mean"])
print("overall_std=", ensemble_one_policy["overall_std"])

# %% [markdown]
# ## Cell 5.3: `.compare_strategies_ensemble()`: All Policies at Once
#
# _Bars_: mean final reward per policy, averaged over random `mu_values`
# configurations
# _Error bars_: standard deviation across configurations

# %%
ensemble_results = ensemble.compare_strategies_ensemble(
    n_trials=5, n_mu_configs=4, epsilon=0.2
)
ensemble.plot_ensemble_comparison(ensemble_results=ensemble_results, epsilon=0.2)

# %% [markdown]
# # Part 6: Composition Examples

# %% [markdown]
# ## Cell 6.1: Smallest Meaningful Object
#
# Just the environment, pulled a few times.

# %%
mini_bandit = sim.MultiArmedBandit(k_machines=2, mu_values=[0.1, 0.3], seed=7)
print([mini_bandit.pull(0) for _ in range(3)])

# %% [markdown]
# ## Cell 6.2: Add a Policy, Manually
#
# Drive the environment with a policy's decisions, without `BanditExperiment`.
# This is exactly what `BanditExperiment.run()` automates.

# %%
mini_strategy = sim.EpsilonGreedyStrategy(epsilon=0.3, seed=7)
mini_bandit.reset()
total = 0.0
for _ in range(10):
    machine_idx = mini_strategy.select_machine(mini_bandit)
    total += mini_bandit.pull(machine_idx)
print("manual total=", total)

# %% [markdown]
# ## Cell 6.3: Combine Into a `BanditExperiment`
#
# Same environment and policy shape, now expressed with the library's own
# orchestration class.

# %%
mini_bandit.reset()
mini_experiment = sim.BanditExperiment(
    bandit=mini_bandit, strategy=mini_strategy, n_coins=10
)
_, _, mini_total = mini_experiment.run()
print("BanditExperiment total=", mini_total)

# %% [markdown]
# ## Cell 6.4: End-to-End: `BanditSimulation`
#
# Repeat Example 3's shape 20 times with varying seeds, to get a statistic
# instead of one anecdote.

# %%
mini_simulation = sim.BanditSimulation(
    k_machines=2, mu_values=[0.1, 0.3], n_coins=10, base_seed=7
)
mini_stats = mini_simulation.run_trials(
    strategy_class=sim.EpsilonGreedyStrategy,
    strategy_params={"epsilon": 0.3, "seed": 0},
    n_trials=20,
)
print(
    "mean_final=",
    mini_stats["mean_final"],
    "std_final=",
    mini_stats["std_final"],
)

# %% [markdown]
# # Part 7: API Patterns

# %% [markdown]
# ## Cell 7.1: Strategy Pattern
#
# Any `Strategy` subclass plugs into the same `BanditExperiment` unchanged.

# %%
for strategy_obj in [
    sim.ExplorationStrategy(seed=0),
    sim.ExploitationStrategy(),
    sim.EpsilonGreedyStrategy(epsilon=0.2, seed=0),
]:
    bandit.reset()
    experiment = sim.BanditExperiment(
        bandit=bandit, strategy=strategy_obj, n_coins=20
    )
    _, _, total = experiment.run()
    print(type(strategy_obj).__name__, "-> total=", total)

# %% [markdown]
# ## Cell 7.2: Keyword-Only Configuration
#
# Every constructor uses `*,` to force keyword arguments: readable call sites,
# no positional-order bugs.

# %%
try:
    sim.MultiArmedBandit(3, [-0.2, 0.0, 0.5], 42)
except TypeError as e:
    print("TypeError=", e)

# %% [markdown]
# ## Cell 7.3: Class-as-Parameter
#
# `.run_trials()` and `.run_ensemble()` take a strategy *class* plus a
# parameter dict, not an instance: a fresh strategy is built per trial so
# each trial gets its own seed.

# %%
print("strategy_class=sim.EpsilonGreedyStrategy (a class, not an instance)")
print("strategy_params={'epsilon': 0.2}")

# %% [markdown]
# # Part 8: Interactive Exploration

# %% [markdown]
# ## Cell 8.1: Introspect the Objects
#
# **Goal**:
# - Practice discovering an unfamiliar API with `dir()` and `help()`
#
# Questions to explore:
# - What happens if `n_trials=1`? Is `std_final` still meaningful?
# - What is the default value of `mu_range` in `BanditEnsemble`?
# - What type does `.select_machine()` return: `int` or `numpy.int64`?

# %%
print(
    "dir(sim.MultiArmedBandit)=",
    [a for a in dir(sim.MultiArmedBandit) if not a.startswith("_")],
)
help(sim.EpsilonGreedyStrategy.select_machine)

# %% [markdown]
# # Part 9: Summary
#
# ## The Mental Model
#
# - A `MultiArmedBandit` is a fixed, hidden-mean environment; a `Strategy`
#   decides which arm to pull without ever seeing those hidden means directly
# - `BanditExperiment` runs one bandit against one strategy for `n_coins`
#   pulls; everything above it just repeats this unit for statistics
# - `BanditSimulation` repeats an experiment across seeds (fixed `mu_values`);
#   `BanditEnsemble` repeats a simulation across random `mu_values` too
# - All orchestration classes take a strategy *class* and parameter dict, so
#   swapping policies never requires touching the environment or the
#   experiment loop
