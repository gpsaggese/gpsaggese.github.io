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
# # Description
#
# - Train a DQN agent with RLlib to sell 100,000 SPY shares in 20 steps
#   within one trading day, minimizing the implementation shortfall (bps)
#   versus the arrival price
# - Calibrate the simulated market on NYSE TAQ 2015-2019 and test on paths
#   calibrated on 2020-2024, split by calm and stressed VIX regimes
# - Compare the agent with TWAP and with selling everything at once
#
# - References:
#   - Spec: `RLlib_Project_Description.md`, Project 1
#   - Almgren and Chriss (2000), "Optimal Execution of Portfolio Transactions"
#   - Almgren, Thum, Hauptmann, and Li (2005), "Direct Estimation of Equity
#     Market Impact"
#   - RLlib: https://docs.ray.io/en/latest/rllib/index.html

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

# System libraries.
import logging
from typing import Any, Callable, Dict, Tuple

# Third-party libraries.
import numpy as np
import pandas as pd
import ray

# %%
import helpers.hnotebook as hnotebook

import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.data_utils as rldautil
import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.rllib_utils as rlutils
import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.taq_utils as rltautil

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
rlutils.init_loggers(_LOG)

# Use IPython `display()` in Jupyter and fall back to `print()` otherwise.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# ## Calibrate the Market
#
# - Train: one calibration on all the 2015-2019 days
# - Test: one calibration per VIX regime on the 2020-2024 days
#   - A day is `calm` if the previous VIX close is below 20, otherwise
#     `stressed`
# - The 1,000 test episodes are split between the regimes in proportion to
#   their days
# - Only summary statistics are shown; the TAQ bars stay local

# %%
total_shares = rlutils.DEFAULT_CONFIG["total_shares"]
n_steps = rlutils.DEFAULT_CONFIG["n_steps"]
train_params = rldautil.calibrate(
    rltautil.load_spy_taq(2015, 2019), n_steps, total_shares
)
vix = rldautil.load_vix()
test_params = rldautil.calibrate_by_regime(
    rltautil.load_spy_taq(2020, 2024), vix, n_steps, total_shares
)
params = {"train 2015-2019": train_params}
params.update({f"test {k}": v for k, v in test_params.items()})
keys = ["n_days", "daily_vol", "adv", "half_spread_bps", "gamma", "eta"]
display(pd.DataFrame(params).loc[keys])

# %%
action = {"action_type": "twap_multiple"}
train_config = {**rldautil.to_env_config(train_params), **action}
test_configs = {
    k: {**rldautil.to_env_config(v), **action} for k, v in test_params.items()
}
n_test = 1000
n_days = {k: v["n_days"] for k, v in test_params.items()}
n_calm = round(n_test * n_days["calm"] / sum(n_days.values()))
regime_episodes = {"calm": n_calm, "stressed": n_test - n_calm}
print("test episodes:", regime_episodes)


# %% [markdown]
# ## Evaluate on the Test Regimes
#
# - Every policy runs on the same 1,000 test paths, so the win rate compares
#   each policy with TWAP episode by episode
# - All costs are in bps of the arrival value:
#   - `shortfall`: the true implementation shortfall, with the price noise
#   - `expected cost`: the same without the P&L of the price noise on the
#     shares still held, which compares the schedules precisely


# %%
def evaluate(
    policies: Dict[str, Callable],
    regime_configs: Dict[str, Dict[str, Any]],
    episodes: Dict[str, int],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Run each policy on the test paths and summarize them by regime.

    :param policies: policy name to policy for `rlutils.run_episodes()`
    :param regime_configs: regime name to environment config
    :param episodes: regime name to number of episodes
    :return: summary with one row per regime and policy, and the inventory
        paths of all the episodes of each policy
    """
    # Use the same seeds for every policy, away from the training seeds.
    seed = 10_000
    results = {
        name: rlutils.run_by_regime(policy, regime_configs, episodes, seed=seed)
        for name, policy in policies.items()
    }
    tables = {}
    for regime in list(episodes) + ["all"]:
        regimes = list(episodes) if regime == "all" else [regime]
        costs = {
            name: pd.concat([res[r][0] for r in regimes], ignore_index=True)
            for name, res in results.items()
        }
        tables[regime] = rlutils.summarize_episodes(costs)
    summary = pd.concat(tables, names=["regime", "policy"])
    inventory = {
        name: np.concatenate([res[r][1] for r in episodes])
        for name, res in results.items()
    }
    return summary, inventory


baselines = {
    "TWAP": lambda env, obs: rlutils.twap_shares(env),
    "sell-all": lambda env, obs: rlutils.sell_all_shares(env),
}
baseline_summary, baseline_inventory = evaluate(
    baselines, test_configs, regime_episodes
)
display(baseline_summary.round(3))

# %% [markdown]
# ## Train the Agent
#
# - `rlutils.train_dqn()` registers `ExecutionEnv` with `register_env`,
#   configures `DQNConfig` with 2 env runners, and calls `Algorithm.train()`
#   60 times (~60,000 env steps, ~3,000 episodes)
# - Each action sells a multiple (0 to 3x) of the TWAP pace, the remaining
#   inventory over the remaining steps, so TWAP is the action 1.0 at every step
# - Two agents differ only in the reward:
#   - `shortfall`: the spec reward, the negative shortfall of each step
#   - `expected_cost`: the same reward minus the P&L of the price shock on
#     the shares still held, which has zero mean whatever the agent does
# - Both rewards have the same expected episode return, but the price noise
#   (~38 bps std per episode) is ~100x the cost differences between schedules
#   (<1 bp), so the `shortfall` reward is mostly noise
# - Training varies from run to run, so each agent is trained with 3 seeds
#   and the results average over them
#   - Each trained agent is evaluated on the test paths right away, then
#     stopped to free the CPUs of its env runners


# %%
def train_and_evaluate(
    name: str,
    env_config: Dict[str, Any],
    regime_configs: Dict[str, Dict[str, Any]],
    seeds: Tuple[int, ...] = (0, 1, 2),
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, pd.DataFrame]]:
    """
    Train one DQN agent per seed and evaluate each one on the test paths.

    :param name: name of the agent in the summary
    :param env_config: training environment config
    :param regime_configs: regime name to test environment config
    :param seeds: training seeds
    :return: summary rows of the agent averaged over the seeds, with the std
        of the expected cost across seeds; inventory paths of all the seeds;
        learning curve of each seed
    """
    rows = []
    paths = []
    curves = {}
    for seed in seeds:
        algo, curves[f"seed {seed}"] = rlutils.train_dqn(env_config, seed=seed)
        policy = rlutils.get_greedy_policy(algo.get_module())
        summary, inventory = evaluate(
            {**baselines, name: policy}, regime_configs, regime_episodes
        )
        algo.stop()
        rows.append(summary.xs(name, level="policy"))
        paths.append(inventory[name])
    grouped = pd.concat(rows).groupby(level=0, sort=False)
    agent = grouped.mean()
    agent["expected cost std over seeds"] = grouped["expected cost"].std()
    agent.index = pd.MultiIndex.from_product(
        [agent.index, [name]], names=["regime", "policy"]
    )
    return agent, np.concatenate(paths), curves


# Start Ray with 4 CPUs: 1 for the driver and learner, 2 for env runners.
ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)
agent_rows = []
inventory = dict(baseline_inventory)
curves = {}
for reward in ("shortfall", "expected_cost"):
    name = f"DQN ({reward})"
    env_config = {**train_config, "reward": reward}
    rows, inventory[name], curves[name] = train_and_evaluate(
        name, env_config, test_configs
    )
    agent_rows.append(rows)

# %% [markdown]
# ### Learning Curves
#
# - Mean episode return per training iteration, including the exploration
#   episodes
# - One plot per agent, with one line per seed, since the `shortfall` curve
#   is ~100x noisier

# %%
for name, seed_curves in curves.items():
    print(name)
    rlutils.plot_learning_curves(seed_curves)

# %% [markdown]
# ## Evaluate the Policy
#
# - Mean and std of the shortfall, mean expected cost, and win rate against
#   TWAP over the 1,000 test episodes, split by VIX regime
# - The DQN rows average the 3 seeds; `expected cost std over seeds` shows
#   how much the expected cost changes from one training run to the next
# - A policy can win more than half the episodes against TWAP and still cost
#   more on average, so compare the expected cost, not only the win rate

# %%
regimes = list(regime_episodes) + ["all"]
summary = pd.concat([baseline_summary] + agent_rows).loc[regimes]
display(summary.round(3))

# %% [markdown]
# ## Visualize the Policy
#
# - Average fraction of the inventory left at each step, over all the test
#   episodes and, for the agents, all the seeds

# %%
rlutils.plot_inventory(inventory)

# %% [markdown]
# ## Order Size Sweep
#
# - 100,000 shares is ~0.14% of SPY's daily volume, so impact is small
# - Larger orders, as a percentage of the 2015-2019 ADV, make the schedule
#   matter more
# - Train 3 `expected_cost` agents per size, one per seed, then evaluate
#   them on the test paths

# %%
rows = {}
for pct_adv in (1.0, 5.0, 10.0):
    shares = round(pct_adv / 100 * train_config["adv"])
    env_config = {
        **train_config,
        "total_shares": shares,
        "reward": "expected_cost",
    }
    configs = {
        k: {**v, "total_shares": shares} for k, v in test_configs.items()
    }
    summary_base, _ = evaluate(baselines, configs, regime_episodes)
    summary_agent, _, _ = train_and_evaluate(
        "DQN (expected_cost)", env_config, configs
    )
    summary_size = pd.concat([summary_base, summary_agent])
    label = f"{pct_adv:.0f}% ADV ({shares / 1e6:.2f}M shares)"
    rows[label] = summary_size.loc["all"]
sweep = pd.concat(rows, names=["order", "policy"])
display(sweep.round(3))

# %%
ray.shutdown()

# %% [markdown]
# ## Summary
#
# - The market is calibrated on NYSE TAQ: volatility, volume, and spread
#   from the data, impact from Almgren et al. (2005)
# - With linear impact and no risk penalty, TWAP has the lowest expected
#   cost, so an agent can at best match it on average
# - The `expected_cost` reward removes the price noise from training without
#   changing the objective
# - The Almgren-Chriss variance penalty (bonus) is where an agent can beat
#   TWAP, since selling earlier lowers the risk
