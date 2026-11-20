"""
Utility functions for the RLlib Optimal Trade Execution notebooks.

- `ExecutionEnv`: Gymnasium environment that liquidates a block of shares.
- TWAP and sell-all baselines.
- Training loop and learning-curve plot for RLlib algorithms.

Import as:

import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.rllib_utils as rlutils
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import helpers.hprint as hprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Environment
# #############################################################################


# Default environment parameters. `daily_vol`, `adv`, `eta`, and `gamma` are
# placeholders until the SPY calibration.
DEFAULT_CONFIG: Dict[str, Any] = {
    # Shares to liquidate.
    "total_shares": 100_000,
    # Trading steps in one day.
    "n_steps": 20,
    # Arrival price.
    "s0": 100.0,
    # Daily volatility of returns.
    "daily_vol": 0.01,
    # Average daily volume in shares.
    "adv": 1_000_000,
    # Temporary impact per unit of traded fraction of daily volume.
    "eta": 0.1,
    # Permanent impact per unit of traded fraction of daily volume.
    "gamma": 0.05,
    # Half the bid-ask spread paid on every share sold, in bps of the price.
    "half_spread_bps": 0.0,
    # Fractions of the remaining inventory that an action can sell.
    "action_fractions": (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
    # How an action sets the shares to sell:
    # - `fraction`: `action_fractions[a]` of the remaining inventory
    # - `twap_multiple`: `twap_multiples[a]` times the TWAP pace, the
    #   remaining inventory over the remaining steps
    "action_type": "fraction",
    # Multiples of the TWAP pace that an action can sell.
    "twap_multiples": (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
    # Reward of each step:
    # - `shortfall`: negative shortfall of the step versus the arrival price
    # - `expected_cost`: same, minus the P&L of the price shock on the
    #   remaining inventory, which has zero mean whatever the action
    "reward": "shortfall",
}

_REWARDS = ("shortfall", "expected_cost")
_ACTION_TYPES = ("fraction", "twap_multiple")


class ExecutionEnv(gym.Env):
    """
    Liquidate `total_shares` in `n_steps` on a simulated price path.

    At step t, selling `n_t` shares with traded fraction `f_t = n_t / adv`:
    - executes at `S_t * (1 - eta * f_t - half_spread)` (temporary impact and
      half the bid-ask spread)
    - moves the price to `S_t * (1 - gamma * f_t)` (permanent impact), then a
      log-normal shock with volatility `daily_vol / sqrt(n_steps)`

    The reward is the negative shortfall of the step in basis points of the
    arrival value `s0 * total_shares`. The episode ends after `n_steps`, and
    any remaining inventory is sold at the last step.

    With `reward="expected_cost"`, the reward also removes the P&L of the
    price shock on the inventory still held. The shortfall of an episode is
    the sum of the step costs plus this P&L, so both rewards have the same
    expected episode return, but the second one has no price noise. Whatever
    the reward, `info` reports both the true `shortfall_bps` and the
    `expected_cost_bps` without the price noise.

    Observation: [time left, inventory left, log(S_t / s0) / daily_vol].
    Action: index into `action_fractions` (fraction of the remaining
    inventory) or, with `action_type="twap_multiple"`, into `twap_multiples`
    (multiple of the TWAP pace).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Build the environment.

        :param config: parameters overriding `DEFAULT_CONFIG`; RLlib passes
            its `env_config` here
        """
        _LOG.debug(hprint.to_str("config"))
        # Fill the missing parameters from the defaults.
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        # Cast the config values to fixed types for the arithmetic below.
        self.total_shares = float(cfg["total_shares"])
        self.n_steps = int(cfg["n_steps"])
        self.s0 = float(cfg["s0"])
        self.daily_vol = float(cfg["daily_vol"])
        self.adv = float(cfg["adv"])
        self.eta = float(cfg["eta"])
        self.gamma = float(cfg["gamma"])
        self.half_spread = float(cfg["half_spread_bps"]) / 1e4
        self.reward = str(cfg["reward"])
        hdbg.dassert_in(self.reward, _REWARDS)
        self.action_type = str(cfg["action_type"])
        hdbg.dassert_in(self.action_type, _ACTION_TYPES)
        self.action_fractions = np.asarray(cfg["action_fractions"], dtype=float)
        self.twap_multiples = np.asarray(cfg["twap_multiples"], dtype=float)
        actions = (
            self.action_fractions
            if self.action_type == "fraction"
            else self.twap_multiples
        )
        # Scale the daily volatility to one step, assuming i.i.d. returns.
        self.step_vol = self.daily_vol / np.sqrt(self.n_steps)
        self.action_space = gym.spaces.Discrete(len(actions))
        # Time and inventory left are in [0, 1]; the price move is unbounded.
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -np.inf], dtype=np.float32),
            high=np.array([1.0, 1.0, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Start a new episode with the full inventory at the arrival price.

        :param seed: seed of the price path
        :param options: unused, required by the Gymnasium API
        :return: first observation and an empty info dict
        """
        _LOG.debug(hprint.to_str("seed"))
        # Seed `self.np_random`, which draws the price shocks.
        super().reset(seed=seed)
        # Track the episode state: time index, inventory, price, and shortfall.
        self.t = 0
        self.inventory = self.total_shares
        self.price = self.s0
        self.shortfall_bps = 0.0
        self.expected_cost_bps = 0.0
        # Build the first observation.
        obs = self._obs()
        _LOG.debug("return=%s", obs)
        return obs, {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Sell the shares selected by `action`.

        :param action: index into `action_fractions` or `twap_multiples`
        :return: observation, reward, terminated, truncated, info
        """
        _LOG.debug("action=%s", action)
        shares = self.action_to_shares(action)
        # Delegate the trade to `step_shares()` and log the result it returns.
        result = self.step_shares(shares)
        _LOG.debug("return=%s", result)
        return result

    def action_to_shares(self, action: int) -> float:
        """
        Convert an action index into shares to sell at the current step.

        :param action: index into `action_fractions` or `twap_multiples`
        :return: shares to sell, before clipping to the remaining inventory
        """
        _LOG.debug("t=%s action=%s", self.t, action)
        if self.action_type == "fraction":
            shares = self.action_fractions[int(action)] * self.inventory
        else:
            twap_pace = self.inventory / (self.n_steps - self.t)
            shares = self.twap_multiples[int(action)] * twap_pace
        _LOG.debug("return=%s", shares)
        return float(shares)

    def step_shares(
        self, shares: float
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Sell `shares` at the current step.

        Baselines call this directly to trade exact share counts.

        :param shares: shares to sell, clipped to the remaining inventory
        :return: observation, reward, terminated, truncated, info
        """
        _LOG.debug("t=%s shares=%s inventory=%s", self.t, shares, self.inventory)
        # Never sell more than the remaining inventory.
        shares = float(np.clip(shares, 0.0, self.inventory))
        # Sell everything left at the last step.
        if self.t == self.n_steps - 1:
            shares = self.inventory
        # Compute the execution price with the temporary impact and the
        # spread.
        frac = shares / self.adv
        exec_price = self.price * (1.0 - self.eta * frac - self.half_spread)
        # Measure the cost against the arrival price, in bps of the arrival
        # value.
        cost = (self.s0 - exec_price) * shares
        reward = -cost / (self.s0 * self.total_shares) * 1e4
        # Accumulate the shortfall: the sum of the step costs in bps.
        self.shortfall_bps -= reward
        self.inventory -= shares
        # Apply the permanent impact, then the price shock.
        impacted_price = self.price * (1.0 - self.gamma * frac)
        shock = self.np_random.standard_normal()
        # The `-0.5 * step_vol**2` drift makes the shock mean-one, so only the
        # impact moves the expected price.
        self.price = impacted_price * np.exp(
            self.step_vol * shock - 0.5 * self.step_vol**2
        )
        # Remove the shock P&L of the shares still held: a price rise lowers
        # the cost of the shares sold later, a fall raises it.
        shock_pnl = (self.price - impacted_price) * self.inventory
        expected_reward = (
            reward - shock_pnl / (self.s0 * self.total_shares) * 1e4
        )
        self.expected_cost_bps -= expected_reward
        if self.reward == "expected_cost":
            reward = expected_reward
        # Advance the time index before checking whether the episode ended.
        self.t += 1
        # Episodes end by termination only, never by truncation.
        terminated = self.t >= self.n_steps
        info = {
            "shares": shares,
            "exec_price": exec_price,
            "shortfall_bps": self.shortfall_bps,
            "expected_cost_bps": self.expected_cost_bps,
        }
        _LOG.debug("reward=%s info=%s", reward, info)
        return self._obs(), float(reward), terminated, False, info

    def _obs(self) -> np.ndarray:
        """
        Build the observation from the current state.

        :return: [time left, inventory left, log(S_t / s0) / daily_vol]
        """
        _LOG.debug(
            "t=%s inventory=%s price=%s", self.t, self.inventory, self.price
        )
        # Normalize time and inventory to [0, 1], as in `observation_space`.
        obs = np.array(
            [
                1.0 - self.t / self.n_steps,
                self.inventory / self.total_shares,
                # Express the price move in units of daily volatility.
                np.log(self.price / self.s0) / self.daily_vol,
            ],
            dtype=np.float32,
        )
        _LOG.debug("return=%s", obs)
        return obs


# #############################################################################
# Baselines
# #############################################################################


def twap_shares(env: ExecutionEnv) -> float:
    """
    Sell the remaining inventory evenly over the remaining steps.

    :param env: environment in the current state
    :return: shares to sell at this step
    """
    _LOG.debug("t=%s inventory=%s", env.t, env.inventory)
    # Spread the remaining inventory evenly over the remaining steps.
    shares = env.inventory / (env.n_steps - env.t)
    _LOG.debug("return=%s", shares)
    return shares


def sell_all_shares(env: ExecutionEnv) -> float:
    """
    Sell the whole inventory at the first step.

    :param env: environment in the current state
    :return: shares to sell at this step
    """
    _LOG.debug("t=%s inventory=%s", env.t, env.inventory)
    # Return the whole inventory, which is zero after the first step.
    shares = env.inventory
    _LOG.debug("return=%s", shares)
    return shares


def run_episodes(
    policy: Callable[[ExecutionEnv, np.ndarray], float],
    *,
    config: Optional[Dict[str, Any]] = None,
    n_episodes: int = 1000,
    seed: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Run a policy on `n_episodes` price paths.

    Policies run with the same `seed` see the same price shocks, so their
    shortfalls can be compared episode by episode.

    :param policy: function mapping the environment and the observation to
        shares to sell
    :param config: environment parameters overriding `DEFAULT_CONFIG`
    :param n_episodes: number of episodes
    :param seed: seed of the first episode; episode i uses `seed + i`
    :return: `shortfall_bps` and `expected_cost_bps` of each episode, and the
        fraction of the inventory left at each step, with shape
        `(n_episodes, n_steps + 1)`
    """
    _LOG.debug(hprint.to_str("policy config n_episodes seed"))
    env = ExecutionEnv(config)
    costs = np.empty((n_episodes, 2))
    inventory = np.empty((n_episodes, env.n_steps + 1))
    for i in range(n_episodes):
        # Use a different seed per episode so each one gets its own price path.
        obs, _ = env.reset(seed=seed + i)
        inventory[i, 0] = 1.0
        terminated = False
        while not terminated:
            obs, _, terminated, _, info = env.step_shares(policy(env, obs))
            inventory[i, env.t] = env.inventory / env.total_shares
        costs[i] = info["shortfall_bps"], info["expected_cost_bps"]
    df = pd.DataFrame(costs, columns=["shortfall_bps", "expected_cost_bps"])
    _LOG.debug("return=%s", df.shape)
    return df, inventory


def run_baseline(
    schedule: Callable[[ExecutionEnv], float],
    *,
    config: Optional[Dict[str, Any]] = None,
    n_episodes: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """
    Run a share schedule on `n_episodes` price paths.

    :param schedule: function mapping the environment to shares to sell
    :param config: environment parameters overriding `DEFAULT_CONFIG`
    :param n_episodes: number of episodes
    :param seed: seed of the first episode; episode i uses `seed + i`
    :return: shortfall in bps for each episode
    """
    _LOG.debug(hprint.to_str("schedule config n_episodes seed"))
    # A schedule is a policy that ignores the observation.
    costs, _ = run_episodes(
        lambda env, obs: schedule(env),
        config=config,
        n_episodes=n_episodes,
        seed=seed,
    )
    shortfalls = costs["shortfall_bps"].to_numpy()
    _LOG.debug("return=%s", shortfalls[:5])
    return shortfalls


def get_greedy_policy(
    module: Any,
) -> Callable[[ExecutionEnv, np.ndarray], float]:
    """
    Turn a trained RLlib `RLModule` into a policy for `run_episodes()`.

    :param module: module returned by `algo.get_module()`
    :return: policy selling the shares of the greedy action
    """
    _LOG.debug("module=%s", type(module).__name__)
    # Import here, so that the environment and the baselines work without
    # `torch`.
    import torch

    def _policy(env: ExecutionEnv, obs: np.ndarray) -> float:
        _LOG.debug("obs=%s", obs)
        batch = {"obs": torch.tensor(obs[None], dtype=torch.float32)}
        out = module.forward_inference(batch)
        # DQN returns the greedy action; PPO returns the action logits.
        if "actions" in out:
            action = int(out["actions"][0])
        else:
            action = int(torch.argmax(out["action_dist_inputs"], dim=-1)[0])
        shares = env.action_to_shares(action)
        _LOG.debug("return=%s", shares)
        return shares

    return _policy


def summarize_episodes(
    costs: Dict[str, pd.DataFrame], benchmark: str = "TWAP"
) -> pd.DataFrame:
    """
    Summarize the costs of policies run on the same price paths.

    :param costs: policy name to the costs of `run_episodes()`
    :param benchmark: policy that the win rate compares against
    :return: one row per policy with the mean and std of the shortfall, the
        mean expected cost (all in bps), and the fraction of episodes with a
        lower shortfall than `benchmark`
    """
    _LOG.debug(hprint.to_str("list(costs) benchmark"))
    hdbg.dassert_in(benchmark, costs)
    benchmark_shortfall = costs[benchmark]["shortfall_bps"].to_numpy()
    rows = {}
    for name, df in costs.items():
        shortfall = df["shortfall_bps"].to_numpy()
        rows[name] = {
            "shortfall mean": shortfall.mean(),
            "shortfall std": shortfall.std(),
            "expected cost": df["expected_cost_bps"].mean(),
            # Episodes are paired by seed, so compare them one by one.
            f"win rate vs {benchmark}": (shortfall < benchmark_shortfall).mean(),
        }
    df = pd.DataFrame(rows).T
    _LOG.debug("return=\n%s", df)
    return df


def run_by_regime(
    policy: Callable[[ExecutionEnv, np.ndarray], float],
    regime_configs: Dict[str, Dict[str, Any]],
    regime_episodes: Dict[str, int],
    *,
    seed: int = 0,
) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
    """
    Run a policy on the price paths of each regime.

    Each regime gets its own block of seeds, so the episodes of different
    regimes never share a price path.

    :param policy: as in `run_episodes()`
    :param regime_configs: regime name to environment parameters
    :param regime_episodes: regime name to number of episodes
    :param seed: seed of the first episode of the first regime
    :return: regime name to the output of `run_episodes()`
    """
    _LOG.debug(hprint.to_str("regime_episodes seed"))
    results = {}
    for name, n_episodes in regime_episodes.items():
        results[name] = run_episodes(
            policy,
            config=regime_configs[name],
            n_episodes=n_episodes,
            seed=seed,
        )
        seed += n_episodes
    _LOG.debug("return=%s", list(results))
    return results


def train_algo(algo: Any, n_iters: int) -> pd.DataFrame:
    """
    Call `algo.train()` `n_iters` times and collect the learning curve.

    :param algo: RLlib `Algorithm` built with `config.build_algo()`
    :param n_iters: number of training iterations
    :return: one row per iteration with the mean episode return, the env
        steps sampled so far, and the wall-clock time
    """
    _LOG.debug(hprint.to_str("n_iters"))
    # Accumulate one metrics row per training iteration.
    rows = []
    for _ in range(n_iters):
        result = algo.train()
        # The sampling metrics of the new API stack live under `env_runners`.
        rows.append(
            {
                "iteration": result["training_iteration"],
                "episode_return_mean": result["env_runners"].get(
                    "episode_return_mean", np.nan
                ),
                "env_steps": result["num_env_steps_sampled_lifetime"],
                "time_s": result["time_total_s"],
            }
        )
        _LOG.info("%s", rows[-1])
    # Index the learning curve by iteration.
    df = pd.DataFrame(rows).set_index("iteration")
    _LOG.debug("return=%s", df.shape)
    return df


def train_dqn(
    env_config: Dict[str, Any], *, n_iters: int = 60, seed: int = 0
) -> Tuple[Any, pd.DataFrame]:
    """
    Train a DQN agent on `ExecutionEnv`.

    :param env_config: environment parameters overriding `DEFAULT_CONFIG`
    :param n_iters: training iterations, each sampling ~1,000 env steps
    :param seed: seed of the env runners and the network initialization
    :return: trained `Algorithm` and the output of `train_algo()`
    """
    _LOG.debug(hprint.to_str("env_config n_iters seed"))
    # Import here, so that the environment and the baselines work without Ray.
    from ray.rllib.algorithms.dqn import DQNConfig
    from ray.tune.registry import register_env

    # The env runners build the environment by name in their own processes.
    register_env("execution_env", lambda cfg: ExecutionEnv(cfg))
    config = (
        DQNConfig()
        .environment("execution_env", env_config=env_config)
        # Sample one full episode per env runner between network updates.
        .env_runners(num_env_runners=2, rollout_fragment_length=20)
        .training(
            lr=5e-4,
            # Episodes have a fixed length, so the step costs are not
            # discounted: the return is minus the episode shortfall.
            gamma=1.0,
            train_batch_size_per_learner=128,
            replay_buffer_config={
                "type": "EpisodeReplayBuffer",
                "capacity": 100_000,
            },
            # Fill the buffer with 100 random episodes before learning.
            num_steps_sampled_before_learning_starts=2000,
            target_network_update_freq=1000,
            double_q=True,
            dueling=True,
        )
        .debugging(seed=seed)
    )
    algo = config.build_algo()
    curve = train_algo(algo, n_iters)
    _LOG.debug("return=%s", curve.shape)
    return algo, curve


def plot_learning_curves(curves: Dict[str, pd.DataFrame]) -> None:
    """
    Plot the mean episode return against the env steps sampled.

    :param curves: algorithm name to the output of `train_algo()`
    """
    _LOG.debug(hprint.to_str("list(curves)"))
    # Create one axis shared by all the curves.
    _, ax = plt.subplots(figsize=(7, 4))
    # Use env steps on the x-axis, since iterations differ across algorithms.
    for name, df in curves.items():
        ax.plot(
            df["env_steps"], df["episode_return_mean"], marker="o", label=name
        )
    ax.set_xlabel("Env steps sampled")
    ax.set_ylabel("Mean episode return")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()


def plot_inventory(inventory: Dict[str, np.ndarray]) -> None:
    """
    Plot the mean fraction of the inventory left at each step.

    :param inventory: policy name to the inventory paths of `run_episodes()`
    """
    _LOG.debug(hprint.to_str("list(inventory)"))
    _, ax = plt.subplots(figsize=(7, 4))
    for name, paths in inventory.items():
        ax.plot(paths.mean(axis=0), marker="o", label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean inventory left (fraction)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()
