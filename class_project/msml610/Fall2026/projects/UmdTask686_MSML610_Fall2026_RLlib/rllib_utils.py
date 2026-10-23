"""
Utility functions for the RLlib Optimal Trade Execution notebooks.

- `ExecutionEnv`: Gymnasium environment that liquidates a block of shares.
- TWAP and sell-all baselines.

Import as:

import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.rllib_utils as rlutils
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

import helpers.hprint as hprint

_LOG = logging.getLogger(__name__)


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
    # Fractions of the remaining inventory that an action can sell.
    "action_fractions": (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
}


class ExecutionEnv(gym.Env):
    """
    Liquidate `total_shares` in `n_steps` on a simulated price path.

    At step t, selling `n_t` shares with traded fraction `f_t = n_t / adv`:
    - executes at `S_t * (1 - eta * f_t)` (temporary impact)
    - moves the price to `S_t * (1 - gamma * f_t)` (permanent impact), then a
      log-normal shock with volatility `daily_vol / sqrt(n_steps)`

    The reward is the negative shortfall of the step in basis points of the
    arrival value `s0 * total_shares`. The episode ends after `n_steps`, and
    any remaining inventory is sold at the last step.

    Observation: [time left, inventory left, log(S_t / s0) / daily_vol].
    Action: index into `action_fractions` (fraction of the remaining inventory).
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
        self.total_shares = float(cfg["total_shares"])
        self.n_steps = int(cfg["n_steps"])
        self.s0 = float(cfg["s0"])
        self.daily_vol = float(cfg["daily_vol"])
        self.adv = float(cfg["adv"])
        self.eta = float(cfg["eta"])
        self.gamma = float(cfg["gamma"])
        self.action_fractions = np.asarray(cfg["action_fractions"], dtype=float)
        # Scale the daily volatility to one step, assuming i.i.d. returns.
        self.step_vol = self.daily_vol / np.sqrt(self.n_steps)
        # One action per fraction of the remaining inventory.
        self.action_space = gym.spaces.Discrete(len(self.action_fractions))
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
        self.t = 0
        self.inventory = self.total_shares
        self.price = self.s0
        self.shortfall_bps = 0.0
        return self._obs(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Sell the fraction of the remaining inventory selected by `action`.

        :param action: index into `action_fractions`
        :return: observation, reward, terminated, truncated, info
        """
        shares = self.action_fractions[int(action)] * self.inventory
        return self.step_shares(shares)

    def step_shares(
        self, shares: float
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Sell `shares` at the current step.

        Baselines call this directly to trade exact share counts.

        :param shares: shares to sell, clipped to the remaining inventory
        :return: observation, reward, terminated, truncated, info
        """
        # Use lazy formatting since this runs at every step of training.
        _LOG.debug("t=%s shares=%s inventory=%s", self.t, shares, self.inventory)
        shares = float(np.clip(shares, 0.0, self.inventory))
        # Sell everything left at the last step.
        if self.t == self.n_steps - 1:
            shares = self.inventory
        # Compute the execution price with the temporary impact.
        frac = shares / self.adv
        exec_price = self.price * (1.0 - self.eta * frac)
        # Measure the cost against the arrival price, in bps of the arrival
        # value.
        cost = (self.s0 - exec_price) * shares
        reward = -cost / (self.s0 * self.total_shares) * 1e4
        self.shortfall_bps -= reward
        self.inventory -= shares
        # Apply the permanent impact, then the price shock.
        shock = self.np_random.standard_normal()
        self.price *= (1.0 - self.gamma * frac) * np.exp(
            self.step_vol * shock - 0.5 * self.step_vol**2
        )
        self.t += 1
        terminated = self.t >= self.n_steps
        info = {
            "shares": shares,
            "exec_price": exec_price,
            "shortfall_bps": self.shortfall_bps,
        }
        _LOG.debug("reward=%s info=%s", reward, info)
        return self._obs(), float(reward), terminated, False, info

    def _obs(self) -> np.ndarray:
        """
        Build the observation from the current state.

        :return: [time left, inventory left, log(S_t / s0) / daily_vol]
        """
        return np.array(
            [
                1.0 - self.t / self.n_steps,
                self.inventory / self.total_shares,
                # Express the price move in units of daily volatility.
                np.log(self.price / self.s0) / self.daily_vol,
            ],
            dtype=np.float32,
        )


# #############################################################################
# Baselines
# #############################################################################


def twap_shares(env: ExecutionEnv) -> float:
    """
    Sell the remaining inventory evenly over the remaining steps.

    :param env: environment in the current state
    :return: shares to sell at this step
    """
    return env.inventory / (env.n_steps - env.t)


def sell_all_shares(env: ExecutionEnv) -> float:
    """
    Sell the whole inventory at the first step.

    :param env: environment in the current state
    :return: shares to sell at this step
    """
    return env.inventory


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
    env = ExecutionEnv(config)
    shortfalls = np.empty(n_episodes)
    for i in range(n_episodes):
        # Use a different seed per episode so each one gets its own price path.
        env.reset(seed=seed + i)
        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step_shares(schedule(env))
        shortfalls[i] = info["shortfall_bps"]
    _LOG.debug("return=%s", shortfalls[:5])
    return shortfalls
