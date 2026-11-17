"""
Import as:

import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.test.test_rllib_utils as ttrlut
"""

import logging
from typing import Any, Dict, Tuple

import gymnasium.utils.env_checker as genvche
import pandas as pd
import torch

import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.rllib_utils as rlutils
import helpers.hunit_test as hunitest

_LOG = logging.getLogger(__name__)

# Deterministic prices: no volatility shock, no permanent impact.
_NO_NOISE_CONFIG = {**rlutils.DEFAULT_CONFIG, "daily_vol": 1e-12, "gamma": 0.0}


# #############################################################################
# Test_ExecutionEnv
# #############################################################################


class Test_ExecutionEnv(hunitest.TestCase):
    """
    Test `rllib_utils.ExecutionEnv` class.
    """

    def helper(
        self, env: rlutils.ExecutionEnv, action: int, seed: int
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        Run one episode taking the same action at every step.

        :param env: environment to run
        :param action: action taken at every step
        :param seed: seed of the price path
        :return: number of steps, total reward, info of the last step
        """
        env.reset(seed=seed)
        n_steps = 0
        total_reward = 0.0
        terminated = False
        # Step until the episode terminates, accumulating the reward.
        while not terminated:
            _, reward, terminated, _, info = env.step(action)
            n_steps += 1
            total_reward += reward
        return n_steps, total_reward, info

    def test1(self) -> None:
        """
        Check that the environment follows the Gymnasium API.
        """
        # Prepare inputs.
        env = rlutils.ExecutionEnv()
        skip_render_check = True
        # Run test and check outputs.
        genvche.check_env(env, skip_render_check=skip_render_check)

    def test2(self) -> None:
        """
        Check that holding until the end still sells all shares.
        """
        # Prepare inputs.
        env = rlutils.ExecutionEnv()
        # Action 0 sells nothing.
        action = 0
        seed = 0
        # Prepare outputs.
        expected = "n_steps=20 inventory=0.0"
        # Run test.
        n_steps, _, _ = self.helper(env, action, seed)
        # Check outputs.
        actual = f"n_steps={n_steps} inventory={env.inventory}"
        self.assert_equal(actual, expected)

    def test3(self) -> None:
        """
        Check that the episode return equals minus the shortfall.
        """
        # Prepare inputs.
        env = rlutils.ExecutionEnv()
        # Action 3 sells 20% of the remaining inventory.
        action = 3
        seed = 1
        # Run test.
        _, total_reward, info = self.helper(env, action, seed)
        # Check outputs.
        actual = f"{total_reward:.6f}"
        expected = f"{-info['shortfall_bps']:.6f}"
        self.assert_equal(actual, expected)

    def test4(self) -> None:
        """
        Check that the `expected_cost` reward removes the price noise.

        Without impact and spread, the expected cost is 0, so the episode
        return is 0 while the true shortfall is not.
        """
        # Prepare inputs.
        cfg = {
            "eta": 0.0,
            "gamma": 0.0,
            "half_spread_bps": 0.0,
            "reward": "expected_cost",
        }
        env = rlutils.ExecutionEnv(cfg)
        # Action 3 sells 20% of the remaining inventory.
        action = 3
        seed = 1
        # Prepare outputs.
        expected = (
            "total_reward=0.000000 expected_cost=0.000000 has_shortfall=True"
        )
        # Run test.
        _, total_reward, info = self.helper(env, action, seed)
        # Check outputs.
        # Add 0.0 to print -0.0 as 0.0.
        actual = (
            f"total_reward={round(total_reward, 6) + 0.0:.6f} "
            f"expected_cost={round(info['expected_cost_bps'], 6) + 0.0:.6f} "
            f"has_shortfall={abs(info['shortfall_bps']) > 1e-3}"
        )
        self.assert_equal(actual, expected)

    def test5(self) -> None:
        """
        Check that the `twap_multiple` action 1.0 sells at the TWAP pace.
        """
        # Prepare inputs.
        cfg = {
            "total_shares": 1000,
            "n_steps": 4,
            "action_type": "twap_multiple",
        }
        env = rlutils.ExecutionEnv(cfg)
        # Action 3 is the multiple 1.0 of the TWAP pace.
        action = 3
        # Prepare outputs.
        expected = "250.0 250.0 250.0 250.0"
        # Run test.
        env.reset(seed=0)
        shares = []
        terminated = False
        while not terminated:
            _, _, terminated, _, info = env.step(action)
            shares.append(info["shares"])
        # Check outputs.
        actual = " ".join(f"{s:.1f}" for s in shares)
        self.assert_equal(actual, expected)


# #############################################################################
# Test_run_baseline
# #############################################################################


class Test_run_baseline(hunitest.TestCase):
    """
    Test `rllib_utils.run_baseline()` function.
    """

    def helper(self, schedule: Any, n_slices: int) -> None:
        """
        Check the shortfall of `schedule` without price noise.

        Selling X shares in `n_slices` equal slices costs
        `eta * (X / n_slices) / adv` in bps.

        :param schedule: baseline schedule to run
        :param n_slices: number of equal slices `schedule` sells in
        """
        # Prepare inputs.
        cfg = _NO_NOISE_CONFIG
        n_episodes = 3
        # Prepare outputs.
        expected_bps = (
            cfg["eta"] * cfg["total_shares"] / n_slices / cfg["adv"] * 1e4
        )
        expected = str([f"{expected_bps:.6f}"] * n_episodes)
        # Run test.
        shortfall = rlutils.run_baseline(
            schedule, config=cfg, n_episodes=n_episodes
        )
        # Check outputs.
        actual = str([f"{x:.6f}" for x in shortfall])
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Check that sell-all costs `eta * X / adv` in bps.
        """
        # Prepare inputs.
        schedule = rlutils.sell_all_shares
        n_slices = 1
        # Run test.
        self.helper(schedule, n_slices)

    def test2(self) -> None:
        """
        Check that TWAP costs `eta * (X / n_steps) / adv` in bps.
        """
        # Prepare inputs.
        schedule = rlutils.twap_shares
        n_slices = _NO_NOISE_CONFIG["n_steps"]
        # Run test.
        self.helper(schedule, n_slices)

    def test3(self) -> None:
        """
        Check that the half spread adds to the sell-all cost.
        """
        # Prepare inputs.
        cfg = {**_NO_NOISE_CONFIG, "half_spread_bps": 1.5}
        n_episodes = 1
        # Prepare outputs.
        impact_bps = cfg["eta"] * cfg["total_shares"] / cfg["adv"] * 1e4
        expected = f"{impact_bps + 1.5:.6f}"
        # Run test.
        shortfall = rlutils.run_baseline(
            rlutils.sell_all_shares, config=cfg, n_episodes=n_episodes
        )
        # Check outputs.
        actual = f"{shortfall[0]:.6f}"
        self.assert_equal(actual, expected)

    def test4(self) -> None:
        """
        Check that TWAP has a lower mean shortfall than sell-all with noise.
        """
        # Prepare inputs.
        n_episodes = 200
        twap_schedule = rlutils.twap_shares
        sell_all_schedule = rlutils.sell_all_shares
        # Prepare outputs.
        expected = "True"
        # Run test.
        twap = rlutils.run_baseline(twap_schedule, n_episodes=n_episodes)
        sell_all = rlutils.run_baseline(
            sell_all_schedule, n_episodes=n_episodes
        )
        # Check outputs.
        actual = str(twap.mean() < sell_all.mean())
        self.assert_equal(actual, expected)


class Test_run_episodes(hunitest.TestCase):
    """
    Test `rllib_utils.run_episodes()` function.
    """

    def test1(self) -> None:
        """
        Check the inventory path of TWAP over 4 steps.
        """
        # Prepare inputs.
        cfg = {**_NO_NOISE_CONFIG, "n_steps": 4}
        n_episodes = 2
        # Prepare outputs.
        expected = "[[1.0, 0.75, 0.5, 0.25, 0.0], [1.0, 0.75, 0.5, 0.25, 0.0]]"
        # Run test.
        _, inventory = rlutils.run_episodes(
            lambda env, obs: rlutils.twap_shares(env),
            config=cfg,
            n_episodes=n_episodes,
        )
        # Check outputs.
        actual = str(inventory.round(6).tolist())
        self.assert_equal(actual, expected)


class _FakeModule:
    """
    Stand-in for an RLlib `RLModule` returning a fixed output.
    """

    def __init__(self, out: Dict[str, Any]) -> None:
        self.out = out

    def forward_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.out


class Test_get_greedy_policy(hunitest.TestCase):
    """
    Test `rllib_utils.get_greedy_policy()` function.
    """

    def helper(self, out: Dict[str, Any], expected: str) -> None:
        """
        Check the shares sold at the first step for a fixed module output.

        :param out: output of `forward_inference()`
        :param expected: expected shares
        """
        # Prepare inputs.
        env = rlutils.ExecutionEnv()
        obs, _ = env.reset(seed=0)
        policy = rlutils.get_greedy_policy(_FakeModule(out))
        # Run test.
        shares = policy(env, obs)
        # Check outputs.
        actual = f"{shares:.1f}"
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Check a DQN output: action 2 sells 10% of 100,000 shares.
        """
        # Prepare inputs.
        out = {"actions": torch.tensor([2])}
        # Prepare outputs.
        expected = "10000.0"
        # Run test.
        self.helper(out, expected)

    def test2(self) -> None:
        """
        Check a PPO output: the highest logit, action 4, sells 30%.
        """
        # Prepare inputs.
        out = {"action_dist_inputs": torch.tensor([[0, 0, 1, 0, 5, 0, 0, 0.0]])}
        # Prepare outputs.
        expected = "30000.0"
        # Run test.
        self.helper(out, expected)


class Test_summarize_episodes(hunitest.TestCase):
    """
    Test `rllib_utils.summarize_episodes()` function.
    """

    def test1(self) -> None:
        """
        Check the shortfall, expected cost, and paired win rate against TWAP.
        """
        # Prepare inputs.
        costs = {
            "TWAP": pd.DataFrame(
                {
                    "shortfall_bps": [1.0, 2.0, 3.0, 4.0],
                    "expected_cost_bps": [0.5, 0.5, 0.5, 0.5],
                }
            ),
            "agent": pd.DataFrame(
                {
                    "shortfall_bps": [0.0, 3.0, 2.0, 5.0],
                    "expected_cost_bps": [0.4, 0.8, 0.6, 1.0],
                }
            ),
        }
        # Prepare outputs.
        expected = r"""
               shortfall mean  shortfall std  expected cost  win rate vs TWAP
        TWAP              2.5       1.118034            0.5               0.0
        agent             2.5       1.802776            0.7               0.5
        """
        # Run test.
        df = rlutils.summarize_episodes(costs)
        # Check outputs.
        actual = df.round(6).to_string()
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)
