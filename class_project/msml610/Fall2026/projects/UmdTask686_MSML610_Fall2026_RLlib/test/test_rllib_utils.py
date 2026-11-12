"""
Import as:

import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.test.test_rllib_utils as ttrlut
"""

import logging
from typing import Any, Dict, Tuple

import gymnasium.utils.env_checker as genvche

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
