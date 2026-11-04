"""
Import as:

import tutorials.gymnasium.test.test_gymnasium_utils as tgtgyuti
"""

import contextlib
import io
import logging
import pprint
import warnings
from typing import Any, Dict, Optional, Tuple

import pytest

# Skip this test suite if gymnasium is not installed (skip for tutorials).
pytest.importorskip("gymnasium")

import gymnasium as gym
import gymnasium.utils.env_checker
import numpy as np

import helpers.hunit_test as hunitest
import tutorials.gymnasium.gymnasium_utils as tgygyuti

_LOG = logging.getLogger(__name__)


# #############################################################################
# _GridEnv
# #############################################################################


class _GridEnv(gym.Env):
    """
    Walk a 3-cell grid: the action moves one cell, and the last cell ends the
    episode with a reward of 1.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pos = 0
        self.observation_space = gym.spaces.Discrete(3)
        self.action_space = gym.spaces.Discrete(2)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        super().reset(seed=seed)
        self._pos = 0
        return self._pos, {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        self._pos += 1 if action == 1 else -1
        self._pos = int(np.clip(self._pos, 0, 2))
        terminated = self._pos == 2
        reward = 1.0 if terminated else 0.0
        return self._pos, reward, terminated, False, {}


# #############################################################################
# _GuessEnv
# #############################################################################


class _GuessEnv(gym.Env):
    """
    Guess a target drawn at reset in one step, with a reward of 1 for a hit.

    The reset draws `integers(25)` from the env generator, the same call that
    `Discrete(25).sample()` makes on the action space generator.
    """

    def __init__(self) -> None:
        super().__init__()
        self._target = 0
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(25)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        super().reset(seed=seed)
        self._target = int(self.np_random.integers(25))
        return 0, {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        reward = 1.0 if action == self._target else 0.0
        return 0, reward, True, False, {}


# #############################################################################
# Test_print_step
# #############################################################################


class Test_print_step(hunitest.TestCase):
    """
    Test `gymnasium_utils.print_step()`.
    """

    def helper(self, obs: Any, expected: str, **kwargs: Any) -> None:
        """
        Print a fixed step and compare the whole output.

        :param obs: observation to print
        :param expected: expected output
        :param kwargs: options of `print_step()`
        """
        # Run test.
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            tgygyuti.print_step(obs, 1.0, True, False, {"key": 1}, **kwargs)
        # Check outputs.
        actual = buffer.getvalue().rstrip("\n")
        self.assert_equal(actual, expected, dedent=True)

    def test1(self) -> None:
        """
        Test the full format of an observation with a shape.
        """
        # Prepare inputs.
        obs = np.array([1, 2])
        # Prepare outputs.
        expected = """
        observation (2,): [1 2]
        reward: 1.0
        terminated: True
        truncated: False
        info: {'key': 1}
        """
        # Run test.
        self.helper(obs, expected)

    def test2(self) -> None:
        """
        Test the compact format of an observation with a shape.
        """
        # Prepare inputs.
        obs = np.array([1, 2])
        # Prepare outputs.
        expected = """
        obs=(2,): [1 2] rew=1.0 term=True trunc=False info={'key': 1}
        """
        # Run test.
        self.helper(obs, expected, compact=True)

    def test3(self) -> None:
        """
        Test the label on an observation with no shape.
        """
        # Prepare inputs.
        obs = 3
        # Prepare outputs.
        expected = """
        run1 observation: 3
        run1 reward: 1.0
        run1 terminated: True
        run1 truncated: False
        run1 info: {'key': 1}
        """
        # Run test.
        self.helper(obs, expected, label="run1")


# #############################################################################
# Test_run_episode
# #############################################################################


class Test_run_episode(hunitest.TestCase):
    """
    Test `gymnasium_utils.run_episode()`.
    """

    def test1(self) -> None:
        """
        Test that the same seed repeats an episode of random actions.
        """
        # Prepare inputs.
        env = gym.make("CartPole-v1")
        # Run test: sample between the runs, so that an action space that is not
        # seeded again continues its stream.
        steps_1, total_reward_1 = tgygyuti.run_episode(env, seed=7)
        env.action_space.sample()
        steps_2, total_reward_2 = tgygyuti.run_episode(env, seed=7)
        # Check outputs.
        self.assertEqual(total_reward_1, total_reward_2)
        self.assert_equal(str(steps_1), str(steps_2))

    def test2(self) -> None:
        """
        Test an episode of a policy that ends exactly at `max_steps`.
        """
        # Prepare inputs.
        env = _GridEnv()
        seen = []

        def _policy(obs: int) -> int:
            seen.append(obs)
            return 1

        # Prepare outputs.
        expected = """
        [Step(obs=1, reward=0.0, terminated=False, truncated=False, info={}),
         Step(obs=2, reward=1.0, terminated=True, truncated=False, info={})]
        """
        # Run test.
        steps, total_reward = tgygyuti.run_episode(
            env, seed=0, policy=_policy, max_steps=2
        )
        # Check outputs.
        self.assert_equal(pprint.pformat(steps), expected, dedent=True)
        self.assertEqual(seen, [0, 1])
        self.assertEqual(total_reward, 1.0)

    def test3(self) -> None:
        """
        Test that a time limit ends the episode as truncated.
        """
        # Prepare inputs.
        env = gym.wrappers.TimeLimit(_GridEnv(), max_episode_steps=2)
        # Run test.
        steps, _ = tgygyuti.run_episode(env, seed=0, policy=lambda _obs: 0)
        # Check outputs.
        self.assertEqual(len(steps), 2)
        self.assertTrue(steps[-1].truncated)
        self.assertFalse(steps[-1].terminated)

    def test4(self) -> None:
        """
        Test that an episode needing one step more than `max_steps` raises.
        """
        # Prepare inputs.
        env = _GridEnv()
        # Prepare outputs.
        expected = "The episode must end within max_steps"
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            tgygyuti.run_episode(env, seed=0, policy=lambda _obs: 1, max_steps=1)
        # Check outputs.
        self.assertIn(expected, str(cm.exception))

    def test5(self) -> None:
        """
        Test that random actions do not copy the draws of the env.
        """
        # Prepare inputs.
        env = _GuessEnv()
        # Run test.
        hits = 0.0
        for seed in range(10):
            _, total_reward = tgygyuti.run_episode(env, seed=seed)
            hits += total_reward
        # Check outputs: copied draws hit all 10, independent ones about 0.4.
        self.assertLessEqual(hits, 2)


# #############################################################################
# Test_step_two_envs
# #############################################################################


class Test_step_two_envs(hunitest.TestCase):
    """
    Test `gymnasium_utils.step_two_envs()`.
    """

    def test1(self) -> None:
        """
        Test that a wrapper changes only what it transforms.
        """
        # Prepare inputs.
        env_a = gym.make("CartPole-v1")
        env_b = gym.wrappers.TransformReward(
            gym.make("CartPole-v1"), lambda reward: 2.0 * float(reward)
        )
        # Run test.
        step_a, step_b = tgygyuti.step_two_envs(env_a, env_b, action=0, seed=3)
        # Check outputs.
        self.assertTrue(np.array_equal(step_a.obs, step_b.obs))
        self.assertEqual(step_b.reward, 2.0 * float(step_a.reward))

    def test2(self) -> None:
        """
        Test that two envs sharing a base env raise `AssertionError`.
        """
        # Prepare inputs.
        base = gym.make("CartPole-v1")
        wrapped = gym.wrappers.TimeLimit(base, max_episode_steps=5)
        # Prepare outputs.
        expected = "The envs must not share a base env"
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            tgygyuti.step_two_envs(base, wrapped, action=0, seed=0)
        # Check outputs.
        self.assertIn(expected, str(cm.exception))


# #############################################################################
# Test_get_episode_returns
# #############################################################################


class Test_get_episode_returns(hunitest.TestCase):
    """
    Test `gymnasium_utils.get_episode_returns()`.
    """

    def test1(self) -> None:
        """
        Test that a policy that aims at every target hits every shot.
        """
        # Prepare inputs.
        grid_size = 3
        n_shots = 4
        n_episodes = 3
        env = tgygyuti.HitTargetEnv(grid_size, n_shots)
        # Prepare outputs.
        expected = "[4.0, 4.0, 4.0]"
        # Run test.
        actual = tgygyuti.get_episode_returns(
            env, n_episodes, seed=0, policy=lambda obs: obs
        )
        # Check outputs.
        self.assert_equal(str(actual), expected)

    def test2(self) -> None:
        """
        Test that episode `i` of random actions runs with seed `seed + i`.
        """
        # Prepare inputs.
        env = gym.make("CartPole-v1")
        seed = 7
        n_episodes = 3
        # Prepare outputs: the return of each seed, one episode at a time.
        expected = [
            tgygyuti.run_episode(env, seed=seed + idx)[1]
            for idx in range(n_episodes)
        ]
        # The returns must differ, or reusing one seed for every episode would
        # also pass.
        self.assertEqual(len(set(expected)), n_episodes)
        # Run test.
        actual = tgygyuti.get_episode_returns(env, n_episodes, seed=seed)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))


# #############################################################################
# TestHitTargetEnv
# #############################################################################


class TestHitTargetEnv(hunitest.TestCase):
    """
    Test `gymnasium_utils.HitTargetEnv`.
    """

    def test1(self) -> None:
        """
        Test that the env passes the Gymnasium env checker with no warning.
        """
        # Prepare inputs.
        grid_size = 5
        n_shots = 10
        env = tgygyuti.HitTargetEnv(grid_size, n_shots)
        # Run test and check outputs: `check_env()` raises on a failed check,
        # and every warning becomes an error.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gymnasium.utils.env_checker.check_env(env, skip_render_check=True)

    def test2(self) -> None:
        """
        Test that only the shot at the target scores, and that it scores 1.
        """
        # Prepare inputs.
        grid_size = 5
        n_shots = 10
        seeds = [0, 1, 2]
        env = tgygyuti.HitTargetEnv(grid_size, n_shots)
        # Run test: from each seeded reset, shoot once at every cell, and keep
        # the cells that score with their reward.
        actual = []
        expected = []
        for seed in seeds:
            target, _ = env.reset(seed=seed)
            scoring_cells = []
            for action in range(grid_size**2):
                env.reset(seed=seed)
                _, reward, _, _, _ = env.step(action)
                if reward != 0.0:
                    scoring_cells.append((action, reward))
            actual.append(scoring_cells)
            # Prepare outputs: the target is the only scoring cell.
            expected.append([(target, 1.0)])
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test3(self) -> None:
        """
        Test that a round of 3 shots ends as truncated, never as terminated.
        """
        # Prepare inputs.
        grid_size = 5
        n_shots = 3
        env = tgygyuti.HitTargetEnv(grid_size, n_shots)
        # Prepare outputs: `(terminated, truncated)` of each step.
        expected = "[(False, False), (False, False), (False, True)]"
        # Run test.
        steps, _ = tgygyuti.run_episode(env, seed=0, policy=lambda obs: obs)
        # Check outputs.
        actual = str([(step.terminated, step.truncated) for step in steps])
        self.assert_equal(actual, expected)
