"""
Utility functions for the gymnasium tutorial notebooks.

Import as:

import tutorials.gymnasium.gymnasium_utils as tgygyuti
"""

import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    SupportsFloat,
    Tuple,
)

import gymnasium as gym
import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Step result printing
# #############################################################################


def print_step(
    obs: Any,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    label: str = "",
    compact: bool = False,
) -> None:
    """
    Print the five return values of `env.step()` in a legible format.

    :param obs: observation from step
    :param reward: reward from step
    :param terminated: whether the episode ended naturally
    :param truncated: whether the episode was cut off externally
    :param info: info dict from step
    :param label: optional prefix label
    :param compact: print in a single compact line
    """
    prefix = f"{label} " if label else ""
    if compact:
        obs_repr = (
            f"{tuple(obs.shape)}: {obs}" if hasattr(obs, "shape") else str(obs)
        )
        print(
            f"{prefix}obs={obs_repr} rew={reward} "
            f"term={terminated} trunc={truncated} info={info}"
        )
    else:
        if hasattr(obs, "shape"):
            obs_shape = tuple(obs.shape)
            print(f"{prefix}observation {obs_shape}: {obs}")
        else:
            print(f"{prefix}observation: {obs}")
        print(f"{prefix}reward: {reward}")
        print(f"{prefix}terminated: {terminated}")
        print(f"{prefix}truncated: {truncated}")
        print(f"{prefix}info: {info}")


# #############################################################################
# Step
# #############################################################################


class Step(NamedTuple):
    """
    One `env.step()` result.
    """

    obs: Any
    reward: SupportsFloat
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


# #############################################################################
# Running environments
# #############################################################################


def run_episode(
    env: gym.Env,
    *,
    seed: int,
    policy: Optional[Callable[[Any], Any]] = None,
    max_steps: int = 10000,
) -> Tuple[List[Step], float]:
    """
    Run one seeded episode until it terminates or is truncated.

    With no `policy`, sample actions from the action space seeded from `seed`, so
    that a rerun repeats the episode. A passed `policy` owns its own randomness.

    :param env: single environment, not a vector one
    :param seed: seed for `env.reset()`, and the source of the action seed
    :param policy: map the last observation to an action, None samples randomly
    :param max_steps: raise if the episode is still running after these steps
    :return: the steps in order, and the total reward
    """
    obs, _info = env.reset(seed=seed)
    # `reset(seed=...)` does not seed the action space.
    if policy is None:
        # Use a child seed: `seed` itself would make the actions copy env draws.
        child_seed = np.random.SeedSequence(seed).spawn(1)[0]
        env.action_space.seed(int(child_seed.generate_state(1)[0]))
    steps: List[Step] = []
    done = False
    while not done:
        hdbg.dassert_lt(
            len(steps), max_steps, "The episode must end within max_steps"
        )
        action = env.action_space.sample() if policy is None else policy(obs)
        step = Step(*env.step(action))
        steps.append(step)
        obs = step.obs
        done = step.terminated or step.truncated
    total_reward = sum(float(step.reward) for step in steps)
    return steps, total_reward


def step_two_envs(
    env_a: gym.Env, env_b: gym.Env, *, action: Any, seed: int
) -> Tuple[Step, Step]:
    """
    Reset two envs with the same seed and apply the same action to both.

    :param env_a: first environment
    :param env_b: second environment, which must not share `env_a`'s base env
    :param action: action sent to both
    :param seed: seed for both resets
    :return: the step result of each env
    """
    # Stepping a shared base env would advance it twice.
    hdbg.dassert_is_not(
        env_a.unwrapped, env_b.unwrapped, "The envs must not share a base env"
    )
    env_a.reset(seed=seed)
    env_b.reset(seed=seed)
    step_a = Step(*env_a.step(action))
    step_b = Step(*env_b.step(action))
    return step_a, step_b


def get_episode_returns(
    env: gym.Env,
    n_episodes: int,
    *,
    seed: int,
    policy: Optional[Callable[[Any], Any]] = None,
) -> List[float]:
    """
    Run seeded episodes and return the total reward of each.

    Episode `i` runs with seed `seed + i`, so that a rerun repeats every episode,
    and two policies run with the same `seed` start each episode from the same
    reset.

    :param env: single environment, not a vector one
    :param n_episodes: number of episodes to run
    :param seed: seed of the first episode
    :param policy: map the last observation to an action, None samples randomly
    :return: the total reward of each episode, in order
    """
    hdbg.dassert_lte(1, n_episodes, "Run at least one episode")
    returns = []
    for idx in range(n_episodes):
        _, total_reward = run_episode(env, seed=seed + idx, policy=policy)
        returns.append(total_reward)
    return returns


# #############################################################################
# Rendering
# #############################################################################


def plot_frame(
    frame: np.ndarray,
    title: str,
    *,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Show an RGB frame returned by `env.render()`.

    :param frame: array of shape `(height, width, 3)`
    :param title: title of the plot
    :param figsize: size of the figure, None uses the notebook default
    """
    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(frame)
    ax.set_title(title)
    ax.axis("off")
    plt.show()


# #############################################################################
# Comparing policies
# #############################################################################


def plot_returns(
    returns_by_policy: Dict[str, List[float]],
    max_return: int,
    title: str,
    *,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Plot a histogram of the episode returns of each policy, side by side.

    :param returns_by_policy: episode returns by policy name
    :param max_return: largest possible return, the right end of the x-axis
    :param title: title of the plot
    :param figsize: size of the figure, None uses the notebook default
    """
    # Build one row per episode, labeled with its policy and the mean of that
    # policy, so that the legend shows the means.
    rows = []
    for name, returns in returns_by_policy.items():
        label = f"{name} (mean {np.mean(returns):.2f})"
        rows += [{"policy": label, "return": ret} for ret in returns]
    df = pd.DataFrame(rows)
    # Draw one bin per integer return, with the policies side by side.
    _, ax = plt.subplots(figsize=figsize)
    sns.histplot(
        data=df, x="return", hue="policy", discrete=True, multiple="dodge", ax=ax
    )
    ax.set_xlim(-0.5, max_return + 0.5)
    ax.set_xticks(range(max_return + 1))
    ax.set_xlabel("return per episode")
    ax.set_ylabel("episodes")
    ax.set_title(title)
    plt.show()


# #############################################################################
# HitTargetEnv
# #############################################################################


class HitTargetEnv(gym.Env):
    """
    Aim at a target that appears on a random cell of a square grid.

    - Observation: the index of the target cell, in `[0, grid_size**2)`
    - Action: the index of the aimed cell
    - Reward: 1 on a hit, 0 on a miss
    - Each shot draws the next target, so the shots are independent
    - A round is truncated after `n_shots` shots, never terminated: its length is
      a time limit, not a terminal state of the task
    """

    def __init__(self, grid_size: int, n_shots: int) -> None:
        """
        Build an env with `grid_size**2` cells.

        :param grid_size: number of cells on each side of the grid
        :param n_shots: number of shots in a round
        """
        super().__init__()
        hdbg.dassert_lte(1, grid_size, "The grid needs at least one cell")
        hdbg.dassert_lte(1, n_shots, "A round needs at least one shot")
        self._n_cells = grid_size**2
        self._n_shots = n_shots
        self.observation_space = gym.spaces.Discrete(self._n_cells)
        self.action_space = gym.spaces.Discrete(self._n_cells)
        # `reset()` sets the target and the shot count of a round.
        self._target = 0
        self._shot = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Start a round and draw its first target.

        :param seed: seed of `self.np_random`, None continues its stream
        :param options: unused
        :return: the first target cell, and an empty info dict
        """
        # Seed `self.np_random`, the only source of randomness of the env.
        super().reset(seed=seed)
        self._shot = 0
        self._target = int(self.np_random.integers(self._n_cells))
        return self._target, {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Score a shot at a cell, then draw the next target.

        :param action: index of the aimed cell
        :return: the next target cell, the reward, `terminated`, `truncated`,
            and an empty info dict
        """
        hdbg.dassert(
            self.action_space.contains(action),
            "The action must be a cell of the grid: %s",
            action,
        )
        # Score the shot against the current target, before drawing the next one.
        reward = 1.0 if action == self._target else 0.0
        self._shot += 1
        truncated = self._shot == self._n_shots
        self._target = int(self.np_random.integers(self._n_cells))
        return self._target, reward, False, truncated, {}
