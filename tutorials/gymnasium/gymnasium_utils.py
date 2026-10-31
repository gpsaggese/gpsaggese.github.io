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

    With no `policy`, sample actions from the action space seeded with `seed`, so
    that a rerun repeats the episode. A passed `policy` owns its own randomness.

    :param env: single environment, not a vector one
    :param seed: seed for `env.reset()` and the action space
    :param policy: map the last observation to an action, None samples randomly
    :param max_steps: raise if the episode is still running after these steps
    :return: the steps in order, and the total reward
    """
    obs, _info = env.reset(seed=seed)
    # `reset(seed=...)` does not seed the action space.
    if policy is None:
        env.action_space.seed(seed)
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
