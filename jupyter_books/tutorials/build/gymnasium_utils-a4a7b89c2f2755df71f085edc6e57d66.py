"""
gymnasium_utils.py

Utility functions for the gymnasium tutorial notebooks.

Import as:

import tutorials.gymnasium.gymnasium_utils as tgygyuti
"""

import logging

_LOG = logging.getLogger(__name__)


# #############################################################################
# Step result printing
# #############################################################################


def print_step(
    obs,
    reward,
    terminated,
    truncated,
    info,
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
