# src/envs/mpe_env.py
"""Environment factory and quick sanity helpers for the MPE simple_spread task."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from mpe2 import simple_spread_v3

__all__ = ["make_mpe_env", "run_random_episode", "print_env_specs"]


def make_mpe_env(
    render_mode: str | None = None,
    N: int = 3,
    local_ratio: float = 0.5,
    max_cycles: int = 25,
):
    """Create and return the PettingZoo parallel simple_spread environment."""
    env = simple_spread_v3.parallel_env(
        N=N,
        local_ratio=local_ratio,
        max_cycles=max_cycles,
        continuous_actions=False,
        render_mode=render_mode,
    )
    env.reset(seed=0)
    return env


def run_random_episode() -> Dict[str, float]:
    """Run one random-policy episode as a manual sanity check (not executed on import)."""
    env = make_mpe_env()
    observations, infos = env.reset()

    total_rewards: Dict[str, float] = {agent: 0.0 for agent in env.agents}

    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

        for agent in rewards:
            total_rewards[agent] += float(rewards[agent])

    env.close()
    return total_rewards


def print_env_specs() -> None:
    """Print observation and action space details for one agent (manual helper)."""
    env = make_mpe_env()
    print("Agents:", env.agents)

    example_agent = env.agents[0]
    obs_space = env.observation_space(example_agent)
    act_space = env.action_space(example_agent)

    print(f"Observation space for {example_agent}: {obs_space}")
    print(f"Action space for {example_agent}: {act_space}")

    env.close()


if __name__ == "__main__":
    rewards = run_random_episode()
    print("Episode finished. Total rewards:", rewards)