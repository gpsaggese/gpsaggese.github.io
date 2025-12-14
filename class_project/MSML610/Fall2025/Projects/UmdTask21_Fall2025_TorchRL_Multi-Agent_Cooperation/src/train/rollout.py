"""Minimal rollout harness to sanity-check env + shared policy wiring (no learning)."""

from __future__ import annotations

import torch
from mpe2 import simple_spread_v3

from src.agent_policy.agent_policy import SharedMLPPolicy
from src.wrappers.wrapper import MPEWrapper

__all__ = ["run_rollout"]


def run_rollout(num_episodes: int = 3) -> None:
    """Run a few episodes using the shared policy (stateless sanity check)."""
    device = "cpu"
    env = MPEWrapper(device=device)

    obs = env.reset()
    num_agents, obs_dim = obs.shape

    tmp_env = simple_spread_v3.parallel_env()
    tmp_env.reset()
    sample_agent = tmp_env.agents[0]
    act_dim = tmp_env.action_space(sample_agent).n
    tmp_env.close()

    policy = SharedMLPPolicy(obs_dim, act_dim).to(device)

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_rewards = torch.zeros(num_agents, device=device)

        while not done:
            actions = policy.act(obs)
            obs, rewards, done = env.step(actions)
            ep_rewards += rewards

        print(f"Episode {ep+1}: rewards per agent = {ep_rewards.tolist()}")


if __name__ == "__main__":
    run_rollout()

