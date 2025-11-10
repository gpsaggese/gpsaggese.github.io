from mpe2 import simple_spread_v3

import torch

from src.train.mpe_wrapper import MPEWrapper
from src.train.policy import SharedMLPPolicy


def run_rollout(num_episodes=3):
    """
    Run a few episodes using the shared policy.
    NOTE: No learning is happening here yet.
    This is just to prove all components connect correctly.
    """
    device = "cpu"
    env = MPEWrapper(device=device)

    # Get obs_dim from one reset
    obs = env.reset()
    num_agents, obs_dim = obs.shape

    # Get act_dim from raw env once
    tmp_env = simple_spread_v3.parallel_env()
    tmp_env.reset()
    sample_agent = tmp_env.agents[0]
    act_dim = tmp_env.action_space(sample_agent).n

    policy = SharedMLPPolicy(obs_dim, act_dim).to(device)

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_rewards = torch.zeros(num_agents, device=device)

        while not done:
            actions = policy.act(obs)          # [num_agents]
            obs, rewards, done = env.step(actions)
            ep_rewards += rewards

        print(f"Episode {ep+1}: rewards per agent = {ep_rewards.tolist()}")


if __name__ == "__main__":
    run_rollout()

