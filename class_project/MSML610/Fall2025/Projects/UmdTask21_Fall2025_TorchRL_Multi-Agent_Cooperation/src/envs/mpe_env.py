# src/envs/mpe_env.py
from mpe2 import simple_spread_v3

def make_mpe_env(render_mode=None, N=3, local_ratio=0.5, max_cycles=25):
    """
    Creates and returns the parallel Simple Spread environment.
    """
    env = simple_spread_v3.parallel_env(
        N=N,
        local_ratio=local_ratio,
        max_cycles=max_cycles,
        continuous_actions=False,
        render_mode=render_mode,
    )
    env.reset(seed=0)
    return env

def run_random_episode():
    """
    Runs one full episode using random actions for all agents
    Sanity check
    """
    env = make_mpe_env()
    observations, infos = env.reset()

    total_rewards = {agent: 0.0 for agent in env.agents}

    # looping until there are no active agents left
    while env.agents:
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # reward accumulation
        for agent in rewards:
            total_rewards[agent] += float(rewards[agent])

    print("Episode finished. Total rewards:", total_rewards)
    env.close()

# policy making guidance helper function
def print_env_specs():
    env = make_mpe_env()
    print("Agents:", env.agents)

    example_agent = env.agents[0]
    obs_space = env.observation_space(example_agent)
    act_space = env.action_space(example_agent)

    print(f"Observation space for {example_agent}: {obs_space}")
    print(f"Action space for {example_agent}: {act_space}")

    env.close()

if __name__ == "__main__":
    run_random_episode()