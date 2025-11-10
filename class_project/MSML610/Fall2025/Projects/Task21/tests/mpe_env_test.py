from src.envs.mpe_env import make_mpe_env

def test_env_one_episode_runs():
    env = make_mpe_env()
    obs, infos = env.reset()

    assert isinstance(obs, dict)
    assert len(env.agents) > 0

    steps = 0
    max_steps = 10

    while env.agents and steps < max_steps:
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }
        obs, rewards, terminations, truncations, infos = env.step(actions)
        steps += 1

    env.close()

if __name__ == "__main__":
    test_env_one_episode_runs()