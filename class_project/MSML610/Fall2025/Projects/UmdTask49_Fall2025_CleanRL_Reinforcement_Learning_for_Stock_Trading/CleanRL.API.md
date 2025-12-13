# CleanRL integration and API

This project uses cleanRL's (https://docs.cleanrl.dev/) single-file RL implementations to train agents on a custom signal-generation environment. The repository contains a small `CleanRL_API/` folder with compact Rl agent scripts and utilities (PPO, SAC, buffers, etc.) copied over from cleanRL's repository.

What you'll find here

- `CleanRL_API/ppo_continuous_action.py`, `CleanRL_API/sac_continuous_action.py`: single-file, canonical cleanRL algorithm implementations adapted for our environment shapes.
- `rl_env.py`: the `SignalTesterEnv` environment used by the agents. This file now exposes a helper to register a gym id so cleanRL scripts (which expect an env id) can call `gym.make("SignalTester-v0")`.
- NOTE: both `ppo_continuous_action.py` and `sac_continuous_action.py` were sligthly updated to ensure they fit into our workflow.

Usage

The PPO and SAC scripts are designed to be imported and run directly from Python code, allowing for seamless integration with the data pipeline and environment registration.

```python
from CleanRL_API.sac_continuous_action import train as train_sac, Args as SACArgs
# or for PPO:
# from CleanRL_API.ppo_continuous_action import train as train_ppo, Args as PPOArgs

# Define training arguments
args = SACArgs(
    env_id="SignalTester-v0",
    total_timesteps=5000,
    policy_lr=3e-4,
    q_lr=1e-3,
    buffer_size=10000,
    gamma=0.99,
    tau=0.005,
    batch_size=256,
    learning_starts=1000,
    policy_frequency=2,
    target_network_frequency=1,
    alpha=0.2,
    autotune=True,
    run_name="my_sac_run",
    seed=42,
    hidden_size=256,  # Use 512+ for high-dim states
)

# Run training
# The train function returns the trained agent
agent = train_sac(args)

# Models are automatically saved to runs/{run_name}/
```

Key points

- CleanRL expects either an environment id (string) or a vectorized environment built from a callable. To keep the cleanRL scripts untouched we register a short-lived gym id that returns a `SignalTesterEnv` instance with precomputed forecasts and news contexts. Use `register_cleanrl_env(...)` inside our experiment startup to make the id available to `gym.make()`.
- The provided PPO and SAC scripts are updated to now automatically use deeper networks as described below;This update made to handle higher dimensional input features. We can train with our environment by running the scripts with `--env_id SignalTester-v0 --num_envs 4 ...` after registering the id.
- The provided PPO and SAC scripts were updated store trained models to `runs/{run_name}/actor.pth`, `runs/{run_name}/qf1.pth`, `runs/{run_name}/qf2.pth` after training completes.
- `CleanRL_API/cleanrl_utils/`: small helpers (buffers, wrappers) used by the algorithms.

`SignalTesterEnv` exposes a 138-dimensional state (uncertainty cones + news context). Both `ppo.py` and `sac_continuous_action.py` use deeper networks with LayerNorm when `obs_dim > 60`: PPO uses a 4-layer architecture (default `--hidden_size 128`), while SAC uses deeper actor/critic networks (default `--hidden_size 256`). For even more capacity, use `--hidden_size 512`. The networks fall back to standard shallow architectures for simple environments, maintaining backward compatibility.
