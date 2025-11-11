# CleanRL integration and API

This project uses cleanRL's (https://docs.cleanrl.dev/) single-file RL implementations to train agents on a custom signal-generation environment. The repository contains a small `CleanRL_API/` folder with compact Rl agent scripts and utilities (PPO, SAC, buffers, etc.) copied over from cleanRL's repository.

What you'll find here

- `CleanRL_API/ppo.py`, `CleanRL_API/sac_continuous_action.py`: single-file, canonical cleanRL algorithm implementations adapted for our environment shapes.
- `rl_env.py`: the `SignalTesterEnv` environment used by the agents. This file now exposes a helper to register a gym id so cleanRL scripts (which expect an env id) can call `gym.make("SignalTester-v0")`.

Usage

The PPO and SAC scripts can be used by spawning a subprocess command. After training, models are automatically saved to `runs/{run_name}/`.
NOTE: debugging Usage

```python
cmd = [
    sys.executable,  # Python interpreter
    str(ppo_script),
    "--env-id", env_id_train,
    "--total-timesteps", str(total_timesteps),
    "--seed", str(SEED),
    "--hidden-size", "256",  # Large network for 138-dim state
    "--track", "False",  # Disable wandb tracking
    "--capture-video", "False",
]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
# Models saved to runs/{run_name}/actor.pth, qf1.pth, qf2.pth
```

Key points

- CleanRL expects either an environment id (string) or a vectorized environment built from a callable. To keep the cleanRL scripts untouched we register a short-lived gym id that returns a `SignalTesterEnv` instance with precomputed forecasts and news contexts. Use `register_cleanrl_env(...)` inside our experiment startup to make the id available to `gym.make()`.
- The provided PPO and SAC scripts are updated to now automatically use deeper networks as described below;This update made to handle higher dimensional input features. We can train with our environment by running the scripts with `--env_id SignalTester-v0 --num_envs 4 ...` after registering the id.
- The provided PPO and SAC scripts were updated store trained models to `runs/{run_name}/actor.pth`, `runs/{run_name}/qf1.pth`, `runs/{run_name}/qf2.pth` after training completes.
- `CleanRL_API/cleanrl_utils/`: small helpers (buffers, wrappers) used by the algorithms.

`SignalTesterEnv` exposes a 138-dimensional state (uncertainty cones + news context). Both `ppo.py` and `sac_continuous_action.py` use deeper networks with LayerNorm when `obs_dim > 60`: PPO uses a 4-layer architecture (default `--hidden_size 128`), while SAC uses deeper actor/critic networks (default `--hidden_size 256`). For even more capacity, use `--hidden_size 512`. The networks fall back to standard shallow architectures for simple environments, maintaining backward compatibility.

TODO: add more info and any diagrams here
