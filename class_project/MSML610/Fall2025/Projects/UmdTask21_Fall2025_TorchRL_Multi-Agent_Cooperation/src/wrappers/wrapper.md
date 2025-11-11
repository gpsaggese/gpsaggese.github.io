# `wrapper.py` — PyTorch-Facing Env Adapter

## Purpose
`MPEWrapper` adapts the PettingZoo **dict-based** parallel MPE env into clean **PyTorch tensors** with a **fixed agent ordering**. This makes training code (A2C/A3C) simple and vectorized. See the project goals and CTDE design in the README.

## Why it matters
- Converts `{agent: obs}` and `{agent: reward}` dicts → **batched tensors**.
- Holds a **stable `self.agents` order** so shapes are consistent across steps.
- Collapses per-agent termination into a single `done_all` for rollout loops.

## Key API
- `MPEWrapper(device="cpu")`  Boots an env from `make_mpe_env()` and snapshots the fixed agent list.
- `reset() -> obs_tensor`  Returns **[num_agents, obs_dim]** float32.
- `step(actions) -> (obs_tensor, rewards_tensor, done_all)`  `actions`: **[num_agents]** long → returns **obs [num_agents, obs_dim]**, **rewards [num_agents]**, **done_all: bool**.

## I/O shapes (typical)
- **Observations**: `[num_agents, obs_dim]` (float32)
- **Actions**: `[num_agents]` (long)
- **Rewards**: `[num_agents]` (float32)

## How it connects
- Consumed by **`rollout.run_rollout`** and future training loops.
- Upstream of **`SharedMLPPolicy`** (produces obs tensors; consumes policy actions).

## Common extensions
- Buffering of **last action** / **agent masks** for variable populations.
- Observation **normalization** and reward **scaling**.
- Vectorization over **multiple parallel envs** (n_envs > 1).
- Support for **continuous actions** (change policy + action conversion).