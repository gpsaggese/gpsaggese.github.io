# `rollout.py` — Sanity-Check Episodes (No Learning)

## Purpose
`run_rollout` wires together **`MPEWrapper`** and **`SharedMLPPolicy`** to run a few episodes and verify that observation → policy → action → env → reward **loops correctly**. No gradients or updates here. See the project goals and CTDE design in the README.

## Why it matters
- Quick **integration test** before implementing advantage computation, losses, and updates.
- Prints **per-agent episodic rewards** to confirm shapes and termination logic.

## Key API
- `run_rollout(num_episodes=3)`
  - Creates `MPEWrapper` and a `SharedMLPPolicy`
  - Loops: `obs → policy.act → env.step → accumulate rewards`

## I/O shapes (recap)
- `obs`: `[num_agents, obs_dim]`
- `actions`: `[num_agents]`
- `rewards`: `[num_agents]`
- `done`: bool (episode termination)

## How it connects
- Serves as the **starting point** for adding A2C/A3C: GAE, entropy bonus, value loss, and optimizer steps.
- Replace random-seeded policy with a **trained** one to evaluate learning progress.

## Common extensions
- Log with **TensorBoard/W&B**.
- Add **seed loops** and average statistics.
- Save **GIFs/videos** using PettingZoo rendering for demos.