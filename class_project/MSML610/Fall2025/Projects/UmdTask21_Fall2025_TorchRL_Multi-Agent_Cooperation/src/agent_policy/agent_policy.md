# `agent_policy.py` — Shared Policy Network (Actors)

## Purpose
Defines a **parameter-shared policy network** (`SharedMLPPolicy`) that all agents use to choose discrete actions from their **local observations**. This matches the **CTDE** setup: actors are decentralized at execution time, while the critic (not shown here) can be centralized during training. See the project goals and CTDE design in the README.

## Why it matters
- **Encourages symmetry** across identical agents and improves sample efficiency.
- Provides a single place to adjust model capacity (layers/hidden size) and exploration behavior.
- Outputs **logits** → softmax → **Categorical distribution** for discrete actions in MPE (`simple_spread`).

## Key API
- `SharedMLPPolicy(obs_dim, act_dim, hidden_dims=[128, 64])`  Build an MLP that maps each agent's observation to action logits.
- `forward(obs) -> logits`  `obs`: tensor of shape **[num_agents, obs_dim]** → returns **[num_agents, act_dim]**.
- `act(obs) -> actions`  Samples integer actions per agent (**[num_agents]**) from a Categorical distribution.

## I/O Shapes (typical)
- **Input**: `obs` from `MPEWrapper.reset/step` → `[num_agents, obs_dim]` (float32)
- **Output**: `logits` → `[num_agents, act_dim]`, `actions` → `[num_agents]` (long)

## How it connects
- Consumes tensors produced by **`MPEWrapper`**.
- Returned actions are fed back to **`MPEWrapper.step(actions)`** during rollouts.
- Plug-in point for adding **entropy bonuses**, **layer norm**, **residuals**, or **attention** later.

## Common extensions
- Swap MLP for **CNN/Transformer/Attention over agents**.
- Add **temperature** or **epsilon-greedy** for exploration.
- Expose `eval()`/`train()` behavior for deterministic vs. stochastic inference.
- Parameterize **value head** here if you want a **shared actor-critic** (MAPPO-style).

## Minimal Usage
```python
from agent_policy import SharedMLPPolicy
policy = SharedMLPPolicy(obs_dim=25, act_dim=5)
actions = policy.act(obs_tensor)   # obs_tensor: [num_agents, obs_dim]
```