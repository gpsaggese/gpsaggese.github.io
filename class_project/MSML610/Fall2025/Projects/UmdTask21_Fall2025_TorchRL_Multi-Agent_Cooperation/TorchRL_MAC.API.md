# TorchRL_MAC API Reference

This document describes the two-layer architecture for multi-agent cooperation on MPE environments using CTDE (Centralized Training, Decentralized Execution) with an A3C (Asynchronous Advantage Actor-Critic) implementation.

---

## Layer 1: Native API (Project Internals)

These modules provide the core MPE environment and policy infrastructure.

### `src/envs/mpe_env.py`

**Purpose:** Factory for PettingZoo MPE parallel environments.

**Key Function:**
```python
make_mpe_env(
    render_mode: str | None = None,
    N: int = 3,                    # number of agents
    local_ratio: float = 0.5,      # reward locality
    max_cycles: int = 25,          # max steps per episode
) -> ParallelEnv
```

**Returns:** PettingZoo `simple_spread_v3.parallel_env` with discrete actions (`continuous_actions=False`).

**Behavior:**
- Resets env with `seed=0` before returning
- Agents list: `env.agents = ["agent_0", "agent_1", "agent_2"]`
- Observation/action spaces accessed via `env.observation_space(agent)`, `env.action_space(agent)`

**Helpers:**
- `run_random_episode() -> Dict[str, float]`: sanity check with random actions, returns total rewards per agent
- `print_env_specs()`: prints spaces for first agent

---

### `src/wrappers/wrapper.py`

**Purpose:** Adapts PettingZoo dict-based I/O to fixed-order PyTorch tensors.

**Class:**
```python
MPEWrapper(device="cpu", **mpe_kwargs)
```

**Attributes:**
- `num_agents: int` — fixed number of agents
- `obs_dim: int` — observation dimension per agent
- `n_actions: int` — discrete action count (from `action_space.n`)
- `agents: List[str]` — stable agent ordering

**Methods:**
```python
reset(seed: int | None = None) -> Tensor
```
- **Returns:** `[num_agents, obs_dim]` float32
- Seeds both wrapper RNG and internal env

```python
step(actions: Tensor) -> Tuple[Tensor, Tensor, bool]
```
- **Input:** `actions` — `[num_agents]` int64, range `[0, n_actions)`
- **Returns:**
  - `obs` — `[num_agents, obs_dim]` float32
  - `rewards` — `[num_agents]` float32
  - `done_all` — bool (aggregates per-agent terminations and truncations)

**Done Semantics:**
- `done_all = all(dones[a] or truncs[a] for a in agents)`
- Episode ends when ALL agents terminate/truncate or max_cycles reached
- Fixed: uses `.item()` for fast tensor-to-bool conversion

**Validation:**
- Checks action shape `[num_agents]`
- Validates discrete action range if `n_actions` is known

---

### `src/agent_policy/agent_policy.py`

**Purpose:** Parameter-shared MLP policy for discrete actions.

**Class:**
```python
SharedMLPPolicy(obs_dim: int, act_dim: int, hidden_dims: Iterable[int] = (128, 64))
```

**Methods:**
```python
forward(obs: Tensor) -> Tensor
```
- **Input:** `[num_agents, obs_dim]`
- **Returns:** logits `[num_agents, act_dim]`

```python
act(obs: Tensor) -> Tensor
```
- **Input:** `[num_agents, obs_dim]`
- **Returns:** sampled actions `[num_agents]` int64

**Design:**
- Same network parameters used for all agents
- ReLU activations between linear layers
- No value head (actor-only)

---

### `src/train/rollout.py`

**Purpose:** Minimal integration test for env + policy wiring (no learning).

**Function:**
```python
run_rollout(num_episodes: int = 3) -> None
```

**Behavior:**
- Creates `MPEWrapper`, infers dims, builds `SharedMLPPolicy`
- Loops: `obs → policy.act → env.step → accumulate rewards`
- Prints per-agent episode rewards
- **No gradients or updates**

---

## Layer 2: Wrapper API (TorchRL_MAC_utils.py)

High-level training utilities and CTDE A2C baseline.

### Configuration Dataclasses

**`EnvConfig`**
```python
@dataclass
class EnvConfig:
    env_name: str = "simple_spread"
    seed: int = 0
    max_steps: int = 25
    device: str = "cpu"
```

**`TrainConfig`**
```python
@dataclass
class TrainConfig:
    gamma: float = 0.99
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    entropy_coef: float = 0.005  # Reduced for convergence
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_episodes: int = 500
    log_every: int = 25
    device: str | None = None  # None = auto-detect MPS/CPU
    n_workers: int = 4  # A3C: number of parallel workers
    t_max: int = 20  # A3C: max steps per worker update
```

**`RolloutBatch`**
```python
@dataclass
class RolloutBatch:
    obs: Tensor        # [T, num_agents, obs_dim]
    actions: Tensor    # [T, num_agents]
    rewards: Tensor    # [T, num_agents]
    dones: Tensor      # [T] (boolean as floats)
```
**Note:** Logprobs, values, and entropies are NOT stored during rollout. They are recomputed during update from stored obs/actions.

---

### Core Functions

**Device Detection**
```python
get_device(device: str | None = None) -> str
```
- Auto-detects best available device: MPS (Apple Silicon) > CPU
- If `device` is provided, returns it as-is
- Used throughout to enable MPS acceleration with CPU fallback

**Environment Creation**
```python
make_env(cfg: EnvConfig) -> MPEWrapper
```
- Seeds torch RNG and env
- Returns wrapper with `max_cycles=cfg.max_steps`
- Auto-detects device via `get_device(cfg.device)`

**Network Builders**
```python
build_shared_actor(obs_dim: int, n_actions: int, hidden_sizes=(128,128)) -> nn.Module
```
- Returns `SharedMLPPolicy` with specified hidden layers

```python
build_central_critic(joint_obs_dim: int, hidden_sizes=(128,128)) -> nn.Module
```
- Returns `CentralCritic` that maps concatenated observations → scalar value
- `joint_obs_dim = num_agents * obs_dim`

**Action Selection**
```python
select_actions(actor: nn.Module, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor]
```
- **Input:** `obs` — `[num_agents, obs_dim]`
- **Returns:**
  - `actions` — `[num_agents]` sampled from Categorical
  - `logprob_mean` — scalar (mean log-prob across agents)
  - `entropy_mean` — scalar (mean entropy across agents)

**Rollout Collection**
```python
collect_episode(wrapper, actor, env_cfg, max_steps=None) -> RolloutBatch
```
- Resets env (no fixed seed for diverse episodes)
- Loops until `done_all` or `max_steps` (defaults to `env_cfg.max_steps`)
- For each step:
  - Samples actions via `select_actions` (with `torch.no_grad()`)
  - Steps env and stores obs, actions, rewards, dones
  - **Does NOT store logprobs/values/entropies** (recomputed in update)
  - Uses `.item()` for fast tensor-to-bool conversion in done check
- Returns `RolloutBatch` with episode trajectory

**Returns and Advantages**
```python
compute_returns_advantages(
    rewards_team: Tensor,  # [T] (sum over agents)
    values: Tensor,        # [T]
    dones: Tensor,         # [T]
    gamma: float
) -> Tuple[Tensor, Tensor]
```
- Uses consistent team reward aggregation (sum over agents)
- Computes discounted returns backward from episode end
- Uses `.item()` for proper conditional check on dones
- `advantages = returns - values`
- **No GAE** (simple TD-λ with λ=1)

**A3C Update**
```python
a3c_update(actor, critic, batch, cfg, optim_actor, optim_critic) -> Dict[str, float]
```
- **Key difference from A2C:** Recomputes logits/logprobs/entropy/values from stored obs/actions
- Aggregates rewards as team reward (sum over agents) for consistent returns and logging
- Computes losses:
  - Policy: `-(logprobs * advantages.detach()).mean()`
  - Value: `MSE(values, returns)`
  - Entropy: `-entropy_coef * entropies.mean()`
- Total loss: `policy + value_coef * value + entropy_term`
- Clips gradients to `max_grad_norm`
- Returns metrics dict: `policy_loss`, `value_loss`, `entropy`, `team_reward`

**Training Loop (A3C)**
```python
train_ctde_a3c(env_cfg: EnvConfig, train_cfg: TrainConfig) -> Dict[str, List[float]]
```

**Parameters:**
- `env_cfg: EnvConfig` — Environment configuration (seed, max_steps, device)
- `train_cfg: TrainConfig` — Training configuration with A3C-specific fields:
  - `max_episodes: int` (e.g., 300) — Total episodes to run across all workers
  - `num_workers: int` (e.g., 4) — Number of parallel worker processes
  - `n_steps: int` (e.g., 10) — Steps per worker rollout before update
  - `device: str` (default "mps") — Global network device for gradient updates
  - `sync_every: int` (default 1) — Frequency of weight sync from global to workers

**Device Behavior:**
- `device="mps"` or `device="cpu"` specified in config applies to **global networks** (gradient computation)
- Worker processes **always use CPU** for rollout collection and local networks
- MPS incompatible with `torch.multiprocessing` shared memory, so workers forced to CPU for correctness
- This design: MPS handles gradients (fast), CPU workers handle environment interaction (stable)

**Workflow:**
1. Creates shared actor/critic networks on specified device
2. Spawns `num_workers` processes with `mp.set_start_method('spawn')`
3. Each worker:
   - Maintains local network copy on CPU
   - Collects `n_steps`-step rollout (no gradients)
   - Recomputes logits/values/entropy from stored obs/actions
   - Computes policy/value/entropy losses
   - Acquires lock, updates global networks, releases lock
   - Periodically syncs weights from global networks
4. Main process:
   - Monitors worker processes
   - Collects episode metrics from shared queue
   - Logs returns and losses
5. Returns `history` dict:
   - `episode_return`, `policy_loss`, `value_loss`, `entropy` — arrays for plotting

**Key Differences from A2C:**
- **Asynchronous:** Workers update global networks independently without waiting for others
- **CPU Workers:** All environment interaction on CPU; only gradient updates on specified device
- **No Replay Buffer:** On-policy learning from fresh worker rollouts
- **Lock-Protected Updates:** Each worker acquires lock before modifying global networks

**Multiprocessing:** Uses `spawn` method for compatibility with GPU memory sharing (MPS/CUDA).

---

## CTDE Concept

**Centralized Training, Decentralized Execution**

### Training Phase
- **Actor (decentralized):** Each agent's action depends only on its local observation `obs[i]`
- **Critic (centralized):** Value function sees all agents' observations concatenated: `joint_obs = [obs[0], obs[1], ..., obs[n-1]]`
- Centralized critic stabilizes training by providing better credit assignment across agents

### Execution Phase
- Only the actor is used
- Each agent acts independently using `actor(obs[i])`
- No communication or global state required at runtime

### Parameter Sharing
- All agents use the **same actor network** (symmetry assumption)
- Improves sample efficiency and generalization
- Reduces total parameter count

---

## Data Flow Diagram

```mermaid
graph LR
    A[reset] -->|obs: n×d| B[Actor per agent]
    B -->|actions: n| C[env.step]
    C -->|obs', rewards: n| D[Centralized Critic]
    C -->|rewards: n| E[Returns/Advantages]
    D -->|value| E
    E -->|advantages| F[Policy Loss]
    B -->|logprobs, entropy| F
    E -->|returns| G[Value Loss]
    D -->|values| G
    F --> H[Grad Update]
    G --> H
    H --> I[Next Episode]
    I --> A
    
    style A fill:#e1f5ff
    style D fill:#ffe1e1
    style H fill:#e1ffe1
```

**Key:**
- **Blue (reset):** Episode initialization with `seed`
- **Red (Critic):** Centralized component sees joint observations
- **Green (Update):** Gradient step updates both actor and critic

---

## Design Decisions

### A3C with Fixed Learning Issues
- Asynchronous parallel workers for better exploration
- Recompute logits/values during update (no stale gradients)
- Consistent team reward aggregation (sum over agents)
- Fixed done semantics: ALL agents must be done (not any)
- MPS device support with automatic CPU fallback
- No GAE (λ=1 returns) for simplicity and debuggability

### Debuggability
- Explicit tensor shapes in docstrings
- Scalar metrics returned from update step
- History dict for easy plotting
- Separate `collect_episode` and `a2c_update` for inspection

### Extensibility
- Configs are dataclasses (easy to serialize/override)
- Modular functions (swap critic, add GAE, vectorize envs)
- Protocol types for actor/critic allow custom implementations

---

## Common Patterns

### Quick Training Run
```python
from TorchRL_MAC_utils import EnvConfig, TrainConfig, train_ctde_a3c

env_cfg = EnvConfig(seed=42, max_steps=25, device=None)  # Auto-detect MPS/CPU
train_cfg = TrainConfig(
    max_episodes=300,    # Total episodes across all workers
    log_every=10,
    device="mps",        # Global networks device (workers always use CPU)
    num_workers=4,       # Number of parallel A3C workers
    n_steps=10,          # Steps per worker rollout
    lr=3e-4
)
history = train_ctde_a3c(env_cfg, train_cfg)

# Plot results
import matplotlib.pyplot as plt
plt.plot(history["episode_return"])
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("A3C Training")
plt.show()
```

### Custom Hyperparameters
```python
train_cfg = TrainConfig(
    gamma=0.95,
    lr_actor=1e-4,
    entropy_coef=0.005,
    max_episodes=500,
    num_workers=8,       # More workers for parallel exploration
    n_steps=15,          # Longer rollouts per worker
    sync_every=1,        # Sync weights after each worker update
    device="mps"         # Global networks on MPS (workers use CPU)
)
```

### Inspect Rollout
```python
from TorchRL_MAC_utils import make_env, build_shared_actor, collect_episode

env = make_env(env_cfg)
obs = env.reset()
n_agents, obs_dim = obs.shape

actor = build_shared_actor(obs_dim, env.n_actions)

# Note: collect_episode no longer needs critic (recomputed in update)
batch = collect_episode(env, actor, env_cfg)
print(batch.rewards.shape, batch.obs.shape)
print(f"Stored: obs, actions, rewards, dones (no logprobs/values)")
```

---

## Backward Compatibility

Legacy helpers from earlier notebooks remain available:
- `build_wrapped_env()` → alias for `MPEWrapper(...)`
- `infer_action_obs_dims()` → shape inference helper
- `make_shared_policy()` → builds actor only
- `run_stateless_rollout()` → no-training sanity check

These are **not** part of the main training flow but useful for quick prototyping.
