# MAC API Documentation

## A) Native APIs Used

### PettingZoo

**Environment Creation:**
```python
from pettingzoo.mpe import simple_reference_v3
env = simple_reference_v3.parallel_env(max_cycles=25, continuous_actions=False)
```

**Parallel Mode Signatures:**

- `reset(seed=None)` → `observations: dict[agent, np.ndarray]`
- `step(actions: dict[agent, action])` → `(observations, rewards, dones, truncations, infos)`
  - All return values are dicts keyed by agent name
  - `rewards[agent]`: float reward for that agent
  - `dones[agent]`: bool indicating episode termination
  - `truncations[agent]`: bool indicating episode truncation
  - `infos[agent]`: dict with auxiliary information

**Key Properties:**
- `env.possible_agents` → `['speaker_0', 'listener_0']`
- `env.observation_space(agent)` → gymnasium Space (Box)
- `env.action_space(agent)` → gymnasium Space (Discrete or Box)

### TorchRL

**PettingZoo Wrapper:**
```python
from torchrl.envs.libs.pettingzoo import PettingZooEnv

env = PettingZooEnv(
    task="mpe/simple_reference_v3",  # or environment instance
    parallel=True,                    # required for parallel envs
    categorical_actions=True,         # True for Discrete, False for Box
    seed=42,
    done_on_any=False
)
```

**Returns:** TorchRL wraps step/reset to return TensorDict instead of plain dicts. However, the underlying `._env` or `.env` provides access to the base PettingZoo environment, which we use for direct interaction in this implementation.

**TorchRL Utilities Used:** Only the PettingZoo wrapper for environment creation; no other TorchRL utilities are used. Rollout collection and training loop are custom implementations.

### Torch

**Discrete Actions:**
- `torch.distributions.Categorical(logits=logits)`
  - `dist.sample()` → action indices
  - `dist.log_prob(action)` → log probabilities
  - `dist.entropy()` → entropy values

**Continuous Actions:**
- `torch.distributions.Normal(mean, std)`
  - `dist.rsample()` → reparameterized samples
  - `dist.log_prob(action_raw)` → log probabilities (pre-squashing)
  - `dist.entropy()` → entropy values

**Action Squashing (Continuous):**
```python
action_raw = dist.rsample()                    # Sample from Gaussian
action = torch.tanh(action_raw)                 # Squash to [-1, 1]
log_prob = dist.log_prob(action_raw)
log_prob -= torch.log(1 - action.pow(2) + 1e-6)  # Tanh correction
action_scaled = action * scale + bias           # Scale to action bounds
```

---

## B) MAC Wrapper Layer

### Architecture

**Centralized A3C:**
- **Separate Actors:** One policy network per agent (`actors[agent_name]`)
- **Centralized Critic:** Single value network taking global state (concatenated observations)
- **Loss:** Policy gradient with shared advantage + value MSE + entropy regularization

### Core API Functions

#### Configuration

```python
def default_cfg() -> MacConfig
```
Returns default configuration with training/eval hyperparameters.

**MacConfig fields:** `num_iters`, `rollout_len`, `num_envs`, `lr`, `gamma`, `hidden_dim`, `actor_layers`, `critic_layers`, `entropy_coef`, `value_coef`, `eval_episodes`, `success_threshold`, `seed`, `device`, `checkpoint_dir`

#### Environment

```python
def make_env(cfg: MacConfig) -> env
```
Creates TorchRL-wrapped PettingZoo environment. Returns environment object.

```python
def get_agent_names(env) -> List[str]
```
Extracts agent names from environment. Returns `['speaker_0', 'listener_0']` for simple_reference.

#### Model Building

```python
def build_modules(cfg: MacConfig, env) -> (actors, critic, specs)
```
**Returns:**
- `actors`: `dict[agent_name, nn.Module]` - Actor network per agent
- `critic`: `nn.Module` - Centralized critic
- `specs`: `dict` - Metadata with keys:
  - `agent_names`: list of agent names
  - `obs_dims`: dict of observation dimensions per agent
  - `action_dims`: dict of action dimensions per agent
  - `action_types`: dict of `'discrete'` or `'continuous'` per agent
  - `action_bounds`: dict of `(low, high)` tuples for continuous actions

#### Rollout Collection

```python
def collect_rollout(cfg, env, actors, critic=None, device=None) -> batch
```
Collects `num_envs × rollout_len` transitions using current policies.

**Returns `batch` dict:**
```python
{
    'obs': {agent: Tensor[T, obs_dim]},        # Observations per agent
    'actions': {agent: Tensor[T, action_dim]}, # Actions per agent
    'logp': {agent: Tensor[T, 1]},             # Log probabilities per agent
    'entropy': {agent: Tensor[T, 1]},          # Entropies per agent
    'reward': Tensor[T],                       # Team rewards (sum across agents)
    'done': Tensor[T],                         # Done flags
    'state': Tensor[T, state_dim],             # Global state (concat obs)
    'values': Tensor[T, 1]                     # State values (if critic provided)
}
```
where `T = num_envs × rollout_len`

#### Loss Computation

```python
def compute_loss(cfg, batch, actors, critic) -> (loss, info)
```
Computes centralized A3C loss from rollout batch.

**Returns:**
- `loss`: scalar tensor for backprop
- `info`: dict with `'loss'`, `'actor_loss'`, `'critic_loss'`, `'entropy'`, `'mean_return'`, `'mean_value'`, `'mean_advantage'`

**Loss formula:**
```
actor_loss = -mean(sum_agents(log_prob_agent × advantage))
critic_loss = MSE(values, returns)
loss = actor_loss + value_coef × critic_loss - entropy_coef × entropy
```

#### Training

```python
def train(cfg: MacConfig) -> (ckpt_path, stats)
```
Full training loop for `num_iters` iterations.

**Returns:**
- `ckpt_path`: str path to saved checkpoint
- `stats`: dict with `'losses'`, `'returns'`, `'entropies'` lists

#### Checkpointing

```python
def load_checkpoint(path, device=None) -> ckpt_dict
```
Loads checkpoint from disk.

**Returns dict:**
```python
{
    'actors': dict[agent, nn.Module],  # Loaded actor networks
    'critic': nn.Module,                # Loaded critic
    'cfg': MacConfig,                   # Config used for training
    'specs': dict,                      # Model specs
    'env': env,                         # Fresh environment instance
    'iteration': int                    # Training iteration checkpoint was saved at
}
```

#### Evaluation

```python
def evaluate(cfg, ckpt_path, mode="normal"|"no_comm") -> metrics
```
Evaluates policy for `eval_episodes` episodes.

**Mode:**
- `"normal"`: standard policy execution
- `"no_comm"`: speaker actions forced to zero/null

**Returns dict:**
```python
{
    'success_rate': float,      # Fraction of successful episodes
    'comm_cost': float,         # Mean communication cost per step
    'comm_gain': float,         # success_normal - success_no_comm (0 if single mode)
    'comm_efficiency': float    # comm_gain / (comm_cost + eps)
}
```

**Success definition:** Episode reward > `-success_threshold` OR `info['is_success']`/`info['success']` if present.

**Communication cost:**
- Discrete: fraction of steps with nonzero speaker action
- Continuous: mean L2 norm of speaker action vector

```python
def evaluate_with_comparison(cfg, ckpt_path) -> metrics
```
Runs evaluation in both `"normal"` and `"no_comm"` modes, computes full metrics including `comm_gain` and `comm_efficiency`.

---

## Quickstart

```python
from mac_utils import default_cfg, train, evaluate_with_comparison

# Configure
cfg = default_cfg()
cfg.num_iters = 1000
cfg.rollout_len = 16
cfg.num_envs = 4

# Train
ckpt_path, stats = train(cfg)

# Evaluate
cfg.eval_episodes = 100
metrics = evaluate_with_comparison(cfg, ckpt_path)

print(f"Success Rate: {metrics['success_rate']:.3f}")
print(f"Comm Efficiency: {metrics['comm_efficiency']:.3f}")
```

**Acceptance Check:**
```python
cfg = default_cfg()
cfg.num_iters = 2
cfg.rollout_len = 8
cfg.num_envs = 2
train(cfg)  # Should complete in < 1 minute on CPU
```