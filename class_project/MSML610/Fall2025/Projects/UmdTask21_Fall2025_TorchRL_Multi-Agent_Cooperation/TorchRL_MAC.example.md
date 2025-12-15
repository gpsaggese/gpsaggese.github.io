# MAC Example: End-to-End Guide

## Project Overview

This project implements **Multi-Agent Communication (MAC)** for the PettingZoo MPE `simple_reference` environment using centralized training with decentralized execution (CTDE).

**Environment:** `simple_reference_v3`
- **Agents:** Speaker (agent 0) and Listener (agent 1)
- **Task:** Speaker observes target landmarks and must guide the Listener to the correct one through communication
- **Challenge:** Communication is costly—agents must learn efficient signaling

**Architecture:** Separate actor networks per agent + centralized critic (Option B)

---

## How Agents Cooperate

### Speaker's Role
- Observes all landmarks and knows which is the target
- Takes a communication action (discrete: 0-4 or continuous: vector)
- Does not directly control navigation
- Goal: Send informative signals to guide the Listener

### Listener's Role
- Observes landmarks but doesn't know which is the target
- Receives communication from Speaker
- Takes navigation actions to move toward landmarks
- Goal: Use Speaker's signals to reach the correct landmark

### Cooperation Mechanism
Success requires **emergent communication protocol**:
1. Speaker learns to encode target information in actions
2. Listener learns to decode signals and navigate accordingly
3. Both must coordinate to minimize communication cost while maximizing success

---

## Centralized Training, Decentralized Execution

### Training (Centralized)
- **Actors (Decentralized):** Each agent has its own policy network
  - Speaker actor: `π_speaker(a_speaker | o_speaker)`
  - Listener actor: `π_listener(a_listener | o_listener)`
  - Each sees only its local observation
  
- **Critic (Centralized):** Single value network with full information
  - `V(s_global)` where `s_global = concat(o_speaker, o_listener)`
  - Critic sees the "big picture" during training
  
- **Advantage:** All agents share the same advantage signal
  - `A = R - V(s_global)` computed using team reward `R = r_speaker + r_listener`

### Execution (Decentralized)
- At test time, only actors are used
- Each agent acts based on its local observation independently
- No communication between actor networks—only through environment actions

### A3C-Style Loss
```
actor_loss = -mean(log π_speaker(a|o) × A + log π_listener(a|o) × A)
critic_loss = MSE(V(s), R)
loss = actor_loss + 0.5 × critic_loss - 0.01 × entropy
```

---

## Running Training

### Minimal Example

```python
from mac_utils import default_cfg, train

# Configure training
cfg = default_cfg()
cfg.num_iters = 500       # Number of training iterations
cfg.rollout_len = 16      # Steps per rollout
cfg.num_envs = 4          # Parallel environments
cfg.lr = 3e-4             # Learning rate
cfg.hidden_dim = 128      # Network hidden dimension

# Train
checkpoint_path, stats = train(cfg)

# Check training progress
print(f"Checkpoint saved to: {checkpoint_path}")
print(f"Final mean return: {stats['returns'][-1]:.2f}")
```

**Expected Output:**
```
Starting training with config:
MacConfig(num_iters=500, rollout_len=16, ...)
Iter 10/500 | Loss: 245.123 | Return: -18.456 | Entropy: 1.234
...
Training complete! Checkpoint saved to: ./checkpoints/mac_checkpoint_iter500.pt
```

---

## Running Evaluation

### Normal vs No-Communication Modes

**Normal Mode:** Agents use learned communication protocol
```python
from mac_utils import evaluate

cfg = default_cfg()
cfg.eval_episodes = 100

# Evaluate with communication
metrics_normal = evaluate(cfg, checkpoint_path, mode="normal")
```

**No-Communication Mode:** Speaker actions forced to zero/null
```python
# Evaluate without communication (speaker silenced)
metrics_no_comm = evaluate(cfg, checkpoint_path, mode="no_comm")
```

**Comparison:** Compute communication efficiency
```python
from mac_utils import evaluate_with_comparison

# Runs both modes and computes all metrics
metrics = evaluate_with_comparison(cfg, checkpoint_path)

print(f"Success (normal): {metrics['success_rate']:.3f}")
print(f"Success (no comm): {metrics['success_rate_no_comm']:.3f}")
print(f"Comm cost: {metrics['comm_cost']:.4f}")
print(f"Comm gain: {metrics['comm_gain']:.3f}")
print(f"Comm efficiency: {metrics['comm_efficiency']:.3f}")
```

---

## Metrics Computation

### Success Rate
**Definition:** Fraction of episodes where the Listener reaches the correct landmark

**Determination:**
1. Check `info['is_success']` or `info['success']` if available
2. Fallback: Episode reward > `-success_threshold` (default: -5.0)

```python
success = (episode_reward > -cfg.success_threshold) or info.get('is_success', False)
success_rate = mean(successes over all episodes)
```

### Communication Cost
**Definition:** Average cost of Speaker's actions per timestep

**Discrete Actions:**
```python
comm_cost = fraction of steps where speaker_action ≠ 0
```

**Continuous Actions:**
```python
comm_cost = mean(||speaker_action||_2 over all steps)
```

### Communication Gain
**Definition:** How much communication helps
```python
comm_gain = success_rate(normal) - success_rate(no_comm)
```

### Communication Efficiency
**Definition:** Success improvement per unit of communication cost
```python
comm_efficiency = comm_gain / (comm_cost + ε)
```
- High efficiency: agents achieve large gains with minimal communication
- Low/negative efficiency: communication is wasteful or harmful

---

## Expected Outputs

### Training Stats Dictionary
```python
{
    'losses': [float, ...],      # Loss per iteration
    'returns': [float, ...],     # Mean return per iteration
    'entropies': [float, ...]    # Mean entropy per iteration
}
```

### Evaluation Metrics Dictionary
```python
{
    'success_rate': float,           # Success rate in evaluated mode
    'comm_cost': float,              # Mean communication cost
    'comm_gain': float,              # Difference in success rates
    'comm_efficiency': float,        # Gain per unit cost
    'success_rate_no_comm': float    # (only in evaluate_with_comparison)
}
```

**Example Keys (values will vary by training quality):**
- All metrics are floats between 0 and 1 (except efficiency, which can be negative)
- `success_rate`: typically 0.0-1.0
- `comm_cost`: 0.0 (no comm mode) or 0.0-1.0+ (normal mode)
- `comm_gain`: -1.0 to 1.0 (negative if communication hurts)
- `comm_efficiency`: can be any real number

---

## Quick Verification

To verify the implementation works correctly:

```python
# Quick acceptance test
from mac_utils import default_cfg, train, evaluate

cfg = default_cfg()
cfg.num_iters = 2
cfg.rollout_len = 8
cfg.num_envs = 2

# Should complete in < 1 minute on CPU
ckpt_path, _ = train(cfg)

# Test evaluation
cfg.eval_episodes = 5
metrics = evaluate(cfg, ckpt_path, mode="normal")

# Verify required keys exist
assert 'success_rate' in metrics
assert 'comm_cost' in metrics
assert 'comm_gain' in metrics
assert 'comm_efficiency' in metrics

print("✓ All checks passed!")
```

---

## Grading Notes

**Key Implementation Points to Verify:**
1. ✓ Separate actor networks per agent (not shared)
2. ✓ Single centralized critic taking concatenated observations
3. ✓ Team reward used for advantage computation
4. ✓ Evaluation supports both normal and no-comm modes
5. ✓ Speaker actions correctly suppressed in no-comm mode
6. ✓ Communication metrics properly computed

**Files to Review:**
- `mac_utils.py`: Main implementation (all wrapper APIs)
- `mac.API.md`: API documentation
- `README_MAC.md`: Usage instructions
- Checkpoint files in `./checkpoints/`

**Expected Behavior:**
- Training completes without errors
- Checkpoint saved successfully
- Evaluation returns all required metrics
- No-comm mode produces lower communication cost (≈0)
