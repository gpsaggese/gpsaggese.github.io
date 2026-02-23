# Goal

This tutorial walks through a complete, runnable example of multi-agent reinforcement learning (MARL) in the Multi-Agent Particle Environment (MPE) using TorchRL. The emphasis is on:

- building an end-to-end MARL pipeline (env → policy/critic → training → evaluation)
- training agents to collaborate on a shared objective
- validating that communication is actually being used, not just "happening"
- documenting real engineering iteration: what failed, how we diagnosed it, and why the final setup was chosen

## Problem setup: MPE simple_reference

We use MPE `simple_reference` / `simple_reference_v3`, a classic communication task:

- Agents must coordinate to reach the correct goal/landmark.
- One agent can "speak" (send a message token).
- Another agent "listens" and moves based on its observation (which includes received communication).

This environment is ideal because it forces coordination + emergent communication: if messages are useful, performance should improve; if messages are random, agents fail.

## System design: CTDE (Centralized Training, Decentralized Execution)

Our training pipeline follows the standard CTDE pattern used in cooperative MARL:

### Decentralized execution

Each agent selects actions from its local observation (no access to full global state at inference).

### Centralized training signal

A centralized critic (value function) can use a global state (or concatenated observations) during training to reduce variance and improve stability.

### Why CTDE matters

- In MARL, the environment is non-stationary because other agents' policies change during learning.
- A centralized critic helps stabilize advantage estimation and reduces noisy learning signals.

## What we tried first: A3C (single-worker and multi-worker)

The project began by directly implementing A3C-style learning, because it is a canonical on-policy approach and matches the provided project description.

### Attempt 1: Single-worker A3C (CTDE)

We first built a single-worker A3C loop to validate:

- action sampling
- stepping TorchRL-wrapped PettingZoo envs
- centralized critic wiring
- advantage/return computation
- gradient updates

This step was important because it gave us a stable base to verify pipeline correctness before adding multi-process complexity.

### Attempt 2: Multi-worker A3C (asynchronous)

We then implemented multi-worker A3C (spawned processes, shared models, asynchronous-style updates). In practice, this caused stability issues that are common when combining:

- MARL + communication (hard exploration + credit assignment early)
- on-policy high-variance updates
- asynchronous / policy-lag effects
- macOS + notebooks constraints (spawn + device limitations)

We tried many tuning passes, but repeatedly observed failure modes where optimization proceeds (loss changes) without actual policy learning.

## Diagnostics: how we proved A3C wasn't learning

We did not treat "loss decreasing" as success. We inspected the actual coordination signals.

### 1) Success rate stayed at 0.0

In a representative run, evaluation success was flat:

- `success = 0.0` across evaluation episodes
- evaluation plot "Eval success" remained at 0.0 across training

This means: regardless of noise in returns, the agents are not achieving the goal.

### 2) Success debugging confirmed the policy is far from goals

We directly inspected distances to each agent's assigned goal landmark:

**Example debug outputs:**

```text
distances=[1.568, 1.887], threshold=0.35
distances=[2.157, 2.993], threshold=0.35
```

Because success requires all distances ≤ 0.35, these values show agents are not even close to reliably reaching goals.

### 3) Communication appeared random (high entropy + high change rate)

We measured message usage two ways:

- **Action-derived**: decode the `SAY` token from the discrete action index
  → `message_entropy`, `message_change_rate`
- **Observation-derived**: infer message token from the comm slice of the next observation
  → `message_entropy_obs`, `message_change_rate_obs`

In the failing A3C setup, both matched closely:

- `message_change_rate ≈ 0.89–0.91`
- `message_entropy ≈ 2.20–2.21`
- obs-derived metrics were consistent (`*_obs`)

**Interpretation:**

- A ~0.9 message change rate means the speaker changes message almost every step.
- Entropy near ~2.2 is close to a near-uniform distribution over ~9–10 symbols.
- Together, this indicates random / unstructured communication, not meaningful signaling.

We even added an automated warning when entropy didn't drop:

```text
[WARNING ep 200] Communication entropy not decreasing!
  Early (ep ~20): 2.199, Late (ep ~200): 2.201
  Agents may not be learning task-relevant messages
```

### 4) Returns alone were misleading

We observed occasional spikes in episode return, but they did not correlate with success. Since we experimented with reward shaping (distance-to-goal terms), raw returns can fluctuate without reflecting actual task completion.

For this task, we treat:

- success rate
- goal distance debug
- communication structure metrics

as the primary indicators of meaningful learning.

## Engineering iteration: what we changed while debugging

Before moving to the final setup, we invested significant effort in ensuring the system was correctly wired and that the failure was truly due to learning instability (not bugs).

### Communication wiring validation

We implemented a deterministic sanity check:

- Vary `SAY` while holding `MOVE` constant → comm slice in next obs must change
- Vary `MOVE` while holding `SAY` constant → comm slice should not change

This ensured action encoding/decoding and comm injection into observations were correct.

### Codec and comm consistency checks

We compared action-derived vs observation-derived comm metrics and flagged divergence:

- large mismatch suggests codec errors or comm not entering obs properly

### Hyperparameter tuning

We ran multiple tuning rounds on:

- learning rate
- entropy coefficient (including decay schedules)
- rollout horizon (`n_steps`)
- episode horizon (`max_cycles`)
- value loss coefficient and grad clipping

### Reward shaping experiments

We tried shaping using goal distance:

```python
reward = reward_base - α * goal_dist
```

to densify early learning signals.

We also validated shaping carefully using success + comm metrics, not returns alone.

### Behavior inspection

We printed decoded (`say`, `move`) tokens and visually inspected whether:

- messages stabilized
- movement became directed
- listener behavior changed under message manipulations

These checks gave us confidence the system wiring was correct and the limitation was primarily optimization stability for A3C in this MARL communication setting.

## Final approach: stable configuration that produced results

After repeatedly seeing A3C stability issues and unstructured communication, we transitioned to the current stable setup used for our final results.

**Key characteristics of the final approach:**

- end-to-end training remained on-policy and CTDE-aligned
- we prioritized stability, measurable progress, and interpretable communication behavior
- evaluation emphasized success and communication evidence, not just return curves

This final configuration is what the example notebook (`TorchRL_MAC.example.ipynb`) runs end-to-end and is the version we recommend as a starting point for future students.

### End-to-end training loop

```mermaid
flowchart LR
  RST[Reset env] --> OBS[Get obs and state]
  OBS --> POL[Actor policy]
  POL --> ACT[Sample actions]
  ACT --> STP[Step env]
  STP --> COL[Collect rewards done obs state]
  COL --> VAL[Central critic]
  VAL --> RET[Compute n step return]
  RET --> ADV[Compute advantage]
  ADV --> LOSS[Total loss<br/>policy value entropy]
  LOSS --> UPD[Optimizer step]
  UPD --> OBS
  ```

## Evaluation methodology: proving cooperation and communication

To evaluate cooperation and communication rigorously, we report:

### Core task metrics

- **Return**: sum of shaped rewards (useful for monitoring but not sufficient)
- **Success**: binary success based on all agents being within `success_dist` of their assigned goals
- **Goal distances**: inspected directly for debugging and interpretability

### Communication metrics (two independent sources)

#### Action-derived

- `message_entropy`
- `message_change_rate`

#### Observation-derived

- `message_entropy_obs`
- `message_change_rate_obs`

**Why both matter:**

- agreement between these indicates comm wiring is correct
- decreasing entropy/change-rate indicates emerging structure and stable signalling

## Expected outputs

When you run the example notebook end-to-end, you should see:

- training curves (return/loss/entropy depending on setup)
- evaluation success statistics
- message entropy + message change rate curves
- printed debug summaries for goal distances and comm metrics

## Takeaways

This project demonstrates a practical MARL workflow:

- implement the algorithm suggested by the prompt (A3C)
- measure learning using task-grounded metrics (success + distances)
- verify communication is real (entropy/change-rate + wiring checks + ablations)
- recognize instability patterns (loss changes without success)
- iterate with tuning, shaping, and debugging
- converge to a stable final setup that produces interpretable results