# Project 3 — Multi-Agent Cooperation (TorchRL + MPE)

## What You’re Actually Building (in Plain English)

**Goal:**  
Design a system of **multiple agents that must work together** (not just coexist) to achieve a shared objective in a simple 2D world.

**Environment:**  
Use **MPE (Multi-Agent Particle Environment)** — think of “little robots” moving in a plane with simple physics and **partial observability**.

**Learning Approach:**  
Apply **Centralized Training, Decentralized Execution (CTDE)**.

- **During training:**  
  A single **centralized critic** sees all agents’ observations (and optionally actions) to learn a good value function and provide better credit assignment.

- **During execution:**  
  Each agent uses its own policy (usually parameter-shared) based only on its **local observation**.

**Baseline Algorithm:**  
An **A3C-style actor–critic**, implemented synchronously as **A2C** first, and optionally extended to asynchronous A3C later.

**What “Cooperation” Means:**  
Agents learn policies that **improve team reward** (e.g., covering different landmarks without collisions) more than if they acted independently.

---

## Key Concepts You Should Feel Comfortable With

### Parameter Sharing
All agents use the **same policy network** but take different actions based on their own observations.  
This encourages symmetry, improves sample efficiency, and simplifies training.

### Centralized Critic
A **value function** that takes the **concatenated observations (and maybe actions)** of all agents and predicts a **global value**.  
It stabilizes training when rewards are shared by the team.

### Credit Assignment
Determines how each agent learns whether *its* action helped the team.  
A centralized critic and shared rewards provide a practical solution.

### Communication
Optional signals agents can pass to each other — continuous vectors or discrete symbols — to coordinate better.  
This will be a **bonus extension** for the project.

---

## Choose Scenarios (and Why)

Pick **one primary** and **one stretch**:

- **Primary:** `simple_spread` — agents must cover distinct landmarks with minimal collisions.  
  → Classic cooperative coordination; easy to visualize and reason about.

- **Stretch A (partial comm):** `simple_reference` — one speaker communicates a target landmark to listeners.  
  → Tests communication and information sharing.

- **Stretch B (noisy/partial info):** Variants or custom wrappers with noisy visibility or rewards.  
  → Tests robustness of learned cooperation.

---

## What to Measure (Objective, Not Vibes)

### Team Success
- Scenario-specific **success rate** (e.g., all landmarks covered within horizon)
- **Average team reward**
- **Collision count**

### Coordination Quality
- Average minimum distance of each landmark to nearest agent  
- Time-to-coverage (how quickly agents achieve goal)  
- Role specialization (do agents consistently choose distinct landmarks?)

### Stability & Robustness
- Mean ± std over random seeds (≥ 3)  
- Sensitivity to entropy coefficient, rollout length, number of parallel envs

### If You Add Communication
- Message usage (entropy/sparsity)  
- Correlation of message statistics with success rate  
- Emergence of interpretable signaling patterns

---

## Experiments You’ll Run (Before Touching Communication)

### Core Ablations (All Else Equal)
1. **Critic type:** centralized critic (CTDE) vs. decentralized per-agent critics  
2. **Reward structure:** pure team reward vs. team + individual shaping (e.g., distance-based)  
3. **Parameter sharing:** with vs. without (separate policies per agent)  
4. **Sync vs. async:** A2C (synchronous) vs. A3C (asynchronous)

### Hyperparameters to Sweep
| Parameter | Values |
|------------|---------|
| Entropy coefficient | 0.005, 0.01, 0.02 |
| Rollout length | 32, 64, 128 |
| # of parallel environments | 8, 16, 32 |
| Learning rate | 2e-4, 3e-4, 5e-4 |

---

## Communication Extension (Bonus Plan)

### Phase 1 — Lightweight (Continuous)
- Each agent learns a **continuous message vector** \(m_i\)  
- Append messages to next-step observations (CommNet-lite)  
- Add entropy regularization to prevent collapse

### Phase 2 — Discrete
- Each agent emits a small **categorical token** per step  
- Use **Gumbel-Softmax** to make it differentiable  
- Evaluate whether meaningful discrete symbols emerge

### Comparisons
- No-comm vs. comm  
- Continuous vs. discrete comm  
- Measure improvement in success rate and speed of coordination

---

## Roles for a Team of 3 (Clear Ownership)

### 🧩 Teammate A — Platform & Environment Lead
- Owns MPE setup, wrappers, seeding, reproducibility
- Sets up logging (TensorBoard / WandB)
- Builds experiment scripts and config files
- Prepares plotting notebooks/dashboards for metrics

### ⚙️ Teammate B — RL Algorithm Lead
- Owns policy/critic design, CTDE scaffolding, GAE advantage calculation
- Implements loss terms, gradient clipping, normalization, stability tricks
- Runs ablations (centralized vs decentralized, reward shaping)
- Conducts hyperparameter sweeps

### 📊 Teammate C — Analysis & Communication Lead
- Designs and evaluates communication extensions
- Defines success metrics for coordination and comm efficiency
- Creates figures, tables, comparative analysis
- Writes final report and presentation slides

**Tip:**  
A and B co-own baseline training.  
C joins after baselines are stable to add communication and analysis.

---

## Milestones & Timeline (4–5 Weeks Example)

### Week 1 — Understanding & Design
- Choose scenario (`simple_spread`), metrics, ablations
- Draw model: parameter-shared actors + centralized critic (CTDE)
- Decide logging schema (e.g., `sspread_ctde_lr3e-4_roll64_e0.01_seed0`)
- **Deliverable:** one-page design doc + experiment matrix

### Week 2 — Baseline Runs & Sanity Checks
- Get stable learning curves on `simple_spread` with CTDE A2C-style
- Verify: increasing entropy → more exploration early; rollout length affects variance
- **Deliverable:** reward vs steps plot (3 seeds), stability notes

### Week 3 — Ablations
- Run critic type (centralized vs decentralized)
- Try reward shaping on/off; parameter sharing on/off
- Report mean ± std over seeds; keep visualizations consistent
- **Deliverable:** ablation table + 2–3 key graphs + short insights

### Week 4 — Communication Extension
- Add continuous comm vector  
- Measure deltas in success, time-to-coverage  
- If time: discrete comm via Gumbel-Softmax  
- **Deliverable:** comm vs no-comm plots + message usage stats

### Week 5 — Polish & Storytelling
- Final hyperparameter sweeps and robustness checks
- Write report and slides: problem → method → baseline → ablations → comm → conclusions
- Create 2–3 min demo clip or GIF of trained policy  
- **Deliverable:** full report, visuals, demo

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-------------|
| Instability / divergence | Smaller LR, gradient clipping, normalized observations, reward scaling |
| Sparse / deceptive rewards | Add light shaping (distance, collision penalties) but keep team reward main driver |
| Overfitting to seeds | Always run ≥3 seeds per configuration |
| Compute / time limits | Use vectorized environments, shorter episodes, log periodically (not every step) |

---

## Documentation & Deliverables

- **Design doc (2–3 pages):** CTDE rationale, network sketches, training loop schematic  
- **Results notebook:** clean plots (reward, success, collisions), ablation tables, commentary  
- **Short video/GIF:** show coordination improvement after training  
- **Final report (4–6 pages):** method, results, limitations, and possible extensions (e.g., MAPPO, attention-based comm)

## TorchRL_MAC Submission Artifacts
- TorchRL_MAC_utils.py: helper entry points to build the wrapped env, infer shapes, and run stateless rollouts.
- TorchRL_MAC.API.md / TorchRL_MAC.API.ipynb: quick API reference and import/shape sanity checks.
- TorchRL_MAC.example.md / TorchRL_MAC.example.ipynb: runnable minimal example wiring env → shared policy → rollout.
- Dockerfile, docker_build.sh, docker_bash.sh: containerized runtime; build with `./docker_build.sh` and open a shell with `./docker_bash.sh` (image tag `torchrl_mac:latest`).

---

## Reading List (Short & Targeted)

- **A3C (Mnih et al., 2016)** — understand asynchronous vs. synchronous actor–critic  
- **CTDE principle** — why centralized critics help in cooperative MARL  
- **PettingZoo MPE docs** — scenario definitions, rewards, and physics  
- *(Optional)* **Communication in MARL** — CommNet, VDN, QMIX for context and ideas

---

## 📚 Supplementary Concept Notes (For Team Reference)

### Centralized Training, Decentralized Execution (CTDE)

**Motivation:**  
Each agent has limited local information. If trained independently, coordination collapses due to ambiguous credit assignment.

**CTDE Idea:**  
During training, agents can share global information (e.g., all observations/actions) through a **centralized critic**.  
During execution, they act independently with only local inputs.

**Analogy:**  
Think of a coach who sees the entire soccer field during practice (centralized training),  
but during the actual match, players act on their local view (decentralized execution).

---

### A3C vs. A2C

| Algorithm | Type | Key Idea |
|------------|------|----------|
| **A3C** | Asynchronous | Many agents interact in separate envs, update shared parameters asynchronously → decorrelated experiences, faster training |
| **A2C** | Synchronous | Multiple envs step in parallel, then synchronize updates → simpler, more reproducible |

A2C = A3C’s synchronized cousin.  
Start with A2C; extend to A3C later for parallelized efficiency.

---

### Actor–Critic Roles

| Component | Learns | Output | Purpose |
|------------|---------|---------|----------|
| **Actor** | Policy (how to act) | Probability distribution over actions | Guides exploration and behavior |
| **Critic** | Value function (how good the current situation is) | Scalar value estimate | Stabilizes training by reducing variance |

Actor proposes → Critic evaluates → both improve together.

---

### What a Policy Is

A **policy** is simply:
\[
\pi(a | s; \theta)
\]
the probability of taking action `a` in state `s`, given parameters `θ`.

Training the actor = adjusting `θ` to make high-reward actions more probable.  
This is what the actor network outputs — a distribution over actions (discrete or continuous).

---

### How It Fits in Your Project

- **Environment (MPE):** provides states, rewards, physics  
- **Actor (shared):** maps each agent’s *local observation* → action distribution  
- **Critic (centralized):** maps all agents’ observations → team value estimate  
- **CTDE:** critic uses global info for training, actors use local info at runtime  
- **Team reward:** shared across agents → encourages cooperation  
- **Policy:** the neural network defining how agents act given their observations

---

### Training Flow (Conceptually)

1. **Collect experiences:** agents act, environment returns next states + shared reward.  
2. **Compute advantage:** use centralized critic to estimate expected value.  
3. **Update networks:**  
   - Actor gradient ∝ advantage × log(probability of taken action)  
   - Critic minimizes (predicted value − actual return)²  
   - Add entropy regularization to maintain exploration.  
4. **Repeat:** agents gradually coordinate and specialize roles.

---

### Expected Learning Outcome

By the end of the project, your team should understand:
- How multi-agent cooperation emerges from shared reward and CTDE.  
- How actor–critic methods (A2C/A3C) stabilize multi-agent training.  
- How communication can be layered on top to enhance coordination.

---

**✅ This document serves as your permanent project reference.**  
Use it for:
- onboarding teammates,
- structuring milestones,
- writing your report,
- and ensuring everyone understands the “why” behind each design choice.
---

## Getting Started

### Prerequisites
- Python 3.10 or 3.11
- pip or conda package manager
- (Optional) Docker for containerized environment

### Local Setup with Virtual Environment

**1. Create and activate virtual environment:**

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n torchrl_mac python=3.10
conda activate torchrl_mac
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Verify installation:**

```python
python -c "import torch; from src.envs import mpe_env; print('Setup OK')"
```

---

### Running Notebooks

**API Demonstration (no training):**

```bash
jupyter notebook TorchRL_MAC.API.ipynb
# or
jupyter lab TorchRL_MAC.API.ipynb
```

This notebook demonstrates the two-layer API without training:
- Native API (src modules)
- Wrapper layer (TorchRL_MAC_utils)
- Forward passes and action selection

**Training Example (full A2C loop):**

```bash
jupyter notebook TorchRL_MAC.example.ipynb
# or
jupyter lab TorchRL_MAC.example.ipynb
```

This notebook trains a CTDE A2C baseline and plots learning curves. Expected runtime: 5-15 minutes for 300 episodes.

**Expected output:**
```
Starting training...
Training complete! Final episode return: -8.34
```

---

### Docker Setup

**Build the image:**

```bash
chmod +x docker_build.sh docker_bash.sh
./docker_build.sh
```

Expected output:
```
Building image torchrl_mac:latest from .
[+] Building 45.2s (12/12) FINISHED
...
```

**Run Jupyter Lab in container:**

```bash
./docker_bash.sh
```

Expected output:
```
Starting Jupyter Lab container (image: torchrl_mac:latest)
Access at: http://localhost:8888
Press Ctrl+C to stop

[I 2025-12-14 12:34:56.789 ServerApp] Jupyter Server 2.x.x is running at:
[I 2025-12-14 12:34:56.789 ServerApp] http://0.0.0.0:8888/lab
```

Open browser to `http://localhost:8888` and run notebooks.

**Stop container:** Press `Ctrl+C` in terminal.

---

## Project Deliverables

### Core Files

1. **TorchRL_MAC_utils.py** — Single reusable module with:
   - Contract layer (EnvConfig, TrainConfig, RolloutBatch dataclasses)
   - Network builders (shared actor, centralized critic)
   - Training loop (train_ctde_a2c)
   - Backward-compatible helpers

2. **TorchRL_MAC.API.md** — API reference documentation:
   - Layer 1: Native API (src modules)
   - Layer 2: Wrapper utilities
   - CTDE concept explanation
   - Mermaid dataflow diagram

3. **TorchRL_MAC.API.ipynb** — Interactive API demonstration:
   - Environment and wrapper usage
   - Actor/critic construction
   - Forward passes without training

4. **TorchRL_MAC.example.md** — Training guide:
   - End-to-end CTDE A2C explanation
   - Loss components and design rationale
   - Training loop flowchart (mermaid)
   - Extension roadmap (GAE → TorchRL → MAPPO)

5. **TorchRL_MAC.example.ipynb** — Runnable training notebook:
   - Full A2C training loop (300 episodes)
   - Learning curve visualization (4-panel matplotlib)
   - Greedy evaluation pattern

### Supporting Files

- **requirements.txt** — Python dependencies (torch, mpe2, pettingzoo, numpy)
- **Dockerfile** — Container definition with Jupyter Lab
- **docker_build.sh** — Image build script
- **docker_bash.sh** — Container launch script with volume mount
- **src/** — Project source modules (envs, wrappers, agent_policy, train)
- **tests/** — Basic environment sanity tests

---

## Quick Reference

**Train baseline from Python:**
```python
from TorchRL_MAC_utils import EnvConfig, TrainConfig, train_ctde_a2c

env_cfg = EnvConfig(seed=42, max_steps=25)
train_cfg = TrainConfig(n_episodes=500, log_every=50)
history = train_ctde_a2c(env_cfg, train_cfg)
```

**Inspect rollout without training:**
```python
from TorchRL_MAC_utils import make_env, build_shared_actor, build_central_critic, collect_episode

env = make_env(env_cfg)
obs = env.reset()
actor = build_shared_actor(obs.shape[1], env.n_actions)
critic = build_central_critic(env.num_agents * obs.shape[1])
batch = collect_episode(env, actor, critic, env_cfg)
```

---

## Troubleshooting

**Import errors:**
- Ensure virtual environment is activated
- Check `PYTHONPATH` includes project root
- Verify all requirements installed: `pip list | grep -E "torch|pettingzoo|mpe"`

**Training diverges (NaN losses):**
- Reduce learning rates in TrainConfig
- Check max_grad_norm (default 0.5)
- Verify environment reset seeding

**Docker port conflicts:**
- Change port mapping in docker_bash.sh: `-p 8889:8888`
- Or stop conflicting services on port 8888

**Slow training:**
- Reduce n_episodes for quick tests
- Check CPU usage (no GPU acceleration by default)
- Consider vectorized environments for production runs