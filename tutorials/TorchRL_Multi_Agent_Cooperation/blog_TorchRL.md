# TorchRL + PettingZoo MPE in 60 Minutes

| title | authors | date | description | categories |
|-------|---------|------|------------|------------|
| TorchRL + PettingZoo MPE in 60 Minutes | Saransh Kumar · Abhinav Singh · Ayush Gaur | 2026-03-03 | End-to-end CTDE MARL tutorial with reproducible training and verified communication | AI Research · Reinforcement Learning · Multi-Agent Systems |

---

## TL;DR

Build a reproducible multi-agent reinforcement learning (MARL) pipeline using TorchRL and PettingZoo’s Multi-Agent Particle Environment (MPE), train using CTDE (Centralized Training, Decentralized Execution), and verify that communication is meaningful using structured diagnostics and causal ablations.

---

## The Problem with Most MARL Tutorials

Most multi-agent reinforcement learning tutorials show:

- Moving loss curves  
- Increasing rewards  
- “Healthy-looking” training  

But they rarely verify:

- Whether agents truly coordinate  
- Whether communication carries useful information  
- Whether evaluation-time success improves  

In multi-agent RL, training curves are not proof of cooperation.

This tutorial focuses on measurable coordination.

---

## What You’ll Build in 60 Minutes

### 1. Environment + Wrapper Layer

- PettingZoo MPE task (e.g., `simple_reference`)
- TorchRL-compatible wrappers
- Explicit message-channel wiring

### 2. CTDE Training Setup

- Decentralized per-agent actors
- Centralized critic
- Stable policy optimization

### 3. Outcome-Aligned Evaluation

- Binary success rate
- Goal-distance debugging
- Structured communication metrics

### 4. Communication Verification

- `message_entropy`
- `message_change_rate`
- Observation-derived verification checks

### 5. Optional Causal Ablations

- `full_comm`
- `disable_comm`
- `random_comm`

If communication matters, success should drop when it is removed.

---

## Why TorchRL + PettingZoo (MPE)?

This stack provides:

- A clean multi-agent API (PettingZoo)
- A PyTorch-native RL pipeline (TorchRL)
- Full control over observations, actions, and message channels
- Lightweight local reproducibility

MPE is ideal for:

- Studying coordination failure modes
- Debugging communication mechanisms
- Practicing CTDE engineering patterns

---

## Final Takeaway

In multi-agent reinforcement learning:

- Loss curves are not evidence of coordination.  
- Reward trends are not proof of cooperation.  

**Success metrics, diagnostics, and ablations are.**