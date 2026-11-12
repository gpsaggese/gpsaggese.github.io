---
title: "TorchRL + PettingZoo MPE in 60 Minutes"
draft: false
authors:
  - SaranshKumar
  - AbhinavSingh
  - AyushGaur
  - gpsaggese
date: 2026-03-27
description:
categories:
  - AI Research
  - Machine Learning
  - Software Engineering
---

TL;DR: Build a reproducible multi-agent reinforcement learning (MARL) pipeline
using TorchRL and PettingZoo's multi-agent particle environment (MPE), train
using centralized training, decentralized execution (CTDE), and verify that
communication is meaningful using structured diagnostics and causal ablations.

<!-- more -->

## Tutorial in 30 Seconds
TorchRL + PettingZoo MPE is a powerful stack for building and verifying
multi-agent reinforcement learning systems with measurable coordination.

Key capabilities:

- **Multi-agent environment integration**: Clean API for wiring PettingZoo
  environments with TorchRL's PyTorch-native RL pipeline
- **Centralized training, decentralized execution (CTDE)**: Stable training with
  decentralized per-agent actors and a centralized critic
- **Communication verification**: Structured diagnostics including message
  entropy, message change rate, and observation-derived verification checks
- **Causal ablations**: Built-in support for testing communication importance
  through full communication, disabled communication, and random communication
  modes
- **Outcome-aligned evaluation**: Binary success rates, goal-distance debugging,
  and structured communication metrics

This tutorial's goal is to show you in 60 minutes:

- The core architecture for building MARL systems with TorchRL and PettingZoo
- How to set up CTDE training with centralized critics and decentralized actors
- How to verify coordination and communication meaning through structured
  diagnostics and causal ablations
- Why training curves alone are not sufficient proof of cooperation

## The Problem with Most MARL Tutorials
Most multi-agent reinforcement learning tutorials show:

- Moving loss curves
- Increasing rewards
- "Healthy-looking" training

But they rarely verify:

- Whether agents truly coordinate
- Whether communication carries useful information
- Whether evaluation-time success improves

In multi-agent RL, training curves are not proof of cooperation. This tutorial
focuses on measurable coordination.

## Official References
- [TorchRL: A PyTorch Reinforcement Learning Library](https://github.com/pytorch/rl)
- [PettingZoo: Parallel Multi-Agent Environment Library](https://pettingzoo.farama.org/)
- [Multi-Agent Particle Environment (MPE)](https://pettingzoo.farama.org/environments/mpe/)

## Tutorial Content
This tutorial includes all the code, notebooks, and Docker containers in
[tutorials/TorchRL_Multi_Agent_Cooperation](https://github.com/gpsaggese/gpsaggese.github.io/tree/master/tutorials/TorchRL_Multi_Agent_Cooperation)

- [`README.md`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/TorchRL_Multi_Agent_Cooperation/README.md):
  Instructions and setup for the tutorial environment
- A Docker system to build and run the environment using our standardized
  approach
- Environment and wrapper components:
  - PettingZoo MPE task wrappers (e.g., `simple_reference`)
  - TorchRL-compatible wrappers
  - Explicit message-channel wiring
- CTDE Training setup:
  - Decentralized per-agent actors
  - Centralized critic
  - Stable policy optimization
- Evaluation and diagnostics:
  - Binary success rate metrics
  - Goal-distance debugging
  - Structured communication metrics including `message_entropy` and
    `message_change_rate`
- Causal ablations:
  - `full_comm`: Standard communication enabled
  - `disable_comm`: Communication disabled
  - `random_comm`: Random communication
