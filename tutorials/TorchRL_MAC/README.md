# TorchRL Multi-Agent Cooperation Tutorial

This 60-minute hands-on tutorial introduces multi-agent reinforcement learning
(MARL) through practical examples using `TorchRL` and `PettingZoo` (MPE). You
will learn to build, train, and evaluate cooperative multi-agent systems with
communication.

## Tutorial in 30 Seconds

**TorchRL** is a scalable, modular library for reinforcement learning research
and **PettingZoo** provides lightweight multi-agent environments.

Key capabilities:

- **Multi-agent reinforcement learning**: Centralized training with
  decentralized execution (CTDE) for cooperative agents
- **Communication protocols**: Build and verify agent communication in
  structured environments
- **TorchRL integration**: Leverage TorchRL's actor-critic utilities for
  scalable training
- **Evaluation metrics**: Structured metrics beyond rewards (success rate,
  communication entropy, goal distance diagnostics)

After this tutorial, you will understand:

- How to integrate PettingZoo multi-agent environments with TorchRL
- The CTDE (Centralized Training, Decentralized Execution) paradigm
- How to verify and debug agent communication
- How to design and interpret structured evaluation metrics for MARL
- Common failure modes in multi-agent training and iteration strategies

## Official References
- [TorchRL documentation](https://pytorch.org/rl/)
- [TorchRL GitHub repo](https://github.com/pytorch/rl)
- [PettingZoo documentation](https://pettingzoo.farama.org/)
- [PettingZoo GitHub repo](https://github.com/Farama-Foundation/PettingZoo)

## Getting Started

### Prerequisites
This tutorial runs in a Docker container with all dependencies pre-configured.
No additional setup is required beyond the steps below.

### Setup Instructions
1. **Navigate to the tutorial directory:**
   ```bash
   > cd tutorials/TorchRL_Multi_Agent_Cooperation
   ```

2. **Build the Docker image:**
   ```bash
   > ./docker_build.sh
   ```

3. **Launch Jupyter Lab:**
   ```bash
   > ./docker_bash.sh
   ```

## Dependency Management

- For more information on the Docker build system refer to
  [Project template readme](/class_project/project_template/docker_scripts.README.md)

## Tutorial Notebooks

Work through the following notebooks in order:

- [`TorchRL_MAC.API.ipynb`](TorchRL_MAC.API.ipynb): Core MARL fundamentals
  - PettingZoo environment setup and wrappers
  - Agent observation and action specs
  - TorchRL actor and critic architecture
  - Rollout and batch collection utilities

- [`TorchRL_MAC.example.ipynb`](TorchRL_MAC.example.ipynb): End-to-end MARL
  training and evaluation
  - Multi-worker A3C training setup
  - Communication verification and debugging
  - Structured evaluation metrics (success rate, message entropy, goal
    distances)
  - MARL failure modes and iteration strategies
  - Final results and diagnostics

- [`TorchRL_MAC_utils.py`](TorchRL_MAC_utils.py): Utility functions supporting
  the tutorial notebooks
  - Environment creation and wrapper logic
  - Actor and critic helpers
  - Rollout and training utilities
  - Evaluation and communication metrics

## Changelog
- 2026-03-01: Initial release
