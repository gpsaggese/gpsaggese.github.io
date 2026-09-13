---
title: "Gymnasium in 30 mins"
draft: false
authors:
    - gpsaggese
date: 2026-06-11
description:
categories:
    - Reinforcement Learning
---

TL;DR `Gymnasium` is a standardized Python API for reinforcement learning
environments. It gives every environment the same `reset()` / `step()` interface,
so the same agent code runs against CartPole, Atari, MuJoCo, or your own custom
problem without changes.

<!-- more -->

## Introduction

- This tutorial shows how to use `Gymnasium` to run reinforcement learning (RL)
  environments and how to build your own

- Reinforcement learning needs an environment: a world the agent acts in, that
  returns observations and rewards. Without a standard, every environment exposes
  a different interface and agent code is not portable

- `Gymnasium` solves this by defining one API that all environments implement:
    - Create an environment with `gym.make()`
    - Start an episode with `env.reset()`
    - Advance one timestep with `env.step(action)`
    - Describe valid observations and actions with `spaces`

- `Gymnasium` is the maintained fork of OpenAI's `Gym`, now developed by the
  [Farama Foundation](https://farama.org/). OpenAI stopped maintaining `Gym` in
  2022; `Gymnasium` is the drop-in successor and the one to use today

### Alternative tools

- **`PettingZoo`**: the same API extended to multi-agent environments
- **`Gymnasium-Robotics`** and **`MuJoCo`**: continuous-control robotics tasks
- **`DeepMind Control Suite`** (`dm_control`): a different control-focused API
- **`Stable-Baselines3`** and **`RLlib`**: RL algorithm libraries that consume
  `Gymnasium` environments (they are agents, not environments)

### Official references

- Installation: <https://gymnasium.farama.org/introduction/installation/>
- Documentation: <https://gymnasium.farama.org/>
- Tutorials: <https://gymnasium.farama.org/tutorials/>
- GitHub: <https://github.com/Farama-Foundation/Gymnasium>

## Prerequisites

- Basic Python and `numpy`
- Familiarity with the RL framing: agent, environment, observation, action,
  reward
- Python 3.10 or later

## Installation

- Install the core library with `pip`:

    ```bash
    > pip install gymnasium
    ```

- Install with extras for specific environment families:

    ```bash
    # Classic control (CartPole, MountainCar, Pendulum).
    > pip install "gymnasium[classic-control]"
    # Box2D (LunarLander, BipedalWalker, CarRacing).
    > pip install "gymnasium[box2d]"
    # Atari games.
    > pip install "gymnasium[atari]"
    # Everything.
    > pip install "gymnasium[all]"
    ```

- Using `uv` (faster):

    ```bash
    > uv pip install "gymnasium[classic-control]"
    ```

### Verifying installation

- Confirm the install and version:

    ```bash
    > python -c "import gymnasium as gym; print(gym.__version__)"
    1.1.1
    ```

## Core Concepts

- **Environment**: the world the agent interacts with. It holds state, accepts
  actions, and returns the next observation and a reward. You create one with
  `gym.make("EnvName-vN")`

- **The step loop**: RL is a loop of observe → act → observe. `env.step(action)`
  returns five values:
    - `observation`: what the agent sees next
    - `reward`: the scalar feedback for the action just taken
    - `terminated`: `True` when the episode ends naturally (goal reached, agent
      failed)
    - `truncated`: `True` when the episode is cut off externally (time limit)
    - `info`: a dict of diagnostics, not used for learning
    - The split between `terminated` and `truncated` matters: a time-limit cutoff
      should not be treated as a real terminal state when bootstrapping values

- **Spaces**: every environment declares an `observation_space` and an
  `action_space`. A space defines the shape, type, and bounds of valid data, and
  can sample random valid values. This is how generic agent code knows what
  actions are legal without hard-coding the environment

## Basic Usage

### Running an episode

- Create an environment, reset it, and loop until the episode ends:

    ```python
    import gymnasium as gym

    # Create the environment; `render_mode="human"` opens a visual window.
    env = gym.make("CartPole-v1", render_mode="human")

    # Start a new episode; always reset before the first step.
    observation, info = env.reset(seed=42)

    episode_over = False
    total_reward = 0.0
    while not episode_over:
        # Pick a random valid action (replace with a policy later).
        action = env.action_space.sample()
        # Apply the action and observe the result.
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # The episode ends on either signal.
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
    env.close()
    ```

- Expected output (rewards vary because actions are random):

    ```bash
    Episode finished! Total reward: 23.0
    ```

- The most common beginner error is calling `step()` before `reset()`. Always
  reset first

### Inspecting spaces

- Look at what the environment expects and returns:

    ```python
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    print("Observations:", env.observation_space)
    print("Actions:", env.action_space)
    print("Sample action:", env.action_space.sample())
    ```

- Expected output:

    ```bash
    Observations: Box([-4.8 ... ], [4.8 ... ], (4,), float32)
    Actions: Discrete(2)
    Sample action: 1
    ```

- `Box(...)` means a continuous 4-dimensional observation; `Discrete(2)` means
  two possible actions (push left or right)

## Spaces in Detail

- The main space types cover most environments:
    - **`Discrete(n)`**: integers `0..n-1`, e.g. a fixed set of actions
    - **`Box(low, high, shape)`**: a continuous vector or tensor with bounds, e.g.
      joint angles or pixel images
    - **`MultiDiscrete([...])`**: several discrete dimensions at once
    - **`MultiBinary(n)`**: a vector of `n` 0/1 flags
    - **`Dict`** and **`Tuple`**: compose the above into structured observations
    - **`Text`** and **`Sequence`**: variable-length text and sequence data

- Create and sample from spaces directly:

    ```python
    import numpy as np
    from gymnasium import spaces

    # Three discrete actions.
    discrete = spaces.Discrete(3)
    print(discrete.sample())          # e.g. 2

    # A continuous 2D vector bounded in [-1, 1].
    box = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    print(box.sample())               # e.g. [ 0.42 -0.17]

    # Check membership.
    print(box.contains(np.array([0.5, 0.5], dtype=np.float32)))  # True
    ```

## Wrappers

- Wrappers modify an environment without changing its code. They share the same
  API, so you can stack them and the result is still an environment

- Common wrappers:
    - **`TimeLimit`**: end episodes after N steps (applied by `gym.make` by
      default)
    - **`RecordEpisodeStatistics`**: track episode return and length in `info`
    - **`RecordVideo`**: save MP4 videos of episodes
    - **`FlattenObservation`**: flatten structured observations into one vector
    - **`RescaleAction`** / **`NormalizeObservation`**: rescale actions or
      normalize observations for stable training

- Apply a wrapper by wrapping the env:

    ```python
    import gymnasium as gym
    from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit

    env = gym.make("CartPole-v1")
    # Cap episodes at 100 steps and record return/length.
    env = TimeLimit(env, max_episode_steps=100)
    env = RecordEpisodeStatistics(env)

    observation, info = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
    # `info` carries episode stats when the episode ends.
    print(info["episode"])            # {'r': ..., 'l': ..., 't': ...}
    env.close()
    ```

## Vectorized Environments

- Training is faster when many copies of an environment run in parallel. A
  vectorized env steps a batch of environments at once and returns batched
  arrays

    ```python
    import gymnasium as gym

    # Run 4 CartPole environments in parallel.
    envs = gym.make_vec("CartPole-v1", num_envs=4)
    observations, infos = envs.reset(seed=0)
    # `actions` is a batch of 4; results are batched along axis 0.
    actions = envs.action_space.sample()
    observations, rewards, terminations, truncations, infos = envs.step(actions)
    print(observations.shape)         # (4, 4)
    print(rewards.shape)              # (4,)
    envs.close()
    ```

- Vectorized envs auto-reset each sub-environment when its episode ends, so the
  batch keeps stepping without manual bookkeeping

## Building a Custom Environment

- To make your own environment, subclass `gym.Env`, declare the two spaces, and
  implement `reset` and `step`:

    ```python
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces

    class GridWorldEnv(gym.Env):
        """Walk a 1-D line to reach the goal at the right end."""

        def __init__(self, size: int = 5) -> None:
            self._size = size
            self._pos = 0
            # Observation: current integer position.
            self.observation_space = spaces.Discrete(size)
            # Actions: 0 = left, 1 = right.
            self.action_space = spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._pos = 0
            return self._pos, {}

        def step(self, action):
            # Move and clamp to the line.
            self._pos += 1 if action == 1 else -1
            self._pos = int(np.clip(self._pos, 0, self._size - 1))
            terminated = self._pos == self._size - 1
            reward = 1.0 if terminated else 0.0
            return self._pos, reward, terminated, False, {}
    ```

- Register it so `gym.make` can find it by name:

    ```python
    gym.register(id="GridWorld-v0", entry_point=GridWorldEnv)
    env = gym.make("GridWorld-v0", size=8)
    ```

## Tutorial Notebooks

Work through the notebooks in this directory:

- [`gymnasium.API.ipynb`](gymnasium.API.ipynb): the core API surface
    - Creating environments and reading spaces
    - The `reset` / `step` loop and the five-tuple return
    - Wrappers and vectorized environments

- [`gymnasium.example.ipynb`](gymnasium.example.ipynb): an end-to-end example
    - Running full episodes with a simple policy
    - Building and registering a custom environment
    - Collecting and plotting episode statistics

- [`gymnasium_utils.py`](gymnasium_utils.py): helper functions used by the
  notebooks

## Getting Started with the Container

- This tutorial runs in a Docker container with dependencies pre-configured

- Build the image:

    ```bash
    > cd tutorials/gymnasium
    > ./docker_build.sh
    ```

- Launch Jupyter Lab:

    ```bash
    > ./docker_jupyter.sh
    ```

- For more on the Docker build system, see the
  [project template README](/class_project/project_template/docker_scripts.README.md)

## Summary

- `Gymnasium` gives RL a single, portable interface: `reset()` to start,
  `step()` to act, `spaces` to describe what is legal
- The five-tuple from `step()` separates natural termination from external
  truncation, which matters for correct value bootstrapping
- Wrappers add behavior without touching environment code, and vectorized envs
  parallelize training
- Building a custom environment is just subclassing `gym.Env` and implementing
  `reset` and `step`

## Changelog
- 2026-06-11: Initial release
