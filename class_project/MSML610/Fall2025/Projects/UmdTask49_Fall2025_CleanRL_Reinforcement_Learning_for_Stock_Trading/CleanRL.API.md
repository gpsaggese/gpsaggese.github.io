# CleanRL API

CleanRL is a Deep Reinforcement Learning library that provides high-quality single-file implementations with research-friendly features. The implementation is clean and simple, yet scalable.

## Core Philosophy

- **Single-file implementation**: Every detail about an algorithm variant is put into a single standalone file. This makes it easy to understand and modify.
- **No shared core**: Unlike modular libraries, CleanRL does not have a shared base class. This prevents complex inheritance chains.

## Key Features

- **Single-file implementation**
- **Benchmarked Implementation** (7+ algorithms and 34+ games)
- **Tensorboard Logging**
- **Local Reproducibility via Seeding**
- **Videos of Gameplay Capturing**
- **Experiment Management with Weights and Biases**

## Usage Pattern

CleanRL is not meant to be imported as a traditional library. The intended workflow is to copy the specific algorithm file into your project and modify it directly.

The `CleanRL_API` folder contains the following implementations:

- `ppo_continuous_action.py`: Proximal Policy Gradient for continuous action spaces.
- `sac_continuous_action.py`: Soft Actor-Critic for continuous action spaces.
- `cleanrl_utils/`: A collection of small utility functions (e.g., Replay Buffers) used by some algorithms like SAC.

The algorithms are typically run as scripts with command-line arguments:

```bash
python CleanRL_API/ppo_continuous_action.py --env-id Hopper-v4 --total-timesteps 50000
```

Alternatively, the scripts in this repository have been adapted to expose a `train` function and an `Args` class, allowing them to be used programmatically (e.g., in a Jupyter Notebook):

```python
from CleanRL_API.sac_continuous_action import train as train_sac, Args as SACArgs

args = SACArgs(
    env_id="Hopper-v4",
    total_timesteps=5000,
    seed=42
)
agent = train_sac(args)
```

## Algorithms

CleanRL supports many popular algorithms. The ones included in this API folder are:

- **Proximal Policy Gradient (PPO)**: A policy gradient method that alternates between sampling data through interaction with the environment and optimizing a "surrogate" objective function.
- **Soft Actor-Critic (SAC)**: An off-policy actor-critic deep RL algorithm based on the maximum entropy reinforcement learning framework.

## Documentation

For full documentation and examples, visit the [official CleanRL documentation](https://docs.cleanrl.dev/).
