# Description

RLlib is a scalable reinforcement learning library built on Ray, with a unified API
for many RL algorithms, multi-agent training, and distributed execution. It solves
the problem of moving an RL experiment from one process on a laptop to many parallel
workers without rewriting the training code. It is worth a 60-minute tutorial because
an `AlgorithmConfig` describes an experiment in one object, and the same object
trains on any Gymnasium environment.

## Technologies Used

RLlib

- Support for a variety of state-of-the-art RL algorithms
- Multi-agent training capabilities
- Built-in support for distributed training
- High-level abstractions for environment creation and training

# Tutorial

- Implement the tutorial "Learn RLlib in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - No earlier tutorial or project uses RLlib, so read the closest RL work
  - Read the `README.md` of `tutorials/gymnasium/` and `tutorials/TorchRL_MAC/`
  - Read the `README.md` of the Fall2025 CleanRL project
    - `class_project/msml610/Fall2025/projects/UmdTask49_Fall2025_CleanRL_Reinforcement_Learning_for_Stock_Trading/`
  - Read the `README.md` of the Ray projects of DATA605 for the Ray setup in Docker
    - `class_project/data605/Spring2025/projects/TutorTask93_Spring2025_Real-Time_Bitcoin_Data_Processing_with_Apache_Ray/`
    - `class_project/data605/Spring2026/projects/UmdTask464_DATA605_Spring2026_Ray_Housing_Price_Prediction/`
- Create `tutorials/RLlib/`, since it does not exist yet
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `rllib_utils.py`
  - `rllib.API.ipynb`
  - `rllib.example.ipynb`

# Project

## Project 1: Basic Reinforcement Learning with Grid World

- **Difficulty**: 1 (Easy)
- **Project Objective**: Train an RLlib agent to learn an optimal policy in a Grid
  World environment, optimizing for the maximum cumulative reward
- **Dataset Suggestions**: Simulated Grid World environment, implemented as a custom
  [Gymnasium](https://gymnasium.farama.org/) environment
- **Tasks**:
  - **Build the Environment**: Implement a custom Grid World as a `gymnasium.Env`
    with `Discrete` observation and action spaces, walls, a goal cell, and trap cells
  - **Define the Problem**: Set the rewards (goal, step penalty, trap) and the
    episode limit, and compute the reward of a random policy and the optimal path
    length
  - **Train the Agent**: Configure a deep Q-learning agent with `DQNConfig`, register
    the environment with `register_env`, and train it with `Algorithm.train`
  - **Evaluate the Policy**: Compute the mean episode reward and the success rate
    over evaluation episodes, and compare the steps to the goal with the optimal path
  - **Visualize the Policy**: Plot the learning curve (mean reward per training
    iteration) and the greedy policy as arrows on the grid
- **Bonus Ideas (Optional)**: Experiment with different reward structures or modify
  the grid layout to create more complex environments; compare `DQNConfig` with
  `PPOConfig`

### Milestones

- Milestone 1: Set up the container and the environment
  - Project tasks: Build the Environment
  - Result: `tutorials/RLlib/` container running Ray, and the Grid World rendered
    with a random-policy episode
- Milestone 2: API notebook
  - Project tasks: Train the Agent
  - Result: `rllib.API.ipynb` covering `AlgorithmConfig`, `PPOConfig`, `DQNConfig`,
    `register_env`, `Algorithm.train`, and `Algorithm.evaluate` on CartPole
- Milestone 3: Example notebook
  - Project tasks: Define the Problem, Train the Agent, Evaluate the Policy,
    Visualize the Policy
  - Result: `rllib.example.ipynb` running end to end

## Project 2: Reinforcement Learning for CartPole Balancing

- **Difficulty**: 2 (Medium)
- **Project Objective**: Develop a reinforcement learning agent that can balance a
  pole on a moving cart, optimizing for the longest time the pole remains upright
- **Dataset Suggestions**: Gymnasium
  [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
  environment (available directly through the library)
- **Tasks**:
  - **Select the Environment**: Import the CartPole environment from Gymnasium
  - **Create the Agent**: Utilize RLlib to implement a Proximal Policy Optimization
    (PPO) agent with `PPOConfig`
  - **Tune the Hyperparameters**: Experiment with different hyperparameters to
    optimize the agent's performance
  - **Evaluate the Agent**: Evaluate the agent's performance and visualize the
    average reward over multiple episodes
- **Bonus Ideas (Optional)**: Compare the performance of different algorithms in
  RLlib (e.g., PPO vs. DQN) on the same task

## Project 3: Autonomous Drone Navigation

- **Difficulty**: 3 (Hard)
- **Project Objective**: Build a reinforcement learning agent that controls a drone
  (or simulated vehicle) to navigate through an environment, avoiding obstacles and
  reaching a target location
- **Dataset Suggestions**: Lightweight default: PyBullet drone environments or the
  Gymnasium
  [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
  environment with continuous actions, for fast training on laptops or Colab
- **Tasks**:
  - **Set Up the Simulation**: Install PyBullet and configure a simple drone or
    continuous-control navigation task (e.g., LunarLander with `continuous=True`)
  - **Implement the Agent**: Use RLlib `SACConfig` (Soft Actor-Critic) for continuous
    action spaces, and define the state (position, velocity, orientation) and the
    action (thrust, pitch, yaw)
  - **Train the Agent**: Train the agent to navigate toward a goal while avoiding
    obstacles, and experiment with different reward functions (e.g., penalties for
    collisions, bonuses for smooth flight)
  - **Evaluate the Performance**: Compute the task completion rate, the average time
    to goal, and the number of collisions across episodes with varied initial
    conditions
  - **Visualize the Flights**: Use 2D or 3D plots in Matplotlib or the PyBullet
    viewer to show the flight paths
- **Bonus Ideas (Optional)**: Explore multi-agent coordination (multiple drones
  reaching goals simultaneously); compare SAC with PPO for continuous navigation; add
  energy efficiency metrics (penalize excessive thrust)
