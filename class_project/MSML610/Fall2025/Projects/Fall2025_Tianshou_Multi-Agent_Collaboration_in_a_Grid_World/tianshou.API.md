# Tianshou API Tutorial  
Tianshou is a PyTorch-based reinforcement learning library with modular APIs for building agents efficiently.

## Table of Contents  
1. Native Documentation
    1. Policy Classes
    2. Collectors and Buffers
    3. Trainers
2. Notebook Documentation
    1. MultiAgentGridWorld
    2. CentralizedEnv
    3. QNet
    
## 1. Native Documentation
Tianshou is a PyTorch-based reinforcement learning library with modular APIs for building agents efficiently. It is structured around a training loop involving a few core components: a Policy, a Collector, a Buffer, and a Trainer.  

### 1.1. Policy Classes
Policy classes represent different Reinforcement Learning training algorithms, such as Deep Q-Network (DQN), Proximal Policy Optimization (PPO), and more. They directly interact with the Machine Learning model to determine how the agent "learns" from each episode of training.

Each Policy inherits from BasePolicy and implements these core methods:  
- **forward**: used for action selection from observations  
- **process_fn**: used to process the replay buffer  
- **update**: used to update during training  

In the case of multi-agent problems, Tianshou also implements a MultiAgentPolicyManager, which accepts a list of policies. However, this class has some restrictions on it, such as requiring a PettingZoo environment for training and a vectorized environment.

### 1.2. Collectors and Buffers  
Information from each training episode, such as agents' observations, actions, rewards, etc. are gathered using a Collector. A Collector is also responsible for managing resets, steps, and episode termination.

Information from a Collector is stored inside a Buffer, which is then given to the Policy for training and updates. The two main buffers are **ReplayBuffer**, which stores off-policy data with sampling, and **OnPolicyBuffer**, which stores sequential on-policy data.  

### 1.3. Trainers
The Trainer is the class that orchestrates the entire training process. It is responsible for invoking all functions required for the training loop, including Policy updates, action selection, information collection, and more. They are also responsible for starting epochs, how frequently to collect information, and callback functions for epsilon annealing. Like with buffers, the two main Trainers are **onpolicy_trainer** and **offpolicy_trainer**.  

## 2. Notebook Documentation  
In order to implement the Tianshou training loop, I created three classes: MultiAgentGridWorld, CentralizedEnv, and QNet. The first two define the environments to use the Tianshou training process on, and the last class defines the neural network that the Tianshou Policy will actually update.  

### 2.1. MultiAgentGridWorld
This class implements a grid world where every agent learns and acts independently. Each agent's goal is to gather an individual set of coins scattered across the grid, while dealing with a limited range of vision. The number of agents, number of coins per agent, grid size, and vision radius are all adjustable parameters.

### 2.2. Centralized Env
This class is a wrapper for MultiAgentGridWorld, where a single policy decides the actions for all agents and agents move at the same time. It is otherwise identical.

### 2.3. QNet
The neural network used to train the policy. It takes the observation space as input, and outputs an action (or set of actions, if using the centralized environment). The network is comprised of a CNN and Fully Connected Network to process separate parts of the observation, which both feed into one more Fully Connected Network.