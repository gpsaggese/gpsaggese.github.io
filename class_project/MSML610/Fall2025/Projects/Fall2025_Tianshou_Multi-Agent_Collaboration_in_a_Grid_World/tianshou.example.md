# Tianshou: Multi-Agent Collaboration in a Grid World  
The corresponding notebook contains stages of implementations of Grid World reinforcement learning training using the environment defined in the API notebook. The notebook demonstrates the following, in order:  
- How the grid world looks with a random policy and a single agent
- Training with a single agent in the grid world, using the neural network defined in the API notebook
- Training with multiple independent agents in the grid world using the same neural network, along with difficulties that arise as a result
- Training with multiple agents controlled by a central policy using the same neural network

## Table of Contents  
1. Grid World Example  
2. Single Agent Baseline  
3. Multiple Independent Agents  
4. Multiple Agents with a Centralized Policy  

## Grid World Example  
The first part of the notebook demonstrates what running the grid world would look like with an agent that takes purely random actions.  

## Single Agent Baseline  
The second part of the notebook trains a single agent in the Grid World environment, which demonstrates great competence in completing its mission of collecting all coins on the grid while being restricted by a limited range of vision. Training utilizes the DQN policy, which is ideal for small discrete action spaces.  

## Multiple Independent Agents  
The third part of the notebook trains 2 independent agents, whose observations all feed into the same neural network. These trained agents are much less effective at collecting all their coins, even though they are effectively acting in complete isolation with no shared rewards or observations. The likely reason for this is that in DQN, agents are treated as part of the environment, and because their policies are constantly evolving, the "optimal" strategy becomes more difficult to learn, even when absolutely nothing has changed.  

## Centralized Policy 
The final part of the notebook trains a single centralized policy that controls both agents. Essentially, rather than 2 independent agents with 4 moves each, this is 1 agent with 16 possible moves (every combination of moves from 2 different agents). Additionally, both agents will "share" their observations with each other. This is done by stacking their observations on top of each other, before passing the combined observations to the neural network.  

The resulting trained policy is much more likely to solve the problem than multiple independent agents. However, since agents' observations are simply stacked on top of each other, with no clear indication as to which set of observations correspond to which agent. This creates a phenomenon where the central policy is most likely to choose the same move for both agents together, rather than making independent choices for each agent.  

The next steps from here would be to modify the centralized environment defined in the API notebook to add some observation that more strongly identifies each agent's set of observations, or to switch to a policy that is more likely to perform well, such as Multi-Agent PPO.