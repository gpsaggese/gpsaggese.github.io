# Algorithm Reference: Papers, Authors, Implementations

# Template
- Prefer arXiv links whenever available for direct access to preprints and
  published papers

- The citation format is
  ```
  Author(s), "Title" (Year) - arxiv.org/abs/1509.06461
  ```
  - If there are more than one author use the last name and "et al."
  - E.g.,
  ```
  Watkins et al., "Q-learning" (1992) - arxiv.org/abs/1509.06461
  ```

- The format for each algorithm is the following
  ```
  ### <topic>
  - **Short description**:
    - ...
  - **Key papers:**
    - Watkins et al., "Q-learning" (1992) - arxiv.org/abs/1509.06461
  - **Key Authors:**
    - Christopher Watkins, Peter Dayan
  - **Key Variants:**
    - van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2015) ...
  - **Python Packages:**
    - `stable-baselines3` (DQN variants)
    - `tensorflow-agents`
    - `pytorch-dqn` (custom implementations)
  ```
- Keep the references in reverse chronological order

## 1. Bayesian Inference

### Bayesian Networks
- **Short description**: Directed acyclic graphical model representing
  probabilistic dependencies; foundation for structured probabilistic reasoning
- **Key papers:**
  - Pearl, "Probabilistic Reasoning in Intelligent Systems" (1988)
  - Koller & Friedman, "Probabilistic Graphical Models" (2009)
- **Key Authors:** Judea Pearl, Daphne Koller, Nir Friedman
- **Python Packages:**
  - `pgmpy` (Probabilistic Graphical Models)
  - `pymc` (Bayesian inference)
  - `networkx` (Graph structures)

### Variational Inference
- **Short description**: Approximate posterior inference via optimization;
  scalable alternative to MCMC for high-dimensional problems
- **Key papers:**
  - Blei et al., "Variational Inference: A Review for Statisticians" (2017) -
    arxiv.org/abs/1601.00670
  - Hoffman et al., "Stochastic Variational Inference" (2013) -
    arxiv.org/abs/1206.7051
- **Key Authors:** David Blei, Michael Jordan, Matthew Hoffman
- **Python Packages:**
  - `pymc` (Variational inference)
  - `edward2` (Probabilistic programming)
  - `pyro` (Probabilistic programming with variational inference)

### Bayesian Neural Networks
- **Short description**: Neural networks with Bayesian treatment of weights;
  principled uncertainty quantification in deep learning
- **Key papers:**
  - Graves, "Practical Variational Inference for Neural Networks" (2011) -
    arxiv.org/abs/1505.05424
  - Gal & Ghahramani, "Dropout as a Bayesian Approximation" (2016) -
    arxiv.org/abs/1506.02142
- **Key Authors:** Yarin Gal, Christos Louizos, Alex Graves
- **Python Packages:**
  - `pymc` (Bayesian neural networks)
  - `edward2` (Bayesian deep learning)
  - `pyro` (Probabilistic deep learning)
  - `tensorflow-probability` (Bayesian modeling)

### MCMC (Markov Chain Monte Carlo)
- **Short description**: Sampling-based inference via Markov chains;
  asymptotically exact posterior approximation for complex models
- **Key papers:**
  - Metropolis et al., "Equations of State Calculations by Fast Computing
    Machines" (1953)
  - Hastings, "Monte Carlo Sampling Methods using Markov Chains and Their
    Applications" (1970)
  - Gelfand & Smith, "Sampling-based approaches to calculating marginal
    densities" (1990)
- **Key Authors:** Nicholas Metropolis, W. Keith Hastings, Andrew Gelfand
- **Python Packages:**
  - `pymc` (MCMC samplers: Metropolis, HMC, NUTS)
  - `stan` (Hamiltonian Monte Carlo)
  - `emcee` (Affine-invariant ensemble sampler)

## 2. Partially Observable / Hidden State Algorithms

### Hidden Markov Models (HMM)
- **Short description**: Probabilistic model for sequences with hidden state
  dynamics; classical approach for time series inference
- **Key papers:**
  - Rabiner, L. R., "A tutorial on hidden Markov models and selected
    applications in speech recognition" (1989) - doi.org/10.1109/5.18626
- **Key Authors:** Lawrence Rabiner
- **Python Packages:**
  - `hmmlearn` (scikit-learn HMM)
  - `pomegranate` (Probabilistic models)
  - `pymc` (Bayesian inference)

### Kalman Filter
- **Short description**: Optimal linear state estimation for Gaussian systems;
  foundational for tracking and sensor fusion
- **Key papers:**
  - Julier & Uhlmann, "Unscented Kalman Filter" (1997)
  - Jazwinski, "Extended Kalman Filter" (1970)
  - Kalman, "A new approach to linear filtering and prediction problems"
    (1960) - doi.org/10.1115/1.3662552
- **Key Authors:** Rudolf Kálmán
- **Python Packages:**
  - `filterpy` (Kalman filters and Bayesian filtering)
  - `scipy.linalg` (Linear algebra for KF)
  - `numpy` (Manual implementations)

### Particle Filtering
- **Short description**: Sequential Monte Carlo for nonlinear/non-Gaussian
  filtering; particle representation of posterior
- **Key papers:**
  - Gordon et al., "Novel approach to nonlinear/non-Gaussian Bayesian state
    estimation" (1993) - doi.org/10.1049/ip-f-2.1993.0015
- **Key Authors:** Neil J. Gordon, David J. Salmond, Adrian F. M. Smith
- **Python Packages:**
  - `filterpy` (Particle filters)
  - `particles` (Sequential Monte Carlo)
  - `PyMC` (Probabilistic inference)

### POMDP (Partially Observable MDP)
- **Short description**: Extension of MDPs for partial observability; solved via
  belief state planning
- **Key papers:**
  - Kaelbling, Littman & Cassandra, "Planning and acting in partially observable
    stochastic domains" (1998) - doi.org/10.1016/S0004-3702(98)00023-X
- **Key Authors:** Leslie Pack Kaelbling, Michael L. Littman, Anthony R
  Cassandra
- **Python Packages:**
  - `pomdp-solve` (Exact POMDP solver)
  - `pomcpow` (Online planning for POMDPs)

### POMCP (Partially Observable Monte Carlo Planning)
- **Short description**: MCTS-based online planning for large POMDPs; scalable
  alternative to belief space planning
- **Key papers:**
  - Silver & Veness, "Monte-Carlo Planning in Large POMDPs" (2010) -
    papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps
- **Key Authors:** David Silver, Joel Veness
- **Python Packages:**
  - Custom implementations
  - `mcts` (General MCTS adaptable to POMDPs)

## 3. Bandit Algorithms

### ε-Greedy
- **Short description**: Simple exploration strategy balancing greedy action
  selection with random exploration
- **Key papers:**
  - Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd
    Edition, 2018)
- **Key Authors:** Richard S. Sutton, Andrew G. Barto

### Upper Confidence Bound (UCB)
- **Short description**: Bandit algorithm using upper confidence bounds to
  balance exploration and exploitation
- **Key papers:**
  - Garivier & Kaufmann, "Optimal Best Arm Identification with Fixed Budget"
    (2016) - arxiv.org/abs/1505.04627
  - Auer et al., "Finite-Time Analysis of the Multiarmed Bandit Problem"
    (2002) - doi.org/10.1023/A:1013689704352
- **Key Authors:** Peter Auer, Nicolò Cesa-Bianchi, Paul Fischer
- **Python Packages:**
  - `vowpalwabbit` (Bandit algorithms)
  - `contextual` (Contextual bandits)
  - `bandito` (Bandit optimization)

### Thompson Sampling
- **Short description**: Bayesian approach to exploration using posterior
  sampling; optimal asymptotic regret bounds
- **Key papers:**
  - Chapelle & Li, "An Empirical Evaluation of Thompson Sampling" (2011) -
    proceedings.neurips.cc/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html
  - Thompson, W. R., "On the likelihood that one unknown probability exceeds
    another" (1933) - doi.org/10.1093/biomet/25.3-4.285
- **Key Authors:** William R. Thompson, Olivier Chapelle, Lihong Li
- **Python Packages:**
  - `vowpalwabbit`
  - `thompson-sampling` (Pure Python)
  - `contextual`

### Contextual Bandits
- **Short description**: Bandit setting with context-dependent rewards; bridges
  bandits and supervised learning
- **Key papers:**
  - Li et al., "A Contextual-Bandit Approach to Personalized News
    Recommendation" (2010) - doi.org/10.1145/1772690.1772758
- **Key Authors:** Lihong Li, Wei Chu, John Langford, Robert E. Schapire
- **Python Packages:**
  - `vowpalwabbit`
  - `contextual` (Full library)
  - `bandit` (Python Bandit)
  - `ml-logger` (Experiment tracking for bandits)

Alternative libraries: mabwiser (for parallelizable contextual multi-armed
bandits), coba (for online contextual bandit research), and space-bandits (for
deep Bayesian approximation)

## 4. Planning and Search Algorithms

### Minimax / Alpha-Beta Pruning
- **Short description**: Classical game tree search with pruning for two-player
  zero-sum games; foundation of game AI
- **Key papers:**
  - Knuth & Moore, "An Analysis of Alpha-Beta Pruning" (1975)
  - Shannon, C. E., "Programming a Computer for Playing Chess" (1950)
- **Key Authors:** Claude Shannon, Donald Knuth, Donald Moore
- **Python Packages:**
  - `python-chess` (Chess engine with minimax)
  - `stockfish` (UCI engine wrapper)
  - Custom game-specific implementations

### A\* Search
- **Short description**: Informed heuristic search combining actual and
  estimated costs; optimal pathfinding algorithm
- **Key papers:**
  - Hart et al., "A Formal Basis for the Heuristic Determination of Minimum Cost
    Paths" (1968)
- **Key Authors:** Peter Hart, Nils Nilsson, Bertram Raphael
- **Python Packages:**
  - `heapq` (Python standard library)
  - `astar` (Pure Python)
  - `networkx` (Graph algorithms)
  - `prm` (Probabilistic Roadmaps)

### MCTS (Monte Carlo Tree Search)
- **Short description**: Best-first planning algorithm using random tree
  exploration; foundation for AlphaGo and game AI
- **Key papers:**
  - Kocsis & Szepesvári, "Bandit based Monte-Carlo Tree Search" (2006) -
    doi.org/10.1007/11871842_29
  - Coulom, "Efficient Selectivity and Backup Operators in Monte-Carlo Tree
    Search" (2006)
- **Key Authors:** Rémi Coulom, Levente Kocsis, Csaba Szepesvári
- **Python Packages:**
  - `pommerman` (with MCTS agents)
  - `mcts` (Pure Python implementation)
  - `alphago-zero-pytorch` (custom)
  - `gym-chess` (chess with MCTS)

### MPC (Model Predictive Control)
- **Short description**: Receding horizon control predicting future states;
  optimal control for systems with known dynamics
- **Key papers:**
  - Qin & Badgwell, "An overview of nonlinear model predictive control
    applications" (2003)
- **Key Authors:** S. Joe Qin, Thomas A. Badgwell
- **Python Packages:**
  - `casadi` (Numeric optimization)
  - `cvxpy` (Convex optimization)
  - `scipy.optimize`
  - `gekko` (Dynamic optimization)

### iLQR (iterative Linear Quadratic Regulator)
- **Short description**: Trajectory optimization via iterative linearization and
  quadratic approximation; local optimal control
- **Key papers:**
  - Li & Todorov, "Iterative Linear Quadratic Regulator Design for Nonlinear
    Biological Movement Systems" (2004)
  - Mayne, "Differential Dynamic Programming" (1966)
- **Key Authors:** David Q. Mayne, Yuval Tassa, Emanuel Todorov
- **Python Packages:**
  - `ilqr` (Pure Python)
  - `PyTorch-based implementations`
  - Part of robotics libraries (Drake, MuJoCo)

### RRT (Rapidly-Exploring Random Tree)
- **Short description**: Randomized motion planning incrementally exploring
  high-dimensional configuration spaces
- **Key papers:**
  - LaValle, "Rapidly-exploring random trees: A new tool for path planning"
    (1998)
- **Key Authors:** Steven M. LaValle
- **Key Variants:** RRT*, Informed RRT*
- **Python Packages:**
  - `pyrrt` (Pure Python)
  - `pybullet` (Includes RRT planning)
  - `ompl` (Open Motion Planning Library)
  - `moveit` (ROS motion planning)

## 5. Core Value-Based RL Algorithms

### Q-Learning
- **Short description**: Foundation of value-based RL; off-policy TD learning
  for optimal action values
- **Key papers:**
  - Watkins & Dayan, "Q-learning" (1992) - doi.org/10.1007/BF00992698
- **Key Authors:** Christopher Watkins, Peter Dayan
- **Key Variants:** van Hasselt et al., "Deep Reinforcement Learning with Double
  Q-learning" (2015) - arxiv.org/abs/1509.06461
- **Python Packages:**
  - `stable-baselines3` (DQN variants)
  - `tensorflow-agents`
  - `pytorch-dqn` (custom implementations)

### SARSA (State-Action-Reward-State-Action)
- **Short description**: First on-policy temporal-difference learning algorithm;
  uses observed action rewards
- **Key papers:**
  - Rummery & Niranjan, "Reinforcement Learning in the Presence of Noise and
    Uncertainty" (1994)
- **Key Authors:** Gavin Rummery, Mahesan Niranjan
- **Python Packages:**
  - `stable-baselines3`
  - Custom RL libraries

### Value Iteration / Policy Iteration
- **Short description**: Dynamic programming methods for solving MDPs with known
  models
- **Key papers:**
  - Puterman, "Markov Decision Processes: Discrete Stochastic Dynamic
    Programming" (2005)
  - Bellman, "Dynamic Programming" (1957)
- **Key Authors:** Richard Bellman, Martin Puterman
- **Python Packages:**
  - `scipy` (for small MDPs)
  - Custom implementations in NumPy

### DQN (Deep Q-Networks)
- **Short description**: Neural network approximation of Q-values; breakthrough
  for high-dimensional control
- **Key papers:**
  - Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013) -
    arxiv.org/abs/1312.5602
- **Key Authors:** Volodymyr Mnih, Koray Kavukcuoglu, David Silver, et al
- **Key Variants:**
  - Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement
    Learning" (2017) - arxiv.org/abs/1710.02298
  - Schaul et al., "Prioritized Experience Replay" (2015) -
    arxiv.org/abs/1511.05952
  - Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning"
    (2015) - arxiv.org/abs/1511.06581
  - Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning"
    (2015) - arxiv.org/abs/1509.06461
- **Python Packages:**
  - `stable-baselines3` (DQN, Double DQN)
  - `tensorflow-agents` (Multiple variants)
  - `pytorch-dqn`
  - `rllib` (Ray RLlib)
  - `keras-rl2`

## 6. Policy-Based Algorithms

### REINFORCE (Policy Gradient)
- **Short description**: On-policy policy gradient method; high variance
  baseline for gradient-based RL
- **Key papers:**
  - Williams, "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning" (1992) - doi.org/10.1007/BF00992696
- **Key Authors:** Ronald J. Williams
- **Python Packages:**
  - `stable-baselines3` (A2C, A3C implementations)
  - `tensorflow-agents`
  - `pytorch` tutorials

### PPO (Proximal Policy Optimization)
- **Short description**: On-policy policy gradient with clipped surrogate loss;
  widely used for continuous control and game AI
- **Key papers:**
  - Schulman et al., "Proximal Policy Optimization Algorithms" (2017) -
    arxiv.org/abs/1707.06347
- **Key Authors:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford,
  Oleg Klimov
- **Python Packages:**
  - `stable-baselines3` (PPO)
  - `rllib` (Ray RLlib)
  - `tensorflow-agents`
  - `openai-baselines`
  - `pytorch-ppo`

### TRPO (Trust Region Policy Optimization)
- **Short description**: On-policy policy optimization with guaranteed monotonic
  improvement; precursor to PPO
- **Key papers:**
  - Schulman et al., "Trust Region Policy Optimization" (2015) -
    arxiv.org/abs/1502.05477
- **Key Authors:** John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan,
  Philipp Moritz
- **Python Packages:**
  - `stable-baselines3` (TRPO)
  - `rllib`
  - `garage` (Reinforcement Learning Toolkit)

### Natural Policy Gradient
- **Short description**: Policy gradient using Fisher information matrix for
  improved convergence; theoretically principled optimization
- **Key papers:**
  - Kakade, "A Natural Policy Gradient" (2001)
  - Amari, "Natural Gradient Works Efficiently in Learning" (1998)
- **Key Authors:** Shun-ichi Amari, Sham Kakade

### Evolutionary Strategies (ES)
- **Short description**: Gradient-free black-box optimization using evolutionary
  population; scalable alternative to gradient-based RL
- **Key papers:**
  - Salimans et al., "Evolution Strategies as a Scalable Alternative to
    Reinforcement Learning" (2016) - arxiv.org/abs/1703.03400
- **Key Authors:** Tim Salimans, Jonathan Ho, Xi Chen, Ilya Sutskever
- **Python Packages:**
  - `evosax` (JAX-based ES)
  - `deap` (Distributed Evolutionary Algorithms in Python)
  - Custom PyTorch implementations

## 7. Actor-Critic Methods

### Actor-Critic (General Framework)
- **Short description**: Combines policy-based (actor) and value-based (critic)
  methods for reduced variance and on-policy learning
- **Key papers:**
  - Konda & Tsitsiklis, "Actor-Critic Algorithms" (2000) -
    doi.org/10.1137/S0363012901385691
- **Key Authors:** Vijay Konda, John Tsitsiklis

### A2C (Advantage Actor-Critic)
- **Short description**: Synchronous version of A3C using advantage function for
  variance reduction
- **Key papers:**
  - Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016) -
    arxiv.org/abs/1602.01783
- **Key Authors:** Volodymyr Mnih, Adrià Puigdomènech Badia, et al
- **Python Packages:**
  - `stable-baselines3` (A2C)
  - `tensorflow-agents`
  - `rllib`
  - `keras-rl2`

### A3C (Asynchronous Advantage Actor-Critic)
- **Short description**: Parallel actor-critic method with asynchronous updates;
  enables efficient distributed training
- **Key papers:**
  - Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016) -
    arxiv.org/abs/1602.01783
- **Key Authors:** Volodymyr Mnih, Adrià Badia, Mircea Gheorghe, et al
- **Python Packages:**
  - `stable-baselines3` (A3C)
  - `tensorflow-agents`
  - `rllib`
  - OpenAI Baselines

### DDPG (Deep Deterministic Policy Gradient)
- **Short description**: Off-policy actor-critic for continuous action spaces
  using deterministic policy and replay buffer
- **Key papers:**
  - Lillicrap et al., "Continuous control with deep reinforcement learning"
    (2015) - arxiv.org/abs/1509.02971
- **Key Authors:** Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel,
  Nicolas Heess, et al
- **Python Packages:**
  - `stable-baselines3` (DDPG)
  - `tensorflow-agents`
  - `rllib`
  - `spinningup` (OpenAI Spinning Up)

### TD3 (Twin Delayed DDPG)
- **Short description**: Improves DDPG by using twin critics, delayed policy
  updates, and action noise to reduce overestimation
- **Key papers:**
  - Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic
    Methods" (2018) - arxiv.org/abs/1802.09477
- **Key Authors:** Scott Fujimoto, Herke van Hoof, David Meger
- **Python Packages:**
  - `stable-baselines3` (TD3)
  - `tensorflow-agents`
  - `rllib`
  - `spinningup`

### SAC (Soft Actor-Critic)
- **Short description**: Maximum entropy off-policy actor-critic combining
  entropy regularization with Q-learning; sample-efficient continuous control
- **Key papers:**
  - Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2018) -
    arxiv.org/abs/1812.05905
  - Haarnoja et al., "Soft Actor-Critic: Off-Policy Deep Reinforcement Learning
    with a Stochastic Actor" (2018) - arxiv.org/abs/1801.01290
- **Key Authors:** Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine
- **Python Packages:**
  - `stable-baselines3` (SAC)
  - `tensorflow-agents`
  - `rllib`
  - `spinningup`

## 8. Integrated Learning & Planning

### Dyna-Q (Integration of Learning and Planning)
- **Short description**: Combines model-free learning with model-based planning
  using learned environment model
- **Key papers:**
  - Moore & Atkeson, "Prioritized Sweeping: Reinforcement Learning with Less
    Data and Less Time" (1993) - doi.org/10.1007/BF00993104
- **Key Authors:** Andrew W. Moore, Christopher G. Atkeson
- **Python Packages:**
  - Custom implementations
  - Part of general RL frameworks

## 9. Deep RL and Foundational Models

### AlphaGo
- **Short description**: Combines deep neural networks with MCTS for superhuman
  Go performance; landmark in AI
- **Key papers:**
  - Silver et al., "Mastering the game of Go with deep neural networks and tree
    search" (2016) - doi.org/10.1038/nature16961
- **Key Authors:** David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, et
  al
- **Key Variants:** AlphaGo Zero, AlphaZero

### AlphaZero
- **Short description**: Self-play RL with neural networks and MCTS; general
  algorithm achieving superhuman performance on multiple games
- **Key papers:**
  - Silver et al., "Mastering Chess and Shogi by Self-Play with a General
    Reinforcement Learning Algorithm" (2017) - arxiv.org/abs/1712.01724
- **Key Authors:** David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis
  Antonoglou, et al
- **Python Packages:**
  - `leela-zero` (Open source Go engine)
  - `leela-chess-zero` (Chess)

### MuZero
- **Short description**: Learned value-equivalent model replacing explicit
  environment; enables planning without forward model
- **Key papers:**
  - Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with
    a Learned Model" (2019) - arxiv.org/abs/1911.08265
- **Key Authors:** Julian Schrittwieser, Thomas Hubert, Amol Mandhane, et al
- **Python Packages:**
  - `mcts` (General MCTS)
  - Custom implementations based on paper

### Hindsight Experience Replay (HER)
- **Short description**: Augments replay buffer with alternative goals; enables
  learning from sparse reward goal-conditioned tasks
- **Key papers:**
  - Andrychowicz et al., "Hindsight Experience Replay" (2017) -
    arxiv.org/abs/1707.01495
- **Key Authors:** Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider,
  et al
- **Python Packages:**
  - `stable-baselines3` (DDPG + HER, SAC + HER)
  - `tensorflow-agents`

## 10. Hierarchical & Modular Approaches

### Hierarchical Reinforcement Learning
- **Short description**: Multi-level abstraction enabling learning at different
  temporal scales; improves sample efficiency on complex tasks
- **Key papers:**
  - Vezhnevets et al., "Feudal Networks for Hierarchical Reinforcement Learning"
    (2017) - arxiv.org/abs/1703.03400
  - Dietterich, "Hierarchical Reinforcement Learning with the MAXQ Value
    Function Decomposition" (2000) - doi.org/10.1613/jair.639
  - Sutton et al., "Between MDPs and semi-MDPs: Learning, Planning, and
    Representing Knowledge at Multiple Temporal Scales" (1999)
- **Key Authors:** Tom Dietterich, Richard S. Sutton, Doina Precup, Satinder
  Singh

### Options Framework
- **Short description**: Temporal abstraction using options (multi-step actions
  with intrinsic policies); hierarchical RL foundation
- **Key papers:**
  - Sutton et al., "Between MDPs and semi-MDPs: Learning, Planning, and
    Representing Knowledge at Multiple Temporal Scales" (1999) -
    doi.org/10.1016/S0004-3702(99)00052-1
- **Key Authors:** Richard S. Sutton, Doina Precup, Satinder Singh

## 11. Offline / Batch Learning

### Batch Q-Learning / Offline RL
- **Short description**: Learning from fixed datasets without online
  interaction; critical for safety-sensitive applications
- **Key papers:**
  - Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning"
    (2020) - arxiv.org/abs/2006.04779
  - Levine et al., "Offline Reinforcement Learning: Tutorial, Review, and
    Perspectives on Open Problems" (2020) - arxiv.org/abs/2005.01643
  - Lange et al., "Batch Reinforcement Learning" (2012) -
    doi.org/10.1007/978-3-642-27645-3_2
- **Key Authors:** Sergey Levine, Aviral Kumar, George Tucker, Justin Fu

### Behavior Cloning
- **Short description**: Supervised learning approach imitating expert behavior
  from demonstrations; simplest imitation learning method
- **Key papers:**
  - Torabi et al., "Behavioral Cloning from Observation" (2018) -
    arxiv.org/abs/1805.01954
- **Python Packages:**
  - `imitation` (Imitation learning library)
  - Custom implementations with PyTorch/TensorFlow

### CQL (Conservative Q-Learning)
- **Short description**: Offline Q-learning with conservative penalty to avoid
  overestimation; enables safe offline learning
- **Key papers:**
  - Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning"
    (2020) - arxiv.org/abs/2006.04779
- **Key Authors:** Aviral Kumar, Aurick Zhou, George Tucker, Sergey Levine
- **Python Packages:**
  - `cql` (Official implementation)
  - `d3rlpy` (Offline RL library)
  - `pyrl` (Policy learning library)

## 12. Game Theory & Multi-Agent Algorithms

### CFR (Counterfactual Regret Minimization)
- **Short description**: Iterative algorithm for computing Nash equilibria in
  imperfect information games; breakthrough for poker
- **Key papers:**
  - Brown et al., "Neural Replicator Dynamics" (2019)
  - Hladík et al., "Solving Imperfect Information Games" (2017)
  - Zinkevich et al., "Regret Minimization in Games with Incomplete Information"
    (2007) -
    papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information
- **Key Authors:** Michael Zinkevich, Michael Bowling
- **Python Packages:**
  - `poker-cfr` (Pure Python CFR)
  - `pykerflop` (Poker with CFR)
  - `imarl` (Imperfect information MARL)

### Nash Equilibrium Solvers
- **Short description**: Computational methods for finding Nash equilibria in
  strategic games
- **Key papers:**
  - Porter, Nudelman & Shoham, "Support enumeration" (2008)
  - Lemke & Howson, "Pivoting algorithm" (1964)
  - Nash, J. F., "Equilibrium points in n-person games" (1950)
- **Python Packages:**
  - `nashpy` (Lemke-Howson algorithm)
  - `gambit` (Gambit Project - equilibrium computation)
  - `pygambit` (Python interface to Gambit)

### QMIX (Mixing Q-Functions)
- **Short description**: Value function factorization for cooperative
  multi-agent RL; enables decentralized execution
- **Key papers:**
  - Rashid et al., "QMIX: Monotonic Value Function Factorisation for
    Decentralised Multi-Agent Reinforcement Learning" (2018) -
    arxiv.org/abs/1803.11485
- **Key Authors:** Tabish Rashid, Mikayel Samvelyan, Christian Schroeder de
  Witt, Gregory Farquhar, et al
- **Python Packages:**
  - `pymarl` (PyMARL - Multi-Agent RL Research Library)
  - `smac` (StarCraft Multi-Agent Challenge)

### MAPPO (Multi-Agent PPO)
- **Short description**: Multi-agent extension of PPO with centralized training
  and decentralized execution
- **Key papers:**
  - Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent
    Games" (2021) - arxiv.org/abs/2108.02556
- **Key Authors:** Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Wang, et al
- **Python Packages:**
  - `pymarl2`
  - `mappo` (Official implementation)
  - `cleanrl` (Clean implementations of RL algorithms)

### MAAC (Multi-Agent Actor-Critic)
- **Short description**: Multi-agent actor-critic with attention mechanism for
  agent communication and coordination
- **Key papers:**
  - Iqbal & Sha, "Actor-Attention-Critic for Multi-Agent Reinforcement Learning"
    (2019) - arxiv.org/abs/1810.02912
- **Key Authors:** Shariq Iqbal, Fei Sha
- **Python Packages:**
  - `maac` (Official PyTorch implementation)
  - `pymarl`

### MADDPG (Multi-Agent DDPG)
- **Short description**: Multi-agent extension of DDPG for mixed
  cooperative-competitive environments
- **Key papers:**
  - Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
    Environments" (2017) - arxiv.org/abs/1706.02891
- **Key Authors:** Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, et al
- **Python Packages:**
  - `maddpg` (Official TensorFlow implementation)
  - `pytorch-maddpg`
  - `openai-multi-agent-particle-envs`

### CommNet (Communication Neural Networks)
- **Short description**: Multi-agent learning with learned communication
  channels; enables emergent agent coordination
- **Key papers:**
  - Sukhbaatar et al., "Learning to Communicate with Deep Multi-Agent
    Reinforcement Learning" (2016) - arxiv.org/abs/1605.06676
- **Key Authors:** Sainbayar Sukhbaatar, Arthur Szlóthy, Gabriel Synnaeve, Rob
  Fergus
- **Python Packages:**
  - `pytorch-geometric` (Graph neural networks)
  - Custom implementations
