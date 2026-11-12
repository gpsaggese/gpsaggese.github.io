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
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `rllib_utils.py`
  - `rllib.API.ipynb`
  - `rllib.example.ipynb`

# Project

## Project 1 (Fall2026): Optimal Trade Execution

- **Project Objective**: Train an RLlib agent to sell a block of shares within one
  trading day in slices, optimizing for the lowest implementation shortfall (in basis
  points) relative to the arrival price, and compare it with a TWAP schedule
- **Dataset Suggestions**:
  - Daily prices and volume of SPY, downloaded with
    [yfinance](https://github.com/ranaroussi/yfinance), to calibrate volatility and
    average daily volume of a simulated market
  - [CBOE Volatility Index (VIXCLS)](https://fred.stlouisfed.org/series/VIXCLS) from
    FRED to label calm and stressed regimes
- **Tasks**:
  - **Build the Environment**: Implement a custom `gymnasium.Env` that liquidates
    100,000 shares in 20 steps on simulated price paths, with temporary and permanent
    price impact proportional to the traded fraction of the daily volume
  - **Define the Problem**: Use the negative shortfall per step as reward, with a
    `Discrete` action for the fraction of the remaining inventory to sell, and
    compute the shortfall of TWAP and of selling everything at once; calibrate on
    2015-2019 and test on paths calibrated on 2020-2024
  - **Train the Agent**: Configure a deep Q-learning agent with `DQNConfig`, register
    the environment with `register_env`, scale the sampling with `env_runners`, and
    train it with `Algorithm.train`
  - **Evaluate the Policy**: Compute the mean and the standard deviation of the
    shortfall in basis points and the win rate against TWAP over 1,000 test
    episodes, split by calm and stressed VIX regimes
  - **Visualize the Policy**: Plot the learning curve (mean reward per training
    iteration) and the average remaining inventory over time for the agent and TWAP
- **Bonus Ideas (Optional)**: Compare `DQNConfig` with `PPOConfig`; add a variance
  penalty to trade off cost and risk as in the Almgren-Chriss model, and use its
  analytic schedule as a stronger baseline

### Milestones

- Milestone 1: Set up the container and the environment
  - Project tasks: Build the Environment
  - Result: project dir created and container running Ray, and a table with the SPY
    volatility and volume calibration and the shortfall of TWAP and of selling
    everything at once
- Milestone 2: API notebook
  - Project tasks: Train the Agent
  - Result: `rllib.API.ipynb` covering `AlgorithmConfig`, `PPOConfig`, `DQNConfig`,
    `register_env`, `Algorithm.train`, and `Algorithm.evaluate` on CartPole
- Milestone 3: Example notebook
  - Project tasks: Define the Problem, Train the Agent, Evaluate the Policy,
    Visualize the Policy
  - Result: `rllib.example.ipynb` running end to end

## Project 2: Portfolio Allocation with Transaction Costs

- **Project Objective**: Train a PPO agent that sets daily weights across industry
  portfolios, optimizing for the out-of-sample Sharpe ratio net of transaction costs
  and for a low maximum drawdown, and compare it with equal-weight and market
  baselines
- **Dataset Suggestions**:
  - [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html):
    `10_Industry_Portfolios_daily_CSV.zip` for the daily returns and
    `F-F_Research_Data_Factors_daily_CSV.zip` for the market return and the
    risk-free rate
- **Tasks**:
  - **Prepare the Returns**: Load the daily returns of the 10 industry portfolios,
    build features from the last 20 daily returns and the 60-day volatility, and
    split by time into train (2000-2012), validation (2013-2017), and test (2018 to
    the last date)
  - **Build the Environment**: Implement a `gymnasium.Env` with a `Box` observation
    (features and current weights), a `Box` action mapped to target weights with a
    softmax, 10 bps of cost on turnover, and the net log return as reward
  - **Train the Agent**: Configure `PPOConfig` with parallel `env_runners`, and tune
    the learning rate, the discount, and the entropy coefficient on the validation
    period with `tune.Tuner`
  - **Evaluate the Strategy**: Compute annualized return, volatility, Sharpe ratio,
    maximum drawdown, and turnover on the test period over 5 seeds, against an equal
    weight portfolio rebalanced monthly and the market portfolio
  - **Visualize the Strategy**: Plot the cumulative wealth and the drawdown of the
    agent and the baselines, and the weights over time as a stacked area chart
- **Bonus Ideas (Optional)**: Add the VIX from FRED as a regime feature; compare
  `PPOConfig` with `SACConfig`; report the results separately on the 2020 crash and
  the 2022 drawdown

## Project 3: Multi-Agent Market Making

- **Project Objective**: Train two competing market-making agents that quote bid and
  ask prices around a replayed BTC/USDT price, optimizing for profit and loss net of
  inventory risk, and test whether the learned quotes beat an Avellaneda-Stoikov
  baseline
- **Dataset Suggestions**:
  [Binance Public Data, BTCUSDT 1-minute klines](https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/1m/)
  (monthly zip files, about 8 months)
- **Tasks**:
  - **Load the Price Paths**: Download the monthly klines, build the mid-price and
    the trailing 60-minute realized volatility, and split by time into train
    (January-June 2025) and test (July-August 2025)
  - **Build the Multi-Agent Environment**: Implement a `MultiAgentEnv` where each of
    the two agents picks bid and ask offsets with a continuous `Box` action, fills
    arrive with a probability that decays with the distance to the mid-price, and the
    tighter quote wins the flow
  - **Train the Agents**: Configure `PPOConfig` with `multi_agent` and a
    `policy_mapping_fn`, train with parallel `env_runners`, and compare independent
    policies with one shared policy
  - **Evaluate the Agents**: Compute the mean profit and loss per episode, its Sharpe
    ratio, the mean absolute inventory, and the fill rate on the test months, against
    fixed-spread quoting and Avellaneda-Stoikov quoting
  - **Analyze the Behavior**: Plot the quoted spread against volatility, show how the
    quotes skew with inventory, and tabulate the results by volatility tercile
- **Bonus Ideas (Optional)**: Add a third agent that trades on short-term momentum to
  create adverse selection; compare `PPOConfig` with `SACConfig`; add a fee and
  rebate schedule
