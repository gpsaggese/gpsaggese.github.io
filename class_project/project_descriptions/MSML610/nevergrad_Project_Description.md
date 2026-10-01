# Description

Nevergrad is an optimization library developed by Facebook Research for
derivative-free optimization: it minimizes a function using only evaluations, without
gradients. It solves black-box problems such as hyperparameter tuning, noisy or
discrete objectives, and multi-objective trade-offs, with one API over many
algorithms (evolution strategies, differential evolution, Bayesian-style methods, and
the `NGOpt` meta-optimizer). It is worth a 60-minute tutorial because the `ask` and
`tell` interface works on any function, and comparing optimizers under the same
budget takes a few lines.

## Technologies Used

Nevergrad

- Offers a collection of optimization algorithms for black-box optimization problems
- Supports both single-objective and multi-objective optimization
- Provides easy integration with Python, allowing for seamless use in machine
  learning workflows
- Defines search spaces with parametrization classes (`ng.p.Scalar`, `ng.p.Log`,
  `ng.p.Choice`, `ng.p.Dict`)

# Tutorial

- Implement the tutorial "Learn Nevergrad in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - No earlier tutorial or project uses Nevergrad, so read the closest optimization
    work
  - Read `tutorials/Ax_Multi_Objective_Optimization/README.md`
  - Read the `README.md` of the Fall2025 Optuna project
    - `class_project/msml610/Fall2025/projects/UmdTask60_Fall2025_Optuna_Customer_Segmentation_Using_Clustering/`
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare briefly with Optuna and `tutorials/Ax_Multi_Objective_Optimization/` from
  the hyperparameter search point of view
- Deliverables:
  - `nevergrad_utils.py`
  - `nevergrad.API.ipynb`
  - `nevergrad.example.ipynb`

# Project

## Project 1 (Fall2026): Cost-Aware Multi-Objective Portfolio Allocation

- **Project Objective**: Optimize the weights of a multi-asset ETF portfolio by
  trading off net return and risk with the multi-objective optimization of Nevergrad,
  and test the Pareto portfolios out of sample against simple baselines
- **Dataset Suggestions**: Daily adjusted prices of nine ETFs from 2006 on (`SPY`,
  `EFA`, `EEM`, `TLT`, `IEF`, `LQD`, `GLD`, `VNQ`, `DBC`), fetched with
  [yfinance](https://pypi.org/project/yfinance/)
  - Use the [Ten Industry Portfolios (Daily)](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_10_ind_port.html)
    from the Ken French library as a static fallback
- **Tasks**:
  - **Acquire the Data**: Fetch the prices, compute the returns, and define
    walk-forward windows of 5 training years and 1 test year, rolled every year, so
    that all estimates use only the training window
  - **Define the Baselines**: Compute the equal-weight portfolio and the closed-form
    minimum-variance portfolio, and report their out-of-sample Sharpe ratio and
    turnover
  - **Define the Objectives**: Write the two losses, the negative annualized return
    net of a cost of 10 basis points per unit of turnover and the annualized
    volatility, and cap each weight at 40% with a constraint
  - **Optimize with Nevergrad**: Define the search space with `ng.p.Dict` made of
    `ng.p.Array` weights and an `ng.p.Choice` of the rebalancing frequency, run the
    `ask` and `tell` loop with both losses, and extract the `pareto_front` of `NGOpt`
    and `RandomSearch` under the same budget
  - **Evaluate Out of Sample**: Select the front point with the highest training
    Sharpe ratio, apply it to the next year, and plot the fronts and the cumulative
    net return against the baselines, with the Sharpe ratio, the maximum drawdown,
    and the turnover of each portfolio
  - **Analyze the Robustness**: Repeat with five seeds and with costs of 0, 10, and
    25 basis points, and report the mean and the standard deviation of the
    out-of-sample Sharpe ratio
- **Bonus Ideas (Optional)**: Replace the volatility with the 95% CVaR and compare
  the fronts; add the maximum drawdown as a third objective

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Acquire the Data, Define the Baselines
  - Result: project dir created and container running, and a table with the
    out-of-sample Sharpe ratio and turnover of the equal-weight and minimum-variance
    portfolios in each walk-forward test year
- Milestone 2: API notebook
  - Project tasks: Optimize with Nevergrad
  - Result: `nevergrad.API.ipynb` covering the parametrization classes `ng.p.Dict`,
    `ng.p.Array`, and `ng.p.Choice`, the optimizers `NGOpt` and `RandomSearch`, the
    `ask` and `tell` loop, constraints, and multi-objective optimization with
    `pareto_front` on a toy function
- Milestone 3: Example notebook
  - Project tasks: Define the Objectives, Optimize with Nevergrad, Evaluate Out of
    Sample, Analyze the Robustness
  - Result: `nevergrad.example.ipynb` running end to end

## Project 2: Tuning a Moving-Average Crossover Strategy

- **Project Objective**: Tune the parameters of a long-or-cash moving-average crossover
  strategy on a broad equity ETF with Nevergrad, and test with walk-forward evaluation
  whether the in-sample gain survives out of sample
- **Dataset Suggestions**: Daily adjusted prices of the `SPY` ETF from 2005 on, fetched
  with [yfinance](https://pypi.org/project/yfinance/)
  - Use the [Fama-French Daily Factors](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html)
    (`Mkt-RF` plus `RF`) as a static fallback, in case Yahoo Finance blocks the
    requests
- **Tasks**:
  - **Load the Prices**: Fetch the daily prices, compute the returns, cache them on
    disk, and define walk-forward windows of 3 training years and 1 test year, rolled
    every year
  - **Define the Strategy**: Implement the crossover signal with a one-day execution
    lag and a cost of 5 basis points per trade, and compute the Sharpe ratio of
    buy-and-hold and of the fixed 50/200-day crossover as baselines
  - **Optimize with Nevergrad**: Define the search space with `ng.p.Dict` made of
    `ng.p.Scalar` windows (with integer casting), an `ng.p.Choice` between simple and
    exponential averages, and an `ng.p.Log` signal band, and run `NGOpt` with
    `minimize` on the negative training Sharpe, comparing it with `OnePlusOne` and
    `RandomSearch` under the same budget
  - **Evaluate Out of Sample**: In each window, optimize on the training years, apply
    the best parameters to the test year, and report the out-of-sample Sharpe ratio,
    the maximum drawdown, and the gap between the in-sample and out-of-sample Sharpe
    ratio for each optimizer
  - **Visualize the Results**: Plot the best training Sharpe ratio as a function of the
    budget for each optimizer, and the cumulative out-of-sample return against
    buy-and-hold
- **Bonus Ideas (Optional)**: Add the maximum drawdown as a second objective and
  extract the `pareto_front`; repeat on a gold ETF (`GLD`) or a bond ETF (`TLT`) and
  compare the results

## Project 3: Yield Curve Calibration with Nelson-Siegel Models

- **Project Objective**: Fit the Nelson-Siegel and Svensson yield curve models to daily
  U.S. Treasury yields with derivative-free optimizers, and measure the pricing error
  on maturities that are left out of the fit
- **Dataset Suggestions**:
  [Daily Treasury Par Yield Curve Rates](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2024)
  (one page and one CSV per year, with maturities from 1 month to 30 years)
  - Use the years 2022 to 2024, which include an inverted curve
- **Tasks**:
  - **Load the Yield Curves**: Download the yearly CSV files, drop the maturities that
    are not available on every day, and convert the maturities to years
  - **Define the Baselines**: Fit linear interpolation and the Diebold-Li model with a
    fixed decay and least-squares betas, and compute their RMSE in basis points on the
    held-out 5-year and 20-year maturities
  - **Calibrate with Nevergrad**: Define the search space with `ng.p.Dict` made of
    `ng.p.Scalar` betas, an `ng.p.Log` decay, and an `ng.p.Choice` between the
    Nelson-Siegel and Svensson forms, and minimize the fit RMSE on the other
    maturities with `NGOpt`, comparing it with `CMA` and `DE` under the same budget
  - **Evaluate Out of Sample**: Calibrate each of the last 250 trading days, and report
    the mean and the standard deviation over the days of the RMSE in basis points on
    the held-out maturities for each method
  - **Visualize the Results**: Plot the fitted and observed curves on one normal day
    and on one inverted day, and the fitted level, slope, and curvature betas against
    the 10-year minus 2-year spread
- **Bonus Ideas (Optional)**: Warm-start each day from the solution of the previous
  day and compare the budget needed with a cold start; compare the results with
  `scipy.optimize.least_squares`
