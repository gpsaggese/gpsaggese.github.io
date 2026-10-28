# Notebook Outline: nevergrad.example.ipynb

- Blueprint for the end-to-end example notebook of the MSML610 Nevergrad project
  - Spec and design decisions: `Docs/PROJECT_PLAN.md`, sections 6-8
  - Each `## Cell` unit becomes a triplet of notebook cells:
    - Goal and Description markdown
    - One code cell
    - Guided usage and Implementation markdown
- Conventions for the implementation:
  - The notebook only orchestrates: every computation and plot lives in
    `nevergrad_utils.py`, imported as `utils`
  - Pipeline functions keep descriptive names, since `nevergrad.API.ipynb` shares
    the module
  - Plot and widget functions of this notebook use the `cellN_M_` prefix of their
    cell
  - Functions marked (new) do not exist yet; the others are in
    `nevergrad_utils.py` with unit tests
  - The optimizer runs of Parts 4-7 are too slow for live widgets:
    - A utils function runs them once with fixed seeds and caches the results in
      `data/` (git-ignored)
    - The widgets only switch between cached results
  - The notebook runs end-to-end after a kernel restart, and headless through the
    paired `nevergrad.example.py`

# Notebook Front Matter

## Title Cell

- Title: `# Nevergrad: Cost-Aware Multi-Objective Portfolio Allocation`
- Scope bullet: use the multi-objective `ask`/`tell` loop of Nevergrad to choose
  weights and a rebalancing frequency for nine ETFs, then test the chosen
  portfolios on years the optimizer never saw
- The pedagogical arc:
  - Data and walk-forward folds
  - Baselines to beat: equal weight and minimum variance
  - Cost-aware objectives and the 40% cap
  - Pareto fronts with `NGOpt` and `RandomSearch`
  - Out-of-sample evaluation against the baselines
  - Robustness across seeds and cost levels
  - Bonus objectives, and the comparison with Optuna and Ax

## Setup Cells

- No `## Imports` header and no `pip` cell: the Docker image has every package
- Third-party imports cell:
  - `%load_ext autoreload` and `%autoreload 2`
  - `logging`, `matplotlib.pyplot`, `nevergrad`, `numpy`, `pandas`
- Last setup cell:
  - Imports `helpers.hnotebook` and `nevergrad_utils as utils`
  - Calls `hnotebook.config_notebook()`, then `utils.init_loggers(_LOG)` (new)
  - Ends with the `display` fallback, so the paired `.py` file runs as a script

# Part 1: Data and Walk-Forward Folds

## Cell 1.1: Load Returns From the Checksummed Cache

**Goal**
- Load the daily simple returns of the nine ETFs, from 2006-02-07 to 2025-12-31
- Show why every result loads from a cache: Yahoo restates adjusted prices, so a
  new download never reproduces the cached file

**Plots and their descriptions**
- `Returns table`: first rows of the returns, one column per ETF, shown with
  `display()`
- `Comments`:
  - Number of rows, first and last date
  - Fetch date and checksum prefix from `data_provenance.json`
  - Largest absolute daily return and its date

**Guided usage**
- Compare the first date with the launch of DBC in February 2006
  - Observe that the data starts on 2006-02-07: the days before the first DBC
    price were dropped, so every ETF has a return on every day
- Read the largest daily move in `Comments`
  - Observe EEM +22.8% on 2008-10-13: a real crisis move, below the 25% limit
    that `check_returns()` enforces

**Implementation** `utils.load_returns("etf")`, `pandas`
- Reads `data/etf_prices.csv` with `load_data()`, which rejects a file whose
  SHA-256 does not match `data_provenance.json`
- Converts prices to simple returns with `compute_returns()`, then validates
  them with `check_returns()`

## Cell 1.2: Risk, Return, and Correlation per ETF

**Goal**
- Place each ETF on the risk-return trade-off that the optimizer searches:
  annualized return against annualized volatility
- See that correlations change over time, so diversification is not guaranteed

**Plots and their descriptions**
- `Risk-return map`: annualized volatility on the x-axis, annualized return on
  the y-axis, one labeled dot per ETF
- `Correlation heatmap`: 9 x 9 correlations of daily returns, with a diverging
  colormap centered at 0
- `Comments`:
  - Selected period
  - Return and volatility of SPY and IEF
  - Correlation between SPY and IEF

**Widgets**
- `period`: dropdown with the full window or one calendar year (2008, 2017,
  2020, 2022)

**Guided usage**
- Select the full window
  - Observe SPY at 12.2% return and 19.4% volatility, against IEF at 3.6% and
    6.9%: no ETF wins on both objectives
- Select 2008, then 2022
  - Observe the SPY-IEF correlation move from -0.47 to +0.19: in 2022 stocks
    and bonds fell together, so diversification failed

**Implementation** `cell1_2_plot_risk_return(returns, figsize=None)` (new)
- Computes each column with `annualized_return()` and `annualized_volatility()`
- Draws the correlations with `pandas.DataFrame.corr()` and `seaborn.heatmap`,
  driven by an `ipywidgets.Dropdown`

## Cell 1.3: Walk-Forward Folds Without Look-Ahead

**Goal**
- Split the data into 15 folds: 5 training years, then the next test year,
  rolled forward one year at a time
- Builds on Cell 1.1: every estimate later in the notebook sees only the
  training slice of its fold

**Plots and their descriptions**
- `Fold table`: start, end, and number of days of each training and test slice
- `Fold timeline`: one horizontal bar per fold, training years in blue and the
  test year in orange, with 2006-2025 on the x-axis
- `Comments`: number of folds, and length of the first training window

**Guided usage**
- Follow one test year down the timeline
  - Observe that it becomes the last training year of the next fold, never the
    other way around
- Look at the fold of 2011
  - Observe a training window from 2006-02-07: about 4.9 years instead of 5

**Implementation** `utils.get_fold_table(returns)`
- Builds the table from `get_test_years()` and `split_fold()`
- Draws the timeline with `cell1_3_plot_fold_timeline(fold_table)` (new), using
  `matplotlib` `barh`

# Part 2: Baselines to Beat

## Cell 2.1: Drift, Rebalancing, and Turnover

**Goal**
- Show how a fixed-weight portfolio is held through a year:
  - Weights drift as prices move
  - Rebalancing trades them back to target
  - Each trade pays `cost_bps` per unit of turnover
- Explain why a simulation replaces `returns @ w`, which hides the daily trades
  needed to hold fixed weights

**Plots and their descriptions**
- `Weights through the year`: stacked area of the daily weights of the
  equal-weight portfolio, with rebalance days marked
- `Cumulative net return`: one line for the selected year
- `Comments`: frequency, cost, total turnover, total cost paid, and net return

**Widgets**
- `test_year`: dropdown over 2011-2025
- `rebal_freq`: dropdown with monthly, quarterly, and annual
- `cost_bps`: slider over 0-50 basis points

**Guided usage**
- Switch `rebal_freq` from monthly to annual
  - Observe the weights drift further from 1/9 between trades, and turnover
    fall: 0.28 a year on average when monthly, 0.08 when annual
- Raise `cost_bps` from 0 to 50
  - Observe the net-return curve step down only on rebalance days

**Implementation** `cell2_1_plot_rebalancing(returns, figsize=None)` (new)
- Gets the test year with `split_fold()` and the targets with `equal_weights()`
- Runs `simulate_portfolio()` with the chosen frequency and cost
- Needs the daily weights path: add an option to `simulate_portfolio()` that
  returns it (new)

## Cell 2.2: Why Minimum Variance Needs a Solver

**Goal**
- Show that the closed form $w \propto \Sigma^{-1} \mathbf{1}$ goes short in
  every fold, so the long-only portfolio needs a numerical solver
- Show that clipping the shorts to 0 is not the long-only optimum
- Builds on Cell 1.2: the closed form hedges ETFs that move together

**Plots and their descriptions**
- `Weights by method`: grouped bars per ETF for three methods:
  - Closed form, where negative bars are shorts
  - Clip-and-renormalize
  - SLSQP long-only
- `Volatility by fold`: training volatility of clipping and of the solver, over
  the 15 folds
- `Comments`:
  - Number of shorts and most negative weight of the closed form
  - IEF weight of the solver
  - Volatility of both long-only methods

**Widgets**
- `test_year`: dropdown over 2011-2025

**Guided usage**
- Select 2019
  - Observe the closed form short 5 ETFs (down to -46%), and the solver reach
    3.9% training volatility against 4.3% for clipping
- Scan `Volatility by fold`
  - Observe the solver line never above clipping, and IEF taking 61-82% of the
    solver portfolio in every fold

**Implementation** `cell2_2_get_min_variance_table(returns)` (new)
- Solves the closed form with `numpy.linalg.solve`, then clips and renormalizes
- Gets the long-only weights from `utils.min_variance_weights()`: SLSQP on the
  annualized covariance with `ftol=1e-12`

## Cell 2.3: The Walk-Forward Baseline Table

**Goal**
- Score both baselines on every test year:
  - Sharpe ratio with the T-bill rate of the test year
  - Turnover, including the year-start trade from the previous holdings
- Fix the numbers that the Nevergrad portfolio must beat in Part 5

**Plots and their descriptions**
- `Baseline table`: T-bill rate, Sharpe, and turnover of each baseline per test
  year, shown with `display()`
- `Sharpe by year`: grouped bars of both baselines per test year, with the
  T-bill rate as a line
- `Comments`:
  - Mean Sharpe and turnover of each baseline
  - Number of years min variance beats equal weight

**Widgets**
- `cost_bps`: dropdown with 0, 10, and 25 basis points

**Guided usage**
- Look at 2023 and 2024
  - Observe the Sharpe of min variance near 0, then -0.60, while T-bills paid
    4.8-5.1%: a bond-heavy portfolio earned less than a riskless loan
- Move `cost_bps` from 0 to 25
  - Observe the mean Sharpe of both baselines fall by about 0.01: at these costs
    the year-to-year swings dominate

**Implementation** `utils.get_baseline_table(returns, rf_monthly, cost_bps=...)`
- Fits the weights on each training slice, simulates the test year chained from
  the previous holdings, and scores it with `sharpe_ratio()`
- Draws the bars with `cell2_3_plot_baseline_sharpe(table)` (new)

# Part 3: Cost-Aware Objectives and the 40% Cap

## Cell 3.1: Map Any Vector to Long-Only Weights

**Goal**
- Show how the optimizer searches unconstrained real vectors, and how softmax
  maps each one to long-only weights that sum to 1
- Note the caveat: softmax never reaches an exact 0 weight

**Plots and their descriptions**
- `Raw vector`: bars of the 9 unconstrained values
- `Weights`: bars of the softmax output, with the 40% cap as a dashed line
- `Comments`: sum, smallest and largest weight, and whether the cap holds

**Widgets**
- `scale`: slider over 0-5 that multiplies a fixed random vector
- `seed`: integer that picks the random vector

**Guided usage**
- Set `scale` to 0
  - Observe equal weights of 1/9, whatever the vector
- Raise `scale` toward 5
  - Observe the weights concentrate until the largest crosses the 40% cap, while
    the smallest stays above 0

**Implementation** `utils.to_weights(x)` (new)
- Subtracts the maximum before exponentiating, for numerical stability
- Draws the bars with `cell3_1_plot_softmax(scale, seed)` (new)

## Cell 3.2: Two Losses: Volatility and Net Return

**Goal**
- Compute the two losses of a portfolio on a training window:
  `[volatility, -net_return]`, both minimized
- Builds on Cell 2.1: the cost comes from the turnover of the same simulator

**Plots and their descriptions**
- `Losses`: bars of the two losses
- `Comments`: volatility, gross return, annual turnover, and net return

**Widgets**
- `portfolio`: dropdown with equal weight, min variance, and 60/40 SPY/IEF
- `test_year`: dropdown that picks the training window
- `rebal_freq`: dropdown with monthly, quarterly, and annual
- `cost_bps`: slider over 0-50 basis points

**Guided usage**
- Switch `portfolio` from equal weight to min variance
  - Observe the volatility loss fall and the return loss rise: the trade-off of
    Cell 1.2, now as two numbers
- Raise `cost_bps`
  - Observe only the return loss move: costs never change volatility

**Implementation** `utils.portfolio_losses(weights, rebal_freq, train_returns,
cost_bps)` (new)
- Computes the volatility as $\sqrt{w^T \Sigma w} \cdot \sqrt{252}$
- Computes the net return as the annualized mean of the net daily returns of
  `simulate_portfolio()`

## Cell 3.3: The 40% Cap as a Cheap Constraint

**Goal**
- Express the cap the way Nevergrad expects: a function that returns a float
  `<= 0` when the constraint holds
- See which portfolios the cap rejects

**Plots and their descriptions**
- `Constraint values`: bars of `max(weights) - 0.40` for equal weight, the min
  variance portfolio of each fold, and 60/40 SPY/IEF, with a line at 0
- `Comments`: constraint value of each portfolio, and which ones are rejected

**Guided usage**
- Read the bar of equal weight
  - Observe -0.29: every weight is 1/9, far below the cap
- Read the bars of min variance
  - Observe positive values in every fold: IEF takes 61-82%, which is why the
    baseline was left uncapped

**Implementation** `utils.cap_constraint(value)` (new)
- Decodes the weights with `to_weights()` and returns `max(weights) - 0.40`
- Is registered with `register_cheap_constraint()` on the parametrization of
  Cell 4.1

# Part 4: Pareto Fronts With Nevergrad

## Cell 4.1: Search Space Over Weights and Rebalancing

**Goal**
- Build the search space as `ng.p.Dict` of `weights` (`ng.p.Array` of 9 values)
  and `rebal` (`ng.p.Choice` of the three frequencies)
- Fix the seed before the first use of the parametrization, so the runs
  reproduce

**Plots and their descriptions**
- `Sample candidates`: table of 5 random candidates, decoded to weights,
  frequency, and cap constraint value
- `Comments`: number of dimensions, frequency choices, and seed

**Widgets**
- `seed`: integer seed

**Guided usage**
- Change `seed`
  - Observe different candidates
- Return to the first `seed`
  - Observe the same candidates as before: the run is reproducible

**Implementation** `utils.make_parametrization(num_assets, seed)` (new)
- Calls `np.random.seed()` before building the parametrization
- Composes `ng.p.Dict`, `ng.p.Array`, and `ng.p.Choice`, then registers
  `cap_constraint()`

## Cell 4.2: One Fold With the ask/tell Loop

**Goal**
- Run `NGOpt` on one training window with the explicit `ask`/`tell` loop,
  telling the list of two losses, then read `optimizer.pareto_front()`
- Check convergence: the front should stop moving as the budget grows

**Plots and their descriptions**
- `Evaluated candidates`: every candidate in (volatility, net return) space, in
  gray
- `Pareto front`: the non-dominated candidates, colored by rebalancing
  frequency
- `Comments`: budget, number of front points, and volatility range of the front

**Widgets**
- `test_year`: dropdown that picks the training window
- `budget`: dropdown with 300, 1000, and 3000 evaluations (cached runs)

**Guided usage**
- Raise `budget` from 300 to 3000
  - Observe whether the front moves toward lower volatility and higher return,
    and fills in
- Compare 1000 with 3000
  - Check whether the front still moves: if it does, the budget is too small

**Implementation** `utils.run_optimizer(name, train_returns, budget, cost_bps,
seed)` (new)
- Loops `ask()`, decodes the candidate, computes `portfolio_losses()`, and calls
  `tell()`
- Optionally sets an upper-bound reference first with
  `ng.p.MultiobjectiveReference`
- Is cached on disk by `utils.load_or_run_fronts()` (new)

## Cell 4.3: NGOpt Versus RandomSearch at Equal Budget

**Goal**
- Compare the fronts of `NGOpt` and `RandomSearch` at the same budget
- Measure "better front" with the hypervolume: the area the front dominates up
  to a reference point

**Plots and their descriptions**
- `Two fronts`: the fronts of both optimizers on the same axes
- `Hypervolume by fold`: bars of both hypervolumes for each of the 15 folds
- `Comments`: both hypervolumes in the selected fold, and the number of folds
  `NGOpt` wins

**Widgets**
- `test_year`: dropdown over the 15 folds

**Guided usage**
- Step through the folds
  - Observe where the `NGOpt` front dominates the `RandomSearch` front
- Find any fold where `RandomSearch` wins
  - Check whether a larger budget changes the result, and report it rather than
    hide it

**Implementation** `utils.hypervolume_2d(front, reference)` (new)
- Sorts the front by volatility and sums the rectangles it dominates
- Draws both fronts with `cell4_3_plot_fronts(...)` (new)

# Part 5: Out-of-Sample Evaluation

## Cell 5.1: Pick the Highest Training-Sharpe Portfolio

**Goal**
- Apply the fixed selection rule: the front point with the highest training
  Sharpe, using the T-bill rate of the training window
- Keep the chosen weights and rebalancing frequency for the test year

**Plots and their descriptions**
- `Front by training Sharpe`: front points colored by training Sharpe, with the
  selected point starred
- `Selected weights`: bars per ETF with the 40% cap as a dashed line, and the
  chosen frequency in the title
- `Comments`: training Sharpe, training T-bill rate, and chosen frequency

**Widgets**
- `test_year`: dropdown over the 15 folds

**Guided usage**
- Step through consecutive years
  - Observe how much the selected weights change from one year to the next:
    each change is a year-start trade in Cell 5.2

**Implementation** `utils.select_max_sharpe(front, train_returns, rf_train)`
(new)
- Scores each front point with `sharpe_ratio()` on its simulated training
  returns, and returns the best weights and frequency

## Cell 5.2: Test-Year Metrics Versus the Baselines

**Goal**
- Run each selected portfolio through its test year, chained from the previous
  holdings like the baselines
- Compare four metrics per year and on average:
  - Cumulative net return
  - Sharpe with the T-bill rate of the test year
  - Maximum drawdown
  - Turnover

**Plots and their descriptions**
- `Metrics table`: the four metrics of the three strategies per test year, with
  a mean row, shown with `display()`
- `Sharpe by year`: grouped bars of the three strategies per test year
- `Comments`:
  - Mean Sharpe of each strategy
  - Number of years Nevergrad beats both baselines
  - Mean turnover of each strategy

**Guided usage**
- Compare the realized turnover of the Nevergrad portfolio with its training
  estimate
  - Observe the gap: the training objective never sees the year-start switch
- Find the years Nevergrad wins and loses
  - Compare them with the rate and correlation regimes of Cells 1.2 and 2.3

**Implementation** `utils.evaluate_out_of_sample(...)` (new)
- Simulates each test year with `simulate_portfolio()`, chained through the end
  weights
- Computes the drawdown with `utils.max_drawdown(net)` (new): the largest
  peak-to-trough fall of the cumulative net return

## Cell 5.3: Cumulative Net Return Over 2011-2025

**Goal**
- Show the whole out-of-sample record of the three strategies as one curve each
- See the drawdowns behind the averages of Cell 5.2

**Plots and their descriptions**
- `Cumulative net return`: one line per strategy, on a log scale
- `Drawdown`: the fall below the previous peak of each strategy over time
- `Comments`: total return, worst drawdown, and its date for each strategy

**Guided usage**
- Look at 2022
  - Compare the depth of the drawdown of the three strategies
- Compare the final values
  - Check whether the strategy with the best mean Sharpe also ends highest

**Implementation** `cell5_3_plot_cumulative_returns(...)` (new)
- Compounds the chained daily net returns of each strategy
- Draws the drawdown from the running maximum of each curve

# Part 6: Robustness Across Seeds and Costs

## Cell 6.1: Seeds Versus Cost Levels

**Goal**
- Repeat the whole pipeline for 5 seeds and costs of 0, 10, and 25 basis points
- Separate optimizer noise (the spread across seeds) from cost sensitivity (the
  slope across costs)

**Plots and their descriptions**
- `Robustness table`: mean and standard deviation of the out-of-sample Sharpe
  across seeds, per cost level
- `Sharpe versus cost`: Nevergrad as mean +- standard deviation error bars, with
  both baselines as lines, re-run at each cost
- `Comments`: the spread across seeds at each cost, and the gap to the best
  baseline

**Guided usage**
- Compare the size of the error bars with the gap to the baselines
  - Check whether the edge, if any, is larger than the seed noise
- Compare the slopes from 0 to 25 basis points
  - Check whether Nevergrad loses more to costs than the baselines

**Implementation** `utils.run_robustness_sweep(returns, rf_monthly, seeds,
costs)` (new)
- Reuses the Part 4 and Part 5 functions with only the seed and cost changed,
  cached on disk

# Part 7: Bonus Objectives and Comparison

## Cell 7.1: CVaR Instead of Volatility

**Goal**
- Replace volatility with the 95% CVaR: the average loss on the worst 5% of
  days
- Check whether optimizing tail risk changes the selected portfolios

**Plots and their descriptions**
- `Two fronts`: the volatility-based and the CVaR-based fronts, both drawn in
  (volatility, net return) space
- `Selected weights`: the weights selected under each risk measure, side by side
- `Comments`: volatility and CVaR of both selected portfolios

**Widgets**
- `test_year`: dropdown over the 15 folds

**Guided usage**
- Step through the folds
  - Check whether the CVaR selection holds less of the ETFs with the largest
    crashes (EEM, VNQ)

**Implementation** `utils.cvar(returns, alpha=0.95)` (new)
- Averages the returns below the 5% quantile, found with `numpy.quantile`

## Cell 7.2: Maximum Drawdown as a Third Objective

**Goal**
- Add the maximum drawdown as a third loss, so the front becomes a surface in
  three dimensions
- See how selecting one point changes with three objectives

**Plots and their descriptions**
- `Front projections`: three 2-D scatter plots of the front, one per pair of
  objectives
- `Comments`: number of front points with two and with three objectives

**Guided usage**
- Compare the number of front points with Cell 4.2
  - Observe whether more candidates become non-dominated as objectives are added

**Implementation** `utils.portfolio_losses(..., objectives=...)` (new option)
- Appends `max_drawdown()` of the simulated training returns to the losses

## Cell 7.3: Nevergrad, Optuna, and Ax for Hyperparameter Search

**Goal**
- Compare the three tools from a hyperparameter-search point of view, as the
  spec requires

**Plots and their descriptions**
- `Comparison table`: a markdown table with one column per tool and rows for:
  - Search-space definition
  - Constraints
  - Multi-objective output
  - Algorithms
  - The `ask`/`tell` interface
- No code cell: the table is the content

**Guided usage**
- Match each row of the table to the cell of this notebook where it appeared
  - E.g., the `ng.p.Dict` of Cell 4.1 against a study in Optuna and a search
    space in Ax

**Implementation** References only
- `tutorials/Ax_Multi_Objective_Optimization/README.md`
- The Fall2025 Optuna project in `class_project/msml610/Fall2025/projects/`:
  - `UmdTask60_Fall2025_Optuna_Customer_Segmentation_Using_Clustering/README.md`

# Summary: What the Results Show

- Final markdown cell with 2-4 standalone takeaways, written once the results
  of Parts 4-6 exist
  - Whether Nevergrad beats the baselines out of sample, and by how much
  - Whether the edge survives the seed noise and the cost levels
