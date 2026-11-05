# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Nevergrad: Cost-Aware Multi-Objective Portfolio Allocation
#
# - This notebook uses the multi-objective `ask`/`tell` loop of Nevergrad to
#   choose weights and a rebalancing frequency for nine ETFs, then tests the
#   chosen portfolios on years the optimizer never saw
# - The pedagogical arc:
#   - Data and walk-forward folds
#   - Baselines to beat: equal weight and minimum variance
#   - Cost-aware objectives and the 40% cap
#   - Pareto fronts with `NGOpt` and `RandomSearch`
#   - Out-of-sample evaluation against the baselines
#   - Robustness across seeds and cost levels
#   - Bonus objectives, and the comparison with Optuna and Ax

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import numpy as np
import pandas as pd

# %%
import helpers.hintrospection as hintros
import helpers.hio as hio
import helpers.hnotebook as hnotebook

import nevergrad_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# # Part 1: Data and Walk-Forward Folds

# %% [markdown]
# ## Cell 1.1: Load Returns From the Checksummed Cache
#
# **Goal**
# - Load the daily simple returns of the nine ETFs, from 2006-02-07 to
#   2025-12-31
# - Show why every result loads from a cache: Yahoo restates adjusted prices, so
#   a new download never reproduces the cached file

# %% [markdown]
# **Description**
# - Inputs
#   - `data_source`: `"etf"` for the nine ETFs, or `"french"` for the Ken French
#     fallback if Yahoo blocks requests
#
# - Panels
#   - `Returns table`: first rows of the returns, one column per ETF
#   - `Comments`:
#     - Shape, first date, and last date
#     - Fetch date and checksum prefix, from `data_provenance.json`
#     - Largest absolute daily return, with its ETF and date

# %%
# Load the cached daily returns of the nine ETFs.
# TODO(Jake): set `data_source`, load the returns with `utils.load_returns()`,
# read the provenance record with `hio.from_json(utils.PROVENANCE_PATH)`, print
# the comments, then display the first rows.

# %% [markdown]
# **Guided usage**
# - Compare the first date with the launch of DBC in February 2006
#   - Observe that the data starts on 2006-02-07: the days before the first DBC
#     price were dropped, so every ETF has a return on every day
# - Read the largest daily move
#   - Observe EEM +22.8% on 2008-10-13: a real crisis move, below the 25% limit
#     that `check_returns()` enforces

# %% [markdown]
# **Implementation** `utils.load_returns(data_source)`
# - Reads `data/etf_prices.csv` with `load_data()`, which rejects a file whose
#   SHA-256 does not match `data_provenance.json`
# - Converts prices to simple returns with `compute_returns()`
# - Validates them with `check_returns()`: no missing values, dates in order, and
#   no daily move above 25%

# %%
hintros.print_obj_info(utils.load_returns)

# %% [markdown]
# ## Cell 1.2: Risk, Return, and Correlation per ETF
#
# **Goal**
# - Place each ETF on the risk-return trade-off that the optimizer searches:
#   annualized return against annualized volatility
# - See that correlations change over time, so diversification is not
#   guaranteed

# %% [markdown]
# **Description**
# - Inputs
#   - `period`: `"all"` for 2006-2025, or one calendar year, e.g., `2008` or
#     `2022`
#
# - Panels
#   - `Risk-return table`: annualized return and volatility of each ETF over
#     `period`
#   - `Correlation matrix`: 9 x 9 correlations of daily returns over `period`
#   - `Comments`: return and volatility of SPY and IEF, and their correlation

# %%
# Compare the annualized return and volatility of each ETF, and their
# correlations.
# TODO(Jake): set `period`, select its rows of `returns`, apply
# `utils.annualized_return` and `utils.annualized_volatility` to each column,
# display both tables, and print the SPY-IEF correlation.

# %% [markdown]
# **Guided usage**
# - Set `period` to `"all"`
#   - Observe SPY at 12.2% return and 19.4% volatility, against IEF at 3.6% and
#     6.9%: no ETF wins on both objectives
# - Set `period` to `2008`, then `2022`
#   - Observe the SPY-IEF correlation move from -0.47 to +0.19: in 2022 stocks
#     and bonds fell together, so diversification failed

# %% [markdown]
# **Implementation** `utils.annualized_return(returns)`,
# `utils.annualized_volatility(returns)`
# - Scales the mean daily return by 252 trading days
# - Scales the sample standard deviation of daily returns by $\sqrt{252}$, which
#   assumes the daily returns are uncorrelated
# - Leaves the correlations to `pandas.DataFrame.corr()`

# %%
hintros.print_obj_info(utils.annualized_return)
hintros.print_obj_info(utils.annualized_volatility)

# %% [markdown]
# ## Cell 1.3: Walk-Forward Folds Without Look-Ahead
#
# **Goal**
# - Split the data into 15 folds: 5 training years, then the next test year,
#   rolled forward one year at a time
# - Builds on Cell 1.1: every estimate later in the notebook sees only the
#   training slice of its fold

# %% [markdown]
# **Description**
# - Inputs
#   - `returns`: the daily returns of Cell 1.1
#
# - Panels
#   - `Fold table`: start, end, and number of days of each training and test
#     slice, one row per test year
#   - `Comments`: number of folds, and number of days in the first training
#     window

# %%
# Build the walk-forward folds and show their boundaries.
# TODO(Jake): call `utils.get_fold_table(returns)`, print the comments, then
# display the table.

# %% [markdown]
# **Guided usage**
# - Follow one test year down the table
#   - Observe that it becomes the last training year of the next fold, never the
#     other way around
# - Look at the fold of 2011
#   - Observe a training window from 2006-02-07 with 1,235 days: about 4.9 years
#     instead of 5

# %% [markdown]
# **Implementation** `utils.get_fold_table(returns, num_train_years=5)`
# - Finds the test years with `get_test_years()`: every year after the first 5
# - Cuts each fold with `split_fold()`, which fails unless the training data ends
#   before the test year starts
# - Records the first date, last date, and number of days of each slice

# %%
hintros.print_obj_info(utils.get_fold_table)

# %% [markdown]
# # Part 2: Baselines to Beat

# %% [markdown]
# ## Cell 2.1: Drift, Rebalancing, and Turnover
#
# **Goal**
# - Show how a fixed-weight portfolio is held through a year:
#   - Weights drift as prices move
#   - Rebalancing trades them back to target
#   - Each trade pays `cost_bps` per unit of turnover
# - Explain why a simulation replaces `returns @ w`, which hides the daily trades
#   needed to hold fixed weights

# %% [markdown]
# **Description**
# - Inputs
#   - `test_year`: the year to simulate, 2011-2025
#   - `cost_bps`: cost per unit of turnover, 0-50 basis points
#
# - Panels
#   - `Turnover table`: one row per rebalancing frequency (monthly, quarterly,
#     annual), with the total turnover and the net return of the equal-weight
#     portfolio in `test_year`
#   - `Comments`: `test_year` and `cost_bps`

# %%
# Hold the equal-weight portfolio through one year at each rebalancing frequency.
# TODO(Jake): set `test_year` and `cost_bps`, take the test slice with
# `utils.split_fold()`, run `utils.simulate_portfolio()` with
# `utils.equal_weights()` for each frequency, and display the turnover table.

# %% [markdown]
# **Guided usage**
# - Set `test_year` to `2023`
#   - Observe a turnover of 0.214 when monthly, 0.093 when quarterly, and 0.000
#     when annual: a single year starts at its target, so annual never trades
#     here; in the chained table of Cell 2.3 it pays the year-start trade
# - Raise `cost_bps` from 10 to 50
#   - Observe the net return of each row fall by about 0.004 times its turnover,
#     and the annual row not move

# %% [markdown]
# **Implementation** `utils.simulate_portfolio(returns, target_weights,
# rebal_freq, cost_bps, initial_weights)`
# - Flags the first trading day of each month, quarter, or year with
#   `get_rebalance_days()`
# - On each flagged day, measures the turnover $\sum_i |w_i^{target} - w_i|$
#   with `compute_turnover()`, then resets the weights to target
# - Earns the day's gross return $w \cdot r$, minus `cost_bps` / 10,000 times the
#   turnover
# - Lets the weights drift: $w \leftarrow w (1 + r) / (1 + w \cdot r)$

# %%
hintros.print_obj_info(utils.simulate_portfolio)

# %% [markdown]
# ## Cell 2.2: Why Minimum Variance Needs a Solver
#
# **Goal**
# - Show that the closed form $w \propto \Sigma^{-1} \mathbf{1}$ goes short in
#   every fold, so the long-only portfolio needs a numerical solver
# - Show that clipping the shorts to 0 is not the long-only optimum
# - Builds on Cell 1.2: the closed form hedges ETFs that move together

# %% [markdown]
# **Description**
# - Inputs
#   - `test_year`: the fold whose weights are compared, 2011-2025
#
# - Panels
#   - `Weights by method`: closed-form, clipped, solver, and equal weights of each
#     ETF in the fold of `test_year`; negative weights are shorts
#   - `Method table`: one row per fold, with the number of closed-form shorts,
#     the training volatility of clipping and of the solver, and the largest
#     weight of the solver
#   - `Comments`: the row of `test_year` in the method table

# %%
# Compare the closed form, clipping, and the solver on the training slices.
# TODO(Jake): set `test_year`, take its training slice with `utils.split_fold()`,
# build `Weights by method` from `utils.closed_form_min_variance_weights()`,
# `utils.clip_and_renormalize()`, `utils.min_variance_weights()`, and
# `utils.equal_weights()`, then display it and
# `utils.cell2_2_get_min_variance_table(returns)`.

# %% [markdown]
# **Guided usage**
# - Set `test_year` to `2023`
#   - Observe the closed form short 4 ETFs, mostly TLT at -0.44 against IEF at
#     +1.41: a hedge between two Treasury funds that move together
#   - Observe clipping reach 6.16% training volatility, against 5.76% for the
#     solver
# - Scan the `solver_vol` and `clip_vol` columns
#   - Observe the solver never above clipping: the function asserts it, since
#     only a solver that stopped early could lose
# - Scan the `top_weight` column
#   - Observe IEF at 61-82% in every fold, above the 40% cap of Part 3: the
#     baseline is deliberately left uncapped, as the spec defines it, and
#     `utils.min_variance_weights(train, max_weight=0.4)` gives a capped version

# %% [markdown]
# **Implementation** `utils.cell2_2_get_min_variance_table(returns,
# num_train_years=5)`
# - Solves the closed form with `closed_form_min_variance_weights()`:
#   `numpy.linalg.solve` on the annualized training covariance
# - Clips and renormalizes it with `clip_and_renormalize()`
# - Solves the long-only problem with `min_variance_weights()`: SLSQP on the
#   annualized covariance, with analytic gradients and `ftol=1e-12`
# - Compares the training volatilities with `annualized_volatility()`

# %%
hintros.print_obj_info(utils.cell2_2_get_min_variance_table)

# %% [markdown]
# ## Cell 2.3: The Walk-Forward Baseline Table
#
# **Goal**
# - Score both baselines on every test year:
#   - Sharpe ratio with the T-bill rate of the test year
#   - Turnover, including the year-start trade from the previous holdings
# - Fix the numbers that the Nevergrad portfolio must beat in Part 5

# %% [markdown]
# **Description**
# - Inputs
#   - `cost_bps`: cost per unit of turnover: 0, 10, or 25 basis points
#   - `rebal_freq`: rebalancing frequency of both baselines, monthly by default
#
# - Panels
#   - `Baseline table`: one row per test year, with the T-bill rate (`rf`) and
#     the Sharpe ratio and turnover of equal weight (`ew_`) and min variance
#     (`mv_`)
#   - `Comments`: mean of each column, and number of years min variance beats
#     equal weight

# %%
# Score both baselines on every test year.
# TODO(Jake): set `cost_bps` and `rebal_freq`, call
# `utils.get_baseline_table(returns, utils.load_risk_free_rate(), ...)`, display
# the table and its column means, and print how often min variance wins.

# %% [markdown]
# **Guided usage**
# - Read the means at 10 basis points
#   - Observe a mean Sharpe of 0.77 for min variance against 0.69 for equal
#     weight, but min variance wins only 7 of the 15 years
# - Look at 2022
#   - Observe both Sharpe ratios negative (-1.20 and -1.59): stocks and bonds
#     fell together, as the +0.19 correlation of Cell 1.2 showed
# - Look at 2023 and 2024
#   - Observe min variance near 0, then -0.60, while T-bills paid 4.83% and
#     5.13%: a bond-heavy portfolio earned less than a riskless loan, and the
#     2023 training window averaged only 1.15%, which is why each test year uses
#     its own rate
# - Compare the turnover columns
#   - Observe equal weight trading more in total (0.28 against 0.22 a year): its
#     volatile ETFs drift and need monthly corrections, while the IEF-heavy min
#     variance portfolio barely drifts
# - Move `cost_bps` from 0 to 25
#   - Observe both mean Sharpe ratios fall by about 0.01: at these costs the
#     year-to-year swings dominate

# %% [markdown]
# **Implementation** `utils.get_baseline_table(returns, rf_monthly, rebal_freq,
# cost_bps, num_train_years)`
# - Fits equal weights and `min_variance_weights()` on each training slice
# - Simulates the test year with `simulate_portfolio()`, starting from the
#   holdings that the same baseline ended the previous year with
# - Scores it with `sharpe_ratio()`, using `annualized_risk_free_rate()` of the
#   T-bills of the test year
# - Sums the daily turnover over the test year

# %%
hintros.print_obj_info(utils.get_baseline_table)
