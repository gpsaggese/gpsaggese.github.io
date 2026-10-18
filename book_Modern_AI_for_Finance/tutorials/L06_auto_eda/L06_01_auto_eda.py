# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Auto EDA: figures of Lesson 06.01
#
# - This notebook generates the figures and tables of the slides
#   `Lesson06.01_Auto_EDA.smd` from simulated data, and saves the figures as
#   PNG files
# - The running example is a daily equity panel, the input of the momentum
#   strategy of Lessons 01.01 and 02.01, with planted defects
# - The pedagogical arc:
#   - Automated profiling: coverage of a panel by asset and date
#   - Finance-specific checks: stale prices and unadjusted splits
#   - Time-aware EDA: structural breaks and regime-aware summaries
#   - Stylized facts as tests of the data
#   - EDA as snooping: every look is a trial

# %%
# %load_ext autoreload
# %autoreload 2

import logging

# %%
import helpers.hintrospection as hintros
import helpers.hmatplotlib as hmatplo
import helpers.hnotebook as hnotebook

import L06_01_auto_eda_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %%
# Directory of the figures, and prefix of the markdown reference to paste in the
# slides.
out_dir_name = "figures"
markdown_path_prefix = "book_Modern_AI_for_Finance/lectures_source"

# Figure sizes: a single panel fits a column of a slide, two stacked panels
# need more height.
figsize = (6.0, 4.0)
figsize_tall = (6.0, 5.0)

# Seed of all the simulations.
seed = 6

# %% [markdown]
# # Part 1: Automated profiling

# %% [markdown]
# ## Cell 1.1: Panel coverage by asset and date
#
# **Goal**
# - Simulate the daily equity panel of the running example
# - See listings, delistings, and a vendor outage that a column-by-column
#   profiler hides
# - See that a survivor-only database never shows a stock that exits

# %% [markdown]
# **Description**
# - Inputs
#   - `n_stocks`: number of stocks
#   - `n_days`: number of business days
#   - `ipo_fraction`: fraction of the stocks that list after the start
#   - `delisting_price`: price, relative to the listing price, that triggers
#     a delisting
#   - `outage_fraction`, `outage_start`, `outage_days`: share of stocks, first
#     day, and length of the vendor outage
#   - `seed`: seed of the random number generator
#
# - Panels
#   - `Coverage`: one row per stock sorted by listing date, one column per
#     month, darker for a larger fraction of days with a price
#   - `Stocks with a price`: count per day in the full database in blue and in
#     a survivor-only database in red

# %%
# Simulate the panel of close prices.
prices_df = utils.simulate_panel(
    n_stocks=200,
    n_days=2520,
    ipo_fraction=0.3,
    delisting_price=0.3,
    outage_fraction=0.25,
    outage_start=1300,
    outage_days=20,
    seed=seed,
)
print("prices_df.shape=", prices_df.shape)
display(prices_df.iloc[-5:, :5])

# %%
# Plot the coverage of the panel and the number of stocks with a price.
fig = utils.cell1_1_plot_panel_coverage(prices_df, figsize=figsize_tall)
hmatplo.save_fig(
    fig,
    f"{out_dir_name}/L06.01.Panel_Coverage.png",
    print_markdown=True,
    path_prefix=markdown_path_prefix,
)

# %% [markdown]
# **Guided usage**
# - Change `delisting_price` from 0.3 to 0.5
#   - Observe more stocks delist, and the gap between the blue and the red
#     curves widen in the early years
# - Change `outage_days` from 20 to 60
#   - Observe a longer dip in the count and a lighter band in the heatmap

# %% [markdown]
# **Implementation** `cell1_1_plot_panel_coverage()`
# - Computes the fraction of business days with a price for each stock and
#   month
# - Sorts the stocks by first and last date with a price
# - Keeps in the survivor-only database the stocks with a price on the last
#   day

# %%
hintros.print_obj_info(utils.cell1_1_plot_panel_coverage)

# %% [markdown]
# # Part 2: Finance-specific checks

# %% [markdown]
# ## Cell 2.1: Stale prices
#
# **Goal**
# - Make the prices of some illiquid stocks stale, as when a stock does not
#   trade and the database repeats the last price
# - Detect stale prices with the fraction of zero returns and the longest flat
#   run
# - See that stale prices make an equal-weighted index look predictable

# %% [markdown]
# **Description**
# - Inputs
#   - `stale_stocks`: stocks whose prices become stale
#   - `stale_prob`: probability that a stale stock does not trade on a day
#   - `zero_fraction_threshold`: fraction of zero returns above which a stock
#     is flagged
#
# - Panels
#   - `Price`: one stale stock, with the true price in gray, the recorded price
#     in blue, and the zero returns in red
#   - `Detector`: fraction of zero returns against the longest flat run for
#     every stock, with the stale stocks in red

# %%
# Make the prices of the first 40 stocks stale.
stale_stocks = list(prices_df.columns[:40])
stale_df = utils.add_stale_prices(
    prices_df, stocks=stale_stocks, stale_prob=0.4, seed=seed
)

# %%
# Plot one stale stock and the detector.
fig = utils.cell2_1_plot_stale_prices(
    prices_df,
    stale_df,
    stale_stocks=stale_stocks,
    stock=stale_stocks[0],
    n_days_shown=60,
    zero_fraction_threshold=0.1,
    figsize=figsize_tall,
)
hmatplo.save_fig(
    fig,
    f"{out_dir_name}/L06.01.Stale_Prices.png",
    print_markdown=True,
    path_prefix=markdown_path_prefix,
)

# %%
# Compare the lag-1 autocorrelation of an equal-weighted index.
index_acf = {
    "clean": utils.compute_index_autocorrelation(prices_df),
    "stale": utils.compute_index_autocorrelation(stale_df),
}
print("index_acf=", index_acf)

# %% [markdown]
# **Guided usage**
# - Change `stale_prob` from 0.4 to 0.1
#   - Observe the stale stocks move toward the cloud of stocks that trade
#     daily, and the index autocorrelation shrink
# - Change `stale_stocks` to the first 100 stocks
#   - Observe the lag-1 autocorrelation of the index grow: the stale stocks
#     catch up with the market move of the previous day

# %% [markdown]
# **Implementation** `cell2_1_plot_stale_prices()`
# - Plots the recorded price as a step function, so that a flat run is visible
# - Computes, for each stock, the fraction of returns equal to 0 and the
#   longest run of zero returns with `compute_stale_stats()`

# %%
hintros.print_obj_info(utils.cell2_1_plot_stale_prices)

# %% [markdown]
# ## Cell 2.2: Unadjusted split
#
# **Goal**
# - Plant a 2-for-1 split that the database does not adjust for
# - Detect it with a robust z-score and the rule that an unadjusted split with
#   ratio `k` produces a return of `1 / k - 1`

# %% [markdown]
# **Description**
# - Inputs
#   - `split_stock`, `split_date`, `ratio`: stock, date, and ratio of the split
#   - `z_threshold`: absolute robust z-score above which a return is flagged
#   - `ratios`: split ratios to test
#   - `tolerance`: largest distance between a return and `1 / k - 1`
#
# - Panels
#   - `Price`: adjusted price in blue and raw price in red
#   - `Returns`: daily returns of the raw price, with the robust band dashed
#     and the split day in red

# %%
# Plant an unadjusted 2-for-1 split in a stock listed on the whole period.
split_stock = "S150"
raw_df = utils.add_unadjusted_split(
    prices_df, stock=split_stock, split_date="2019-06-03", ratio=2.0
)

# %%
# Plot the split and the robust band.
fig = utils.cell2_2_plot_split_detection(
    prices_df,
    raw_df,
    stock=split_stock,
    ratio=2.0,
    z_threshold=8.0,
    figsize=figsize_tall,
)
hmatplo.save_fig(
    fig,
    f"{out_dir_name}/L06.01.Split_Detection.png",
    print_markdown=True,
    path_prefix=markdown_path_prefix,
)

# %%
# Scan the whole panel for extreme returns and split ratios.
candidates_df = utils.detect_split_candidates(
    raw_df, z_threshold=8.0, ratios=[2.0, 3.0, 1.5, 4.0], tolerance=0.01
)
print("n_flagged=", len(candidates_df))
print("n_split_like=", int(candidates_df["split_ratio"].notna().sum()))
display(candidates_df.sort_values("robust_z").head(10))

# %% [markdown]
# **Guided usage**
# - Change `z_threshold` from 8 to 5
#   - Observe many more flagged returns: heavy tails make extreme returns
#     common, and only the split-ratio rule separates the split from a crash
# - Change `ratio` from 2 to 1.5
#   - Observe a -33.3% return, which matches a 3-for-2 split

# %% [markdown]
# **Implementation** `cell2_2_plot_split_detection()`
# - Computes the band as the median return plus or minus `z_threshold` times
#   the scaled MAD
# - Marks the most negative return as the split candidate

# %%
hintros.print_obj_info(utils.cell2_2_plot_split_detection)

# %% [markdown]
# # Part 3: Time-aware EDA

# %% [markdown]
# ## Cell 3.1: Structural breaks with CUSUM
#
# **Goal**
# - Simulate a strategy with a volatility regime and a later alpha decay
# - See that a CUSUM test detects the break in variance but not the break in
#   mean

# %% [markdown]
# **Description**
# - Inputs
#   - `segments`: list of `(n_days, annual_sharpe, annual_vol)` for each regime
#   - `seed`: seed of the random number generator
#
# - Panels
#   - `Cumulative return`: cumulative return of the strategy, with the true
#     breaks dotted
#   - `CUSUM`: absolute CUSUM statistic of a break in mean in blue and of a
#     break in variance in red, with the 5% critical value dashed

# %%
# Simulate 3 calm years, 1.5 crisis years, 1.5 calm years, and 4 years in
# which the edge is gone.
segments = [
    (756, 1.0, 0.10),
    (378, 1.0, 0.20),
    (378, 1.0, 0.10),
    (1008, 0.0, 0.10),
]
strategy_returns = utils.simulate_strategy_returns(segments=segments, seed=seed)
break_days = [756, 1134, 1512]

# %%
# Plot the cumulative return and the CUSUM statistics.
fig = utils.cell3_1_plot_structural_breaks(
    strategy_returns, break_days=break_days, figsize=figsize_tall
)
hmatplo.save_fig(
    fig,
    f"{out_dir_name}/L06.01.Structural_Breaks.png",
    print_markdown=True,
    path_prefix=markdown_path_prefix,
)

# %% [markdown]
# **Guided usage**
# - Change the Sharpe ratio of the first three regimes from 1.0 to 2.0
#   - Observe the mean CUSUM cross the 5% line: only a large drop in the mean is
#     visible in 10 years of daily data
# - Change the volatility of the crisis from 0.20 to 0.12
#   - Observe the variance CUSUM still cross the line: breaks in volatility are
#     much easier to detect than breaks in mean

# %% [markdown]
# **Implementation** `cell3_1_plot_structural_breaks()`
# - Computes the OLS-CUSUM of the demeaned returns and the cumulative sum of
#   squares with `compute_cusum_stats()`
# - Compares the maximum of each absolute statistic with the 5% critical value
#   1.358 of a Brownian bridge

# %%
hintros.print_obj_info(utils.cell3_1_plot_structural_breaks)

# %% [markdown]
# ## Cell 3.2: Regime-aware summaries
#
# **Goal**
# - Compare the summary of the full sample with the summaries by regime

# %%
# Summarize the strategy over the full sample and by regime.
summary_df = utils.cell3_2_summarize_by_period(
    strategy_returns,
    boundaries=break_days,
    labels=["calm", "crisis", "calm again", "edge gone"],
)
display(summary_df.round(2))

# %% [markdown]
# # Part 4: Stylized facts

# %% [markdown]
# ## Cell 4.1: Stylized facts test suite
#
# **Goal**
# - Compute the stylized-fact statistics of a clean index and of four
#   corrupted versions
# - See that each data defect breaks a different stylized fact

# %%
# Build the clean and corrupted price series.
corrupted_prices = utils.build_corrupted_prices(
    n_days=2520, missing_prob=0.3, split_ratio=2.0, seed=seed
)

# %%
# Compute the test suite on each version.
facts_df = utils.cell4_1_compute_stylized_facts_table(
    corrupted_prices, max_lag=10
)
display(facts_df.round(3))

# %% [markdown]
# # Part 5: EDA as snooping

# %% [markdown]
# ## Cell 5.1: Every look is a trial
#
# **Goal**
# - Test more and more noise features against noise returns, and keep the
#   feature with the best in-sample IC
# - See the best IC grow with the number of looks and vanish on the holdout

# %% [markdown]
# **Description**
# - Inputs
#   - `n_in`, `n_out`: number of in-sample and holdout days
#   - `looks`: numbers of looks to evaluate
#   - `n_reps`: number of repetitions of the simulation
#
# - Panels
#   - `IC`: mean IC of the best feature in-sample in blue and on the holdout in
#     red, with the 10%-90% band shaded and the single-test 5% threshold dashed

# %%
# Simulate the best of up to 1,000 noise features, 5 years in-sample and 5
# years of holdout.
n_in = 1260
looks = [1, 3, 10, 30, 100, 300, 1000]
snooping_df = utils.simulate_snooping(
    n_in=n_in, n_out=1260, looks=looks, n_reps=50, seed=seed
)
display(snooping_df.groupby("looks")[["ic_in", "ic_out"]].mean().round(3))

# %%
# Plot the IC of the selected feature against the number of looks.
fig = utils.cell5_1_plot_snooping(snooping_df, n_in=n_in, figsize=figsize)
hmatplo.save_fig(
    fig,
    f"{out_dir_name}/L06.01.Snooping_Looks.png",
    print_markdown=True,
    path_prefix=markdown_path_prefix,
)

# %% [markdown]
# **Guided usage**
# - Change `n_in` from 1260 to 5040
#   - Observe the best in-sample IC shrink by half: a 4x longer history halves
#     the noise, but does not remove the selection effect
# - Change `looks` to stop at 10
#   - Observe the best IC stay close to the single-test threshold

# %% [markdown]
# **Implementation** `cell5_1_plot_snooping()`
# - Simulates independent Gaussian targets and features with
#   `simulate_snooping()`, so every true IC is 0
# - Keeps, for each number of looks, the feature with the largest absolute
#   in-sample IC, and measures it on the holdout with the in-sample sign

# %%
hintros.print_obj_info(utils.cell5_1_plot_snooping)
