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
# # Data engineering: figures of Lesson 04.01
#
# - This notebook generates the figures and tables of the slides
#   `Lesson04.01_Data_Engineering.smd` from simulated data, and saves the
#   figures as PNG files
# - The running example is a value signal built on quarterly fundamentals, with
#   release dates, restatements, delistings, splits, and dividends planted in
#   the data
# - The pedagogical arc:
#   - Point-in-time correctness: as-of joins, release lags, and restatements
#   - Survivorship bias: the delisting return that a database leaves out
#   - Reference data: split and dividend adjustment, and time zones

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import pandas as pd

# %%
import helpers.hintrospection as hintros
import helpers.hmatplotlib as hmatplo
import helpers.hnotebook as hnotebook

import L04_01_data_engineering_utils as utils

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
seed = 4

# %% [markdown]
# # Part 1: Point-in-time correctness

# %% [markdown]
# ## Cell 1.1: Simulate fundamentals with release dates and restatements
#
# **Goal**
# - Simulate the quarterly fundamentals of 1000 stocks over 15 years
# - See that each record has a period end (`datadate`) and a later release date
#   (`rdq`)
# - See that some records are restated after the release

# %%
# Simulate the fundamentals of the running example.
fund_df = utils.simulate_fundamentals(
    n_stocks=1000,
    n_quarters=60,
    first_quarter_end="2005-03-31",
    persistence=0.9,
    restated_fraction=0.15,
    seed=seed,
)
print("fund_df.shape=", fund_df.shape)
display(fund_df.head())

# %%
# Summarize the release lag, i.e., the days between the period end and the
# release.
print("lag_days.describe()=")
display(fund_df["lag_days"].describe())
print("share restated=", fund_df["restate_date"].notna().mean())

# %%
# Simulate the monthly returns that react to the news when it is released.
month_ends = utils.make_month_ends(fund_df, n_extra_months=8)
returns_df = utils.simulate_monthly_returns(
    fund_df,
    month_ends,
    monthly_drift=0.006,
    monthly_vol=0.08,
    dof=4.0,
    value_beta=0.003,
    surprise_beta=0.02,
    restate_beta=0.02,
    seed=seed,
)
print("returns_df.shape=", returns_df.shape)

# %% [markdown]
# ## Cell 1.2: As-of join on a small example
#
# **Goal**
# - Join fundamentals to monthly decisions in two ways, on the period end and
#   on the release date
# - See that the only change is the key of the join, and that the first key
#   uses numbers that were not public yet

# %%
# Build one stock with three quarters of fundamentals, and monthly decisions.
fund_small_df, decision_df = utils.build_asof_example()
display(fund_small_df)

# %%
# Join on the period end: each decision takes the latest quarter that has
# already ended.
naive_df = pd.merge_asof(
    decision_df,
    fund_small_df,
    left_on="decision_date",
    right_on="datadate",
    by="stock",
)
display(naive_df)
# Outcome: on 2024-01-31 the decision uses the quarter ended 2023-12-31, but
# that quarter was released on 2024-02-22.

# %%
# Join on the release date: each decision takes the latest quarter that has
# already been released.
asof_df = pd.merge_asof(
    decision_df,
    fund_small_df,
    left_on="decision_date",
    right_on="rdq",
    by="stock",
)
display(asof_df)
# Outcome: on 2024-01-31 there is no value, and each value appears only after
# its release date.

# %% [markdown]
# ## Cell 1.3: Release lag and look-ahead
#
# **Goal**
# - Measure the information coefficient (IC) of the value signal when the value
#   is available a fixed lag after the period end
# - Compare it with the IC when the value is available on the true release
#   date
# - See that a short lag assumption leaks the earnings news into the signal

# %%
# Compute the mean IC for each assumed lag after the quarter end.
lags = [0, 30, 45, 60, 75, 90, 105, 120]
ic_by_lag_s = utils.compute_ic_by_assumed_lag(
    fund_df, returns_df, lags=lags, min_stocks=500
)
display(ic_by_lag_s.to_frame())

# %%
# Compute the mean IC when each value is available on its release date.
signal_asof_df = utils.build_signal(
    fund_df, month_ends, mode="first", assumed_lag_days=0
)
ic_asof_s = utils.compute_monthly_ic(signal_asof_df, returns_df, min_stocks=500)
ic_asof = float(ic_asof_s.mean())
print("ic_asof=", ic_asof)

# %% [markdown]
# **Description**
# - Inputs
#   - `lags`: assumed days between the period end and the availability of the
#     value
#   - `ic_asof`: mean IC when the value is available on the true release date
#
# - Panels
#   - `Release lag`: histogram of the days between the period end and the
#     release, with the mean in red
#   - `IC by assumed lag`: mean IC against the assumed lag in blue, and the
#     as-of IC as a green dashed line

# %%
# Plot the release lags and the IC against the assumed lag.
fig = utils.cell1_3_plot_release_lag(
    fund_df, ic_by_lag_s, ic_asof, figsize=figsize_tall
)
hmatplo.save_fig(
    fig,
    f"{out_dir_name}/L04.01.Release_Lag.png",
    print_markdown=True,
    path_prefix=markdown_path_prefix,
)

# %% [markdown]
# **Guided usage**
# - Compare the IC at an assumed lag of 0 and 30 days with the IC at 120 days
#   - Observe the IC fall to the as-of level only once the assumed lag exceeds
#     almost all the true lags
# - Compare the IC at an assumed lag of 45 days with the as-of line
#   - Observe that a common rule of thumb still leaks the news of the slow
#     reporters

# %% [markdown]
# **Implementation** `cell1_3_plot_release_lag()`
# - Draws the histogram of `lag_days` and marks its mean
# - Plots the mean IC of each assumed lag, computed by
#   `compute_ic_by_assumed_lag()` with `build_signal(mode="period_end_lag")`
# - Draws the IC with true release dates as a horizontal line

# %%
hintros.print_obj_info(utils.cell1_3_plot_release_lag)

# %% [markdown]
# ## Cell 1.4: Restatements and vintages
#
# **Goal**
# - Compare three ways to read a restated value: the latest vintage, the point
#   in time vintage, and the first reported value
# - See that the latest vintage knows the restatement before it happens, and
#   inflates the IC

# %%
# Compute the mean IC of the signal under each vintage rule.
ic_by_vintage_s = utils.compute_ic_by_vintage_rule(
    fund_df, returns_df, min_stocks=500
)
display(ic_by_vintage_s.to_frame())

# %%
# Pick a restated record with a typical release lag.
record = utils.pick_restated_record(fund_df)
display(record.to_frame("record"))

# %% [markdown]
# **Description**
# - Inputs
#   - `record`: one restated record, with its period end, release date, and
#     restatement date
#   - `ic_by_vintage_s`: mean IC under each vintage rule
#
# - Panels
#   - `Vintages`: the value known at each date in blue, and the value that a
#     database keeping only the latest vintage shows today in red
#   - `IC by vintage rule`: mean IC of the latest vintage, the point in time
#     vintage, and the first reported value

# %%
# Plot the vintages of one record and the IC of each vintage rule.
fig = utils.cell1_4_plot_restatements(
    record, ic_by_vintage_s, figsize=figsize_tall
)
hmatplo.save_fig(
    fig,
    f"{out_dir_name}/L04.01.Restatements.png",
    print_markdown=True,
    path_prefix=markdown_path_prefix,
)

# %% [markdown]
# **Guided usage**
# - Compare the red dashed line with the blue line before the restatement date
#   - Observe that the latest vintage shows, from the quarter end, a value that
#     nobody knew until the restatement
# - Compare the first bar with the second bar
#   - Observe the gain of the latest vintage over the point in time vintage:
#     the revision predicts the price reaction that follows it

# %% [markdown]
# **Implementation** `cell1_4_plot_restatements()`
# - Draws the value known between the release and the restatement, and the
#   value known after the restatement, as solid blue segments
# - Draws the restated value over the whole period as a red dashed segment
# - Plots the three mean IC values as bars

# %%
hintros.print_obj_info(utils.cell1_4_plot_restatements)

# %% [markdown]
# ## Cell 1.5: A guard that fails loudly
#
# **Goal**
# - Check that every joined row was known before the decision
# - See that the guard counts the future rows of the period end join, and
#   passes on the as-of join

# %%
# Count the rows that use a value released after the decision.
n_future_naive = utils.count_future_records(
    naive_df, decision_col="decision_date", knowledge_col="rdq"
)
print("n_future_naive=", n_future_naive)
# Outcome: the period end join uses future values on five of the eight
# decisions.

# %%
# Assert that the as-of join has no future row; the same call on `naive_df`
# raises an error.
utils.assert_point_in_time(
    asof_df, decision_col="decision_date", knowledge_col="rdq"
)
n_future_asof = utils.count_future_records(
    asof_df, decision_col="decision_date", knowledge_col="rdq"
)
print("n_future_asof=", n_future_asof)

# %% [markdown]
# # Part 2: Survivorship bias

# %% [markdown]
# ## Cell 2.1: The delisting return
#
# **Goal**
# - Simulate stocks that delist for performance reasons, cheap stocks more
#   often
# - Compare the return of the cheapness deciles when the delisting return is
#   true, missing, or imputed
# - See that a missing delisting return inflates the value premium

# %%
# Simulate the monthly returns of 3000 stocks over 20 years, with delistings.
returns_del_df, is_delisting_df, cheapness_s = utils.simulate_delisting_panel(
    n_stocks=3000,
    n_months=240,
    base_hazard=0.0015,
    hazard_slope=1.0,
    premium_slope=0.0012,
    mean_delisting_return=-0.30,
    delisting_vol=0.25,
    seed=seed,
)
print("returns_del_df.shape=", returns_del_df.shape)
print("share delisted=", (is_delisting_df.sum() > 0).mean())

# %%
# Compute the annual return of each cheapness decile under each rule for the
# delisting month.
decile_df = pd.DataFrame(
    {
        rule: utils.compute_decile_returns(
            returns_del_df,
            is_delisting_df,
            cheapness_s,
            delisting_rule=rule,
            imputed_return=-0.30,
        )
        for rule in ["true", "missing", "imputed"]
    }
)
display((100 * decile_df).round(2))

# %% [markdown]
# **Description**
# - Inputs
#   - `mean_delisting_return`: mean return of the delisting month, -30%
#   - `base_hazard`, `hazard_slope`: monthly probability of delisting, and how
#     fast it grows with cheapness
#   - `imputed_return`: return used when the delisting return is missing, -30%
#
# - Panels
#   - `Delisting returns`: histogram of the return of the delisting months,
#     with the mean in red
#   - `Decile returns`: annual return of the ten cheapness deciles with the true
#     delisting return in blue, missing in red, and imputed in green

# %%
# Plot the delisting returns and the decile returns.
delisting_returns = returns_del_df.to_numpy()[is_delisting_df.to_numpy()]
fig = utils.cell2_1_plot_delisting_returns(
    delisting_returns, decile_df, figsize=figsize_tall
)
hmatplo.save_fig(
    fig,
    f"{out_dir_name}/L04.01.Delisting_Returns.png",
    print_markdown=True,
    path_prefix=markdown_path_prefix,
)

# %% [markdown]
# **Guided usage**
# - Compare the red and the blue curves at decile 10
#   - Observe that the missing delisting returns raise the return of the
#     cheapest decile, where most delistings happen
# - Compare the spread in the legend of the three rules
#   - Observe that the imputed return recovers most of the true spread

# %% [markdown]
# **Implementation** `cell2_1_plot_delisting_returns()`
# - Plots the histogram of the returns of the delisting months
# - Plots the annual return of each decile under the three rules, computed by
#   `compute_decile_returns()`, and writes the spread of decile 10 minus
#   decile 1 in the legend

# %%
hintros.print_obj_info(utils.cell2_1_plot_delisting_returns)

# %% [markdown]
# # Part 3: Reference data

# %% [markdown]
# ## Cell 3.1: Split and dividend adjustment
#
# **Goal**
# - Simulate the raw prices of a stock with quarterly dividends and splits
# - Adjust the prices for splits and dividends
# - See the gap between the price-only return and the total return

# %%
# Simulate 20 years of raw prices with dividends and splits.
price_df = utils.simulate_dividend_payer(
    n_years=20,
    start_price=60.0,
    annual_drift=0.11,
    annual_vol=0.25,
    quarterly_yield=0.005,
    split_price=120.0,
    split_ratio=2,
    seed=3,
)
print("number of splits=", (price_df["split_ratio"] > 1).sum())
print("number of dividends=", (price_df["dividend"] > 0).sum())

# %%
# Adjust the raw prices, and compare the worst daily return before and after.
adjusted_df = utils.adjust_prices(price_df)
print("worst raw return=", price_df["close"].pct_change().min())
print("worst adjusted return=", adjusted_df["fully_adjusted"].pct_change().min())

# %% [markdown]
# **Description**
# - Inputs
#   - `quarterly_yield`: dividend over the previous close, 0.5% each quarter
#   - `split_price`, `split_ratio`: raw close above which the stock splits 2 for
#     1
#
# - Panels
#   - `Growth of $1`: value of $1 invested at the start with dividends
#     reinvested in blue, and with the price only in red
#   - `Adjustment factor`: factor that turns the raw close into the adjusted
#     close, for splits and dividends in blue and for splits only in orange

# %%
# Plot the growth of $1 and the adjustment factor.
fig = utils.cell3_1_plot_adjusted_prices(adjusted_df, figsize=figsize_tall)
hmatplo.save_fig(
    fig,
    f"{out_dir_name}/L04.01.Adjusted_Prices.png",
    print_markdown=True,
    path_prefix=markdown_path_prefix,
)

# %% [markdown]
# **Guided usage**
# - Compare the two curves in the top panel at the last date
#   - Observe the gap: the dividends that the price-only series leaves out
# - Follow the blue and the orange lines in the bottom panel
#   - Observe the large steps at the splits and the small steps at the
#     dividends

# %% [markdown]
# **Implementation** `cell3_1_plot_adjusted_prices()`
# - Rescales the split-adjusted and the fully adjusted closes to start at 1
# - Plots the factors computed by `adjust_prices()` as step functions on a log
#   scale

# %%
hintros.print_obj_info(utils.cell3_1_plot_adjusted_prices)

# %% [markdown]
# ## Cell 3.2: The price on a past date, as known and as adjusted today
#
# **Goal**
# - Compare the close that traded on a past date with the close that a
#   database built today shows for that date
# - See that adjusted prices change every time a new split or dividend
#   happens

# %%
# Compare the raw and the adjusted close on one past date.
adjustment_df = utils.make_adjustment_table(adjusted_df, "2010-06-30")
display(adjustment_df)
# Outcome: the stock traded at about 98 dollars, and today's database shows
# about 37 dollars for the same day.

# %% [markdown]
# ## Cell 3.3: Time zones and daylight saving
#
# **Goal**
# - Convert the same local close, 16:00 in New York, to UTC over two years
# - Measure the gap between New York and London, which is not constant
# - List the local days that do not have 24 hours

# %%
# Compute the UTC hour of the New York close and the New York / London gap.
close_df = utils.compute_close_in_utc("2024-01-01", "2025-12-31")
print("days by UTC hour of the close=")
display(close_df["ny_close_utc_hour"].value_counts())

# %%
# Count the days per year where the gap is 4 hours instead of 5.
off_df = close_df[close_df["ny_minus_london_hours"] != 5.0]
print("days with a different gap, by year=")
display(off_df.groupby(pd.DatetimeIndex(off_df.index).year).size())
print("first and last day of the first window=")
print(off_df.index[0], off_df.index[off_df.index < "2024-06-01"][-1])

# %%
# List the local days with 23 or 25 hours in 2024.
print("New York=")
display(
    utils.list_non_24_hour_days(
        "2024-01-01", "2024-12-31", time_zone="America/New_York"
    )
)
print("London=")
display(
    utils.list_non_24_hour_days(
        "2024-01-01", "2024-12-31", time_zone="Europe/London"
    )
)

# %% [markdown]
# **Description**
# - Inputs
#   - `start`, `end`: first and last calendar day of the table
#
# - Panels
#   - `UTC hour of the close`: UTC hour of 16:00 in New York, 21 in winter and
#     20 in summer
#   - `New York minus London`: gap in hours between noon in New York and noon
#     in London, shaded in gray on the days where it is not 5

# %%
# Plot the UTC hour of the close and the New York / London gap.
fig = utils.cell3_3_plot_utc_close(close_df, figsize=figsize_tall)
hmatplo.save_fig(
    fig,
    f"{out_dir_name}/L04.01.UTC_Close.png",
    print_markdown=True,
    path_prefix=markdown_path_prefix,
)

# %% [markdown]
# **Guided usage**
# - Follow the top line across the two clock changes of each year
#   - Observe that the same local close moves by one hour in UTC
# - Look at the gray bands in the bottom panel
#   - Observe that the gap between the two cities drops to 4 hours for about
#     4 weeks a year, since the clocks change on different dates

# %% [markdown]
# **Implementation** `cell3_3_plot_utc_close()`
# - Localizes 16:00 and noon in each city with the time zone database, and
#   converts them to UTC
# - Plots the UTC hour of the close and the gap as step functions

# %%
hintros.print_obj_info(utils.cell3_3_plot_utc_close)
