# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Description
#
# Step 1 of the plan on #733: load 5-minute OHLCV bars for BTC and ETH and
# clean them.
#
# - Load the bars from the Binance monthly files (cached as parquet)
# - Put the bars on a regular 5-minute grid and find the gaps
# - Fill gaps of up to 3 bars, leave longer gaps empty and split the data
#   into segments at each one, so later features and labels never span a gap
# - Flag zero-volume bars and extreme returns (kept, not dropped)
# - Summarize the data quality per coin

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import pandas as pd

import data_utils as dutil

# %% [markdown]
# ## Config

# %%
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "5m"
START = "2020-01-01"
END = "2026-10-01"
# Fill gaps of up to 15 minutes, leave longer ones empty.
MAX_FILL_BARS = 3
# Flag a return if it is larger than 10 rolling stds over the past day.
VOL_WINDOW_BARS = 288
EXTREME_STD = 10.0

# %% [markdown]
# ## Load and clean

# %%
raw = {}
clean = {}
for symbol in SYMBOLS:
    raw[symbol] = dutil.load_ohlcv(symbol, START, END, interval=INTERVAL)
    clean[symbol] = dutil.clean_bars(
        raw[symbol],
        interval=INTERVAL,
        max_fill_bars=MAX_FILL_BARS,
        vol_window_bars=VOL_WINDOW_BARS,
        extreme_std=EXTREME_STD,
    )
clean["BTCUSDT"].head()

# %% [markdown]
# ## Data quality table

# %%
quality = pd.DataFrame(
    {
        symbol: dutil.compute_quality_table(
            raw[symbol], clean[symbol], interval=INTERVAL
        )
        for symbol in SYMBOLS
    }
)
quality

# %% [markdown]
# ## Gaps
#
# The longest gaps, with their length in bars.

# %%
gaps = dutil.find_gaps(raw["BTCUSDT"], interval=INTERVAL)
gaps.sort_values("n_bars", ascending=False).head(10)

# %% [markdown]
# Check if the two coins have their gaps at the same times.

# %%
gaps_eth = dutil.find_gaps(raw["ETHUSDT"], interval=INTERVAL)
print("Same gaps for BTC and ETH:", gaps.equals(gaps_eth))

# %% [markdown]
# ## Flagged bars
#
# Extreme returns per year. Most should come from known market events.

# %%
extreme_per_year = pd.DataFrame(
    {
        symbol: clean[symbol]["is_extreme"]
        .groupby(clean[symbol].index.year)
        .sum()
        for symbol in SYMBOLS
    }
)
extreme_per_year

# %%
# The largest moves among the flagged bars.
flagged = clean["BTCUSDT"].loc[clean["BTCUSDT"]["is_extreme"], ["close", "ret"]]
flagged.reindex(flagged["ret"].abs().sort_values(ascending=False).index).head(10)

# %%
# Zero-volume bars per day.
zero_vol = clean["BTCUSDT"].index[clean["BTCUSDT"]["is_zero_volume"]]
zero_vol.to_series().dt.date.value_counts().sort_index()

# %% [markdown]
# ## Plot

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for ax, symbol in zip(axes, SYMBOLS):
    df = clean[symbol]
    ax.plot(df.index, df["close"], linewidth=0.5)
    flagged = df.index[df["is_extreme"]]
    ax.scatter(
        flagged,
        df.loc[flagged, "close"],
        color="red",
        s=5,
        label="extreme return",
    )
    gap_starts = dutil.find_gaps(raw[symbol], interval=INTERVAL)["start"]
    for gap_start in gap_starts:
        ax.axvline(gap_start, color="gray", linewidth=0.5, linestyle=":")
    ax.set_title(f"{symbol} 5-minute close (dotted lines: gaps)")
    ax.set_yscale("log")
    ax.legend()
plt.tight_layout()

# %% [markdown]
# ## Observations
#
# - The data is clean overall: no duplicates, and only 462 of 709,920 bars
#   (0.07%) are missing for each coin
# - BTC and ETH have the same 15 gaps at the same times, so these are Binance
#   outages and maintenance, not problems with one coin. The longest is
#   5h50m on 2020-02-19. Most are in 2020-2021, and the last one is on
#   2023-03-24
# - Every gap is at least 12 bars (1 hour) long, so nothing was filled. The
#   data splits into 16 segments, and features and labels will be computed
#   inside each one
# - The 42 zero-volume bars are on 4 days, and on 3 of them they come right
#   before a gap (2020-12-21, 2021-02-11, 2023-03-24). Trading stopped before
#   the data stopped, so the prices in those bars are stale. For now they are
#   only flagged; we may want to treat them as part of the gap
# - About 300 bars per coin (0.05%) have an extreme return, more than 10
#   times the past day's std. The largest ones are real crashes and
#   rebounds: March 2020 (+18% in 5 minutes on 2020-03-13), the May 2021
#   crash, and 2025-10-10. They are kept, since fat tails are part of what we
#   study
