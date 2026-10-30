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
# Step 1 of the plan on #733: load 5-minute OHLCV bars for BTC and ETH, clean
# them, and compute features and labels.
#
# - Load the bars from the Binance monthly files (cached as parquet)
# - Put the bars on a regular 5-minute grid and find the gaps
# - Fill gaps of up to 3 bars, leave longer gaps empty and split the data
#   into segments at each one, so later features and labels never span a gap
# - Flag zero-volume bars and extreme returns (kept, not dropped)
# - Summarize the data quality per coin
# - Compute features (past data only) and forward-return labels for each
#   horizon, inside each segment
# - Summarize the labels per horizon on non-overlapping decision times

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import pandas as pd

import data_utils as dutil
import tradability_utils as tu

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
# Holding periods for the labels.
HORIZONS = ["5m", "15m", "1h", "4h", "1d"]

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
# ## Features
#
# Each row is what is known at the close of the bar: lagged returns,
# volatility, volume, range, momentum, time, and the other coin's returns.

# %%
other = {"BTCUSDT": "ETHUSDT", "ETHUSDT": "BTCUSDT"}
features = {}
for symbol in SYMBOLS:
    features[symbol] = dutil.compute_features(
        clean[symbol],
        other=clean[other[symbol]],
        other_name=other[symbol][:3].lower(),
        interval=INTERVAL,
    )
features["BTCUSDT"].describe().T

# %%
# Share of missing values per feature: segment starts and long gaps.
pd.DataFrame({symbol: features[symbol].isna().mean() for symbol in SYMBOLS})

# %% [markdown]
# ## Labels per horizon
#
# For each horizon, keep one decision time every horizon so the trades do
# not overlap, then compare:
# - `share_up`: share of up moves, i.e., the hit rate of always buying. It
#   uses all decision times (the average over the possible start bars), since
#   with one start bar it depends on the time of day at long horizons
# - `p_star_eff`: hit rate needed for a 95% chance of profit, with N_eff as in
#   Exercise 1

# %%
labels = {}
for symbol in SYMBOLS:
    labels[symbol] = dutil.compute_labels(
        clean[symbol], HORIZONS, interval=INTERVAL
    )
rows = []
for symbol in SYMBOLS:
    for horizon in HORIZONS:
        sample = dutil.sample_non_overlapping(
            labels[symbol], horizon, interval=INTERVAL
        )
        sample = sample.dropna(subset=[f"label_{horizon}"])
        fwd_ret = sample[f"fwd_ret_{horizon}"]
        n_eff = tu.compute_effective_n(fwd_ret)
        rows.append(
            {
                "symbol": symbol,
                "horizon": horizon,
                "n": len(sample),
                "share_up": labels[symbol][f"label_{horizon}"].mean(),
                "mean_abs_ret": fwd_ret.abs().mean(),
                "n_eff": round(n_eff),
                "p_star_eff": tu.compute_min_hit_rate(n_eff),
            }
        )
label_stats = pd.DataFrame(rows).set_index(["symbol", "horizon"])
label_stats.round(4)

# %%
# Plot the share of up moves and p* against the horizon.
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, symbol in zip(axes, SYMBOLS):
    stats = label_stats.loc[symbol]
    ax.plot(stats.index, stats["share_up"], marker="o", label="share up")
    ax.plot(stats.index, stats["p_star_eff"], marker="o", label="corrected p*")
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.set_title(symbol)
    ax.set_xlabel("horizon")
    ax.set_ylabel("hit rate")
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
#
# Features and labels:
# - 17 features per bar. Less than 5% are missing, all at the start of a
#   segment where the 7-day windows are not full yet
# - Both lines in the last plot go up with the horizon. The share of up moves
#   goes up because BTC and ETH rose a lot over 2020-2026, and the drift adds
#   up over longer holds. p* goes up because there are fewer independent
#   trades. This is the horizon tradeoff from the spec, in the real data
# - Always buying is above p* at 1h and 4h for both coins, but below it at
#   5m and 1d. This does not mean buying is a strategy: the drift is specific
#   to this period, and p* assumes the hits do not depend on the size of the
#   move. But it is the baseline any model in step 2 has to beat
# - At 1d the share of up moves depends on the time of day we decide at:
#   from 50.6% to 53.1% for BTC, since there are only 2,448 days. Results at
#   1d will be noisy
