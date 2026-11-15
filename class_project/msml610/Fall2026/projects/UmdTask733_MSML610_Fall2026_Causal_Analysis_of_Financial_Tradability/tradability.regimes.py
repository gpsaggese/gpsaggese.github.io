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
# Step 3 of the plan on #733: does the answer of step 2 change with the
# market conditions? As GP asked, this only describes what happens, without
# a causal analysis (`D17`).
#
# - Same data, models, and trades as `tradability.models.ipynb`: BTC and
#   ETH, 5-minute bars, three models, walk forward by year, non-overlapping
#   trades in 2022-01-01 to 2026-10-01
# - Each trade is labeled with the market conditions at decision time, using
#   past data only (`regime_utils.compute_regimes()`):
#   - volatility: the 7-day volatility against its terciles over the past
#     year: calm, normal, volatile
#   - trend: the 7-day return divided by its expected size from the 7-day
#     volatility (t-statistic): falling (t < -1), flat, rising (t > 1)
# - The models are not refit per regime: the same trades are split by the
#   conditions they were made in
# - Each PnL per trade comes with its standard error, since some groups are
#   small

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import data_utils as dutil
import matplotlib.pyplot as plt
import pandas as pd
import predict_utils as putil
import regime_utils as rutil

# %% [markdown]
# ## Config

# %%
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "5m"
START = "2020-01-01"
END = "2026-10-01"
HORIZONS = ["5m", "15m", "1h", "4h", "1d"]
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
# Costs per round trip, as in the cost sweep of `tradability.models.ipynb`.
COSTS = [0.0, 0.0002, 0.0005, 0.001]
MOMENTUM_COLUMNS = {
    "5m": "ret_1",
    "15m": "ret_3",
    "1h": "ret_12",
    "4h": "ret_48",
    "1d": "mom_1d",
}
MODELS = ["always_buy", "momentum", "logistic"]
REGIME_TYPES = {
    "vol_regime": rutil.VOL_REGIMES,
    "trend_regime": rutil.TREND_REGIMES,
}

# %% [markdown]
# ## Data, features, labels, regimes

# %%
clean = {}
for symbol in SYMBOLS:
    raw = dutil.load_ohlcv(symbol, START, END, interval=INTERVAL)
    clean[symbol] = dutil.clean_bars(raw, interval=INTERVAL)
other = {"BTCUSDT": "ETHUSDT", "ETHUSDT": "BTCUSDT"}
features = {}
labels = {}
regimes = {}
for symbol in SYMBOLS:
    features[symbol] = dutil.compute_features(
        clean[symbol],
        other=clean[other[symbol]],
        other_name=other[symbol][:3].lower(),
        interval=INTERVAL,
    )
    labels[symbol] = dutil.compute_labels(
        clean[symbol], HORIZONS, interval=INTERVAL
    )
    regimes[symbol] = rutil.compute_regimes(features[symbol], interval=INTERVAL)

# %% [markdown]
# ## How often each regime happens
#
# Share of the bars in 2022-2026.

# %%
test_start = f"{TEST_YEARS[0]}-01-01"
rows = []
for symbol in SYMBOLS:
    df = regimes[symbol].loc[test_start:]
    for regime_type, names in REGIME_TYPES.items():
        share = df[regime_type].value_counts(normalize=True)
        for name in names:
            rows.append(
                {
                    "symbol": symbol,
                    "regime_type": regime_type,
                    "regime": name,
                    "share": share[name],
                }
            )
shares = pd.DataFrame(rows).pivot_table(
    index=["regime_type", "regime"], columns="symbol", values="share"
)
shares.round(3)

# %% [markdown]
# ## Predict and evaluate per regime

# %%
rows = []
trades = {}
for symbol in SYMBOLS:
    for horizon in HORIZONS:
        pos = {
            "always_buy": putil.predict_always_buy(features[symbol]),
            "momentum": putil.predict_momentum(
                features[symbol], MOMENTUM_COLUMNS[horizon]
            ),
            "logistic": putil.predict_logistic(
                features[symbol],
                labels[symbol],
                horizon,
                TEST_YEARS,
                interval=INTERVAL,
            ),
        }
        # One trade every horizon, in the test years only.
        sample = dutil.sample_non_overlapping(
            labels[symbol], horizon, interval=INTERVAL
        )
        sample = sample[sample.index >= test_start]
        fwd_ret = sample[f"fwd_ret_{horizon}"]
        for model in MODELS:
            trades[(symbol, horizon, model)] = pd.DataFrame(
                {"pos": pos[model].reindex(sample.index), "fwd_ret": fwd_ret}
            )
            for regime_type in REGIME_TYPES:
                stats = putil.evaluate_by_group(
                    pos[model].reindex(sample.index),
                    fwd_ret,
                    regimes[symbol][regime_type].reindex(sample.index),
                )
                stats["symbol"] = symbol
                stats["horizon"] = horizon
                stats["model"] = model
                stats["regime_type"] = regime_type
                rows.append(stats.reset_index(names="regime"))
results = pd.concat(rows, ignore_index=True)
# How many standard errors the PnL per trade is from zero.
results["t_stat"] = results["gross_bps"] / results["se_bps"]
cols = ["symbol", "regime_type", "model", "regime", "horizon"]
results = results.set_index(cols).sort_index()
results[
    ["n", "hit_rate", "weighted_hit_rate", "gross_bps", "se_bps", "t_stat"]
].round(3)

# %% [markdown]
# ## Volatility: PnL per trade of the logistic regression
#
# PnL per trade before costs (bp) and how many standard errors it is from
# zero.

# %%
idx = pd.IndexSlice
logistic_vol = results.loc[idx[:, "vol_regime", "logistic"], :]
table = logistic_vol[["gross_bps", "se_bps", "t_stat"]].unstack("horizon")
table = table.reindex(columns=HORIZONS, level="horizon")
table = table.droplevel(["regime_type", "model"])
table = table.reindex(rutil.VOL_REGIMES, level="regime")
table.round(2)

# %% [markdown]
# ## Is it one year?
#
# PnL per trade before costs (bp) of the logistic regression in volatile
# markets, for each test year.

# %%
rows = []
for symbol in SYMBOLS:
    for horizon in HORIZONS:
        df = trades[(symbol, horizon, "logistic")]
        vol_regime = regimes[symbol]["vol_regime"].reindex(df.index)
        df = df[vol_regime == "volatile"]
        stats = putil.evaluate_by_group(
            df["pos"], df["fwd_ret"], pd.Series(df.index.year, index=df.index)
        )
        stats["symbol"] = symbol
        stats["horizon"] = horizon
        rows.append(stats.reset_index(names="year"))
by_year = pd.concat(rows, ignore_index=True)
by_year = by_year.pivot_table(
    index=["symbol", "year"], columns="horizon", values="gross_bps"
)[HORIZONS]
by_year.round(2)

# %% [markdown]
# ## Weighted hit rate per regime
#
# Share of the total |r| traded in the right direction. The PnL before costs
# is positive only above 0.5. Bars: 2 standard errors. The PnL per trade is
# (2 * whr - 1) * mean |r|, so the standard error of the weighted hit rate is
# se / (2 * mean |r|).

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
for row, regime_type in enumerate(REGIME_TYPES):
    for col, symbol in enumerate(SYMBOLS):
        ax = axes[row, col]
        for i, regime in enumerate(REGIME_TYPES[regime_type]):
            sub = results.loc[idx[symbol, regime_type, "logistic", regime], :]
            sub = sub.reindex(HORIZONS)
            err = 2 * sub["se_bps"] / (2 * sub["mean_abs_bps"])
            ax.errorbar(
                HORIZONS,
                sub["weighted_hit_rate"],
                yerr=err,
                marker="o",
                capsize=3,
                color=f"C{i}",
                label=regime,
            )
        ax.axhline(0.5, color="gray", linestyle=":")
        ax.set_title(f"{symbol}, logistic regression, by {regime_type}")
        ax.set_xlabel("horizon")
        ax.set_ylabel("weighted hit rate")
        ax.legend(fontsize=8)
plt.tight_layout()

# %% [markdown]
# ## Best horizon per regime and cost
#
# For each regime and cost, the model and horizon with the highest PnL per
# trade after costs, and how many standard errors its PnL before costs is
# from zero. A negative net PnL means nothing makes money in that regime at
# that cost.

# %%
# The best model and horizon is the one with the highest PnL before costs,
# since the cost is the same for every trade.
flat = results.reset_index()
keys = ["symbol", "regime_type", "regime"]
best_rows = flat.groupby(keys, observed=True)["gross_bps"].idxmax()
best = flat.loc[best_rows].set_index(keys)
best = best[["model", "horizon", "gross_bps", "t_stat"]]
for cost in COSTS:
    best[f"net_bps at {cost * 1e4:g} bp"] = best["gross_bps"] - cost * 1e4
best.round(2)

# %% [markdown]
# ## Trend: always buy and the logistic regression
#
# PnL per trade before costs (bp) by trend regime, and its t-statistic.

# %%
trend = results.loc[idx[:, "trend_regime", ["always_buy", "logistic"]], :]
table = trend[["gross_bps", "t_stat"]].unstack("horizon")
table = table.reindex(columns=HORIZONS, level="horizon")
table = table.droplevel("regime_type")
table.round(2)

# %% [markdown]
# ## Observations
#
# How often:
# - In 2022-2026 the market was calm about 40% of the time, normal 30%, and
#   volatile 30% (the thresholds come from the past year, and 2022-2026 was
#   calmer than 2020-2021). It was flat 71-73% of the time, rising 16-17%,
#   falling 11-12%
#
# Volatility:
# - The logistic regression's edge is in volatile markets only. There its
#   weighted hit rate goes up with the horizon, from 0.50 at 5m to 0.54 at
#   4h and 0.55-0.58 at 1d. In calm and normal markets it stays at about
#   0.50 or below (the one exception, ETH at 1d in normal markets with 0.52,
#   is within its error), and the model loses a little at most horizons
# - In volatile markets the PnL per trade before costs is 1.7 bp (BTC) and
#   2.2 bp (ETH) at 1h, 7.7 and 10.9 bp at 4h, and 35 and 28 bp at 1d. From
#   1h to 4h it is 2.6 to 3.5 standard errors from zero on both coins
# - So the step 2 average (0-2 bp, nothing after costs) hides two different
#   markets: no edge in calm and normal times, and an edge that grows with
#   the horizon in volatile times
# - It is not one lucky year: at 4h the volatile-market PnL is positive in 9
#   of the 10 coin-years, but it gets smaller over time, and in 2026 it is
#   1.4 bp (BTC) and -2.5 bp (ETH)
# - With costs: at 4h in volatile markets the PnL per trade (7.7-10.9 bp) is
#   about the 10 bp cost, so it would pay a cost of 5 bp. At 1d it is above
#   10 bp, but on only about 500 trades per coin, so the uncertainty is
#   large (BTC 35 +/- 15 bp, ETH 28 +/- 18 bp)
# - Always buying does not show the same pattern (at most 1.5 standard
#   errors), so the edge comes from the model, not from the market going up
#   more in volatile times
#
# Trend:
# - The trend splits show much less. Always buying after a rising week makes
#   a little more at every horizon (ETH: 2.4-2.8 standard errors from 5m to
#   4h; BTC: 1.5-1.8), so a weekly up move tends to continue a bit
# - Short moves reverse in every regime: momentum at 15m loses in all of
#   them, the most in flat markets (BTC: 5.8 standard errors)
#
# Best horizon:
# - In every regime the highest PnL per trade is at 1d, since that is where
#   the moves are largest. But the 1d results are noisy: none is more than
#   2.4 standard errors from zero. The volatile-market edge is clearer from
#   1h to 4h, where there are many more trades
#
# Caveats:
# - Each regime type splits the trades into 90 groups (2 coins, 5 horizons,
#   3 models, 3 regimes). At 2 standard errors, about 5 would look
#   significant by chance. The volatile-market result holds on both coins
#   and at 4 horizons, so it is unlikely to be chance, but it should be
#   checked on new data
# - The regime definitions (terciles over the past year, |t| > 1) were set
#   once and not tuned
