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
# # Online Prediction of Next-Day Stock Direction with River
#
# **Question.** Can an online model, updated once per day as new prices
# arrive, predict whether a stock closes higher tomorrow better than simple
# baselines, and does it adapt when market conditions change?
#
# **Approach.**
# 1. Download daily prices for a basket of large-cap stocks (cached locally).
# 2. Validate and clean the data.
# 3. Build features that use only information available at today's close.
# 4. Replay history one trading day at a time: predict, then learn once the
#    label is revealed the next day.
# 5. Compare against baselines with tests that account for correlated errors.
#
# Daily stock direction is close to a coin flip, so the goal is a rigorous
# measurement of whatever edge exists (possibly none), not a headline number.

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import pandas as pd
from river import compose, linear_model, preprocessing

import river_utils as rivutil

logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# Set to True to run the notebook offline on synthetic prices.
USE_SYNTHETIC = False

# %% [markdown]
# ## 1. Data acquisition

# %%
if USE_SYNTHETIC:
    raw = rivutil.make_synthetic_ohlcv(n_days=1500, seed=0)
else:
    raw = rivutil.download_ohlcv(
        rivutil.DEFAULT_TICKERS, rivutil.DEFAULT_START, rivutil.DEFAULT_END
    )
print(raw.shape)
raw.head()

# %% [markdown]
# ## 2. Validation and cleaning

# %%
display(rivutil.validate_ohlcv(raw))
clean = rivutil.clean_ohlcv(raw)

# %% [markdown]
# ## 3. Features and labels
#
# Every feature for day `t` is computed from prices up to the close of day
# `t`. The label is the sign of the return from close `t` to close `t+1`.
# Features are not normalized here: normalizing with full-sample statistics
# would leak the future. The River pipeline scales them online instead.

# %%
feats = rivutil.build_features(clean)
panel = rivutil.add_direction_label(feats)
feature_cols = rivutil.feature_columns(panel)
print(f"{len(panel)} rows, {len(feature_cols)} features")
print(f"Share of up days: {panel['target_up'].mean():.3f}")
panel[feature_cols].describe().T

# %% [markdown]
# ## 4. First online model: logistic regression
#
# Test-then-train with a one-trading-day label delay. The first 250 trading
# days (about a year) are a warm-up and are not scored.

# %%
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression(),
)
preds = rivutil.run_prequential(model, panel, feature_cols, warmup_days=250)
rivutil.summarize_predictions(preds).round(4)

# %%
print(rivutil.binomial_test_vs_baseline(preds))
print(rivutil.block_bootstrap_accuracy(preds))

# %%
# Rolling 250-day accuracy shows whether performance is stable over time.
daily_acc = (
    preds.assign(correct=preds["y_true"] == preds["y_pred"])
    .groupby("date")["correct"]
    .mean()
)
ax = daily_acc.rolling(250).mean().plot(figsize=(10, 4))
ax.axhline(0.5, color="gray", linestyle="--")
ax.set_ylabel("Rolling 250-day accuracy")
plt.show()

# %% [markdown]
# ## 5. Baselines (next milestone)
#
# - Majority class learned online (`dummy.PriorClassifier`)
# - Persistence: predict today's direction for tomorrow
# - Batch logistic regression / gradient boosting retrained walk-forward
#   (expanding window, monthly refit) on the same timeline

# %% [markdown]
# ## 6. River model comparison (next milestone)
#
# - Hoeffding tree and Hoeffding adaptive tree
# - Adaptive random forest (`forest.ARFClassifier`)
# - Online bagging / ensembles combining the above

# %% [markdown]
# ## 7. Drift and adaptation (next milestone)
#
# - ADWIN on the error stream; mark detected drifts against known regimes
#   (2020 COVID crash, 2022 rate hikes)
# - Compare a static model, a model with forgetting, and an adaptive tree

# %% [markdown]
# ## 8. Economic evaluation (next milestone)
#
# - Long/flat strategy from predictions, with transaction costs, versus
#   buy-and-hold; report Sharpe ratio and drawdown

# %% [markdown]
# ## 9. Conclusions (next milestone)
