# ---
# jupyter:
#   jupytext:
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
from river import dummy
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
# - Always up (online): predicts the majority class, learned day by day
# - Persistence: predicts that tomorrow repeats today's direction
# - Batch logistic regression and batch gradient boosting: retrained on
#   the first trading day of each month on all earlier data (expanding window)
#
# All models are scored on the same predictions, so they can be compared row by
# row with McNemar's test.

# %%
# Online baselines use the same test-then-train loop as the River model.
all_preds = {"river_logreg": preds}
all_preds["always_up_online"] = rivutil.run_prequential(
    dummy.PriorClassifier(), panel, feature_cols, warmup_days=250
)
all_preds["persistence"] = rivutil.run_prequential(
    rivutil.PersistenceClassifier(), panel, feature_cols, warmup_days=250
)
# Batch models retrained monthly on all past data.
all_preds["batch_logreg_monthly"] = rivutil.run_walk_forward(
    lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    panel, feature_cols, warmup_days=250,
)
all_preds["batch_hgb_monthly"] = rivutil.run_walk_forward(
    lambda: HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.05, random_state=0
    ),
    panel, feature_cols, warmup_days=250,
)

# %%
comparison = rivutil.compare_models(all_preds, reference="always_up_online")
comparison.round(4)

# %%
# Rolling 250-day accuracy of every model.
fig, ax = plt.subplots(figsize=(10, 4))
for name, p in all_preds.items():
    daily = (p["y_true"] == p["y_pred"]).groupby(p["date"]).mean()
    daily.rolling(250).mean().plot(ax=ax, label=name)
ax.axhline(0.5, color="gray", linestyle="--")
ax.set_ylabel("Rolling 250-day accuracy")
ax.legend(fontsize=8)
plt.show()

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
