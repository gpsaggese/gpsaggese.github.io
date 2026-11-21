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
# **Findings (baselines)**
# - No model beats always predicting "up" (52.75%, 95% block-bootstrap CI
#   51.97-53.61%); every model that tries to predict is significantly worse
#   (McNemar).
# - Accuracy falls in the same order as the share of "down" calls (100% up:
#   52.8%, 76%: 52.0%, 67%: 51.5%, 61%: 50.3%, 53%: 49.4%). The models' "down"
#   calls are right less often than the base rate, so each one costs accuracy.
# - Only always-up has a log loss below ln(2) = 0.693: the learning models'
#   probabilities are worse than predicting 50/50 every day.
# - Persistence scores 49.35%, so a reversal rule would score about 50.65%: a
#   small short-term reversal effect, still below always-up.
# - Retraining monthly in batch beat the untuned online logistic regression
#   (52.0% vs. 50.3%); section 6 tests whether tuning closes this gap.

# %% [markdown]
# ## 6. River model comparison
#
# Eleven RIver models run on the same stream with identical one-day label delay: six settings of the online logistic regression (learning rate x L2 penalty) and five other model families (naive Bayes, Hoeffding tree, Hoeffding adaptive tree, adaptive random forest, bagged Hoeffding trees). 
#
# To avoid overstating the tuned model, the logistic regression setting is
# chosen on 2011-2014 only (by log loss), and every model is then compared on
# 2015-2026. The key column is `down_call_accuracy`: a model can only beat
# always predicting "up" if its "down" calls are right more than half the time.

# %%
# Six settings of the online logistic regression plus five other River models.
# The `lr=lr, l2=l2` defaults freeze each loop value inside its lambda;
# without them every lambda would use the last values of the loop.
logreg_grid = {
    f"river_logreg_lr{lr}_l2{l2}": (
        lambda lr=lr, l2=l2: rivutil.make_river_logreg(lr, l2)
    )
    for lr in (0.01, 0.001, 0.0001)
    for l2 in (0.0, 0.01)
}
river_preds, river_seconds = rivutil.run_models(
    {**logreg_grid, **rivutil.make_river_models()},
    panel, feature_cols, warmup_days=250,
)

# %%
# Choose the logistic regression setting on 2011-2014 only.
SPLIT_DATE = "2015-01-01"
early, late = rivutil.split_preds({**all_preds, **river_preds}, SPLIT_DATE)
selection = rivutil.compare_models(
    {k: early[k] for k in logreg_grid},
    reference="river_logreg_lr0.01_l20.0",
    n_boot=200,
)
best_logreg = selection["log_loss"].idxmin()
print("Chosen on 2011-2014:", best_logreg)
selection.sort_values("log_loss").round(4)

# %%
# Compare everything on 2015-2026.
pd.set_option("display.max_columns", None)
report_models = list(dict.fromkeys([
    "always_up_online", "persistence",
    "batch_logreg_monthly", "batch_hgb_monthly",
    "river_logreg_lr0.01_l20.0", best_logreg,
    *rivutil.make_river_models(),
]))
results = rivutil.compare_models(
    {k: late[k] for k in report_models}, reference="always_up_online"
)
results["seconds_full_run"] = river_seconds
results.round(4)

# %% [markdown]
# **Findings (River models, test period 2015-2026)**
# - Tuning matters: the learning rate chosen on 2011-2014 (0.0001) raised the
#   online logistic regression from 50.3% to 52.0% and its log loss from 0.725
#   to 0.693 (= ln 2), closing the gap to the monthly batch model (52.1%).
# - It improves by learning less: the chosen setting is the smallest in the
#   grid, and the model's share of "up" calls rises from 62% to 85%, so it
#   behaves more like always predicting "up".
# - More complex River models do not help: Hoeffding trees, the Hoeffding
#   adaptive tree, the adaptive random forest, bagged trees and naive Bayes
#   all score 49.9-51.1%. The single trees are overconfident (log loss 0.90-0.93).
# - No model's "down" calls carry information: every model is right on 45-48%
#   of its "down" calls, the same as the 47% share of down days. AUCs are
#   0.49-0.51. No model beats always "up" (53.0%, 95% CI 52.1-54.0%).
# - McNemar p-values are optimistic because the 8 stocks move together; the
#   block-bootstrap intervals of the two best models (tuned River and batch
#   logistic regression) only just reach always-up's accuracy.

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
