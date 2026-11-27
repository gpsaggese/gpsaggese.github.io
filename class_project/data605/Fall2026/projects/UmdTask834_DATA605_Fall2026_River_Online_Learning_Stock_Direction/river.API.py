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
# # River API Tour
#
# This notebook introduces the parts of the [River](https://riverml.xyz/) API
# used in `river.example.ipynb`, on small built-in data so every cell runs in
# seconds and without network access.
#
# Topics:
# - The online learning contract: one sample at a time, as a `dict`
# - `learn_one` / `predict_one` / `predict_proba_one`
# - Pipelines and online preprocessing (`StandardScaler`)
# - Incremental metrics
# - Progressive validation (test-then-train)
# - Concept-drift detection (`ADWIN`)
#
# References:
# - River documentation: https://riverml.xyz/
# - Montiel et al., "River: machine learning for streaming data in Python",
#   JMLR 22(110), 2021

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import itertools
import logging

import numpy as np
from river import (
    compose,
    datasets,
    drift,
    evaluate,
    linear_model,
    metrics,
    preprocessing,
    tree,
)

logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# ## 1. A stream is an iterator of `(x, y)` pairs
#
# `x` is a plain `dict` of feature name to value, not a NumPy row. This lets
# features appear or disappear over time, which batch libraries cannot
# handle. `datasets.Phishing` ships with River, so no download is needed.

# %%
dataset = datasets.Phishing()
x, y = next(iter(dataset))
print(dataset)
print(x)
print(y)

# %% [markdown]
# ## 2. The online learning contract
#
# A River model holds its own state and is updated one observation at a time
# with `learn_one`. There is no `fit(X, y)` over a full dataset.

# %%
model = linear_model.LogisticRegression()
# Before learning anything the model returns an uninformative prediction.
print("Before:", model.predict_proba_one(x))
model.learn_one(x, y)
print("After one update:", model.predict_proba_one(x))

# %% [markdown]
# ## 3. Pipelines and online preprocessing
#
# Scaling with statistics from the whole dataset leaks future information into
# past predictions. `preprocessing.StandardScaler` instead keeps running means
# and variances and updates them sample by sample. The `|` operator chains
# steps into a `compose.Pipeline`.

# %%
model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
model

# %%
# The scaler's running statistics after a few samples.
for x, y in itertools.islice(datasets.Phishing(), 50):
    model.learn_one(x, y)
scaler = model["StandardScaler"]
print({k: round(v, 3) for k, v in list(scaler.means.items())[:4]})

# %% [markdown]
# ## 4. Incremental metrics
#
# Metrics are updated with one `(y_true, y_pred)` pair at a time and can be
# read at any point of the stream.

# %%
acc = metrics.Accuracy()
for y_true, y_pred in [(True, True), (False, True), (True, True)]:
    acc.update(y_true, y_pred)
acc

# %% [markdown]
# ## 5. Progressive validation (test-then-train)
#
# Each sample is first used to test the model, then to train it. Every
# prediction is therefore made on unseen data, and all data is used for both
# purposes. This is the streaming replacement for k-fold cross-validation,
# which would train on the future and test on the past.

# %%
model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
evaluate.progressive_val_score(
    datasets.Phishing(), model, metrics.ROCAUC(), print_every=250
)

# %% [markdown]
# Comparing a linear model with an incremental decision tree (Hoeffding tree)
# on the same stream:

# %%
for name, m in {
    "logreg": preprocessing.StandardScaler() | linear_model.LogisticRegression(),
    "hoeffding_tree": tree.HoeffdingTreeClassifier(grace_period=50),
}.items():
    score = evaluate.progressive_val_score(datasets.Phishing(), m, metrics.Accuracy())
    print(f"{name:>15}: {score}")

# %% [markdown]
# ## 6. Delayed labels
#
# In finance the label (tomorrow's price move) arrives one step after the
# prediction. `progressive_val_score` supports this with `moment` and
# `delay`. With a multi-ticker panel the delay must be per trading day, not
# per row; `river_utils.run_prequential` implements that explicitly.

# %% [markdown]
# ## 7. Concept-drift detection with ADWIN
#
# ADWIN keeps a window of recent values and drops old data when the mean of
# the recent part differs significantly from the older part. Below, the
# error rate of a stream jumps from 30% to 60% at step 1000.

# %%
rng = np.random.default_rng(0)
errors = np.concatenate([rng.random(1000) < 0.3, rng.random(1000) < 0.6])
detector = drift.ADWIN()
for i, e in enumerate(errors):
    detector.update(int(e))
    if detector.drift_detected:
        print(f"Drift detected at step {i}")

# %% [markdown]
# ## Summary
#
# The example notebook combines these pieces: daily stock features are
# streamed as dicts through an online pipeline, evaluated test-then-train with
# a one-day label delay, and monitored for drift.
