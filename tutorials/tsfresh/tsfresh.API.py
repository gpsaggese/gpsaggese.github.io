# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # tsfresh API Overview
#
# **tsfresh** is a Python library for automated **time series feature
# extraction**.
#
# - Computes 100+ statistical, temporal, and frequency-domain features
#   from raw time series data automatically.
# - Primary use cases: classification, regression, anomaly detection.
# - Removes the need for manual feature engineering.
#
# This notebook covers:
# - Preparing data in **long format** (required by tsfresh).
# - Extracting features with pre-defined parameter sets.
# - Selecting relevant features with statistical tests.
# - Integrating tsfresh into a scikit-learn Pipeline.
#
# Reference: [tsfresh documentation](https://tsfresh.readthedocs.io/)

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import numpy as np
import pandas as pd

import tsfresh
import tsfresh.feature_extraction as fe
from tsfresh import extract_features, select_features

logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# ## Architecture Overview
#
# tsfresh's workflow has four stages:
#
# 1. **Data Preparation** – convert raw time series to *long format*.
# 2. **Feature Extraction** – apply built-in calculators to produce a feature
#    matrix.
# 3. **Feature Selection** – keep only statistically significant features.
# 4. **Model Training** – feed selected features into any scikit-learn
#    estimator.

# %% [markdown]
# ## Data Handling
#
# tsfresh requires a **long-format DataFrame** with these columns:
#
# | Column  | Description                                     |
# |---------|-------------------------------------------------|
# | `id`    | entity identifier (one integer per time series) |
# | `time`  | time index (integer or timestamp)               |
# | `value` | measurement value                               |
# | `kind`  | optional: channel/sensor name (multivariate)    |
#
# Each unique `(id, kind)` pair defines one time series.

# %%
# Build a tiny univariate example: 3 series, each 10 timesteps.
np.random.seed(42)
n_series = 3
n_time = 10

rows = []
for sid in range(n_series):
    signal = np.sin(np.linspace(0, 2 * np.pi, n_time)) + np.random.randn(n_time) * 0.1
    for t, v in enumerate(signal):
        rows.append({"id": sid, "time": t, "value": v})

df_uni = pd.DataFrame(rows)
print(df_uni.head(12))

# %%
# Multivariate example: add a second channel "acc_y".
rows_multi = []
for sid in range(n_series):
    for kind in ["acc_x", "acc_y"]:
        signal = np.random.randn(n_time)
        for t, v in enumerate(signal):
            rows_multi.append({"id": sid, "time": t, "kind": kind, "value": v})

df_multi = pd.DataFrame(rows_multi)
print(df_multi.head(8))

# %% [markdown]
# ## Feature Extraction
#
# tsfresh provides three pre-defined parameter sets for controlling how many
# features are calculated:
#
# | Parameter Set                | Features | Speed   |
# |------------------------------|----------|---------|
# | `MinimalFCParameters`        | ~7       | fastest |
# | `EfficientFCParameters`      | ~800     | medium  |
# | `ComprehensiveFCParameters`  | >1,500   | slowest |

# %%
# Extract features with MinimalFCParameters (fast, good for prototyping).
settings_minimal = fe.MinimalFCParameters()
X_min = extract_features(
    df_uni,
    column_id="id",
    column_sort="time",
    column_value="value",
    default_fc_parameters=settings_minimal,
    disable_progressbar=True,
)
X_min = X_min.fillna(0)
print(f"MinimalFCParameters → {X_min.shape[1]} features")
print(X_min)

# %%
# Extract features from multivariate data (uses column_kind).
settings_efficient = fe.EfficientFCParameters()
X_eff = extract_features(
    df_multi,
    column_id="id",
    column_sort="time",
    column_kind="kind",
    column_value="value",
    default_fc_parameters=settings_minimal,  # Keep minimal for speed.
    disable_progressbar=True,
)
X_eff = X_eff.fillna(0)
print(f"Multivariate EfficientFCParameters → {X_eff.shape[1]} features")
print(X_eff.head())

# %% [markdown]
# ## Feature Selection
#
# `select_features()` runs statistical relevance tests (FRESH algorithm) to
# keep only features that are significantly correlated with the target `y`.
#
# - Supports both **classification** (integer labels) and **regression**
#   (float targets).
# - Internally uses Benjamini-Hochberg FDR correction.

# %%
# Create a toy binary target: series 0 & 1 are class 0, series 2 is class 1.
y = pd.Series({0: 0, 1: 0, 2: 1})

X_selected = select_features(X_min, y)
print(
    f"Selected {X_selected.shape[1]} / {X_min.shape[1]} features: "
    f"{list(X_selected.columns)}"
)

# %% [markdown]
# ## scikit-learn Pipeline Integration
#
# tsfresh ships a `RelevantFeatureAugmenter` transformer that fits into
# standard scikit-learn `Pipeline` objects.

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from tsfresh.transformers import RelevantFeatureAugmenter

# Build pipeline: feature augmentation → classifier.
pipeline = Pipeline(
    [
        (
            "augmenter",
            RelevantFeatureAugmenter(
                column_id="id",
                column_sort="time",
                column_value="value",
                default_fc_parameters=fe.MinimalFCParameters(),
                disable_progressbar=True,
            ),
        ),
        ("classifier", RandomForestClassifier(n_estimators=10, random_state=0)),
    ]
)

# For the pipeline we need a DataFrame indexed by id.
# The augmenter works on the *original* long-format data set via timeseries_container.
pipeline.set_params(augmenter__timeseries_container=df_uni)
pipeline.fit(pd.DataFrame(index=y.index), y)
print("Pipeline trained successfully.")
print("Predicted:", pipeline.predict(pd.DataFrame(index=y.index)))

# %% [markdown]
# ## Custom Feature Calculators
#
# You can register your own feature function by decorating it with
# `@set_property("fctype", "simple")` and including it in a
# `CustomFCParameters` dict.

# %%
from tsfresh.feature_extraction.feature_calculators import set_property


@set_property("fctype", "simple")
def range_value(x):
    """Return max - min of the time series."""
    return float(np.max(x) - np.min(x))


custom_settings = {"range_value": None}
X_custom = extract_features(
    df_uni,
    column_id="id",
    column_sort="time",
    column_value="value",
    default_fc_parameters=custom_settings,
    disable_progressbar=True,
)
print("Custom feature (range_value):")
print(X_custom)
