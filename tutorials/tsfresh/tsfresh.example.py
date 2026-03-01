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
# # tsfresh: Human Activity Recognition
#
# This notebook demonstrates an end-to-end time series classification pipeline
# using **tsfresh** on the
# [UCI HAR dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
# (smartphone accelerometer & gyroscope data).
#
# **Workflow:**
# 1. Download the UCI HAR dataset.
# 2. Load raw inertial sensor signals (9 channels, 128 timesteps per sample).
# 3. Convert to long format required by tsfresh.
# 4. Extract statistical features with `tsfresh.extract_features`.
# 5. Select the most relevant features with `tsfresh.select_features`.
# 6. Train a `RandomForestClassifier` to predict 6 activity classes.
# 7. Evaluate with accuracy, macro-F1, and macro-ROC-AUC.
# 8. Inspect top feature importances.

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, label_binarize

import tsfresh_utils as ttsfuti

logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# ## Part 1: Download & Load Data
#
# The UCI HAR dataset contains recordings from 30 subjects performing
# 6 activities while carrying a waist-mounted smartphone with inertial sensors:
#
# | Label | Activity             |
# |-------|----------------------|
# | 1     | WALKING              |
# | 2     | WALKING_UPSTAIRS     |
# | 3     | WALKING_DOWNSTAIRS   |
# | 4     | SITTING              |
# | 5     | STANDING             |
# | 6     | LAYING               |
#
# Each sample contains 128 time-steps captured at 50 Hz (~2.56 seconds).
# Nine sensor channels are provided (body acceleration x/y/z, gyroscope x/y/z,
# total acceleration x/y/z).

# %%
# Download and extract the dataset (skips download if already present).
har_root = ttsfuti.fetch_har_data(out_dir="./uci_har_data")
print("Dataset root:", har_root)

# %%
# Load train and test splits.
train_arrays, y_train_raw, subj_train = ttsfuti.load_inertial_split(
    "train", har_root
)
test_arrays, y_test_raw, subj_test = ttsfuti.load_inertial_split(
    "test", har_root
)

channels = list(train_arrays.keys())
n_train = train_arrays[channels[0]].shape[0]
n_test = test_arrays[channels[0]].shape[0]
n_time = train_arrays[channels[0]].shape[1]
print(
    f"Train: {n_train} samples  |  Test: {n_test} samples  "
    f"|  Timesteps: {n_time}  |  Channels: {len(channels)}"
)

# %% [markdown]
# ## Part 2: Convert to Long Format
#
# tsfresh expects data in long format with columns:
# - `id` – unique integer per sample.
# - `time` – timestep index (0 … 127).
# - `kind` – sensor channel name.
# - `value` – sensor reading.
#
# This lets tsfresh handle all 9 channels as a multivariate signal.

# %%
# Convert train split.
X_long_train = ttsfuti.to_long_format(train_arrays, start_id=0)
# Convert test split with ids that continue after train ids.
X_long_test = ttsfuti.to_long_format(
    test_arrays, start_id=n_train
)

print("Long-format train shape:", X_long_train.shape)
print(X_long_train.head(8))

# %%
# Build label Series indexed by sample id.
y_train = pd.Series(y_train_raw, index=np.arange(n_train))
y_test = pd.Series(y_test_raw, index=np.arange(n_train, n_train + n_test))

# Encode labels to 0-based integers.
le = LabelEncoder()
y_train_enc = pd.Series(
    le.fit_transform(y_train), index=y_train.index
)
y_test_enc = pd.Series(
    le.transform(y_test), index=y_test.index
)
print("Activity classes:", le.classes_)

# %% [markdown]
# ## Part 3: Feature Extraction
#
# We use `MinimalFCParameters` (mean, std, min, max, length, sum, median)
# for speed.  Switch to `EfficientFCParameters` or
# `ComprehensiveFCParameters` for a richer but slower feature set.

# %%
import tsfresh.feature_extraction as fe

fc_parameters = fe.MinimalFCParameters()

# Extract features for the training set.
X_feat_train = ttsfuti.extract_tsfresh_features(
    X_long_train, fc_parameters=fc_parameters
)
print("Train feature matrix shape:", X_feat_train.shape)

# %%
# Extract features for the test set using the same settings.
X_feat_test = ttsfuti.extract_tsfresh_features(
    X_long_test, fc_parameters=fc_parameters
)
print("Test feature matrix shape:", X_feat_test.shape)

# %% [markdown]
# ## Part 4: Feature Selection
#
# `tsfresh.select_features` runs the FRESH algorithm — hypothesis tests
# per feature — to keep only the features significantly correlated with the
# target labels.  This reduces noise and improves generalisation.

# %%
X_sel_train = ttsfuti.select_tsfresh_features(X_feat_train, y_train_enc)
print(
    f"Selected {X_sel_train.shape[1]} features out of {X_feat_train.shape[1]}"
)

# Align test features to the same columns selected from training.
X_sel_test = X_feat_test.reindex(
    columns=X_sel_train.columns, fill_value=0.0
)

# %% [markdown]
# ## Part 5: Classification & Evaluation
#
# We train a `RandomForestClassifier` and report:
# - **Accuracy** – fraction of correctly classified samples.
# - **Macro-F1** – averaged F1 score across all 6 classes.
# - **Macro-ROC-AUC** – one-vs-rest AUC averaged across classes.

# %%
clf = ttsfuti.train_classifier(X_sel_train, y_train_enc)

# %%
results = ttsfuti.evaluate_classifier(clf, X_sel_test, y_test_enc)
print(f"Accuracy    : {results['accuracy']:.3f}")
print(f"Macro-F1    : {results['macro_f1']:.3f}")
print(f"Macro-ROC-AUC: {results['macro_roc_auc']:.3f}")

# %%
# Full classification report.
proba = clf.predict_proba(X_sel_test)
pred = proba.argmax(axis=1)
print(
    metrics.classification_report(
        y_test_enc,
        pred,
        target_names=[str(c) for c in le.classes_],
    )
)

# %% [markdown]
# ## Part 6: Visualisation
#
# ### One-vs-Rest ROC Curves

# %%
n_classes = len(le.classes_)
y_test_bin = label_binarize(
    y_test_enc, classes=np.arange(n_classes)
)

plt.figure(figsize=(8, 6))
for k, cls in enumerate(le.classes_):
    fpr, tpr, _ = metrics.roc_curve(y_test_bin[:, k], proba[:, k])
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {cls} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC Curves — HAR (RandomForest + tsfresh)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Top Feature Importances

# %%
importances = pd.Series(
    clf.feature_importances_, index=X_sel_train.columns
).sort_values(ascending=False)

topk = 25
plt.figure(figsize=(8, 5))
ax = importances.iloc[:topk].sort_values().plot(kind="barh")
ax.set_title(f"Top {topk} tsfresh Feature Importances")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.show()
