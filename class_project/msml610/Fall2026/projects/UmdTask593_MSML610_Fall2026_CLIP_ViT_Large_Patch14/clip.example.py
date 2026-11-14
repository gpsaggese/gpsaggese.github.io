# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Multimodal Sentiment Analysis with CLIP
#
# This notebook predicts the sentiment (negative / neutral / positive) of
# tweets that contain both an image and a text, using frozen
# **CLIP ViT-L/14** embeddings.
#
# Plan:
# 1. Load the MVSA-Multiple posts and their cached CLIP embeddings.
# 2. Build baselines: majority class, zero-shot CLIP, logistic regression.
# 3. Train a custom fusion head on top of the frozen embeddings.
# 4. Compare all models on the same test set.
#
# The data preparation and embedding extraction are in `clip_utils.py`; see
# `clip.API.ipynb` for an introduction to the CLIP API.
#
# References:
# - Niu et al., *Sentiment Analysis on Multi-view Social Data*, MMM 2016
#   (MVSA dataset)
# - Xu and Mao, *MultiSentiNet*, CIKM 2017 (label aggregation)
# - Radford et al., *Learning Transferable Visual Models From Natural Language
#   Supervision*, 2021 (CLIP)

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

# System libraries.
import logging
import os

# Third-party libraries.
import pandas as pd
import torch

# Use all CPU cores for the CLIP text encoder.
torch.set_num_threads(os.cpu_count())

# %%
import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebook

import clip_utils as cliputil

_LOG = logging.getLogger(__name__)

# Initialize notebook configuration and logging.
hdbg.init_logger(verbosity=logging.INFO)
hnotebook.config_notebook()
cliputil.init_loggers(_LOG)

# %% [markdown]
# ## Load the Data
#
# The first run of the cells below builds the post table and extracts the
# CLIP embeddings (about 3 hours on CPU). Later runs load both from
# `data/processed/` in a few seconds.

# %%
LABELS_PATH = "data/processed/labels.csv"
CACHE_PATH = "data/processed/clip_embeddings.npz"

# Build the post table and the embedding cache only if they are missing.
if not os.path.exists(LABELS_PATH):
    cliputil.build_mvsa_table("data/raw/MVSA", LABELS_PATH)
if not os.path.exists(CACHE_PATH):
    cliputil.extract_embeddings(pd.read_csv(LABELS_PATH), CACHE_PATH)
df, emb = cliputil.load_project_data(LABELS_PATH, CACHE_PATH)
print(f"Posts: {len(df)}  Image embeddings: {emb['image'].shape}  "
      f"Text embeddings: {emb['text'].shape}")

# %% [markdown]
# The classes are imbalanced, and the stratified split keeps the same
# proportions in train, validation, and test.

# %%
cliputil.summarize_splits(df)

# %% [markdown]
# ## Evaluation Protocol
#
# - **Main metric: macro F1**, the average F1 over the 3 classes. With about
#   2/3 positive posts, a model that always says "positive" already gets high
#   accuracy, while its macro F1 stays low.
# - Hyperparameters are chosen on the **validation** split. The **test**
#   split is used only to report the final numbers.
# - **Same training budget for every trained model**: the linear baselines
#   and the 3 fusion heads all use the same loop (class-weighted
#   cross-entropy, AdamW, batch 128, learning rate 1e-3, weight decay 1e-2)
#   for exactly `EPOCHS` epochs. The weights of the epoch with the best
#   validation macro F1 are kept.

# %%
EPOCHS = 20
TRAIN = dict(epochs=EPOCHS, batch_size=128, lr=1e-3, weight_decay=1e-2)

y = df["label_id"].to_numpy()
train_idx = cliputil.split_indices(df, "train")
val_idx = cliputil.split_indices(df, "val")
test_idx = cliputil.split_indices(df, "test")

results = {}
test_preds = {}
histories = {}


def add_result(name, model, make_batch):
    """
    Evaluate a trained model on the test split and store the results.
    """
    probs, gates = cliputil.predict_head(model, make_batch, test_idx)
    y_pred = probs.argmax(axis=1)
    results[name] = cliputil.evaluate(y[test_idx], y_pred)
    test_preds[name] = (y[test_idx], y_pred)
    return gates

# %% [markdown]
# ## Baseline 0: Majority Class
#
# Always predict the most frequent training label. Every model must beat
# this.

# %%
_, y_train = cliputil.split_xy(df, emb["image"], "train")
_, y_test = cliputil.split_xy(df, emb["image"], "test")

y_pred = cliputil.majority_baseline(y_train, len(y_test))
results["Majority class"] = cliputil.evaluate(y_test, y_pred)
test_preds["Majority class"] = (y_test, y_pred)
cliputil.results_table(results)

# %% [markdown]
# ## Baseline 1: Zero-Shot CLIP
#
# No training: compare each image and each text with one prompt per class,
# as in `clip.API.ipynb`, and average the two probability vectors. Only the
# prompts need the CLIP text encoder.

# %%
model, processor = cliputil.load_clip()

image_prompts = [
    "a photo with a negative mood",
    "a photo with a neutral mood",
    "a photo with a positive mood",
]
text_prompts = [
    "a negative message",
    "a neutral message",
    "a positive message",
]

probs_img = cliputil.zero_shot_probs(
    emb["image"][test_idx], image_prompts, model, processor
)
probs_txt = cliputil.zero_shot_probs(
    emb["text"][test_idx], text_prompts, model, processor
)
y_pred = ((probs_img + probs_txt) / 2).argmax(axis=1)
name = "Zero-shot (image+text)"
results[name] = cliputil.evaluate(y[test_idx], y_pred)
test_preds[name] = (y[test_idx], y_pred)
cliputil.results_table(results)

# %% [markdown]
# ## Baseline 2: Logistic Regression on Frozen Embeddings
#
# A linear classifier (logistic regression) on 3 inputs:
# - `image`: image embedding only
# - `text`: text embedding only
# - `concat`: both embeddings side by side
#
# The features are standardized with the training mean and std. It is
# trained with the same loop and number of epochs as the fusion heads.

# %%
for kind in cliputil.FEATURE_SETS:
    X = cliputil.build_features(emb, kind)
    name = f"LogReg ({kind})"
    clf = cliputil.make_linear_head(X, train_idx)
    clf, histories[name] = cliputil.train_head(
        clf, cliputil.feature_batch(X), y, train_idx, val_idx, **TRAIN
    )
    add_result(name, clf, cliputil.feature_batch(X))

# %% [markdown]
# ## Baseline Results

# %%
cliputil.results_table(results)

# %% [markdown]
# Each row of a confusion matrix sums to 1, so the diagonal is the recall of
# each class.

# %%
cliputil.plot_confusion_matrices(test_preds)

# %% [markdown]
# ## Custom Fusion Heads
#
# We train 3 small heads on top of the frozen CLIP features, with the same
# training settings (`TRAIN`) as the linear baselines.
#
# | Head | Input | Idea |
# |---|---|---|
# | H1 Stack | 768-d embeddings | standardize, average image and text, linear layer |
# | H2 Gated fusion | 768-d embeddings | text is the anchor; the image is added through a learned gate |
# | H3 Cross-attention | token features | image and text tokens attend to each other |

# %% [markdown]
# ### H1: Stack Head
#
# Each modality is standardized with the training mean and std, the two are
# averaged with fixed weights 1/2, and one linear layer predicts the class.
# This tests whether summing in CLIP's shared space works as well as
# concatenation.

# %%
emb_batch = cliputil.embedding_batch(emb)

h1 = cliputil.make_stack_head(emb, train_idx)
print("H1 parameters:", cliputil.count_params(h1))
h1, histories["H1 Stack"] = cliputil.train_head(
    h1, emb_batch, y, train_idx, val_idx, **TRAIN
)
add_result("H1 Stack", h1, emb_batch)
cliputil.results_table(results)

# %% [markdown]
# ### H2: Gated Fusion (anchor = text)
#
# The text is the main input. The image is added as `z + g * a`, where the
# gate `g` (one number per post, between 0.05 and 0.95) is computed from both
# modalities and their CLIP cosine similarity. The gate starts at 0.5.

# %%
h2 = cliputil.GatedFusionHead(anchor="text")
print("H2 parameters:", cliputil.count_params(h2))
h2, histories["H2 Gated"] = cliputil.train_head(
    h2, emb_batch, y, train_idx, val_idx, **TRAIN
)
h2_gates = add_result("H2 Gated", h2, emb_batch)
cliputil.results_table(results)

# %% [markdown]
# The gate shows how much each post relies on the image. A higher mean `g`
# for a class means the head uses the image more for that class.

# %%
cliputil.gate_table(h2_gates, y[test_idx])

# %% [markdown]
# ### H3: Token-Level Cross-Modal Attention
#
# H3 needs the token features (CLS + 8x8 image patches, and all text tokens),
# extracted once with `python extract_tokens.py text` and
# `python extract_tokens.py image`. They are read from disk batch by batch.

# %%
TOKENS_DIR = "data/processed/clip_tokens"
tok = cliputil.load_token_features(TOKENS_DIR)
assert (tok["ids"] == df["id"].to_numpy()).all(), "Token features do not match df"
tok_batch = cliputil.token_batch(tok)

h3 = cliputil.TokenCrossAttentionHead()
print("H3 parameters:", cliputil.count_params(h3))
h3, histories["H3 Cross-attn"] = cliputil.train_head(
    h3, tok_batch, y, train_idx, val_idx, **TRAIN
)
add_result("H3 Cross-attn", h3, tok_batch)
cliputil.results_table(results)

# %% [markdown]
# ## Results

# %%
cliputil.plot_history(histories)

# %%
cliputil.results_table(results)

# %%
cliputil.plot_confusion_matrices(test_preds)

# %% [markdown]
# ## Summary
#
# TODO: summarize the comparison once the results are in.
