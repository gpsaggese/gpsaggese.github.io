# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 13F stock recommendations
#
# - Implicit ALS on position size, vs recommending the most widely held names
# - Then which stocks sit near each other in the factor space

# %%
# %load_ext autoreload
# %autoreload 2

import logging
import os
import sys

import matplotlib.pyplot as plt
import pyspark.sql.functions as F
import seaborn as sns

# %%
# Find helpers_root from the project directory or the repo root.
_path = os.getcwd()
while not os.path.isdir(os.path.join(_path, "helpers_root", "helpers")):
    _parent = os.path.dirname(_path)
    if _parent == _path:
        break
    _path = _parent
sys.path.insert(0, os.path.join(_path, "helpers_root"))

import helpers.hnotebook as hnotebook

import mllib_utils as utils

hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# ## Ratings
#
# - $\log(1 + \mathrm{value})$, value in dollars
# - An open is a CUSIP held next quarter and not this quarter
# - `cold_share`: that CUSIP is missing from the training quarter, so ALS
#   has no item factor and it is left out of the metric
# - `median_positions` is why the top 10 skips the current book

# %%
holdings = utils.ingest_holdings(utils.project_data_dir())
# one row per quarter. the matrix is large
by_quarter = holdings.groupBy(
    F.col("report_period").cast("string").alias("quarter")
).agg(
    F.count("*").alias("rows"),
    F.countDistinct("cik").alias("managers"),
    F.countDistinct("cusip").alias("cusips"),
)
display(by_quarter.orderBy("quarter").toPandas())
# log so a $1B stake doesn't drown a $1M one
ratings = holdings.withColumn("rating", F.log1p("value")).persist()
ratings.count()
display(utils.open_summary(ratings))

# %% [markdown]
# ## ALS vs most-held
#
# - Tune `rank`, `regParam`, and `alpha` on the earliest pair only
# - `tuned` is that pair. `held_out` is the later ones
# - Both lists drop stocks already held, then keep 10
# - Small / mid / large are terciles of training-quarter dollars

# %%
# 2 x 2 x 2 = 8 models. this is the slow cell
grid = utils.tune_als(
    ratings, ranks=[8, 16], reg_params=[0.05, 0.1], alphas=[1.0, 40.0]
)
display(grid)
best = grid.iloc[0]
metrics, model, items = utils.score_quarters(
    ratings, int(best["rank"]), float(best["reg_param"]), float(best["alpha"])
)
overall = metrics[metrics["bucket"] == "all"]
display(overall.drop(columns=["rank", "reg_param", "alpha"]))
by_size = metrics[metrics["bucket"] != "all"]
display(
    by_size[
        [
            "test_period",
            "role",
            "bucket",
            "model",
            "precision_at_10",
            "ndcg_at_10",
            "n_managers",
            "cold_share",
        ]
    ]
)

# %%
# taller bar wins. first quarter on the x-axis is the one we tuned on
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
panels = (("precision_at_10", "precision@10"), ("ndcg_at_10", "NDCG@10"))
for ax, (column, title) in zip(axes, panels):
    sns.barplot(
        data=overall,
        x="test_period",
        y=column,
        hue="model",
        hue_order=["ALS", "most held"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylim(0, None)
axes[0].set_ylabel("score")
if axes[1].get_legend() is not None:
    axes[1].get_legend().remove()
fig.tight_layout()
display(fig)

# %% [markdown]
# ## Neighbors and clusters
#
# - Cosine similarity of the latest quarter's item factors
# - `KMeans` with 8 clusters. `examples` is the five most-held names

# %%
display(utils.item_neighbors(model, items, utils.KNOWN_STOCKS))
summary, points = utils.cluster_items(model, items, k=8)
display(summary)
# 2D view of the factors, color is the cluster
plot_df = points.copy()
plot_df["cluster"] = plot_df["cluster"].astype(str)
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    data=plot_df, x="x", y="y", hue="cluster", s=12, alpha=0.7, ax=ax
)
ax.set_xlabel("component 1")
ax.set_ylabel("component 2")
fig.tight_layout()
display(fig)
