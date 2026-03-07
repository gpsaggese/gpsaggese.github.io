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
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set plotting style.
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %%
import helpers.hmatplotlib as hmatplo
import helpers.hmodule as hmodule
import helpers.hpandas_display as hpandisp

import msml610_utils as ut
import L08_01_causal_inference_utils as mtl0cireout

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %%
hmodule.install_module_if_not_present(
  "dataframe_image",
  use_activate=True,
)

# %%
# from cycler import cycler

# default_cycler = (
#     cycler(color=["0.3", "0.5", "0.7", "0.5"])
#     + cycler(linestyle=["-", "--", ":", "-."])
#     + cycler(marker=["o", "v", "d", "p"])
# )

# color = ["0.3", "0.5", "0.7", "0.5"]
# linestyle = ["-", "--", ":", "-."]
# marker = ["o", "v", "d", "p"]

# plt.rc("font", size=20)

# %% [markdown]
# # Cell 1: Sales example

# %%
dir_name = "L08_data"
# #!ls $dir_name

out_dir_name = "figures/"

markdown_path_prefix="msml610/lectures_source"
# # cp msml610/lectures_source/figures/L08*.png msml610/lectures_source/figures

# %%
data = mtl0cireout.load_xmas_sales_data(dir_name)
print(data.shape)
data.head(6)

# %%
xmas_sales_df_png = os.path.join(out_dir_name, 'L08.4.xmas_sales_df.png')
hpandisp.convert_df_to_png(
    data.head(6),
    xmas_sales_df_png,
    index=True,
    print_markdown=True,
    markdown_path_prefix=markdown_path_prefix
)

# %% [markdown]
# - **Purpose**: Compare sales outcomes between stores with and without price cuts
# - **What it shows**: Box plots of weekly sales amounts for treated (cut prices) and control (no price cut) groups
# - **Key insight**: Visual evidence suggesting price cuts increase sales, but this may reflect confounding rather than true causal effect

# %%
fig = mtl0cireout.plot_xmas_sales_boxplot(data)
xmas_boxplot_png = os.path.join(out_dir_name, "L08.4.xmas_boxplot.png")
hmatplo.save_fig(
    fig,
    xmas_boxplot_png,
    print_markdown=True,
    path_prefix=markdown_path_prefix
)

# %% [markdown]
# ## Cell 2: Conceptual Example

# %%
# # i = unit identifier
# # y0, y1 = outcomes under control and treatment
# # t = treatment indicator
# # x = group
# df1 = pd.DataFrame(
#     dict(
#         i=[1, 2, 3, 4, 5, 6],
#         y0=[200, 120, 300, 450, 600, 600],
#         y1=[220, 140, 400, 500, 600, 800],
#         t=[0, 0, 0, 1, 1, 1],
#         x=[0, 0, 1, 0, 0, 1],
#     )
# )
# df1

# %%
# # Select the outcome based on the treatment.
# df1["y"] = (df1["t"] * df1["y1"] + (1 - df1["t"]) * df1["y0"]).astype(int)

# # Treatment effect.
# df1["te"] = df1["y1"] - df1["y0"]

# df1

# %%
# df2 = pd.DataFrame(
#     dict(
#         i=[1, 2, 3, 4, 5, 6],
#         y0=[
#             200,
#             120,
#             300,
#             np.nan,
#             np.nan,
#             np.nan,
#         ],
#         y1=[np.nan, np.nan, np.nan, 500, 600, 800],
#         t=[0, 0, 0, 1, 1, 1],
#         x=[0, 0, 1, 0, 0, 1],
#     )
# )
# df2

# %%
# # Select the outcome based on the treatment.
# df2["y"] = (df2["t"] * df2["y1"] + (1 - df2["t"]) * df2["y0"]).astype(int)

# # Treatment effect.
# df2["te"] = df2["y1"] - df2["y0"]

# df2

# %% [markdown]
# ## Cell 3: Visual Analysis of Bias in Sales Example

# %% [markdown]
# - **Purpose**: Visualize scatter points and regression lines for treated and control stores
# - **What it shows**: Treated stores (red) and control stores (blue) with their respective regression trends
# - **Key insight**: Within each group, the relationship between baseline sales and treatment appears similar, but overall pooled relationship is different

# %%
fig = mtl0cireout.plot_sales_bias_analysis(data)
bias_analysis0_png = os.path.join(out_dir_name, "L08.4.Association_Causation_Bias0.png")
hmatplo.save_fig(
    fig,
    bias_analysis0_png,
    print_markdown=True,
    path_prefix=markdown_path_prefix
)

# %% [markdown]
# - **Purpose**: Compare pooled vs. stratified regression models with synthetic data
# - **What it shows**: Left panel shows single trend line across all data; right panel shows separate trend lines for large and small businesses
# - **Key insight**: Simpson's paradox emerges when aggregation obscures group-level trends; stratification reveals the true relationships

# %%
fig = mtl0cireout.plot_single_vs_separate_trends()
bias_analysis1_png = os.path.join(out_dir_name, "L08.4.Association_Causation_Bias1.png")
hmatplo.save_fig(
    fig,
    bias_analysis1_png,
    print_markdown=True,
    path_prefix=markdown_path_prefix
)

# %% [markdown]
# ## Cell 4: Simpson's Paradox
#
# - **Purpose**: Illustrate Simpson's paradox where aggregate and group-level trends contradict
# - **What it shows**: Two groups (blue and red) with positive within-group trends, but negative overall trend
# - **Key insight**: Ignoring confounding variables (like business size) leads to contradictory causal conclusions

# %%
fig = mtl0cireout.plot_simpsons_paradox()
simpsons_paradox_png = os.path.join(out_dir_name, "L08.4.Simpson_Paradox.png")
hmatplo.save_fig(
    fig,
    simpsons_paradox_png,
    print_markdown=True,
    path_prefix=markdown_path_prefix
)
