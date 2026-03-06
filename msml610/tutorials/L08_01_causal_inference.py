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
import msml610_utils as ut
import msml610.tutorials.L08_01_causal_inference_utils as mtl0cireout

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %%
import helpers.hmodule as hmodule
hmodule.install_module_if_not_present(
  "dataframe_image",
  use_activate=True,
)

# %%
from cycler import cycler

default_cycler = (
    cycler(color=["0.3", "0.5", "0.7", "0.5"])
    + cycler(linestyle=["-", "--", ":", "-."])
    + cycler(marker=["o", "v", "d", "p"])
)

color = ["0.3", "0.5", "0.7", "0.5"]
linestyle = ["-", "--", ":", "-."]
marker = ["o", "v", "d", "p"]

plt.rc("font", size=20)

# %% [markdown]
# # Cell 1: Sales example

# %%
dir_name = "L09_data"
# #!ls $dir_name

out_dir_name = "figures/L09"

# %%
data = pd.read_csv(dir_name + "/xmas_sales.csv")
data["is_on_sale"] = data["is_on_sale"].astype(float)
print(data.shape)
data.head(6)

# %%
import helpers.hpandas_display as hpandisp
hpandisp.convert_df_to_png(data.head(6), os.path.join(out_dir_name, 'xmas_sales_df.png'), index=True,
                           print_markdown=True,
                           markdown_path_prefix="msml610/lectures_source")
# # cp msml610/tutorials/figures/L09/* msml610/lectures_source/figures/L09/

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.boxplot(y="weekly_amount_sold", x="is_on_sale", data=data,
            ax=ax)

ax.set_xlabel("is_on_sale", fontsize=20)
ax.set_ylabel("weekly_amount_sold", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=18)

import helpers.hmatplotlib as hmatplo
hmatplo.save_fig(fig, os.path.join(out_dir_name, "xmas_boxplot.png"),
                  print_markdown=True,
                  path_prefix="msml610/lectures_source")

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

# %%
mtl0cireout.plot_sales_bias_analysis(data, marker)

# %%
mtl0cireout.plot_single_vs_separate_trends()

# %% [markdown]
# # Cell 4: Simpson's Paradox

# %%
mtl0cireout.plot_simpsons_paradox()
