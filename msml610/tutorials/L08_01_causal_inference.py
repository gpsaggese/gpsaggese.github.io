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
#
# - **Purpose**: Configure the notebook environment with required libraries and plotting style
# - **What it does**: Loads data science libraries (pandas, numpy, matplotlib, seaborn), enables autoreload for interactive development, and sets consistent plotting defaults
# - **Key insight**: Consistent styling across all plots ensures reproducibility and professional appearance

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

# %% [markdown]
# - **Purpose**: Import project-specific utilities and configure logging
# - **What it does**: Loads the notebook configuration utilities and the causal inference helper module with plotting and analysis functions
# - **Key insight**: Follows project conventions by using aliased imports (ut, mtl0cireout) for clarity

# %%
import msml610_utils as ut
import msml610.tutorials.L08_01_causal_inference_utils as mtl0cireout

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# - **Purpose**: Ensure optional dependencies are available in the notebook environment
# - **What it does**: Installs the dataframe_image package if not already present
# - **Key insight**: Allows dynamic dependency installation for features like converting dataframes to images

# %%
import helpers.hmodule as hmodule
hmodule.install_module_if_not_present(
  "dataframe_image",
  use_activate=True,
)

# %% [markdown]
# - **Purpose**: Set up consistent visualization parameters for all plots
# - **What it does**: Defines color schemes, line styles, markers, and font sizes for reproducible visualization
# - **Key insight**: Pre-configured cyclers ensure consistency and accessibility across multiple plot types

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
#
# - **Purpose**: Load and explore the Christmas sales dataset for causal inference analysis
# - **What it shows**: Real-world data with store information, seasonality, pricing actions, and sales outcomes
# - **Key insight**: Observational data where stores make pricing decisions based on their own characteristics

# %%
dir_name = "L09_data"
# #!ls $dir_name

out_dir_name = "figures/L09"

# %% [markdown]
# - **Purpose**: Load and examine the structure of the sales data
# - **What it shows**: The dataset contains 2000 observations with 5 columns: store ID, weeks until Christmas, average weekly sales, price cut indicator, and observed sales
# - **Key insight**: The is_on_sale column indicates treatment (price cut), while weekly_amount_sold is the outcome of interest

# %%
data = pd.read_csv(dir_name + "/xmas_sales.csv")
data["is_on_sale"] = data["is_on_sale"].astype(float)
print(data.shape)
data.head(6)

# %% [markdown]
# - **Purpose**: Create a visual representation of the data table for presentation
# - **What it does**: Converts the first 6 rows of the dataset into a PNG image
# - **Key insight**: Makes the data structure accessible in documents and presentations

# %%
import helpers.hpandas_display as hpandisp
hpandisp.convert_df_to_png(data.head(6), os.path.join(out_dir_name, 'xmas_sales_df.png'), index=True,
                           print_markdown=True,
                           markdown_path_prefix="msml610/lectures_source")
# # cp msml610/tutorials/figures/L09/* msml610/lectures_source/figures/L09/

# %% [markdown]
# - **Purpose**: Compare sales outcomes between stores with and without price cuts
# - **What it shows**: Box plots of weekly sales amounts for treated (cut prices) and control (no price cut) groups
# - **Key insight**: Visual evidence suggesting price cuts increase sales, but this may reflect confounding rather than true causal effect

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
#
# - **Purpose**: Introduce the potential outcomes framework fundamental to causal inference
# - **What it shows**: How each unit has two potential outcomes (y0 under control, y1 under treatment) and heterogeneous treatment effects
# - **Key insight**: The fundamental problem of causal inference is that we observe only one outcome per unit, never both

# %% [markdown]
# - **Purpose**: Illustrate the Rubin causal model with unit-level data
# - **What it shows**: Each unit i has potential outcomes y0 and y1 depending on treatment assignment
# - **Key insight**: Treatment effects (te = y1 - y0) vary across units; estimating these differences requires careful identification strategies

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

# %% [markdown]
# - **Purpose**: Calculate observed outcomes and individual-level treatment effects
# - **What it shows**: How observed outcome y depends on treatment: when treated (t=1) we see y1, otherwise y0
# - **Key insight**: Treatment effects vary from 0 (units 5) to 100 (unit 3); ignoring stratification would bias estimates

# %%
# # Select the outcome based on the treatment.
# df1["y"] = (df1["t"] * df1["y1"] + (1 - df1["t"]) * df1["y0"]).astype(int)

# # Treatment effect.
# df1["te"] = df1["y1"] - df1["y0"]

# df1

# %% [markdown]
# - **Purpose**: Demonstrate the fundamental problem of causal inference with missing counterfactuals
# - **What it shows**: The unobserved potential outcomes as NaN; we never observe both y0 and y1 for any unit
# - **Key insight**: This missing data structure is why causal inference requires identifying assumptions (randomization, no confounding, etc.)

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

# %% [markdown]
# - **Purpose**: Calculate observed outcomes with missing counterfactual data
# - **What it shows**: When we only observe treated units under treatment and control units under control, counterfactuals are missing
# - **Key insight**: Selecting observed values vs unobserved counterfactuals illustrates why assignment mechanism matters

# %%
# # Select the outcome based on the treatment.
# df2["y"] = (df2["t"] * df2["y1"] + (1 - df2["t"]) * df2["y0"]).astype(int)

# # Treatment effect.
# df2["te"] = df2["y1"] - df2["y0"]

# df2

# %% [markdown]
# ## Cell 3: Visual Analysis of Bias in Sales Example
#
# - **Purpose**: Demonstrate the importance of stratification in causal inference
# - **What it shows**: The difference between fitting a single regression line to pooled data vs. fitting separate lines for each subgroup
# - **Key insight**: The choice of regression model (pooled vs. stratified) can lead to different conclusions about the relationship between pricing and sales for different business sizes

# %% [markdown]
# - **Purpose**: Visualize scatter points and regression lines for treated and control stores
# - **What it shows**: Treated stores (red) and control stores (blue) with their respective regression trends
# - **Key insight**: Within each group, the relationship between baseline sales and treatment appears similar, but overall pooled relationship is different

# %%
mtl0cireout.plot_sales_bias_analysis(data, marker)

# %% [markdown]
# - **Purpose**: Compare pooled vs. stratified regression models with synthetic data
# - **What it shows**: Left panel shows single trend line across all data; right panel shows separate trend lines for large and small businesses
# - **Key insight**: Simpson's paradox emerges when aggregation obscures group-level trends; stratification reveals the true relationships

# %%
mtl0cireout.plot_single_vs_separate_trends()

# %% [markdown]
# ## Cell 4: Simpson's Paradox
#
# - **Purpose**: Illustrate Simpson's paradox where aggregate and group-level trends contradict
# - **What it shows**: Two groups (blue and red) with positive within-group trends, but negative overall trend
# - **Key insight**: Ignoring confounding variables (like business size) leads to contradictory causal conclusions

# %%
mtl0cireout.plot_simpsons_paradox()
