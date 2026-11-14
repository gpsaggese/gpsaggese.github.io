# ---
# jupyter:
#   jupytext:
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
# # Imports

# %%
# %load_ext autoreload
# %autoreload 2

# System libraries.
import logging

# Third-party libraries.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Use this for most notebooks.
import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebook

_LOG = logging.getLogger(__name__)

# Initialize notebook configuration and logging.
hdbg.init_logger(verbosity=logging.INFO)
hnotebook.config_notebook()

# Convert `display` into `print()`.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# # Load Data

# %% [markdown]
# Heart disease dataset for binary classification. The Cleveland dataset has very few missing values and the combined dataset has many missing values in certain columns. The combined dataset also has one additional column for which institute the record came from.

# %%
heart_disease_cleveland = pd.read_csv('./datasets/heart_disease_cleveland.csv')
heart_disease_combined = pd.read_csv('./datasets/heart_disease_combined.csv')

# %%
heart_disease_cleveland.head()

# %%
heart_disease_combined = heart_disease_combined.drop(columns=['source'])
heart_disease_combined.head()

# %% [markdown]
# NYC Yellow Taxi dataset for regression

# %%
yellow_taxi = pd.read_parquet('./datasets/yellow_tripdata_2026-09.parquet')

# %%
yellow_taxi = yellow_taxi.drop(columns=['store_and_fwd_flag','request_source']).sample(n=1000,random_state=42)
yellow_taxi.head()

# %% [markdown]
# # EDA

# %% [markdown]
# ## Heart Disease Cleveland

# %% [markdown]
# ### Manual

# %%
num_rows = len(heart_disease_cleveland)
print('number of rows:',num_rows)
print('number of missing values:')
print(heart_disease_cleveland.isna().sum())
print('mean values:')
print(heart_disease_cleveland.mean())

columns = heart_disease_cleveland.columns
fig, (axs) = plt.subplots(1, 14)

for i in range(len(axs)):
    axs[i].boxplot(heart_disease_cleveland[columns[i]].dropna())
    axs[i].set_title(columns[i])
plt.show()

# %% [markdown]
# ## Heart Disease Combined

# %% [markdown]
# ### Manual

# %%
num_rows = len(heart_disease_combined)
print('number of rows:',num_rows)
print('number of missing values:')
print(heart_disease_combined.isna().sum())
print('mean values:')
print(heart_disease_combined.mean())

columns = heart_disease_combined.columns
fig, (axs) = plt.subplots(1, 14)

for i in range(len(axs)):
    axs[i].boxplot(heart_disease_combined[columns[i]].dropna())
    axs[i].set_title(columns[i])
plt.show()

# %% [markdown]
# ## Yellow Taxi

# %% [markdown]
# ### Manual

# %%
num_rows = len(yellow_taxi)
print('number of rows:',num_rows)
print('number of missing values:')
print(yellow_taxi.isna().sum())
print('mean values:')
print(yellow_taxi.mean())

columns = yellow_taxi.drop(columns=['tpep_pickup_datetime','tpep_dropoff_datetime']).columns
fig, (axs) = plt.subplots(2, 9)

for i in range(len(columns)):
    axs[i//9][i%9].boxplot(yellow_taxi[columns[i]].dropna())
    axs[i//9][i%9].set_title(columns[i])
plt.show()

# %%
