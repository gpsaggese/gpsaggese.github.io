# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Imports

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# +
import helpers.hgoogle_drive_api as hgodrapi

hgodrapi.install_needed_modules()

# +
import logging

import pandas as pd

# /venv/lib/python3.12/site-packages/gspread_pandas/spread.py:401: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)` .replace("", np.nan)
pd.set_option("future.no_silent_downcasting", True)

import helpers.hdbg as hdbg
import helpers.henv as henv
import helpers.hpandas as hpandas
import helpers.hprint as hprint

#
hdbg.init_logger(verbosity=logging.INFO)
_LOG = logging.getLogger(__name__)

#
_LOG.info("%s", henv.get_system_signature()[0])
hnotebook.config_notebook()

# +
import helpers.hgoogle_drive_api as hgodrapi

# Get credentials first.
credentials = hgodrapi.get_credentials(
    service_key_path="/home/.config/gspread_pandas/google_secret.json"
)

import ck_marketing.workflows as ckmktwf
import ck_marketing.plugins as ckmktpi

import importlib

importlib.reload(ckmktwf)
importlib.reload(ckmktpi)
# -

# # Load Contact data

contact_dfs = []

# +
# normalize = True
normalize = False
url = "https://docs.google.com/spreadsheets/d/14ekmwAgpIxRzrf8EZeaZB8hudT7-PHGYb0BzK4jpdj8"
tab_name = "phantombuster-all-leads-12262025.csv"
tag = ""
df_tmp = ckmktwf.get_data_from_PhantomBuster_All_Leads(
    url, tab_name, tag, normalize
)

df_tmp.head(2)
# -

df_tmp.nunique()

df_tmp.iloc[-1]

df_tmp["createdBy"].unique()

# tag = "Deleted Phantom (1314085960650129)"
# tag = "[Deleted Phantom] (undefined)', 'LinkedIn Outreach - MA"
# tag = 'LinkedIn Auto Invitation Accepter'
tag = "MA_targets_10k_persons"
mask = df_tmp["createdBy"] == tag
mask.sum()

# +
url = "https://docs.google.com/spreadsheets/d/1uSS5h5f3bcWxCFxg90TTB46296JgnFGQeKDaIKtz_84/edit"
df1 = ckmktwf.get_gsheet_to_df(url, None)
# print(hpandas.head(df1))
print(df1.shape)
display(df1.head(2))

url = "https://docs.google.com/spreadsheets/d/1PZkWY8TYuEx7_WapG74AiwLLoLrDeGuHrQT8y3N4GXg/edit"
df2 = ckmktwf.get_gsheet_to_df(url, None)
# display(df2.head(2))
print(df2.shape)
# -

df1.equals(df2)


contact_dfs.append(df_tmp)

ckmktwf.print_contact_df_stats(df_tmp)

# ## Clean up names

# +
contact_df = ckmktpi.clean_and_track_name_changes(contact_df)

debug_df = ckmktpi.get_debug_clean_name_df(contact_df)
ckmktpi.get_clean_name_stats(contact_df)
# -

hpandas.filter_df(debug_df, "is_modified", True).head(2)

# +
contact_df = ckmktpi.merge_clean_names_df(contact_df)

contact_df.head(1)

# +
# hgodrapi.to_gsheet(
#     hgodrapi.get_credentials(),
#     contact_df,
#     url: str,
#     freeze_rows=True,
# ) -> None:

# +
# ckmktwf.save_to_gsheet(contact_df)
# -

# # Load into DB

db_path = "test.sql"

ckmktwf.get_table_count(db_path, "Contact")

# +
crm_df = ckmktwf.get_as_df(db_path, "Contact")

print(hpandas.head(crm_df))
# -

crm_df.origin.unique()

# mode = "assume_no_overlap"
mode = "keep_existing"
_ = ckmktwf.insert_contact_df(db_path, contact_df, mode)

# +
crm_df = ckmktwf.get_as_df(db_path, "Contact")

print(hpandas.head(crm_df))
print(crm_df.origin.unique())

ckmktwf.get_table_count(db_path, "Contact")
