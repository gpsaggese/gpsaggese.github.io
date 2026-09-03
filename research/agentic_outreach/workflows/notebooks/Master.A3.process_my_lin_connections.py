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

# +
# if False:
# #     !sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet jupyterlab-vim)"
# #     !jupyter labextension enable
# -

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# +
import helpers.hgoogle_drive_api as hgodrapi

hgodrapi.install_needed_modules()

import helpers.hllm_cli as hllmcli

hllmcli.install_needed_modules()
# -

# Make sure to use the central cache.
# !cd /app

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
import gspread
import gspread_pandas

# gspread_pandas.conf.get_config()
print(gspread_pandas.conf.get_config()["project_id"])
print(gspread.__version__)
print(gspread_pandas.__version__)

import helpers.hgoogle_drive_api as hgodrapi

# Get credentials first.
credentials = hgodrapi.get_credentials(
    service_key_path="/home/.config/gspread_pandas/google_secret.json"
)

import ck_marketing.workflows as ckmktwf
import ck_marketing.plugins as ckmktpi


# +
import importlib

importlib.reload(ckmktwf)
importlib.reload(ckmktpi)
importlib.reload(hpandas)
# -

# # Load Contact data

contact_dfs = []

# ## GP_LIn_Connections (2024-12-31)

normalize = True
# normalize = False
df_tmp = ckmktwf.get_data_from_GP_LIn_connections(normalize)
print(hpandas.head(df_tmp))
contact_dfs.append(df_tmp)

ckmktwf.print_contact_df_stats(df_tmp)

# ## GP_Lin_Connections (2025_12_05)

# +
# The problem is that:
# 1) It's not easy to export all the LinkedIn connections at once
# 2) We have exported in chunks
# 3) Some of the connections were enriched

# The solution is to find the connections that are not enriched.

# GP_LinkedIn_Connections_2024_12_31
# url1 = "https://docs.google.com/spreadsheets/d/19ziUmqbPaUO73cqlJB1F9y-j1Oq98nMzo6wTmzyVnwg/edit?gid=568530034#gid=568530034"
# tab_name1 = "Sheet1"
# df1 = hgodrapi.from_gsheet(url1, tab_name=tab_name1, credentials=credentials)
df1 = ckmktwf.get_data_from_GP_LIn_connections(normalize)
print(df1.shape)

# GP_LinkedIn_Connections_2025_12_05
url2 = "https://docs.google.com/spreadsheets/d/1vz4cYvWOjkIkNQghIhj6DBKUr6OZ7bSEUKNlv0ic-Xk/edit?gid=1753306632#gid=1753306632"
tab_name2 = "Sheet3"
tag = "GP_Lin_Connections_after_2025_12_05"
normalize = True
# df2 = hgodrapi.from_gsheet(url2, tab_name=tab_name2, credentials=credentials)
df2 = ckmktwf.get_data_from_LinkedIn_Connections_Exports(
    url2, tab_name2, tag, normalize
)
print(df2.shape)
# -

df1.head(2)

df2.head(2)

print(df1.shape)
print(df2.shape)

# +
cols = ["first_name", "last_name"]

print(df1[cols].dropna())
print(df2[cols].dropna())

len(set(df2[cols]) - set(df1[cols]))

# +
diff = df1.merge(
    df2,
    on=cols,
    # how="left",
    how="outer",
    indicator=True,
)

print("common=", (diff[diff["_merge"] == "both"]).shape)
print("df1 only=", (diff[diff["_merge"] == "left_only"]).shape)
print("df2 only=", (diff[diff["_merge"] == "right_only"]).shape)
# -

# Find the values that are only df2.
df2_only = df2.merge(
    diff.loc[diff["_merge"].eq("right_only"), cols].drop_duplicates(),
    on=cols,
    how="inner",
)

df2_only.head(3)

contact_dfs.append(df2)

# # Concat all

print(len(contact_dfs))
# Contacts from LinkedIn connections might have no emails.
allow_no_emails = True
contact_df = pd.concat(contact_dfs)
debug = ""
# debug = "duplicated_emails"
# debug = "remove_chinese_names"
# debug = "remove_empty_first_name"
# debug = "clean_company_names"
# debug = "clean_linkedin_emails"
# debug = "clean_linkedin_websites"
contact_df = ckmktwf.clean_up_contact_df(
    contact_df, allow_no_emails=allow_no_emails, debug=debug
)
display(contact_df.head(2))

contact_df.iloc[0]

ckmktwf.print_contact_df_detailed_stats(contact_df)

ckmktwf.sanity_check_contact_df(contact_df)

ckmktwf.print_contact_df_stats(contact_df)

contact_df["origin"].value_counts()

# +
# ckmktwf.save_to_gsheet(contact_df)
# -

# ## Clean up names

# +
contact_df = ckmktpi.clean_and_track_name_changes(contact_df)

debug_df = ckmktpi.get_debug_clean_name_df(contact_df)
ckmktpi.get_clean_name_stats(contact_df)
# -

hpandas.filter_df(debug_df, "is_modified", True).head(10)

# +
contact_df = ckmktpi.merge_clean_names_df(contact_df)

contact_df.head(1)
# -

contact_df.shape

# +
# hgodrapi.save_df_to_tmp_gsheet(contact_df)
# -

# # Classify

df = contact_df.head(500)

# hgodrapi.save_df_to_tmp_gsheet(df, remove_empty_columns=True)
url = "https://docs.google.com/spreadsheets/d/1y7A__hpV8n9nyQpNLDso6fGVsjBu5LDfgx1xYU6cwOI/edit?gid=0#gid=0"
hgodrapi.save_df_to_tmp_gsheet(df, url=url, tab_name="before")

hllmcli.shutup_llm_logging()

model = "gpt-4o-mini"
df = ckmktwf.classify_industry_type_executive(df, model=model)

hpandas.display_value_counts_stats_df(
    df,
    col_names=["type", "industry", "executive"],
    num_rows=10,  # Number of top rows to return
)

# +
# hpandas.remove_empty_columns(hpandas.filter_df(df, "type", "unknown"))
# -

df.head(2)

hgodrapi.save_df_to_tmp_gsheet(df, remove_empty_columns=True)

# # Load into DB

assert 0

db_path = "test.sql"

ckmktwf.print_table_schema(db_path, tables=["Contact"])

ckmktwf.get_table_count(db_path, "Contact")

# +
crm_df = ckmktwf.get_as_df(db_path, "Contact")

hpandas.head(crm_df)
# -

crm_df.origin.unique()

# mode = "assume_no_overlap"
mode = "assume_idempotent"
ckmktwf.insert_contact_df(db_path, contact_df, mode)

# +
crm_df = ckmktwf.get_as_df(db_path, "Contact")

print(hpandas.head(crm_df))
print(crm_df.origin.unique())

ckmktwf.get_table_count(db_path, "Contact")
