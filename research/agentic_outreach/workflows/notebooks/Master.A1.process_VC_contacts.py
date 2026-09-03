# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
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
# #     !sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet --upgrade google-api-python-client)"
# -

# !cd /app

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

import importlib

importlib.reload(ckmktwf)
importlib.reload(ckmktpi)
# -

# # Load Contact data

contact_dfs = []

# ## Search4.FinTech_VC_in_US.SalesNavigator

normalize = True
df_tmp = ckmktwf.get_data_from_PB_SalesNavigator_Connections_Export(normalize)
contact_dfs.append(df_tmp)

ckmktwf.print_contact_df_stats(df_tmp)

print(df_tmp.iloc[0])

# ## VC_search_export

normalize = True
df_tmp = ckmktwf.get_data_from_LinkedIn2(normalize)
contact_dfs.append(df_tmp)

ckmktwf.print_contact_df_stats(df_tmp)

# +
# ckmktwf.print_contact_df_detailed_stats(df_tmp)
# -

# ## Search7.AI_VC_in_US.gsheet

normalize = True
# normalize = False
df_tmp = ckmktwf.get_data_from_LinkedIn3(normalize)
contact_dfs.append(df_tmp)

ckmktwf.print_contact_df_stats(df_tmp)

# ## VC Tier 1 - Partners

normalize = True
df_tmp = ckmktwf.get_data_from_VCSheet(normalize)
contact_dfs.append(df_tmp)

ckmktwf.print_contact_df_stats(df_tmp)

# ## VC_Tier_2_Partners

normalize = True
df_tmp = ckmktwf.get_data_from_LinkedIn5(normalize)
contact_dfs.append(df_tmp)

ckmktwf.print_contact_df_stats(df_tmp)

# ## VCSheet

normalize = True
# normalize = False
df_tmp = ckmktwf.get_data_from_VCSheet(normalize)
contact_dfs.append(df_tmp)

ckmktwf.print_contact_df_stats(df_tmp)

# ## Euro-VCs

normalize = True
# normalize = False
df_tmp = ckmktwf.get_data_from_EuroVC(normalize)
contact_dfs.append(df_tmp)

ckmktwf.print_contact_df_stats(df_tmp)

# ## Folkapp

normalize = True
# normalize = False
df_tmp = ckmktwf.get_data_from_FolkApp(normalize)
contact_dfs.append(df_tmp)

ckmktwf.print_contact_df_stats(df_tmp)

# # Concat all

contact_df = pd.concat(contact_dfs)
debug = ""
# debug = "duplicated_emails"
# debug = "remove_chinese_names"
# debug = "remove_empty_first_name"
# debug = "clean_company_names"
# debug = "clean_linkedin_emails"
# debug = "clean_linkedin_websites"
contact_df = ckmktwf.clean_up_contact_df(contact_df, debug=debug)
display(contact_df.head(2))

contact_df.iloc[0]

ckmktwf.print_contact_df_detailed_stats(contact_df)

ckmktwf.sanity_check_contact_df(contact_df)

ckmktwf.print_contact_df_stats(contact_df)

display(contact_df["origin"].value_counts().to_frame())

# +
# ckmktwf.save_to_gsheet(contact_df)
# -

# # Process Contact data

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
# ckmktwf.save_to_gsheet(df)
# -

# # Load into DB

db_path = "test.sql"

# mode = "assume_no_overlap"
# mode = "assume_idempotent"
mode = "keep_new"
log_level = logging.DEBUG
df_out = ckmktwf.insert_contact_df(
    db_path, contact_df, mode, log_level=log_level
)

ckmktwf.print_contact_stats(db_path)
