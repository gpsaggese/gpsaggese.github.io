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
normalize = True
# normalize = False
url = "https://docs.google.com/spreadsheets/d/1uSS5h5f3bcWxCFxg90TTB46296JgnFGQeKDaIKtz_84/edit?gid=2041749088#gid=2041749088"
tab_name = "LinkedIn_Profile_Scraper"
tag = "MA_targets_10k_persons"
df_tmp = ckmktwf.get_data_from_LinkedIn_Profile_Scraper(
    url, tab_name, tag, normalize
)

df_tmp.head(2)
# -

contact_dfs.append(df_tmp)

ckmktwf.print_contact_df_stats(df_tmp)

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

# # Process Contact data

# ## Read data

# +
# assert 0
# -

if False:
    # url = "https://docs.google.com/spreadsheets/d/1nlfOHLUo2iTuNtFb2T5ZJXZZnztAnr3WzG9Jm5lbk6I"
    url = "https://docs.google.com/spreadsheets/d/1HanQiolicUToLgQFv1__cPOP4WLks5r7pHcgu2P8BjM"
    contact_df = ckmdatal.get_cached_sheet_to_df(url, "Sheet1")
    contact_df.set_index("hash", drop=True, inplace=True)
    #
    hpandas.head(contact_df)

# +
# contact_df = ckmktwf.clean_up_contact_df(contact_df,

# +
# ckmktwf.sanity_check_contact_df(contact_df)

# +
# ckmktwf.print_contact_df_stats(contact_df)

# +
# contact_df["origin"].value_counts()

# +
# stats_df = ckmktwf.print_contact_df_detailed_stats(contact_df)
# -

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
