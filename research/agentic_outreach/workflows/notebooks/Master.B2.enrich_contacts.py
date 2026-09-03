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

# +
db_path = "test.sql"

crm_df = ckmktwf.get_as_df(db_path, "Contact")

print(hpandas.head(crm_df))
# -

# ## Enrich data

# mode = "AssumeEmailValid"
mode = "FromScratch"
# dry_run = True
dry_run = False
# enrich_kwargs = {"issue_warnings": False}
enrich_kwargs = {}
contact_df2 = ckmktpi.hunterio_enrich_contact_df(
    contact_df,
    "first_name",
    "last_name",
    "company_name",
    mode=mode,
    dry_run=dry_run,
)
display(contact_df2.head(1))

print(contact_df2.columns)

ckmktwf.get_column_stats(
    contact_df2,
    ["email", "email_verification", "hunterio.email"],
    info_mode="only_pct",
)

ckmktwf.flush_cache_to_disk()  # "find_email")

ckmktwf.flush_cache_to_disk()  # "find_email")

# ## Merge Hunter.io enrichment

print(contact_df2.columns)

del contact_df_enriched

contact_df3 = ckmktpi.merge_hunterio_values(contact_df2)

ckmktwf.sanity_check_contact_df(contact_df3)

print(contact_df3.columns)

hpandas.head(contact_df3, num_rows=2)

ckmktwf.get_column_stats(contact_df3, ["email", "email_verification"])

stats_df = ckmktwf.print_contact_df_detailed_stats(contact_df3)

# +
# hyamm.save_to_gsheet(contact_df_enriched, name="contact_df_enriched")
# -

# ## Infer category

hpandas.get_value_counts_stats_df(contact_df3, "category")

contact_df_tmp = ckmktwf.infer_category(contact_df3)

# ## Validate email

ckmktpi.get_account_info()

contact_df4 = contact_df3.copy()
print(contact_df4.shape)
contact_df4.head(2)

ckmktwf.print_contact_df_stats(contact_df4)

ckmktwf.enable_cache_perf("verify_email")
mode = "AssumeEmailValid"
dry_run = True
is_company = True
contact_df5 = ckmktpi.process_email(
    contact_df4,
    "first_name",
    "last_name",
    "company_name",
    is_company,
    mode=mode,
    dry_run=dry_run,
)

ckmktwf.save_to_gsheet(contact_df5)

ckmktwf.print_contact_df_stats(contact_df5)

# +
# diagnostics_df = ckmktpi.get_diagnostic_df(contact_df_tmp)
# display(diagnostics_df.head(2))

# import pprint
# pprint.pprint(ckmktpi.get_stats(diagnostics_df), sort_dicts=False)
# -

ckmktwf.force_cache_from_disk()

ckmktwf.flush_cache_to_disk()  # "find_email")

print(ckmktwf.cache_stats_to_str())

cols = [
    "first_name",
    "last_name",
    "email",
    "company_name",
    "hunterio.email",
    "is_email_changed",
]
mask = contact_df_tmp["is_email_changed"]
contact_df_tmp.loc[mask][cols]

contact_df_tmp["hunterio.email_verification"].unique()

# ## Check category

contact_df_tmp2 = contact_df_tmp.copy()
display(contact_df_tmp2.head(1))

hpandas.get_value_counts_stats_df(contact_df_tmp2, "category", num_rows=20)

# +
# categories = ["Venture Fund (inferred)", "Venture Fund", "Venture Capital & Private Equity", "Accelerator", "Corporate VC", "Family Office"]
# contact_df_tmp2 = hpandas.filter_df(contact_df_tmp2, "category", categories)
# contact_df_tmp2 = hpandas.filter_df(contact_df_tmp2, "email", "_nan_", invert=True)

# +
# hpandas.filter_df(contact_df_tmp2, "category", "").head(3)
# -

contact_df_tmp2 = hpandas.filter_df(
    contact_df_tmp2, "category", ["Financial Services", "Family Office"]
)
# contact_df_tmp2 = hpandas.filter_df(contact_df_tmp2, "email", "_nan_", invert=True)
# mask = contact_df_tmp2["company_name"].isin(["Engineers Gate", "Teza Technologies", "RavenPack"])
# contact_df_tmp2 = contact_df_tmp2[~mask]

ckmktwf.sanity_check_contact_df(contact_df_tmp2)

# +
mode = "AssumeEmailValid"
contact_df_tmp3 = ckmktpi.process_email(
    contact_df_tmp2,
    "first_name",
    "last_name",
    "company_name",
    mode=mode,
    is_company=True,
)

contact_df_tmp3["email_verification"] = contact_df_tmp3[
    "hunterio.email_verification"
]
# -

contact_df_tmp3.head()

ckmktwf.save_to_gsheet(contact_df_tmp3, name="financial")

hpandas.get_value_counts_stats_df(contact_df_tmp3, "origin", num_rows=20)

contact_df = contact_df_tmp3

# # YAMM pipeline

# ## Read and merge data

yamm_df = ckmktwf.get_yamm_results()

display(yamm_df.head(3))

yamm_df["campaign_name"].unique()

ckmktwf.yamm_stats_to_pct(yamm_df)

ckmktwf.yamm_stats_by_campaign(yamm_df)

# +
# hyamm.save_to_gsheet(yamm_df)
# -

# ## Load data

url = "https://docs.google.com/spreadsheets/d/1KosRM5j6cFz8mm3Aw-ctIncAPOLx5bF1bkcEFrHnJsY"
yamm_df = ckmdatal.get_cached_gsheet_to_df(url, "Sheet1")
print(yamm_df.shape)
display(yamm_df.head(5))

# hyamm.yamm_stats(yamm_df)
ckmktwf.yamm_stats_to_pct(yamm_df)

# ## Update Contact_df with YAMM data

print(contact_df.shape)
contact_yamm_df = ckmktwf.update_contact_df_with_yamm_df(contact_df, yamm_df)
print(contact_yamm_df.shape)

# campaign_name = "campaign0_VC_causify"
campaign_name = "campaign2_VC_UMD"
# Pick the one already sent.
# df_tmp = hpandas.filter_df(contact_yamm_df, campaign_name, "", invert=True)
df_tmp = hpandas.filter_df(contact_yamm_df, campaign_name, "", invert=False)
ckmktwf.get_short_contact_df(df_tmp)

# +
# mask = yamm_df["campaign_name"] == "campaign2_VC_UMD"
# print(mask.sum())

# +
# mask2 = contact_yamm_df["campaign2_VC_UMD"] != ""
# print(mask2.sum())
# -

# # Extract YAMM / LIN campaign

assert 0

# campaign_col_name = "campaign0_VC_causify"
# campaign_col_name = "campaign1_VC_causify"
campaign_col_name = "campaign2_VC_UMD"
type_ = "email"
# type_ = "linkedin"
# num_rows = 10
num_rows = -1
campaign_df, contact_df2 = ckmktwf.select_campaign(
    contact_yamm_df, campaign_col_name, type_, num_rows, seed=2
)

contact_df2.head(2)

hpandas.filter_df(contact_df2, campaign_col_name, "selected").head(2)

print(campaign_df.shape)
campaign_df.head()

ckmktwf.save_to_gsheet(campaign_df, name="campaign_3_LIN_VC")

# # One-offs

#
(
    (
        "Wave1-20241210-folkapp1",
        "campaign0_VC_causify",
        "https://docs.google.com/spreadsheets/d/1mwRy0yTTCnTR14npWe7xATBYLb7DV9Pt1a2p4DjloQA",
        ["YAMM-20241210", "YAMM-20241210-1", "YAMM-20241210-2"],
    ),
)
#
(
    (
        "Wave2-20241210-folkapp1",
        "campaign0_VC_causify",
        "https://docs.google.com/spreadsheets/d/1eufg2XREYbXnCy8tygGKAkDigM0OE_fJRmnHTDxFQ8A",
        ["YAMM-2024-12-"],
    ),
)
#
(
    (
        "campaign_1_batch1",
        "campaign1_VC_causify",
        "https://docs.google.com/spreadsheets/d/10bWbYHdzl5KvvccHI5grtquFraO29MFP3iBcwkuVj1A",
        ["Sheet1", "Sheet2"],
    ),
)
#
(
    (
        "Campaign2_UMD_YAMM",
        "campaign2_VC_UMD",
        "https://docs.google.com/spreadsheets/d/1rpM5MeMtAwRvbV1fCngKD4"
        "-xe7Wc19ikvs7ljx9HIeA",
        ["2024-12-28"],
    ),
)

# +
# url = "https s://docs.google.com/spreadsheets/d/1rpM5MeMtAwRvbV1fCngKD4-xe7Wc19ikvs7ljx9HIeA"
# url = "https://docs.google.com/spreadsheets/d/1mwRy0yTTCnTR14npWe7xATBYLb7DV9Pt1a2p4DjloQA"
url = "https://docs.google.com/spreadsheets/d/1eufg2XREYbXnCy8tygGKAkDigM0OE_fJRmnHTDxFQ8A"
tab_name = "YAMM-2024-12-"
df = ckmdatal.get_cached_gsheet_to_df(url, tab_name)
# display(df.head(3))

res_df = df.merge(
    contact_df, left_on="email", right_on="email", how="left"
)  # [["hash"] + df.columns.tolist()]
hash_ = res_df["hash"]
res_df = df.copy()
res_df.insert(0, "hash", hash_)

res_df.head()
# -

hpandas.filter_df(res_df, "hash", "", check_value=False)

ckmktwf.save_to_gsheet(res_df, name="test1")

url = "https://docs.google.com/spreadsheets/d/1zJtF9q6NC9hEM3arUxr7vzVNmSyfxrybcYNkOaisjC4/edit?gid=796026511#gid=796026511"
df = ckmdatal.get_cached_gsheet_to_df(url, "SaaS Angel Investors (Globally)")

ckmktwf.save_to_gsheet(df)
