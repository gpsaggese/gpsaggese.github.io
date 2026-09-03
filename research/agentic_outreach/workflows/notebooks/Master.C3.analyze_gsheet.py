# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Import

# %%
if False:
    # !sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet jupyterlab-vim)"
    # !jupyter labextension enable

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# %%
import helpers.hgoogle_drive_api as hgodrapi

hgodrapi.install_needed_modules()

# %%
import logging

import helpers.hdbg as hdbg
import helpers.henv as henv
import helpers.hprint as hprint

print(henv.get_system_signature()[0])

hnotebook.config_notebook()

# hdbg.init_logger(verbosity=logging.DEBUG)
hdbg.init_logger(verbosity=logging.INFO)
# hdbg.test_logger()
_LOG = logging.getLogger(__name__)

# %% [markdown]
# ## Reload

# %%
import importlib
import pandas as pd

import ck_marketing.workflows as ckmktwf
import ck_marketing.plugins as ckmktpi
import helpers.hgoogle_drive_api as hgodrapi

importlib.reload(ckmktwf)
importlib.reload(ckmktpi)

# Get credentials first
credentials = hgodrapi.get_credentials(
    service_key_path="/home/.config/gspread_pandas/google_secret.json"
)

# %% [markdown]
# # Load data

# %% [markdown]
# ## pitchbook.Outreach_AI_companies

# %%
url = "https://docs.google.com/spreadsheets/d/1GnnmtGTrHDwMP77VylEK0bSF_RLUV5BWf1iGmxuBQpI"
hgodrapi.print_info_about_google_url(url)

# %%
tab_name = "_all_"
ckmktwf.analyze_data_type(url, tab_name)

# %%
tab_names = ["2025-12-03", "2025-12-03-v2"]
normalize = True
contact_df = ckmktwf.get_data_from_fuzzy_column_matching(url, tab_names, normalize)

display(contact_df.head(2))
print(contact_df.shape)

# %%
merge_status_tabs = ["2025-12-03", "2025-12-03-v2"]
contact_df = ckmktwf.merge_yamm_results_from_gsheet(contact_df, url, merge_status_tabs, print_results=True)

# %%
contact_df

# %% [markdown]
# ## Pitchbook.results

# %%
url = "https://docs.google.com/spreadsheets/d/1aKzWUw9mwP-2_vzz27ggeLe1sgF9OrWRWmGMo9Dk9bU"
hgodrapi.print_info_about_google_url(url)

# %%
tab_name = "_all_"
ckmktwf.analyze_data_type(url, tab_name)

# %%
#tab_name = "200AI.Corp_dev"
#tab_name = "Fort500.Corp_dev"
tab_name = "_all_"
normalize = True
contact_df = ckmktwf.get_data_from_fuzzy_column_matching(url, tab_name, normalize)
contact_df.head(2)

# %%
# No merge status in this data.
#merge_status_tabs = ["200AI.Corp_dev", "Fort500.Corp_dev"]
# df = ckmktwf.merge_yamm_results_from_gsheet(contact_df, url, merge_status_tabs)

# %% [markdown]
# ## Acq outreach

# %%
url = "https://docs.google.com/spreadsheets/d/1HyglraD02TJwp16wkU_yZZ6jJW51LnBdlaWsVSTNLws"
hgodrapi.print_info_about_google_url(url)

# %%
tab_name = "_all_"
ckmktwf.analyze_data_type(url, tab_name)

# %%
# tab_names = ['Sheet1', 'Sheet14', 'all', 'Sheet2', '2025-12-12', '2025-12-16', '2025-12-16-v2', 'Sheet12', 'Sheet13']
# tab_name = tab_names[0]
# normalize = True
# contact_df = ckmktwf.get_data_from_fuzzy_column_matching(url, tab_name, normalize)
# contact_df.head(2)

# %%
tab_name = "all"
normalize = True
contact_df = ckmktwf.get_data_from_fuzzy_column_matching(url, tab_name, normalize)
contact_df.head(2)

# %%
merge_status_tabs = ['Sheet1', 'Sheet14', 'Sheet2', '2025-12-12', '2025-12-16', '2025-12-16-v2']

contact_df = ckmktwf.merge_yamm_results_from_gsheet(contact_df, url, merge_status_tabs)

# %%
df.head(2)

# %%
ckmktwf.print_stats(contact_df)

# %% [markdown]
# ## pitchbook.Outreach_AI_companies

# %%
url = "https://docs.google.com/spreadsheets/d/1GnnmtGTrHDwMP77VylEK0bSF_RLUV5BWf1iGmxuBQpI"
hgodrapi.print_info_about_google_url(url)

# %%
tab_name = "_all_"
ckmktwf.analyze_data_type(url, tab_name)

# %%
tab_name = ['2025-12-03', '2025-12-03-v2']
normalize = True
contact_df = ckmktwf.get_data_from_fuzzy_column_matching(url, tab_name, normalize)
contact_df.head(2)
print(contact_df.shape)

# %%
merge_status_tabs = ['2025-12-03', '2025-12-03-v2']
contact_df = ckmktwf.merge_yamm_results_from_gsheet(contact_df, url, merge_status_tabs)

display(contact_df.head(2))

ckmktwf.print_stats(contact_df)

# %% [markdown]
# ## MA_targets_10k_persons.2025-12-25

# %%
url = "https://docs.google.com/spreadsheets/d/1uSS5h5f3bcWxCFxg90TTB46296JgnFGQeKDaIKtz_84"
hgodrapi.print_info_about_google_url(url)

tab_name = "_all_"
ckmktwf.analyze_data_type(url, tab_name)

# %% [markdown]
# ## Connect to MA on LinkedIn

# %%
url = "https://docs.google.com/spreadsheets/d/1tXnir8P5VjBxvbQOgJZmArRa7K__J7Fy90rWZtIxXTE"
hgodrapi.print_info_about_google_url(url)

tab_name = "_all_"
ckmktwf.analyze_data_type(url, tab_name)

# %%
url = "https://docs.google.com/spreadsheets/d/1vfWbjN7aYYOdRKwvGs03p-wMQVKgRkpjvVpb-lbhe4A"
hgodrapi.print_info_about_google_url(url)

tab_name = "_all_"
ckmktwf.analyze_data_type(url, tab_name)

# %%
url = "https://docs.google.com/spreadsheets/d/1t4zpgpNMVoBXnjOy1b8e0F15dPHUyxqXAYtp9yFewE4"
hgodrapi.print_info_about_google_url(url)

tab_name = "_all_"
ckmktwf.analyze_data_type(url, tab_name)

# %%
url = "https://docs.google.com/spreadsheets/d/1ehxS764nQYJAtz24YG2oxNPmLYbY04AYWQCU9nDQSko"
hgodrapi.print_info_about_google_url(url)

tab_name = "_all_"
ckmktwf.analyze_data_type(url, tab_name)

# %%
url = "https://docs.google.com/spreadsheets/d/14ekmwAgpIxRzrf8EZeaZB8hudT7-PHGYb0BzK4jpdj8/edit?gid=887396004#gid=887396004"
hgodrapi.print_info_about_google_url(url)

tab_name = "_all_"
ckmktwf.analyze_data_type(url, tab_name, log_level=logging.INFO)

# %%
db_path = "test.db"
id_ = 1
name = "email.partnership.top200ai.cxo"
type_ = "email"
description = ""
message = """
Hi Aaron,

I'm GP Saggese, and I teach machine learning at the University of Maryland, where my team is developing new methods that let LLMs reason explicitly about probability and time.

We're exploring partnerships with product teams who could benefit from advanced AI that understands probability and time. Given Box's focus on collaboration and document management, I see a few areas where these capabilities could be impactful, such as:
- predicting document utilization trends for better resource allocation
- forecasting compliance timelines within workflows
- analyzing user engagement patterns to enhance the Box experience

We have deployed this to early partners, and we have already $1M ARR in less than 12 months.

Would someone on your team be open to a brief 10-minute conversation to see if any of this aligns with Box's roadmap?
"""
start_date = pd.Timestamp.now()
num_targets = 0

ckmktwf.create_campaign_info_id(
    db_path, id_, name, type_, description, message, start_date, num_targets
)

# %% [markdown]
# ## Acq12

# %%
url = "https://docs.google.com/spreadsheets/d/1pQ8JCCI9bMQ_woOZJ0P4uRdUZvZP-rPJv9xnycRMw2Y/edit?gid=1923055155#gid=1923055155"

hgodrapi.print_info_about_google_url(url)

tab_name = "All"
ckmktwf.analyze_data_type(url, tab_name)

# %%
email_tab = "All"
merge_status_tabs = "20250210-1 20250211-2 20250212-3 20250212-4 20250213-5 20250213-6".split()

df = ckmktwf.load_all_yamm_results(url, email_tab, merge_status_tabs)
