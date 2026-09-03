# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# #!sudo /bin/bash -c "(source /venv/bin/activate; pip install --upgrade google-api-python-client)"

# %%
import importlib
import os

import ck_marketing.plugins.hunterio.hunterio_api as cmphhuap
import helpers.hgoogle_file_api as hgofiapi

importlib.reload(hgofiapi)
importlib.reload(cmphhuap)

# %%
hunter_api = os.environ.get("HUNTER_API_KEY")
drop_api = os.environ.get("DROPCONTACT_API_KEY")

# %%
creds = hgofiapi.get_credentials()

# %%
link = "https://docs.google.com/spreadsheets/d/1o07XnIArFdIjz0jTZuyPkldxBczhQhMywdF9Xe9kNmo/edit?gid=1379255156#gid=1379255156"
df = hgofiapi.read_google_file(link, "chatgpt_extract", credentials=creds)

# %%
df.head()

# %%
df[["firstName", "lastName"]] = df["Name"].str.split(" ", n=1, expand=True)

# %%
df.head()

# %%
df.shape

# %%
df_test = df.iloc[50:60]

# %%
df_test

# %%
acc_info = ckmktpi.get_account_info()
acc_info

# %%
res = ckmktpi.process_email(
    df_test, "firstName", "lastName", "Company", is_company=True
)

# %%
res
