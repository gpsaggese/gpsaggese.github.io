# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# <a name='description'></a>
# # Description
#
# This notebook examines ...

# %%
# #!sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet jupyterlab-vim)"
# #!jupyter labextension enable

# %%
# %load_ext autoreload
# %autoreload 2

import logging
import requests
import os

import pandas as pd

# /venv/lib/python3.12/site-packages/gspread_pandas/spread.py:401: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)` .replace("", np.nan)
pd.set_option("future.no_silent_downcasting", True)

import helpers.hdbg as hdbg
import helpers.henv as henv
import helpers.hprint as hprint

# hcacsimp.cache_stats_to_str()
# hcacsimp.reset_cache()

hdbg.init_logger(verbosity=logging.INFO)
_LOG = logging.getLogger(__name__)

_LOG.info("%s", henv.get_system_signature()[0])
hnotebook.config_notebook()

# %% [markdown]
# # HunterIO

# %%
import ck_marketing.plugins as ckmktpi

import importlib

importlib.reload(ckmktpi)
import pprint

# %% [markdown]
# ## Account info

# %%
# os.environ["HUNTER_API_KEY"] = ""

# %%
pprint.pprint(ckmktpi.get_hunterio_account_info())

# %% [markdown]
# ## API request

# %%
url = f"https://api.hunter.io/v2/domain-search?company=BraunHagey%20Borden%20LLP&limit=1&api_key={os.environ['HUNTER_API_KEY']}"

response = requests.get(url)
response.raise_for_status()
data = response.json()
pprint.pprint(data)

# %% [markdown]
# ## Domain search

# %%
company = "BraunHagey Borden LLP"
pprint.pprint(ckmktpi.hunterio_domain_search(company, limit=2))

# %% [markdown]
# ## Email search

# %%
first_name = "Lauren"
last_name = "Chase"
company = "BraunHagey Borden LLP"
pprint.pprint(ckmktpi.hunterio_email_finder(first_name, last_name, company))

# %%
first_name = "Lauren"
last_name = "Chase"
company = "BraunHagey Borden LLP"
email = ckmktpi.find_email_intrinsic(
    first_name,
    last_name,
    company,
    is_company=True,
)
print(email)

# %%
domain = "upenn.edu"
email = ckmktpi.find_email_intrinsic(
    first_name,
    last_name,
    domain,
    is_company=False,
)
print(email)
