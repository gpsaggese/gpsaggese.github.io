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
import logging
import os

import helpers.hdbg as hdbg
import helpers.henv as henv
import helpers.hprint as hprint
from ck_marketing.plugins.hunterio.hunterio_api import (
    GoogleSheetsHelper,
    HunterIO,
)

# %%
# Configure logger.
hdbg.init_logger(verbosity=logging.INFO)
_LOG = logging.getLogger(__name__)

# Print system signature.
_LOG.info("%s", henv.get_system_signature()[0])

# Configure the notebook style.
hnotebook.config_notebook()

# %% [markdown]
# # Clean Profiles

# %%
hunter_api_key = os.getenv("Hunter_API_KEY")

# %%
# Google Drive Setup.
google_creds_path = "service.json"
google_sheet_helper = GoogleSheetsHelper(google_creds_path)
file_id = "1UOov0ZmZoCVBIUg825k9LgT8m6p43aW91aR2WmJNDcg"

# %%
df = google_sheet_helper.read_sheet(file_id)

# %%
df.head()

# %% [markdown]
# # Verify Email

# %%
hunter_instance = HunterIO(hunter_api_key)
verified_df = hunter_instance.verify_emails(df, "Person - Email - Work")

# %%
sheet = google_sheet_helper.google_account.open_by_key(file_id)
cleaned_profiles_tab = sheet.add_worksheet(
    title="hunter_verification", rows="100", cols="20"
)
google_sheet_helper.write_results(file_id, verified_df, "hunter_verification")

# %%
verified_df.head()
