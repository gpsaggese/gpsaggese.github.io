"""
Import as:

import ck_marketing.one_off_notebooks.CsfyTask_Power_Marketers_TX_Sales_Leads as cmooncpmtsl
"""

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Power Marketers TX - Sales Leads Generation
#
# Pipeline:
# 1. Read company names from Google Sheets
# 2. Clean company names (strip legal suffixes like LLC, Inc., etc.)
# 3. Generate Sales Navigator search URLs in batches of 50 companies
# 4. Use PhantomBuster to scrape each batch (Sales Navigator Search Export)
# 5. Use Hunter.io to fetch email addresses for all scraped profiles
# 6. Write final results back to Google Sheets
#
# **Prerequisites:**
# - Set env var `LINKEDIN_SESSION_COOKIE` to your LinkedIn `li_at` cookie value
# - Google credentials at `/home/.config/gspread_pandas/google_secret.json`

# %%
import subprocess

# Install missing dependencies before any other imports.
subprocess.check_call(
    [
        "sudo",
        "/venv/bin/pip",
        "install",
        "-q",
        "google-api-python-client",
        "google-auth",
        "gspread",
        "gspread-pandas",
        "openpyxl",
    ]
)

import copy
import logging
import os
import re
import time
from typing import List

import pandas as pd
from tqdm.autonotebook import tqdm

import ck_marketing.plugins.hunterio.hunterio_api as cmphhuap
import ck_marketing.plugins.linkedin.phantombuster_api as cmplphap
import ck_marketing.plugins.linkedin.sales_navigator_query as cmplsnaqu
import helpers.hdbg as hdbg

hdbg.init_logger(verbosity=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# # Configuration

# %%
# API keys.
os.environ.setdefault("Phantom_API_KEY", "")
os.environ.setdefault("HUNTER_API_KEY", "")

# %%
# LinkedIn session cookie (li_at).
LINKEDIN_SESSION_COOKIE = os.environ.get("LINKEDIN_SESSION_COOKIE", "")

# %%
# Google Sheets URL containing the list of power marketer companies.
GSHEET_URL = (
    "https://docs.google.com/spreadsheets/d/1eFpExrbxbEhT_HgfBoTyXVEoQpSsLdHv"
    "/edit?usp=drive_link&ouid=108015864448096120019&rtpof=true&sd=true"
)

# %%
# Base Sales Navigator URL.
# The CURRENT_COMPANY filter will be replaced per batch;
# the REGION (Texas) and CURRENT_TITLE (CEO/President/VP) filters are preserved.
BASE_SN_URL = (
    "https://www.linkedin.com/sales/search/people?query=(recentSearchParam%3A"
    "(id%3A5295873122%2CdoLogHistory%3Atrue)%2Cfilters%3AList((type%3ACURRENT_COMPANY"
    "%2Cvalues%3AList((text%3AAES%2520MARKETING%2520AND%2520TRADING%2520LLC%2Cselection"
    "Type%3AINCLUDED)%2C(id%3Aurn%253Ali%253Aorganization%253A57660281%2Ctext%3AALGONQUIN"
    "%2520ENERGY%2520SERVICES%2520INC.%2CselectionType%3AINCLUDED)%2C(id%3Aurn%253Ali"
    "%253Aorganization%253A105168594%2Ctext%3AAlliance%2520Power%2520Generation%2520"
    "Company%2CselectionType%3AINCLUDED)%2C(text%3ABAYES%2520POWER%2520LLC%2CselectionType"
    "%3AINCLUDED)%2C(text%3ADC%2520Energy%2520Texas%2CselectionType%3AINCLUDED)%2C(id%3A"
    "urn%253Ali%253Aorganization%253A36634%2Ctext%3ATenaska%2520Power%2520Services%2520Co."
    "%2CselectionType%3AINCLUDED)%2C(text%3ADC%2520ENERGY%2520DAKOTA%2520LLC%2CselectionType"
    "%3AINCLUDED)%2C(text%3ACOSERV%2520ELECTRIC%2CselectionType%3AINCLUDED)%2C(text%3ADENTON"
    "%2520MUNICIPAL%2520ELECTRIC%2CselectionType%3AINCLUDED)%2C(text%3ADERIVA%2520ENERGY"
    "%2520SERVICES%2CselectionType%3AINCLUDED)%2C(text%3ADESLYS%2520LLC%2CselectionType%3A"
    "INCLUDED)%2C(text%3ADIRECT%2520ENERGY%2CselectionType%3AINCLUDED)%2C(text%3ADRW%2520"
    "ENERGY%2520TRADING%2CselectionType%3AINCLUDED)%2C(text%3ADTE%2520ENERGY%2520TRADING"
    "%2CselectionType%3AINCLUDED)%2C(text%3ADV%2520TRADING%2CselectionType%3AINCLUDED)%2C"
    "(text%3ADXT%2520COMMODITIES%2CselectionType%3AINCLUDED)%2C(text%3ADYNASTY%2520ENERGY"
    "%2520CALIFORNIA%2CselectionType%3AINCLUDED)%2C(text%3ADYNASTY%2520POWER%2CselectionType"
    "%3AINCLUDED)%2C(text%3AEAST%2520TEXAS%2520ELECTRIC%2520COOPERATIVE%2CselectionType%3A"
    "INCLUDED)%2C(text%3AEDC%2520POWER%2CselectionType%3AINCLUDED)%2C(text%3AEDF%2520TRADING"
    "%2520NORTH%2520AMERICA%2CselectionType%3AINCLUDED)%2C(text%3AEDP%2520RENEWABLES%2520"
    "NORTH%2520AMERICA%2CselectionType%3AINCLUDED)%2C(text%3AENEL%2520TRADING%2520NORTH"
    "%2520AMERICA%2CselectionType%3AINCLUDED)))%2C(type%3AREGION%2Cvalues%3AList((id%3A"
    "102748797%2Ctext%3ATexas%252C%2520United%2520States%2CselectionType%3AINCLUDED)))%2C"
    "(type%3ACURRENT_TITLE%2Cvalues%3AList((id%3A8%2Ctext%3AChief%2520Executive%2520Officer"
    "%2CselectionType%3AINCLUDED)%2C(id%3A6%2Ctext%3APresident%2CselectionType%3AINCLUDED)"
    "%2C(id%3A7%2Ctext%3AVice%2520President%2CselectionType%3AINCLUDED)))))&sessionId=Qpl"
    "MEC%2FDRyGZxrHGosa%2BzQ%3D%3D&viewAllFilters=true"
)

# Number of companies per PhantomBuster batch (LinkedIn SN limit is ~100, use 50 to be safe).
BATCH_SIZE = 50

# %% [markdown]
# # Step 1: Read Company Names from Google Sheets

# %%
import io

import googleapiclient.discovery as godisc
from google.oauth2 import service_account

# The file is an Excel file stored in Google Drive (not a native Google Sheet),
# so we use the Drive API to export it as CSV instead of gspread.
_SHEET_ID = "1eFpExrbxbEhT_HgfBoTyXVEoQpSsLdHv"
_CREDS_PATH = "/app/.google_credentials/google_secret.json"
_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
credentials = service_account.Credentials.from_service_account_file(
    _CREDS_PATH, scopes=_SCOPES  # gitleaks:allow
)
drive_service = godisc.build(
    "drive", "v3", credentials=credentials, cache_discovery=False
)
# File is a raw Excel (.xlsx) in Drive, not a native Google Sheet, so download raw bytes.
request = drive_service.files().get_media(fileId=_SHEET_ID)
xlsx_bytes = request.execute()
companies_df = pd.read_excel(io.BytesIO(xlsx_bytes))
_LOG.info("Loaded %d rows from Google Sheets", len(companies_df))

# %%
# Inspect columns to identify the company name column.
print("Available columns:", companies_df.columns.tolist())
print(companies_df.head(3))

# %% [markdown]
# # Step 2: Clean Company Names

# %%
# Legal suffixes to strip (case-insensitive word boundaries).
_LEGAL_SUFFIXES = [
    r"\bLLC\b",
    r"\bL\.L\.C\.\b",
    r"\bINC\b",
    r"\bINC\.\b",
    r"\bCO\b",
    r"\bCO\.\b",
    r"\bCORP\b",
    r"\bCORP\.\b",
    r"\bCORPORATION\b",
    r"\bLTD\b",
    r"\bLTD\.\b",
    r"\bLP\b",
    r"\bL\.P\.\b",
    r"\bPLC\b",
    r"\bGMBH\b",
    r"\bS\.A\.\b",
    r"\bBV\b",
    r"\bPTE\b",
    r"\bPTY\b",
    r"\bCRRAH\b",
]


def clean_company_name(name: str) -> str:
    """
    Remove legal suffixes from a company name.

    :param name: raw company name string.
    :return: cleaned company name.
    """
    if not isinstance(name, str):
        return ""
    name = name.strip()
    for suffix in _LEGAL_SUFFIXES:
        name = re.sub(suffix, "", name, flags=re.IGNORECASE)
    # Remove trailing commas and collapse whitespace.
    name = re.sub(r",\s*$", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


# %%
# Column name in the sheet that contains the company name.
COMPANY_COL = "NAME"

if COMPANY_COL not in companies_df.columns:
    # Try to find a reasonable match automatically.
    candidates = [
        c
        for c in companies_df.columns
        if "company" in c.lower() or "name" in c.lower()
    ]
    _LOG.warning("Column '%s' not found. Candidates: %s", COMPANY_COL, candidates)
    COMPANY_COL = candidates[0] if candidates else companies_df.columns[0]
    _LOG.info("Using column: '%s'", COMPANY_COL)

raw_names = companies_df[COMPANY_COL].dropna().unique().tolist()
_LOG.info("Unique raw company names: %d", len(raw_names))

# %%
cleaned_names = [clean_company_name(n) for n in raw_names]
cleaned_names = sorted(set(n for n in cleaned_names if n))
_LOG.info("Unique cleaned company names: %d", len(cleaned_names))
print(cleaned_names[:10])

# %% [markdown]
# # Step 3: Generate Sales Navigator URLs per Batch

# %%
# Parse the base URL to extract its filter structure (region, title).
parsed_base = cmplsnaqu.parse_query(BASE_SN_URL)
_LOG.info(
    "Base URL filter types: %s", list(parsed_base.get("filters", {}).keys())
)


# %%
def build_sn_url_for_batch(company_batch: List[str], parsed_base: dict) -> str:
    """
    Build a Sales Navigator search URL for a batch of company names.

    Replaces the CURRENT_COMPANY filter in the base URL with the given
    company names while preserving all other filters (REGION,
    CURRENT_TITLE).

    :param company_batch: list of cleaned company name strings.
    :param parsed_base: parsed query dict from
        `cmplsnaqu.parse_query()`.
    :return: full Sales Navigator search URL.
    """
    query = copy.deepcopy(parsed_base)
    query["filters"]["CURRENT_COMPANY"] = [
        {"text": name, "selectionType": "INCLUDED"} for name in company_batch
    ]
    return cmplsnaqu.generate_query(query)


# %%
# Split cleaned company names into batches.
batches = [
    cleaned_names[i : i + BATCH_SIZE]
    for i in range(0, len(cleaned_names), BATCH_SIZE)
]
_LOG.info("Total batches: %d (BATCH_SIZE=%d)", len(batches), BATCH_SIZE)

# %%
# Preview batch 1 URL.
preview_url = build_sn_url_for_batch(batches[0], parsed_base)
print(f"Batch 1 URL (first 300 chars):\n{preview_url[:300]}...")

# %% [markdown]
# # Step 4: PhantomBuster – Create, Launch, Collect

# %%
phantom = cmplphap.Phantom()

# %%
# Verify PhantomBuster connectivity and list existing agents.
existing_phantoms = phantom.get_all_phantoms()
_LOG.info("Existing PhantomBuster agents:\n%s", existing_phantoms)

# %%
all_results = []

for batch_idx, company_batch in enumerate(batches):
    _LOG.info(
        "--- Batch %d/%d (%d companies) ---",
        batch_idx + 1,
        len(batches),
        len(company_batch),
    )
    sn_url = build_sn_url_for_batch(company_batch, parsed_base)
    agent_name = f"power_mktr_tx_b{batch_idx + 1:02d}"
    # 1. Create the phantom.
    create_resp = phantom.create_sales_nav_phantom(
        agent_name=agent_name,
        sales_nav_query=sn_url,
        linkedin_session_cookie=LINKEDIN_SESSION_COOKIE,
    )
    agent_id = str(create_resp.get("id", ""))
    if not agent_id:
        _LOG.error(
            "Batch %d: failed to create phantom. Response: %s",
            batch_idx + 1,
            create_resp,
        )
        continue
    _LOG.info(
        "Batch %d: created phantom '%s' (id=%s)",
        batch_idx + 1,
        agent_name,
        agent_id,
    )
    # 2. Launch and wait for results.
    try:
        batch_df = phantom.launch_and_get_df(agent_id)
        batch_df["_batch"] = batch_idx + 1
        all_results.append(batch_df)
        _LOG.info("Batch %d: scraped %d profiles", batch_idx + 1, len(batch_df))
    except Exception as e:
        _LOG.error("Batch %d: error during scraping: %s", batch_idx + 1, e)
    # 3. Delete the phantom to keep the account clean.
    try:
        phantom.delete_phantom(agent_id)
        _LOG.info("Batch %d: deleted phantom %s", batch_idx + 1, agent_id)
    except Exception as e:
        _LOG.warning(
            "Batch %d: could not delete phantom %s: %s",
            batch_idx + 1,
            agent_id,
            e,
        )
    # Brief pause between batches to avoid rate-limiting.
    if batch_idx < len(batches) - 1:
        time.sleep(5)

# %%
PROFILES_CHECKPOINT = "/app/power_marketers_tx_profiles.csv"
if all_results:
    profiles_df = pd.concat(all_results, ignore_index=True)
    # Save immediately so PhantomBuster results are never lost.
    profiles_df.to_csv(PROFILES_CHECKPOINT, index=False)
    _LOG.info(
        "Scraped %d profiles → saved to %s", len(profiles_df), PROFILES_CHECKPOINT
    )
else:
    _LOG.warning(
        "No profiles collected. Check LINKEDIN_SESSION_COOKIE and PhantomBuster setup."
    )
    profiles_df = pd.DataFrame()

# %%
if not profiles_df.empty:
    print("Scraped columns:", profiles_df.columns.tolist())
    profiles_df.head(3)

# %% [markdown]
# # Step 5: Fetch Emails with Hunter.io

# %%
# PhantomBuster Sales Navigator Search Export columns:
#   firstName, lastName, companyName, title, profileUrl, ...
FIRST_NAME_COL = "firstName"
LAST_NAME_COL = "lastName"
COMPANY_COL_PB = "companyName"


def hunterio_find_email(first_name: str, last_name: str, company: str) -> str:
    """
    Find an email for a person at a company using Hunter.io email-finder API.

    :param first_name: person's first name.
    :param last_name: person's last name.
    :param company: company name.
    :return: found email address, or a sentinel string:
        - `_missing_info_` if name/company is empty
        - `_nan_` if hunter.io found nothing
        - `_error_` on API error
    """
    if not first_name or not last_name or not company:
        return "_missing_info_"
    try:
        data = cmphhuap.hunterio_email_finder(
            first_name=str(first_name).strip(),
            last_name=str(last_name).strip(),
            company=str(company).strip(),
        )
        email = data.get("data", {}).get("email")
        return email if email else "_nan_"
    except Exception as e:
        _LOG.warning(
            "Hunter.io error for %s %s @ %s: %s",
            first_name,
            last_name,
            company,
            e,
        )
        return "_error_"


# %%
if not profiles_df.empty:
    # Check that required columns exist.
    for col in [FIRST_NAME_COL, LAST_NAME_COL, COMPANY_COL_PB]:
        if col not in profiles_df.columns:
            _LOG.warning(
                "Column '%s' not found. Available: %s",
                col,
                profiles_df.columns.tolist(),
            )
    # Filter rows that have the minimum required fields.
    mask = (
        profiles_df[FIRST_NAME_COL].notna()
        & profiles_df[LAST_NAME_COL].notna()
        & profiles_df[COMPANY_COL_PB].notna()
        & (profiles_df[FIRST_NAME_COL] != "")
        & (profiles_df[LAST_NAME_COL] != "")
        & (profiles_df[COMPANY_COL_PB] != "")
    )
    profiles_df = profiles_df[mask].copy()
    _LOG.info("Profiles with required name/company fields: %d", len(profiles_df))

# %%
OUTPUT_PATH = "/app/power_marketers_tx_leads.csv"
if not profiles_df.empty:
    profiles_df["hunterio.email"] = ""
    for idx, row in tqdm(profiles_df.iterrows(), total=len(profiles_df)):
        email = hunterio_find_email(
            row[FIRST_NAME_COL], row[LAST_NAME_COL], row[COMPANY_COL_PB]
        )
        profiles_df.loc[idx, "hunterio.email"] = email
        # Save after every row so progress is never lost.
        profiles_df.to_csv(OUTPUT_PATH, index=False)
    _LOG.info("Hunter.io email enrichment complete.")
    # Summary stats.
    total = len(profiles_df)
    found = profiles_df["hunterio.email"].str.contains("@", na=False).sum()
    _LOG.info(
        "Email found: %d / %d (%.1f%%)",
        found,
        total,
        100 * found / total if total else 0,
    )
    profiles_df.head()

# %% [markdown]
# # Step 6: Save Results

# %%
if not profiles_df.empty:
    profiles_df.to_csv(OUTPUT_PATH, index=False)
    _LOG.info("Final results saved to: %s", OUTPUT_PATH)

# NOTE: Write-back to Google Sheets skipped — the source file is an Excel
# file in Drive (not a native Google Sheet), so gspread cannot write to it.
# The final output is the CSV at /tmp/power_marketers_tx_leads.csv.

# %%
# Final summary.
if not profiles_df.empty:
    total = len(profiles_df)
    with_email = profiles_df["hunterio.email"].str.contains("@", na=False).sum()
    missing = (profiles_df["hunterio.email"] == "_missing_info_").sum()
    nan = (profiles_df["hunterio.email"] == "_nan_").sum()
    errors = (profiles_df["hunterio.email"] == "_error_").sum()
    print(f"Total profiles:   {total}")
    print(f"Emails found:     {with_email} ({100 * with_email / total:.1f}%)")
    print(f"Missing info:     {missing}")
    print(f"Not found (_nan): {nan}")
    print(f"Errors:           {errors}")
