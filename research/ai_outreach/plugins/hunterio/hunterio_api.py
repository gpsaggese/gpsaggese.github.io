"""
Import as:

import ai_outreach.plugins.hunterio.hunterio_api as aophhuap
"""

import logging
import os
import re
from typing import Any, Dict, cast

import pandas as pd
import requests  # type: ignore[import-untyped]
import tqdm

import helpers.hcache_simple as hcacsimp
import helpers.hdbg as hdbg

_LOG = logging.getLogger(__name__)


# #############################################################################
# Account
# #############################################################################


def get_hunterio_account_info() -> Dict[str, Any]:
    """
    Get the account info from Hunter.io.

    :return: A dictionary with the account info.
    """
    HUNTER_API_KEY = os.environ["HUNTER_API_KEY"]
    url = f"https://api.hunter.io/v2/account?api_key={HUNTER_API_KEY}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()["data"]
    requests_ = data.get("requests", {})
    result = {
        "searches": requests_.get("searches", {}),
        "verifications": requests_.get("verifications", {}),
        "reset_date": data.get("reset_date", ""),
    }
    return result


def get_hunterio_email_stats() -> Dict[str, Any]:
    """
    Get email search and verification stats from the Hunter.io account.

    :return: Email search and verification stats.
    """
    account_info = get_hunterio_account_info()
    data = {
        "searches": account_info["searches"],
        "verifications": account_info["verifications"],
        "reset_date": account_info["reset_date"],
    }
    return data


# #############################################################################
# Domain search
# #############################################################################


@hcacsimp.simple_cache()
def hunterio_domain_search(
    domain: str,
    *,
    limit: int = 10,
    offset: int = 0,
    type_: str = "",
    seniority: str = "",
    department: str = "",
) -> Dict[str, Any]:
    """
    Search for email addresses associated with a domain.

    :param domain: The domain to search for.
    :param limit: Maximum number of results.
    :param offset: Offset for pagination.
    :param type_: Type of email (personal or generic).
    :param seniority: Seniority level filter.
    :param department: Department filter.
    :return: Search results data.
    """
    HUNTER_API_KEY = os.environ["HUNTER_API_KEY"]
    url = "https://api.hunter.io/v2/domain-search"
    params = {
        "domain": domain,
        "api_key": HUNTER_API_KEY,
        "limit": limit,
        "offset": offset,
    }
    if type_:
        params["type"] = type_
    if seniority:
        params["seniority"] = seniority
    if department:
        params["department"] = department
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = cast(Dict[str, Any], response.json())
    return data


# #############################################################################
# Email finder
# #############################################################################


@hcacsimp.simple_cache()
def hunterio_email_finder(
    domain: str,
    first_name: str,
    last_name: str,
) -> Dict[str, Any]:
    """
    Find the email address of a person given their name and company domain.

    :param domain: The company domain.
    :param first_name: The person's first name.
    :param last_name: The person's last name.
    :return: Email finder results data.
    """
    HUNTER_API_KEY = os.environ["HUNTER_API_KEY"]
    url = "https://api.hunter.io/v2/email-finder"
    params = {
        "domain": domain,
        "first_name": first_name,
        "last_name": last_name,
        "api_key": HUNTER_API_KEY,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = cast(Dict[str, Any], response.json())
    return data


def hunterio_find_emails_from_df(
    df: pd.DataFrame,
    *,
    incremental: bool = True,
    log_level: int = logging.DEBUG,
) -> pd.DataFrame:
    """
    Find emails for a DataFrame of contacts using HunterIO.

    :param df: DataFrame with columns [first_name, last_name,
        company_domain].
    :param incremental: If True, skip rows that already have emails.
    :param log_level: Logging level.
    :return: DataFrame with updated email columns.
    """
    df = df.copy()
    hdbg.dassert_is_subset(
        ["first_name", "last_name", "company_domain"], df.columns
    )
    # Initialize email columns if they don't exist.
    for col in ["email", "email_verification", "email_timestamp"]:
        if col not in df.columns:
            df[col] = ""
    # Process each row.
    for idx in tqdm.tqdm(df.index, desc="Finding emails"):
        row = df.loc[idx]
        # Skip if already has email and incremental mode.
        if incremental and row.get("email", "") not in ["", "nan"]:
            _LOG.log(
                log_level,
                "Skipping %s %s (already has email)",
                row["first_name"],
                row["last_name"],
            )
            continue
        domain = row["company_domain"]
        if not domain or domain == "nan":
            continue
        # Clean domain.
        domain = re.sub(r"^https?://", "", domain)
        domain = re.sub(r"^www\.", "", domain)
        domain = domain.split("/")[0]
    return df


# #############################################################################
# Email verifier
# #############################################################################


@hcacsimp.simple_cache()
def hunterio_verify_email(email: str) -> Dict[str, Any]:
    """
    Verify an email address using Hunter.io.

    :param email: The email to verify.
    :return: Verification results data.
    """
    HUNTER_API_KEY = os.environ["HUNTER_API_KEY"]
    url = "https://api.hunter.io/v2/email-verifier"
    params = {
        "email": email,
        "api_key": HUNTER_API_KEY,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = cast(Dict[str, Any], response.json())
    return data


def hunterio_verify_emails_from_df(
    df: pd.DataFrame,
    *,
    incremental: bool = True,
    log_level: int = logging.DEBUG,
) -> pd.DataFrame:
    """
    Verify emails for a DataFrame of contacts using HunterIO.

    :param df: DataFrame with 'email' column.
    :param incremental: If True, skip rows that already have
        verification.
    :param log_level: Logging level.
    :return: DataFrame with updated email_verification and
        email_verification_timestamp columns.
    """
    df = df.copy()
    hdbg.dassert_in("email", df.columns)
    if "email_verification" not in df.columns:
        df["email_verification"] = ""
    if "email_verification_timestamp" not in df.columns:
        df["email_verification_timestamp"] = ""
    for idx in tqdm.tqdm(df.index, desc="Verifying emails"):
        row = df.loc[idx]
        email = row["email"]
        if not email or email == "nan" or email == "":
            continue
        # Skip if already verified and incremental mode.
        if incremental and row.get("email_verification", "") not in [
            "",
            "nan",
        ]:
            _LOG.log(log_level, "Skipping %s (already verified)", email)
            continue
        data = hunterio_verify_email(email)
        if data:
            df.loc[idx, "email_verification"] = data.get("status", "")
            df.loc[idx, "email_verification_timestamp"] = str(pd.Timestamp.now())
    return df
