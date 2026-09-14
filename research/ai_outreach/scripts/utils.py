"""
Shared utilities for ai_outreach CLI scripts.

Import as:

import ai_outreach.scripts.utils as aouscuti
"""

import logging

import pandas as pd

import helpers.hdbg as hdbg

_LOG = logging.getLogger(__name__)


# #############################################################################
# File I/O.
# #############################################################################


def load_contacts_csv(filepath: str) -> pd.DataFrame:
    """
    Load a contacts CSV and fill NaN with empty string.

    :param filepath: path to CSV file
    :return: loaded contacts DataFrame
    """
    hdbg.dassert_path_exists(filepath)
    df = pd.read_csv(filepath)
    df = df.fillna("")
    df = df.astype(str)
    _LOG.info("Loaded %d contacts from %s", len(df), filepath)
    return df


def save_contacts_csv(df: pd.DataFrame, filepath: str) -> None:
    """
    Save a contacts DataFrame to CSV.

    :param df: contacts DataFrame
    :param filepath: output CSV path
    """
    df.to_csv(filepath, index=False)
    _LOG.info("Saved %d contacts to %s", len(df), filepath)


# #############################################################################
# Filtering.
# #############################################################################


def filter_mailing_list(
    df: pd.DataFrame,
    *,
    require_email: bool = True,
    require_verified: bool = False,
) -> pd.DataFrame:
    """
    Filter a contacts DataFrame to produce a mailing list.

    :param df: enriched contacts DataFrame
    :param require_email: only include rows with non-empty email
    :param require_verified: only include rows with verified email
    :return: filtered DataFrame ready for sending
    """
    out = df.copy()
    # Filter by email presence.
    if require_email:
        has_email = (out["email"] != "") & (out["email"] != "nan")
        _LOG.info(
            "Filtering by email: %d / %d have email",
            has_email.sum(),
            len(out),
        )
        out = out[has_email]
    # Filter by verification status.
    if require_verified and "email_verification" in out.columns:
        is_verified = out["email_verification"].isin(["valid", "accept_all"])
        _LOG.info(
            "Filtering by verification: %d / %d verified",
            is_verified.sum(),
            len(out),
        )
        out = out[is_verified]
    return out


# #############################################################################
# Google Sheets upload.
# #############################################################################


def update_google_sheet(
    df: pd.DataFrame,
    *,
    sheet_name: str = "outreach_results",
) -> None:
    """
    Upload a DataFrame to Google Sheets.

    :param df: DataFrame to upload
    :param sheet_name: name of the Google Sheet
    """
    import ai_outreach.workflows.contact_df_utils as aowcdfut

    aowcdfut.save_to_gsheet(df, name=sheet_name, use_timestamp=True)
    _LOG.info("Uploaded %d rows to Google Sheet: %s", len(df), sheet_name)
