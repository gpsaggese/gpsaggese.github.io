"""
Import as:

import ck_marketing.workflows.yamm_utils as cmwoyaut
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from IPython.display import display

import helpers.hdbg as hdbg
import helpers.hgoogle_drive_api as hgodrapi
import helpers.hprint as hprint

_LOG = logging.getLogger(__name__)


def update_df_with_yamm_bounced(
    contact_df: pd.DataFrame,
    yamm_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    *,
    email_col1: str = "email",
    email_col2: str = "Email",
    yamm_col: str = "Merge status",
    debug: bool = False,
) -> pd.DataFrame:
    """
    Update a contact df with the YAMM results.

    :param contact_df: The contact DataFrame.
    :param yamm_df: The YAMM results DataFrame.
    :param timestamp: The timestamp to associate with the email verification.
    :param email_col1: The column name of the email in the contact DataFrame.
    :param email_col2: The column name of the email in the YAMM results
        DataFrame.
    :param yamm_col: The column name of the YAMM status in the YAMM results
        DataFrame.
    :param debug: Whether to print debug information.
    :return: The updated contact DataFrame.
    """
    _LOG.info(
        "Updating contact df (%s rows) with YAMM results (%s rows)",
        contact_df.shape[0],
        yamm_df.shape[0],
    )
    contact_df = contact_df.copy()
    yamm_df = yamm_df.copy()
    # Find keys present in `yamm_df` but missing from `contact_df`.
    hdbg.dassert_in(email_col1, contact_df.columns)
    hdbg.dassert_in(email_col2, yamm_df.columns)
    missing_in_df1 = set(yamm_df[email_col2].tolist()) - set(
        contact_df[email_col1].tolist()
    )
    hdbg.dassert_eq(
        len(missing_in_df1),
        0,
        (
            f"{len(missing_in_df1)} emails in `yamm_df` are not present in `contact_df`. "
            f"Examples: {list(missing_in_df1)[:5]}"
        ),
    )
    # Merge just the "Merge status" onto `contact_df`.
    out = contact_df.merge(
        yamm_df[[email_col2, yamm_col]],
        left_on=email_col1,
        right_on=email_col2,
        how="left",
    )
    hdbg.dassert_eq(out.shape[0], contact_df.shape[0])
    # Add the "yamm.bounced" column.
    out["yamm.bounced"] = out[yamm_col] == "BOUNCED"
    hdbg.dassert_eq(
        out["yamm.bounced"].sum(), (yamm_df[yamm_col] == "BOUNCED").sum()
    )
    # Update email verification status for bounced emails.
    before_num_valid = (out["email_verification"] == "_yamm_bounced_").sum()
    out["email_verification"] = np.where(
        out["yamm.bounced"],
        "_yamm_bounced_",
        out["email_verification"],
    )
    out["email_verification_timestamp"] = np.where(
        out["yamm.bounced"],
        str(timestamp),
        out["email_verification_timestamp"],
    )
    after_num_valid = (out["email_verification"] == "_yamm_bounced_").sum()
    # _LOG.info("Changed state to _yamm_bounced_: %s",
    #     hprint.perc(after_num_valid - before_num_valid, after_num_valid))
    _LOG.info(
        "Changed state to _yamm_bounced_: %s -> %s",
        before_num_valid,
        after_num_valid,
    )
    if not debug:
        out = out.drop(columns=[yamm_col, email_col2])
    return out


def update_df_with_multiple_yamm_bounced(
    contact_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    *,
    linkedin_url_col: str = "LinkedInURL",
    email_col: str = "Email",
    yamm_col: str = "Merge status",
    debug: bool = False,
) -> pd.DataFrame:
    """
    Update a contact df with the YAMM results from multiple YAMM campaigns.

    Same interface as `update_df_with_yamm_bounced()`.
    """
    urls = [
        (
            "https://docs.google.com/spreadsheets/d/1HyglraD02TJwp16wkU_yZZ6jJW51LnBdlaWsVSTNLws/edit?gid=0#gid=0",
            "Sheet1",
        ),
        (
            "https://docs.google.com/spreadsheets/d/1HyglraD02TJwp16wkU_yZZ6jJW51LnBdlaWsVSTNLws/edit?gid=0#gid=0",
            "Sheet2",
        ),
        # ("https://docs.google.com/spreadsheets/d/1HyglraD02TJwp16wkU_yZZ6jJW51LnBdlaWsVSTNLws/edit?gid=0#gid=0", "2025-12-12"),
    ]
    for url, tab_name in urls:
        yamm_df = hgodrapi.get_cached_gsheet_to_df(url, tab_name)
        # TODO(gp): Pass parameters.
        contact_df = update_df_with_yamm_bounced(contact_df, yamm_df, timestamp)
    return contact_df


# #############################################################################


# TODO(gp): -> _normalize_yamm_schema() ?
def normalize_yamm_schema(
    df: pd.DataFrame, cols_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Normalize the schema of a YAMM DataFrame.

    :param df: The DataFrame to normalize.
    :param cols_map: A mapping of column names from the original schema
        to the desired schema.
    :return: The normalized DataFrame.
    """
    # Rename columns.
    hdbg.dassert_is_subset(
        cols_map.keys(),
        df.columns,
        "All columns to rename must be present in df",
    )
    df = df[cols_map.keys()]
    df_out = df.rename(columns=cols_map, inplace=False)
    return df_out


def get_yamm_results(normalize: bool = True) -> pd.DataFrame:
    """
    Get the results from all the YAMM campaigns.

    :param normalize: Whether to normalize the schema and remove duplicates.
    :return: The result DataFrame with columns: hash, origin, campaign_name,
        first_name, email, merge_status.
    """
    data = [
        #
        (
            "Wave1-20241210-folkapp1",
            "campaign0_VC_causify",
            "https://docs.google.com/spreadsheets/d"
            "/1mwRy0yTTCnTR14npWe7xATBYLb7DV9Pt1a2p4DjloQA",
            ["YAMM-20241210", "YAMM-20241210-1", "YAMM-20241210-2"],
        ),
        #
        (
            "Wave2-20241210-folkapp1",
            "campaign0_VC_causify",
            "https://docs.google.com/spreadsheets/d"
            "/1eufg2XREYbXnCy8tygGKAkDigM0OE_fJRmnHTDxFQ8A",
            ["YAMM-2024-12-"],
        ),
        #
        (
            "campaign_1_batch1",
            "campaign1_VC_causify",
            "https://docs.google.com/spreadsheets/d"
            "/10bWbYHdzl5KvvccHI5grtquFraO29MFP3iBcwkuVj1A",
            ["Sheet1", "Sheet2"],
        ),
        #
        (
            "Campaign2_UMD_YAMM",
            "campaign2_VC_UMD",
            "https://docs.google.com/spreadsheets/d/1rpM5MeMtAwRvbV1fCngKD4"
            "-xe7Wc19ikvs7ljx9HIeA",
            ["2024-12-28", "2024-12-30", "2025-01-02"],
        ),
    ]
    #
    yamm_dfs = []
    cols = [
        "hash",
        "origin",
        "campaign_name",
        "first_name",
        "email",
        "Merge status",
    ]
    for origin, campaign_name, url, tab_names in data:
        for tab_name in tab_names:
            _LOG.debug("Reading %s -> %s", url, tab_name)
            yamm_df = hgodrapi.get_cached_gsheet_to_df(url, tab_name)
            yamm_df["origin"] = origin
            yamm_df["campaign_name"] = campaign_name
            _LOG.debug("Read %s -> %s", tab_name, yamm_df.shape[0])
            hdbg.dassert_is_subset(cols, yamm_df.columns)
            yamm_df = yamm_df[cols]
            yamm_dfs.append(yamm_df)
    # Concatenate all the DataFrames.
    yamm_dfs = pd.concat(yamm_dfs)
    _LOG.info("Read %s rows", yamm_dfs.shape[0])
    # Normalize the schema if requested.
    if normalize:
        cols_map = {
            "hash": "hash",
            "origin": "origin",
            "campaign_name": "campaign_name",
            "first_name": "first_name",
            "email": "email",
            "Merge status": "merge_status",
        }
        yamm_dfs = normalize_yamm_schema(yamm_dfs, cols_map)
        display(yamm_dfs.head())
        # Remove duplicates.
        num_before = yamm_dfs.shape[0]
        valid_mask = yamm_dfs.duplicated(
            subset=["first_name", "email", "campaign_name"]
        )
        num_after = (~valid_mask).sum()
        _LOG.info(
            "Removed %s duplicates"
            % hprint.perc(num_before - num_after, num_before)
        )
        yamm_dfs = yamm_dfs[~valid_mask]
    _LOG.info("Number of rows: %s", yamm_dfs.shape[0])
    return yamm_dfs


# Total sent    726      100%
# BOUNCED        20      2.8%     Email not delivered (e.g., email account deactivate)
# EMAIL_OPENED  266      36.6%    User opened email
# RESPONDED      24      3.4%     User responded
# EMAIL_CLICKED  31      4.4%     User clicked one link
# EMAIL_SENT    385      53.0%    Email sent but not opened
# UNSUBSCRIBED    0      0.0%     Unsubscribed
# DELIVERED     706      97.2%    Email was actually delivered
#   = total_sent - bounced


# TODO: -> get_yamm_stats
def yamm_stats(df: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate statistics for a YAMM campaign DataFrame.

    :param df: The DataFrame containing YAMM campaign data.
    :return: A dictionary with counts of different email statuses.
    """
    # Group by merge status.
    col_name = "merge_status"
    hdbg.dassert_in(col_name, df.columns)
    df_stats = df.groupby(col_name)[col_name].count()
    vals = df_stats.to_dict()
    #
    yamm_status = {
        "BOUNCED": "bounced",
        "EMAIL_OPENED": "opened",
        "RESPONDED": "responded",
        "EMAIL_CLICKED": "clicked",
        "EMAIL_SENT": "unopened",
        "UNSUBSCRIBED": "unsubscribed",
    }
    vals2 = {}
    for k, v in yamm_status.items():
        vals2[v] = int(vals.get(k, 0))
    #
    vals2["total"] = df.shape[0]
    vals2["delivered"] = vals2["total"] - vals2["bounced"]
    return vals2


def yamm_stats_to_pct(
    obj: Union[pd.DataFrame, Dict], *, name: str = ""
) -> pd.DataFrame:
    """
    Convert YAMM campaign statistics to percentages.

    :param obj: A DataFrame containing YAMM campaign data or a dictionary with
        YAMM campaign statistics.
    :param name: Optional name of the campaign.
    :return: A DataFrame with the percentage statistics of the YAMM campaign.
    """
    if isinstance(obj, pd.DataFrame):
        vals = yamm_stats(obj)
    elif isinstance(obj, dict):
        vals = obj
    else:
        raise ValueError("Invalid object type '%s'" % type(obj))
    # Convert to percentages.
    res = {}
    res["total"] = int(vals["total"])
    yamm_ordered = (
        "delivered bounced opened responded clicked unopened unsubscribed"
    ).split()
    for k in yamm_ordered:
        hdbg.dassert_lte(vals[k], vals["total"])
        res[k] = vals[k] / vals["total"]
        res[k] = float("%.1f" % (res[k] * 100))
    res = pd.Series(res)
    res = res.to_frame().T
    if name != "":
        res.index = [name]
    return res


def yamm_stats_by_campaign(yamm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate YAMM campaign statistics for each campaign.

    :param yamm_df: The DataFrame containing YAMM campaign data.
    ```
                            total   delivered bounced opened responded clicked  unopened unsubscribed
    campaign0_VC_causify    727.0   97.2      2.8     36.9   3.3       4.3     52.7      0.0
    campaign1_VC_causify    59.0    98.3      1.7     15.3   0.0       8.5     74.6      0.0
    campaign2_VC_UMD        249.0   99.2      0.8     49.0   3.6       2.4     43.8      0.4
    total                   1035.0  97.8      2.2     38.6   3.2       4.1     51.8      0.1
    ```
    :return: A DataFrame with the statistics for each campaign.
    """
    stats_dfs = []
    # Calculate statistics for each campaign.
    for campaign_name, yamm_df_tmp in yamm_df.groupby("campaign_name"):
        stats_df = yamm_stats_to_pct(yamm_df_tmp, name=campaign_name)
        stats_dfs.append(stats_df)
    # Add the total.
    stats_df = yamm_stats_to_pct(yamm_df, name="total")
    stats_dfs.append(stats_df)
    # Create the final df.
    stats_dfs = pd.concat(stats_dfs)
    return stats_dfs


# #############################################################################


def _update_contact_df_with_yamm_df(
    contact_df: pd.DataFrame,
    campaign_id: str,
    yamm_df: pd.DataFrame,
    key_col: str,
) -> pd.DataFrame:
    """
    Update the master list with the YAMM results.

    :param contact_df: The master list DataFrame.
    :param campaign_id: The ID of the campaign to update.
    :param yamm_df: The YAMM results DataFrame.
    :param key_col: The column name of the key to use for the update.
    :return: The updated master list DataFrame.
    """
    if campaign_id not in contact_df.columns:
        contact_df[campaign_id] = ""
    # Create a mapping from key to index.
    hdbg.dassert_in(key_col, contact_df.columns)
    key_to_idx = {row[key_col]: idx for idx, row in contact_df.iterrows()}
    # For each row in the YAMM results, update the master list.
    num_updates = 0
    for _, row in yamm_df.iterrows():
        key = row[key_col]
        if key not in key_to_idx:
            _LOG.warning("Can't find key '%s' in the master list", key)
            continue
        merge_status = row["merge_status"]
        idx = key_to_idx[key]
        _LOG.debug("Updating %s %s", idx, merge_status)
        contact_df.loc[idx, campaign_id] = merge_status
        #
        num_updates += 1
    _LOG.info("num_updates=%s", hprint.perc(num_updates, yamm_df.shape[0]))
    return contact_df


def update_contact_df_with_yamm_df(
    contact_df: pd.DataFrame,
    yamm_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Update the master list with the YAMM results.

    :param contact_df: The master list DataFrame.
    :param yamm_df: The YAMM results DataFrame.
    :return: The updated contact DataFrame.
    """
    contact_df = contact_df.copy()
    # Update for each campaign.
    for campaign_id, yamm_df_tmp in yamm_df.groupby("campaign_name"):
        _LOG.info("Updating for campaign_id='%s'", campaign_id)
        contact_df = _update_contact_df_with_yamm_df(
            contact_df, campaign_id, yamm_df_tmp, "hash"
        )
    return contact_df


# #############################################################################


def merge_yamm_results_into_contact_df(
    contact_df: pd.DataFrame,
    df_yamm: pd.DataFrame,
    *,
    email_col_contact: Optional[str] = None,
    email_col_yamm: Optional[str] = None,
    merge_status_col_yamm: Optional[str] = None,
    assert_or_warn: str = "assert",
    print_results: bool = True,
) -> pd.DataFrame:
    """
    Merge YAMM results, updating the `email_verification` column.

    If a value in df_yamm[merge_status_col_yamm] is 'BOUNCED', set
    'email_verification' to '_bounced_' for that email.

    :param contact_df: The DataFrame to merge the YAMM results into.
    :param df_yamm: The YAMM results DataFrame.
    :param email_col_contact: The column name of the email in the contact
        DataFrame.
    :param email_col_yamm: The column name of the email in the YAMM results
        DataFrame.
    :param merge_status_col_yamm: The column name of the YAMM status in the
        YAMM results DataFrame.
    :return: The DataFrame containing the merged YAMM results.
    """
    # Handle default column names.
    if email_col_contact is None:
        email_col_contact = "email"
    if email_col_yamm is None:
        email_col_yamm = "Email"
    if merge_status_col_yamm is None:
        # Look for the merge status column in the YAMM results DataFrame.
        merge_status_col_yamms = ["Merge status", "merge_status", "Mergestatus"]
        merge_status_col_yamm = "Invalid %s" % str(merge_status_col_yamms)
        for merge_status_col_yamm_tmp in merge_status_col_yamms:
            if merge_status_col_yamm_tmp in df_yamm.columns:
                merge_status_col_yamm = merge_status_col_yamm_tmp
                break
    hdbg.dassert_in(email_col_contact, contact_df.columns)
    if "email_verification" not in contact_df.columns:
        contact_df["email_verification"] = ""
    if assert_or_warn == "assert":
        hdbg.dassert_in(email_col_yamm, df_yamm.columns)
        hdbg.dassert_in(merge_status_col_yamm, df_yamm.columns)
    else:
        if email_col_yamm not in df_yamm.columns:
            _LOG.warning(
                "No email_col_yamm='%s' column in df_yamm.columns=%s",
                email_col_yamm,
                df_yamm.columns,
            )
            return contact_df
        if merge_status_col_yamm not in df_yamm.columns:
            _LOG.warning(
                "No merge_status_col_yamm='%s' column in df_yamm.columns=%s",
                merge_status_col_yamm,
                df_yamm.columns,
            )
            return contact_df
    # Build a mapping from email to new email_verification value.
    # Only assign "_bounced_" if status is BOUNCED, otherwise leave as is.
    status_map = df_yamm.set_index(email_col_yamm)[merge_status_col_yamm]
    # Create a mapping from Email to either "_bounced_" (if BOUNCED) or leave as is (None).
    email_verification_map = status_map.apply(
        lambda x: "_bounced_" if str(x).strip().upper() == "BOUNCED" else None
    )
    # Compute how many bounces.
    num_bounced = sum(email_verification_map == "_bounced_")
    # Assign: where map has value "_bounced_", assign it, otherwise retain the old value.
    # Compute which were changed.
    orig_ev = contact_df["email_verification"].copy()
    new_ev = (
        contact_df[email_col_contact]
        .map(email_verification_map)
        .combine_first(contact_df["email_verification"])
    )
    num_changed = (orig_ev != new_ev).sum()
    contact_df["email_verification"] = new_ev
    #
    if print_results:
        txt = "Found %s bounced emails in df_yamm\n" % hprint.perc(
            num_bounced, df_yamm.shape[0]
        )
        txt += (
            "Changed email_verification for %s contacts in contact_df"
            % hprint.perc(num_changed, contact_df.shape[0])
        )
        _LOG.info(txt)
    return contact_df


def merge_yamm_results_from_gsheet(
    df: pd.DataFrame,
    url: str,
    merge_status_tabs: List[str],
    **kwargs: Dict[str, Any],
) -> pd.DataFrame:
    """
    Load data and merge YAMM results.

    :param url: The URL of the Google Sheet.
    :param tabs: The tabs to load.
    :return: The DataFrame containing all YAMM results.
    """
    credentials = hgodrapi.get_credentials()
    for tab_name in merge_status_tabs:
        df_tmp = hgodrapi.from_gsheet(
            url, credentials=credentials, tab_name=tab_name
        )
        df = merge_yamm_results_into_contact_df(df, df_tmp, **kwargs)
    return df


# #############################################################################


def print_stats(contact_df: pd.DataFrame) -> None:
    """
    Print statistics about the contact DataFrame.

    :param contact_df: The contact DataFrame.
    """
    num_emails = (contact_df["email"] != "").sum()
    _LOG.info(
        "Number of emails: %s", hprint.perc(num_emails, contact_df.shape[0])
    )
    if "linkedin_url" in contact_df.columns:
        num_linkedins = (contact_df["linkedin_url"] != "").sum()
        _LOG.info(
            "Number of linkedins: %s",
            hprint.perc(num_linkedins, contact_df.shape[0]),
        )
    else:
        _LOG.warning(
            "No 'linkedin_url' column in contact_df.columns=%s",
            contact_df.columns,
        )
    _LOG.info("Stats:\n%s", contact_df["email_verification"].value_counts())
