"""
Import as:

import ck_marketing.workflows.campaign_utils as cmwocaut
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

import helpers.hdbg as hdbg

# import helpers.hcache_simple as hcacsimp
import helpers.hprint as hprint

_LOG = logging.getLogger(__name__)


def select_campaign(
    contact_df: pd.DataFrame,
    campaign_col_name: str,
    type_: str,
    num_rows: int,
    *,
    seed: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select a subset of rows from the master list DataFrame for a campaign.

    :param contact_df: The master list DataFrame.
    :param campaign_col_name: The column name indicating the campaign
        status.
    :param type_: The type of selection (e.g., 'email', 'linkedin')
    :param num_rows: The number of rows to select.
    :param seed: The random seed for reproducibility.
    :return: A df containing the selected campaign
    """
    contact_df = contact_df.copy()
    if campaign_col_name not in contact_df.columns:
        contact_df[campaign_col_name] = ""
    # Filter the campaign_df.
    campaign_df = contact_df
    # 1) Remove the one already sent.
    col_name = campaign_col_name
    valid_mask = campaign_df[col_name] == ""
    print(
        "Selected %s from %s"
        % (hprint.perc(valid_mask.sum(), len(valid_mask)), col_name)
    )
    campaign_df = campaign_df[valid_mask]
    # 2) Select the type of campaign.
    if type_ == "email":
        # Select rows with email.
        col_name = "email"
        valid_mask = campaign_df[col_name] != ""
        print(
            "Selected %s from %s"
            % (hprint.perc(valid_mask.sum(), len(valid_mask)), col_name)
        )
        campaign_df = campaign_df[valid_mask]
        # Select rows with email verification.
        col_name = "email_verification"
        valid_mask = campaign_df[col_name].isin({"valid", "all_valid"})
        print(
            "Selected %s from %s"
            % (hprint.perc(valid_mask.sum(), len(valid_mask)), col_name)
        )
        campaign_df = campaign_df[valid_mask]
        #
        campaign_col_names = (
            "hash first_name last_name company_name email email_verification"
        ).split()
    elif type_ == "linkedin":
        # Select rows with LinkedIn.
        col_name = "linkedin_url"
        valid_mask = campaign_df[col_name] != ""
        print(
            "Selected %s from %s"
            % (hprint.perc(valid_mask.sum(), len(valid_mask)), col_name)
        )
        campaign_df = campaign_df[valid_mask]
        campaign_col_names = (
            "hash first_name last_name company_name linkedin_url"
        ).split()
    else:
        raise ValueError("Invalid type_='%s'" % type_)
    # 3) Pick random `num_rows` rows.
    campaign_df = campaign_df[campaign_col_names]
    campaign_df.sort_values(by=["hash"], inplace=True)
    campaign_df = campaign_df.reset_index(drop=True)
    index = campaign_df.index
    hdbg.dassert_eq(index[0], 0)
    hdbg.dassert_eq(index[-1], len(campaign_df) - 1)
    np.random.seed(seed)
    if num_rows < 0:
        num_rows = len(campaign_df)
    index = np.random.choice(index, num_rows, replace=False)
    index = sorted(index)
    campaign_df = campaign_df.iloc[index]
    # Update the master list.
    hashes = campaign_df["hash"]
    indices = contact_df["hash"].isin(hashes)
    contact_df.loc[indices, campaign_col_name] = "selected"
    return campaign_df, contact_df


def get_short_contact_df(contact_df: pd.DataFrame) -> pd.DataFrame:
    col_names = [
        "hash",
        "first_name",
        "last_name",
        "email",
        "linkedin_url",
        "company_name",
    ]
    col_names.extend(
        [
            col_name
            for col_name in contact_df.columns
            if col_name.startswith("campaign")
        ]
    )
    return contact_df[col_names]
