"""
All the functions to load data from Google Sheets into a Contact_df.

Import as:

import ck_marketing.workflows.data_loaders_utils as cmwdlout
"""

import logging
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import pprint

import helpers.hdbg as hdbg

import helpers.hprint as hprint

_LOG = logging.getLogger(__name__)


# #############################################################################
# Contact df schema.
# #############################################################################


# This is the schema of the Contact table, after the raw data has been
# normalized.
_contact_df_schema = [
    "hash",
    # What's the source of the data (e.g., Search4.FinTech_VC_in_US,
    # VCSheet_Query1, Euro-VC-LinkedIn, etc.)
    "origin",
    # When it was loaded from the source (e.g., 2023-11-09T20:34:20.691Z).
    "origin_timestamp",
    "first_name",
    "last_name",
    # Email.
    "email",
    # When this email was generated.
    "email_timestamp",
    # State of email verification (e.g., _accept_all_, ...).
    "email_verification",
    # When this email was verified last time.
    "email_verification_timestamp",
    "linkedin_url",
    # E.g., Senior Advisor.
    "job_title",
    # The last update of this profile.
    "linked_timestamp",
    "company_name",
    "company_domain",
    # Public, private.
    "company_type",
    #
    "industry",
    "city",
    "country",
    # When it was enriched last time (e.g., to update job, company).
    "enrichment_timestamp",
    # VC, CxO, product, m&a, student, strategy, hiring
    # See ck_marketing/plugins/pitchbook/position_departments.grouped.txt
    "type",
    # Bio of the person.
    "biography",
    # # TODO(gp): This should go in a different table.
    # # E.g., Seed,Convertible Note,Series A,Pre-Seed
    # "stages",
    # # E.g., Valuation / Revenue Threshold.
    # "restrictions",
    # # E.g., AI & Machine Learning
    # "industry",
    # # Angel, VC, PE, Family Office, Corporate VC, Accelerator, Incubator
    # "category",
    # "notes",
]


def get_contact_df_schema() -> List[str]:
    """
    Get the schema of the Contact table.
    """
    return _contact_df_schema.copy()


# Template for the column mapping from a gsheet to the Contact schema.
# You should copy this template and modify it to fit the specific data.
#
# lower_cols_map = {
#     "origin": "origin",
#     # When it was loaded from the source.
#     "timestamp": "origin.timestamp",
#     "first_name": "first_name",
#     "last_name": "last_name",
#     # Email.
#     "email": "email",
#     # When this email was generated.
#     "email.timestamp": "email.timestamp",
#     "email_verification": "email_verification",
#     # When this email was verified last time.
#     "email_verification.timestamp": "email_verification.timestamp",
#     "linkedin_url": "linkedin_url",
#     "job_title": "job_title",
#     "biography": "...",
#     "company_name": "company_name",
#     "company_domain": "company_domain",
#     "city": "city",
#     # When it was enriched last time (e.g., to update job, company).
#     "enrichment.timestamp",
#     # VC, customer, student, CxO, ...
#     "type",
# }


_ColumnMap = Dict[str, str]


def _resolve_None_in_cols_map(cols_map: _ColumnMap) -> _ColumnMap:
    """
    Resolve the `None` values in a column mapping, since `None` means pass-
    through.
    """
    cols_map_out = {}
    for k, v in cols_map.items():
        if v is None:
            v = k
        cols_map_out[k] = v
    _LOG.debug("%s", cols_map_out)
    hdbg.dassert_no_duplicates(cols_map_out.keys())
    hdbg.dassert_no_duplicates(cols_map_out.values())
    return cols_map_out


def _rename_columns_to_contact_schema(
    df: pd.DataFrame,
    cols_map: _ColumnMap,
) -> Tuple[pd.DataFrame, _ColumnMap]:
    """
    Rename the columns of a DataFrame to match the Contact schema.

    :param df: The DataFrame to rename the columns of.
    :param cols_map: A dictionary mapping the original column names to
        the new column names.
    :return: A tuple containing the DataFrame with the renamed columns and
        the dictionary mapping the original column names to the new column names.
    """
    df = df.copy()
    # Resolve the pass-through columns, marked with a target value of `None`.
    cols_map = _resolve_None_in_cols_map(cols_map)
    # Rename the columns, making sure that they are all available.
    hdbg.dassert_is_subset(
        cols_map.keys(),
        df.columns,
        "All columns must be in the schema",
    )
    hdbg.dassert_no_duplicates(df.columns.tolist())
    df.rename(columns=cols_map, inplace=True, errors="raise")
    hdbg.dassert_no_duplicates(df.columns.tolist())
    return (df, cols_map)


def split_first_last_name(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    """
    Split the first and last name from a column.

    :param df: The DataFrame to split the first and last name from.
    :param name_col: The column to split the first and last name from.
    :return: A DataFrame with the first and last name columns.
    """
    df = df.copy()
    df.insert(1, "first_name", "")
    df.insert(2, "last_name", "")
    for idx, v in enumerate(df[name_col]):
        data = v.split()
        if len(data) > 0:
            df.loc[idx, "first_name"] = data[0]
        if len(data) > 1:
            df.loc[idx, "last_name"] = " ".join(data[1:])
    return df


# The difference between `normalize()` and `sanity_check()` is that:
# - `normalize()` performs some transformations to the data;
# - `sanity_check()` just checks that the data is in the correct format.Q

# We distinguish tokens as:
# - "empty space": the data is missing since there was no attempt to fill it in
# - a "nan": the data was attempted to be filled in, and was not found


def normalize_contact_schema(
    df: pd.DataFrame, cols_map: _ColumnMap, *, allow_subset: bool = False
) -> pd.DataFrame:
    """
    Normalize a dataframe to use the Contact schema, based on the column reamp.

    The output has:
    - The columns in the Contact schema.
    - All tokens are marked as `_..._` to separate them from actual values
    - Empty values means that the data was not attempted

    :param df: The DataFrame to normalize.
    :param cols_map: A dictionary mapping the original column names to
        the new column names.
    :param allow_subset: Whether to allow the subset of columns that are in the
        schema.
    """
    df = df.copy()
    _LOG.debug(hprint.to_str("df.columns"))
    # 1) Convert data into Contact schema.
    if allow_subset:
        # Find the subset of columns that are in the schema.
        cols_map = {k: v for k, v in cols_map.items() if k in df.columns}
        hdbg.dassert_lte(
            1, len(cols_map.keys()), "At least one column must be in the schema"
        )
    df, cols_map = _rename_columns_to_contact_schema(df, cols_map)
    _LOG.debug(hprint.to_str("df.columns"))
    df = df[cols_map.values()]
    contact_schema = get_contact_df_schema()
    hdbg.dassert_is_subset(
        df.columns, contact_schema, "All columns must be in contact schema"
    )
    # 2) Create missing columns.
    for col in contact_schema:
        if col not in df.columns:
            df[col] = ""
    hdbg.dassert_eq(sorted(df.columns), sorted(contact_schema))
    # 3) Remove empty spaces from all columns.
    for col in df.columns:
        df[col] = df[col].str.strip()
    # 4) Use only canonical values for `email` column.
    if "email" in df.columns:
        # Create a mask where the '@' symbol is absent.
        mask = df["email"].apply(lambda x: "@" not in str(x))
        # Find unique values where '@' is missing.
        token_values = df["email"][mask].unique()
        replacement_dict = {
            email: np.nan for email in token_values if email not in ["", "nan"]
        }
        df["email"] = df["email"].replace(replacement_dict)
        corrected_token_values = df["email"][mask].unique()
        hdbg.dassert_is_subset(corrected_token_values, ["", "nan", np.nan])
    # 5) Use only canonical values for `email_verification` column.
    if "email_verification" in df.columns:
        hdbg.dassert_is_subset(
            df["email_verification"].unique(),
            ["", "valid", "accept_all", "unknown", "invalid"],
        )
        df["email_verification"] = df["email_verification"].replace(
            {
                "accept_all": "_accept_all_",
                "valid": "_valid_",
                "unknown": "_unknown_,",
                "invalid": "_invalid_",
            }
        )
    # Reorder columns.
    hdbg.dassert_no_duplicates(df.columns.to_list())
    df = df[contact_schema]
    return df


# #############################################################################


def fuzzy_column_matching(
    cols: List[str],
    *,
    log_level: int = logging.DEBUG,
    print_results: bool = False,
) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """
    Match column names to standard schema using fuzzy matching.

    :param cols: List of column names to match
    :param log_level: The log level for debug output.
    :param print_results: Whether to print the results.
    :return: Tuple of (matched_map, unmatched_keys, matched_multiple_times_keys, unmatched_cols)
    """
    hdbg.dassert_no_duplicates(cols)
    #
    _LOG.log(log_level, "cols=%s", cols)
    #
    map_ = {
        "full_name": ["name", "people"],
        "first_name": [],
        "last_name": [],
        "email": ["professionalemail1"],
        "email_verification": [
            "Verification status",
            "hunterio.email_verification",
        ],
        "personal_email": ["personalemail1"],
        "linkedin_url": [
            "profileurl",
            "LinkedInURL",
            "LinkedIn URL",
            "defaultProfileUrl",
            "linkedinProfileUrl",
            "linkedin",
        ],
        "job_title": ["primary position", "title"],
        "company_name": ["primary company", "company"],
        "company_domain": [
            "website",
            "website1",
            "companywebsite",
            "Company Domain Name",
            "Primary Company Website",
        ],
        "company_type": ["Primary Company Type"],
        "company_industry": [],
        "city": ["location"],
        "country": ["country/territory/region", "location"],
        "biography": [
            "summary",
            "linkedinheadline",
            "linkedindescription",
            "linkedinjobdescription",
        ],
        "timestamp": ["refreshedat"],
    }
    # Add the key to the list of values.
    for key, value in map_.items():
        hdbg.dassert_isinstance(value, list)
        hdbg.dassert_not_in(key, value)
        value.insert(0, key)
    # For each value in keys of column_map, add values to expanded_col_map.
    map2_ = {}
    for key, value in map_.items():
        expanded_values = []
        for v in value:
            # Add original (already lowercase from map_ initialization).
            if v not in expanded_values:
                expanded_values.append(v)
            # Add version with underscores replaced by spaces.
            v_spaces = v.replace("_", " ")
            if v_spaces not in expanded_values:
                expanded_values.append(v_spaces)
            # Add version with underscores removed.
            v_no_underscore = v.replace("_", "")
            if v_no_underscore not in expanded_values:
                expanded_values.append(v_no_underscore)
            # Add version with no spaces.
            v_no_spaces = v.replace(" ", "")
            if v_no_spaces not in expanded_values:
                expanded_values.append(v_no_spaces)
        map2_[key] = expanded_values
        hdbg.dassert_no_duplicates(map2_[key])
    _LOG.log(log_level, "map2_=\n%s", pprint.pformat(map2_))
    # Match columns.
    out_map = {}
    for key, values in map2_.items():
        _LOG.log(log_level, "Matching '%s' to '%s'", key, values)
        for val in values:
            for col in cols:
                if val.lower() == col.lower() and key not in out_map and col not in out_map.values():
                    out_map[key] = col
                    _LOG.log(log_level, "-> Matched '%s' to '%s'", key, col)
    _LOG.log(log_level, hprint.to_str("out_map"))
    # Print the keys that were not matched.
    unmatched_keys = []
    for key, values in map2_.items():
        if key not in out_map:
            _LOG.log(log_level, f"Key '{key}' was not matched")
            unmatched_keys.append(key)
    _LOG.log(log_level, hprint.to_str("unmatched_keys"))
    # Find the keys in cols that were not matched.
    unmatched_cols = []
    for col in cols:
        if col not in out_map.values():
            _LOG.log(log_level, f"Column '{col}' was not matched")
            unmatched_cols.append(col)
    _LOG.log(log_level, hprint.to_str("unmatched_cols"))
    #
    if print_results:
        _LOG.info("# cols=%s", cols)
        _LOG.info("out_map=\n\t%s", out_map)
        _LOG.info("unmatched_keys=%s", unmatched_keys)
        _LOG.info("unmatched_cols=%s", unmatched_cols)
    # Return the map.
    return out_map, unmatched_keys, unmatched_cols
