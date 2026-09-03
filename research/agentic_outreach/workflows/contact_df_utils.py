"""
Import as:

import ck_marketing.workflows.contact_df_utils as cmwcdfut
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Union

import gspread_pandas
import numpy as np
import pandas as pd
from IPython.display import display

import helpers.hdbg as hdbg

import ck_marketing.workflows.data_loaders_utils as cmwdlout
import helpers.hgoogle_drive_api as hgodrapi
import helpers.hpandas as hpandas
import helpers.hprint as hprint
import helpers.hsystem as hsystem


_LOG = logging.getLogger(__name__)


# #############################################################################
# Process contact df.
# #############################################################################


def add_hash(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a hash column to the DataFrame.

    :param df: The DataFrame to add the hash column to.
    :return: The DataFrame with the hash column added.
    """
    df = df.copy()
    keys = df["first_name"] + df["last_name"] + df["linkedin_url"]
    df["hash"] = keys.apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    # Reorder columns to put hash first.
    cols = ["hash"] + [col for col in df.columns if col != "hash"]
    df = df[cols]
    # df.set_index("hash", drop=True, inplace=True)
    # df.sort_index(inplace=True)
    return df


# TODO(gp): Move to helpers somewhere.
email_regex = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def is_valid_email(email: str) -> bool:
    """
    Check if an email is valid.

    :param email: The email to check.
    :return: True if the email is valid, False otherwise.
    """
    return bool(re.match(email_regex, email))


url_regex = re.compile(
    r"^(https?|ftp):\/\/([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,6}(\:[0-9]{1,5})?(\/[^\s]*)?$"
)


def is_valid_url(url: str) -> bool:
    return bool(re.match(url_regex, url))


def is_linkedin_url(url: str) -> bool:
    return "linkedin.com" in url


def clean_up_contact_df(
    df: pd.DataFrame, *, allow_no_emails: bool = False, debug: str = ""
) -> pd.DataFrame:
    """
    Clean up the contact DataFrame by performing several operations.

    :param df: The DataFrame to clean up.
    :param allow_no_emails: Whether to allow rows with no emails.
    :param debug: The debug phase to stop at.
    :returns: The cleaned-up DataFrame.
    """
    result = {}
    # - Convert everything into strings.
    df = df.astype(str)
    # - Remove empty spaces.
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()
    # - Remove duplicated rows unless they have non-null values.
    phase = "duplicated_emails"
    num_before = df.shape[0]
    valid_mask = df.apply(
        lambda row: row["email"] == "", axis=1
    ) | ~df.duplicated(subset=["email"])
    num_after = valid_mask.sum()
    result[phase] = hprint.perc(num_before - num_after, num_before)
    if debug == phase:
        display(df[~valid_mask])
        assert 0
    df = df[valid_mask]
    if not allow_no_emails:
        hdbg.dassert_ne((df["email"] != "").sum(), 0)
    # - Ensure that there are no duplicated emails.
    df_tmp = df[lambda row: row["email"] != ""]
    hdbg.dassert(not df_tmp.duplicated(subset=["email"]).any())
    # - Remove invalid emails.
    phase = "remove_invalid_emails"
    num_before = df.shape[0]
    valid_mask = (df["email"] == "") | df["email"].apply(is_valid_email)
    num_after = valid_mask.sum()
    result[phase] = hprint.perc(num_before - num_after, num_before)
    if debug == phase:
        display(df[~valid_mask])
        assert 0
    df = df[valid_mask]
    # - Remove names with Chinese characters.
    phase = "remove_chinese_names"
    num_before = df.shape[0]

    def contains_chinese(text: str) -> bool:
        chinese_characters = re.compile(r"[\u4E00-\u9FFF]")
        return bool(chinese_characters.search(text))

    valid_mask = df["first_name"].apply(
        lambda value: not contains_chinese(value)
    )
    num_after = valid_mask.sum()
    result[phase] = hprint.perc(num_before - num_after, num_before)
    if debug == phase:
        display(df[~valid_mask])
        assert 0
    df = df[valid_mask]
    # - Remove rows with empty first name.
    phase = "remove_empty_first_names"
    num_before = df.shape[0]
    valid_mask = df["first_name"] != ""
    num_after = valid_mask.sum()
    result[phase] = hprint.perc(num_before - num_after, num_before)
    if debug == phase:
        display(df[~valid_mask])
        assert 0
    df = df[valid_mask]
    # - Move urls from the `company_domain` column.
    phase = "clean_company_names"
    num_before = df.shape[0]
    is_url = df["company_name"].apply(is_valid_url)
    result[phase] = hprint.perc(is_url.sum(), num_before)
    if debug == phase:
        display(df[is_url])
        assert 0
    srs = df["company_name"].copy()
    df["company_domain"] = np.where(is_url, srs, df["company_domain"])
    df["company_name"] = np.where(is_url, "", srs)
    # - Remove `nan` from the LinkedIn column.
    phase = "clean_linkedin_nans"
    is_nan = df["linkedin_url"].apply(lambda value: value == "nan")
    result[phase] = hprint.perc(is_nan.sum(), num_before)
    if debug == phase:
        display(df[is_nan])
        assert 0
    df["linkedin_url"] = np.where(is_nan, "", df["linkedin_url"])
    # - Move emails from the LinkedIn column to the `email` column.
    phase = "clean_linkedin_emails"
    is_email = df["linkedin_url"].apply(lambda value: is_valid_email(value))
    result[phase] = hprint.perc(is_email.sum(), num_before)
    if debug == phase:
        display(df[is_email])
        assert 0
    srs = df["linkedin_url"].copy()
    df["email"] = np.where(is_email, srs, df["email"])
    df["linkedin_url"] = np.where(is_email, "", srs)
    # - Move websites from the LinkedIn column to the `company_domain` column.
    phase = "clean_linkedin_websites"
    is_website = df["linkedin_url"].apply(
        lambda value: not is_linkedin_url(value) and is_valid_url(value)
    )
    result[phase] = hprint.perc(is_website.sum(), num_before)
    if debug == phase:
        display(df[is_website])
        assert 0
    srs = df["linkedin_url"].copy()
    df["company_domain"] = np.where(is_website, srs, df["company_domain"])
    df["linkedin_url"] = np.where(is_website, "", srs)
    # - Remove rows with empty emails.
    phase = "clean_emails"
    num_before = df.shape[0]
    invalid_mask = df["email"].str.contains("mailto:")
    result[phase] = hprint.perc(invalid_mask.sum(), num_before)
    if debug == phase:
        display(df[invalid_mask])
        assert 0
    df.loc[invalid_mask, "email"] = df.loc[invalid_mask, "email"].str.replace(
        "mailto:", ""
    )
    # Add hash.
    df = add_hash(df)
    # Sort.
    # df.sort_values(by=["first_name", "last_name"], inplace=True)
    # df.sort_values(by=["hash"], inplace=True)
    #
    result_df = pd.Series(result).to_frame()
    # result_df = result_df.T
    display(result_df)
    return df


def sanity_check_contact_df(
    df: pd.DataFrame,
    *,
    verbose: bool = False,
    debug: str = "",
) -> None:
    """
    Perform a sanity check on the contact DataFrame.

    If one of the checks fails, print the invalid rows and raise an
    exception.

    :param df: The DataFrame to check.
    :param verbose: Whether to print detailed information.
    :param debug: The debug phase to stop at.
    """
    # Remove the columns that start with `campaign_`.
    col_names = [
        col_name
        for col_name in df.columns
        if not col_name.startswith("campaign_")
    ]
    df = df[col_names]
    contact_schema = cmwdlout.get_contact_df_schema()
    if set(df.columns) != set(contact_schema):
        diff1 = set(df.columns.tolist()) - set(contact_schema)
        diff2 = set(contact_schema) - set(df.columns.tolist())
        _LOG.warning(
            "All columns must be in Contact schema: diff1=%s diff2=%s"
            % (diff1, diff2)
        )
    # 1) `linkedin_url` column has empty values or valid LinkedIn URLs.
    phase = "linkedin_url"
    valid_mask = df["linkedin_url"].apply(
        lambda value: value == "" or is_linkedin_url(value)
    )
    if debug == phase and not valid_mask.all():
        print("Invalid %s linkedin_url values" % (~valid_mask).sum())
        if verbose:
            display(df[~valid_mask])
        assert 0
    # 2) `email` column has empty values or valid emails.
    phase = "email"
    valid_mask = df["email"].apply(
        lambda value: value == "" or is_valid_email(value)
    )
    if debug == phase and not valid_mask.all():
        print("Invalid %s email values" % (~valid_mask).sum())
        if verbose:
            display(df[~valid_mask])
        assert 0
    # 3) `email_verification` column.
    phase = "email_verification"
    valid_mask = df["email_verification"].isin(
        ["valid", "accept_all", "invalid", "unknown", ""]
    )
    if debug == phase and not valid_mask.all():
        print("Invalid %s email_verification values" % (~valid_mask).sum())
        if verbose:
            display(df[~valid_mask])
        assert 0
    # Email and company domain if they are not empty, should match.
    cols = ["email", "company_domain"]
    df_tmp = df[cols].copy()
    df_tmp["email_domain"] = df_tmp["email"].str.extract(".*@(.*)")
    df_tmp["email_domain"] = df_tmp["email_domain"].str.lower()
    # Remove suffix.
    # df_tmp["email_domain"] = df_tmp["email_domain"].str.extract(r"^(?:https?:\/\/)?(?:www\.)?([^\/]+)\.[^\.]+$")
    hpandas.impute_nans(df_tmp, "email_domain", "")
    df_tmp["actual_domain"] = df_tmp["company_domain"].str.extract(
        r"^(?:https?:\/\/)?(?:www\.)?(?:[^\/]+\.)?([^\/]+\.[^\/]+)"
    )
    df_tmp["actual_domain"] = df_tmp["actual_domain"].str.lower()
    # df_tmp["actual_domain"] = df_tmp["actual_domain"].str.extract(r"^([^\/]+)\.[^\.]+$")
    hpandas.impute_nans(df_tmp, "actual_domain", "")
    df_tmp["both_specified"] = (df_tmp["email_domain"] != "") & (
        df_tmp["actual_domain"] != ""
    )
    df_tmp["is_equal"] = df_tmp["email_domain"] == df_tmp["actual_domain"]
    df_tmp["check"] = np.where(
        df_tmp["both_specified"], 2 * df_tmp["is_equal"] - 1, 0
    )
    df_tmp["check"] = df_tmp["check"].replace(
        {-1: "mismatch", 0: "nan", 1: "match"}
    )
    # TODO(gp): Lots of mismatches are like:
    # email_domain  actual_domain
    # 12    pointninecap.com    pointnine.com
    # 25    magiclab.co bumble.com
    # 85    eventures.vc    headline.com
    # 92    mobeus.co.uk    co.uk
    # 94    transi.st   efundsforschools.com
    # 113   gmail.com   alliancetechventures.com
    # 144   holtzbrinck.com holtzbrinck-digital.com
    # 170   metaprop.org    metaprop.vc
    # 185   benchmarkcapital.co.za  co.za
    # 199   nyaamerica.org  basisset.com
    # df_tmp.head()
    print((df_tmp["check"].value_counts() / df_tmp.shape[0]).to_dict())


def print_contact_df_stats(
    df: pd.DataFrame,
    *,
    debug: str = "",
    email_col: str = "email",
    email_verification_col: str = "email_verification",
) -> None:
    """
    Print statistics for a Contact df, without applying any modification.

    :param df: The DataFrame to analyze.
    :param debug: The debug phase to stop at.
    :param email_col: The name of the email column.
    :param email_verification_col: The name of the email verification
        column.
    """
    display(df.head(1))
    #
    result_df = {}
    result_df["num_rows"] = df.shape[0]
    # 1) Check if there are duplicates for sure.
    phase = "count_no_dups"
    df_no_dups = df.drop_duplicates(subset=["first_name", "last_name", "email"])
    num_rows_no_dups = df_no_dups.shape[0]
    result_df[phase] = hprint.perc(num_rows_no_dups, df.shape[0])

    if debug == phase and (num_rows_no_dups != df.shape[0]):
        duplicated = df.duplicated(subset=["first_name", "last_name", "email"])
        display(df[duplicated])
        assert 0
    df = df_no_dups
    # 2) Remove names with non-ASCII characters
    phase = "count_no_ascii"
    valid_ascii = df["first_name"].apply(lambda x: x.isascii())
    num_rows_only_ascii = valid_ascii.sum()
    result_df[phase] = hprint.perc(num_rows_only_ascii, df.shape[0])
    if debug == phase and (num_rows_only_ascii != df.shape[0]):
        display(df[valid_ascii])
        assert 0
    df = df[valid_ascii]
    # 3) Report stats about email.
    phase = "count_email"
    if email_col in df.columns:
        valid_mask = (df[email_col] != "") & (df[email_col] != "_nan_")
        num_valid = valid_mask.sum()
        result_df[phase] = hprint.perc(num_valid, df.shape[0])
        if debug == phase and (num_valid != df.shape[0]):
            display(df[~valid_mask])
            assert 0
    else:
        _LOG.warning("No 'email=%s' column", email_col)
    # 4) Report stats about email verification.
    phase = "count_email_verification"
    if email_verification_col in df.columns:
        valid_mask = [
            (not v.startswith("_")) and (v != "")
            for v in df[email_verification_col]
        ]
        num_valid = sum(valid_mask)
        result_df[phase] = hprint.perc(num_valid, df.shape[0])
        if debug == phase and (num_valid != df.shape[0]):
            display(df[~valid_mask])
            assert 0
    else:
        _LOG.warning("No '%s' column", email_verification_col)
    # 4) Check for same first / last name.
    phase = "count_name_dups"
    duplicated = df.duplicated(subset=["first_name", "last_name"])
    result_df[phase] = hprint.perc(duplicated.sum(), df.shape[0])
    # 5) Report origin.
    phase = "count_origin"
    col_name = "origin"
    valid_mask = df[col_name] != ""
    num_valid = sum(valid_mask)
    result_df[phase] = hprint.perc(num_valid, df.shape[0])
    if debug == phase and (num_valid != df.shape[0]):
        display(df[~valid_mask])
        assert 0
    #
    result_df = pd.Series(result_df).to_frame()
    display(result_df)


# TODO(gp): -> display_heatmap_contact_df
def print_contact_df_detailed_stats(
    df: pd.DataFrame, *, mode: str = "only_pct"
) -> None:
    """
    This is similar to `get_column_stats()` but it shows a heatmap of the stats
    and understands the semantics of a contact df.
    """
    stats_df = []

    def _perc(a: float, b: float) -> float:
        return hprint.perc(a, b, only_perc=True, use_float=True)

    for col_name in df.columns:
        # Compute the stats.
        type_ = str(df[col_name].dtype)
        num_empty_vals = int((df[col_name] == "").sum())
        num_tokens_vals = int((df[col_name] == "nan").sum())
        num_nan_vals = int((df[col_name] == "nan").sum())
        num_unique_vals = int(len(df[col_name].unique()))
        num_invalid_vals = int(
            (df[col_name] == "").sum() + (df[col_name] == "nan").sum()
        )
        num_valid_vals = df.shape[0] - num_invalid_vals
        # Package the results.
        stats_df_tmp = [
            col_name,
            type_,
            num_valid_vals,
            _perc(num_valid_vals, df.shape[0]),
            num_invalid_vals,
            _perc(num_invalid_vals, df.shape[0]),
            num_unique_vals,
            _perc(num_unique_vals, df.shape[0]),
            num_empty_vals,
            _perc(num_empty_vals, df.shape[0]),
            num_nan_vals,
            _perc(num_nan_vals, df.shape[0]),
        ]
        stats_df_tmp = pd.DataFrame(stats_df_tmp).T
        stats_df.append(stats_df_tmp)
    #
    stats_df = pd.concat(stats_df, axis=0)
    stats_df.columns = [
        "col_name",
        "type",
        "valid",
        "valid [pct]",
        "invalid",
        "invalid [pct]",
        "unique",
        "unique [pct]",
        "empty",
        "empty [pct]",
        "nan",
        "nan [pct]",
    ]
    stats_df.index = range(0, stats_df.shape[0])
    if mode == "only_pct":
        columns = [
            "col_name",
            "valid [pct]",
            "unique [pct]",
            "invalid [pct]",
            "empty [pct]",
            "nan [pct]",
        ]
        hdbg.dassert_is_subset(columns, stats_df.columns)
        stats_df = stats_df[columns]
        #
        stats_df.set_index("col_name", inplace=True)
        stats_df = stats_df.astype(float)
        display(hpandas.heatmap_df(stats_df))
    else:
        hpandas.heatmap_df(stats_df)


def add_info_to_result(
    result: Dict[str, Any], tag: str, a: float, b: float, info_mode: str
) -> Dict[str, Any]:
    """
    Add information to a result dictionary.

    :param result: The result dictionary to add information to.
    :param tag: The tag to add information to.
    :param a: The value to add information to.
    :param b: The value to add information to.
    :param info_mode: The mode of information to add.
    :return: The result dictionary with the information added.
    """
    if info_mode == "all":
        # 4225 / 7377 = 57.27%
        vals = [(tag, hprint.perc(a, b))]
    elif info_mode == "only_pct":
        # 57.27%
        vals = [(tag, hprint.perc(a, b, only_perc=True))]
    elif info_mode == "only_num":
        # 4225
        vals = [(tag, str(a))]
    elif info_mode == "num_pct":
        # 4225, 57.27%
        vals = [
            (tag, str(a)),
            (tag + " [%]", hprint.perc(a, b, only_perc=True)),
        ]
    elif info_mode == "num_den_pct":
        # 4225/7377, 57.27%
        vals = [
            (tag, hprint.perc(a, b, only_fraction=True)),
            (tag + " [%]", hprint.perc(a, b, only_perc=True)),
        ]
    else:
        raise ValueError("Invalid info_mode='%s'" % info_mode)
    for tag_tmp, val in vals:
        hdbg.dassert_not_in(tag_tmp, result.keys())
        result[tag_tmp] = val
    return result


# TODO(gp): Generalize and move to hpandas?
def get_column_stats(
    contact_df: pd.DataFrame,
    col_name: Union[List[str], str],
    *,
    mode: str = "print_df",
    info_mode: str = "only_pct",
) -> Any:
    """
    Compute statistics for a column or list of columns in a contact_df.

    :param contact_df: The DataFrame containing the data.
    :param col_name: The column name or list of column names to compute
        stats for.
    :param mode: The mode of output
    :returns: The result in one of different formats
    """
    hdbg.dassert_isinstance(col_name, (str, list))
    # If a list of columns was passed compute the stats for each one.
    if isinstance(col_name, list):
        dfs = []
        for col_name_ in col_name:
            df_tmp = get_column_stats(
                contact_df, col_name_, mode="df", info_mode=info_mode
            )
            dfs.append(df_tmp)
        df = pd.concat(dfs)
        if mode == "df":
            return df
        elif mode == "print_df":
            display(df)
            return None
        else:
            raise ValueError("Invalid mode='%s'" % mode)
    # Collect results.
    result = {}
    #
    vals = contact_df[col_name]
    num_vals = len(vals)
    # result["num_vals"] = num_vals
    result = add_info_to_result(result, "num_vals", num_vals, None, "only_num")
    #
    unique_vals = vals.unique()
    num_unique_vals = len(unique_vals)
    # result["num_unique_vals"] = hprint.perc(num_unique_vals, num_vals)
    result = add_info_to_result(
        result, "num_unique_vals", num_unique_vals, num_vals, info_mode
    )
    #
    num_empty_vals = sum(t == "" for t in vals)
    # result["num_empty_vals"] = hprint.perc(num_empty_vals, num_vals)
    result = add_info_to_result(
        result, "num_empty_vals", num_empty_vals, num_vals, info_mode
    )
    #
    tokens = [t for t in vals if t.startswith("_")]
    num_tokens = len(tokens)
    # result["num_tokens"] = hprint.perc(num_tokens, num_vals)
    result = add_info_to_result(
        result, "num_tokens", num_tokens, num_vals, info_mode
    )
    #
    unique_tokens = sorted(list(set(tokens)))
    result["unique_tokens"] = "%s %s" % (
        len(unique_tokens),
        " ".join(unique_tokens),
    )
    # Return values.
    if mode in ("str", "print_str"):
        txt = ["%s=%s" % (k, v) for k, v in result.items()]
        txt = "\n".join(txt)
        if mode == "str":
            value = txt
        elif mode == "print_str":
            print(txt)
            value = None
        else:
            raise ValueError("Invalid mode='%s'" % mode)
    elif mode in ("df", "print_df"):
        df = pd.Series(result).to_frame().T
        df.index = [col_name]
        if mode == "df":
            value = df
        elif mode == "print_df":
            display(df)
            value = None
        else:
            raise ValueError("Invalid mode='%s'" % mode)
    else:
        raise ValueError("Invalid mode='%s'" % mode)
    return value


# #############################################################################
# Infer category
# #############################################################################


def infer_category(
    contact_df: pd.DataFrame,
    *,
    leave_debug_cols: bool = False,
    log_level: int = logging.DEBUG,
) -> pd.DataFrame:
    contact_df = contact_df.copy()
    src_col_names = contact_df.columns
    # Select the empty category.
    main_mask = contact_df["category"] == ""
    _LOG.log(
        log_level,
        "Empty category %s",
        hprint.perc(main_mask.sum(), len(main_mask)),
    )
    #
    masks = {}
    stats_df = []

    def _append_mask(mask, tag):
        mask = pd.Series(mask, index=main_mask.index)
        masks[tag] = mask & main_mask
        # print("%s: %s" % (tag, hprint.perc(sum(mask), len(mask))))
        stats_df.append([tag, sum(mask), 100.0 * sum(mask) / len(mask)])

    keyword = "vc"
    mask = [keyword in val.lower() for val in contact_df["company_domain"]]
    _append_mask(mask, "vc_in_domain")
    #
    keyword = "vc"
    mask = [keyword in val.lower() for val in contact_df["company_name"]]
    _append_mask(mask, "vc_in_name")
    #
    keyword = "venture"
    mask = [keyword in val.lower() for val in contact_df["company_name"]]
    _append_mask(mask, "venture_in_name")
    #
    mask = contact_df["stages"] != ""
    _append_mask(mask, "stages")
    #
    keyword = "partner"
    mask = [keyword in val.lower() for val in contact_df["job_title"]]
    _append_mask(mask, "partner_in_job")
    #
    keyword = "vc"
    mask = [keyword in val.lower() for val in contact_df["job_title"]]
    _append_mask(mask, "vc_in_job")
    #
    keyword = "invest"
    mask = [keyword in val.lower() for val in contact_df["job_title"]]
    _append_mask(mask, "invest_in_job")
    #
    keyword = "venture"
    mask = [keyword in val.lower() for val in contact_df["job_title"]]
    _append_mask(mask, "venture_in_job")
    #
    keyword = "director"
    mask = [keyword in val.lower() for val in contact_df["job_title"]]
    _append_mask(mask, "director_in_job")
    #
    keyword = "scout"
    mask = [keyword in val.lower() for val in contact_df["job_title"]]
    _append_mask(mask, "scout_in_job_title")
    #
    keyword = "eir"
    mask = [keyword in val.lower() for val in contact_df["job_title"]]
    _append_mask(mask, "eir_in_job_title")
    #
    keyword = "in residence"
    mask = [keyword in val.lower() for val in contact_df["job_title"]]
    _append_mask(mask, "in_residence_in_job_title")
    #
    keyword = "capital"
    mask = [keyword in val.lower() for val in contact_df["company_name"]]
    _append_mask(mask, "capital_in_company_name")
    # Compute the stats from the masks.
    stats_df = pd.DataFrame(stats_df, columns=["tag", "num", "pct [%]"])
    stats_df.sort_values(by="num", ascending=False, inplace=True)
    stats_df.set_index("tag", inplace=True)
    display(stats_df)
    # Decorate with the mask.
    for tag, mask in masks.items():
        contact_df[tag] = False
        contact_df.loc[mask, tag] = True
    # Mark as `is_vc` for the rows that have at lease one mask.
    contact_df["is_vc"] = contact_df[masks.keys()].sum(axis=1)
    _LOG.log(log_level, contact_df[masks.keys()].sum(axis=0))
    display(hpandas.get_value_counts_stats_df(contact_df, "is_vc"))
    #
    contact_df["category"] = np.where(
        contact_df["is_vc"] > 0,
        "Venture Fund (inferred)",
        contact_df["category"],
    )
    #
    if not leave_debug_cols:
        contact_df = contact_df[src_col_names]
    #
    return contact_df


# TODO(gp): Use hgoogle_drive_api
def save_to_gsheet(
    df: pd.DataFrame,
    *,
    name: str = "display_tmp",
    use_timestamp: bool = False,
) -> None:
    """
    Save a DataFrame to a Google Sheet.

    :param df: The DataFrame to save.
    :param name: The name of the Google Sheet.
    :param use_timestamp: Whether to append a timestamp to the name.
    """
    if use_timestamp:
        timestamp = hsystem.get_timestamp()
        name += "." + timestamp
    # Save in gp/test.
    folder_id = "1HTyRpbb4tFqRxjX6yQgosmCcpVxcF6X9"
    client = gspread_pandas.Client()
    spreadsheet = client.create(name, folder_id=folder_id)
    # Connect to the newly created spreadsheet using Spread
    spread = gspread_pandas.Spread(spreadsheet.url)
    # create_sheet=True,
    # create_spread=True)
    _LOG.info("Saving to %s ..." % spread.url)
    spread_id = spread.spread.id
    # Write DataFrame to the Google Sheet (this creates the sheet if it
    # doesn't exist).
    tab_name = "Sheet1"
    spread.df_to_sheet(df, index=True, sheet=tab_name, start="A1", replace=True)
    #
    # TODO(gp): These functions are now private in hgoogle_drive_api.
    # Consider refactoring this code to use hgodrapi.to_gsheet() instead of
    # gspread_pandas directly, or request that these functions be made public.
    credentials = hgodrapi.get_credentials()
    hgodrapi._freeze_rows_in_gsheet(
        credentials,
        spread_id,
        num_rows_to_freeze=1,
        tab_name=tab_name,
    )
    #
    hgodrapi._set_row_height_in_gsheet(
        credentials,
        spread_id,
        height=20,
        tab_name=tab_name,
    )
    _LOG.info("Saved to %s", name)
