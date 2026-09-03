#!/usr/bin/env python

"""
Import as:

import ck_marketing.manage_contact_db as ckmacodb
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import sqlite3
from tqdm import tqdm
from IPython.display import display

import ck_marketing.workflows.data_loaders_utils as cmwdlout
import ck_marketing.workflows.contact_df_utils as cmwcdfut
import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hpandas as hpandas
import helpers.hsystem as hsystem


_LOG = logging.getLogger(__name__)


# Design invariants:
# - Tables in `sqlite`are created with a query to specify the schema, primary
#   key, and foreign keys.
# - Insertions are done creating dataframes which map exactly to the table schema.
# - Df are then inspected and inserted into the database.
# - Tables and dfs have a 1:1 mapping.

# - Examples of objects are "Contact", "Campaign_info", "Campaign_targets".
#
# - There is a mapping of columns from df to table schema, e.g., `_<object>_mapping`.
#  - E.g., `_contact_mapping`
#
# - Functions that check that a df has the correct schema for a table are called
#   `check_<object>_df_schema`.
#
# - Functions that get the SQL insertion statement for an object are called
#  `get_<object>_sql_schema`
#   - E.g., `get_contact_sql_schema`.
#
# - Functions to insert a df into a table are called `insert_<object>_df`
#  - E.g., `insert_contact_df`.


# #############################################################################
# Utils functions
# #############################################################################


def _check_df_schema(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    """
    Check that a df has the correct schema for a table with the given columns.

    :param df: DataFrame to validate
    :param expected_columns: Columns of a dataframe compatible with the table
        with the given columns.
    :return: True if schema is valid
    :raises: AssertionError if schema is invalid
    """
    _LOG.debug("Validating contact DataFrame schema")
    # Get actual columns from DataFrame.
    actual_columns = df.columns.tolist()
    # Check for missing required columns.
    missing_columns = list(set(expected_columns) - set(actual_columns))
    hdbg.dassert(
        not missing_columns,
        "Missing required columns in contact DataFrame: %s",
        missing_columns,
    )
    # Check for extra columns (warn but don't fail).
    extra_columns = list(set(actual_columns) - set(expected_columns))
    if extra_columns:
        _LOG.warning(
            "Extra columns in contact DataFrame (will be ignored): %s",
            extra_columns,
        )
    # Validate that 'hash' column is present (primary key).
    hdbg.dassert_in("hash", df.columns, "Primary key 'hash' is required")
    # TODO(ai_gp2): Make sure there are no repeated data.
    _LOG.debug("DataFrame schema validation passed")
    return True


def _get_sql_schema(
    table_name: str,
    mapping: Dict[str, str],
    *,
    foreign_keys: Optional[List[str]] = None,
) -> str:
    """
    Get CREATE TABLE statement for `table_name`.

    :param table_name: Name of the table to create
        - E.g., "Contact", "Campaign_info", "Campaign_targets"
    :param mapping: Mapping of column names to SQL definitions. E.g.,
        ```
        {
        "hash": "VARCHAR(255) PRIMARY KEY",
        "first_name": "VARCHAR(100)"
        }
        ```
    :param foreign_keys: List of foreign key constraints. E.g.,
        ```
        ["FOREIGN KEY (campaign_id) REFERENCES Campaign_info(id)"]
        ```
    :return: SQL CREATE TABLE statement
    """
    lines = []
    for col_name, sql_definition in mapping.items():
        lines.append("    " + col_name + " " + sql_definition)
    create_table_stmt = (
        "CREATE TABLE IF NOT EXISTS "
        + table_name
        + " (\n"
        + ",\n".join(lines)
        + (",\n" + ",\n".join(foreign_keys) if foreign_keys else "")
        + "\n)"
    )
    _LOG.debug("sql=%s", create_table_stmt)
    return create_table_stmt


def _insert_df(
    db_path: str,
    df: pd.DataFrame,
    primary_key: str,
    table_name: str,
    overlap_cols: List[str],
    mode: str,
    *,
    log_level: int = logging.DEBUG,
) -> pd.DataFrame:
    """
    Insert a DataFrame `df` into `table_name` with overlap checks.

    :param db_path: Path to SQLite database file
    :param df: Data frame with data to insert into `table_name`
    :param primary_key: Primary key column name of `df` (e.g., `hash`)
    :param table_name: Name of the table to insert `df` into
    :param overlap_cols: Columns to check for overlap
        - E.g., ["first_name", "last_name", "linkedin_url"]
    :param mode: Insertion mode
        - "assume_no_overlap": Assert that there's no overlap on `overlap_cols`
        - "assume_idempotent": Check and assume that the data is identical to
          the data in the database.
        - "keep_new": Overwrite existing entries with new data where overlap
          occurs
        - "keep_existing": Keep existing entries, don't overwrite overlapped
         rows
    :return: DataFrame of rows that were inserted or updated in the database
    """
    _LOG.debug(
        "Insert %d rows into table '%s' with mode=%s",
        len(df),
        table_name,
        mode,
    )
    hdbg.dassert_in(
        mode,
        ["assume_no_overlap", "assume_idempotent", "keep_new", "keep_existing"],
    )
    # Connect to DB.
    conn = sqlite3.connect(db_path)
    # Check that the database has the same columns as the df.
    db_df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 0", conn)
    hdbg.dassert_eq_all(
        db_df.columns.tolist(),
        df.columns.tolist(),
        "Database and df have different columns",
    )
    #
    hdbg.dassert_is_subset(
        overlap_cols, df.columns, "overlap_cols must be a subset of df.columns"
    )
    # Only fetch `overlap_cols` for the initial overlap check.
    cols = ", ".join(overlap_cols)
    db_df_overlap = pd.read_sql_query(f"SELECT {cols} FROM {table_name}", conn)
    _LOG.log(log_level, "db_df_overlap=\n%s", hpandas.head(db_df_overlap))
    # Merge new and existing by identifying columns to count overlaps.
    merged = df.merge(
        db_df_overlap,
        how="inner",
        on=overlap_cols,
        indicator=True,
    )
    _LOG.log(log_level, "merged=\n%s", hpandas.head(merged))
    num_overlap = len(merged)
    _LOG.log(logging.DEBUG, "num_overlap=%d", num_overlap)
    # Modify the database.
    if mode == "assume_no_overlap":
        hdbg.dassert_eq(
            num_overlap,
            0,
            f"Overlap detected in keys ({cols}); aborting insertion.",
        )
        insert_df = df
    elif mode == "assume_idempotent":
        # Check that overlapping rows have identical data.
        if num_overlap > 0:
            _LOG.log(
                log_level,
                "num_overlap=%d > 0, checking for identical data",
                num_overlap,
            )
            # Get all columns from database.
            db_df_all = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            # Get the overlapping rows from df using merged index.
            df_overlap = df.loc[merged.index]
            # Merge db_df_all with df_overlap on overlap_cols to get matching rows.
            db_df_overlap = db_df_all.merge(
                df_overlap[overlap_cols], on=overlap_cols, how="inner"
            )
            # Sort both dataframes by overlap_cols for consistent comparison.
            df_overlap_sorted = df_overlap.sort_values(
                by=overlap_cols
            ).reset_index(drop=True)
            db_df_overlap_sorted = db_df_overlap.sort_values(
                by=overlap_cols
            ).reset_index(drop=True)
            # Check that the df and db_df_overlap have the same data.
            if not df_overlap_sorted.equals(db_df_overlap_sorted):
                txt = (
                    "df_overlap and db_df_overlap have different data\n"
                    + "df_overlap=\n"
                    + hpandas.head(df_overlap_sorted)
                    + "\n"
                    + "db_df_overlap=\n"
                    + hpandas.head(db_df_overlap_sorted)
                )
                _LOG.error(txt)
                raise AssertionError(txt)
        # Find rows that are not in the database by checking overlap_cols.
        db_df_all = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        merged_all = df.merge(
            db_df_all[overlap_cols], on=overlap_cols, how="left", indicator=True
        )
        insert_df = df[merged_all["_merge"] == "left_only"]
        _LOG.log(log_level, "insert_df=\n%s", hpandas.head(insert_df))
    elif mode == "keep_new":
        # Upsert all rows.
        insert_df = df
    elif mode == "keep_existing":
        # Only insert rows not already present.
        tmp = df.merge(
            db_df_overlap, on=overlap_cols, how="left", indicator=True
        )
        insert_df = tmp[tmp["_merge"] == "left_only"]
        insert_df = insert_df[df.columns]
        _LOG.debug(
            "%d row(s) will be inserted; %d skipped because of overlap.",
            len(insert_df),
            num_overlap,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    _LOG.log(log_level, "insert_df=\n%s", hpandas.head(insert_df))
    # Insert one at a time and commit at the very end.
    if not insert_df.empty:
        columns = insert_df.columns.tolist()
        placeholders = ", ".join(["?"] * len(columns))
        sql = f"INSERT OR REPLACE INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        values = [tuple(row) for row in insert_df.to_numpy()]
        _LOG.debug("Executing single-row inserts for %d rows.", len(values))
        cursor = conn.cursor()
        for value in tqdm(values, desc="Inserting rows"):
            cursor.execute(sql, value)
        conn.commit()
        _LOG.debug("%d rows inserted successfully (mode=%s)", len(values), mode)
    else:
        _LOG.debug("No rows to insert (mode=%s)", mode)
    conn.close()
    # Return the DataFrame with the rows that were inserted or updated.
    return insert_df


# #############################################################################
# Contact
# #############################################################################


# A `contact` is a person who is a target of a campaign with information about
# them and their contact information.
# The fields are defined in the `_contact_df_schema` in `data_loaders.py`.


_contact_mapping = {
    "hash": "VARCHAR(255) PRIMARY KEY",
    "origin": "VARCHAR(255)",
    "origin_timestamp": "TIMESTAMP",
    "first_name": "VARCHAR(100)",
    "last_name": "VARCHAR(100)",
    "email": "VARCHAR(255)",
    "email_timestamp": "TIMESTAMP",
    "email_verification": "VARCHAR(32)",
    "email_verification_timestamp": "TIMESTAMP",
    "linkedin_url": "VARCHAR(512)",
    "job_title": "VARCHAR(255)",
    "linked_timestamp": "TIMESTAMP",
    "company_name": "VARCHAR(255)",
    "company_domain": "VARCHAR(255)",
    "company_type": "VARCHAR(255)",
    "industry": "VARCHAR(255)",
    "city": "VARCHAR(255)",
    "country": "VARCHAR(255)",
    "enrichment_timestamp": "TIMESTAMP",
    "type": "VARCHAR(64)",
    "biography": "TEXT",
}


def _get_contact_sql_schema() -> str:
    """
    Get CREATE TABLE statement for `Contact`.

    :return: SQL CREATE TABLE statement
    """
    hdbg.dassert_set_eq(
        _contact_mapping.keys(),
        cmwdlout.get_contact_df_schema(),
        "Contact mapping and contact schema have different columns",
    )
    #
    sql_schema = _get_sql_schema("Contact", _contact_mapping)
    return sql_schema


def check_contact_df_schema(contact_df: pd.DataFrame) -> bool:
    """
    Check that a DataFrame has the correct schema for `Contact`.

    :param contact_df: DataFrame to validate
    :return: True if schema is valid
    :raises: AssertionError if schema is invalid
    """
    return _check_df_schema(contact_df, cmwdlout.get_contact_df_schema())


def insert_contact_df(
    db_path: str,
    contact_df: pd.DataFrame,
    mode: str,
    *,
    log_level: int = logging.DEBUG,
) -> pd.DataFrame:
    check_contact_df_schema(contact_df)
    #
    primary_key = "hash"
    # We use the first name, last name, and linkedin url to check for overlap.
    overlap_cols = ["first_name", "last_name", "linkedin_url"]
    table_name = "Contact"
    return _insert_df(
        db_path,
        contact_df,
        primary_key,
        table_name,
        overlap_cols,
        mode,
        log_level=log_level,
    )


# TODO(ai_gp3): Generalize to other tables by extracting the logic and passing
# primary_key_columns as an argument.
def _query_contacts(
    db_path: str,
    *,
    limit: Optional[int] = None,
    where_clause: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Query contacts from Contact.

    :param db_path: Path to SQLite database file
    :param limit: Maximum number of records to return
    :param where_clause: Optional WHERE clause for filtering
    :return: List of contact records as dictionaries
    """
    _LOG.info("Querying contacts from database")
    # Connect to database.
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Build query.
    sql = "SELECT * FROM Contact"
    if where_clause:
        sql += f" WHERE {where_clause}"
    if limit:
        sql += f" LIMIT {limit}"
    # Execute query.
    _LOG.debug("Executing SQL: %s", sql)
    cursor.execute(sql)
    rows = cursor.fetchall()
    # Convert to list of dictionaries.
    results = [dict(row) for row in rows]
    conn.close()
    _LOG.info("Retrieved %d contacts", len(results))
    return results


# #############################################################################
# Campaign
# #############################################################################

# - A campaign is a group of contacts that are targeted with a message.
#   - `Campaign_info`: information about the campaign itself
#   - `Campaign_targets`: information about each target of the campaign
#     including who it is, when it was sent, and the status of the message.


# #############################################################################
# Campaign_info
# #############################################################################

# A `Campaign_info` has:
# - an id: e.g., 1
# - a name: e.g., "Connect to MA on LinkedIn"
# - a type: e.g., "LinkedIn" or "Email"
# - a description: e.g., "Reach out to M&A professionals on LinkedIn"
# - message: "Hi, I'm interested in M&A. Can you help me?"
# - start date: e.g., 2025-01-01
# - end date: e.g., 2025-01-01
# - last_update_timestamp: when the corresponding `Campaign_targets_date` data
#   was last updated

_campaign_info_mapping = {
    "id": "VARCHAR(255) PRIMARY KEY",
    "name": "VARCHAR(255)",
    "type": "VARCHAR(255)",
    "description": "TEXT",
    "message": "TEXT",
    "start_date": "TIMESTAMP",
    "end_date": "TIMESTAMP",
    "num_targets": "INTEGER",
    "last_update_timestamp": "TIMESTAMP",
}


def _get_campaign_info_sql_schema() -> str:
    """
    Get CREATE TABLE statement for Campaign_info.
    """
    sql_schema = _get_sql_schema("Campaign_info", _campaign_info_mapping)
    return sql_schema


def check_campaign_info_df_schema(campaign_info_df: pd.DataFrame) -> bool:
    """
    Check that a DataFrame has the correct schema for `Campaign_info`.

    :param campaign_info_df: DataFrame to validate
    :return: True if schema is valid
    :raises: AssertionError if schema is invalid
    """
    return _check_df_schema(campaign_info_df, _campaign_info_mapping.keys())


# #############################################################################
# Campaign_targets
# #############################################################################


# A `Campaign_targets` stores all the information about the contacts that
# were targeted with a message
# The campaign is created with:
# - campaign_id
# - first_name
# - last_name
# - email_or_linkedin_url
# - num_targets: number of targets in the campaign
#
# The campain is updated with:
# - send_timestamp: when the message was sent
# - status (if the campaign is a YAMM campaign or a PB campaign)


_campaign_target_mapping = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "campaign_id": "VARCHAR(255)",
    "first_name": "VARCHAR(255)",
    "last_name": "VARCHAR(255)",
    "email_or_linkedin_url": "VARCHAR(512)",
    "send_timestamp": "TIMESTAMP",
    "status": "VARCHAR(64)",
    # Foreign key handled separately in SQL schema generation
}


def _get_campaign_targets_sql_schema() -> str:
    """
    Get CREATE TABLE statement for Campaign_targets.

    :return: SQL CREATE TABLE statement
    """
    sql_schema = _get_sql_schema(
        "Campaign_targets",
        _campaign_target_mapping,
        foreign_keys=["FOREIGN KEY (campaign_id) REFERENCES Campaign_info(id)"],
    )
    return sql_schema


def check_campaign_targets_df(campaign_targets_df: pd.DataFrame) -> bool:
    """
    Check that a DataFrame has the correct schema for `Campaign_targets`.

    :param campaign_targets_df: DataFrame to validate
    :return: True if schema is valid
    :raises: AssertionError if schema is invalid
    """
    return _check_df_schema(
        campaign_targets_df, cmwdlout.get_campaign_targets_df_schema()
    )


def create_campaign_info_id(
    db_path: str,
    id_: Optional[int],
    name: str,
    type_: str,
    description: str,
    message: str,
    start_date: pd.Timestamp,
    num_targets: int,
    *,
    log_level: int = logging.DEBUG,
) -> pd.DataFrame:
    """
    Create a campaign info dataframe.

    :param db_path: Path to the SQLite database file.
    :param id_: The id of the campaign.
    :param name: The name of the campaign.
    :param type_: The type of the campaign.
    :param description: The description of the campaign.
    :param message: The message of the campaign.
    :param start_date: The start date of the campaign.
    :param num_targets: The number of targets of the campaign.
    :param log_level: The log level to use.
    :return: The campaign info dataframe.
    """
    hdbg.dassert_in(type_, ["linkedin", "email"], "Invalid campaign type")
    # Connect to database.
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Get the next id by querying max existing id.
        if id_ is None:
            cursor.execute("SELECT MAX(CAST(id AS INTEGER)) FROM Campaign_info")
            result = cursor.fetchone()[0]
            if result is None:
                # No campaigns exist yet, start at 1.
                id_ = 1
            else:
                id_ = result + 1
            _LOG.log(log_level, "Auto-generated id=%d", id_)
    # Create Campaign_info dataframe.
    hdbg.dassert_lte(0, id_, "Invalid campaign id")
    campaign_info_df = pd.DataFrame(
        {
            "name": [name],
            "type": [type_],
            "description": [description],
            "message": [message],
            "start_date": [start_date],
            "num_targets": [num_targets],
            "last_update_timestamp": [str(pd.Timestamp.now())],
        },
        index=pd.Index([id_], name="id"),
    )
    return campaign_info_df


# TODO(gp): Move to campaign_utils.py?
def create_campaign_tables(
    db_path: str,
    #
    id_: int,
    name: str,
    type_: str,
    description: str,
    message: str,
    start_date: pd.Timestamp,
    #
    target_df: pd.DataFrame,
    target_col: str,
    *,
    log_level: int = logging.DEBUG,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create Campaign_info and Campaign_targets tables.

    :param db_path: Path to the SQLite database file.
    :param id: The ID of the campaign.
    :param name: The name of the campaign.
    :param type_: The type of the campaign.
    :param description: The description of the campaign.
    :param message: The message of the campaign.
    :param start_date: The start date of the campaign.
    :param target_df: The DataFrame containing the targets (as emails or linkedin urls)
    :param target_col: The column name containing the target email or linkedin url.
    :param log_level: The log level to use.
    :return: The Campaign_info and Campaign_targets DataFrames.
    """
    # check_campaign_info_df_schema(campaign_info_df)
    check_campaign_targets_df(campaign_targets_df)
    _LOG.info("Creating campaign '%s' with %d targets", name, len(target_df))
    hdbg.dassert_in(type_, ["linkedin", "email"], "Invalid campaign type")
    # Connect to database.
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Get the next id by querying max existing id.
        if id_ is None:
            cursor.execute("SELECT MAX(CAST(id AS INTEGER)) FROM Campaign_info")
            result = cursor.fetchone()[0]
            if result is None:
                # No campaigns exist yet, start at 1.
                id_ = 1
            else:
                id_ = result + 1
            _LOG.log(log_level, "Auto-generated id=%d", id_)
    # Create Campaign_info dataframe.
    campaign_info_df = pd.DataFrame(
        {
            "id": id_,
            "name": name,
            "type": type_,
            "description": description,
            "message": message,
            "start_date": start_date,
            "num_targets": len(target_df),
            "last_update_timestamp": str(pd.Timestamp.now()),
        }
    )
    # Create Campaign_targets dataframe.
    campaign_targets_df = pd.DataFrame(
        {
            "campaign_id": id_,
            "first_name": target_df["first_name"],
            "last_name": target_df["last_name"],
            "email_or_linkedin_url": target_df[target_col],
            "send_timestamp": None,
            "status": None,
        }
    )
    return campaign_info_df, campaign_targets_df


# #############################################################################
# LinkedIn table
# #############################################################################


_linkedin_mapping = {
    "linkedinProfileId": "BIGINT PRIMARY KEY",
    "companyIndustry": "VARCHAR(255)",
    "companyName": "VARCHAR(255)",
    "firstName": "VARCHAR(255)",
    "lastName": "VARCHAR(255)",
    "linkedinCompanyUrl": "VARCHAR(512)",
    "linkedinCompanySlug": "VARCHAR(255)",
    "linkedinFollowersCount": "INTEGER",
    "linkedinHeadline": "TEXT",
    "linkedinIsHiringBadge": "BOOLEAN",
    "linkedinIsOpenToWorkBadge": "BOOLEAN",
    "linkedinJobDateRange": "VARCHAR(255)",
    "linkedinPreviousCompanySlug": "VARCHAR(255)",
    "linkedinPreviousJobDateRange": "VARCHAR(255)",
    "linkedinPreviousJobDescription": "TEXT",
    "linkedinPreviousJobTitle": "VARCHAR(255)",
    "linkedinPreviousSchoolDegree": "VARCHAR(255)",
    "linkedinProfileSlug": "VARCHAR(255)",
    "linkedinProfileUrl": "VARCHAR(512)",
    "linkedinProfileUrn": "VARCHAR(255)",
    "linkedinSchoolUrl": "VARCHAR(512)",
    "linkedinSchoolCompanySlug": "VARCHAR(255)",
    "linkedinSchoolDegree": "VARCHAR(255)",
    "linkedinSchoolName": "VARCHAR(255)",
    "linkedinSkillsLabel": "TEXT",
    "location": "VARCHAR(255)",
    "previousCompanyName": "VARCHAR(255)",
    "connectionDegree": "VARCHAR(10)",
    "refreshedAt": "TIMESTAMP",
    "mutualConnectionsUrl": "VARCHAR(512)",
    "connectionsUrl": "VARCHAR(512)",
    "linkedinConnectionsCount": "INTEGER",
    "profileUrl": "VARCHAR(512)",
    "linkedinDescription": "TEXT",
    "linkedinJobDescription": "TEXT",
    "linkedinPreviousJobLocation": "VARCHAR(255)",
    "linkedinPreviousSchoolUrl": "VARCHAR(512)",
    "linkedinPreviousSchoolCompanySlug": "VARCHAR(255)",
    "linkedinPreviousSchoolDescription": "TEXT",
    "linkedinPreviousSchoolName": "VARCHAR(255)",
    "linkedinSchoolDescription": "TEXT",
    "linkedinPreviousSchoolDateRange": "VARCHAR(255)",
    "linkedinSchoolDateRange": "VARCHAR(255)",
}


def _get_linkedin_sql_schema() -> str:
    """
    Get CREATE TABLE statement for LinkedIn.

    :return: SQL CREATE TABLE statement
    """
    sql_schema = _get_sql_schema("LinkedIn", _linkedin_mapping)
    return sql_schema


def check_linkedin_df_schema(linkedin_df: pd.DataFrame) -> bool:
    """
    Check that a DataFrame has the correct schema for `LinkedIn`.

    :param linkedin_df: DataFrame to validate
    :return: True if schema is valid
    :raises: AssertionError if schema is invalid
    """
    return _check_df_schema(linkedin_df, _linkedin_mapping.keys())


def insert_linkedin_df(
    db_path: str,
    linkedin_df: pd.DataFrame,
    mode: str,
    *,
    log_level: int = logging.DEBUG,
) -> pd.DataFrame:
    primary_key = "linkedinProfileId"
    overlap_cols = ["linkedinProfileId"]
    check_linkedin_df_schema(linkedin_df)
    table_name = "LinkedIn"
    return _insert_df(
        db_path,
        linkedin_df,
        primary_key,
        table_name,
        overlap_cols,
        mode,
        log_level=log_level,
    )


def _query_linkedin(
    db_path: str,
    *,
    limit: Optional[int] = None,
    where_clause: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Query LinkedIn records from LinkedIn.

    :param db_path: Path to SQLite database file
    :param limit: Maximum number of records to return
    :param where_clause: Optional WHERE clause for filtering
    :return: List of LinkedIn records as dictionaries
    """
    _LOG.info("Querying LinkedIn records from database")
    # Connect to database.
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Build query.
    sql = "SELECT * FROM LinkedIn"
    if where_clause:
        sql += f" WHERE {where_clause}"
    if limit:
        sql += f" LIMIT {limit}"
    # Execute query.
    _LOG.debug("Executing SQL: %s", sql)
    cursor.execute(sql)
    rows = cursor.fetchall()
    # Convert to list of dictionaries.
    results = [dict(row) for row in rows]
    conn.close()
    _LOG.info("Retrieved %d LinkedIn records", len(results))
    return results


# #############################################################################
# Database
# #############################################################################


def create_db(
    db_path: str,
    *,
    delete_existing_db: bool = False,
    log_level: int = logging.DEBUG,
) -> None:
    """
    Initialize a new SQLite database for CRM data.
    Creates the required tables if they do not already exist.

    :param db_path: Path to the SQLite database file.
    :param delete_existing_db: Whether to delete the existing database file.
    :param log_level: The log level to use.
    """
    _LOG.info(f"Initializing database at: {db_path}")
    if delete_existing_db:
        _LOG.warning("Creating database from scratch")
        msg = "Are you sure you want to create the database from scratch?"
        hsystem.query_yes_no(msg, abort_on_no=True)
        # Delete database file if it exists.
        if os.path.exists(db_path):
            os.remove(db_path)
    # Ensure target directory exists.
    hio.create_enclosing_dir(db_path, incremental=True)
    # Open connection.
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Create Contact table.
        _LOG.debug("Ensuring Contact exists ...")
        sql_schema = _get_contact_sql_schema()
        _LOG.log(log_level, "sql_schema\n%s", sql_schema)
        cursor.execute(sql_schema)
        # Create Campaign_info table.
        _LOG.debug("Ensuring Campaign_info exists ...")
        sql_schema = _get_campaign_info_sql_schema()
        _LOG.log(log_level, "sql_schema\n%s", sql_schema)
        cursor.execute(sql_schema)
        # Create Campaign_targets table.
        _LOG.debug("Ensuring Campaign_targets exists ...")
        sql_schema = _get_campaign_targets_sql_schema()
        _LOG.log(log_level, "sql_schema\n%s", sql_schema)
        cursor.execute(sql_schema)
        # Create LinkedIn table.
        _LOG.debug("Ensuring LinkedIn exists ...")
        sql_schema = _get_linkedin_sql_schema()
        _LOG.log(log_level, "sql_schema\n%s", sql_schema)
        cursor.execute(sql_schema)
        #
        conn.commit()
    _LOG.info(f"Database initialized at: {db_path}")


def print_table_schema(
    db_path: str,
    *,
    tables: Optional[List[str]] = None,
    print_schema: bool = False,
) -> None:
    """
    Print the schema for each table in the database.

    :param db_path: Path to SQLite database file
    :param tables: List of table names to print the schema for
    :param print_schema: Whether to print the schema for each table
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Get all table names if not provided.
        if tables is None:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
        # Print schema for each table.
        for table in tables:
            # TODO(ai_gp3): Use hprint.frame
            # TODO(ai_gp3): Use _LOG
            print(f"\n{'=' * 80}")
            print(f"Table: {table}")
            print(f"{'=' * 80}")
            if print_schema:
                # Get column information using PRAGMA.
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                print("\nColumns:")
                print(
                    f"{'Name':<30} {'Type':<15} {'NotNull':<10} {'Default':<15} {'PK':<5}"
                )
                print("-" * 80)
                for col in columns:
                    cid, name, col_type, notnull, dflt_value, pk = col
                    dflt_str = str(dflt_value) if dflt_value is not None else ""
                    print(
                        f"{name:<30} {col_type:<15} {str(bool(notnull)):<10} {dflt_str:<15} {str(bool(pk)):<5}"
                    )
            # Get row count.
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"\nRow count: {count}")


def print_table_stats(
    db_path: str, *, tables: Optional[List[str]] = None
) -> None:
    """
    Print stats for each table: list of columns and number of rows.

    :param db_path: Path to SQLite database file
    :param tables: List of table names to print the stats for
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Get all table names if not provided.
        if tables is None:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
        # Print stats for each table.
        for table in tables:
            print(f"\n{'=' * 80}")
            print(f"Table: {table}")
            print(f"{'=' * 80}")
            # Get columns.
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in cursor.fetchall()]
            print(f"Columns ({len(columns)}): {columns}")
            # Get row count.
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"Row count: {count}")


# TODO(ai_gp3): -> get_table_as_df
def get_as_df(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Get all rows from a table in the database as a DataFrame.

    :param db_path: Path to SQLite database file
    :param table_name: Name of the table to get rows from
    :return: DataFrame with all rows
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        _LOG.debug("Retrieved %d rows from table '%s'", len(df), table_name)
    return df


# TODO(ai_gp3): -> get_table_row_count()
def get_table_count(db_path: str, table_name: str) -> int:
    """
    Get the number of rows in a table in the database.

    :param db_path: Path to SQLite database file
    :param table_name: Name of the table to get the number of rows from
    :return: Number of rows in the table
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
    _LOG.info("Table '%s' has %d rows", table_name, count)
    return count


# TODO(ai_gp3): Generalize to other tables passing tables: Optional[List[str]
def _export_table_to_csv(
    db_path: str,
    table_name: str,
    # TODO(ai_gp3): -> output_dir: str
    output_name: str,
) -> None:
    """
    Export all tables to CSV files.

    :param db_path: Path to SQLite database file
    :param output_dir: Directory to save CSV files
    """
    _LOG.debug("Exporting table '%s' to CSV file '%s'", table_name, output_name)
    # Create output directory.
    hio.create_enclosing_dir(output_name, incremental=True)
    # Connect to database.
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Build query.
    sql = f"SELECT * FROM {table_name}"
    # Execute query.
    _LOG.debug("Exporting %s", table_name)
    cursor.execute(sql)
    rows = cursor.fetchall()
    if rows:
        with open(output_name, "w", encoding="utf-8") as f:
            # Write header.
            f.write(",".join(rows[0].keys()) + "\n")
            # Write rows.
            for row in rows:
                values = [str(v) if v is not None else "" for v in row]
                f.write(",".join(values) + "\n")
        _LOG.debug("Exported %d contacts to '%s'", len(rows), output_name)
    else:
        _LOG.debug("No contacts to export")
    conn.close()


def export_db_to_csv(db_path, output_dir: str) -> None:
    """
    Export both tables to CSV files.

    :param db_path: Path to SQLite database file
    :param output_dir: Directory to save CSV files
    """
    _LOG.info("Exporting tables to CSV in directory '%s' ...", output_dir)
    _export_table_to_csv(db_path, "Contact", "contact.csv")
    _export_table_to_csv(db_path, "LinkedIn", "linked_in.csv")
    _export_table_to_csv(db_path, "Campaign_info", "campaign_info.csv")
    _export_table_to_csv(db_path, "Campaign_targets", "campaign_targets.csv")
    _LOG.info("Exporting tables to CSV done")


# TODO(ai_gp3): Add an export to Gsheet


def print_contact_stats(
    db_path: str, *, num_rows: int = 1, verbose: bool = False
) -> None:
    crm_df = get_as_df(db_path, "Contact")
    print("Contact contact shape:", crm_df.shape)
    #
    print("Contact head:")
    display(crm_df.head(num_rows))
    display(crm_df["origin"].value_counts().to_frame())
    if verbose:
        cmwcdfut.print_contact_df_detailed_stats(crm_df)
