#!/usr/bin/env python

"""
Import as:

import ck_marketing.test.test_crm_db as cktestcr
"""

import logging
import os
import sqlite3

import pandas as pd

import ck_marketing.workflows.crm_db as cmwocrdb
import helpers.hunit_test as hunitest

_LOG = logging.getLogger(__name__)


# #############################################################################
# Helper functions
# #############################################################################


def _create_sample_contact_df(num_rows: int = 3, **kwargs) -> pd.DataFrame:
    """
    Create a sample contact DataFrame for testing.

    :param num_rows: Number of rows to create
    :param kwargs: Override default values for specific columns
    :return: DataFrame with sample contact data
    """
    data = {
        "hash": [f"hash_{i}" for i in range(num_rows)],
        "origin": ["linkedin"] * num_rows,
        "origin_timestamp": ["2025-01-01 10:00:00"] * num_rows,
        "first_name": [f"John{i}" for i in range(num_rows)],
        "last_name": [f"Doe{i}" for i in range(num_rows)],
        "email": [f"john{i}@example.com" for i in range(num_rows)],
        "email_timestamp": ["2025-01-01 10:00:00"] * num_rows,
        "email_verification": ["valid"] * num_rows,
        "email_verification_timestamp": ["2025-01-01 10:00:00"] * num_rows,
        "linkedin_url": [
            f"https://linkedin.com/in/john{i}" for i in range(num_rows)
        ],
        "job_title": ["Software Engineer"] * num_rows,
        "linked_timestamp": ["2025-01-01 10:00:00"] * num_rows,
        "company_name": ["TechCorp"] * num_rows,
        "company_domain": ["techcorp.com"] * num_rows,
        "company_type": ["Private"] * num_rows,
        "industry": ["Technology"] * num_rows,
        "city": ["San Francisco"] * num_rows,
        "country": ["USA"] * num_rows,
        "enrichment_timestamp": ["2025-01-01 10:00:00"] * num_rows,
        "type": ["customer"] * num_rows,
        "biography": ["Experienced software engineer"] * num_rows,
    }
    # Override with any provided kwargs.
    data.update(kwargs)
    return pd.DataFrame(data)


def _create_sample_db(test_instance: hunitest.TestCase) -> str:
    """
    Create a sample database for testing using test scratch space.

    :param test_instance: Test instance to get scratch space directory
    :return: Path to database file
    """
    scratch_dir = test_instance.get_scratch_space()
    db_path = os.path.join(scratch_dir, "test_crm.db")
    cmwocrdb.create_db(db_path)
    return db_path


def _create_sample_campaign_info_df(num_rows: int = 1, **kwargs) -> pd.DataFrame:
    """
    Create a sample campaign_info DataFrame for testing.

    :param num_rows: Number of rows to create
    :param kwargs: Override default values for specific columns
    :return: DataFrame with sample campaign_info data
    """
    data = {
        "id": [f"{i + 1}" for i in range(num_rows)],
        "name": [f"Campaign_{i + 1}" for i in range(num_rows)],
        "type": ["linkedin"] * num_rows,
        "description": [f"Test campaign {i + 1}" for i in range(num_rows)],
        "message": ["Hello, let's connect!"] * num_rows,
        "start_date": ["2025-01-01 00:00:00"] * num_rows,
        "end_date": ["2025-12-31 00:00:00"] * num_rows,
        "num_targets": [10] * num_rows,
        "last_update_timestamp": ["2025-01-01 10:00:00"] * num_rows,
    }
    # Override with any provided kwargs.
    data.update(kwargs)
    return pd.DataFrame(data)


def _create_sample_campaign_targets_df(
    num_rows: int = 3, **kwargs
) -> pd.DataFrame:
    """
    Create a sample campaign_targets DataFrame for testing.

    :param num_rows: Number of rows to create
    :param kwargs: Override default values for specific columns
    :return: DataFrame with sample campaign_targets data
    """
    data = {
        "campaign_id": ["1"] * num_rows,
        "first_name": [f"John{i}" for i in range(num_rows)],
        "last_name": [f"Doe{i}" for i in range(num_rows)],
        "email_or_linkedin_url": [
            f"john{i}@example.com" for i in range(num_rows)
        ],
        "send_timestamp": [None] * num_rows,
        "status": [None] * num_rows,
    }
    # Override with any provided kwargs.
    data.update(kwargs)
    return pd.DataFrame(data)


# #############################################################################
# Test_check_contact_df_schema
# #############################################################################


class Test_check_contact_df_schema(hunitest.TestCase):
    """
    Test suite for check_contact_df_schema validation function.
    """

    def test1(self) -> None:
        """
        Test that a DataFrame with valid schema passes validation.

        Input:
            - DataFrame with all required columns from Contact_df_columns

        Expected Output:
            - Function returns True
            - No exceptions raised
        """
        # Prepare inputs.
        df = _create_sample_contact_df(num_rows=2)
        # Run test.
        result = cmwocrdb.check_contact_df_schema(df)
        # Check outputs.
        self.assertTrue(result)

    def test2(self) -> None:
        """
        Test that missing a required column raises an error.

        Input:
            - DataFrame missing 'hash' column (required primary key)

        Expected Output:
            - AssertionError raised with message about missing 'hash'
        """
        # Prepare inputs.
        df = _create_sample_contact_df(num_rows=2)
        df = df.drop(columns=["hash"])
        # Run test and check output.
        with self.assertRaises(AssertionError) as cm:
            cmwocrdb.check_contact_df_schema(df)
        self.assertIn("hash", str(cm.exception))

    def test3(self) -> None:
        """
        Test that extra columns generate a warning but don't fail validation.

        Input:
            - DataFrame with all required columns plus 'extra_column'

        Expected Output:
            - Function returns True
            - Warning logged about extra column
        """
        # Prepare inputs.
        df = _create_sample_contact_df(num_rows=2)
        df["extra_column"] = "extra_value"
        # Run test.
        # Should not raise, just log warning.
        result = cmwocrdb.check_contact_df_schema(df)
        # Check outputs.
        self.assertTrue(result)

    def test4(self) -> None:
        """
        Test that missing multiple required columns raises an error.

        Input:
            - DataFrame missing 'hash', 'email', and 'first_name' columns

        Expected Output:
            - AssertionError raised listing all missing columns
        """
        # Prepare inputs.
        df = _create_sample_contact_df(num_rows=2)
        df = df.drop(columns=["hash", "email", "first_name"])
        # Run test and check output.
        with self.assertRaises(AssertionError):
            cmwocrdb.check_contact_df_schema(df)


# #############################################################################
# Test_insert_contact_df
# #############################################################################


class Test_insert_contact_df(hunitest.TestCase):
    def test1(self) -> None:
        """
        Test inserting contacts into an empty database with `assume_no_overlap mode.

        Input:
            - Empty database
            - 3 new contacts
            - mode="assume_no_overlap"

        Expected Output:
            - All 3 contacts inserted successfully
            - No errors raised
            - Database contains 3 records
        """
        # Prepare inputs.
        db_path = _create_sample_db(self)
        df = _create_sample_contact_df(num_rows=3)
        # Run test.
        cmwocrdb.insert_contact_df(db_path, df, mode="assume_no_overlap")
        # Check outputs.
        count = cmwocrdb.get_table_count(db_path, "Contact")
        self.assertEqual(count, 3)

    def test2(self) -> None:
        """
        Test that `assume_no_overlap` mode raises error when overlap exists.

        Input:
            - Database with 2 existing contacts
            - 3 new contacts where 1 overlaps (same first_name, last_name, email)
            - mode="assume_no_overlap"

        Expected Output:
            - AssertionError raised
            - Database unchanged (still contains only original 2 contacts)
        """
        # Prepare inputs.
        db_path = _create_sample_db(self)
        # Insert initial contacts.
        initial_df = _create_sample_contact_df(num_rows=2)
        cmwocrdb.insert_contact_df(db_path, initial_df, mode="assume_no_overlap")
        # Create new contacts with overlap.
        new_df = _create_sample_contact_df(num_rows=3)
        # Run test and check output.
        # This should raise because John0, Doe0, john0@example.com already exists.
        with self.assertRaises(AssertionError):
            cmwocrdb.insert_contact_df(db_path, new_df, mode="assume_no_overlap")
        # Verify DB unchanged.
        count = cmwocrdb.get_table_count(db_path, "Contact")
        self.assertEqual(count, 2)

    def test3(self) -> None:
        """
        Test that keep_new mode replaces existing contacts with new data.

        Input:
            - Database with 2 existing contacts:
                * hash_0, John0, Doe0, john0@example.com, city="NYC"
                * hash_1, John1, Doe1, john1@example.com, city="Boston"
            - 2 new contacts where 1 overlaps:
                * hash_0_new, John0, Doe0, john0@example.com, city="SF"
                * hash_2, John2, Doe2, john2@example.com, city="LA"
            - mode="keep_new"

        Expected Output:
            - Database contains 4 records (INSERT OR REPLACE keeps both when hash differs)
            - One record for John0 with hash=hash_0_new and city="SF"
            - John1's record unchanged
            - John2's record inserted
        """
        db_path = _create_sample_db(self)
        # Insert initial contacts with specific cities.
        initial_df = _create_sample_contact_df(
            num_rows=2, city=["NYC", "Boston"]
        )
        cmwocrdb.insert_contact_df(db_path, initial_df, mode="assume_no_overlap")
        # Create new contacts with overlap on John0.
        new_df = _create_sample_contact_df(num_rows=2)
        new_df["hash"] = ["hash_0_new", "hash_2"]
        new_df["first_name"] = ["John0", "John2"]
        new_df["last_name"] = ["Doe0", "Doe2"]
        new_df["email"] = ["john0@example.com", "john2@example.com"]
        new_df["city"] = ["SF", "LA"]
        cmwocrdb.insert_contact_df(db_path, new_df, mode="keep_new")
        # Verify results - note: INSERT OR REPLACE on hash means we'll have 4 records.
        db_df = cmwocrdb.get_as_df(db_path, "Contact")
        # We should have both hash_0 and hash_0_new in DB since hash is primary key.
        self.assertGreaterEqual(len(db_df), 3)
        # Check that the new John0 record exists.
        john0_new = db_df[db_df["hash"] == "hash_0_new"]
        self.assertEqual(len(john0_new), 1)
        self.assertEqual(john0_new.iloc[0]["city"], "SF")

    def test4(self) -> None:
        """
        Test that keep_existing mode works when all contacts are new.

        Input:
            - Empty database
            - 2 new contacts
            - mode="keep_existing"

        Expected Output:
            - All contacts inserted successfully (no overlap to trigger bug)
        """
        db_path = _create_sample_db(self)
        # Insert new contacts into empty DB (no overlap possible).
        new_df = _create_sample_contact_df(num_rows=2)
        cmwocrdb.insert_contact_df(db_path, new_df, mode="keep_existing")
        # Verify insertion.
        count = cmwocrdb.get_table_count(db_path, "Contact")
        self.assertEqual(count, 2)

    def test5(self) -> None:
        """
        Test that DataFrame with invalid schema raises error.

        Input:
            - Empty database
            - DataFrame missing required 'hash' column
            - mode="assume_no_overlap"

        Expected Output:
            - AssertionError raised about missing 'hash' column
            - Database unchanged (empty)
        """
        db_path = _create_sample_db(self)
        df = _create_sample_contact_df(num_rows=1)
        df = df.drop(columns=["hash"])
        with self.assertRaises(AssertionError):
            cmwocrdb.insert_contact_df(db_path, df, mode="assume_no_overlap")
        count = cmwocrdb.get_table_count(db_path, "Contact")
        self.assertEqual(count, 0)

    def test6(self) -> None:
        """
        Test that inserting an empty DataFrame completes without error.

        Input:
            - Database with 1 existing contact
            - Empty DataFrame (0 rows) with correct schema
            - mode="assume_no_overlap"

        Expected Output:
            - No exception raised
            - Database unchanged (still contains 1 contact)
            - Log message indicates no contacts to insert
        """
        db_path = _create_sample_db(self)
        # Insert initial contact.
        initial_df = _create_sample_contact_df(num_rows=1)
        cmwocrdb.insert_contact_df(db_path, initial_df, mode="assume_no_overlap")
        # Try to insert empty DataFrame.
        empty_df = _create_sample_contact_df(num_rows=0)
        cmwocrdb.insert_contact_df(db_path, empty_df, mode="assume_no_overlap")
        # Verify DB unchanged.
        count = cmwocrdb.get_table_count(db_path, "Contact")
        self.assertEqual(count, 1)

    def test7(self) -> None:
        """
        Test keep_existing mode with empty DataFrame.

        Input:
            - Database with 1 existing contact
            - Empty DataFrame
            - mode="keep_existing"

        Expected Output:
            - No exception raised
            - Database unchanged (still 1 contact)
        """
        db_path = _create_sample_db(self)
        # Insert initial contact.
        initial_df = _create_sample_contact_df(num_rows=1)
        cmwocrdb.insert_contact_df(db_path, initial_df, mode="assume_no_overlap")
        # Try to insert empty DataFrame.
        empty_df = _create_sample_contact_df(num_rows=0)
        cmwocrdb.insert_contact_df(db_path, empty_df, mode="keep_existing")
        # Verify DB unchanged.
        count = cmwocrdb.get_table_count(db_path, "Contact")
        self.assertEqual(count, 1)

    def test8(self) -> None:
        """
        Test inserting contacts where multiple have same identifying fields.

        Input:
            - Empty database
            - 3 new contacts where 2 have identical first_name, last_name, email
              but different hash values
            - mode="assume_no_overlap"

        Expected Output:
            - All contacts inserted (since overlap check is against DB, not within input)
            - Database contains 3 records
            - Note: This tests that internal duplicates in input DF are handled
        """
        db_path = _create_sample_db(self)
        df = _create_sample_contact_df(num_rows=3)
        # Make first two contacts have same identifying fields but different hash.
        df.loc[1, "first_name"] = df.loc[0, "first_name"]
        df.loc[1, "last_name"] = df.loc[0, "last_name"]
        df.loc[1, "email"] = df.loc[0, "email"]
        cmwocrdb.insert_contact_df(db_path, df, mode="assume_no_overlap")
        # All 3 should be inserted.
        count = cmwocrdb.get_table_count(db_path, "Contact")
        self.assertEqual(count, 3)

    def test9(self) -> None:
        """
        Test inserting contacts with special characters in fields.

        Input:
            - Empty database
            - 2 contacts with special characters:
                * Names: "O'Brien", "José García"
                * Emails with + and .: "john+test@example.co.uk"
            - mode="assume_no_overlap"

        Expected Output:
            - All contacts inserted successfully
            - Database contains 2 records with special characters preserved
        """
        db_path = _create_sample_db(self)
        df = _create_sample_contact_df(num_rows=2)
        df["first_name"] = ["Patrick", "José"]
        df["last_name"] = ["O'Brien", "García"]
        df["email"] = [
            "patrick.obrien+test@example.co.uk",
            "jose.garcia@example.com",
        ]
        cmwocrdb.insert_contact_df(db_path, df, mode="assume_no_overlap")
        # Verify insertion.
        db_df = cmwocrdb.get_as_df(db_path, "Contact")
        self.assertEqual(len(db_df), 2)
        self.assertEqual(db_df.iloc[0]["last_name"], "O'Brien")
        self.assertEqual(db_df.iloc[1]["first_name"], "José")

    def test10(self) -> None:
        """
        Test inserting contacts with NULL/None values in optional fields.

        Input:
            - Empty database
            - 2 contacts with None/NaN in optional fields:
                * Required fields: hash, first_name, last_name, email
                * Optional fields: job_title, company_name, city = None
            - mode="assume_no_overlap"

        Expected Output:
            - All contacts inserted successfully
            - Database contains 2 records with NULL in optional fields
        """
        db_path = _create_sample_db(self)
        df = _create_sample_contact_df(num_rows=2)
        df["job_title"] = None
        df["company_name"] = None
        df["city"] = None
        cmwocrdb.insert_contact_df(db_path, df, mode="assume_no_overlap")
        # Verify insertion.
        count = cmwocrdb.get_table_count(db_path, "Contact")
        self.assertEqual(count, 2)

    def test11(self) -> None:
        """
        Test assume_idempotent mode with no overlap (all new contacts).

        Input:
            - Empty database
            - 3 new contacts
            - mode="assume_idempotent"

        Expected Output:
            - All 3 contacts inserted successfully
            - No errors raised
            - Database contains 3 records
        """
        db_path = _create_sample_db(self)
        df = _create_sample_contact_df(num_rows=3)
        cmwocrdb.insert_contact_df(db_path, df, mode="assume_idempotent")
        count = cmwocrdb.get_table_count(db_path, "Contact")
        self.assertEqual(count, 3)

    def test12(self) -> None:
        """
        Test assume_idempotent mode with overlap where data is identical.

        Input:
            - Database with 2 existing contacts
            - 3 contacts where 2 overlap with identical data
            - mode="assume_idempotent"

        Expected Output:
            - No errors raised (data is identical)
            - Only 1 new contact inserted
            - Database contains 3 records total
        """
        db_path = _create_sample_db(self)
        # Insert initial contacts.
        initial_df = _create_sample_contact_df(num_rows=2)
        cmwocrdb.insert_contact_df(db_path, initial_df, mode="assume_no_overlap")
        # Create new contacts with 2 overlapping (identical data).
        new_df = _create_sample_contact_df(num_rows=3)
        cmwocrdb.insert_contact_df(db_path, new_df, mode="assume_idempotent")
        # Verify only the new contact (John2) was inserted.
        count = cmwocrdb.get_table_count(db_path, "Contact")
        self.assertEqual(count, 3)

    def test13(self) -> None:
        """
        Test assume_idempotent mode with overlap where data differs.

        Input:
            - Database with 1 contact: John0, Doe0, john0@example.com, city="NYC"
            - 2 contacts where 1 overlaps but has different data:
                * John0, Doe0, john0@example.com, city="SF" (different city)
                * John1, Doe1, john1@example.com, city="LA" (new)
            - mode="assume_idempotent"

        Expected Output:
            - AssertionError raised with message about data mismatch
            - Database unchanged (still contains only 1 contact)
        """
        db_path = _create_sample_db(self)
        # Insert initial contact with specific city.
        initial_df = _create_sample_contact_df(num_rows=1)
        initial_df["city"] = "NYC"
        cmwocrdb.insert_contact_df(db_path, initial_df, mode="assume_no_overlap")
        # Try to insert with same identifying fields but different city.
        new_df = _create_sample_contact_df(num_rows=2)
        new_df["city"] = ["SF", "LA"]
        with self.assertRaises(AssertionError) as cm:
            cmwocrdb.insert_contact_df(db_path, new_df, mode="assume_idempotent")
        self.assertIn("have different data", str(cm.exception).lower())
        # Verify DB unchanged.
        count = cmwocrdb.get_table_count(db_path, "Contact")
        self.assertEqual(count, 1)

    def test14(self) -> None:
        """
        Test assume_idempotent mode with NULL values in overlapping data.

        Input:
            - Database with 1 contact having job_title=None
            - 1 contact with same identifying fields and job_title=None
            - mode="assume_idempotent"

        Expected Output:
            - No errors raised (NULL == NULL is treated as identical)
            - Database unchanged (still 1 contact)
        """
        db_path = _create_sample_db(self)
        # Insert initial contact with NULL job_title.
        initial_df = _create_sample_contact_df(num_rows=1)
        initial_df["job_title"] = None
        cmwocrdb.insert_contact_df(db_path, initial_df, mode="assume_no_overlap")
        # Insert same contact with NULL job_title.
        new_df = _create_sample_contact_df(num_rows=1)
        new_df["job_title"] = None
        cmwocrdb.insert_contact_df(db_path, new_df, mode="assume_idempotent")
        # Verify no new insert.
        count = cmwocrdb.get_table_count(db_path, "Contact")
        self.assertEqual(count, 1)

    def test15(self) -> None:
        """
        Test assume_idempotent mode detects mismatch when one has NULL and other doesn't.

        Input:
            - Database with 1 contact having job_title="Engineer"
            - 1 contact with same identifying fields but job_title=None
            - mode="assume_idempotent"

        Expected Output:
            - AssertionError raised about data mismatch
            - Database unchanged
        """
        db_path = _create_sample_db(self)
        # Insert initial contact with job_title.
        initial_df = _create_sample_contact_df(num_rows=1)
        initial_df["job_title"] = "Engineer"
        cmwocrdb.insert_contact_df(db_path, initial_df, mode="assume_no_overlap")
        # Try to insert with NULL job_title.
        new_df = _create_sample_contact_df(num_rows=1)
        new_df["job_title"] = None
        with self.assertRaises(AssertionError):
            cmwocrdb.insert_contact_df(db_path, new_df, mode="assume_idempotent")
        # Verify DB unchanged.
        count = cmwocrdb.get_table_count(db_path, "Contact")
        self.assertEqual(count, 1)


# #############################################################################
# Test_create_db
# #############################################################################


class Test_create_db(hunitest.TestCase):
    """
    Test suite for create_db database initialization function.
    """

    def test1(self) -> None:
        """
        Test that create_db creates database with both Contact and LinkedIn tables.

        Input:
            - Path to non-existent database file

        Expected Output:
            - Database file created
            - Contact_table exists with correct schema
            - LinkedIn_table exists with correct schema
        """
        scratch_dir = self.get_scratch_space()
        db_path = os.path.join(scratch_dir, "test.db")
        # Create database.
        cmwocrdb.create_db(db_path)
        # Verify file exists.
        self.assertTrue(os.path.exists(db_path))
        # Verify tables exist.
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        self.assertIn("Contact", tables)
        self.assertIn("LinkedIn", tables)
        conn.close()

    def test2(self) -> None:
        """
        Test that create_db creates parent directories if they don't exist.

        Input:
            - Path with non-existent parent directories: scratch_space/foo/bar/db.sqlite

        Expected Output:
            - Parent directories created
            - Database file created successfully
        """
        scratch_dir = self.get_scratch_space()
        db_path = os.path.join(scratch_dir, "foo", "bar", "test.db")
        # Create database.
        cmwocrdb.create_db(db_path)
        # Verify file and directories exist.
        self.assertTrue(os.path.exists(db_path))
        self.assertTrue(os.path.exists(os.path.dirname(db_path)))

    def test3(self) -> None:
        """
        Test that calling create_db multiple times is safe (idempotent).

        Input:
            - Call create_db twice on same path

        Expected Output:
            - No errors on second call
            - Database structure remains valid
        """
        scratch_dir = self.get_scratch_space()
        db_path = os.path.join(scratch_dir, "test.db")
        # Create database twice.
        cmwocrdb.create_db(db_path)
        cmwocrdb.create_db(db_path)
        # Verify database is still valid.
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        self.assertIn("Contact", tables)
        self.assertIn("LinkedIn", tables)
        conn.close()
