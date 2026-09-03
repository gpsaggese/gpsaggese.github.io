import logging
from typing import List, Dict

import pandas as pd

import ck_marketing.workflows.data_loaders_utils as cmwdlout
import helpers.hunit_test as hunitest


_LOG = logging.getLogger(__name__)

# #############################################################################
# Test_normalize_contact_schema
# #############################################################################


class Test_normalize_contact_schema(hunitest.TestCase):
    def test1(self) -> None:
        """
        Test column renaming.
        """
        # Input data.
        df = pd.DataFrame(
            {"First Name": ["Alice"], "Email Address": ["alice@example.com"]}
        )
        cols_map = {"First Name": "first_name", "Email Address": "email"}
        # Call function to test.
        actual = cmwdlout.normalize_contact_schema(df, cols_map)
        actual = actual.to_csv(index=False)
        expected = r"""
        hash,origin,origin_timestamp,first_name,last_name,email,email_timestamp,email_verification,email_verification_timestamp,linkedin_url,job_title,linked_timestamp,company_name,company_domain,company_type,industry,city,country,enrichment_timestamp,type,biography
        ,,,Alice,,alice@example.com,,,,,,,,,,,,,,,
        """
        # Check output.
        self.assert_equal(expected, actual, fuzzy_match=True)

    def test2(self) -> None:
        """
        Test handling of whitespace in column values.
        """
        # Input data.
        df = pd.DataFrame(
            {"first_name": ["    John "], "email": [" john@example.com "]}
        )
        cols_map = {"first_name": "first_name", "email": "email"}
        # Call function to test.
        actual = cmwdlout.normalize_contact_schema(df, cols_map)
        actual = actual.to_csv(index=False)
        expected = r"""
        hash,origin,origin_timestamp,first_name,last_name,email,email_timestamp,email_verification,email_verification_timestamp,linkedin_url,job_title,linked_timestamp,company_name,company_domain,company_type,industry,city,country,enrichment_timestamp,type,biography
        ,,,John,,john@example.com,,,,,,,,,,,,,,,
        """
        # Check output.
        self.assert_equal(actual, expected, fuzzy_match=True)

    def test3(self) -> None:
        """
        Test that the email_verification column uses only canonical values.
        """
        # Input data.
        df = pd.DataFrame(
            {
                "first_name": ["    John "],
                "email": [" john@example.com "],
                "email_verification": ["valid"],
            }
        )
        cols_map = {
            "first_name": "first_name",
            "email": "email",
            "email_verification": "email_verification",
        }
        # Call the function to test.
        actual = cmwdlout.normalize_contact_schema(df, cols_map)
        actual = actual.to_csv(index=False)
        expected = r"""
        hash,origin,origin_timestamp,first_name,last_name,email,email_timestamp,email_verification,email_verification_timestamp,linkedin_url,job_title,linked_timestamp,company_name,company_domain,company_type,industry,city,country,enrichment_timestamp,type,biography
        ,,,John,,john@example.com,,_valid_,,,,,,,,,,,,,
        """
        # Check output.
        self.assert_equal(actual, expected, fuzzy_match=True)

    def test4(self) -> None:
        """
        Test the final column order matches the Contact schema.
        """
        # Input data.
        df = pd.DataFrame(
            {"first_name": ["    John "], "Email": [" john@example.com "]}
        )
        cols_map = {"Email": "email", "first_name": "first_name"}
        # Call the function to test.
        actual = cmwdlout.normalize_contact_schema(df, cols_map)
        actual = actual.to_csv(index=False)
        expected = r"""
        hash,origin,origin_timestamp,first_name,last_name,email,email_timestamp,email_verification,email_verification_timestamp,linkedin_url,job_title,linked_timestamp,company_name,company_domain,company_type,industry,city,country,enrichment_timestamp,type,biography
        ,,,John,,john@example.com,,,,,,,,,,,,,,,
        """
        # Check output.
        self.assert_equal(actual, expected, fuzzy_match=True)

    def test5(self) -> None:
        """
        Test for None values in cols_map.
        """
        # Input data.
        df = pd.DataFrame(
            {"first_name": ["Alice Smith"], "email": ["alice@example.com"]}
        )
        cols_map = {"email": None, "first_name": None}
        # Call the function to test.
        result = cmwdlout.normalize_contact_schema(df, cols_map)
        actual = result.to_csv(index=False)
        expected = r"""
        hash,origin,origin_timestamp,first_name,last_name,email,email_timestamp,email_verification,email_verification_timestamp,linkedin_url,job_title,linked_timestamp,company_name,company_domain,company_type,industry,city,country,enrichment_timestamp,type,biography
        ,,,Alice Smith,,alice@example.com,,,,,,,,,,,,,,,
        """
        # Check output.
        self.assert_equal(actual, expected, fuzzy_match=True)

    def test6(self) -> None:
        """
        Test handling of empty DataFrame.
        """
        # Prepare input data.
        df = pd.DataFrame()
        cols_map = {}
        # Call function to test.
        actual = cmwdlout.normalize_contact_schema(df, cols_map)
        # Check output.
        self.assertTrue(actual.empty, "Returned DataFrame is not empty")


# #############################################################################
# Test_fuzzy_column_matching1
# #############################################################################


class Test_fuzzy_column_matching1(hunitest.TestCase):
    def helper(
        self,
        cols: List[str],
        expected_out_map: Dict[str, str],
        expected_unmatched_keys: List[str],
        expected_unmatched_cols: List[str],
    ) -> None:
        # Call function to test.
        out_map, unmatched_keys, unmatched_cols = cmwdlout.fuzzy_column_matching(
            cols
        )
        self.assertEqual(out_map, expected_out_map)
        self.assertEqual(unmatched_keys, expected_unmatched_keys)
        self.assertEqual(unmatched_cols, expected_unmatched_cols)

    def test1(self) -> None:
        """
        Test basic exact matching with lowercase columns.
        """
        cols = ["email", "first_name", "last_name"]
        expected_out_map = {
            "email": "email",
            "first_name": "first_name",
            "last_name": "last_name",
        }
        expected_unmatched_keys = [
            "full_name",
            "email_verification",
            "personal_email",
            "linkedin_url",
            "job_title",
            "company_name",
            "company_domain",
            "company_type",
            "company_industry",
            "city",
            "country",
            "biography",
            "timestamp",
        ]
        expected_unmatched_cols = []
        self.helper(
            cols,
            expected_out_map,
            expected_unmatched_keys,
            expected_unmatched_cols,
        )

    def test2(self) -> None:
        """
        Test matching with mixed case columns.
        """
        cols = ["Email", "FirstName", "LastName"]
        expected_out_map = {
            "email": "Email",
            "first_name": "FirstName",
            "last_name": "LastName",
        }
        expected_unmatched_keys = [
            "full_name",
            "email_verification",
            "personal_email",
            "linkedin_url",
            "job_title",
            "company_name",
            "company_domain",
            "company_type",
            "company_industry",
            "city",
            "country",
            "biography",
            "timestamp",
        ]
        expected_unmatched_cols = []
        self.helper(
            cols,
            expected_out_map,
            expected_unmatched_keys,
            expected_unmatched_cols,
        )

    def test3(self) -> None:
        """
        Test matching with underscore variations.
        """
        cols = ["first name", "firstname", "job_title"]
        expected_out_map = {
            "first_name": "first name",
            "job_title": "job_title",
        }
        expected_unmatched_keys = [
            "full_name",
            "last_name",
            "email",
            "email_verification",
            "personal_email",
            "linkedin_url",
            "company_name",
            "company_domain",
            "company_type",
            "company_industry",
            "city",
            "country",
            "biography",
            "timestamp",
        ]
        expected_unmatched_cols = ["firstname"]
        self.helper(
            cols,
            expected_out_map,
            expected_unmatched_keys,
            expected_unmatched_cols,
        )

    def test4(self) -> None:
        """
        Test with alias columns (e.g., profileUrl -> linkedin_url).
        """
        cols = ["profileurl", "professionalemail1", "refreshedat"]
        expected_out_map = {
            "email": "professionalemail1",
            "linkedin_url": "profileurl",
            "timestamp": "refreshedat",
        }
        expected_unmatched_keys = [
            "full_name",
            "first_name",
            "last_name",
            "email_verification",
            "personal_email",
            "job_title",
            "company_name",
            "company_domain",
            "company_type",
            "company_industry",
            "city",
            "country",
            "biography",
        ]
        expected_unmatched_cols = []
        self.helper(
            cols,
            expected_out_map,
            expected_unmatched_keys,
            expected_unmatched_cols,
        )

    def test5(self) -> None:
        """
        Test unmatched columns are reported correctly.
        """
        cols = ["unknown_column", "random_field", "email"]
        expected_out_map = {
            "email": "email",
        }
        expected_unmatched_keys = [
            "full_name",
            "first_name",
            "last_name",
            "email_verification",
            "personal_email",
            "linkedin_url",
            "job_title",
            "company_name",
            "company_domain",
            "company_type",
            "company_industry",
            "city",
            "country",
            "biography",
            "timestamp",
        ]
        expected_unmatched_cols = ["unknown_column", "random_field"]
        self.helper(
            cols,
            expected_out_map,
            expected_unmatched_keys,
            expected_unmatched_cols,
        )

    def test6(self) -> None:
        """
        Test that location matches city (first match wins).
        """
        cols = ["location"]
        expected_out_map = {
            "city": "location",
        }
        expected_unmatched_keys = [
            "full_name",
            "first_name",
            "last_name",
            "email",
            "email_verification",
            "personal_email",
            "linkedin_url",
            "job_title",
            "company_name",
            "company_domain",
            "company_type",
            "company_industry",
            "country",
            "biography",
            "timestamp",
        ]
        expected_unmatched_cols = []
        self.helper(
            cols,
            expected_out_map,
            expected_unmatched_keys,
            expected_unmatched_cols,
        )

    def test7(self) -> None:
        """
        Test with empty column list.
        """
        cols = []
        expected_out_map = {}
        expected_unmatched_keys = [
            "full_name",
            "first_name",
            "last_name",
            "email",
            "email_verification",
            "personal_email",
            "linkedin_url",
            "job_title",
            "company_name",
            "company_domain",
            "company_type",
            "company_industry",
            "city",
            "country",
            "biography",
            "timestamp",
        ]
        expected_unmatched_cols = []
        self.helper(
            cols,
            expected_out_map,
            expected_unmatched_keys,
            expected_unmatched_cols,
        )

    def test8(self) -> None:
        """
        Test with complex aliases (e.g., Primary Position -> job_title).
        """
        cols = ["primary position", "primary company", "companywebsite"]
        expected_out_map = {
            "company_domain": "companywebsite",
            "company_name": "primary company",
            "job_title": "primary position",
        }
        expected_unmatched_keys = [
            "full_name",
            "first_name",
            "last_name",
            "email",
            "email_verification",
            "personal_email",
            "linkedin_url",
            "company_type",
            "company_industry",
            "city",
            "country",
            "biography",
            "timestamp",
        ]
        expected_unmatched_cols = []
        self.helper(
            cols,
            expected_out_map,
            expected_unmatched_keys,
            expected_unmatched_cols,
        )
