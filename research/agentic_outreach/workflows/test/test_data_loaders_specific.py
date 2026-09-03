import logging
import os
import unittest.mock

import pandas as pd
import pytest

import ck_marketing.workflows.data_loaders_utils as cmwdlout
import ck_marketing.workflows.data_loaders_specific as cmwdlosp
import helpers.hunit_test as hunitest


_LOG = logging.getLogger(__name__)


def _load_mock_data(test_instance: hunitest.TestCase) -> pd.DataFrame:
    """
    Load mock data from a CSV file.

    :param test_instance: instance used to retrieve the input data
    :return: mock data with NaN values replaced by empty strings
    """
    dir_name = test_instance.get_input_dir()
    test_csv_path = os.path.join(dir_name, "test.csv")
    mock_data = pd.read_csv(test_csv_path)
    # Use empty string for empty value to avoid mismatch.
    df = mock_data.fillna("").astype(str)
    return df


def _run_test(
    test_instance: hunitest.TestCase,
    function_name: str,
    normalize: bool,
    *,
    mock_func_name: str = "helpers.hgoogle_drive_api.get_cached_gsheet_to_df",
    use_mock_data: bool = True,
) -> pd.DataFrame:
    """
    Test scraping function using mock or real data.

    - 'use_mock_data' is set to True by default for all functions. The switch
      enables data mocking instead of extracting real data from files which
      depends ongoogle APIs.
    - When set to False, the real time data is extracted and tested upon. The
      tests are completed in much less time with mock data. Set to False for
      testing real data [Google API dependant when set to false].

    :param test_instance: instance used to retrieve test input data
    :param function_name: name of the function to be tested from the ckmdatal
        module
    :param normalize: flag indicating whether to normalize the data
    :param use_mock_data: flag indicating whether to use mock data
    :return: data containing the first row of the function output
    """
    if use_mock_data:
        mock_data = _load_mock_data(test_instance)
        with unittest.mock.patch(mock_func_name) as mock_func:
            _LOG.info("Mock data used for tests")
            mock_func.side_effect = lambda *args, **kwargs: mock_data.copy()
            df_out = getattr(cmwdlosp, function_name)(
                normalize=normalize, verbose=False
            )
            df_out = df_out[:1]
    else:
        _LOG.info("Real data used for tests")
        df_out = getattr(cmwdlosp, function_name)(
            normalize=normalize, verbose=False
        )
        df_out = df_out[:1]
    return df_out


# #############################################################################
# Test_get_data_from_PB_SalesNavigator_Connections_Export
# #############################################################################


class Test_get_data_from_PB_SalesNavigator_Connections_Export(hunitest.TestCase):
    def test1(self) -> None:
        """
        Test retrieval of scraped data from LinkedIn after normalization on
        real or mock data.
        """
        df = _run_test(
            self,
            "get_data_from_PB_SalesNavigator_Connections_Export",
            normalize=True,
        )
        # Check schema after data normalization.
        contact_schema = cmwdlout.get_contact_df_schema()
        expected_columns: set[str] = set(contact_schema)
        self.assertSetEqual(set(df.columns), expected_columns)
        # Check output.
        actual = df.to_csv(index=False)
        self.check_string(actual)

    def test2(self) -> None:
        """
        Test retrieval of scraped data from LinkedIn with no normalization on
        real or mock data.
        """
        df = _run_test(
            self,
            "get_data_from_PB_SalesNavigator_Connections_Export",
            normalize=False,
        )
        actual = df.to_csv(index=False)
        # Check output.
        self.check_string(actual)


# #############################################################################
# Test_extract_and_validate_email
# #############################################################################


class Test_extract_and_validate_email(hunitest.TestCase):
    """
    Test email extraction and validation.
    """

    def test1(self) -> None:
        """
        Test when dataframe has single email.
        """
        # Prepare input.
        data_single_email = {
            "hunter_extracted_email": ["test1@example.com"],
            "dropcontact_mail": ["nan"],
            "all_emails": ["nan"],
        }
        # Call function to test.
        df = pd.DataFrame(data_single_email)
        actual = cmwdlosp._extract_and_validate_email(df)
        expected = ["test1@example.com"]
        # Check return of single found mail.
        self.assertEqual(actual, expected)

    def test2(self) -> None:
        """
        Test when dataframe has multiple emails.
        """
        # Prepare inputs.
        data_multiple_emails = {
            "hunter_extracted_email": ["test1@example.com"],
            "dropcontact_mail": ["test2@example.com"],
            "all_emails": ["nan"],
        }
        df = pd.DataFrame(data_multiple_emails)
        # Call function to test.
        with self.assertRaises(ValueError) as cm:
            cmwdlosp._extract_and_validate_email(df)
        actual = str(cm.exception)
        expected = r"""
        Multiple emails found: {test1@example.com, test2@example.com}
        """
        # Check output.
        self.assert_equal(actual, expected, fuzzy_match=True)

    def test3(self) -> None:
        """
        Test when there is no mails.
        """
        # Prepare input.
        data_no_email = {
            "hunter_extracted_email": ["nan"],
            "dropcontact_mail": ["nan"],
            "all_emails": ["nan"],
        }
        df_no_email = pd.DataFrame(data_no_email)
        expected = ["nan"]
        # Call function to test.
        actual = cmwdlosp._extract_and_validate_email(df_no_email)
        # Check output.
        self.assertEqual(actual, expected)

    def test4(self) -> None:
        """
        Test when the dataframe is empty.
        """
        # Prepare input.
        df_empty = pd.DataFrame(
            columns=["hunter_extracted_email", "dropcontact_mail", "all_emails"]
        )
        # Call function to test.
        actual = cmwdlosp._extract_and_validate_email(df_empty)
        expected: list[str] = []
        # Check output.
        self.assertEqual(actual, expected)

    def test5(self) -> None:
        """
        Test when dataframe is mix of empty and non empty mails.
        """
        # Prepare input.
        data_mixed_email = {
            "hunter_extracted_email": ["nan", "test1@example.com"],
            "dropcontact_mail": ["test2@example.com", "nan"],
            "all_emails": ["nan", "nan"],
        }
        df_mixed_email = pd.DataFrame(data_mixed_email)
        # Call function to test.
        actual = cmwdlosp._extract_and_validate_email(df_mixed_email)
        expected = ["test2@example.com", "test1@example.com"]
        # Check output.
        self.assertEqual(actual, expected)


# #############################################################################
# Test_get_data_from_LinkedIn2
# #############################################################################


class Test_get_data_from_LinkedIn2(hunitest.TestCase):
    def test1(self) -> None:
        """
        Test retrieval of scraped data from LinkedIn2 after normalization for
        real or mock data.
        """
        df = _run_test(self, "get_data_from_LinkedIn2", normalize=True)
        # Check schema after data normalization.
        contact_schema = cmwdlout.get_contact_df_schema()
        expected_columns: set[str] = set(contact_schema)
        self.assertSetEqual(set(df.columns), expected_columns)
        # Check output.
        actual = df.to_csv(index=False)
        self.check_string(actual)

    def test2(self) -> None:
        """
        Test retrieval of scraped data from LinkedIn2 without normalization for
        real or mock data.
        """
        df = _run_test(self, "get_data_from_LinkedIn2", normalize=False)
        df = df.to_csv(index=False)
        # Replace 'True' with 'TRUE' and 'False' with 'FALSE' to avoid string mismatch.
        actual = df.replace(",True,", ",TRUE,").replace(",False,", ",FALSE,")
        # Check output.
        self.check_string(actual)


# #############################################################################
# Test_get_data_from_LinkedIn3
# #############################################################################


@pytest.mark.skip(reason="Fix mocking")
class Test_get_data_from_LinkedIn3(hunitest.TestCase):
    def test1(self) -> None:
        """
        Test retrieval of scraped data from LinkedIn3 after normalization for
        real or mock data.
        """
        df = _run_test(
            self,
            "get_data_from_LinkedIn3",
            normalize=True,
            mock_func_name="ck_marketing.workflows.data_loaders_utils.get_cached_gsheet_to_df2",
        )
        # Check schema after data normalization.
        contact_schema = cmwdlout.get_contact_df_schema()
        expected_columns: set[str] = set(contact_schema)
        self.assertSetEqual(set(df.columns), expected_columns)
        actual = df.to_csv(index=False)
        # Check output.
        self.check_string(actual)

    def test2(self) -> None:
        """
        Test retrieval of scraped data from LinkedIn3 without normalization for
        real or mock data.
        """
        df = _run_test(
            self,
            "get_data_from_LinkedIn3",
            normalize=False,
            mock_func_name="ck_marketing.workflows.data_loaders_utils.get_cached_gsheet_to_df2",
        )
        df = df.to_csv(index=False)
        # Replace 'True' with 'TRUE' and 'False' with 'FALSE' to avoid string mismatch.
        actual = df.replace(",True,", ",TRUE,").replace(",False,", ",FALSE,")
        # Check output.
        self.check_string(actual)


# #############################################################################
# Test_get_data_from_LinkedIn4
# #############################################################################


class Test_get_data_from_LinkedIn4(hunitest.TestCase):
    def test1(self) -> None:
        """
        Test retrieval of scraped data from LinkedIn4 after normalization for
        real or mock data.
        """
        use_mock_data = True

        if use_mock_data:
            dir_name = self.get_input_dir()
            test_csv_path = os.path.join(dir_name, "test.csv")
            mock_data = pd.read_csv(test_csv_path)
            # Use empty string for empty value to avoid mismatch.
            mock_main_data = mock_data.fillna("").astype(str)
            mock_validity_data = pd.DataFrame(
                {
                    "email_first": ["dk@accel.com"],
                    "hunter_verification": ["valid"],
                }
            )
            with unittest.mock.patch(
                "helpers.hgoogle_drive_api.get_cached_gsheet_to_df"
            ) as mock_func:
                # Replace the actual function call with a predefined mock dataset
                # by returning a copy of `mock_data`,ensuring consistent test results.
                mock_func.side_effect = lambda url, tab_name: (
                    mock_main_data
                    if tab_name in ("Sheet1", "Sheet2", "Sheet3")
                    else mock_validity_data
                )
                df_out = cmwdlosp.get_data_from_LinkedIn4(
                    normalize=True, verbose=False
                )
                df_out = df_out[:1]
        else:
            df_out = cmwdlosp.get_data_from_LinkedIn4(
                normalize=True, verbose=False
            )
            df_out = df_out[:1]
        # Check schema after data normalization.
        contact_schema = cmwdlout.get_contact_df_schema()
        expected_columns: set[str] = set(contact_schema)
        self.assertSetEqual(set(df_out.columns), expected_columns)
        actual = df_out.to_csv(index=False)
        # Check output.
        self.check_string(actual)

    def test2(self) -> None:
        """
        Test retrieval of scraped data from LinkedIn4 without normalization for
        real or mock data.
        """

        use_mock_data = True

        if use_mock_data:
            dir_name = self.get_input_dir()
            test_csv_path = os.path.join(dir_name, "test.csv")
            pd.read_csv(test_csv_path)
            mock_main_data = pd.read_csv(test_csv_path)
            # Use empty string for empty value to avoid mismatch.
            mock_validity_data = pd.DataFrame(
                {
                    "email_first": ["dk@accel.com"],
                    "hunter_verification": ["valid"],
                }
            )
            with unittest.mock.patch(
                "helpers.hgoogle_drive_api.get_cached_gsheet_to_df"
            ) as mock_func:
                # Replace the actual function call with a predefined mock dataset
                # by returning a copy of `mock_data`,ensuring consistent test results.
                mock_func.side_effect = lambda url, tab_name: (
                    mock_main_data
                    if tab_name in ("Sheet1", "Sheet2", "Sheet3")
                    else mock_validity_data
                )
                df_not_norm = cmwdlosp.get_data_from_LinkedIn4(
                    normalize=False, verbose=False
                )
                df_not_norm = df_not_norm[:1]
        else:
            df_not_norm = cmwdlosp.get_data_from_LinkedIn4(
                normalize=False, verbose=False
            )
            df_not_norm = df_not_norm[:1]
        actual = df_not_norm.to_csv(index=False)
        # Check output.
        self.check_string(actual)


# #############################################################################
# Test_get_data_from_LinkedIn5
# #############################################################################


class Test_get_data_from_LinkedIn5(hunitest.TestCase):
    def test1(self) -> None:
        """
        Test retrieval of scraped data from LinkedIn5 after normalization for
        real or mock data.
        """
        df = _run_test(self, "get_data_from_LinkedIn5", normalize=True)
        # Check schema after data normalization.
        contact_schema = cmwdlout.get_contact_df_schema()
        expected_columns: set[str] = set(contact_schema)
        self.assertSetEqual(set(df.columns), expected_columns)
        actual = df.to_csv(index=False)
        # Check output.
        self.check_string(actual)

    def test2(self) -> None:
        """
        Test retrieval of scraped data from LinkedIn5 without normalization for
        real or mock data.
        """
        df = _run_test(self, "get_data_from_LinkedIn5", normalize=False)
        actual = df.to_csv(index=False)
        # Check output.
        self.check_string(actual)


# #############################################################################
# Test_get_data_from_VCSheet
# #############################################################################


class Test_get_data_from_VCSheet(hunitest.TestCase):
    def test1(self) -> None:
        """
        Test retrieval of scraped data from VCSheet after normalization for
        real or mock data.
        """
        df = _run_test(self, "get_data_from_VCSheet", normalize=True)
        # Check schema after data normalization.
        contact_schema = cmwdlout.get_contact_df_schema()
        expected_columns: set[str] = set(contact_schema)
        self.assertSetEqual(set(df.columns), expected_columns)
        actual = df.to_csv(index=False)
        # Check output.
        self.check_string(actual)

    def test2(self) -> None:
        """
        Test retrieval of scraped data from VCSheet without normalization for
        real or mock data.
        """
        df = _run_test(self, "get_data_from_VCSheet", normalize=False)
        actual = df.to_csv(index=False)
        # Check output.
        self.check_string(actual)


# #############################################################################
# Test_get_data_from_EuroVC
# #############################################################################


class Test_get_data_from_EuroVC(hunitest.TestCase):
    def test1(self) -> None:
        """
        Test retrieval of scraped data from EuroVC after normalization for real
        or mock data.
        """
        df = _run_test(self, "get_data_from_EuroVC", normalize=True)
        # Check schema after data normalization.
        contact_schema = cmwdlout.get_contact_df_schema()
        expected_columns: set[str] = set(contact_schema)
        self.assertSetEqual(set(df.columns), expected_columns)
        actual = df.to_csv(index=False)
        # Check output.
        self.check_string(actual)

    def test2(self) -> None:
        """
        Test retrieval of scraped data from EuroVC without normalization for
        real or mock data.
        """
        df = _run_test(self, "get_data_from_EuroVC", normalize=False)
        df = df.to_csv(index=False)
        # Replace 'True' with 'true' and 'False' with 'false' to avoid string mismatch.
        actual = df.replace(",True,", ",true,").replace(",False,", ",false,")
        # Check output.
        self.check_string(actual)


# #############################################################################
# Test_get_data_from_FolkApp
# #############################################################################


class Test_get_data_from_FolkApp(hunitest.TestCase):
    def test1(self) -> None:
        """
        Test retrieval of scraped data from FolkApp after normalization for
        real or mock data.
        """
        df = _run_test(self, "get_data_from_FolkApp", normalize=True)
        # Check schema after data normalization.
        contact_schema = cmwdlout.get_contact_df_schema()
        expected_columns: set[str] = set(contact_schema)
        self.assertSetEqual(set(df.columns), expected_columns)
        actual = df.to_csv(index=False)
        # Check output.
        self.check_string(actual)

    def test2(self) -> None:
        """
        Test retrieval of scraped data from FolkApp without normalization for
        real or mock data.
        """
        df = _run_test(self, "get_data_from_FolkApp", normalize=False)
        actual = df.to_csv(index=False)
        # Check output.
        self.check_string(actual)


# #############################################################################
# Test_get_data_from_hedge_fund_list
# #############################################################################


class Test_get_data_from_hedge_fund_list(hunitest.TestCase):
    def test1(self) -> None:
        """
        Test retrieval of scraped data from HedgeFund after normalization for
        real or mock data.
        """
        df = _run_test(self, "get_data_from_hedge_fund_list", normalize=True)
        # Check schema after data normalization.
        contact_schema = cmwdlout.get_contact_df_schema()
        expected_columns: set[str] = set(contact_schema)
        self.assertSetEqual(set(df.columns), expected_columns)
        # Check output.
        actual = df.to_csv(index=False)
        self.check_string(actual)

    def test2(self) -> None:
        """
        Test retrieval of scraped data from HedgeFund without normalization for
        real or mock data.
        """
        df = _run_test(self, "get_data_from_hedge_fund_list", normalize=False)
        actual = df.to_csv(index=False)
        # Check output.
        self.check_string(actual)


# #############################################################################
# Test_get_data_from_GP_LIn_connections
# #############################################################################


class Test_get_data_from_GP_LIn_connections(hunitest.TestCase):
    def test1(self) -> None:
        """
        Test retrieval of scraped data from GP LinkedIn after normalization for
        real or mock data.
        """
        df = _run_test(self, "get_data_from_GP_LIn_connections", normalize=True)
        # Check schema after data normalization.
        contact_schema = cmwdlout.get_contact_df_schema()
        expected_columns: set[str] = set(contact_schema)
        self.assertSetEqual(set(df.columns), expected_columns)
        actual = df.to_csv(index=False)
        # Check output.
        self.check_string(actual)

    def test2(self) -> None:
        """
        Test retrieval of scraped data from GP Link without normalization for
        real or mock data.
        """
        df = _run_test(self, "get_data_from_GP_LIn_connections", normalize=False)
        actual = df.to_csv(index=False)
        actual = actual.replace(",False", ",FALSE")
        # Check output.
        self.check_string(actual)
