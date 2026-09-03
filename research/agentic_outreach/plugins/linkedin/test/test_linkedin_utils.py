import logging
import os

import pandas as pd

import ck_marketing.plugins.linkedin.linkedin_utils as cmplliut
import helpers.hs3 as hs3
import helpers.hunit_test as hunitest

_LOG = logging.getLogger(__name__)

# #############################################################################
# TestCleanFirstLastName
# #############################################################################


class TestCleanFirstLastName(hunitest.TestCase):
    def test1(self) -> None:
        # Mapping of edge case names to expected outputs.
        # Input: (first_name, last_name)
        # Output: (cleaned_first_name, cleaned_last_name, first_alias, second_alias)
        test_cases = {
            # Already in a clean state: no transformations
            ("John", "Doe"): ("John", "Doe", "", ""),
            ("AJ", "Cawood"): ("AJ", "Cawood", "", ""),
            ("Ann", "Miura-Ko"): ("Ann", "Miura-Ko", "", ""),
            ("D", "Sharma"): ("D", "Sharma", "", ""),
            # Test cases that require transformations.
            ("Adam", "Ibarra, CPA PFP"): ("Adam", "Ibarra", "", ""),
            ("John", "Doe, PhD"): ("John", "Doe", "", ""),
            (
                "Adrianna",
                "Samaniego (she/her/ella)",
            ): ("Adrianna", "Samaniego", "", ""),
            ("Amanda", "(she/her)"): ("Amanda", "", "", ""),
            ("Dr. Marie", "Curie"): ("Marie", "Curie", "", ""),
            ("Gianpiero (JP)", "Balestrieri"): (
                "Gianpiero",
                "Balestrieri",
                "JP",
                "",
            ),
            ("Amanda", "Townsend (she/her)"): ("Amanda", "Townsend", "", ""),
            ("Baylor", "Y."): ("Baylor", "Y", "", ""),
            ("Christine S.", "Li-AuYeung, CPA, MBA"): (
                "Christine",
                "Li-Auyeung",
                "",
                "",
            ),
            ("Dr. Felix", "Cardenas"): ("Felix", "Cardenas", "", ""),
            ("Erez", "Hevroni, CEPA, AWMA, AAMS"): (
                "Erez",
                "Hevroni",
                "",
                "",
            ),
            ("Jean-paul (j.p.", "Sanday"): (
                "Jean-Paul",
                "Sanday",
                "J.P.",
                "",
            ),
            ("John", "Doe, PhD (he/him)"): ("John", "Doe", "", ""),
            ("Alicia", "O'Connell"): ("Alicia", "O'Connell", "", ""),
            ("Jack", "DuBro"): ("Jack", "DuBro", "", ""),
            ("Abbie J.", "Cohen"): ("Abbie", "Cohen", "", ""),
        }
        # Collect actual outputs.
        cleaned_outputs = {}
        for key, _ in test_cases.items():
            first_name, last_name = key
            cleaned_first = cmplliut.clean_first_last_name(
                name=first_name, is_last_name=False
            )
            cleaned_last = cmplliut.clean_first_last_name(
                name=last_name, is_last_name=True
            )
            # Combine cleaned first and last names along with the nickname.
            combined_output = (
                cleaned_first[0],
                cleaned_last[0],
                cleaned_first[1],
                cleaned_last[1],
            )
            cleaned_outputs[key] = combined_output
        # Compare all outputs.
        self.assertEqual(cleaned_outputs, test_cases)

    # #########################################################################

    def get_data(
        self,
        s3_path: str,
        aws_profile: str,
        file_name: str,
    ) -> pd.DataFrame:
        """
        Get data from S3.

        :param s3_path: path to a CSV file in S3
        :param aws_profile: AWS profile to use
        :param file_name: name of the CSV file to load
        :return: df
        """
        dir_name = self.get_scratch_space()
        hs3.copy_data_from_s3_to_local_dir(s3_path, dir_name, aws_profile)
        file_path = os.path.join(dir_name, file_name)
        df = pd.read_csv(file_path)
        return df

    def test2(self) -> None:
        """
        Integration test that:
        1) Fetches a CSV file of names from S3.
        2) Cleans the first and last names.
        3) Computes simple statistics about how many names were changed.
        """
        # Create a df with first and last names.
        s3_path = "s3://cryptokaizen-data-test/ck_marketing/"
        df = self.get_data(
            s3_path, aws_profile="ck", file_name="clean_up_names.csv"
        )
        # Clean first and last names.
        df = cmplliut.clean_and_track_name_changes(
            df, first_name_col="first_name", last_name_col="last_name"
        )
        # Compute number of changes.
        _LOG.debug("modified=\n%s", df[df["is_modified"]])
        cleaned_count = df["is_modified"].sum()
        expected_count = 245
        # Compare results.
        self.assertEqual(cleaned_count, expected_count)
        df_str = df.to_string()
        self.check_string(df_str)
