import logging
import os

import helpers.hunit_test as hunitest
import research.Noesis.noesis_settings as rnonoset

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_read_dotenv
# #############################################################################


class Test_read_dotenv(hunitest.TestCase):
    """
    Test parsing of dotenv files.
    """

    def test1(self) -> None:
        """
        Test that comments, blanks, and quotes are handled.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), ".env")
        with open(path, "w") as f:
            f.write("# comment\n\nA=1\nB=\"two\"\nC='three'\nnot a pair\n")
        # Prepare outputs.
        expected = {"A": "1", "B": "two", "C": "three"}
        # Run test.
        actual = rnonoset.read_dotenv(path)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test2(self) -> None:
        """
        Test that a missing file yields an empty dict.
        """
        # Run test.
        actual = rnonoset.read_dotenv("/nonexistent/.env")
        # Check outputs.
        self.assertEqual(actual, {})


# #############################################################################
# Test_load_settings
# #############################################################################


class Test_load_settings(hunitest.TestCase):
    """
    Test building `Settings` from a dotenv file and the environment.
    """

    def helper(self, environ: dict, dotenv: str) -> rnonoset.Settings:
        """
        Load settings from `dotenv` text overlaid with `environ`.
        """
        path = os.path.join(self.get_scratch_space(), ".env")
        with open(path, "w") as f:
            f.write(dotenv)
        settings = rnonoset.load_settings(environ=environ, dotenv_path=path)
        return settings

    def test1(self) -> None:
        """
        Test the defaults when nothing is set.
        """
        # Prepare outputs.
        expected = rnonoset.Settings(
            openrouter_api_key="",
            database_url=rnonoset.DEFAULT_DATABASE_URL,
            model_id=rnonoset.DEFAULT_MODEL_ID,
            max_tokens_cap=256,
            spend_cap_usd=50.0,
            timeout_multiplier=2.0,
            seed_path=rnonoset.DEFAULT_SEED_PATH,
        )
        # Run test.
        actual = self.helper({}, "")
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test2(self) -> None:
        """
        Test that the environment wins over the dotenv file.
        """
        # Prepare inputs.
        environ = {"NOESIS_SPEND_CAP_USD": "10"}
        dotenv = 'OPENROUTER_API_KEY="sk-or-x"\nNOESIS_SPEND_CAP_USD=99\n'
        # Run test.
        actual = self.helper(environ, dotenv)
        # Check outputs.
        self.assertEqual(actual.openrouter_api_key, "sk-or-x")
        self.assertEqual(actual.spend_cap_usd, 10.0)

    def test3(self) -> None:
        """
        Test that a non-numeric value fails loudly, naming the setting.
        """
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            self.helper({"NOESIS_TIMEOUT_MULTIPLIER": "abc"}, "")
        # Check outputs.
        self.assertIn("NOESIS_TIMEOUT_MULTIPLIER", str(cm.exception))

    def test4(self) -> None:
        """
        Test that zero and negative values are rejected.
        """
        # Run test and check outputs.
        for name, value in [
            ("NOESIS_MAX_TOKENS_CAP", "0"),
            ("NOESIS_SPEND_CAP_USD", "-1"),
        ]:
            with self.assertRaises(AssertionError):
                self.helper({name: value}, "")

    def test5(self) -> None:
        """
        Test that a fractional token cap is rejected.
        """
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            self.helper({"NOESIS_MAX_TOKENS_CAP": "10.5"}, "")
