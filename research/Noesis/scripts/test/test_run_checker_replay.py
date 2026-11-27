"""
Test the runnable offline checker replay command.
"""

import argparse
import pathlib
import tempfile

import helpers.hunit_test as hunitest
import research.Noesis.scripts.run_checker_replay as rnscchre


class Test_parse_seeds(hunitest.TestCase):
    """
    Test deterministic seed-list parsing.
    """

    def test1(self) -> None:
        """
        Test a valid list preserves signed integer seeds.
        """
        self.assertEqual(rnscchre.parse_seeds("0, 2,-3"), (0, 2, -3))

    def test2(self) -> None:
        """
        Test empty, duplicate, and noninteger seed lists fail clearly.
        """
        for value in ("", "0,", "1,1", "a"):
            with self.assertRaises(argparse.ArgumentTypeError):
                rnscchre.parse_seeds(value)


class Test_main(hunitest.TestCase):
    """
    Test the one-command experiment and artifact contract.
    """

    def test1(self) -> None:
        """
        Test the default-sized acceptance run writes all three outputs.
        """
        with tempfile.TemporaryDirectory() as directory:
            args = rnscchre._parse_args(
                [
                    "--seeds",
                    "0",
                    "--contracts_per_seed",
                    "1000",
                    "--output_dir",
                    directory,
                ]
            )
            # Run test.
            actual = rnscchre._main(args)
            # Check outputs.
            self.assertEqual(actual, 0)
            output_dir = pathlib.Path(directory)
            self.assertTrue((output_dir / "checker_replay.json").is_file())
            self.assertTrue((output_dir / "checker_replay.csv").is_file())
            self.assertTrue((output_dir / "checker_replay.md").is_file())

    def test2(self) -> None:
        """
        Test invalid contract counts fail before the experiment runs.
        """
        args = rnscchre._parse_args(["--contracts_per_seed", "0"])
        with self.assertRaises(ValueError):
            rnscchre._main(args)
