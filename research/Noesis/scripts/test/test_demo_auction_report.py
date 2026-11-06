"""
Smoke-test the offline reviewer command and its PNG artifact.

Import as:

import research.Noesis.scripts.test.test_demo_auction_report as rnsctedear
"""

import os
import shlex
import sys

import helpers.hsystem as hsystem
import helpers.hunit_test as hunitest


class Test_demo_auction_report(hunitest.TestCase):
    """
    Run the public command rather than mocking report or plotting helpers.
    """

    def test1(self) -> None:
        """
        Display every report and export a PNG with explicit mocked fulfillment.
        """
        # Prepare inputs: use test-owned scratch space for the only file output.
        path = os.path.join(self.get_scratch_space(), "auction.png")
        command = (
            f"{shlex.quote(sys.executable)} -m "
            "research.Noesis.scripts.demo_auction_report "
            f"--scenario excess_demand --plot {shlex.quote(path)}"
        )
        # Run test.
        return_code, output = hsystem.system_to_string(command)
        # Check outputs: useful table headings and a real image, not log chatter.
        labels = [
            line
            for line in output.splitlines()
            if line in ("orders=", "contracts=", "tiers=", "unmatched=")
        ]
        self.assertEqual(
            (return_code, labels),
            (0, ["orders=", "contracts=", "tiers=", "unmatched="]),
        )
        self.assertTrue("Fulfillment is MOCKED" in output)
        with open(path, "rb") as stream:
            self.assertEqual(
                stream.read(8), bytes([137, 80, 78, 71, 13, 10, 26, 10])
            )
