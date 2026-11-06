import logging
import os
from typing import Any, Dict, List, Tuple

import helpers.hio as hio
import helpers.hprint as hprint
import helpers.hunit_test as hunitest
import yaml

import research.Estimate_academic_paper_quality.populate_affiliation_tiers as popatier
import research.Estimate_academic_paper_quality.report_papers as repopapers

_LOG = logging.getLogger(__name__)

# The names below are fixtures of the tests, not real ranks.
_ROWS = [
    (3, "Test Institute of Technology (TIT)"),
    (1, "Alpha University"),
    (21, "Beta University"),
    (150, "Far University"),
]


def _build(
    rows: List[Tuple[int, str]], existing: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build the YAML text and read it back.
    """
    text = popatier._build_yaml(
        rows,
        existing,
        as_of="Test ranking, accessed 2026-01-01",
        source="https://example.com/ranking",
        tier_5_max_rank=20,
        tier_4_max_rank=100,
    )
    return yaml.safe_load(text)


# #############################################################################
# Test_parse_rank
# #############################################################################


class Test_parse_rank(hunitest.TestCase):
    """
    Test the parsing of the rank text of a QS row.
    """

    def helper(self, text: str, expected: int) -> None:
        # Run test.
        actual = popatier._parse_rank(text)
        # Check outputs.
        self.assertEqual(actual, expected)

    def test1(self) -> None:
        """
        Test a plain rank.
        """
        self.helper("7", 7)

    def test2(self) -> None:
        """
        Test a tie.
        """
        self.helper("=20", 20)

    def test3(self) -> None:
        """
        Test a band takes its first number.
        """
        self.helper("101-110", 101)

    def test4(self) -> None:
        """
        Test an open band.
        """
        self.helper("601+", 601)

    def test5(self) -> None:
        """
        Test a text with no number gives 0.
        """
        self.helper("-", 0)


# #############################################################################
# Test_split_name
# #############################################################################


class Test_split_name(hunitest.TestCase):
    """
    Test that the acronym in parentheses becomes an alias.
    """

    def test1(self) -> None:
        """
        Test a name with an acronym.
        """
        # Run test.
        actual = popatier._split_name("Test Institute of Technology (TIT)")
        # Check outputs.
        expected = ("Test Institute of Technology", ["TIT"])
        self.assertEqual(actual, expected)

    def test2(self) -> None:
        """
        Test a name with no parentheses.
        """
        # Run test.
        actual = popatier._split_name("  Alpha   University ")
        # Check outputs.
        expected = ("Alpha University", [])
        self.assertEqual(actual, expected)


# #############################################################################
# Test_read_qs_rows
# #############################################################################


class Test_read_qs_rows(hunitest.TestCase):
    """
    Test the reading of the QS CSV.
    """

    def write_csv(self, text: str) -> str:
        path = os.path.join(self.get_scratch_space(), "qs.csv")
        hio.to_file(path, text)
        return path

    def test1(self) -> None:
        """
        Test that the columns are found, a BOM is dropped, and a bad row is
        skipped.
        """
        # Prepare inputs.
        text = "﻿Rank,Institution,Country\n1,Alpha University,X\n=3,Beta University,Y\n-,No Rank University,Z\n"
        path = self.write_csv(text)
        # Run test.
        actual = popatier._read_qs_rows(path, rank_col="", name_col="")
        # Check outputs.
        expected = [(1, "Alpha University"), (3, "Beta University")]
        self.assertEqual(actual, expected)

    def test2(self) -> None:
        """
        Test that the only header with `rank` in it is the rank column.
        """
        # Prepare inputs.
        path = self.write_csv("2026 Rank,University\n5,Alpha University\n")
        # Run test.
        actual = popatier._read_qs_rows(path, rank_col="", name_col="")
        # Check outputs.
        expected = [(5, "Alpha University")]
        self.assertEqual(actual, expected)

    def test3(self) -> None:
        """
        Test that the columns can be named with the flags.
        """
        # Prepare inputs.
        path = self.write_csv("Pos,Who\n5,Alpha University\n")
        # Run test.
        actual = popatier._read_qs_rows(path, rank_col="Pos", name_col="Who")
        # Check outputs.
        expected = [(5, "Alpha University")]
        self.assertEqual(actual, expected)

    def test4(self) -> None:
        """
        Test that the read fails if two headers can be the rank.
        """
        # Prepare inputs.
        path = self.write_csv("2025 Rank,2026 Rank,University\n5,4,Alpha\n")
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            popatier._read_qs_rows(path, rank_col="", name_col="")


# #############################################################################
# Test_build_yaml
# #############################################################################


class Test_build_yaml(hunitest.TestCase):
    """
    Test the text of the YAML file.
    """

    def test1(self) -> None:
        """
        Test the file for an empty existing file.
        """
        # Run test.
        actual = popatier._build_yaml(
            _ROWS,
            {},
            as_of="Test ranking, accessed 2026-01-01",
            source="https://example.com/ranking",
            tier_5_max_rank=20,
            tier_4_max_rank=100,
        )
        # Check outputs.
        expected = hprint.dedent(
            """
            # Dated list of the institutions that score 5 and 4 in the affiliation signal.
            #
            # The script `populate_affiliation_tiers.py` writes the entries that have a
            # `qs_rank` key, from a CSV exported from the QS World University Rankings site.
            # Never fill them from the memory of an LLM (`docs/03-decisions.md`, D15).
            # An entry with no `qs_rank` key is a hand entry (an industry lab, a central
            # bank, NBER): the script keeps it. Add the `aliases` of an entry by hand: the
            # script keeps them.
            #
            # How to fill the file: `affiliation_tiers.yaml.README.md`.
            as_of: "Test ranking, accessed 2026-01-01"
            source: "https://example.com/ranking"
            # QS rank 1 to 20, plus leading industry research labs, central banks, NBER.
            tier_5:
              - name: "Alpha University"
                aliases: []
                qs_rank: 1
              - name: "Test Institute of Technology"
                aliases: ["TIT"]
                qs_rank: 3
            # QS rank 21 to 100, plus established industry research groups.
            tier_4:
              - name: "Beta University"
                aliases: []
                qs_rank: 21
            """
        )
        self.assert_equal(actual, expected, remove_lead_trail_empty_lines=True)

    def test2(self) -> None:
        """
        Test that hand entries and hand aliases survive, and a stale QS entry
        goes away.
        """
        # Prepare inputs.
        existing = {
            "tier_5": [
                {"name": "Alpha University", "aliases": ["Alpha"], "qs_rank": 9},
                {"name": "Old University", "aliases": [], "qs_rank": 2},
                {"name": "Test Central Bank", "aliases": ["TCB"]},
            ],
            "tier_4": [
                {"name": "Test Lab", "aliases": [], "note": "hand"},
                {"name": "Beta University", "aliases": []},
            ],
        }
        # Run test.
        actual = _build(_ROWS, existing)
        # Check outputs.
        names_5 = [entry["name"] for entry in actual["tier_5"]]
        names_4 = [entry["name"] for entry in actual["tier_4"]]
        self.assertEqual(
            names_5,
            [
                "Alpha University",
                "Test Institute of Technology",
                "Test Central Bank",
            ],
        )
        self.assertEqual(names_4, ["Beta University", "Test Lab"])
        # The alias of the old file is kept and the rank is the new one.
        self.assertEqual(actual["tier_5"][0]["aliases"], ["Alpha"])
        self.assertEqual(actual["tier_5"][0]["qs_rank"], 1)
        # The extra key of a hand entry is kept.
        self.assertEqual(actual["tier_4"][1]["note"], "hand")
        # The hand entry with a QS name became a QS entry.
        self.assertEqual(actual["tier_4"][0]["qs_rank"], 21)

    def test3(self) -> None:
        """
        Test that a tie at the last rank stays in the tier 5 and that a repeated
        name is dropped.
        """
        # Prepare inputs.
        rows = [
            (20, "Gamma University"),
            (20, "Delta University"),
            (20, "Gamma University"),
        ]
        # Run test.
        actual = _build(rows, {})
        # Check outputs.
        names_5 = [entry["name"] for entry in actual["tier_5"]]
        self.assertEqual(names_5, ["Gamma University", "Delta University"])
        self.assertEqual(actual["tier_4"], [])

    def test4(self) -> None:
        """
        Test that the report script reads the file and matches a paper name.
        """
        # Prepare inputs.
        actual = _build(_ROWS, {})
        tiers = repopapers._parse_tiers(actual)
        # Run test.
        match = repopapers._match_tier("Test Institute of Technology", tiers)
        short_match = repopapers._match_tier("TIT, Department of Math", tiers)
        # Check outputs.
        self.assertEqual(match, (5, "Test Institute of Technology"))
        self.assertEqual(short_match, (5, "Test Institute of Technology"))
        self.assertEqual(tiers.as_of, "Test ranking, accessed 2026-01-01")
