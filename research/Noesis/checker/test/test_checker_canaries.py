"""
Test canary-bank validation and secret deterministic selection.
"""

import collections
import json
import os

import helpers.hunit_test as hunitest

import research.Noesis.checker.checker_canaries as rnochcan
import research.Noesis.checker.checker_grading as rnochgra
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.scripts.generate_canary_bank as rnscaba

_NOESIS_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_PRODUCTION_BANK_DIR = os.path.join(_NOESIS_DIR, "data", "canaries")


def _canary(identifier: str, family: str = "math") -> rnochmod.Canary:
    return rnochmod.Canary(
        identifier,
        "v1",
        family,
        "Question %s" % identifier,
        "1",
        "integer",
    )


# #############################################################################
# Test_load_canary_bank
# #############################################################################


class Test_load_canary_bank(hunitest.TestCase):
    """
    Test JSONL bank loading and actionable errors.
    """

    def test1(self) -> None:
        """
        Test two valid records load in file order.
        """
        records = [
            {
                "id": "math-v1-001",
                "bank_version": "v1",
                "family": "math",
                "prompt": "What is zero plus one?",
                "answer": "1",
                "normalizer": "integer",
            },
            {
                "id": "logic-v1-001",
                "bank_version": "v1",
                "family": "logic",
                "prompt": "Return true.",
                "answer": "true",
                "normalizer": "alnum_lower",
            },
        ]
        path = os.path.join(self.get_scratch_space(), "bank.jsonl")
        with open(path, "w", encoding="utf-8") as file_:
            file_.write("\n".join(json.dumps(record) for record in records))
        actual = rnochcan.load_canary_bank(self.get_scratch_space(), "v1")
        self.assertEqual(
            [canary.canary_id for canary in actual],
            ["math-v1-001", "logic-v1-001"],
        )

    def test2(self) -> None:
        """
        Test malformed records report their source line.
        """
        path = os.path.join(self.get_scratch_space(), "bank.jsonl")
        with open(path, "w", encoding="utf-8") as file_:
            file_.write("{}\nnot-json\n")
        with self.assertRaisesRegex(rnochcan.CanaryBankError, "bank.jsonl:1"):
            rnochcan.load_canary_bank(self.get_scratch_space(), "v1")


# #############################################################################
# Test_production_canary_bank
# #############################################################################


class Test_production_canary_bank(hunitest.TestCase):
    """
    Test the committed v1 candidate bank is complete and reproducible.
    """

    def test1(self) -> None:
        """
        Test the bank has 300 valid, balanced, self-grading candidates.
        """
        canaries = rnochcan.load_canary_bank(_PRODUCTION_BANK_DIR, "v1")
        family_counts = collections.Counter(canary.family for canary in canaries)
        self.assertEqual(len(canaries), 300)
        self.assertEqual(
            family_counts,
            {
                "arithmetic": 50,
                "code-trace": 50,
                "modular": 50,
                "ordering": 50,
                "string": 50,
                "table": 50,
            },
        )
        grades = [
            rnochgra.grade_response(canary, canary.answer) for canary in canaries
        ]
        self.assertTrue(all(grade.correct for grade in grades))

    def test2(self) -> None:
        """
        Test committed records exactly match the deterministic generator.
        """
        actual = rnochcan.load_canary_bank(_PRODUCTION_BANK_DIR, "v1")
        generated_records = [
            record
            for records in rnscaba.build_bank().values()
            for record in records
        ]
        expected = tuple(
            rnochcan.parse_canary(record) for record in generated_records
        )
        self.assertEqual(actual, expected)


# #############################################################################
# Test_select_canaries
# #############################################################################


class Test_select_canaries(hunitest.TestCase):
    """
    Test selection is stable, secret, unique, and family-balanced.
    """

    def test1(self) -> None:
        """
        Test repeatability, uniqueness, and equal family representation.
        """
        bank = tuple(
            _canary("%s-v1-%03d" % (family, number), family)
            for family in ("math", "logic", "code")
            for number in range(5)
        )
        first = rnochcan.select_canaries(bank, 42, 9, "private-secret")
        second = rnochcan.select_canaries(bank, 42, 9, "private-secret")
        self.assertEqual(first, second)
        self.assertEqual(
            [item.canary_id for item in first],
            [
                "math-v1-002",
                "code-v1-004",
                "logic-v1-000",
                "math-v1-004",
                "code-v1-000",
                "logic-v1-004",
                "math-v1-001",
                "code-v1-002",
                "logic-v1-001",
            ],
        )
        self.assertEqual(len({item.canary_id for item in first}), 9)
        counts = {
            family: sum(item.family == family for item in first)
            for family in ("math", "logic", "code")
        }
        self.assertEqual(counts, {"math": 3, "logic": 3, "code": 3})

    def test2(self) -> None:
        """
        Test a different contract receives a different selection.
        """
        bank = tuple(_canary("math-v1-%03d" % i) for i in range(20))
        first = rnochcan.select_canaries(bank, 1, 5, "private-secret")
        second = rnochcan.select_canaries(bank, 2, 5, "private-secret")
        self.assertNotEqual(first, second)

    def test3(self) -> None:
        """
        Test an opaque monitoring key is repeatable and contract-independent.
        """
        bank = tuple(_canary("math-v1-%03d" % i) for i in range(20))
        first = rnochcan.select_canaries_for_key(
            bank, "monitoring-run:7", 5, "private-secret"
        )
        second = rnochcan.select_canaries_for_key(
            bank, "monitoring-run:7", 5, "private-secret"
        )
        contract = rnochcan.select_canaries(bank, 7, 5, "private-secret")
        self.assertEqual(first, second)
        self.assertNotEqual(first, contract)
