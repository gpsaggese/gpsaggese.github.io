"""
Test deterministic canary grading.
"""

from typing import List, Optional

import helpers.hunit_test as hunitest

import research.Noesis.checker_grading as rnochgra
import research.Noesis.checker_models as rnochmod


def _canary(answer: str, normalizer: str) -> rnochmod.Canary:
    return rnochmod.Canary(
        "test-canary", "v1", "test", "Return the answer.", answer, normalizer
    )


# #############################################################################
# Test_grade_response
# #############################################################################


class Test_grade_response(hunitest.TestCase):
    """
    Test normalized exact-match grading and ownership of failures.
    """

    def test1(self) -> None:
        """
        Test equivalent answers accepted by every non-exact normalizer.
        """
        cases = [
            ("alnum_lower", "Canberra.", " canBERRA "),
            ("integer", "1,000", "+1000.0"),
            ("decimal", "001.2300", "1.23"),
            ("json_canonical", '{"a":1,"b":[2]}', '{"b":[2],"a":1}'),
            ("code_output", "a  \r\nb\n", "a\nb"),
        ]
        for normalizer, expected, response in cases:
            result = rnochgra.grade_response(_canary(expected, normalizer), response)
            self.assertTrue(result.correct)
            self.assertEqual(result.reason, rnochgra.MATCH)

    def test2(self) -> None:
        """
        Test empty and malformed responses are seller failures.
        """
        responses: List[Optional[str]] = [None, "", "about seven"]
        reasons = [
            rnochgra.EMPTY_RESPONSE,
            rnochgra.EMPTY_RESPONSE,
            rnochgra.INVALID_RESPONSE,
        ]
        for response, reason in zip(responses, reasons):
            result = rnochgra.grade_response(_canary("7", "integer"), response)
            self.assertFalse(result.correct)
            self.assertEqual(result.reason, reason)

    def test3(self) -> None:
        """
        Test bad expected data and unknown graders are checker errors.
        """
        cases = [
            (_canary("seven", "integer"), rnochgra.INVALID_EXPECTED_ANSWER),
            (_canary("...", "alnum_lower"), rnochgra.INVALID_EXPECTED_ANSWER),
            (_canary("7", "mystery"), rnochgra.UNKNOWN_NORMALIZER),
        ]
        for canary, reason in cases:
            result = rnochgra.grade_response(canary, "7")
            self.assertIsNone(result.correct)
            self.assertEqual(result.reason, reason)


# #############################################################################
# Test_normalize_decimal
# #############################################################################


class Test_normalize_decimal(hunitest.TestCase):
    """
    Test unsafe numeric values are rejected.
    """

    def test1(self) -> None:
        """
        Test all non-finite decimal spellings.
        """
        for value in ["NaN", "Infinity", "-Infinity"]:
            with self.assertRaises(rnochgra.NormalizationError):
                rnochgra.normalize_decimal(value)


class Test_normalize_json(hunitest.TestCase):
    """
    Test strict JSON parsing.
    """

    def test1(self) -> None:
        """
        Test non-standard JSON constants are rejected.
        """
        with self.assertRaises(rnochgra.NormalizationError):
            rnochgra.normalize_json('{"value": NaN}')
