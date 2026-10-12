"""
Implement deterministic canary normalizers and exact-match grading.

Import as:

import research.Noesis.checker_grading as rnochgra
"""

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Callable, Dict, FrozenSet, Optional

import research.Noesis.checker_models as rnochmod

GRADER_VERSION = "m2-v1"

MATCH = "MATCH"
VALUE_MISMATCH = "VALUE_MISMATCH"
EMPTY_RESPONSE = "EMPTY_RESPONSE"
INVALID_RESPONSE = "INVALID_RESPONSE"
INVALID_EXPECTED_ANSWER = "INVALID_EXPECTED_ANSWER"
UNKNOWN_NORMALIZER = "UNKNOWN_NORMALIZER"


class NormalizationError(ValueError):
    """
    Signal that a value does not match a normalizer's expected shape.
    """


def normalize_exact(value: str) -> str:
    """
    Trim surrounding whitespace and preserve everything else.
    """
    return value.strip()


def normalize_alnum_lower(value: str) -> str:
    """
    Ignore case, whitespace, and punctuation.
    """
    return "".join(character.lower() for character in value if character.isalnum())


def normalize_integer(value: str) -> str:
    """
    Canonicalize a response containing only one integer value.
    """
    candidate = value.strip().replace(",", "")
    if not re.fullmatch(r"[+-]?\d+(?:\.0+)?", candidate):
        raise NormalizationError("expected one integer")
    try:
        number = Decimal(candidate)
    except InvalidOperation as error:
        raise NormalizationError("invalid integer") from error
    if number != number.to_integral_value():
        raise NormalizationError("expected one integer")
    return str(int(number))


def normalize_decimal(value: str) -> str:
    """
    Canonicalize a response containing only one finite decimal value.
    """
    candidate = value.strip().replace(",", "")
    try:
        number = Decimal(candidate)
    except InvalidOperation as error:
        raise NormalizationError("invalid decimal") from error
    if not number.is_finite():
        raise NormalizationError("decimal must be finite")
    if number == 0:
        return "0"
    return format(number.normalize(), "f")


def _reject_json_constant(value: str) -> None:
    raise NormalizationError("invalid JSON constant %r" % value)


def normalize_json(value: str) -> str:
    """
    Parse JSON and serialize it with stable key ordering and spacing.
    """
    try:
        parsed = json.loads(value, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, TypeError) as error:
        raise NormalizationError("invalid JSON") from error
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"))


def normalize_code_output(value: str) -> str:
    """
    Normalize line endings and trailing whitespace in program output.
    """
    lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


_NORMALIZERS: Dict[str, Callable[[str], str]] = {
    "exact": normalize_exact,
    "alnum_lower": normalize_alnum_lower,
    "integer": normalize_integer,
    "decimal": normalize_decimal,
    "json_canonical": normalize_json,
    "code_output": normalize_code_output,
}


def normalizer_names() -> FrozenSet[str]:
    """
    Return the supported normalizer identifiers.
    """
    return frozenset(_NORMALIZERS)


def get_normalizer(name: str) -> Callable[[str], str]:
    """
    Resolve a configured normalizer.
    """
    try:
        return _NORMALIZERS[name]
    except KeyError as error:
        raise KeyError("unknown normalizer %r" % name) from error


def grade_response(
    canary: rnochmod.Canary, response: Optional[str]
) -> rnochmod.GradeResult:
    """
    Normalize the expected and actual values, then compare exactly.
    """
    try:
        normalizer = get_normalizer(canary.normalizer)
    except KeyError:
        return rnochmod.GradeResult(
            None, UNKNOWN_NORMALIZER, None, None, GRADER_VERSION
        )
    try:
        expected = normalizer(canary.answer)
    except NormalizationError:
        return rnochmod.GradeResult(
            None, INVALID_EXPECTED_ANSWER, None, None, GRADER_VERSION
        )
    if not expected:
        return rnochmod.GradeResult(
            None, INVALID_EXPECTED_ANSWER, None, None, GRADER_VERSION
        )
    if response is None or not response.strip():
        return rnochmod.GradeResult(
            False, EMPTY_RESPONSE, expected, None, GRADER_VERSION
        )
    try:
        actual = normalizer(response)
    except NormalizationError:
        return rnochmod.GradeResult(
            False, INVALID_RESPONSE, expected, None, GRADER_VERSION
        )
    correct = actual == expected
    reason = MATCH if correct else VALUE_MISMATCH
    return rnochmod.GradeResult(correct, reason, expected, actual, GRADER_VERSION)
