"""
Load, validate, and select from a versioned canary bank.

Import as:

import research.Noesis.checker_canaries as rnochcan
"""

import collections
import glob
import hashlib
import hmac
import json
import os
import random
import re
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import research.Noesis.checker_grading as rnochgra
import research.Noesis.checker_models as rnochmod

_CANARY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")


class CanaryBankError(ValueError):
    """
    Signal a malformed or unsatisfiable canary bank.
    """


def _required_text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CanaryBankError("%r must be a non-empty string" % field)
    return value.strip()


def parse_canary(record: Mapping[str, object]) -> rnochmod.Canary:
    """
    Parse one JSON-compatible record into an immutable canary.
    """
    seed = record.get("generator_seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise CanaryBankError("'generator_seed' must be an integer or null")
    return rnochmod.Canary(
        canary_id=_required_text(record, "id"),
        bank_version=_required_text(record, "bank_version"),
        family=_required_text(record, "family"),
        prompt=_required_text(record, "prompt"),
        answer=_required_text(record, "answer"),
        normalizer=_required_text(record, "normalizer"),
        generator_seed=seed,
    )


def validate_canary(canary: rnochmod.Canary) -> Tuple[str, ...]:
    """
    Return every validation error for one canary.
    """
    errors = []
    if not _CANARY_ID_RE.fullmatch(canary.canary_id):
        errors.append("id must be 3-80 lowercase URL-safe characters")
    if not canary.bank_version:
        errors.append("bank_version is required")
    if not canary.family:
        errors.append("family is required")
    if not canary.prompt.strip():
        errors.append("prompt is required")
    if not canary.answer.strip():
        errors.append("answer is required")
    if canary.normalizer not in rnochgra.normalizer_names():
        errors.append("unknown normalizer %r" % canary.normalizer)
    return tuple(errors)


def validate_bank(canaries: Iterable[rnochmod.Canary]) -> Tuple[str, ...]:
    """
    Return record, duplicate-id, and cross-version errors.
    """
    errors = []
    seen = set()
    versions = set()
    count = 0
    for canary in canaries:
        count += 1
        versions.add(canary.bank_version)
        if canary.canary_id in seen:
            errors.append("duplicate canary id %r" % canary.canary_id)
        seen.add(canary.canary_id)
        errors.extend(
            "%s: %s" % (canary.canary_id, message)
            for message in validate_canary(canary)
        )
    if count == 0:
        errors.append("canary bank is empty")
    if len(versions) > 1:
        errors.append("canary bank mixes versions: %r" % sorted(versions))
    return tuple(errors)


def load_canary_bank(directory: str, bank_version: str) -> Tuple[rnochmod.Canary, ...]:
    """
    Load and validate every JSONL file in a bank directory.
    """
    canaries = []
    paths = sorted(glob.glob(os.path.join(directory, "*.jsonl")))
    if not paths:
        raise CanaryBankError("no JSONL files found in %s" % directory)
    for path in paths:
        with open(path, encoding="utf-8") as file_:  # pylint: disable=W1514
            lines = file_.read().splitlines()
        for line_number, raw_line in enumerate(lines, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise CanaryBankError(
                    "%s:%s: invalid JSON: %s" % (path, line_number, error.msg)
                ) from error
            if not isinstance(record, dict):
                raise CanaryBankError(
                    "%s:%s: each line must be a JSON object" % (path, line_number)
                )
            try:
                canary = parse_canary(record)
            except CanaryBankError as error:
                raise CanaryBankError(
                    "%s:%s: %s" % (path, line_number, error)
                ) from error
            if canary.bank_version != bank_version:
                raise CanaryBankError(
                    "%s:%s: expected bank version %r, found %r"
                    % (
                        path,
                        line_number,
                        bank_version,
                        canary.bank_version,
                    )
                )
            canaries.append(canary)
    errors = validate_bank(canaries)
    if errors:
        raise CanaryBankError("; ".join(errors))
    return tuple(canaries)


def group_by_family(
    canaries: Iterable[rnochmod.Canary],
) -> Dict[str, Tuple[rnochmod.Canary, ...]]:
    """
    Group canaries with stable family and canary ordering.
    """
    groups = collections.defaultdict(list)
    for canary in sorted(canaries, key=lambda item: item.canary_id):
        groups[canary.family].append(canary)
    return {family: tuple(groups[family]) for family in sorted(groups)}


def _selection_seed(secret: str, contract_id: int) -> int:
    if not secret:
        raise CanaryBankError("selection secret must not be empty")
    if contract_id <= 0:
        raise CanaryBankError("contract_id must be positive")
    digest = hmac.new(
        secret.encode(), str(contract_id).encode(), hashlib.sha256
    ).digest()
    return int.from_bytes(digest[:8], "big")


def select_canaries(
    canaries: Sequence[rnochmod.Canary],
    contract_id: int,
    count: int,
    secret: str,
) -> Tuple[rnochmod.Canary, ...]:
    """
    Select a secret, repeatable, family-balanced contract batch.
    """
    if count < 1:
        raise CanaryBankError("selection count must be positive")
    errors = validate_bank(canaries)
    if errors:
        raise CanaryBankError("; ".join(errors))
    if count > len(canaries):
        raise CanaryBankError(
            "requested %s canaries from a bank of %s" % (count, len(canaries))
        )
    generator = random.Random(_selection_seed(secret, contract_id))
    groups: Dict[str, List[rnochmod.Canary]] = {
        family: list(items) for family, items in group_by_family(canaries).items()
    }
    for items in groups.values():
        generator.shuffle(items)
    families = list(groups)
    generator.shuffle(families)
    selected = []
    while len(selected) < count:
        made_progress = False
        for family in families:
            if groups[family] and len(selected) < count:
                selected.append(groups[family].pop())
                made_progress = True
        if not made_progress:
            break
    return tuple(selected)
