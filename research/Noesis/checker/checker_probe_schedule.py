"""
Build secret, deterministic probe schedules for active contract windows.

Import as:

import research.Noesis.checker.checker_probe_schedule as rnochprsc
"""

import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Iterable, Sequence, Tuple

import research.Noesis.checker.checker_canaries as rnochcan
import research.Noesis.checker.checker_models as rnochmod

_MIN_BUCKET_POSITION = 0.2
_BUCKET_POSITION_WIDTH = 0.6


class ProbeScheduleError(ValueError):
    """
    Signal an invalid or unsatisfiable probe schedule.
    """


def _secret_bucket_position(
    secret: str,
    contract_id: int,
    probe_slot: int,
) -> float:
    """
    Return a secret position in the middle 60% of one time bucket.
    """
    if not secret:
        raise ProbeScheduleError("schedule secret must not be empty")
    message = "probe-schedule:%s:%s" % (contract_id, probe_slot)
    digest = hmac.new(secret.encode(), message.encode(), hashlib.sha256).digest()
    unit_interval = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return _MIN_BUCKET_POSITION + _BUCKET_POSITION_WIDTH * unit_interval


def build_probe_schedule(
    contract: rnochmod.ContractWindow,
    canaries: Sequence[rnochmod.Canary],
    secret: str,
) -> Tuple[rnochmod.ProbeScheduleItem, ...]:
    """
    Select canaries and spread them across an active contract window.

    The window is divided into one equal bucket per probe. One secret,
    deterministic timestamp is selected from the middle 60% of each bucket.
    This avoids start/end bursts while keeping schedules replayable.
    """
    if contract.state != "ACTIVE":
        raise ProbeScheduleError("contract must be ACTIVE")
    if contract.window_start is None:
        raise ProbeScheduleError("contract window_start is required")
    if contract.window_start.utcoffset() is None:
        raise ProbeScheduleError("contract window_start must be timezone-aware")
    if contract.target_canaries < 1:
        raise ProbeScheduleError("target_canaries must be positive")
    if contract.maximum_seconds < 1:
        raise ProbeScheduleError("maximum_seconds must be positive")
    try:
        selected = rnochcan.select_canaries(
            canaries,
            contract.terms.contract_id,
            contract.target_canaries,
            secret,
        )
    except rnochcan.CanaryBankError as error:
        raise ProbeScheduleError(str(error)) from error
    bucket_seconds = contract.maximum_seconds / contract.target_canaries
    schedule = []
    for probe_slot, canary in enumerate(selected, start=1):
        bucket_position = _secret_bucket_position(
            secret, contract.terms.contract_id, probe_slot
        )
        offset_seconds = (probe_slot - 1 + bucket_position) * bucket_seconds
        schedule.append(
            rnochmod.ProbeScheduleItem(
                contract_id=contract.terms.contract_id,
                probe_slot=probe_slot,
                canary_id=canary.canary_id,
                due_at=contract.window_start + timedelta(seconds=offset_seconds),
            )
        )
    return tuple(schedule)


def get_due_probes(
    schedule: Sequence[rnochmod.ProbeScheduleItem],
    now: datetime,
    claimed_slots: Iterable[int] = (),
) -> Tuple[rnochmod.ProbeScheduleItem, ...]:
    """
    Return due, unclaimed probes in stable chronological order.
    """
    if now.utcoffset() is None:
        raise ProbeScheduleError("now must be timezone-aware")
    claimed = set(claimed_slots)
    return tuple(
        item
        for item in sorted(
            schedule,
            key=lambda candidate: (candidate.due_at, candidate.probe_slot),
        )
        if item.due_at <= now and item.probe_slot not in claimed
    )
