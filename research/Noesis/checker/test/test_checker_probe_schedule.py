"""
Test secret deterministic scheduling across contract evaluation windows.
"""

import datetime
from typing import Tuple

import helpers.hunit_test as hunitest
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_probe_schedule as rnochprsc


def _canaries() -> Tuple[rnochmod.Canary, ...]:
    return tuple(
        rnochmod.Canary(
            "%s-v1-%03d" % (family, number),
            "v1",
            family,
            "Question %s %s" % (family, number),
            "1",
            "integer",
        )
        for family in (
            "arithmetic",
            "code",
            "logic",
            "ordering",
            "string",
            "table",
        )
        for number in range(1, 3)
    )


def _contract(
    *,
    contract_id: int = 42,
    target_canaries: int = 10,
    maximum_seconds: int = 60,
    state: str = "ACTIVE",
    window_start: datetime.datetime = datetime.datetime(
        2026, 1, 1, tzinfo=datetime.timezone.utc
    ),
) -> rnochmod.ContractWindow:
    return rnochmod.ContractWindow(
        terms=rnochmod.ContractTerms(contract_id, 2.0, 0.9, 0.85),
        state=state,
        target_requests=20,
        target_canaries=target_canaries,
        maximum_seconds=maximum_seconds,
        checker_policy="m2-v1",
        window_start=window_start,
        window_end=None,
        seller_id="seller_1",
    )


# #############################################################################
# Test_build_probe_schedule
# #############################################################################


class Test_build_probe_schedule(hunitest.TestCase):
    """
    Test selection and timing across one active contract window.
    """

    def test1(self) -> None:
        """
        Test ten unique, balanced probes occupy ten separate time buckets.
        """
        contract = _contract()
        canaries = _canaries()
        schedule = rnochprsc.build_probe_schedule(
            contract, canaries, "schedule-secret"
        )
        self.assertEqual(
            [item.probe_slot for item in schedule], list(range(1, 11))
        )
        self.assertEqual(len({item.canary_id for item in schedule}), 10)
        families = {canary.canary_id: canary.family for canary in canaries}
        counts = {
            family: sum(families[item.canary_id] == family for item in schedule)
            for family in set(families.values())
        }
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        bucket_seconds = 6
        for index, item in enumerate(schedule):
            offset = (item.due_at - contract.window_start).total_seconds()
            self.assertGreaterEqual(offset, index * bucket_seconds + 1.2)
            self.assertLess(offset, index * bucket_seconds + 4.8)
        deadline = contract.window_start + datetime.timedelta(seconds=60)
        self.assertLess(schedule[-1].due_at, deadline)

    def test2(self) -> None:
        """
        Test one secret reproduces a schedule and another changes it.
        """
        contract = _contract()
        canaries = _canaries()
        first = rnochprsc.build_probe_schedule(contract, canaries, "secret-a")
        repeated = rnochprsc.build_probe_schedule(contract, canaries, "secret-a")
        different = rnochprsc.build_probe_schedule(contract, canaries, "secret-b")
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)

    def test3(self) -> None:
        """
        Test inactive contracts are rejected before scheduling.
        """
        with self.assertRaisesRegex(
            rnochprsc.ProbeScheduleError, "must be ACTIVE"
        ):
            rnochprsc.build_probe_schedule(
                _contract(state="PASSED"), _canaries(), "secret"
            )


# #############################################################################
# Test_get_due_probes
# #############################################################################


class Test_get_due_probes(hunitest.TestCase):
    """
    Test time and claimed-slot filtering without database access.
    """

    def test1(self) -> None:
        """
        Test only elapsed and unclaimed slots become due.
        """
        contract = _contract(target_canaries=3)
        schedule = rnochprsc.build_probe_schedule(contract, _canaries(), "secret")
        before_first = schedule[0].due_at - datetime.timedelta(microseconds=1)
        self.assertEqual(
            rnochprsc.get_due_probes(schedule, before_first),
            (),
        )
        due = rnochprsc.get_due_probes(
            schedule,
            schedule[1].due_at,
            claimed_slots=(1,),
        )
        self.assertEqual([item.probe_slot for item in due], [2])
        after_window = contract.window_start + datetime.timedelta(seconds=61)
        due = rnochprsc.get_due_probes(
            schedule,
            after_window,
            claimed_slots=(1, 3),
        )
        self.assertEqual([item.probe_slot for item in due], [2])
