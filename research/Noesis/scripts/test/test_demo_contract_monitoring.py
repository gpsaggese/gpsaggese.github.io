"""
Test the deterministic contract-monitoring demo.

Import as:

import research.Noesis.scripts.test.test_demo_contract_monitoring as rnsttdcm
"""

import asyncio

import helpers.hunit_test as hunitest

import research.Noesis.scripts.demo_contract_monitoring as rnscdemo


class Test_build_report(hunitest.TestCase):
    """
    Test `demo_contract_monitoring.build_report()`.
    """

    def test1(self) -> None:
        """
        Test all four monitoring outcomes and the sink publication policy.
        """
        expected = """
        1) Seeded 4 ACTIVE contracts with buyer evidence.
        2) Prober persisted 7 scheduled canaries before checker evaluation.
        3) Checker evaluated 4 ready windows:
           contract=1 seller=healthy_seller close=TARGETS_REACHED buyer=2 canary=2 verdict=PASSED published=yes
           contract=2 seller=slow_seller close=TARGETS_REACHED buyer=2 canary=2 verdict=FAILED published=yes
           contract=3 seller=weak_seller close=TARGETS_REACHED buyer=2 canary=2 verdict=FAILED published=yes
           contract=4 seller=sparse_seller close=DEADLINE_REACHED buyer=0 canary=1 verdict=INSUFFICIENT published=no
        4) Persistence boundary finalized 4 contracts exactly once.
        5) RecordingVerdictSink received 3 actionable reports: 1:PASSED, 2:FAILED, 3:FAILED
           Contract 4 stayed sink-silent because INSUFFICIENT is non-actionable.
        """
        actual = asyncio.run(rnscdemo.build_report())
        self.assert_equal(actual, expected, dedent=True)
