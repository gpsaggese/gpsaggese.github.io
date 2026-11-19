"""
Test offline replay execution through the production checker logic.
"""

import helpers.hunit_test as hunitest
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_replay as rnochrep
import research.Noesis.checker.checker_replay_runner as rnochrer
import research.Noesis.checker.checker_verdicts as rnochver

# #############################################################################
# Test_evaluate_replay_contract
# #############################################################################


class Test_evaluate_replay_contract(hunitest.TestCase):
    """
    Test one contract through evidence, readiness, and verdict production.
    """

    def test1(self) -> None:
        """
        Test target-reaching all-success evidence produces a passing verdict.
        """
        # Prepare inputs.
        scenario = rnochrep.ReplayScenario("all_success", 50, 10, 1.0, 1.0)
        generator = rnochrep.ReplayObservationGenerator(scenario, 0)
        # Run test.
        actual = rnochrer.evaluate_replay_contract(scenario, generator, 1)
        # Check outputs.
        self.assertEqual(
            actual.readiness.reason,
            rnochmod.WindowCloseReason.TARGETS_REACHED,
        )
        self.assertEqual(actual.decision.verdict, rnochmod.Verdict.PASSED)
        self.assertEqual(actual.decision.latency.total, 50)
        self.assertEqual(actual.decision.quality.total, 10)

    def test2(self) -> None:
        """
        Test sparse evidence closes at the deadline as insufficient.
        """
        # Prepare inputs.
        scenario = rnochrep.ReplayScenario("sparse", 19, 7, 1.0, 1.0)
        generator = rnochrep.ReplayObservationGenerator(scenario, 0)
        # Run test.
        actual = rnochrer.evaluate_replay_contract(scenario, generator, 1)
        # Check outputs.
        self.assertEqual(
            actual.readiness.reason,
            rnochmod.WindowCloseReason.DEADLINE_REACHED,
        )
        self.assertEqual(
            actual.decision.verdict,
            rnochmod.Verdict.INSUFFICIENT,
        )
        self.assertFalse(actual.decision.latency.sufficient)
        self.assertFalse(actual.decision.quality.sufficient)

    def test3(self) -> None:
        """
        Test latency and quality degradation remain separately attributable.
        """
        # Prepare inputs.
        scenarios = (
            rnochrep.ReplayScenario("slow", 50, 10, 0.0, 1.0),
            rnochrep.ReplayScenario("weak", 50, 10, 1.0, 0.0),
        )
        expected_reasons = (
            (rnochver.LATENCY_BELOW_PROMISE,),
            (rnochver.QUALITY_BELOW_PROMISE,),
        )
        # Run test.
        actual = [
            rnochrer.evaluate_replay_contract(
                scenario,
                rnochrep.ReplayObservationGenerator(scenario, 0),
                index,
            )
            for index, scenario in enumerate(scenarios, start=1)
        ]
        # Check outputs.
        self.assert_equal(
            str([outcome.decision.reason_codes for outcome in actual]),
            str(list(expected_reasons)),
        )


# #############################################################################
# Test_analytic_verdict_rates
# #############################################################################


class Test_analytic_verdict_rates(hunitest.TestCase):
    """
    Test analytic probabilities derived from the production confidence rule.
    """

    def test1(self) -> None:
        """
        Test the documented weak-model detection probability.
        """
        # Prepare inputs.
        scenario = rnochrep.ReplayScenario("weak", 50, 10, 1.0, 0.5)
        # Run test.
        actual = rnochrer.analytic_verdict_rates(scenario)
        # Check outputs.
        self.assertAlmostEqual(actual.quality_failed, 0.828125)
        self.assertAlmostEqual(actual.latency_failed, 0.0)
        self.assertAlmostEqual(actual.failed, 0.828125)

    def test2(self) -> None:
        """
        Test verdict probabilities form a complete distribution.
        """
        for scenario in rnochrep.DEFAULT_REPLAY_SCENARIOS:
            # Run test.
            actual = rnochrer.analytic_verdict_rates(scenario)
            # Check outputs.
            self.assertAlmostEqual(
                actual.passed + actual.failed + actual.insufficient,
                1.0,
            )

    def test3(self) -> None:
        """
        Test sparse all-success evidence is analytically insufficient.
        """
        # Prepare inputs.
        scenario = rnochrep.ReplayScenario("sparse", 19, 7, 1.0, 1.0)
        # Run test.
        actual = rnochrer.analytic_verdict_rates(scenario)
        # Check outputs.
        self.assertEqual(actual.passed, 0.0)
        self.assertEqual(actual.failed, 0.0)
        self.assertEqual(actual.insufficient, 1.0)


# #############################################################################
# Test_run_replay_scenario
# #############################################################################


class Test_run_replay_scenario(hunitest.TestCase):
    """
    Test deterministic aggregation over repeated synthetic contracts.
    """

    def test1(self) -> None:
        """
        Test identical scenario and seed produce identical summaries.
        """
        # Prepare inputs.
        scenario = rnochrep.ReplayScenario("sample", 50, 10, 0.85, 0.8)
        # Run test.
        left = rnochrer.run_replay_scenario(scenario, 4, 100)
        right = rnochrer.run_replay_scenario(scenario, 4, 100)
        # Check outputs.
        self.assert_equal(str(left), str(right))

    def test2(self) -> None:
        """
        Test all slow contracts fail only their latency promise.
        """
        # Prepare inputs.
        scenario = rnochrep.ReplayScenario("slow", 50, 10, 0.0, 1.0)
        # Run test.
        actual = rnochrer.run_replay_scenario(scenario, 0, 25)
        # Check outputs.
        self.assertEqual(actual.verdict_counts["FAILED"], 25)
        self.assertEqual(actual.latency_failure_count, 25)
        self.assertEqual(actual.quality_failure_count, 0)
        self.assertEqual(actual.both_failure_count, 0)
        self.assertEqual(actual.observed_failure_rate, 1.0)

    def test3(self) -> None:
        """
        Test insufficient evidence remains safe across repeated contracts.
        """
        # Prepare inputs.
        scenario = rnochrep.ReplayScenario("sparse", 19, 7, 1.0, 1.0)
        # Run test.
        actual = rnochrer.run_replay_scenario(scenario, 0, 25)
        # Check outputs.
        self.assertEqual(actual.verdict_counts["INSUFFICIENT"], 25)
        self.assertEqual(actual.verdict_counts["FAILED"], 0)
        self.assertEqual(actual.observed_failure_rate, 0.0)

    def test4(self) -> None:
        """
        Test 1,000 weak-model contracts match the analytic prediction.
        """
        # Prepare inputs.
        scenario = rnochrep.ReplayScenario("weak", 50, 10, 1.0, 0.5)
        # Run test.
        actual = rnochrer.run_replay_scenario(scenario, 0, 1000)
        # Check outputs.
        self.assertTrue(rnochrer.agrees_with_analytic_prediction(actual))
        self.assertAlmostEqual(actual.analytic_rates.failed, 0.828125)
