"""
Test deterministic synthetic observation generation for checker replay.
"""

import helpers.hunit_test as hunitest
import research.Noesis.checker.checker_replay as rnochrep
import research.Noesis.gateway.gateway_providers as rnogapro

# #############################################################################
# Test_ReplayScenario
# #############################################################################


class Test_ReplayScenario(hunitest.TestCase):
    """
    Test replay scenario configuration and validation.
    """

    def test1(self) -> None:
        """
        Test the frozen experiment scenario matrix.
        """
        # Prepare outputs.
        expected = (
            "healthy:50:10:0.97:0.98\n"
            "slow_only:50:10:0.6:0.98\n"
            "weak_model_only:50:10:0.97:0.5\n"
            "borderline_quality:50:10:0.97:0.8\n"
            "borderline_latency:50:10:0.85:0.98\n"
            "insufficient_evidence:19:7:1.0:1.0"
        )
        # Run test.
        actual = "\n".join(
            "%s:%s:%s:%s:%s"
            % (
                scenario.name,
                scenario.buyer_requests,
                scenario.canaries,
                scenario.latency_compliance_probability,
                scenario.quality_probability,
            )
            for scenario in rnochrep.DEFAULT_REPLAY_SCENARIOS
        )
        # Check outputs.
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Test invalid counts and probabilities are rejected.
        """
        # Prepare inputs.
        invalid_arguments = (
            ("", 1, 1, 0.5, 0.5),
            ("negative_buyers", -1, 1, 0.5, 0.5),
            ("negative_canaries", 1, -1, 0.5, 0.5),
            ("bad_latency", 1, 1, 1.1, 0.5),
            ("bad_quality", 1, 1, 0.5, -0.1),
        )
        # Run test and check outputs.
        for arguments in invalid_arguments:
            with self.assertRaises(AssertionError):
                rnochrep.ReplayScenario(*arguments)


# #############################################################################
# Test_ReplayObservationGenerator
# #############################################################################


class Test_ReplayObservationGenerator(hunitest.TestCase):
    """
    Test deterministic buyer and canary observation generation.
    """

    def test1(self) -> None:
        """
        Test equal inputs produce equal observations across generators.
        """
        # Prepare inputs.
        scenario = rnochrep.ReplayScenario("sample", 5, 3, 0.6, 0.5)
        left_generator = rnochrep.ReplayObservationGenerator(scenario, 7)
        right_generator = rnochrep.ReplayObservationGenerator(scenario, 7)
        # Run test.
        left = left_generator.generate()
        right = right_generator.generate()
        # Check outputs.
        self.assert_equal(str(left), str(right))

    def test2(self) -> None:
        """
        Test probability extremes and request field separation.
        """
        # Prepare inputs.
        successful = rnochrep.ReplayScenario("successful", 3, 2, 1.0, 1.0)
        failing = rnochrep.ReplayScenario("failing", 3, 2, 0.0, 0.0)
        # Run test.
        successful_rows = rnochrep.ReplayObservationGenerator(
            successful,
            0,
        ).generate()
        failing_rows = rnochrep.ReplayObservationGenerator(failing, 0).generate()
        # Check outputs.
        self.assert_equal(
            str(
                [
                    (
                        row.request_id,
                        row.status,
                        row.latency_ms,
                        row.is_canary,
                        row.canary_correct,
                    )
                    for row in successful_rows
                ]
            ),
            "[(1, 'ok', 1200.0, False, None), "
            "(2, 'ok', 1200.0, False, None), "
            "(3, 'ok', 1200.0, False, None), "
            "(4, 'ok', 1200.0, True, True), "
            "(5, 'ok', 1200.0, True, True)]",
        )
        self.assertTrue(all(row.status == rnogapro.OK for row in failing_rows))
        self.assert_equal(
            str([row.latency_ms for row in failing_rows[:3]]),
            "[1800.0, 1800.0, 1800.0]",
        )
        self.assert_equal(
            str([row.canary_correct for row in failing_rows[3:]]),
            "[False, False]",
        )

    def test3(self) -> None:
        """
        Test repeated generation advances streams and unique request IDs.
        """
        # Prepare inputs.
        scenario = rnochrep.ReplayScenario("sequential", 2, 1, 0.5, 0.5)
        generator = rnochrep.ReplayObservationGenerator(
            scenario,
            11,
            first_request_id=101,
        )
        # Run test.
        first = generator.generate()
        second = generator.generate()
        # Check outputs.
        self.assert_equal(
            str([row.request_id for row in first + second]),
            "[101, 102, 103, 104, 105, 106]",
        )
        self.assertNotEqual(str(first), str(second))

    def test4(self) -> None:
        """
        Test invalid generator settings are rejected.
        """
        # Prepare inputs.
        scenario = rnochrep.ReplayScenario("sample", 1, 1, 0.5, 0.5)
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnochrep.ReplayObservationGenerator(
                scenario,
                0,
                latency_threshold_ms=0,
            )
        with self.assertRaises(AssertionError):
            rnochrep.ReplayObservationGenerator(
                scenario,
                0,
                first_request_id=0,
            )
        with self.assertRaises(AssertionError):
            rnochrep.ReplayObservationGenerator(scenario, True)
