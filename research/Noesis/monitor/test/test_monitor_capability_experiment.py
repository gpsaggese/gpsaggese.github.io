"""
Test deterministic, blinded capability-drift experiments.
"""

import research.Noesis.monitor.monitor_capability_experiment as rnmoncaex

import helpers.hunit_test as hunitest

# #############################################################################
# Test_generate_capability_trial
# #############################################################################


class Test_generate_capability_trial(hunitest.TestCase):
    """
    Test raw observation generation and truth separation.
    """

    def test1(self) -> None:
        """
        Test identical protocol, scenario, and seed reproduce every row.
        """
        protocol = rnmoncaex.CapabilityExperimentProtocol(trials_per_scenario=1)
        scenario = rnmoncaex.DEFAULT_CAPABILITY_SCENARIOS[1]
        first = rnmoncaex.generate_capability_trial(protocol, scenario, 7)
        second = rnmoncaex.generate_capability_trial(protocol, scenario, 7)
        self.assertEqual(first, second)
        self.assertEqual(first.truth.scenario_name, scenario.name)
        self.assertEqual(len(first.baseline_windows), 5)
        self.assertEqual(len(first.post_windows), 15)


# #############################################################################
# Test_run_capability_experiment
# #############################################################################


class Test_run_capability_experiment(hunitest.TestCase):
    """
    Test end-to-end detector evaluation on planted controls.
    """

    def test1(self) -> None:
        """
        Test healthy and full-swap controls separate in a seeded rehearsal.
        """
        protocol = rnmoncaex.CapabilityExperimentProtocol(trials_per_scenario=20)
        scenarios = (
            rnmoncaex.DEFAULT_CAPABILITY_SCENARIOS[0],
            rnmoncaex.DEFAULT_CAPABILITY_SCENARIOS[1],
        )
        actual = rnmoncaex.run_capability_experiment(protocol, scenarios)
        self.assertTrue(actual.passed)
        self.assertEqual(actual.scenarios[0].detection_rate, 0.0)
        self.assertGreaterEqual(actual.scenarios[1].detection_rate, 0.8)

    def test2(self) -> None:
        """
        Test operational failures remain insufficient, not degraded.
        """
        protocol = rnmoncaex.CapabilityExperimentProtocol(trials_per_scenario=2)
        scenario = rnmoncaex.DEFAULT_CAPABILITY_SCENARIOS[-2]
        actual = rnmoncaex.run_capability_experiment(protocol, (scenario,))
        self.assertTrue(actual.passed)
        self.assertEqual(actual.scenarios[0].detection_rate, 0.0)
        self.assertEqual(actual.scenarios[0].insufficient_rate, 1.0)
