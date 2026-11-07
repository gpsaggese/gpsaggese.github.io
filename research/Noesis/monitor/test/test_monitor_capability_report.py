"""
Test reproducible capability-drift reports.
"""

import dataclasses
import os

import research.Noesis.monitor.monitor_capability_experiment as rnmoncaex
import research.Noesis.monitor.monitor_capability_report as rnmoncare

import helpers.hunit_test as hunitest

# #############################################################################
# Test_write_capability_artifacts
# #############################################################################


class Test_write_capability_artifacts(hunitest.TestCase):
    """
    Test the review-artifact contract.
    """

    def test1(self) -> None:
        """
        Test a small experiment writes every deterministic artifact.
        """
        protocol = rnmoncaex.CapabilityExperimentProtocol(trials_per_scenario=2)
        scenarios = (
            rnmoncaex.DEFAULT_CAPABILITY_SCENARIOS[0],
            rnmoncaex.DEFAULT_CAPABILITY_SCENARIOS[1],
        )
        result = rnmoncaex.run_capability_experiment(protocol, scenarios)
        power = rnmoncaex.run_capability_power_analysis(
            protocol,
            1,
            window_sizes=(60,),
            baseline_window_counts=(5,),
        )
        output_dir = self.get_scratch_space()
        paths = rnmoncare.write_capability_artifacts(result, power, output_dir)
        for path in dataclasses.astuple(paths):
            self.assertTrue(os.path.isfile(path), path)
        with open(paths.summary_path, encoding="utf-8") as file_:
            summary = file_.read()
        self.assertIn('"schema_version"', summary)
        self.assertIn('"full_weak_model_swap"', summary)
