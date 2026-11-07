"""
Test the runnable offline capability-drift experiment.
"""

import os
import sys
from unittest import mock

import research.Noesis.scripts.run_model_drift_capability_experiment as rnscmdce

import helpers.hunit_test as hunitest

# #############################################################################
# Test_main
# #############################################################################


class Test_main(hunitest.TestCase):
    """
    Test the one-command Phase 1 artifact contract.
    """

    def test1(self) -> None:
        """
        Test a short accepted rehearsal writes the complete artifact set.
        """
        output_dir = os.path.join(self.get_scratch_space(), "artifacts")
        args = [
            "run_model_drift_capability_experiment.py",
            "--trials_per_scenario",
            "20",
            "--power_trials",
            "1",
            "--dst_dir",
            output_dir,
            "-v",
            "INFO",
        ]
        with mock.patch.object(sys, "argv", args):
            rnscmdce._main(rnscmdce._parse())
        expected = (
            "protocol.json",
            "trial_results.csv",
            "scenario_summary.csv",
            "power_analysis.csv",
            "summary.json",
            "findings.md",
            "detection_power.svg",
            "false_alarm_rate.svg",
            "power_vs_query_count.svg",
        )
        for file_name in expected:
            self.assertTrue(os.path.isfile(os.path.join(output_dir, file_name)))
