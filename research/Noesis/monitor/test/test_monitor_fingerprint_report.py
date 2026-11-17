"""
Test behavioral-fingerprint review artifacts.
"""

import dataclasses
import os
import tempfile
import unittest

import research.Noesis.monitor.monitor_fingerprint_experiment as rnmonfiex
import research.Noesis.monitor.monitor_fingerprint_report as rnmonfire


def _always_null_test(
    reference: rnmonfiex.CompletionPool,
    candidate: rnmonfiex.CompletionPool,
    protocol: rnmonfiex.FingerprintProtocol,
    seed: int,
) -> rnmonfiex.FingerprintTestResult:
    _ = reference, candidate, protocol, seed
    return rnmonfiex.FingerprintTestResult(1.0, 0.0)


class TestWriteFingerprintArtifacts(unittest.TestCase):
    """
    Test the fingerprint artifact contract.
    """

    def test1(self) -> None:
        """
        Test a small A/A run writes all declared artifacts.
        """
        protocol = rnmonfiex.FingerprintProtocol(trials_per_scenario=1)
        result = rnmonfiex.run_fingerprint_experiment(
            protocol,
            (rnmonfiex.DEFAULT_FINGERPRINT_SCENARIOS[0],),
            _always_null_test,
        )
        with tempfile.TemporaryDirectory() as output_dir:
            paths = rnmonfire.write_fingerprint_artifacts(result, output_dir)
            for path in dataclasses.astuple(paths):
                self.assertTrue(os.path.isfile(path), path)
            with open(paths.findings_path, encoding="utf-8") as file_:
                findings = file_.read()
        self.assertIn("synthetic implementation control", findings)
        self.assertIn("Live-provider conclusions supported: `none`", findings)
