"""Test blinded behavioral-fingerprint controls."""

import unittest

import research.Noesis.monitor.monitor_fingerprint_experiment as rnmonfiex


def _fake_test(
    reference: rnmonfiex.CompletionPool,
    candidate: rnmonfiex.CompletionPool,
    protocol: rnmonfiex.FingerprintProtocol,
    seed: int,
) -> rnmonfiex.FingerprintTestResult:
    """
    Detect the synthetic alternative vocabulary without planted labels.
    """
    _ = reference, seed
    alternative_count = sum(
        "shortcut" in text
        or "alternate system" in text
        or "compact heuristic" in text
        for text in candidate.completions
    )
    detected = alternative_count >= protocol.sample_size // 2
    return rnmonfiex.FingerprintTestResult(
        p_value=0.0 if detected else 1.0,
        statistic=float(alternative_count),
    )


class TestBuildComparisonPools(unittest.TestCase):
    """
    Test independent pools and exact deterministic mixtures.
    """

    def test1(self) -> None:
        """
        Test a 25 percent mixture replaces exactly 15 of 60 rows.
        """
        protocol = rnmonfiex.FingerprintProtocol(trials_per_scenario=1)
        records = rnmonfiex.generate_synthetic_completion_triples(protocol, 7)
        first = rnmonfiex.build_comparison_pools(records, 0.25, 7)
        second = rnmonfiex.build_comparison_pools(records, 0.25, 7)
        self.assertEqual(first, second)
        reference, candidate, substituted_ids = first
        self.assertEqual(len(substituted_ids), 15)
        self.assertEqual(reference.prompt_ids, candidate.prompt_ids)
        substituted = set(substituted_ids)
        per_prompt = {
            prompt_id: sum(
                record.record_id in substituted
                for record in records
                if record.prompt_id == prompt_id
            )
            for prompt_id in range(protocol.prompt_count)
        }
        self.assertEqual(set(per_prompt.values()), {3})

    def test2(self) -> None:
        """
        Test A/A uses independent reference draws rather than reused text.
        """
        protocol = rnmonfiex.FingerprintProtocol(trials_per_scenario=1)
        records = rnmonfiex.generate_synthetic_completion_triples(protocol, 9)
        reference, candidate, substituted_ids = rnmonfiex.build_comparison_pools(
            records, 0.0, 9
        )
        self.assertEqual(substituted_ids, ())
        self.assertNotEqual(reference.completions, candidate.completions)


class TestRunFingerprintExperiment(unittest.TestCase):
    """
    Test blinded control aggregation with an injected detector.
    """

    def test1(self) -> None:
        """
        Test A/A, full-swap, and half-swap gates separate.
        """
        protocol = rnmonfiex.FingerprintProtocol(trials_per_scenario=3)
        scenarios = (
            rnmonfiex.DEFAULT_FINGERPRINT_SCENARIOS[0],
            rnmonfiex.DEFAULT_FINGERPRINT_SCENARIOS[1],
            rnmonfiex.DEFAULT_FINGERPRINT_SCENARIOS[-1],
        )
        result = rnmonfiex.run_fingerprint_experiment(
            protocol, scenarios, _fake_test
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.scenarios[0].false_alarm_rate, 0.0)
        self.assertEqual(result.scenarios[1].detection_rate, 1.0)
        self.assertEqual(result.scenarios[2].detection_rate, 1.0)
