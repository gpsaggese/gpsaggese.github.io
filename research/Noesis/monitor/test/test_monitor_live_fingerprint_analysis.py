"""Test analysis of saved live fingerprint pools."""

import os
import tempfile
import unittest

import research.Noesis.monitor.monitor_fingerprint_experiment as rnmonfiex
import research.Noesis.monitor.monitor_live_fingerprint_analysis as rnmonlifa


def _rows():
    rows = []
    for pool in ("reference-a", "reference-b", "alternative"):
        for prompt_index in range(2):
            for sample_index in range(2):
                rows.append(
                    {
                        "call_id": "%s-%s-%s"
                        % (pool, prompt_index, sample_index),
                        "provider_status": "ok",
                        "attribution_matches": True,
                        "usage_complete": True,
                        "response": "%s response %s %s"
                        % (pool, prompt_index, sample_index),
                        "metadata": [
                            ["pool", pool],
                            ["prompt_index", str(prompt_index)],
                            ["sample_index", str(sample_index)],
                            ["bank_hash", "abc"],
                        ],
                    }
                )
    return rows


def _fake_test(reference, candidate, protocol, seed):
    _ = (reference, protocol, seed)
    alternative_count = sum(
        value.startswith("alternative") for value in candidate.completions
    )
    p_value = 0.01 if alternative_count >= 2 else 0.50
    return rnmonfiex.FingerprintTestResult(
        p_value=p_value, statistic=float(alternative_count)
    )


class TestLiveFingerprintAnalysis(unittest.TestCase):
    """Test strict pool construction and scenario summaries."""

    def test1(self):
        protocol = rnmonfiex.FingerprintProtocol(
            trials_per_scenario=1,
            prompt_count=2,
            samples_per_prompt=2,
            permutations=10,
        )
        records = rnmonlifa.build_live_completion_triples(_rows(), protocol)
        self.assertEqual(len(records), 4)
        trials = rnmonlifa.run_live_fingerprint_tests(
            records, protocol, test_func=_fake_test, partial_trials=3
        )
        summaries = {
            row["scenario"]: row
            for row in rnmonlifa.summarize_live_fingerprint_tests(trials)
        }
        self.assertEqual(summaries["healthy_a_a"]["detection_rate"], 0.0)
        self.assertEqual(summaries["full_model_swap"]["detection_rate"], 1.0)
        self.assertEqual(summaries["partial_substitution_50"]["detection_rate"], 1.0)

    def test2(self):
        protocol = rnmonfiex.FingerprintProtocol(
            trials_per_scenario=1,
            prompt_count=2,
            samples_per_prompt=2,
            permutations=10,
        )
        rows = _rows()
        rows[0]["provider_status"] = "timeout"
        rows[0]["response"] = None
        with self.assertRaisesRegex(ValueError, "unusable calls"):
            rnmonlifa.build_live_completion_triples(rows, protocol)

    def test3(self):
        protocol = rnmonfiex.FingerprintProtocol(
            trials_per_scenario=1,
            prompt_count=2,
            samples_per_prompt=2,
            permutations=10,
        )
        with self.assertRaisesRegex(ValueError, "at least 12"):
            rnmonlifa.build_live_completion_triples(_rows()[:-1], protocol)

    def test4(self):
        protocol = rnmonfiex.FingerprintProtocol(
            trials_per_scenario=1,
            prompt_count=2,
            samples_per_prompt=2,
            permutations=10,
        )
        rows = _rows()
        failed = dict(rows[0])
        rows[0]["provider_status"] = "timeout"
        rows[0]["response"] = None
        replacement = dict(failed)
        replacement["call_id"] = "topup-" + str(failed["call_id"])
        rows.append(replacement)
        records = rnmonlifa.build_live_completion_triples(rows, protocol)
        self.assertEqual(len(records), 4)

    def test5(self):
        protocol = rnmonfiex.FingerprintProtocol(
            trials_per_scenario=1,
            prompt_count=2,
            samples_per_prompt=2,
            permutations=10,
        )
        records = rnmonlifa.build_live_completion_triples(_rows(), protocol)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "checkpoint.jsonl")
            first = rnmonlifa.run_live_fingerprint_tests(
                records,
                protocol,
                test_func=_fake_test,
                partial_trials=1,
                checkpoint_path=checkpoint,
            )

            def _must_not_run(*_args):
                raise AssertionError("completed checkpoint should be reused")

            second = rnmonlifa.run_live_fingerprint_tests(
                records,
                protocol,
                test_func=_must_not_run,
                partial_trials=1,
                checkpoint_path=checkpoint,
            )
        self.assertEqual(first, second)
