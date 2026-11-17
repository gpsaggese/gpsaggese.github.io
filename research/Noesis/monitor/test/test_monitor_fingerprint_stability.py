"""Test the compressed 70B fingerprint stability plan."""

import unittest

import research.Noesis.monitor.monitor_fingerprint_stability as rnmonfist
import research.Noesis.monitor.monitor_live_fingerprint as rnmonlifp


class TestFingerprintStabilityPlan(unittest.TestCase):
    def test1(self):
        prompts = tuple(
            rnmonlifp.LiveFingerprintPrompt("prompt-%s" % index, "Prompt")
            for index in range(5)
        )
        calls = rnmonfist.build_stability_calls(prompts)
        summary = rnmonfist.summarize_stability_plan(calls)
        self.assertEqual(len(calls), 180)
        self.assertEqual(
            summary["pool_counts"],
            {"reference-c": 60, "reference-d": 60, "reference-e": 60},
        )
        self.assertEqual({call.route_name for call in calls}, {
            "cloudflare-llama-3.3-70b"
        })
        self.assertTrue(all(call.sampling_temperature == 1.0 for call in calls))
        self.assertEqual(
            rnmonfist.FingerprintStabilityProtocol()
            .make_execution_policy()
            .maximum_cost_usd,
            0.50,
        )


if __name__ == "__main__":
    unittest.main()
