"""Test the paid Lane 2 fingerprint collection plan."""

import tempfile
import unittest

import research.Noesis.monitor.monitor_live_fingerprint as rnmonlifp


def _prompts():
    return tuple(
        rnmonlifp.LiveFingerprintPrompt("prompt-%s" % index, "Explain %s" % index)
        for index in range(5)
    )


class TestLiveFingerprintPlan(unittest.TestCase):
    """Test exact plan balance and stable validation."""

    def test1(self):
        protocol = rnmonlifp.LiveFingerprintProtocol()
        calls = rnmonlifp.build_live_fingerprint_calls(_prompts())
        summary = rnmonlifp.summarize_live_fingerprint_plan(calls)
        self.assertEqual(len(calls), 180)
        self.assertEqual(
            summary["pool_counts"],
            {"alternative": 60, "reference-a": 60, "reference-b": 60},
        )
        self.assertEqual(
            summary["route_counts"],
            {
                "cloudflare-llama-3.1-8b": 60,
                "cloudflare-llama-3.3-70b": 120,
            },
        )
        self.assertEqual({call.batch for call in calls}, {1, 2, 3})
        self.assertTrue(
            all(call.sampling_temperature == 1.0 for call in calls)
        )
        self.assertEqual(protocol.make_execution_policy().maximum_cost_usd, 2.0)

    def test2(self):
        with self.assertRaisesRegex(ValueError, "exactly 5"):
            rnmonlifp.build_live_fingerprint_calls(_prompts()[:4])

    def test3(self):
        calls = rnmonlifp.build_live_fingerprint_calls(_prompts())
        prior = [{"call_id": call.call_id} for call in calls[:17]]
        remaining = rnmonlifp.build_unattempted_live_fingerprint_calls(
            calls, prior
        )
        self.assertEqual(remaining, calls[17:])

    def test4(self):
        calls = rnmonlifp.build_live_fingerprint_calls(_prompts())
        prior = [{"call_id": calls[1].call_id}]
        with self.assertRaisesRegex(ValueError, "exact plan prefix"):
            rnmonlifp.build_unattempted_live_fingerprint_calls(calls, prior)

    def test5(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as file_:
            file_.write('{"prompt_id":"one","prompt":"Text"}\n')
            file_.flush()
            loaded = rnmonlifp.load_live_fingerprint_bank(file_.name)
        self.assertEqual(loaded[0].prompt_id, "one")
