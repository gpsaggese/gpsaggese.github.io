"""Test durable, immutable paid-plan artifacts."""

import os
import tempfile
import unittest

import research.Noesis.monitor.monitor_paid_execution as rnmonpaex
import research.Noesis.monitor.monitor_paid_report as rnmonpare


def _call(prompt="Reply OK."):
    return rnmonpaex.PaidCallSpec(
        call_id="call-1",
        batch=1,
        route_name="route",
        provider_slug="provider",
        expected_provider_name="Provider",
        model_id="model",
        prompt=prompt,
        max_tokens=8,
        timeout_s=30.0,
    )


class TestWritePaidPlan(unittest.TestCase):
    """Test idempotent review and conflict rejection."""

    def test1(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "plan.json")
            policy = rnmonpaex.PaidExecutionPolicy(1, 0.25, 0.01)
            rnmonpare.write_paid_plan(
                path, "test-v1", (_call(),), policy, {"bank": "hash"}
            )
            rnmonpare.write_paid_plan(
                path, "test-v1", (_call(),), policy, {"bank": "hash"}
            )
            with self.assertRaisesRegex(ValueError, "different paid plan"):
                rnmonpare.write_paid_plan(
                    path,
                    "test-v1",
                    (_call("Different prompt."),),
                    policy,
                    {"bank": "hash"},
                )
