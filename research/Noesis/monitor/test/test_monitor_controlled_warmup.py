"""Test the $10 controlled warm-up plan."""

import unittest

import research.Noesis.monitor.monitor_controlled_warmup as rnmoncowar
import research.Noesis.monitor.monitor_experiment_manifest as rnmonexma


def _make_bank():
    items = []
    for family in rnmonexma.DEFAULT_PHASE_ZERO_MANIFEST.benchmark_families:
        for index in range(12):
            items.append(
                rnmoncowar.WarmupItem(
                    item_id="%s-%02d" % (family.name, index),
                    family=family.name,
                    prompt="Solve item %s." % index,
                    grader_id="test-grader-v1",
                    grader_config_hash="a" * 64,
                )
            )
    return tuple(items)


class TestBuildWarmupCalls(unittest.TestCase):
    """Test balance, bounds, and reproducibility."""

    def test1(self):
        items = _make_bank()
        first = rnmoncowar.build_warmup_calls(items)
        second = rnmoncowar.build_warmup_calls(items)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 240)
        summary = rnmoncowar.summarize_warmup_plan(first)
        self.assertEqual(set(summary["route_counts"].values()), {60})
        self.assertEqual(set(summary["family_counts"].values()), {48})
        self.assertEqual({call.batch for call in first}, {1, 2, 3, 4})

    def test2(self):
        items = _make_bank()[:-1]
        with self.assertRaises(ValueError):
            rnmoncowar.build_warmup_calls(items)

    def test3(self):
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            rnmoncowar.WarmupItem(
                item_id="math-00",
                family="math",
                prompt="Solve this.",
                grader_id="test-grader-v1",
                grader_config_hash="not-a-digest",
            )

    def test4(self):
        calls = rnmoncowar.build_warmup_calls(_make_bank())
        prior = ({"call_id": calls[0].call_id}, {"call_id": calls[1].call_id})
        remaining = rnmoncowar.build_unattempted_calls(calls, prior)
        self.assertEqual(len(remaining), 238)
        self.assertEqual(remaining[0], calls[2])
        with self.assertRaisesRegex(ValueError, "exact plan prefix"):
            rnmoncowar.build_unattempted_calls(
                calls, ({"call_id": calls[1].call_id},)
            )
