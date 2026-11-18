"""Test the frozen 60-item warm-up benchmark and deterministic graders."""

import unittest

import research.Noesis.monitor.monitor_controlled_warmup as rnmoncowar
import research.Noesis.monitor.monitor_experiment_manifest as rnmonexma
import research.Noesis.monitor.monitor_warmup_benchmark as rnmonwaben


class TestWarmupBenchmark(unittest.TestCase):
    """Test balance, hash binding, and representative normalizers."""

    def test1(self):
        bank, configs = rnmonwaben.build_bank_and_configs()
        self.assertEqual(len(bank), 60)
        self.assertEqual(len(configs), 60)
        rnmoncowar.validate_warmup_bank(
            bank,
            rnmonexma.DEFAULT_PHASE_ZERO_MANIFEST,
            rnmoncowar.WarmupProtocol(),
        )
        rnmonwaben.validate_grader_configs(
            bank, {config.item_id: config for config in configs}
        )
        calls = rnmoncowar.build_warmup_calls(bank)
        self.assertEqual(len(calls), 240)

    def test2(self):
        cases = (
            ("integer-v1", 421, "ANSWER: 421", True),
            ("integer-v1", 421, "421 words", False),
            ("fraction-v1", "5/14", "ANSWER: 5/14", True),
            ("choice-v1", "B", "answer: b", True),
            ("exact-text-v1", "3 3", "3 3", True),
            ("exact-text-v1", "3 3", "The answer is 3 3", False),
            ("json-v1", {"a": 1}, '{"a": 1}', True),
        )
        for grader_id, expected, response, correct in cases:
            with self.subTest(grader_id=grader_id, response=response):
                self.assertEqual(
                    rnmonwaben.grade_response(
                        grader_id, expected, response
                    ),
                    correct,
                )

    def test3(self):
        bank, configs = rnmonwaben.build_bank_and_configs()
        altered = dict((config.item_id, config) for config in configs)
        config = altered[bank[0].item_id]
        altered[bank[0].item_id] = rnmonwaben.GraderConfig(
            config.item_id, config.grader_id, 999
        )
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            rnmonwaben.validate_grader_configs(bank, altered)

    def test4(self):
        self.assertTrue(
            rnmonwaben.grade_substantive_answer(
                "integer-v1",
                421,
                "Work shown here.\nANSWER: 421",
            )
        )
        self.assertTrue(
            rnmonwaben.grade_substantive_answer(
                "choice-v1",
                "B",
                "ANSWER: B) higher apparent Km and unchanged Vmax",
            )
        )
        self.assertFalse(
            rnmonwaben.grade_response(
                "integer-v1",
                421,
                "Work shown here.\nANSWER: 421",
            )
        )
