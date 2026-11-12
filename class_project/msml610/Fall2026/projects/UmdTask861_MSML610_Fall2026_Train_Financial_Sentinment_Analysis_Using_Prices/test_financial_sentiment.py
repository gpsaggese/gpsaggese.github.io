"""Regression checks for the independent portable course replay."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import financial_sentiment_utils as utils


class PortableEvidenceTests(unittest.TestCase):
    def test_real_refit_agrees_with_original_solver(self):
        rows = utils.refit_baselines()
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["test_rows"] == 87 for r in rows))
        self.assertLess(max(r["max_prediction_difference"] for r in rows), 1e-6)

    def test_training_outcome_crossing_cutoff_is_rejected(self):
        packet = json.loads(Path(__file__).with_name("data").joinpath("historical_features.json").read_text())
        packet["partitions"]["train"]["samples"][0]["exit_at"] = "2019-01-02T14:30:00Z"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(packet))
            with self.assertRaisesRegex(ValueError, "crosses test boundary"):
                utils.refit_baselines(path)

    def test_curve_summary_tampering_is_rejected(self):
        _, strategy, _ = utils.load_evidence()
        bad = copy.deepcopy(strategy)
        bad["curves"][-1]["B"] += 1
        with self.assertRaisesRegex(ValueError, "disagree"):
            utils.audit_strategy(bad)

    def test_daily_and_intraday_drawdowns_are_not_conflated(self):
        _, strategy, _ = utils.load_evidence()
        for row in utils.audit_strategy(strategy)["models"]:
            self.assertGreaterEqual(row["daily_close_drawdown"] + 1e-9, row["reported_intraday_drawdown"])

    def test_label_maturity_and_nonfinite_prices(self):
        self.assertAlmostEqual(utils.price_label([100, 105], [100, 102], 0, 1)["excess_return"], 0.03)
        for args in [([100, 105], [100, 102], 0, 2), ([100, float("nan")], [100, 102], 0, 1)]:
            with self.assertRaises(ValueError):
                utils.price_label(*args)


if __name__ == "__main__":
    unittest.main()
