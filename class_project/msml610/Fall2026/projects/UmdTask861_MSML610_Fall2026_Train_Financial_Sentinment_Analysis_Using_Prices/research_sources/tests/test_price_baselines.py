"""Check paired comparison and fitting boundaries, without market-performance claims."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from price_baselines import compare_baselines, evaluate_dataset, main, synthetic_dataset


def split_fixture():
    def row(index, partition):
        first = index / 100
        second = ((index * 7) % 13 - 6) / 100
        tone = ((index * 3) % 7 - 3) / 3
        return {"sample_id": f"{partition}-{index}", "status": "Eligible", "label_status": "Available",
                "feature_values": {"momentum_5d": first, "momentum_20d": second, "tone_score": tone},
                "excess_return": .5 * first + .2 * second + .03 * tone}
    return {"train": [row(i, "train") for i in range(20)],
            "test": [row(i, "test") for i in range(20, 26)], "horizon_sessions": 10}


class BaselineComparisonTests(unittest.TestCase):
    def test_every_model_uses_same_paired_samples_and_reports_loss_difference(self):
        split = split_fixture()
        result = compare_baselines(split, data_kind="synthetic")
        self.assertEqual(result["evaluation_status"], "Synthetic check completed")
        self.assertEqual(len(result["models"]), 5)
        self.assertEqual(result["mode"], "Synthetic contract check")
        expected = [r["sample_id"] for r in split["test"]]
        for model in result["models"].values():
            self.assertEqual([r["sample_id"] for r in model["predictions"]], expected)
            self.assertEqual(model["fit"]["fitted_sample_ids"], result["paired_train_ids"])
        delta = result["paired_loss_deltas"][0]
        self.assertEqual(delta["candidate"], "price_tone_ridge")
        expected_delta = result["models"]["price_tone_ridge"]["metrics"]["mae"] - result["models"]["price_ridge"]["metrics"]["mae"]
        self.assertAlmostEqual(delta["mean_absolute_error_delta"], expected_delta)

    def test_test_outcomes_cannot_change_fit_or_predictions(self):
        original = split_fixture()
        altered = copy.deepcopy(original)
        for row in altered["test"]:
            row["excess_return"] = -100 * row["excess_return"]
        first = compare_baselines(original, data_kind="synthetic")
        second = compare_baselines(altered, data_kind="synthetic")
        for name in first["models"]:
            self.assertEqual(first["models"][name]["fit"], second["models"][name]["fit"])
            self.assertEqual([r["predicted_excess_return"] for r in first["models"][name]["predictions"]],
                             [r["predicted_excess_return"] for r in second["models"][name]["predictions"]])
        self.assertNotEqual(first["models"]["price_ridge"]["metrics"], second["models"]["price_ridge"]["metrics"])

    def test_test_feature_distribution_never_refits_training_scaler(self):
        split = split_fixture()
        first = compare_baselines(split, data_kind="synthetic")
        altered = copy.deepcopy(split)
        altered["test"][-1]["feature_values"]["momentum_5d"] = 100
        second = compare_baselines(altered, data_kind="synthetic")
        self.assertEqual(first["models"]["price_ridge"]["fit"], second["models"]["price_ridge"]["fit"])
        self.assertEqual(first["models"]["price_ridge"]["predictions"][0], second["models"]["price_ridge"]["predictions"][0])

    def test_missing_tone_removes_row_from_all_models_without_neutral_imputation(self):
        split = split_fixture()
        split["test"][0]["feature_values"]["tone_score"] = None
        split["train"][0]["feature_values"].pop("momentum_20d")
        result = compare_baselines(split, data_kind="synthetic")
        self.assertEqual(result["coverage"]["paired_train"], 19)
        self.assertEqual(result["coverage"]["paired_test"], 5)
        self.assertEqual(len(result["excluded"]), 2)
        for model in result["models"].values():
            self.assertEqual(model["metrics"]["samples"], 5)
            self.assertNotIn("test-20", [r["sample_id"] for r in model["predictions"]])

    def test_missing_targets_are_excluded_without_turning_them_into_zero(self):
        split = split_fixture()
        split["test"][0]["excess_return"] = None
        result = compare_baselines(split, data_kind="observed")
        self.assertEqual(result["coverage"]["paired_test"], 5)
        self.assertIn("excess_return", result["excluded"][0]["fields"])

    def test_empty_or_insufficient_training_is_not_evaluated(self):
        for n in (0, 4):
            with self.subTest(n=n):
                split = split_fixture()
                split["train"] = split["train"][:n]
                result = compare_baselines(split, data_kind="synthetic")
                self.assertEqual(result["evaluation_status"], "Not evaluated")
                self.assertEqual(result["models"], {})

    def test_constant_features_are_stable_and_predict_training_mean(self):
        split = split_fixture()
        for row in split["train"] + split["test"]:
            row["feature_values"] = {"momentum_5d": .1, "momentum_20d": .2, "tone_score": 0}
        result = compare_baselines(split, data_kind="synthetic")
        for model in result["models"].values():
            self.assertEqual(model["predictions"][0]["predicted_excess_return"], model["fit"]["intercept"])

    def test_duplicate_partition_ids_or_retrospective_rows_are_rejected(self):
        split = split_fixture()
        split["test"][0]["sample_id"] = split["train"][0]["sample_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            compare_baselines(split, data_kind="synthetic")
        split = split_fixture()
        split["train"][0]["status"] = "Retrospective"
        with self.assertRaisesRegex(ValueError, "eligible"):
            compare_baselines(split, data_kind="synthetic")

    def test_numeric_or_mode_errors_fail_explicitly(self):
        for value in (True, "0.1", float("nan"), float("inf")):
            with self.subTest(value=value):
                split = split_fixture()
                split["test"][0]["feature_values"]["tone_score"] = value
                with self.assertRaises(ValueError):
                    compare_baselines(split, data_kind="synthetic")
        for alpha in (0, -1, True, float("inf")):
            with self.assertRaises(ValueError):
                compare_baselines(split_fixture(), data_kind="synthetic", ridge_alpha=alpha)
        with self.assertRaisesRegex(ValueError, "data_kind"):
            compare_baselines(split_fixture(), data_kind="unspecified")

    def test_extra_future_outcome_fields_are_never_used_as_features(self):
        split = split_fixture()
        first = compare_baselines(split, data_kind="synthetic")
        for row in split["train"] + split["test"]:
            row["feature_values"]["future_excess_return"] = row["excess_return"]
        self.assertEqual(first, compare_baselines(split, data_kind="synthetic"))

    def test_comparison_does_not_mutate_input(self):
        split = split_fixture()
        before = copy.deepcopy(split)
        compare_baselines(split, data_kind="synthetic")
        self.assertEqual(split, before)


class BaselineContractIntegrationTests(unittest.TestCase):
    def test_synthetic_ten_session_contract_runs_deterministically(self):
        dataset = synthetic_dataset()
        before = copy.deepcopy(dataset)
        labels, split, report = evaluate_dataset(dataset)
        self.assertEqual(dataset, before)
        self.assertEqual(report["evaluation_status"], "Synthetic check completed")
        self.assertGreaterEqual(report["coverage"]["paired_train"], 5)
        self.assertGreater(report["coverage"]["paired_test"], 0)
        self.assertEqual(report, evaluate_dataset(dataset)[2])
        self.assertTrue(all(row["status"] == "Eligible" for row in split["train"] + split["test"]))
        self.assertTrue(all(row["horizon_sessions"] == 10 for row in labels))

    def test_backfilled_raw_archive_cannot_become_forward_evaluation(self):
        dataset = synthetic_dataset()
        for price in dataset["prices"]:
            price["return_basis"] = "raw_price"
        for row in dataset["observations"]:
            row["historical_availability_verified"] = False
            row["response_received_at"] = dataset["as_of"]
        labels, split, report = evaluate_dataset(dataset)
        self.assertEqual(report["evaluation_status"], "Not evaluated")
        self.assertEqual(report["coverage"]["paired_train"], 0)
        self.assertFalse(split["train"])
        self.assertTrue(all(row["status"] != "Eligible" for row in labels))

    def test_demo_cannot_be_reclassified_as_observed_market_evidence(self):
        dataset = synthetic_dataset()
        dataset["data_kind"] = "observed"
        with self.assertRaisesRegex(ValueError, "Synthetic fixtures"):
            evaluate_dataset(dataset)
        dataset.pop("synthetic_fixture")
        with self.assertRaisesRegex(ValueError, "Synthetic fixtures"):
            evaluate_dataset(dataset)

    def test_cli_uses_new_private_directory_and_never_changes_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "dataset.json"
            original = json.dumps(synthetic_dataset()).encode()
            source.write_bytes(original)
            output = directory / "new-run"
            main(["--input", str(source), "--output-dir", str(output)])
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual((output / ".gitignore").read_text(), "*\n")
            self.assertEqual(json.loads((output / "comparison.json").read_text())["mode"], "Synthetic contract check")
            previous = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                main(["--input", str(source), "--output-dir", str(output)])
            self.assertEqual(previous, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_demo_cli_preserves_the_exact_input_hash(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "demo"
            main(["--demo", "--output-dir", str(output), "--horizon-sessions", "5"])
            canonical = (output / "synthetic_input.json").read_bytes()
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["input_sha256"], hashlib.sha256(canonical).hexdigest())


if __name__ == "__main__":
    unittest.main()
