"""Verify the offline course entry point with temporary artificial inputs only.

The imported synthetic dataset checks software contracts, never predictive skill.
Archive fixtures deliberately lack availability/calendar/total-return evidence.
"""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_course
import data_preflight
from price_baselines import evaluate_dataset, synthetic_dataset
import test_data_preflight as archive_fixtures


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CourseEntryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def prepared(self, dataset=None, *, name="run"):
        source = self.root / f"{name}-input.json"
        source.write_text(json.dumps(synthetic_dataset() if dataset is None else dataset) + "\n",
                          encoding="utf-8")
        output = self.root / name
        result = run_course.run_pipeline(dataset_path=source, output_dir=output)
        return result, source, output

    def archive_fixture(self):
        fixture = archive_fixtures.DataPreflightTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def dry_run(self, fixture, output):
        return run_course.run_pipeline(
            dry_run=True, input_dir=fixture.primary, legacy_alpaca_dir=fixture.legacy,
            month="2026-09", as_of="2026-10-16T12:00:00Z", output_dir=output)

    def test_archive_dry_run_reads_real_files_but_does_not_invoke_models_or_labels(self):
        fixture = self.archive_fixture()
        original = {p: p.read_bytes() for directory in (fixture.primary, fixture.legacy)
                    for p in directory.iterdir()}
        output = self.root / "preflight"
        with mock.patch.object(run_course, "evaluate_dataset") as evaluate:
            report = self.dry_run(fixture, output)
        evaluate.assert_not_called()
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["execution_status"], "Dry-run completed")
        self.assertEqual(report["evaluation_status"], "Not evaluated")
        self.assertEqual(report["labels_created"], 0)
        self.assertTrue((output / "source_family_audit.json").is_file())
        self.assertEqual(report["source_family_audits"]["month_news"]["eligible_source_records"], 0)
        self.assertIn("earnings_call", report["source_family_audits"]["month_news"]["missing_source_families"])
        self.assertFalse(report["market_predictive_skill_established"])
        for name in ("labels.json", "split.json", "comparison.json"):
            self.assertFalse((output / name).exists())
        self.assertEqual(original, {p: p.read_bytes() for p in original})
        preflight = read_json(output / "preflight.json")
        self.assertEqual(report["archive_summary"], preflight["summary"])
        self.assertIn("independent_versioned_complete_exchange_calendar", report["missing_inputs"])
        self.assertIn("corporate_actions_and_consistent_open_total_return_indices", report["missing_inputs"])
        self.assertIn("prepared_dataset.json following required_contract.json", report["missing_inputs"])
        self.assertIn("evaluation.training_cutoff/test_start/test_end fixed before evaluation",
                      report["missing_inputs"])
        manifest = read_json(output / "run_manifest.json")
        self.assertEqual(manifest["input_files_sha256"], preflight["input_files_sha256"])
        self.assertTrue(manifest["input_files_sha256"])
        for path, expected in manifest["input_files_sha256"].items():
            self.assertEqual(digest(Path(path)), expected)

    def test_missing_contract_scaffolding_reports_exact_paths_without_evaluation(self):
        dataset = synthetic_dataset()
        dataset.pop("data_kind")
        dataset["calendar"].pop("calendar_version")
        dataset["evaluation"].pop("test_end")
        with mock.patch.object(run_course, "evaluate_dataset") as evaluate:
            report, _, output = self.prepared(dataset)
        evaluate.assert_not_called()
        self.assertEqual(report["exit_code"], 2)
        self.assertEqual(set(report["missing_inputs"]),
                         {"data_kind", "calendar.calendar_version", "evaluation.test_end"})
        self.assertEqual(report["evaluation_status"], "Not evaluated")
        self.assertEqual(report["labels_created"], 0)
        self.assertEqual(read_json(output / "report.json"), report)
        self.assertTrue((output / "required_contract.json").is_file())
        self.assertFalse((output / "comparison.json").exists())

    def test_optional_source_snapshot_audit_and_exact_fingerprints_reach_runner_report(self):
        fixture = self.archive_fixture()
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        source = snapshot / "input-evidence.json"
        source.write_text("{}\n")
        output = self.root / "with-source-snapshot"
        with mock.patch.object(data_preflight, "_snapshot_source_records", return_value=(
                [], {str(source): digest(source)}, {"snapshot_analysis_cutoff": "2026-10-12T16:00:00Z"})) as adapt:
            with mock.patch.object(run_course, "evaluate_dataset") as evaluate:
                report = run_course.run_pipeline(
                    dry_run=True, input_dir=fixture.primary, legacy_alpaca_dir=fixture.legacy,
                    month="2026-09", as_of="2026-10-16T12:00:00Z",
                    source_snapshot_dir=snapshot, output_dir=output)
        adapt.assert_called_once_with(snapshot)
        evaluate.assert_not_called()
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["source_family_audits"]["snapshot"]["cutoff"], "2026-10-12T16:00:00Z")
        self.assertIn("snapshot.earnings_call.retained_text_with_publication_version_and_actual_receipt",
                      report["missing_inputs"])
        self.assertEqual(read_json(output / "run_manifest.json")["input_files_sha256"][str(source)], digest(source))
        self.assertEqual(set(read_json(output / "source_family_audit.json")), {"month_news", "snapshot"})
        self.assertFalse((output / "labels.json").exists())

    def test_optional_source_archive_and_its_symlink_cannot_receive_outputs(self):
        fixture = self.archive_fixture()
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        alias = self.root / "snapshot-alias"
        alias.symlink_to(snapshot, target_is_directory=True)
        for output in (snapshot / "new", alias / "new"):
            with self.subTest(output=output):
                with mock.patch.object(run_course, "audit_archives") as audit:
                    with self.assertRaisesRegex(ValueError, "optional source snapshot"):
                        run_course.run_pipeline(
                            dry_run=True, input_dir=fixture.primary, legacy_alpaca_dir=fixture.legacy,
                            month="2026-09", as_of="2026-10-16T12:00:00Z",
                            source_snapshot_dir=snapshot, output_dir=output)
                audit.assert_not_called()
                self.assertFalse(output.exists())

    def test_prepared_input_cannot_silently_ignore_source_snapshot_argument(self):
        with self.assertRaisesRegex(ValueError, "omit archive arguments"):
            run_course.run_pipeline(demo=True, source_snapshot_dir=self.root,
                                    output_dir=self.root / "not-created")
        self.assertFalse((self.root / "not-created").exists())

    def test_missing_whole_calendar_evaluation_or_dataset_shape_stays_blocked(self):
        cases = (
            ({"data_kind": "observed"}, {"as_of", "calendar", "prices", "observations", "evaluation"}),
            ([], {"dataset (JSON object)"}),
            ({"calendar": [], "evaluation": "unknown", "prices": {}, "observations": {}},
             {"data_kind", "as_of", "calendar (object)", "evaluation (object)",
              "prices (list)", "observations (list)"}),
        )
        for index, (dataset, expected) in enumerate(cases):
            with self.subTest(dataset=dataset):
                with mock.patch.object(run_course, "evaluate_dataset") as evaluate:
                    report, _, output = self.prepared(dataset, name=f"missing-{index}")
                evaluate.assert_not_called()
                self.assertEqual(set(report["missing_inputs"]), expected)
                self.assertEqual(report["exit_code"], 2)
                self.assertFalse((output / "labels.json").exists())

    def test_missing_session_open_reports_its_exact_calendar_path_before_evaluation(self):
        dataset = synthetic_dataset()
        dataset["calendar"]["sessions"][0].pop("open_at")
        with mock.patch.object(run_course, "evaluate_dataset") as evaluate:
            report, _, output = self.prepared(dataset)
        evaluate.assert_not_called()
        self.assertEqual(report["missing_inputs"], ["calendar.sessions[0].open_at"])
        self.assertEqual(report["exit_code"], 2)
        self.assertEqual(report["evaluation_status"], "Not evaluated")
        self.assertFalse((output / "labels.json").exists())

    def test_missing_per_row_evidence_is_not_promoted_to_eligible(self):
        dataset = synthetic_dataset()
        missing_receipt, late_receipt = dataset["observations"][:2]
        missing_receipt.pop("response_received_at")
        late_receipt["response_received_at"] = late_receipt["prediction_completed_at"]
        dataset["prices"] = [p for p in dataset["prices"]
                             if not (p["symbol"] == "DEMO"
                                     and p["session_date"] == dataset["observations"][2]["decision_session"])]
        report, _, output = self.prepared(dataset)
        labels = {row["sample_id"]: row for row in read_json(output / "labels.json")}
        self.assertIn("observations[0].response_received_at", report["missing_inputs"])
        self.assertEqual(labels[missing_receipt["sample_id"]]["status"], "Retrospective")
        self.assertIn("time_missing:response_received_at", labels[missing_receipt["sample_id"]]["reasons"])
        self.assertEqual(labels[late_receipt["sample_id"]]["status"], "Unavailable")
        self.assertIn("after_cutoff:response_received_at", labels[late_receipt["sample_id"]]["reasons"])
        absent = labels[dataset["observations"][2]["sample_id"]]
        self.assertEqual(absent["status"], "Unavailable")
        self.assertIn("required_open_price_missing", absent["reasons"])
        split = read_json(output / "split.json")
        selected = {row["sample_id"] for side in ("train", "test") for row in split[side]}
        self.assertTrue(set(labels).difference(selected))
        for observation in dataset["observations"][:3]:
            self.assertNotIn(observation["sample_id"], selected)

    def test_raw_price_history_remains_retrospective_and_keeps_negative_result(self):
        dataset = synthetic_dataset()
        for price in dataset["prices"]:
            price["return_basis"] = "raw_price"
        report, _, output = self.prepared(dataset)
        labels = read_json(output / "labels.json")
        self.assertTrue(all(row["status"] == "Retrospective" for row in labels))
        self.assertTrue(all("raw_price_not_total_return" in row["reasons"] for row in labels))
        self.assertEqual(report["exit_code"], 2)
        self.assertEqual(report["eligible_labels"], 0)
        self.assertEqual(report["evaluation_status"], "Not evaluated")
        self.assertEqual(read_json(output / "comparison.json")["models"], {})
        self.assertEqual(read_json(output / "split.json")["train"], [])
        self.assertTrue(report["blocking_reasons"])

    def test_insufficient_paired_training_retains_labels_split_and_missing_fields(self):
        dataset = synthetic_dataset()
        for observation in dataset["observations"]:
            observation["feature_values"].pop("tone_score")
            observation["feature_provenance"].pop("tone_score")
        report, _, output = self.prepared(dataset)
        comparison = read_json(output / "comparison.json")
        split = read_json(output / "split.json")
        self.assertEqual(report["exit_code"], 2)
        self.assertEqual(report["labels_created"], 45)
        self.assertEqual(report["eligible_labels"], 45)
        self.assertEqual(comparison["evaluation_status"], "Not evaluated")
        self.assertEqual(comparison["coverage"]["paired_train"], 0)
        self.assertGreater(len(split["train"]), 0)
        self.assertGreater(len(split["test"]), 0)
        self.assertEqual(comparison["models"], {})
        self.assertEqual(report["paired_missing_field_exclusions"], comparison["excluded"])
        self.assertTrue(all(row["fields"] == ["tone_score"] for row in comparison["excluded"]))
        self.assertEqual(len(comparison["excluded"]), len(split["train"]) + len(split["test"]))
        self.assertIn("observations[0].feature_values.tone_score", report["missing_inputs"])

    def test_entry_reuses_evaluator_and_preserves_transitive_out_of_window_event_purging(self):
        dataset = synthetic_dataset()
        train, test, bridge = (dataset["observations"][index] for index in (0, 27, 44))
        train["source_ids"] = ["chain-article"]
        train["event_cluster_ids"] = ["earlier-event"]
        test["source_ids"] = ["test-article"]
        test["event_cluster_ids"] = ["later-event"]
        bridge["source_ids"] = ["chain-article"]
        bridge["event_cluster_ids"] = ["later-event"]
        bridge["historical_availability_verified"] = False
        with mock.patch.object(run_course, "evaluate_dataset", wraps=evaluate_dataset) as evaluate:
            report, _, output = self.prepared(dataset)
        evaluate.assert_called_once()
        self.assertEqual(evaluate.call_args.kwargs, {"horizon_sessions": 10})
        split = read_json(output / "split.json")
        self.assertNotIn(train["sample_id"], {row["sample_id"] for row in split["train"]})
        exclusions = {row["sample_id"]: row["reasons"] for row in split["excluded"]}
        self.assertEqual(exclusions[train["sample_id"]], ["transitive_group_shared_with_test"])
        self.assertIn("sample_not_eligible:Retrospective", exclusions[bridge["sample_id"]])
        self.assertIn(test["sample_id"], {row["sample_id"] for row in split["test"]})
        self.assertEqual(report["exit_code"], 0)

    def test_all_models_use_same_complete_case_ids_after_entry_point_filtering(self):
        dataset = synthetic_dataset()
        for index, feature in ((0, "tone_score"), (27, "momentum_20d")):
            dataset["observations"][index]["feature_values"].pop(feature)
            dataset["observations"][index]["feature_provenance"].pop(feature)
        report, _, output = self.prepared(dataset)
        comparison = read_json(output / "comparison.json")
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(len(comparison["models"]), 5)
        for model in comparison["models"].values():
            self.assertEqual(model["fit"]["fitted_sample_ids"], comparison["paired_train_ids"])
            self.assertEqual([p["sample_id"] for p in model["predictions"]], comparison["paired_test_ids"])
        self.assertNotIn(dataset["observations"][0]["sample_id"], comparison["paired_train_ids"])
        self.assertNotIn(dataset["observations"][27]["sample_id"], comparison["paired_test_ids"])
        self.assertEqual(len(report["paired_missing_field_exclusions"]), 2)

    def test_prepared_synthetic_contract_is_labeled_software_check_with_pending_research(self):
        report, _, output = self.prepared()
        self.assertEqual(report["mode"], "Synthetic contract check")
        self.assertEqual(report["evaluation_status"], "Synthetic check completed")
        self.assertFalse(report["market_predictive_skill_established"])
        self.assertIn("pending", report["execution_status"].lower())
        for name in ("archive_to_company_day_adapter", "price_supervised_text_model",
                     "lm_and_full_text_baseline_comparison", "large_move_news_attribution"):
            self.assertEqual(report["stages"][name]["status"], "Pending")
            self.assertIn(name, report["pending_research_stages"])
        self.assertFalse((output / "synthetic_input.json").exists())

    def test_synthetic_cannot_be_reclassified_as_observed_in_prepared_mode(self):
        for marker_present in (True, False):
            with self.subTest(marker_present=marker_present):
                dataset = synthetic_dataset()
                dataset["data_kind"] = "observed"
                if not marker_present:
                    dataset.pop("synthetic_fixture")
                report, _, output = self.prepared(dataset, name=f"relabel-{marker_present}")
                self.assertEqual(report["exit_code"], 2)
                self.assertEqual(report["evaluation_status"], "Not evaluated")
                self.assertTrue(any("Synthetic fixtures" in reason for reason in report["blocking_reasons"]))
                self.assertFalse((output / "labels.json").exists())
                self.assertFalse((output / "comparison.json").exists())

    def test_manifest_commits_exact_input_source_and_output_bytes_without_mutation(self):
        report, source, output = self.prepared()
        before = source.read_bytes()
        manifest = read_json(output / "run_manifest.json")
        self.assertEqual(manifest["input_files_sha256"], {str(source.resolve()): digest(source)})
        self.assertEqual(set(Path(p).name for p in manifest["generation_sources_sha256"]),
                         set(run_course.MODULES))
        for path, expected in manifest["generation_sources_sha256"].items():
            self.assertEqual(digest(Path(path)), expected)
        for name, expected in manifest["outputs"].items():
            self.assertEqual((output / name).stat().st_size, expected["bytes"])
            self.assertEqual(digest(output / name), expected["sha256"])
        self.assertIn("report.json", manifest["outputs"])
        self.assertIn("required_contract.json", manifest["outputs"])
        self.assertEqual(report, read_json(output / "report.json"))
        self.assertEqual((output / ".gitignore").read_text(), "*\n")
        self.assertEqual(output.stat().st_mode & 0o777, 0o700)
        self.assertEqual(source.read_bytes(), before)

    def test_demo_retains_exact_synthetic_input_and_output_hash(self):
        output = self.root / "demo"
        report = run_course.run_pipeline(output_dir=output, demo=True)
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["mode"], "Synthetic contract check")
        manifest = read_json(output / "run_manifest.json")
        self.assertEqual(manifest["outputs"]["synthetic_input.json"]["sha256"],
                         digest(output / "synthetic_input.json"))
        self.assertTrue(read_json(output / "synthetic_input.json")["synthetic_fixture"])

    def test_existing_directory_or_dangling_symlink_is_never_replaced(self):
        existing = self.root / "existing"
        existing.mkdir()
        marker = existing / "marker.txt"
        marker.write_text("Keep existing work\n")
        symlink = self.root / "dangling-link"
        symlink.symlink_to(self.root / "does-not-exist", target_is_directory=True)
        for output in (existing, symlink):
            with self.subTest(output=output):
                with mock.patch.object(run_course, "evaluate_dataset") as evaluate:
                    with self.assertRaises(FileExistsError):
                        run_course.run_pipeline(output_dir=output, demo=True)
                evaluate.assert_not_called()
        self.assertEqual(marker.read_text(), "Keep existing work\n")
        self.assertTrue(symlink.is_symlink())
        self.assertFalse((self.root / "does-not-exist").exists())

    def test_dry_run_output_inside_either_archive_or_aliased_archive_is_rejected(self):
        fixture = self.archive_fixture()
        alias = self.root / "primary-alias"
        alias.symlink_to(fixture.primary, target_is_directory=True)
        for output in (fixture.primary / "new", fixture.legacy / "new", alias / "new"):
            with self.subTest(output=output):
                with mock.patch.object(run_course, "audit_archives") as audit:
                    with self.assertRaisesRegex(ValueError, "outside both input archives"):
                        self.dry_run(fixture, output)
                audit.assert_not_called()
                self.assertFalse(output.exists())

    def test_input_change_during_evaluation_invalidates_run_before_publication(self):
        source = self.root / "input.json"
        source.write_text(json.dumps(synthetic_dataset()))
        output = self.root / "invalidated"

        def changing_input(dataset, **settings):
            result = evaluate_dataset(dataset, **settings)
            source.write_text(source.read_text() + "\n")
            return result

        with mock.patch.object(run_course, "evaluate_dataset", side_effect=changing_input):
            with self.assertRaisesRegex(ValueError, "changed during execution"):
                run_course.run_pipeline(dataset_path=source, output_dir=output)
        self.assertFalse(output.exists())

    def test_input_integrity_failure_is_reported_without_labels_or_comparison(self):
        fixture = self.archive_fixture()
        path = fixture.primary / "news_raw_000.json"
        raw = read_json(path)
        raw["news"][0]["content"] = "Tampered fixture after hashing"
        path.write_text(json.dumps(raw))
        output = self.root / "tampered-archive"
        report = self.dry_run(fixture, output)
        self.assertEqual(report["exit_code"], 2)
        self.assertEqual(report["evaluation_status"], "Not evaluated")
        self.assertTrue(any("integrity" in reason for reason in report["blocking_reasons"]))
        self.assertEqual(report["stages"]["input_quality_audit"]["status"], "Blocked")
        self.assertFalse((output / "preflight.json").exists())
        self.assertFalse((output / "labels.json").exists())

    def test_cli_exit_codes_distinguish_software_success_from_blocked_data(self):
        successful = self.root / "successful-cli"
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            code = run_course.main(["--demo", "--output-dir", str(successful)])
        self.assertEqual(code, 0)
        printed = json.loads(captured.getvalue())
        self.assertEqual(printed["evaluation_status"], "Synthetic check completed")
        missing = self.root / "missing.json"
        missing.write_text("{}\n")
        blocked = self.root / "blocked-cli"
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            code = run_course.main(["--dataset", str(missing), "--output-dir", str(blocked)])
        self.assertEqual(code, 2)
        printed = json.loads(captured.getvalue())
        self.assertEqual(printed["exit_code"], 2)
        self.assertEqual(printed["evaluation_status"], "Not evaluated")
        self.assertEqual(printed["labels_created"], 0)

    def test_modes_are_exclusive_and_archive_scope_cannot_override_prepared_clock(self):
        combinations = ({}, {"demo": True, "dataset_path": self.root / "input.json"},
                        {"dry_run": True}, {"demo": True, "as_of": "2026-10-16T12:00:00Z"})
        for index, settings in enumerate(combinations):
            output = self.root / f"invalid-mode-{index}"
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    run_course.run_pipeline(output_dir=output, **settings)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
