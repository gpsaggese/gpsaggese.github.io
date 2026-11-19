"""Synthetic, offline contract tests; no market-performance claims or I/O."""

from copy import deepcopy
from datetime import datetime
import json
import unittest
from zoneinfo import ZoneInfo

from research.financial_sentiment_using_prices.price_labels import build_labels, make_time_split


NY = ZoneInfo("America/New_York")
# An explicit synthetic exchange-calendar fixture, including gaps. The module
# must count these supplied sessions, never insert weekends or missing dates.
DAYS = ["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07",
        "2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13", "2026-10-14",
        "2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20", "2026-10-21",
        "2026-10-22", "2026-10-23", "2026-10-26", "2026-10-27", "2026-10-28",
        "2026-10-29", "2026-10-30", "2026-11-02", "2026-11-03", "2026-11-04",
        "2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11",
        "2026-11-12", "2026-11-13", "2026-11-16", "2026-11-17", "2026-11-18"]


def local(day, clock):
    return datetime.fromisoformat(day + "T" + clock).replace(tzinfo=NY).isoformat()


def observation(index=0, symbol="PLTR", **changes):
    day = DAYS[index]
    row = {"sample_id": f"{symbol}:{day}", "symbol": symbol, "decision_session": day,
           "source_ids": [f"source:{symbol}:{day}"], "event_cluster_ids": [f"event:{symbol}:{day}"],
           "feature_values": {"momentum_5d": .02, "tone_score": .3},
           "feature_provenance": {name: {"available_at": local(day, "08:40:00"),
                                        "provenance_id": f"synthetic:{name}:{day}"}
                                  for name in ("momentum_5d", "tone_score")},
           "published_at": local(day, "07:00:00"), "updated_at": local(day, "07:15:00"),
           "response_received_at": local(day, "08:30:00"),
           "extraction_completed_at": local(day, "09:00:00"),
           "prediction_completed_at": local(day, "09:10:00"),
           "historical_availability_verified": True, "sector_symbol": "IGV"}
    row.update(changes)
    return row


def dataset(observations=None):
    sessions = [{"date": day, "open_at": local(day, "09:30:00"), "close_at": local(day, "16:00:00")}
                for day in DAYS]
    return {"data_kind": "synthetic", "as_of": "2026-12-01T23:00:00Z",
            "calendar": {"calendar_id": "synthetic-only", "calendar_version": "v1", "complete": True,
                         "sessions": sessions},
            "prices": [{"symbol": symbol, "session_date": day, "open_value": base + index,
                        "return_basis": "total_return_index", "available_at": local(day, "09:31:00"),
                        "provenance_id": f"synthetic-price:{symbol}:{day}"}
                       for symbol, base in (("PLTR", 100), ("SPY", 200), ("IGV", 50))
                       for index, day in enumerate(DAYS)],
            "observations": observations if observations is not None else [observation()]}


class PriceLabelTests(unittest.TestCase):
    def test_ten_session_open_label_uses_decimal_returns_and_is_pure(self):
        data = dataset()
        before = deepcopy(data)
        row = build_labels(data)[0]
        self.assertEqual(row["status"], "Eligible")
        self.assertEqual(row["label_status"], "Available")
        self.assertTrue(row["historical_availability_verified"])
        self.assertEqual(row["entry_at"], "2026-10-01T13:30:00Z")
        self.assertEqual(row["exit_at"], "2026-10-15T13:30:00Z")
        self.assertEqual(row["label_available_at"], "2026-10-15T13:31:00Z")
        self.assertAlmostEqual(row["stock_return"], .1)
        self.assertAlmostEqual(row["spy_return"], .05)
        self.assertAlmostEqual(row["excess_return"], .05)
        self.assertAlmostEqual(row["excess_vs_sector"], -.1)
        self.assertEqual(row["feature_values"], data["observations"][0]["feature_values"])
        row["feature_values"]["tone_score"] = 999
        self.assertEqual(data, before)
        json.dumps(build_labels(data), allow_nan=False)

    def test_five_and_twenty_sessions_have_distinct_exact_endpoints(self):
        for horizon, expected in ((5, DAYS[5]), (10, DAYS[10]), (20, DAYS[20])):
            with self.subTest(horizon=horizon):
                row = build_labels(dataset(), horizon)[0]
                self.assertEqual(row["exit_at"][:10], expected)
                self.assertEqual(row["horizon_sessions"], horizon)
        for horizon in (0, 1, 7, True, 10.0):
            with self.assertRaisesRegex(ValueError, "5, 10 or 20"):
                build_labels(dataset(), horizon)

    def test_daylight_saving_and_standard_time_cutoffs(self):
        rows = build_labels(dataset([observation(0), observation(22)]))
        self.assertEqual(rows[0]["feature_cutoff"], "2026-10-01T12:45:00Z")
        self.assertEqual(rows[0]["prediction_deadline"], "2026-10-01T13:15:00Z")
        self.assertEqual(rows[1]["feature_cutoff"], "2026-11-02T13:45:00Z")
        self.assertEqual(rows[1]["prediction_deadline"], "2026-11-02T14:15:00Z")
        self.assertEqual(rows[1]["entry_at"], "2026-11-02T14:30:00Z")

    def test_explicit_holiday_gap_is_never_filled(self):
        data = dataset()
        del data["calendar"]["sessions"][3]
        row = build_labels(data, 5)[0]
        self.assertEqual(row["exit_at"][:10], DAYS[6])
        self.assertNotEqual(row["exit_at"][:10], DAYS[5])

    def test_incomplete_calendar_missing_tail_and_non_session_are_unavailable(self):
        data = dataset()
        data["calendar"]["complete"] = False
        row = build_labels(data)[0]
        self.assertEqual(row["status"], "Unavailable")
        self.assertIn("calendar_not_complete", row["reasons"])
        data = dataset([observation(30)])
        row = build_labels(data)[0]
        self.assertIsNone(row["exit_at"])
        self.assertIn("calendar_horizon_missing", row["reasons"])
        data = dataset([observation(decision_session="2026-10-03")])
        row = build_labels(data)[0]
        self.assertIn("decision_session_not_in_calendar", row["reasons"])

    def test_calendar_requires_explicit_version_and_unique_ordered_aware_sessions(self):
        for corruption in ("version", "complete", "duplicate", "order", "timezone", "wrong_day", "open_early"):
            with self.subTest(corruption=corruption):
                data = dataset()
                sessions = data["calendar"]["sessions"]
                if corruption == "version":
                    del data["calendar"]["calendar_version"]
                elif corruption == "complete":
                    del data["calendar"]["complete"]
                elif corruption == "duplicate":
                    sessions.insert(1, deepcopy(sessions[0]))
                elif corruption == "order":
                    sessions.reverse()
                elif corruption == "timezone":
                    sessions[0]["open_at"] = "2026-10-01T09:30:00"
                elif corruption == "wrong_day":
                    sessions[0]["open_at"] = local(DAYS[1], "09:30:00")
                else:
                    sessions[0]["open_at"] = local(DAYS[0], "09:00:00")
                with self.assertRaises(ValueError):
                    build_labels(data)

    def test_missing_endpoint_is_never_replaced_by_another_session(self):
        data = dataset()
        data["prices"] = [point for point in data["prices"]
                          if not (point["symbol"] == "PLTR" and point["session_date"] == DAYS[10])]
        row = build_labels(data)[0]
        self.assertEqual(row["status"], "Unavailable")
        self.assertIsNone(row["excess_return"])
        self.assertIn("required_open_price_missing", row["reasons"])

    def test_duplicate_and_invalid_prices_raise(self):
        data = dataset()
        data["prices"].append(deepcopy(data["prices"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate price"):
            build_labels(data)
        for value in (0, -1, True, "100", float("nan"), float("inf")):
            with self.subTest(value=value):
                data = dataset()
                data["prices"][0]["open_value"] = value
                with self.assertRaises(ValueError):
                    build_labels(data)

    def test_raw_prices_are_only_retrospective_and_mixed_basis_is_unavailable(self):
        data = dataset()
        for point in data["prices"]:
            point["return_basis"] = "raw_price"
        row = build_labels(data)[0]
        self.assertEqual(row["status"], "Retrospective")
        self.assertEqual(row["label_status"], "Available")
        self.assertAlmostEqual(row["excess_return"], .05)
        self.assertIn("raw_price_not_total_return", row["reasons"])
        data["prices"][0]["return_basis"] = "total_return_index"
        row = build_labels(data)[0]
        self.assertEqual(row["status"], "Unavailable")
        self.assertIn("inconsistent_return_basis", row["reasons"])

    def test_not_yet_mature_label_does_not_expose_outcomes(self):
        data = dataset()
        data["as_of"] = "2026-10-15T13:29:59Z"
        row = build_labels(data)[0]
        self.assertEqual(row["status"], "Pending")
        self.assertEqual(row["label_status"], "Pending")
        self.assertIsNone(row["stock_return"])
        self.assertIsNone(row["excess_return"])
        self.assertIn("horizon_not_mature", row["reasons"])

    def test_endpoint_receipt_controls_maturity_even_after_exit(self):
        data = dataset()
        data["as_of"] = "2026-10-15T13:30:30Z"
        row = build_labels(data)[0]
        self.assertEqual(row["status"], "Pending")
        self.assertEqual(row["label_available_at"], "2026-10-15T13:31:00Z")
        self.assertIsNone(row["excess_return"])
        data["prices"][10]["available_at"] = None
        row = build_labels(data)[0]
        self.assertEqual(row["status"], "Unavailable")
        self.assertIn("price_availability_missing", row["reasons"])

    def test_endpoint_price_receipts_before_their_open_are_unavailable(self):
        for symbol, day in (("PLTR", DAYS[0]), ("PLTR", DAYS[10]),
                            ("SPY", DAYS[0]), ("SPY", DAYS[10]), ("IGV", DAYS[10])):
            with self.subTest(symbol=symbol, day=day):
                data = dataset()
                point = next(point for point in data["prices"]
                             if point["symbol"] == symbol and point["session_date"] == day)
                point["available_at"] = local(day, "09:29:59")
                row = build_labels(data)[0]
                self.assertEqual(row["status"], "Unavailable")
                self.assertFalse(row["historical_availability_verified"])
                if symbol == "IGV":
                    self.assertIsNone(row["sector_return"])
                    self.assertIn("sector_price_available_before_open", row["reasons"])
                else:
                    self.assertIsNone(row["excess_return"])
                    self.assertIn("price_available_before_open", row["reasons"])

    def test_empty_source_or_event_groups_never_become_eligible(self):
        for field in ("source_ids", "event_cluster_ids"):
            with self.subTest(field=field):
                data = dataset()
                data["observations"][0][field] = []
                row = build_labels(data)[0]
                self.assertEqual(row["status"], "Retrospective")
                self.assertEqual(row["label_status"], "Available")
                self.assertFalse(row["historical_availability_verified"])
                self.assertIn(field + "_missing", row["reasons"])

    def test_unknown_source_or_feature_evidence_never_becomes_verified(self):
        for field in ("response_received_at", "extraction_completed_at", "prediction_completed_at",
                      "historical_availability_verified", "feature_provenance"):
            with self.subTest(field=field):
                data = dataset()
                del data["observations"][0][field]
                row = build_labels(data)[0]
                self.assertEqual(row["status"], "Retrospective")
                self.assertFalse(row["historical_availability_verified"])
        data = dataset()
        del data["prices"][0]["provenance_id"]
        self.assertEqual(build_labels(data)[0]["status"], "Retrospective")

    def test_known_late_versions_receipts_and_computation_are_unavailable(self):
        for field, clock in (("published_at", "08:46:00"), ("updated_at", "08:46:00"),
                             ("response_received_at", "08:46:00"),
                             ("extraction_completed_at", "09:16:00"),
                             ("prediction_completed_at", "09:16:00")):
            with self.subTest(field=field):
                data = dataset()
                data["observations"][0][field] = local(DAYS[0], clock)
                row = build_labels(data)[0]
                self.assertEqual(row["status"], "Unavailable")
                self.assertFalse(row["historical_availability_verified"])
                self.assertTrue(any(reason.startswith("after_") for reason in row["reasons"]))

    def test_raw_feature_after_0845_is_not_excused_by_early_inference(self):
        data = dataset()
        data["observations"][0]["feature_provenance"]["momentum_5d"]["available_at"] = local(DAYS[0], "09:00:00")
        row = build_labels(data)[0]
        self.assertEqual(row["status"], "Unavailable")
        self.assertIn("feature_available_after_cutoff:momentum_5d", row["reasons"])

    def test_features_cannot_arrive_after_claimed_extraction_or_prediction(self):
        data = dataset()
        item = data["observations"][0]
        item.update(published_at=local(DAYS[0], "06:00:00"), updated_at=local(DAYS[0], "06:30:00"),
                    response_received_at=local(DAYS[0], "07:00:00"),
                    extraction_completed_at=local(DAYS[0], "07:30:00"),
                    prediction_completed_at=local(DAYS[0], "08:00:00"))
        row = build_labels(data)[0]
        self.assertEqual(row["status"], "Unavailable")
        self.assertFalse(row["historical_availability_verified"])
        self.assertEqual(row["signal_available_at"], "2026-10-01T12:40:00Z")
        self.assertFalse(any(reason.startswith("time_order:") for reason in row["reasons"]))
        for name in item["feature_values"]:
            self.assertIn("feature_available_after_extraction_completed_at:" + name, row["reasons"])
            self.assertIn("feature_available_after_prediction_completed_at:" + name, row["reasons"])

    def test_exact_cutoff_and_deadline_are_allowed_but_bad_order_is_rejected(self):
        data = dataset()
        item = data["observations"][0]
        item.update(response_received_at=local(DAYS[0], "08:45:00"),
                    extraction_completed_at=local(DAYS[0], "09:15:00"),
                    prediction_completed_at=local(DAYS[0], "09:15:00"))
        self.assertEqual(build_labels(data)[0]["status"], "Eligible")
        item["prediction_completed_at"] = local(DAYS[0], "09:14:59")
        self.assertEqual(build_labels(data)[0]["status"], "Unavailable")

    def test_outcomes_cannot_enter_features_or_change_existing_feature_values(self):
        data = dataset()
        first = build_labels(data)[0]
        data["prices"][10]["open_value"] = 999
        second = build_labels(data)[0]
        self.assertEqual(first["feature_values"], second["feature_values"])
        self.assertNotEqual(first["excess_return"], second["excess_return"])
        for name in ("excess_return", "stock_return", "future_price", "label_value", "outcome_answer"):
            with self.subTest(name=name):
                data = dataset()
                data["observations"][0]["feature_values"][name] = .3
                with self.assertRaisesRegex(ValueError, "Outcome fields"):
                    build_labels(data)

    def test_sector_is_optional_and_missing_sector_never_invents_a_return(self):
        data = dataset()
        data["prices"] = [point for point in data["prices"] if point["symbol"] != "IGV"]
        row = build_labels(data)[0]
        self.assertEqual(row["status"], "Eligible")
        self.assertIsNone(row["sector_return"])
        self.assertIsNone(row["excess_vs_sector"])
        self.assertAlmostEqual(row["excess_return"], .05)

    def test_naive_input_times_duplicate_company_days_and_nonfinite_features_fail(self):
        for mutation in ("as_of", "receipt", "feature_time", "duplicate", "feature_value"):
            with self.subTest(mutation=mutation):
                data = dataset()
                if mutation == "as_of":
                    data["as_of"] = "2026-12-01T23:00:00"
                elif mutation == "receipt":
                    data["observations"][0]["response_received_at"] = "2026-10-01T08:30:00"
                elif mutation == "feature_time":
                    data["observations"][0]["feature_provenance"]["tone_score"]["available_at"] = "2026-10-01T08:30:00"
                elif mutation == "duplicate":
                    data["observations"].append(observation(sample_id="other"))
                else:
                    data["observations"][0]["feature_values"]["tone_score"] = float("nan")
                with self.assertRaises(ValueError):
                    build_labels(data)


class TimeSplitTests(unittest.TestCase):
    def split(self, rows):
        return make_time_split(rows, local(DAYS[15], "09:00:00"),
                               local(DAYS[15], "09:30:00"), local(DAYS[18], "09:30:00"))

    def test_split_keeps_only_previously_mature_labels_and_later_entries(self):
        rows = build_labels(dataset([observation(i) for i in (0, 5, 6, 15, 17, 18)]))
        before = deepcopy(rows)
        split = self.split(rows)
        self.assertEqual([row["sample_id"] for row in split["train"]], [observation(0)["sample_id"]])
        self.assertEqual([row["sample_id"] for row in split["test"]],
                         [observation(i)["sample_id"] for i in (15, 17)])
        excluded = {row["sample_id"]: row["reasons"] for row in split["excluded"]}
        for index in (5, 6):
            self.assertIn("training_label_not_mature_before_cutoff", excluded[observation(index)["sample_id"]])
        self.assertIn("outside_requested_windows", excluded[observation(18)["sample_id"]])
        split["train"][0]["feature_values"]["tone_score"] = 999
        self.assertEqual(rows, before)

    def test_maturity_equal_to_training_cutoff_is_not_allowed(self):
        rows = build_labels(dataset([observation(0), observation(15)]))
        rows[0]["label_available_at"] = local(DAYS[15], "09:00:00")
        split = self.split(rows)
        self.assertEqual(split["train"], [])
        self.assertIn("training_label_not_mature_before_cutoff", split["excluded"][0]["reasons"])

    def test_shared_article_across_stocks_and_shared_events_are_purged_from_train(self):
        observations = [observation(0), observation(1), observation(15, "SPY"), observation(16, "IGV")]
        observations[0]["source_ids"] = observations[2]["source_ids"] = ["same-article-multistock"]
        observations[1]["event_cluster_ids"] = observations[3]["event_cluster_ids"] = ["same-event"]
        rows = build_labels(dataset(observations))
        split = self.split(rows)
        self.assertEqual(split["train"], [])
        self.assertEqual(len(split["test"]), 2)
        reasons = {item for row in split["excluded"] for item in row["reasons"]}
        self.assertEqual(reasons, {"source_shared_with_test", "event_cluster_shared_with_test"})

    def test_outside_window_bridge_connects_train_and_test_transitively(self):
        observations = [observation(0, source_ids=["u"], event_cluster_ids=["d"]),
                        observation(19, source_ids=["u"], event_cluster_ids=["e"]),
                        observation(15, source_ids=["v"], event_cluster_ids=["e"])]
        rows = build_labels(dataset(observations))
        split = self.split(rows)
        self.assertEqual(split["train"], [])
        self.assertEqual([row["sample_id"] for row in split["test"]], [observations[2]["sample_id"]])
        reasons = {row["sample_id"]: row["reasons"] for row in split["excluded"]}
        self.assertEqual(reasons[observations[0]["sample_id"]], ["transitive_group_shared_with_test"])
        self.assertEqual(reasons[observations[1]["sample_id"]], ["outside_requested_windows"])

    def test_multistock_article_chain_uses_even_retrospective_bridge_ids(self):
        observations = [observation(0, source_ids=["original"], event_cluster_ids=["early-event"]),
                        observation(1, "SPY", source_ids=["original"], event_cluster_ids=["followup"]),
                        observation(19, "IGV", source_ids=["later-copy"], event_cluster_ids=["followup"],
                                    historical_availability_verified=False),
                        observation(15, source_ids=["later-copy"], event_cluster_ids=["test-event"])]
        rows = build_labels(dataset(observations))
        self.assertEqual(rows[2]["status"], "Retrospective")
        split = self.split(rows)
        self.assertEqual(split["train"], [])
        self.assertEqual(len(split["test"]), 1)
        reasons = {row["sample_id"]: row["reasons"] for row in split["excluded"]}
        for observation_ in observations[:2]:
            self.assertEqual(reasons[observation_["sample_id"]], ["transitive_group_shared_with_test"])
        self.assertIn("sample_not_eligible:Retrospective", reasons[observations[2]["sample_id"]])

    def test_reconstructed_raw_unknown_and_pending_rows_are_not_evaluation_samples(self):
        rows = build_labels(dataset([observation(0), observation(15)]))
        for status, label_status in (("Retrospective", "Available"), ("Pending", "Pending"),
                                     ("Unavailable", "Unavailable")):
            with self.subTest(status=status):
                changed = deepcopy(rows)
                changed[0].update(status=status, label_status=label_status)
                changed[1].update(status=status, label_status=label_status)
                split = self.split(changed)
                self.assertEqual(split["train"], [])
                self.assertEqual(split["test"], [])
                self.assertEqual(len(split["excluded"]), 2)

    def test_bad_boundaries_and_duplicate_labels_fail(self):
        rows = build_labels(dataset())
        with self.assertRaisesRegex(ValueError, "training_cutoff"):
            make_time_split(rows, "2026-10-20T00:00:00Z", "2026-10-19T00:00:00Z", "2026-10-21T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            make_time_split(rows, "2026-10-20", "2026-10-21T00:00:00Z", "2026-10-22T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
            self.split(rows + deepcopy(rows))

    def test_training_cutoff_cannot_follow_actual_test_prediction_completion(self):
        rows = build_labels(dataset([observation(0), observation(15)]))
        with self.assertRaisesRegex(ValueError, "follows test prediction"):
            make_time_split(rows, local(DAYS[15], "09:11:00"),
                            local(DAYS[15], "09:30:00"), local(DAYS[18], "09:30:00"))
        allowed = make_time_split(rows, local(DAYS[15], "09:10:00"),
                                  local(DAYS[15], "09:30:00"), local(DAYS[18], "09:30:00"))
        self.assertEqual(len(allowed["test"]), 1)
        rows[1]["prediction_completed_at"] = local(DAYS[15], "09:20:00")
        with self.assertRaisesRegex(ValueError, "follows test prediction"):
            make_time_split(rows, local(DAYS[15], "09:16:00"),
                            local(DAYS[15], "09:30:00"), local(DAYS[18], "09:30:00"))

    def test_split_rejects_mixed_horizons_bases_and_calendar_metadata(self):
        rows = build_labels(dataset([observation(0), observation(15)]))
        for field, other in (("horizon_sessions", 5), ("horizon_sessions", 20),
                             ("return_basis", "raw_price"), ("calendar_id", "other-calendar"),
                             ("calendar_version", "v2")):
            with self.subTest(field=field, other=other):
                changed = deepcopy(rows)
                changed[1][field] = other
                with self.assertRaisesRegex(ValueError, "Mixed " + field):
                    self.split(changed)

    def test_pending_unknown_basis_is_not_a_second_basis_or_a_test_sample(self):
        data = dataset([observation(0), observation(15)])
        data["as_of"] = local(DAYS[20], "12:00:00")
        rows = build_labels(data)
        self.assertEqual(rows[1]["status"], "Pending")
        self.assertIsNone(rows[1]["return_basis"])
        result = self.split(rows)
        self.assertEqual(len(result["train"]), 1)
        self.assertEqual(result["test"], [])


if __name__ == "__main__":
    unittest.main()
