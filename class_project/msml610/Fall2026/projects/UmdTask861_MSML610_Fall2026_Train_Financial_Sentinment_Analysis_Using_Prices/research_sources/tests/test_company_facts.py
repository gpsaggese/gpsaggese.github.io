"""Synthetic SEC data only; no network calls or trading-app imports."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import pytest


_path = Path(__file__).resolve().parents[1] / "company_facts.py"
_spec = importlib.util.spec_from_file_location("company_facts_test_target", _path)
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)
RECEIPT = "2026-10-28T20:00:00+00:00"
IDENTITY = {"cik": "0001045810"}
REVENUE = cf.METRICS["revenue"][0]


def fact(value=100, start="2026-04-01", end="2026-06-30", filed="2026-08-01", accession="original"):
    row = dict(val=value, end=end, filed=filed, accn=accession, fy=2026, fp="Q2", form="10-Q")
    if start is not None:
        row["start"] = start
    return row


def payload(tags):
    return {"cik": 1045810, "entityName": "Synthetic Company", "facts": {"us-gaap": tags}}


def normalize(data, cutoff="2026-10-28"):
    return cf.normalize_company(data, "NVDA", IDENTITY, RECEIPT, cutoff)


def test_cutoff_selects_then_restatement_preserves_both_versions_and_receipt_unknown():
    data = payload({REVENUE: {"units": {"USD": [fact(), fact(80, filed="2026-10-01", accession="amended")]}}})
    old = normalize(data, "2026-09-01")
    new = normalize(data)
    assert len(old["records"]) == len(new["records"]) == 2
    old_value = cf.summarize_company(old)["metrics"]["revenue"]["units"]["USD"]["latest_eight_quarter_length_periods"][0]
    new_value = cf.summarize_company(new)["metrics"]["revenue"]["units"]["USD"]["latest_eight_quarter_length_periods"][0]
    assert old_value["value"] == 100 and new_value["value"] == 80
    assert new_value["version_count"] == 2
    assert old["records"][1]["exclusion_reasons"] == ["filed_after_cutoff"]
    assert all(r["historical_pipeline_received_at"] is None for r in new["records"])
    assert all(r["response_received_at"] == RECEIPT and not r["historical_availability_verified"] for r in new["records"])


def test_q2_filing_keeps_single_quarter_separate_from_ytd_and_never_invents_eight_quarters():
    revenue = [fact(100), fact(190, start="2026-01-01")]
    ocf = [fact(10 * year, start=f"{year}-01-01", end=f"{year}-06-30", filed=f"{year}-08-01", accession=f"ytd-{year}")
           for year in range(2019, 2027)]
    data = payload({REVENUE: {"units": {"USD": revenue}}, cf.METRICS["operating_cash_flow"][0]: {"units": {"USD": ocf}}})
    n = normalize(data)
    summary = cf.summarize_company(n)
    units = summary["metrics"]["revenue"]["units"]["USD"]
    assert units["period_type_counts"] == {"quarter_length_duration": 1, "six_month_cumulative_duration": 1}
    ocf_units = summary["metrics"]["operating_cash_flow"]["units"]["USD"]
    assert ocf_units["period_type_counts"] == {"six_month_cumulative_duration": 8}
    assert not ocf_units["eight_reported_quarter_length_periods_available"]
    assert ocf_units["latest_eight_quarter_length_periods"] == []


def test_revenue_aliases_not_added_and_currency_units_not_mixed():
    data = payload({REVENUE: {"units": {"USD": [fact(100)], "EUR": [fact(90)]}},
                    "Revenues": {"units": {"USD": [fact(100)]}}})
    units = cf.summarize_company(normalize(data))["metrics"]["revenue"]["units"]
    assert units["USD"]["latest_eight_quarter_length_periods"][0]["value"] == 100
    assert units["EUR"]["latest_eight_quarter_length_periods"][0]["value"] == 90
    assert units["USD"]["latest_eight_quarter_length_periods"][0]["available_tags"] == sorted([REVENUE, "Revenues"])


def test_same_filing_conflicting_values_are_not_silently_selected():
    data = payload({REVENUE: {"units": {"USD": [fact(100), fact(101)]}}})
    metric = cf.summarize_company(normalize(data))["metrics"]["revenue"]["units"]["USD"]
    assert metric["latest_eight_quarter_length_periods"] == []
    latest = metric["latest_by_period_type"]["quarter_length_duration"]
    assert latest["value"] is None and latest["conflicting_latest_values"]


def test_multiple_start_dates_at_one_quarter_end_do_not_inflate_quarter_count():
    data = payload({REVENUE: {"units": {"USD": [fact(start=f"2026-04-{day:02d}") for day in range(1, 9)]}}})
    unit = cf.summarize_company(normalize(data))["metrics"]["revenue"]["units"]["USD"]
    assert not unit["eight_reported_quarter_length_periods_available"]
    assert unit["latest_eight_quarter_length_periods"] == []
    assert unit["ambiguous_quarter_end_dates"] == ["2026-06-30"]


def test_wrong_cik_is_rejected_and_future_cutoff_not_permitted():
    data = payload({})
    changed = deepcopy(data)
    changed["cik"] = 123
    with pytest.raises(ValueError, match="CIK"):
        normalize(changed)
    with pytest.raises(ValueError, match="Cutoff"):
        normalize(data, "2026-10-29")


def test_one_get_per_company_http403_retained_without_retry_or_user_agent_disclosure(tmp_path):
    inputs = tmp_path / "identity"
    inputs.mkdir()
    for index, symbol in enumerate(cf.SYMBOLS, start=1):
        (inputs / f"{symbol}_submissions.response").write_text(json.dumps(dict(cik=index, name=f"Fake {symbol}", tickers=[symbol])))
    calls = []
    class Response:
        def read(self, count):
            return b"Access denied"
        def getcode(self):
            return 403
        def close(self):
            pass
    def fake_fetch(url, user_agent):
        calls.append(url)
        return Response()
    out = tmp_path / "capture"
    manifest = cf.collect_company_facts(out, inputs, fetch=fake_fetch,
                                        clock=lambda: datetime(2026, 10, 28, 20, tzinfo=timezone.utc), pause=lambda _: None)
    assert len(calls) == 4 and len(set(calls)) == 4
    assert manifest["status"] == "partial_or_failed" and manifest["company_count"] == 0
    assert all(row["http_status"] == 403 and row["status"] == "http_failed" for row in manifest["requests"])
    assert (out / "NVDA_companyfacts.response").read_bytes() == b"Access denied"
    assert not list(out.glob("*_normalized.json"))
    with pytest.raises(FileExistsError):
        cf.collect_company_facts(out, inputs, fetch=fake_fetch)
    assert len(calls) == 4


def test_one_verified_archived_issuer_can_be_collected_without_guessing_other_identities(tmp_path):
    inputs = tmp_path / "identity"
    inputs.mkdir()
    (inputs / "NVDA_submissions.response").write_text(json.dumps(dict(cik=1045810, name="Synthetic Company", tickers=["NVDA"])))
    calls = []
    class Response:
        def read(self, count):
            return json.dumps(payload({REVENUE: {"units": {"USD": [fact()]}}})).encode()
        def getcode(self):
            return 200
        def close(self):
            pass
    def fake_fetch(url, user_agent):
        calls.append(url)
        return Response()
    manifest = cf.collect_company_facts(tmp_path / "capture", inputs, symbols=["NVDA"], fetch=fake_fetch,
                                        clock=lambda: datetime(2026, 10, 28, 20, tzinfo=timezone.utc), pause=lambda _: None)
    assert len(calls) == 1 and "CIK0001045810.json" in calls[0]
    assert manifest["status"] == "complete" and manifest["company_count"] == 1
    with pytest.raises(ValueError, match="declared company"):
        cf.collect_company_facts(tmp_path / "bad", inputs, symbols=["UNKNOWN"], fetch=fake_fetch)
    assert len(calls) == 1
