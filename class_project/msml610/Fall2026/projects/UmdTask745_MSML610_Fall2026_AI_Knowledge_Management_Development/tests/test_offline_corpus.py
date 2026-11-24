"""Integration: the real build output must match the measurements and gold labels in specs/."""
import json

import pytest

from adrqa.config import DATA_DIR, GOLD_DIR, SPECS_DIR

pytestmark = [pytest.mark.corpus, pytest.mark.skipif(not (DATA_DIR / "staleness.json").exists(), reason="run build first")]


def data(name):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def measured():
    return json.loads((SPECS_DIR / "corpus_measurements.json").read_text(encoding="utf-8"))["records"]


def test_extraction_matches_gold():
    gold = json.loads((GOLD_DIR / "extraction.json").read_text(encoding="utf-8"))["records"]
    records = {r["id"]: r for r in data("records.json")}
    for g in gold:
        r = records[g["id"]]
        assert (r["title"], r["date"], r["status"]) == (g["title"], g["date"], g["status"]), g["id"]
        assert r["references"] == sorted(g["references"]), g["id"]
        core = [{k: s[k] for k in ("target", "scope", "statement")} for s in r["supersedes"]]  # ignore source/evidence
        assert core == g["supersedes"], g["id"]
        for field in ("decision", "rationale", "approach"):
            assert r["fields"][field].startswith(g[f"{field}_anchor"]), (g["id"], field)


def test_links_match_measurements(measured):
    links = data("links.json")
    for rid, m in measured.items():
        assert sorted({l["file_id"] for l in links if l["record_id"] == rid}) == m["linked_files"], rid


def test_staleness_matches_measurements(measured):
    for s in data("staleness.json"):
        m = measured[s["record_id"]]
        assert s["stale"] == m["stale"], s["record_id"]
        assert s["churn_code"] == m["churn_code"], s["record_id"]
        assert s["dead_identifiers"] == m["dead_identifiers"], s["record_id"]


def test_eight_scored_two_null_atomikos_zero():
    scores = {s["record_id"]: s for s in data("staleness.json")}
    assert sum(s["stale"] is not None for s in scores.values()) == 8
    assert scores["2026-09-24-remove-atomikos"]["stale"] == 0.0
    assert scores["2024-10-10-daps-deprecation"]["reason"] == "no_linked_files"


def test_graph_has_expected_edges():
    edges = data("graph.json")["edges"]
    kinds = [e["kind"] for e in edges]
    assert kinds.count("supersedes") == 1 and kinds.count("references") == 7


def test_adr_commit_signal_adds_no_links():
    assert all(r["code_files_still_present"] == 0 for r in data("adr_commit_signal.json"))
    assert not [l for l in data("links.json") if l["signal"] == "adr_commit"]


def test_gap_report_shape():
    gaps = data("gap_report.json")
    assert len(gaps) == 20 and gaps == sorted(gaps, key=lambda e: -e["gap"])
