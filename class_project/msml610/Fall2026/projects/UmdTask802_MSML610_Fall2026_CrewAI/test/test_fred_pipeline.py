"""
Tests for the FRED macro briefing pipeline.
"""

from pathlib import Path

import pandas as pd
import pytest

import evaluation
import fred_tools as ft
import reporting


_TEST_REQUIRED_VALUES = {
    "cpi_latest": "110.0",
    "cpi_prior_year": "100.0",
    "inflation_yoy": "10.0%",
    "unemployment_latest": "4.5%",
    "unemployment_prior_year": "5.0%",
    "unemployment_change": "-0.5",
    "fed_funds_latest": "3.5%",
    "fed_funds_prior_year": "4.0%",
    "fed_funds_change": "-0.5",
    "yield_spread_latest": "0.5",
    "yield_spread_prior_year": "-0.25",
    "yield_curve_sign": "positive",
}


def _build_required_claim_table(
    *,
    overrides: dict[str, str] | None = None,
    omitted: set[str] | None = None,
) -> str:
    """Build a labeled Markdown claims table for evaluation tests."""

    values = dict(_TEST_REQUIRED_VALUES)
    values.update(overrides or {})
    omitted = omitted or set()
    rows = [
        "| Claim | Value |",
        "|---|---|",
    ]
    rows.extend(
        f"| {claim} | {value} |"
        for claim, value in values.items()
        if claim not in omitted
    )
    return "\n".join(rows)


def _write_series(
    snapshot_dir: Path,
    series_id: str,
    dates: list[str],
    values: list[float],
) -> None:
    """Write a small synthetic FRED-format CSV file."""
    dataframe = pd.DataFrame(
        {
            "observation_date": dates,
            series_id: values,
        }
    )
    dataframe.to_csv(
        snapshot_dir / f"{series_id}.csv",
        index=False,
    )


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> Path:
    """Create a deterministic four-series FRED snapshot."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    dates = [
        "2025-01-01",
        "2026-01-01",
    ]
    _write_series(
        snapshot,
        "CPIAUCSL",
        dates,
        [100.0, 110.0],
    )
    _write_series(
        snapshot,
        "UNRATE",
        dates,
        [5.0, 4.5],
    )
    _write_series(
        snapshot,
        "FEDFUNDS",
        dates,
        [4.0, 3.5],
    )
    _write_series(
        snapshot,
        "T10Y2Y",
        dates,
        [-0.25, 0.5],
    )
    return snapshot


def test_calculate_macro_statistics(
    snapshot_dir: Path,
) -> None:
    """Check all deterministic macro calculations."""
    statistics = ft.calculate_macro_statistics(snapshot_dir)

    assert statistics["inflation_yoy_pct"] == pytest.approx(10.0)
    assert statistics["unemployment_rate_pct"] == pytest.approx(4.5)
    assert statistics["unemployment_change_pp"] == pytest.approx(-0.5)
    assert statistics["fed_funds_rate_pct"] == pytest.approx(3.5)
    assert statistics["fed_funds_change_pp"] == pytest.approx(-0.5)
    assert statistics["yield_spread_pp"] == pytest.approx(0.5)
    assert statistics["yield_curve_sign"] == "positive"


def test_evaluate_numeric_claims() -> None:
    """Check correct and incorrect numeric-claim detection."""
    text = (
        "Inflation was 10.0%, unemployment was 4.5%, "
        "and an unsupported claim was 99.0%."
    )
    result = evaluation.evaluate_numeric_claims(
        text,
        [10.0, 4.5],
    )

    assert result["correct_claims"] == 2
    assert result["total_claims"] == 3
    assert result["fraction_correct"] == pytest.approx(2 / 3, abs=1e-4)


def test_evaluate_required_claims_complete(
    snapshot_dir: Path,
) -> None:
    """Check a complete and correct required-claims table."""

    expected = evaluation.get_expected_claims(snapshot_dir)
    text = _build_required_claim_table()
    result = evaluation.evaluate_required_claims(text, expected)

    assert result["correct_claims"] == 12
    assert result["submitted_claims"] == 12
    assert result["required_claims"] == 12
    assert result["accuracy"] == 1.0
    assert result["completeness"] == 1.0
    assert result["fraction_correct"] == 1.0


def test_evaluate_required_claims_missing_and_incorrect(
    snapshot_dir: Path,
) -> None:
    """Penalize an incorrect field and an omitted required field."""

    expected = evaluation.get_expected_claims(snapshot_dir)
    text = _build_required_claim_table(
        overrides={"cpi_latest": "999.0"},
        omitted={"yield_spread_prior_year"},
    )
    result = evaluation.evaluate_required_claims(text, expected)

    assert result["correct_claims"] == 10
    assert result["submitted_claims"] == 11
    assert result["required_claims"] == 12
    assert result["accuracy"] == pytest.approx(10 / 11, abs=1e-4)
    assert result["completeness"] == pytest.approx(11 / 12, abs=1e-4)
    assert result["fraction_correct"] == pytest.approx(
        10 / 12,
        abs=1e-4,
    )


def test_evaluate_required_claims_rounding(
    snapshot_dir: Path,
) -> None:
    """Accept a correctly rounded value under the correct label."""

    expected = evaluation.get_expected_claims(snapshot_dir)
    expected["inflation_yoy"] = 3.717250844
    text = _build_required_claim_table(
        overrides={"inflation_yoy": "3.72%"},
    )
    result = evaluation.evaluate_required_claims(text, expected)

    inflation_claim = next(
        claim
        for claim in result["claims"]
        if claim["claim"] == "inflation_yoy"
    )
    assert inflation_claim["correct"] is True

def test_export_final_report(
    snapshot_dir: Path,
    tmp_path: Path,
) -> None:
    """Check Markdown and chart export."""
    briefing_path = tmp_path / "briefing.md"
    briefing_path.write_text(
        "# Test Briefing\n\nAll values are verified.\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"

    report_path, chart_path = reporting.export_final_report(
        snapshot_dir,
        briefing_path,
        output_dir,
    )

    assert report_path.exists()
    assert chart_path.exists()
    assert chart_path.stat().st_size > 0

    report_text = report_path.read_text(encoding="utf-8")
    assert "## Summary Table" in report_text
    assert "10.00%" in report_text
    assert "fred_macro_series.png" in report_text