"""
Generate a single-LLM baseline and evaluate numeric claims.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from crewai import LLM

import fred_tools as ft


_MONTHS = (
    "January|February|March|April|May|June|July|August|"
    "September|October|November|December"
)


def generate_baseline(
    snapshot_dir: Path,
    llm_name: str,
) -> str:
    """
    Generate a macro briefing using one LLM and no tools.
    """
    profiles = ft.profile_fred_snapshot(snapshot_dir)
    prompt = f"""
You are writing a concise macroeconomic briefing.

Using only the raw FRED profile below, calculate and discuss:
- year-over-year CPI inflation,
- the change in unemployment,
- the change in the federal funds rate,
- the sign of the yield curve spread.

Include inflation, labor, interest rates, and the yield curve.
Include observation dates and keep the response below 350 words.
Do not use external information.

Raw profile:
{json.dumps(profiles, indent=2)}
"""
    llm = LLM(model=llm_name)
    return str(llm.call(prompt))


def _remove_dates_and_series_names(text: str) -> str:
    """Remove dates and series-name numbers before extracting claims."""
    text = re.sub(
        rf"\b(?:{_MONTHS})\s+\d{{1,2}},\s+\d{{4}}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", text)
    text = re.sub(
        r"\b10-Year Minus 2-Year\b",
        "yield spread",
        text,
        flags=re.IGNORECASE,
    )
    return text


def extract_numeric_claims(text: str) -> list[dict[str, Any]]:
    """
    Extract non-date numeric claims from a briefing.
    """
    cleaned = _remove_dates_and_series_names(text)
    pattern = r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?(?![\w.])"
    claims = []

    for match in re.finditer(pattern, cleaned):
        raw_value = match.group(0)
        numeric_text = raw_value.rstrip("%")
        value = float(numeric_text)

        # Ignore standalone years that were not part of a recognized date.
        if value.is_integer() and 1900 <= value <= 2100:
            continue

        decimals = (
            len(numeric_text.split(".", maxsplit=1)[1])
            if "." in numeric_text
            else 0
        )
        claims.append(
            {
                "claim": raw_value,
                "value": value,
                "decimals": decimals,
            }
        )

    return claims


def get_expected_values(snapshot_dir: Path) -> list[float]:
    """
    Recompute valid numeric values directly from the saved CSV files.
    """
    profiles = ft.profile_fred_snapshot(snapshot_dir)
    statistics = ft.calculate_macro_statistics(snapshot_dir)

    expected_values = []
    for profile in profiles.values():
        expected_values.append(float(profile["latest_value"]))
        if profile["prior_year_value"] is not None:
            expected_values.append(float(profile["prior_year_value"]))

    for value in statistics.values():
        if isinstance(value, (int, float)):
            expected_values.append(float(value))
            expected_values.append(abs(float(value)))

    return expected_values


def evaluate_numeric_claims(
    text: str,
    expected_values: list[float],
) -> dict[str, Any]:
    """
    Compute the fraction of numeric claims matching within rounding.
    """
    claims = extract_numeric_claims(text)
    evaluated_claims = []

    for claim in claims:
        tolerance = 0.5 * (10 ** (-claim["decimals"])) + 1e-12
        is_correct = any(
            abs(claim["value"] - expected) <= tolerance
            for expected in expected_values
        )
        evaluated_claims.append(
            {
                **claim,
                "correct": is_correct,
            }
        )

    correct_count = sum(
        claim["correct"] for claim in evaluated_claims
    )
    total_count = len(evaluated_claims)
    fraction_correct = (
        correct_count / total_count if total_count else 0.0
    )

    return {
        "correct_claims": correct_count,
        "total_claims": total_count,
        "fraction_correct": round(fraction_correct, 4),
        "claims": evaluated_claims,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the CrewAI brief with a single-LLM baseline."
    )
    parser.add_argument(
        "--snapshot-dir",
        required=True,
        help="Directory containing the saved FRED CSV files.",
    )
    parser.add_argument(
        "--crew-brief",
        required=True,
        help="Markdown file produced by the CrewAI workflow.",
    )
    parser.add_argument(
        "--llm",
        default=os.getenv("CREWAI_MODEL"),
        help="CrewAI model identifier.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for baseline and evaluation results.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.llm:
        raise ValueError(
            "Provide --llm or set the CREWAI_MODEL environment variable."
        )

    snapshot_dir = Path(args.snapshot_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crew_text = Path(args.crew_brief).read_text(encoding="utf-8")
    baseline_text = generate_baseline(snapshot_dir, args.llm)

    baseline_path = output_dir / "single_llm_baseline.md"
    baseline_path.write_text(
        baseline_text + "\n",
        encoding="utf-8",
    )

    expected_values = get_expected_values(snapshot_dir)
    report = {
        "crew": evaluate_numeric_claims(
            crew_text,
            expected_values,
        ),
        "single_llm_baseline": evaluate_numeric_claims(
            baseline_text,
            expected_values,
        ),
    }

    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "Crew accuracy:",
        report["crew"]["fraction_correct"],
    )
    print(
        "Single-LLM baseline accuracy:",
        report["single_llm_baseline"]["fraction_correct"],
    )
    print(f"Baseline saved to: {baseline_path}")
    print(f"Evaluation saved to: {report_path}")


if __name__ == "__main__":
    main()