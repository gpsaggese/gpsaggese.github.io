"""
Run a reproducible multi-snapshot FRED briefing evaluation.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import evaluation
import fred_tools as ft


DEFAULT_CUTOFFS = (
    "2019-06-30",
    "2020-05-31",
    "2022-06-30",
    "2023-07-31",
)


def prepare_snapshot(
    cutoff: str,
    data_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Download and truncate one FRED snapshot at a historical cutoff."""

    snapshot_dir = data_dir / cutoff
    expected_paths = [
        snapshot_dir / f"{series_id}.csv"
        for series_id in ft.FRED_SERIES
    ]

    if not force and all(path.exists() for path in expected_paths):
        return snapshot_dir

    paths = ft.download_all_fred_series(
        data_dir=str(data_dir),
        download_date=cutoff,
        observation_cutoff=cutoff,
    )
    return next(iter(paths.values())).parent


def summarize_snapshot(
    cutoff: str,
    snapshot_dir: Path,
) -> dict[str, Any]:
    """Return the deterministic economic conditions for one snapshot."""

    profiles = ft.profile_fred_snapshot(snapshot_dir)
    statistics = ft.calculate_macro_statistics(snapshot_dir)

    return {
        "cutoff": cutoff,
        "latest_dates": {
            series_id: profile["latest_date"]
            for series_id, profile in profiles.items()
        },
        "inflation_yoy_pct": statistics["inflation_yoy_pct"],
        "unemployment_rate_pct": statistics[
            "unemployment_rate_pct"
        ],
        "fed_funds_rate_pct": statistics["fed_funds_rate_pct"],
        "yield_spread_pp": statistics["yield_spread_pp"],
        "yield_curve_sign": statistics["yield_curve_sign"],
    }


def evaluate_snapshot_outputs(
    snapshot_dir: Path,
    crew_text: str,
    baseline_text: str,
) -> dict[str, Any]:
    """Evaluate CrewAI and baseline outputs with identical rules."""

    expected_claims = evaluation.get_expected_claims(snapshot_dir)
    expected_values = evaluation.get_expected_values(snapshot_dir)

    crew = evaluation.evaluate_required_claims(
        crew_text,
        expected_claims,
    )
    crew["all_numeric_claims"] = evaluation.evaluate_numeric_claims(
        crew_text,
        expected_values,
    )

    baseline = evaluation.evaluate_required_claims(
        baseline_text,
        expected_claims,
    )
    baseline["all_numeric_claims"] = (
        evaluation.evaluate_numeric_claims(
            baseline_text,
            expected_values,
        )
    )

    return {
        "crew": crew,
        "single_llm_baseline": baseline,
    }


def aggregate_reports(
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate required-claim metrics across all snapshots."""

    methods = {}
    for method in ("crew", "single_llm_baseline"):
        results = [report[method] for report in reports.values()]
        snapshot_count = len(results)

        total_correct = sum(
            result["correct_claims"] for result in results
        )
        total_required = sum(
            result["required_claims"] for result in results
        )

        methods[method] = {
            "snapshot_count": snapshot_count,
            "mean_accuracy": round(
                sum(result["accuracy"] for result in results)
                / snapshot_count,
                4,
            ),
            "mean_completeness": round(
                sum(result["completeness"] for result in results)
                / snapshot_count,
                4,
            ),
            "mean_fraction_correct": round(
                sum(result["fraction_correct"] for result in results)
                / snapshot_count,
                4,
            ),
            "total_correct_claims": total_correct,
            "total_required_claims": total_required,
            "pooled_fraction_correct": round(
                total_correct / total_required,
                4,
            ),
        }

    return {
        "snapshot_count": len(reports),
        "cutoffs": list(reports),
        "methods": methods,
    }


def run_crew_for_snapshot(
    cutoff: str,
    data_dir: Path,
    llm: str,
    project_dir: Path,
) -> str:
    """Run the CrewAI CLI and return its generated briefing."""

    command = [
        sys.executable,
        "crewai.example.py",
        "--data-dir",
        str(data_dir),
        "--download-date",
        cutoff,
        "--observation-cutoff",
        cutoff,
        "--llm",
        llm,
    ]
    subprocess.run(
        command,
        cwd=project_dir,
        check=True,
        env=os.environ.copy(),
    )

    generated_path = (
        project_dir
        / "outputs"
        / f"fred_macro_briefing_{cutoff}.md"
    )
    return generated_path.read_text(encoding="utf-8")


def render_markdown_summary(
    reports: dict[str, dict[str, Any]],
    aggregate: dict[str, Any],
) -> str:
    """Render per-snapshot and aggregate results as Markdown."""

    lines = [
        "# Multi-Snapshot Evaluation",
        "",
        "| Cutoff | Crew score | Crew completeness | "
        "Baseline score | Baseline completeness |",
        "|---|---:|---:|---:|---:|",
    ]

    for cutoff, report in reports.items():
        crew = report["crew"]
        baseline = report["single_llm_baseline"]
        lines.append(
            f"| {cutoff} | {crew['fraction_correct']:.4f} | "
            f"{crew['completeness']:.4f} | "
            f"{baseline['fraction_correct']:.4f} | "
            f"{baseline['completeness']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Results",
            "",
            "| Method | Correct | Required | Mean accuracy | "
            "Mean completeness | Pooled score |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    labels = {
        "crew": "Three-agent CrewAI workflow",
        "single_llm_baseline": "Single-LLM baseline",
    }
    for method, result in aggregate["methods"].items():
        lines.append(
            f"| {labels[method]} | "
            f"{result['total_correct_claims']} | "
            f"{result['total_required_claims']} | "
            f"{result['mean_accuracy']:.4f} | "
            f"{result['mean_completeness']:.4f} | "
            f"{result['pooled_fraction_correct']:.4f} |"
        )

    lines.extend(
        [
            "",
            "The snapshots use current FRED series truncated at historical "
            "observation dates. They are not revision-aware ALFRED vintages.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare and optionally evaluate historical FRED snapshots."
        )
    )
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        default=list(DEFAULT_CUTOFFS),
        help="Historical observation cutoff dates.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/fred/historical",
        help="Root directory for historical snapshots.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/multi_snapshot",
        help="Directory for per-snapshot and aggregate results.",
    )
    parser.add_argument(
        "--llm",
        default=os.getenv("CREWAI_MODEL"),
        help="Model identifier used for both evaluated methods.",
    )
    parser.add_argument(
        "--run-llm",
        action="store_true",
        help="Run CrewAI and baseline model calls.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate cached snapshots and model outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare snapshots and optionally run the full comparison."""

    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_llm and not args.llm:
        raise ValueError(
            "Provide --llm or set CREWAI_MODEL when using --run-llm."
        )

    reports = {}
    summaries = []

    for cutoff in args.cutoffs:
        snapshot_dir = prepare_snapshot(
            cutoff,
            data_dir,
            force=args.force,
        )
        summary = summarize_snapshot(cutoff, snapshot_dir)
        summaries.append(summary)

        cutoff_output_dir = output_dir / cutoff
        cutoff_output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = cutoff_output_dir / "snapshot_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )

        if not args.run_llm:
            continue

        crew_path = cutoff_output_dir / "crew_briefing.md"
        baseline_path = (
            cutoff_output_dir / "single_llm_baseline.md"
        )

        if args.force or not crew_path.exists():
            crew_text = run_crew_for_snapshot(
                cutoff,
                data_dir,
                args.llm,
                project_dir,
            )
            crew_path.write_text(
                crew_text.rstrip() + "\n",
                encoding="utf-8",
            )
        else:
            crew_text = crew_path.read_text(encoding="utf-8")

        if args.force or not baseline_path.exists():
            baseline_text = evaluation.generate_baseline(
                snapshot_dir,
                args.llm,
            )
            baseline_path.write_text(
                baseline_text.rstrip() + "\n",
                encoding="utf-8",
            )
        else:
            baseline_text = baseline_path.read_text(
                encoding="utf-8"
            )

        report = evaluate_snapshot_outputs(
            snapshot_dir,
            crew_text,
            baseline_text,
        )
        reports[cutoff] = report

        report_path = cutoff_output_dir / "evaluation_report.json"
        report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

    preparation_path = output_dir / "snapshot_summaries.json"
    preparation_path.write_text(
        json.dumps(summaries, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.run_llm:
        print(json.dumps(summaries, indent=2))
        print(f"Snapshot summaries saved to: {preparation_path}")
        print("No LLM calls were made.")
        return

    aggregate = aggregate_reports(reports)
    aggregate_path = output_dir / "aggregate_report.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )

    markdown_path = output_dir / "aggregate_report.md"
    markdown_path.write_text(
        render_markdown_summary(reports, aggregate),
        encoding="utf-8",
    )

    print(json.dumps(aggregate, indent=2))
    print(f"Aggregate JSON saved to: {aggregate_path}")
    print(f"Aggregate Markdown saved to: {markdown_path}")


if __name__ == "__main__":
    main()
