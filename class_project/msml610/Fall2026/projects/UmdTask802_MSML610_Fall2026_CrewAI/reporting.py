"""
Create the final FRED briefing report with a summary table and chart.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import fred_tools as ft


def create_summary_table(snapshot_dir: Path) -> str:
    """
    Create a Markdown summary table from deterministic statistics.
    """
    stats = ft.calculate_macro_statistics(snapshot_dir)

    rows = [
        (
            "CPI inflation",
            f'{stats["inflation_yoy_pct"]:.2f}%',
            "Year-over-year",
            stats["inflation_latest_date"],
        ),
        (
            "Unemployment rate",
            f'{stats["unemployment_rate_pct"]:.1f}%',
            f'{stats["unemployment_change_pp"]:+.1f} pp',
            stats["unemployment_latest_date"],
        ),
        (
            "Federal funds rate",
            f'{stats["fed_funds_rate_pct"]:.2f}%',
            f'{stats["fed_funds_change_pp"]:+.2f} pp',
            stats["fed_funds_latest_date"],
        ),
        (
            "10Y–2Y Treasury spread",
            f'{stats["yield_spread_pp"]:.2f} pp',
            stats["yield_curve_sign"].title(),
            stats["yield_spread_latest_date"],
        ),
    ]

    lines = [
        "| Indicator | Latest value | Annual change / status | Date |",
        "|---|---:|---:|---|",
    ]
    lines.extend(
        f"| {indicator} | {value} | {change} | {date} |"
        for indicator, value, change, date in rows
    )
    return "\n".join(lines)


def create_fred_chart(
    snapshot_dir: Path,
    output_path: Path,
) -> None:
    """
    Create a four-panel line chart from the saved FRED snapshot.
    """
    chart_config = {
        "CPIAUCSL": ("Consumer Price Index", "Index"),
        "UNRATE": ("Unemployment Rate", "Percent"),
        "FEDFUNDS": ("Federal Funds Rate", "Percent"),
        "T10Y2Y": (
            "10-Year Minus 2-Year Treasury Spread",
            "Percentage points",
        ),
    }

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8),
        constrained_layout=True,
    )

    for axis, (series_id, (title, ylabel)) in zip(
        axes.flat,
        chart_config.items(),
    ):
        series = ft.load_fred_series(
            snapshot_dir / f"{series_id}.csv",
            series_id,
        )
        axis.plot(
            series.index,
            series.values,
            linewidth=1.4,
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Observation date")
        axis.grid(alpha=0.3)

        if series_id == "T10Y2Y":
            axis.axhline(
                0,
                color="black",
                linewidth=0.8,
                linestyle="--",
            )

    figure.suptitle(
        "FRED Macroeconomic Series",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def export_final_report(
    snapshot_dir: Path,
    briefing_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Export the final Markdown report and chart.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_path = output_dir / "fred_macro_series.png"
    report_path = output_dir / "fred_macro_report.md"

    create_fred_chart(snapshot_dir, chart_path)
    briefing = briefing_path.read_text(encoding="utf-8").strip()
    summary_table = create_summary_table(snapshot_dir)

    report = (
        f"{briefing}\n\n"
        "## Summary Table\n\n"
        f"{summary_table}\n\n"
        "## FRED Series Chart\n\n"
        "![FRED macroeconomic series](fred_macro_series.png)\n"
    )
    report_path.write_text(report, encoding="utf-8")
    return report_path, chart_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the final FRED macro briefing."
    )
    parser.add_argument(
        "--snapshot-dir",
        required=True,
        help="Directory containing the saved FRED CSV files.",
    )
    parser.add_argument(
        "--briefing",
        required=True,
        help="CrewAI-generated Markdown briefing.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for the report and chart.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report_path, chart_path = export_final_report(
        Path(args.snapshot_dir),
        Path(args.briefing),
        Path(args.output_dir),
    )
    print(f"Report saved to: {report_path}")
    print(f"Chart saved to: {chart_path}")


if __name__ == "__main__":
    main()