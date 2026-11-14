#!/usr/bin/env python

"""
Break down one run of every agent by snippet type and OWASP category.

Reads the normalized findings in `results/normalized/`, scores them with
`scoring_lib`, prints recall, strict recall, and false-positive tables per
category, and saves the same tables as Markdown.

Strict recall is `N/A` for agents that report no line numbers.

# Break down `r1` and save `results/comparison_tables.md`
> analysis/breakdown.py --run_id r1
"""

import argparse
import logging
import os
from typing import Dict, List

import pandas as pd

import helpers.hdbg as hdbg
import helpers.hparser as hparser

import benchmark_lib as benlib
import findings_lib as finlib
import scoring_lib as scolib

_LOG = logging.getLogger(__name__)

# Label columns to break the scores down by.
_DIMS = ("type", "owasp_2025")

# Metric columns to print, one table each.
_METRICS = ("recall", "recall_strict", "fp")

# #############################################################################


def _fraction(hits: pd.Series) -> str:
    """
    Format the number of true values in `hits` as `k/n`, or `-` if empty.
    """
    if len(hits) == 0:
        return "-"
    return f"{int(hits.sum())}/{len(hits)}"


def _breakdown(
    labels: pd.DataFrame,
    findings: pd.DataFrame,
    runs: pd.DataFrame,
    *,
    line_window: int,
) -> pd.DataFrame:
    """
    Score each run and count hits per category.

    :return: one row per (dim, category, agent) with `_METRICS` as `k/n`
    """
    rows: List[Dict[str, str]] = []
    for run in runs.to_dict("records"):
        mask = (findings["agent"] == run["agent"]) & (
            findings["run_id"] == run["run_id"]
        )
        run_findings = findings[mask]
        scores = scolib.score_snippets(
            labels, run_findings, line_window=line_window
        ).merge(labels[["snippet_id", *_DIMS]], on="snippet_id")
        # Agents without line numbers can't be scored on localization.
        no_lines = (
            len(run_findings) > 0
            and (run_findings["line"] == finlib.NO_LINE).all()
        )
        for dim in _DIMS:
            for category, group in scores.groupby(dim):
                vuln = group[group["vulnerable"]]
                safe = group[~group["vulnerable"]]
                rows.append(
                    {
                        "dim": dim,
                        "category": category,
                        "agent": run["agent"],
                        "recall": _fraction(vuln["flagged"]),
                        "recall_strict": (
                            "N/A"
                            if no_lines
                            else _fraction(vuln["localized_strict"])
                        ),
                        "fp": _fraction(safe["flagged"]),
                    }
                )
    return pd.DataFrame(rows)


def _to_markdown(df: pd.DataFrame) -> str:
    """
    Render a pivot table as a GitHub Markdown table.
    """
    header = [df.index.name or "", *map(str, df.columns)]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for index, row in df.iterrows():
        cells = [str(index), *("" if pd.isna(v) else str(v) for v in row)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "--benchmark_dir",
        action="store",
        default="benchmark",
        help="Dir containing the snippets and `labels.csv`",
    )
    parser.add_argument(
        "--results_dir",
        action="store",
        default="results/normalized",
        help="Dir with normalized findings",
    )
    parser.add_argument(
        "--run_id",
        action="store",
        default="r1",
        help="Run to break down, for every agent that has it",
    )
    parser.add_argument(
        "--output_file",
        action="store",
        default="results/comparison_tables.md",
        help="Markdown file to write the tables to",
    )
    parser.add_argument(
        "--line_window",
        action="store",
        type=int,
        default=scolib.LINE_WINDOW,
        help="Max distance between a finding and a bug line",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    labels = benlib.load_labels(os.path.join(args.benchmark_dir, "labels.csv"))
    findings = finlib.read_findings(args.results_dir)
    runs = finlib.read_runs(args.results_dir)
    runs = runs[runs["run_id"] == args.run_id]
    hdbg.dassert_lt(0, len(runs), "No runs with run_id '%s'", args.run_id)
    df = _breakdown(labels, findings, runs, line_window=args.line_window)
    agents = ", ".join(sorted(runs["agent"]))
    sections = [
        f"# Per-category comparison: run `{args.run_id}`",
        "",
        f"Agents: {agents}. Cells are `hits/total`; recall and strict recall "
        "count vulnerable snippets, fp counts safe snippets. Strict recall is "
        "`N/A` for agents that report no line numbers.",
        "",
        f"Generated by `analysis/breakdown.py --run_id {args.run_id}`.",
    ]
    for dim in _DIMS:
        dim_df = df[df["dim"] == dim]
        for metric in _METRICS:
            table = dim_df.pivot(
                index="category", columns="agent", values=metric
            )
            print(f"\n== {dim}: {metric} ==")
            print(table.to_string())
            sections += ["", f"## {dim}: {metric}", "", _to_markdown(table)]
    with open(args.output_file, "w") as f:
        f.write("\n".join(sections) + "\n")
    _LOG.info("Wrote tables to '%s'", args.output_file)


if __name__ == "__main__":
    _main(_parse())
