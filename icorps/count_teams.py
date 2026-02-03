#!/usr/bin/env python

"""
Count teams by National I-Corps Status from a CSV file.

This script accepts a CSV file containing I-Corps team data and provides three
types of analysis:
1. Count teams by status code
2. Count teams by status code for UMD teams only
3. Count teams by NSF Background (HRT/HCT) and UMD status

Example usage:
    > icorps/count_teams.py --input data.csv
    > icorps/count_teams.py --input data.csv --output_file results.txt

The input CSV should have header row with columns:
- "National I-Corps Status" (required)
- "UMD Team" (optional, for Step 2)
- "NSF Background" (optional, for Step 3)
"""

import argparse
import csv
import logging
import os
from typing import Dict, List, Tuple

import helpers.hdbg as hdbg
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)

# Define all valid status codes in order.
_ALL_STATUSES = [
    "03",
    "04",
    "A.01",
    "A.02",
    "A.03",
    "A.04",
    "A.05",
    "A.06",
    "A.07",
    "B.01",
    "B.02",
    "B.03",
    "C.01",
    "C.02",
    "C.03",
    "C.04",
    "D.01",
    "D.02",
    "D.03",
    "D.04",
    "D.05",
    "E.02",
    "E.04",
    "E.08",
]

# Map status codes to their descriptions.
_STATUS_NAMES = {
    "03": "Eligible",
    "04": "Expressed Interest",
    "A.01": "Executive Summary Review",
    "A.02": "Executive Summary Feedback Sent",
    "A.03": "Mentor Search",
    "A.04": "PI Search",
    "A.05": "Send LOR instructions",
    "A.06": "Waiting for LOR Draft from Team",
    "A.07": "Pre-Interview Document Review",
    "B.01": "Going through Regionals",
    "B.02": "Paused",
    "B.03": "Withdrawn",
    "C.01": "Scheduling for Interview",
    "C.02": "Interview Scheduled",
    "C.03": "Interview done",
    "C.04": "Post-Interview Action Required",
    "D.01": "Accepted by Hub",
    "D.02": "Waiting for LOR from Dan",
    "D.03": "Send NSF Application Instructions",
    "D.04": "NSF Application Sent",
    "D.05": "Applied to NSF",
    "E.02": "Accepted by NSF",
    "E.04": "Rejected",
    "E.08": "Completed",
}

# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        action="store",
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output_file",
        action="store",
        default=None,
        help="Path to output file (if not specified, print to stdout)",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _read_csv_data(
    csv_file_path: str,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Read CSV file and return all rows with fieldnames.

    :param csv_file_path: path to CSV file
    :return: tuple of (rows list, fieldnames list)
    """
    # Verify file exists.
    hdbg.dassert(
        os.path.isfile(csv_file_path),
        "Input file does not exist:",
        csv_file_path,
    )
    # Read and parse CSV file.
    rows = []
    with open(csv_file_path, "r", encoding="utf-8") as f:
        csv_reader = csv.DictReader(f)
        fieldnames = csv_reader.fieldnames
        # Verify required column exists.
        hdbg.dassert(
            "National I-Corps Status" in fieldnames,
            "CSV file missing required column:",
            "National I-Corps Status",
        )
        rows = list(csv_reader)
    _LOG.debug("Read %d rows from CSV", len(rows))
    return rows, fieldnames


def _count_teams_by_status(rows: List[Dict[str, str]]) -> Dict[str, int]:
    """
    Count teams by their National I-Corps Status.

    :param rows: list of row dictionaries from CSV
    :return: dictionary mapping status codes to counts
    """
    status_counts: Dict[str, int] = {}
    for row in rows:
        status = row["National I-Corps Status"].strip()
        if status:
            # Extract status code (e.g., "A.04" from "A.04: PI Search").
            status_code = status.split(":")[0].strip()
            status_counts[status_code] = status_counts.get(status_code, 0) + 1
    _LOG.debug("Step 1: Counted %d status entries", sum(status_counts.values()))
    return status_counts


def _count_umd_teams_by_status(
    rows: List[Dict[str, str]],
    *,
    has_umd_column: bool,
) -> Dict[str, int]:
    """
    Count UMD teams by their National I-Corps Status.

    :param rows: list of row dictionaries from CSV
    :param has_umd_column: whether CSV has UMD Team column
    :return: dictionary mapping status codes to counts
    """
    if not has_umd_column:
        _LOG.debug("Step 2: Skipped (no UMD Team column)")
        return {}
    status_counts: Dict[str, int] = {}
    for row in rows:
        # Check if team is UMD team (column value "1" or "true").
        umd_team = row.get("UMD Team", "").strip().lower()
        if umd_team in ["1", "true"]:
            status = row["National I-Corps Status"].strip()
            if status:
                # Extract status code.
                status_code = status.split(":")[0].strip()
                status_counts[status_code] = (
                    status_counts.get(status_code, 0) + 1
                )
    _LOG.debug(
        "Step 2: Counted %d UMD team entries", sum(status_counts.values())
    )
    return status_counts


def _count_nsf_background_matrix(
    rows: List[Dict[str, str]],
    *,
    has_umd_column: bool,
    has_nsf_column: bool,
) -> Dict[str, Dict[str, int]]:
    """
    Count teams by NSF Background and UMD status.

    :param rows: list of row dictionaries from CSV
    :param has_umd_column: whether CSV has UMD Team column
    :param has_nsf_column: whether CSV has NSF Background column
    :return: nested dictionary with counts
    """
    if not has_nsf_column:
        _LOG.debug("Step 3: Skipped (no NSF Background column)")
        return {}
    # Initialize matrix.
    matrix = {
        "UMD": {"HRT": 0, "HCT": 0},
        "Other": {"HRT": 0, "HCT": 0},
    }
    # Track teams with missing NSF background.
    missing_nsf_teams = []
    for row in rows:
        nsf_background = row.get("NSF Background", "").strip()
        # Assert if NSF background is missing.
        if not nsf_background or nsf_background not in ["HRT", "HCT"]:
            team_name = row.get("Team Name", "Unknown")
            missing_nsf_teams.append(team_name)
            continue
        # Determine if UMD team.
        is_umd = False
        if has_umd_column:
            umd_team = row.get("UMD Team", "").strip().lower()
            is_umd = umd_team in ["1", "true"]
        # Update matrix.
        team_type = "UMD" if is_umd else "Other"
        matrix[team_type][nsf_background] += 1
    # Assert if there are teams with missing NSF background.
    if missing_nsf_teams:
        _LOG.warning(
            "Teams with missing or invalid NSF Background:" +
            ", ".join(map(str, missing_nsf_teams))
        )
    _LOG.debug("Step 3: Counted NSF background matrix")
    return matrix


def _format_output(
    status_counts: Dict[str, int],
    umd_status_counts: Dict[str, int],
    nsf_matrix: Dict[str, Dict[str, int]],
    *,
    has_umd_column: bool,
    has_nsf_column: bool,
) -> str:
    """
    Format all results as tab-separated text.

    :param status_counts: counts from step 1
    :param umd_status_counts: counts from step 2
    :param nsf_matrix: matrix from step 3
    :param has_umd_column: whether CSV has UMD Team column
    :param has_nsf_column: whether CSV has NSF Background column
    :return: formatted output string
    """
    lines = []
    # Step 1: All teams by status.
    lines.append("=" * 60)
    lines.append("STEP 1: Count of all teams by status")
    lines.append("=" * 60)
    lines.append("Status\tCount")
    for status_code in _ALL_STATUSES:
        count = status_counts.get(status_code, 0)
        status_name = _STATUS_NAMES[status_code]
        lines.append(f"{status_code}: {status_name}\t{count}")
    lines.append("")
    # Step 2: UMD teams by status.
    if has_umd_column and umd_status_counts:
        lines.append("=" * 60)
        lines.append("STEP 2: Count of UMD teams by status")
        lines.append("=" * 60)
        lines.append("Status\tCount")
        for status_code in _ALL_STATUSES:
            count = umd_status_counts.get(status_code, 0)
            status_name = _STATUS_NAMES[status_code]
            lines.append(f"{status_code}: {status_name}\t{count}")
        lines.append("")
    # Step 3: NSF Background matrix.
    if has_nsf_column and nsf_matrix:
        lines.append("=" * 60)
        lines.append("STEP 3: NSF Background matrix")
        lines.append("=" * 60)
        # Header row.
        lines.append("\tHRT\tHCT\tTotal")
        # UMD row.
        umd_hrt = nsf_matrix["UMD"]["HRT"]
        umd_hct = nsf_matrix["UMD"]["HCT"]
        umd_total = umd_hrt + umd_hct
        lines.append(f"UMD\t{umd_hrt}\t{umd_hct}\t{umd_total}")
        # Other row.
        other_hrt = nsf_matrix["Other"]["HRT"]
        other_hct = nsf_matrix["Other"]["HCT"]
        other_total = other_hrt + other_hct
        lines.append(f"Other\t{other_hrt}\t{other_hct}\t{other_total}")
        # Total row.
        total_hrt = umd_hrt + other_hrt
        total_hct = umd_hct + other_hct
        grand_total = total_hrt + total_hct
        lines.append(f"Total\t{total_hrt}\t{total_hct}\t{grand_total}")
        lines.append("")
    return "\n".join(lines)


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    _LOG.info("Processing CSV file: %s", args.input)
    # Read CSV data.
    rows, fieldnames = _read_csv_data(args.input)
    has_umd_column = "UMD Team" in fieldnames
    has_nsf_column = "NSF Background" in fieldnames
    _LOG.info(
        "CSV columns: UMD Team=%s, NSF Background=%s",
        has_umd_column,
        has_nsf_column,
    )
    # Step 1: Count all teams by status.
    status_counts = _count_teams_by_status(rows)
    # Step 2: Count UMD teams by status.
    umd_status_counts = _count_umd_teams_by_status(
        rows,
        has_umd_column=has_umd_column,
    )
    # Step 3: Count NSF background matrix.
    nsf_matrix = _count_nsf_background_matrix(
        rows,
        has_umd_column=has_umd_column,
        has_nsf_column=has_nsf_column,
    )
    # Format and output results.
    output_text = _format_output(
        status_counts,
        umd_status_counts,
        nsf_matrix,
        has_umd_column=has_umd_column,
        has_nsf_column=has_nsf_column,
    )
    if args.output_file:
        _LOG.info("Writing results to file: %s", args.output_file)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(output_text)
    else:
        print(output_text)


if __name__ == "__main__":
    _main(_parse())
