#!/usr/bin/env python

"""
Merge MHTML CSV files with separate name and data sections.

This script processes CSV files that have:
- A header line with format #People(N)
- A section with names (one per line)
- A section with CSV data (comma-separated)

The script merges these sections into a proper CSV with standardized columns.

Import as:

import ck_marketing.plugins.pitchbook.merge_mhtml_csv_files as ckplpime

Examples:
# Merge a single file.
> merge_mhtml_csv_files.py --input_file output/out1.csv --output_file merged.csv

# Merge multiple files.
> merge_mhtml_csv_files.py --input_file output/out1.csv --input_file output/out2.csv --output_file merged.csv

# Merge all CSV files in a directory.
> merge_mhtml_csv_files.py --input_dir output --output_file merged.csv
"""

import argparse
import csv
import glob
import io
import logging
import os
from typing import List, Tuple

import helpers.hdbg as hdbg
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)

# Expected output columns (20 columns total).
OUTPUT_COLUMNS = [
    "People",
    "LinkedIn URL",
    "Last Name",
    "First Name",
    "Primary Company",
    "Primary Position",
    "Biography",
    "Board Seats",
    "Roles",
    "Deal Roles",
    "Location",
    "Address Line 1",
    "Address Line 2",
    "City",
    "State/Province",
    "Post Code",
    "Country/Territory/Region",
    "Phone",
    "Fax",
    "Email",
]


# #############################################################################
# File parsing
# #############################################################################


def _parse_input_file(file_path: str) -> Tuple[List[str], List[List[str]]]:
    """
    Parse an input CSV file with separate name and data sections.

    The file format is:
    - Line 1: #People(N) - header with count
    - Lines 2-K: Names (one per line)
    - Lines K+1-end: CSV data rows

    :param file_path: path to the input file
    :return: tuple of (names list, csv data rows list)
    """
    _LOG.info("Reading input file='%s'", file_path)
    hdbg.dassert_path_exists(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    hdbg.dassert_lte(
        2, len(lines), "File must have at least 2 lines (header + data)"
    )
    # Parse header.
    header = lines[0].strip()
    # Remove surrounding quotes if present.
    if header.startswith('"') and header.endswith('"'):
        header = header[1:-1]
    _LOG.debug("Header: %s", header)
    hdbg.dassert(
        header.startswith("#People("),
        "First line must start with '#People(' but got: %s",
        header,
    )
    # Find where CSV data starts (lines with URLs or many commas).
    csv_start_idx = None
    for i in range(1, len(lines)):
        line = lines[i]
        if "http" in line or line.count(",") > 5:
            csv_start_idx = i
            _LOG.debug("CSV data starts at line %s (1-indexed: %s)", i, i + 1)
            break
    hdbg.dassert_is_not(
        csv_start_idx,
        None,
        "Could not find CSV data section in file: %s",
        file_path,
    )
    # Extract names (lines between header and CSV data start).
    names = [line.strip() for line in lines[1:csv_start_idx] if line.strip()]
    _LOG.info("Parsed %s names from file", len(names))
    # Extract CSV data rows.
    # Note: We need to use csv.reader properly to handle multi-line quoted fields.
    # Join the CSV section back into a string and parse it with io.StringIO.
    csv_content = "".join(lines[csv_start_idx:])
    csv_rows = []
    try:
        reader = csv.reader(io.StringIO(csv_content))
        for row in reader:
            if row:  # Skip empty rows.
                csv_rows.append(row)
    except Exception as e:
        _LOG.error("Failed to parse CSV data section: %s", e)
        raise
    _LOG.info("Parsed %s CSV data rows from file", len(csv_rows))
    return names, csv_rows


def _normalize_csv_row(name: str, csv_row: List[str]) -> List[str]:
    """
    Normalize a CSV row to have all 20 expected output columns.

    The input CSV row has variable fields:
    - Field 0: LinkedIn URL
    - Field 1: Last Name
    - Field 2: First Name
    - Field 3: Primary Company
    - Field 4: Primary Position
    - Field 5: Biography
    - Field 6: Board Seats (number)
    - Field 7: Roles (number)
    - Field 8: Deal Roles (number)
    - Field 9: Location
    - Field 10+: Variable address/contact fields

    :param name: the person's full name from the names section
    :param csv_row: the CSV data row with variable number of fields
    :return: normalized row with exactly 20 fields matching OUTPUT_COLUMNS
    """
    # Create output row with empty strings.
    output_row = [""] * len(OUTPUT_COLUMNS)
    # Set the People column (from names section).
    output_row[0] = name
    # Map fields from CSV row to output columns.
    # Handle variable number of fields in input.
    field_count = len(csv_row)
    if field_count >= 1:
        output_row[1] = csv_row[0]  # LinkedIn URL
    if field_count >= 2:
        output_row[2] = csv_row[1]  # Last Name
    if field_count >= 3:
        output_row[3] = csv_row[2]  # First Name
    if field_count >= 4:
        output_row[4] = csv_row[3]  # Primary Company
    if field_count >= 5:
        output_row[5] = csv_row[4]  # Primary Position
    if field_count >= 6:
        output_row[6] = csv_row[5]  # Biography
    if field_count >= 7:
        output_row[7] = csv_row[6]  # Board Seats
    if field_count >= 8:
        output_row[8] = csv_row[7]  # Roles
    if field_count >= 9:
        output_row[9] = csv_row[8]  # Deal Roles
    if field_count >= 10:
        output_row[10] = csv_row[9]  # Location
    # Handle variable address/contact fields.
    # The remaining fields can be:
    # - Address Line 1, Address Line 2, City, State, Zip, Country, Phone, Fax, Email
    # - Address Line 1, City, State, Zip, Country, Phone, Fax, Email (no Address Line 2)
    # - Address Line 1, City, State, Zip, Country, Phone, Email (no Address Line 2, no Fax)
    # etc.
    #
    # We need to intelligently map these based on field count.
    remaining_fields = csv_row[10:] if field_count > 10 else []
    num_remaining = len(remaining_fields)
    if num_remaining >= 1:
        output_row[11] = remaining_fields[0]  # Address Line 1
    if num_remaining >= 2:
        # Could be Address Line 2 or City - hard to distinguish.
        # Use heuristic: if next field looks like a city name (short), this is Address Line 2.
        # Otherwise, this is City and there's no Address Line 2.
        # For simplicity, we'll assume:
        # - If we have 9 remaining fields: all fields present
        # - If we have 8 remaining fields: missing Fax
        # - If we have 7 remaining fields: missing Address Line 2 and Fax
        # - etc.
        if num_remaining == 9:
            # All fields present.
            output_row[12] = remaining_fields[1]  # Address Line 2
            output_row[13] = remaining_fields[2]  # City
            output_row[14] = remaining_fields[3]  # State/Province
            output_row[15] = remaining_fields[4]  # Post Code
            output_row[16] = remaining_fields[5]  # Country/Territory/Region
            output_row[17] = remaining_fields[6]  # Phone
            output_row[18] = remaining_fields[7]  # Fax
            output_row[19] = remaining_fields[8]  # Email
        elif num_remaining == 8:
            # Missing Fax.
            output_row[12] = remaining_fields[1]  # Address Line 2
            output_row[13] = remaining_fields[2]  # City
            output_row[14] = remaining_fields[3]  # State/Province
            output_row[15] = remaining_fields[4]  # Post Code
            output_row[16] = remaining_fields[5]  # Country/Territory/Region
            output_row[17] = remaining_fields[6]  # Phone
            output_row[19] = remaining_fields[7]  # Email
        elif num_remaining == 7:
            # Missing Address Line 2 and Fax.
            output_row[13] = remaining_fields[1]  # City
            output_row[14] = remaining_fields[2]  # State/Province
            output_row[15] = remaining_fields[3]  # Post Code
            output_row[16] = remaining_fields[4]  # Country/Territory/Region
            output_row[17] = remaining_fields[5]  # Phone
            output_row[19] = remaining_fields[6]  # Email
        elif num_remaining == 6:
            # Missing Address Line 2, Fax, and Phone.
            output_row[13] = remaining_fields[1]  # City
            output_row[14] = remaining_fields[2]  # State/Province
            output_row[15] = remaining_fields[3]  # Post Code
            output_row[16] = remaining_fields[4]  # Country/Territory/Region
            output_row[19] = remaining_fields[5]  # Email
        elif num_remaining == 5:
            # Missing Address Line 2, Fax, Phone, and Email.
            output_row[13] = remaining_fields[1]  # City
            output_row[14] = remaining_fields[2]  # State/Province
            output_row[15] = remaining_fields[3]  # Post Code
            output_row[16] = remaining_fields[4]  # Country/Territory/Region
        elif num_remaining == 4:
            # Only City, State, Zip, Country.
            output_row[13] = remaining_fields[1]  # City
            output_row[14] = remaining_fields[2]  # State/Province
            output_row[15] = remaining_fields[3]  # Post Code
            # Note: Could be Country, but hard to distinguish.
        elif num_remaining == 3:
            # Only City, State, Country.
            output_row[13] = remaining_fields[1]  # City
            output_row[14] = remaining_fields[2]  # State/Province
        elif num_remaining == 2:
            # Only City, State.
            output_row[13] = remaining_fields[1]  # City
    return output_row


def _merge_files(input_files: List[str], output_file: str) -> None:
    """
    Merge one or more input files into a single output CSV.

    :param input_files: list of input file paths
    :param output_file: path to the output CSV file
    """
    _LOG.info("Merging %s input files", len(input_files))
    all_names = []
    all_csv_rows = []
    # Parse all input files.
    for input_file in input_files:
        names, csv_rows = _parse_input_file(input_file)
        all_names.extend(names)
        all_csv_rows.extend(csv_rows)
    # Assert that name count matches CSV row count.
    _LOG.info(
        "Total names: %s, Total CSV rows: %s", len(all_names), len(all_csv_rows)
    )
    hdbg.dassert_eq(
        len(all_names),
        len(all_csv_rows),
        "Number of names (%s) must equal number of CSV data rows (%s)",
        len(all_names),
        len(all_csv_rows),
    )
    # Merge names with CSV data.
    _LOG.info("Merging names with CSV data")
    merged_rows = []
    for name, csv_row in zip(all_names, all_csv_rows):
        normalized_row = _normalize_csv_row(name, csv_row)
        merged_rows.append(normalized_row)
    # Write output CSV.
    _LOG.info("Writing output to file='%s'", output_file)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header row.
        writer.writerow(OUTPUT_COLUMNS)
        # Write data rows.
        writer.writerows(merged_rows)
    _LOG.info("Successfully wrote %s rows to output file", len(merged_rows))


# #############################################################################
# Main execution
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    """
    Parse command line arguments.

    :return: argument parser
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Create mutually exclusive group for input specification.
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_file",
        action="append",
        dest="input_files",
        help="Input CSV file(s) to merge. Can be specified multiple times.",
    )
    input_group.add_argument(
        "--input_dir",
        help="Directory containing CSV files to merge. All .csv files will be processed.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output CSV file path",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    """
    Main execution function.

    :param parser: argument parser
    """
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Collect input files.
    if args.input_dir:
        # Find all CSV files in the directory.
        _LOG.info("Searching for CSV files in directory='%s'", args.input_dir)
        hdbg.dassert_path_exists(args.input_dir)
        hdbg.dassert_dir_exists(args.input_dir)
        csv_pattern = os.path.join(args.input_dir, "*.csv")
        input_files = sorted(glob.glob(csv_pattern))
        _LOG.info("Found %s CSV files in directory", len(input_files))
        hdbg.dassert_ne(
            input_files,
            [],
            "No CSV files found in directory: %s",
            args.input_dir,
        )
    else:
        # Use the provided input files.
        input_files = args.input_files
        hdbg.dassert_ne(
            input_files, [], "At least one input file must be specified"
        )
    # Validate that all input files exist.
    for input_file in input_files:
        hdbg.dassert_path_exists(input_file)
    # Merge files.
    _merge_files(input_files, args.output_file)
    _LOG.info("Merge completed successfully")


if __name__ == "__main__":
    _main(_parse())
