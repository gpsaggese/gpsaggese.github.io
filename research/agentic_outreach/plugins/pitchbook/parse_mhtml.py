#!/usr/bin/env python

"""
Parse MHTML files and extract HTML content, tables, or div-based grids.

This script can:
- Extract HTML from MHTML files.
- Print the DOM structure of the HTML.
- Parse and extract traditional HTML tables.
- Parse and extract div-based data grids (modern web apps).
- Process multiple MHTML files from a directory.

Import as:

import dev_scripts_helpers.scraping_script.parse_mhtml as dsspamht

Examples:
# Print DOM structure.
> python parse_mhtml.py --input_file file.mhtml

# Extract all HTML tables to CSV.
> python parse_mhtml.py --input_file file.mhtml --mode tables --output_dir ./output

# Extract div-based grids to CSV.
> python parse_mhtml.py --input_file file.mhtml --mode grids --output_dir ./output

# Extract both tables and grids.
> python parse_mhtml.py --input_file file.mhtml --mode all --output_dir ./output

# Extract a specific table/grid by index to a directory.
> python parse_mhtml.py --input_file file.mhtml --mode grids --table_index 0 --output_dir ./output

# Extract single table/grid to a file (automatically uses first if only one exists).
> python parse_mhtml.py --input_file file.mhtml --mode tables --output_file ./output.csv

# Process all MHTML files in a directory and save as out001.csv, out002.csv, etc.
> python parse_mhtml.py --input_dir ./mhtml_files --mode tables --output_dir ./output
"""

import argparse
import csv
import glob
import logging
import os
from email import policy
from email.parser import BytesParser
from typing import List, Optional

from bs4 import BeautifulSoup

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)

# #############################################################################
# HTML extraction
# #############################################################################


def _extract_html_from_mhtml(path: str) -> Optional[str]:
    """
    Extract HTML string from first text/html part in MHTML file.

    :param path: path to the MHTML file
    :return: HTML content as string or None if not found
    """
    _LOG.debug("Extracting HTML from MHTML file='%s'", path)
    hdbg.dassert_path_exists(path)
    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    # Check if the root itself is text/html.
    if msg.get_content_type() == "text/html":
        _LOG.debug("Found HTML in root message")
        return msg.get_content()
    # Walk all parts to find the first text/html.
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                _LOG.debug("Found HTML in multipart message")
                return part.get_content()
    _LOG.warning("No text/html part found in MHTML file")
    return None


# #############################################################################
# DOM printing
# #############################################################################


def _print_dom(node, *, depth: int = 0, max_depth: int = 10) -> None:
    """
    Recursively print DOM structure showing tag names and text snippets.

    :param node: BeautifulSoup node to print
    :param depth: current depth in the tree
    :param max_depth: maximum depth to traverse
    """
    from bs4.element import Tag

    indent = "  " * depth
    # Stop deep recursion if page is huge.
    if depth > max_depth:
        _LOG.debug("Max depth reached at depth=%s", depth)
        print(indent + "... (max depth reached)")
        return
    # Only process elements (tags), skip NavigableString etc.
    if isinstance(node, Tag):
        # Get small snippet of text inside this tag.
        text = node.get_text(strip=True)
        if len(text) > 60:
            text = text[:57] + "..."
        print(f"{indent}<{node.name}>  {repr(text)}")
        # Recurse into children.
        for child in node.children:
            _print_dom(child, depth=depth + 1, max_depth=max_depth)


def _validate_table_structure(
    table_data: List[List[str]], table_index: int = 0
) -> None:
    """
    Validate that all rows in a table have the same number of columns.

    :param table_data: table as list of rows, where each row is a list of cell
        values
    :param table_index: index of table for logging purposes
    :raises ValueError: if rows have inconsistent column counts
    """
    if not table_data:
        _LOG.warning("Table %s is empty", table_index)
        return
    # Get the number of columns from the first row.
    expected_cols = len(table_data[0])
    _LOG.debug(
        "Validating table %s structure: expected %s columns",
        table_index,
        expected_cols,
    )
    # Check each row has the same number of columns.
    inconsistent_rows = []
    for row_idx, row in enumerate(table_data):
        actual_cols = len(row)
        if actual_cols != expected_cols:
            inconsistent_rows.append((row_idx, actual_cols))
            _LOG.warning(
                "Table %s, row %s: expected %s columns but found %s",
                table_index,
                row_idx,
                expected_cols,
                actual_cols,
            )
    if inconsistent_rows:
        error_msg = (
            f"Table {table_index} has inconsistent column counts. "
            f"Expected {expected_cols} columns in all rows, but found:\n"
        )
        for row_idx, actual_cols in inconsistent_rows:
            error_msg += f"  Row {row_idx}: {actual_cols} columns\n"
        raise ValueError(error_msg.strip())
    _LOG.debug("Table %s structure is valid", table_index)


# #############################################################################
# Table parsing
# #############################################################################


def _extract_tables_from_html(html: str) -> List[List[List[str]]]:
    """
    Extract all tables from HTML content.

    :param html: HTML content as string
    :return: list of tables, where each table is a list of rows, and each row is
        a list of cell values
    """
    _LOG.debug("Parsing HTML to extract tables")
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    _LOG.info("Found %s HTML tables", len(tables))
    tables_data = []
    for table_idx, table in enumerate(tables):
        _LOG.debug("Processing HTML table %s", table_idx)
        table_data = []
        # Process all rows in the table.
        rows = table.find_all("tr")
        for row_idx, row in enumerate(rows):
            # Extract cells from row (both th and td).
            cells = row.find_all(["th", "td"])
            row_data = []
            for cell in cells:
                # Get text content, stripping whitespace.
                # Keep empty cells to preserve table structure.
                cell_text = cell.get_text(strip=True)
                row_data.append(cell_text)
            # Always append row_data to preserve table structure, even if all cells are empty.
            table_data.append(row_data)
        if table_data:
            # Validate table structure before adding.
            try:
                _validate_table_structure(table_data, table_index=table_idx)
                tables_data.append(table_data)
            except ValueError as e:
                _LOG.error(
                    "Skipping table %s due to validation error: %s", table_idx, e
                )
                # Still append the table but log the issue.
                tables_data.append(table_data)
    return tables_data


def _extract_data_table_pattern(row_divs: List) -> List[List[str]]:
    """
    Extract data from data-table__row pattern.

    :param row_divs: list of row div elements
    :return: grid data as list of rows
    """
    import re

    grid_data = []
    for row_idx, row in enumerate(row_divs):
        _LOG.debug("Processing data-table row %s", row_idx)
        # Look for cells within the row.
        cells = row.find_all("div", class_=re.compile(r"data-table__cell"))
        row_data = []
        for cell in cells:
            # Try to find entity text in custom format.
            entity = cell.find("div", class_="custom-cell-format__fixed-entity")
            if entity:
                cell_text = entity.get_text(strip=True)
            else:
                # Fall back to cell text.
                cell_text = cell.get_text(strip=True)
            # Keep empty cells to preserve table structure.
            row_data.append(cell_text)
        # Always append row_data to preserve table structure, even if all cells are empty.
        grid_data.append(row_data)
    return grid_data


def _extract_aria_grid_pattern(role_rows: List) -> List[List[str]]:
    """
    Extract data from ARIA grid pattern with role attributes.

    :param role_rows: list of elements with role=row
    :return: grid data as list of rows
    """
    grid_data = []
    for row_idx, row in enumerate(role_rows):
        _LOG.debug("Processing ARIA row %s", row_idx)
        # Look for cells with role="gridcell" or role="cell".
        cells = row.find_all(
            attrs={"role": ["gridcell", "cell", "columnheader"]}
        )
        row_data = []
        for cell in cells:
            # Keep empty cells to preserve table structure.
            cell_text = cell.get_text(strip=True)
            row_data.append(cell_text)
        # Always append row_data to preserve table structure, even if all cells are empty.
        grid_data.append(row_data)
    return grid_data


def _extract_generic_grid_pattern(generic_rows: List) -> List[List[str]]:
    """
    Extract data from generic grid/row pattern.

    :param generic_rows: list of row-like div elements
    :return: grid data as list of rows
    """
    import re

    grid_data = []
    for row_idx, row in enumerate(generic_rows):
        _LOG.debug("Processing generic row %s", row_idx)
        # Look for cell-like divs.
        cells = row.find_all(
            "div",
            class_=re.compile(r"(cell|col)", re.IGNORECASE),
            recursive=False,
        )
        if not cells:
            # If no cells found, use direct children.
            cells = [c for c in row.children if hasattr(c, "get_text")]
        row_data = []
        for cell in cells:
            if hasattr(cell, "get_text"):
                # Keep empty cells to preserve table structure.
                cell_text = cell.get_text(strip=True)
                row_data.append(cell_text)
        # Always append row_data to preserve table structure, even if all cells are empty.
        grid_data.append(row_data)
    return grid_data


def _extract_div_grids_from_html(html: str) -> List[List[List[str]]]:
    """
    Extract div-based data grids from HTML content.

    Looks for common patterns in div-based grids such as data tables,
    virtual grids, etc.

    :param html: HTML content as string
    :return: list of grids, where each grid is a list of rows, and each row is
        a list of cell values
    """
    import re

    _LOG.debug("Parsing HTML to extract div-based grids")
    soup = BeautifulSoup(html, "html.parser")
    grids_data = []
    # Pattern 1: Look for data-table__row pattern (common in many frameworks).
    row_divs = soup.find_all("div", class_=re.compile(r"data-table__row"))
    if row_divs:
        _LOG.info("Found %s rows in data-table pattern", len(row_divs))
        grid_data = _extract_data_table_pattern(row_divs)
        if grid_data:
            # Validate grid structure before adding.
            try:
                _validate_table_structure(grid_data, table_index=len(grids_data))
                grids_data.append(grid_data)
            except ValueError as e:
                _LOG.error("Grid validation error: %s", e)
                # Still append the grid but log the issue.
                grids_data.append(grid_data)
    # Pattern 2: Look for role="row" ARIA pattern.
    role_rows = soup.find_all(attrs={"role": "row"})
    if role_rows:
        _LOG.info("Found %s rows with role=row attribute", len(role_rows))
        grid_data = _extract_aria_grid_pattern(role_rows)
        if grid_data:
            # Validate grid structure before adding.
            try:
                _validate_table_structure(grid_data, table_index=len(grids_data))
                grids_data.append(grid_data)
            except ValueError as e:
                _LOG.error("Grid validation error: %s", e)
                # Still append the grid but log the issue.
                grids_data.append(grid_data)
    # Pattern 3: Look for generic grid-row or table-row classes.
    if not grids_data:
        generic_rows = soup.find_all(
            "div", class_=re.compile(r"(grid-row|table-row|row)", re.IGNORECASE)
        )
        if generic_rows:
            _LOG.info("Found %s rows in generic pattern", len(generic_rows))
            grid_data = _extract_generic_grid_pattern(generic_rows)
            if grid_data:
                # Validate grid structure before adding.
                try:
                    _validate_table_structure(
                        grid_data, table_index=len(grids_data)
                    )
                    grids_data.append(grid_data)
                except ValueError as e:
                    _LOG.error("Grid validation error: %s", e)
                    # Still append the grid but log the issue.
                    grids_data.append(grid_data)
    _LOG.info("Found %s div-based grids in HTML", len(grids_data))
    return grids_data


def _save_table_to_csv(table_data: List[List[str]], output_path: str) -> None:
    """
    Save a table to a CSV file.

    :param table_data: table as list of rows, where each row is a list of cell
        values
    :param output_path: path to save the CSV file
    """
    _LOG.info("Saving table to file='%s'", output_path)
    hdbg.dassert_ne(table_data, [], "Table data cannot be empty")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(table_data)
    _LOG.info("Table saved successfully")


def _print_table_summary(tables_data: List[List[List[str]]]) -> None:
    """
    Print a summary of extracted tables.

    :param tables_data: list of tables
    """
    _LOG.info("Printing table summary")
    print(f"\nFound {len(tables_data)} tables:")
    for idx, table_data in enumerate(tables_data):
        num_rows = len(table_data)
        num_cols = len(table_data[0]) if table_data else 0
        print(f"  Table {idx}: {num_rows} rows x {num_cols} columns")
        # Print first few cells as preview.
        if table_data:
            print(f"    Preview: {table_data[0][:3]}")


# #############################################################################
# Main execution
# #############################################################################


def _process_single_mhtml_file(
    mhtml_file: str,
    *,
    mode: str,
    max_depth: int = 10,
) -> List[List[List[str]]]:
    """
    Process a single MHTML file and extract tables/grids.

    :param mhtml_file: path to MHTML file
    :param mode: parsing mode (dom, tables, grids, all)
    :param max_depth: maximum depth for DOM traversal (for dom mode)
    :return: list of extracted tables/grids, or empty list for dom mode
    """
    _LOG.info("Processing MHTML file='%s'", mhtml_file)
    hdbg.dassert_path_exists(mhtml_file)
    # Extract HTML from MHTML file.
    html = _extract_html_from_mhtml(mhtml_file)
    hdbg.dassert_is_not(
        html, None, "No text/html part found in MHTML file: %s", mhtml_file
    )
    # Process based on mode.
    if mode == "dom":
        # Print DOM structure.
        _LOG.info("Printing DOM structure for file='%s'", mhtml_file)
        soup = BeautifulSoup(html, "html.parser")
        root = soup.html or soup
        _print_dom(root, max_depth=max_depth)
        return []
    # Extract tables and/or grids.
    all_data = []
    if mode in ["tables", "all"]:
        _LOG.info("Extracting HTML tables from file='%s'", mhtml_file)
        tables_data = _extract_tables_from_html(html)
        all_data.extend(tables_data)
    if mode in ["grids", "all"]:
        _LOG.info("Extracting div-based grids from file='%s'", mhtml_file)
        grids_data = _extract_div_grids_from_html(html)
        all_data.extend(grids_data)
    # Check if any data was found.
    if not all_data:
        _LOG.warning(
            "No tables or grids found in HTML from file: %s", mhtml_file
        )
    return all_data


def _parse() -> argparse.ArgumentParser:
    """
    Parse command line arguments.

    :return: argument parser
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Input mode: either single file or directory.
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_file",
        help="Path to single MHTML file to parse",
    )
    input_group.add_argument(
        "--input_dir",
        help="Directory containing MHTML files to process (non-recursive)",
    )
    parser.add_argument(
        "--mode",
        choices=["dom", "tables", "grids", "all"],
        default="dom",
        help="Parsing mode: dom (print DOM structure), tables (extract HTML tables), grids (extract div-based grids), or all (extract both tables and grids)",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save extracted tables (only for tables mode)",
    )
    parser.add_argument(
        "--output_file",
        help="File path to save extracted table (only for tables mode). Mutually exclusive with --output_dir. If multiple tables exist, use --table_index to specify which one. Only valid with --input_file.",
    )
    parser.add_argument(
        "--table_index",
        type=int,
        help="Extract only the table at this index. With --input_file, extracts the specified table from that file. With --input_dir, extracts the specified table from each file in the directory.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Maximum depth for DOM traversal (only for dom mode)",
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
    # Validate output options.
    if args.output_dir and args.output_file:
        raise ValueError(
            "Cannot specify both --output_dir and --output_file. Use one or the other."
        )
    # Validate that --output_file is only used with --input_file.
    if args.input_dir:
        if args.output_file:
            raise ValueError(
                "--output_file can only be used with --input_file, not --input_dir"
            )
        if args.mode == "dom":
            raise ValueError(
                "--input_dir mode requires --mode to be tables, grids, or all (not dom)"
            )
        if not args.output_dir:
            raise ValueError(
                "--input_dir mode requires --output_dir to be specified"
            )
    # Handle single file mode.
    if args.input_file:
        all_data = _process_single_mhtml_file(
            args.input_file,
            mode=args.mode,
            max_depth=args.max_depth,
        )
        # For dom mode, we're done (DOM was printed in the function).
        if args.mode == "dom":
            return
        # Check if any data was found.
        hdbg.dassert_ne(
            all_data,
            [],
            "No tables or grids found in HTML from file: %s",
            args.input_file,
        )
        # Print summary.
        _print_table_summary(all_data)
        # Save data if output directory specified.
        if args.output_dir:
            hio.create_dir(args.output_dir, incremental=True)
            # Determine which tables/grids to save.
            if args.table_index is not None:
                hdbg.dassert_lte(
                    args.table_index,
                    len(all_data) - 1,
                    "Table/grid index out of range: %s",
                    args.table_index,
                )
                data_to_save = [(args.table_index, all_data[args.table_index])]
            else:
                data_to_save = list(enumerate(all_data))
            # Save each table/grid to CSV.
            for idx, table_data in data_to_save:
                output_path = f"{args.output_dir}/table_{idx}.csv"
                _save_table_to_csv(table_data, output_path)
        elif args.output_file:
            # Save data if output file specified.
            hdbg.dassert_is_not(
                args.table_index,
                None,
                "--output_file requires --table_index to be specified",
            )
            hdbg.dassert_lte(
                args.table_index,
                len(all_data) - 1,
                "Table/grid index out of range: %s",
                args.table_index,
            )
            table_data = all_data[args.table_index]
            _save_table_to_csv(table_data, args.output_file)
    # Handle directory mode.
    elif args.input_dir:
        hdbg.dassert_path_exists(args.input_dir)
        # Find all MHTML files in the directory (non-recursive).
        pattern = os.path.join(args.input_dir, "*.mhtml")
        mhtml_files = sorted(glob.glob(pattern))
        hdbg.dassert_ne(
            mhtml_files,
            [],
            "No MHTML files found in directory: %s",
            args.input_dir,
        )
        _LOG.info(
            "Found %s MHTML files in directory='%s'",
            len(mhtml_files),
            args.input_dir,
        )
        # Create output directory.
        hio.create_dir(args.output_dir, incremental=True)
        # Process all files and collect all tables.
        all_tables_from_all_files = []
        for mhtml_file in mhtml_files:
            tables_data = _process_single_mhtml_file(
                mhtml_file,
                mode=args.mode,
                max_depth=args.max_depth,
            )
            if tables_data:
                # If table_index specified, only extract that specific table.
                if args.table_index is not None:
                    if args.table_index < len(tables_data):
                        tables_to_add = [tables_data[args.table_index]]
                        _LOG.info(
                            "Extracted table/grid %s from file='%s'",
                            args.table_index,
                            os.path.basename(mhtml_file),
                        )
                    else:
                        _LOG.warning(
                            "Table/grid index %s not found in file='%s' (found %s tables/grids)",
                            args.table_index,
                            os.path.basename(mhtml_file),
                            len(tables_data),
                        )
                        tables_to_add = []
                else:
                    tables_to_add = tables_data
                    _LOG.info(
                        "Extracted %s tables/grids from file='%s'",
                        len(tables_data),
                        os.path.basename(mhtml_file),
                    )
                all_tables_from_all_files.extend(tables_to_add)
            else:
                _LOG.warning(
                    "No tables/grids found in file='%s'",
                    os.path.basename(mhtml_file),
                )
        # Check if any data was found across all files.
        hdbg.dassert_ne(
            all_tables_from_all_files,
            [],
            "No tables or grids found in any MHTML files in directory: %s",
            args.input_dir,
        )
        # Print summary.
        _LOG.info(
            "Total tables/grids extracted: %s", len(all_tables_from_all_files)
        )
        _print_table_summary(all_tables_from_all_files)
        # Save all tables with sequential numbering (out001.csv, out002.csv, etc.).
        for idx, table_data in enumerate(all_tables_from_all_files):
            # Use 3-digit zero-padded numbering: out001.csv, out002.csv, etc.
            output_filename = f"out{idx + 1:03d}.csv"
            output_path = os.path.join(args.output_dir, output_filename)
            _save_table_to_csv(table_data, output_path)


if __name__ == "__main__":
    _main(_parse())
