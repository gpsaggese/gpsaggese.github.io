#!/usr/bin/env python

"""
CRM utility commands for ai_outreach.

These are helper commands for managing the SQLite contact database.
The main outreach workflow is in ai_outreach/scripts/:
  - outreach_search.py   persona -> SN query -> PhantomBuster -> CSV
  - outreach_enrich.py   CSV -> emails + verify -> enriched CSV
  - outreach_send.py     mailing list + template -> send emails
  - outreach_run_all.py  single entry point for the full pipeline

USAGE:
    # Initialize a new CRM database.
    python -m ai_outreach.cli.main crm-init --db-path crm.db

    # Import contacts from CSV.
    python -m ai_outreach.cli.main crm-import \
        --db-path crm.db --csv-file contacts.csv --mode keep_existing

    # Export CRM tables to CSV.
    python -m ai_outreach.cli.main crm-export \
        --db-path crm.db --output-dir ./export/

    # Show CRM statistics.
    python -m ai_outreach.cli.main crm-stats --db-path crm.db

    # Run arbitrary SQL query.
    python -m ai_outreach.cli.main crm-query \
        --db-path crm.db --sql "SELECT COUNT(*) FROM Contact"

Import as:

import ai_outreach.cli.main as aoclmain
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from typing import Callable, Optional

import pandas as pd

import ai_outreach.workflows.crm_db as aowocrdb
import helpers.hdbg as hdbg
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)


# #############################################################################
# _HelpFormatter
# #############################################################################


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawTextHelpFormatter,
):
    """
    Argparse help formatter that:
      - preserves newlines (RawTextHelpFormatter)
      - automatically shows defaults (ArgumentDefaultsHelpFormatter)
    """


def _epilog() -> str:
    """
    Return examples shown at the bottom of `--help`.
    """
    return """Examples:
  # Initialize a new CRM database.
  python -m ai_outreach.cli.main crm-init --db-path crm.db

  # Import contacts from CSV (preview first).
  python -m ai_outreach.cli.main crm-import --db-path crm.db --csv-file contacts.csv --dry-run
  python -m ai_outreach.cli.main crm-import --db-path crm.db --csv-file contacts.csv --mode keep_existing

  # Export CRM tables.
  python -m ai_outreach.cli.main crm-export --db-path crm.db --output-dir ./export/

  # Show statistics (optionally include schema).
  python -m ai_outreach.cli.main crm-stats --db-path crm.db --verbose-stats

  # Run an ad-hoc SQL query.
  python -m ai_outreach.cli.main crm-query --db-path crm.db --sql "SELECT COUNT(*) FROM Contact"
"""


# #############################################################################
# CRM subcommands.
# #############################################################################


def _run_crm_init(args: argparse.Namespace) -> None:
    """
    Initialize a new CRM database.
    """
    aowocrdb.create_db(args.db_path)
    _LOG.info("CRM database initialized at: %s", args.db_path)


def _run_crm_import(args: argparse.Namespace) -> None:
    """
    Import contacts from CSV into CRM.
    """
    hdbg.dassert_path_exists(args.csv_file)
    hdbg.dassert_path_exists(args.db_path)
    df = pd.read_csv(args.csv_file)
    _LOG.info("Loaded %d contacts from %s", len(df), args.csv_file)
    # Handle dry run.
    if args.dry_run:
        _LOG.info("[DRY RUN] Would import %d contacts", len(df))
        _LOG.info("Preview:\n%s", df.head())
        return
    # Insert contacts.
    inserted_df = aowocrdb.insert_contact_df(args.db_path, df, mode=args.mode)
    _LOG.info("Inserted %d contacts into CRM", len(inserted_df))


def _run_crm_export(args: argparse.Namespace) -> None:
    """
    Export CRM tables to CSV files.
    """
    hdbg.dassert_path_exists(args.db_path)
    # Handle dry run.
    if args.dry_run:
        _LOG.info("[DRY RUN] Would export CRM tables to %s", args.output_dir)
        return
    # Export tables.
    aowocrdb.export_db_to_csv(args.db_path, args.output_dir)
    _LOG.info("CRM exported to %s", args.output_dir)


def _run_crm_stats(args: argparse.Namespace) -> None:
    """
    Show CRM summary statistics.
    """
    hdbg.dassert_path_exists(args.db_path)
    aowocrdb.print_table_stats(args.db_path)
    if args.verbose_stats:
        aowocrdb.print_table_schema(args.db_path, print_schema=True)


def _run_crm_query(args: argparse.Namespace) -> None:
    """
    Run arbitrary SQL query against the CRM.
    """
    hdbg.dassert_path_exists(args.db_path)
    _LOG.debug("Executing SQL: %s", args.sql)
    with sqlite3.connect(args.db_path) as conn:
        df = pd.read_sql_query(args.sql, conn)
    # Save or print results.
    if args.output_file:
        df.to_csv(args.output_file, index=False)
        _LOG.info("Query results saved to %s", args.output_file)
    else:
        _LOG.info("Query results:\n%s", df.to_string(index=False))


# #############################################################################
# Parser.
# #############################################################################


_CommandFn = Callable[[argparse.Namespace], None]


def _add_common_db_arg(parser: argparse.ArgumentParser) -> None:
    """
    Add a consistent DB path argument to a subcommand parser.
    """
    parser.add_argument("--db-path", required=True, help="Path to SQLite DB file")


def _parse() -> argparse.ArgumentParser:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "CRM utility commands for ai_outreach.\n"
            "These commands manage the SQLite contact database used by outreach workflows."
        ),
        epilog=_epilog(),
        formatter_class=_HelpFormatter,
    )
    hparser.add_verbosity_arg(parser)
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        help="Available commands",
    )
    # Make a subcommand required, so `ai-outreach` without args prints help + exits non-zero.
    subparsers.required = True
    # crm-init
    p = subparsers.add_parser(
        "crm-init",
        help="Initialize a new CRM database",
        formatter_class=_HelpFormatter,
    )
    _add_common_db_arg(p)
    p.set_defaults(func=_run_crm_init)
    # crm-import.
    p = subparsers.add_parser(
        "crm-import",
        help="Import contacts from CSV into the CRM database",
        formatter_class=_HelpFormatter,
    )
    inputs = p.add_argument_group("Inputs")
    inputs.add_argument("--db-path", required=True, help="Path to SQLite DB file")
    inputs.add_argument("--csv-file", required=True, help="Path to input CSV")
    behavior = p.add_argument_group("Behavior")
    behavior.add_argument(
        "--mode",
        default="keep_existing",
        choices=[
            "assume_no_overlap",
            "assume_idempotent",
            "keep_new",
            "keep_existing",
        ],
        help="How to handle duplicates/overlaps",
    )
    behavior.add_argument(
        "--dry-run", action="store_true", help="Preview only (no writes)"
    )
    p.set_defaults(func=_run_crm_import)
    # crm-export.
    p = subparsers.add_parser(
        "crm-export",
        help="Export CRM tables to CSV files",
        formatter_class=_HelpFormatter,
    )
    _add_common_db_arg(p)
    p.add_argument("--output-dir", required=True, help="Directory for CSV output")
    p.add_argument(
        "--dry-run", action="store_true", help="Preview only (no writes)"
    )
    p.set_defaults(func=_run_crm_export)
    # crm-stats.
    p = subparsers.add_parser(
        "crm-stats",
        help="Show CRM summary statistics",
        formatter_class=_HelpFormatter,
    )
    _add_common_db_arg(p)
    p.add_argument(
        "--verbose-stats", action="store_true", help="Also print table schemas"
    )
    p.set_defaults(func=_run_crm_stats)
    # crm-query.
    p = subparsers.add_parser(
        "crm-query",
        help="Run an SQL query against the CRM database",
        formatter_class=_HelpFormatter,
    )
    _add_common_db_arg(p)
    p.add_argument("--sql", required=True, help="SQL query to execute")
    p.add_argument(
        "--output-file", default=None, help="If set, save results to CSV here"
    )
    p.set_defaults(func=_run_crm_query)
    return parser


def _main(argv: Optional[list[str]] = None) -> None:
    """
    Entry point for the CRM CLI.

    :param argv: Optional explicit argv for testing. If None, uses
        sys.argv.
    """
    parser = _parse()
    args = parser.parse_args(argv)
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # `subparsers.required = True` guarantees we have a command and func.
    func: _CommandFn = args.func
    func(args)


if __name__ == "__main__":
    _main()
