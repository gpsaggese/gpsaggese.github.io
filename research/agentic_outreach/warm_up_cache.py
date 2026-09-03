#!/usr/bin/env python

"""
Warm up LLM cache by classifying LinkedIn connections from Google Sheet.

This script reads LinkedIn connection data from a Google Sheet, classifies
the connections by industry/type/executive level using LLM, and saves the
results back to the same sheet.

Usage:
  > warm_up_cache.py
  > warm_up_cache.py --model gpt-4
  > warm_up_cache.py --batch_size 100

Import as:

import ck_marketing.warm_up_cache as ckwaupca
"""

import argparse
import logging

import ck_marketing.workflows as ckmktwf
import helpers.hdbg as hdbg
import helpers.hllm_cli as hllmcli
import helpers.hgoogle_drive_api as hgodrapi
import helpers.hpandas as hpandas
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)

# Google Sheet URL and tab names.
_GSHEET_URL = "https://docs.google.com/spreadsheets/d/1y7A__hpV8n9nyQpNLDso6fGVsjBu5LDfgx1xYU6cwOI/edit?gid=1094411348#gid=1094411348"
_INPUT_TAB_NAME = "before"
_OUTPUT_TAB_NAME = "after"

# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        type=int,
        default=25,
        help="Number of items to process in each batch",
    )
    parser.add_argument(
        "--model",
        action="store",
        # default="gpt-5-nano",
        default="gpt-4o-mini",
        help="LLM model to use for classification (e.g., gpt-4, claude-3-opus)",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    hllmcli.shutup_llm_logging()
    # Get credentials for Google API.
    _LOG.info("Getting Google API credentials")
    credentials = hgodrapi.get_credentials()
    # Read data from Google Sheet.
    _LOG.info(
        "Reading data from Google Sheet: %s (tab=%s)",
        _GSHEET_URL,
        _INPUT_TAB_NAME,
    )
    df = hgodrapi.from_gsheet(
        _GSHEET_URL, credentials=credentials, tab_name=_INPUT_TAB_NAME
    )
    _LOG.info("Read %d rows and %d columns", len(df), len(df.columns))
    # Import here to avoid circular imports and ensure modules are loaded after logger init.
    # Classify industry, type, and executive level.
    _LOG.info(
        "Classifying industry, type, and executive level using model=%s",
        args.model,
    )
    df = ckmktwf.classify_industry_type_executive(
        df,
        batch_size=args.batch_size,
        model=args.model,
        display_stats=False,
    )
    _LOG.info("Classification completed")
    # Display statistics.
    _LOG.info("Displaying statistics")
    col_names = ["type", "industry", "executive"]
    hpandas.display_value_counts_stats_df(df, col_names, num_rows=10)
    # Save results to Google Sheet.
    _LOG.info(
        "Saving results to Google Sheet: %s (tab=%s)",
        _GSHEET_URL,
        _OUTPUT_TAB_NAME,
    )
    hgodrapi.save_df_to_tmp_gsheet(
        df,
        url=_GSHEET_URL,
        tab_name=_OUTPUT_TAB_NAME,
        remove_empty_columns=True,
    )
    _LOG.info("Results saved successfully")


if __name__ == "__main__":
    _main(_parse())
