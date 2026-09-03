#!/usr/bin/env python

"""
Example script demonstrating industry classification using LLM.

This script creates a sample dataframe from LinkedIn connection data and
classifies the company industries using an LLM.

Usage:
  > classify_industry_example.py
  > classify_industry_example.py --model gpt-4

Import as:

import ck_marketing.workflows.classify_industry_example as ckwoclae
"""

import argparse
import logging

import pandas as pd

import ck_marketing.workflows.classify_utils as cmwoclut
import helpers.hdbg as hdbg
import helpers.hllm_cli as hllmcli
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)

# #############################################################################


def _create_sample_dataframe() -> pd.DataFrame:
    """
    Create a sample dataframe with LinkedIn connection data.

    :return: dataframe with sample data
    """
    # Create sample data from instr.md.
    data = {
        "index": [1635, 1789, 1280, 606],
        "hash": [
            "e2478d5b0fcb5c19f2008dc7d507083d",
            "960addbdc2970f9beb8fe4e2c1166569",
            "665a0a6ead504830d9dec8cbc227cd52",
            "d209f6d5092f6fbd7ec3a4d76b1aab05",
        ],
        "origin": [
            "PB.LIN_Connections_Exports.GP_Lin_Connections_2024_12_31",
            "PB.LIN_Connections_Exports.GP_Lin_Connections_2024_12_31",
            "PB.LIN_Connections_Exports.GP_Lin_Connections_2024_12_31",
            "PB.LIN_Connections_Exports.GP_Lin_Connections_after_2025_12_05",
        ],
        "origin_timestamp": [
            "2024-12-31T19:10:27.069Z",
            "2024-12-31T19:12:19.286Z",
            "2024-12-31T18:36:58.020Z",
            "2025-12-09T23:40:27.199Z",
        ],
        "first_name": ["A Blake", "Aadit", "Aakarsh", "Aakash"],
        "last_name": ["Cooper", "Narayanaswamy", "Ramchandani", "Bansal"],
        "email": [None, None, None, None],
        "email_timestamp": [None, None, None, "2025-12-09T23:40:27.199Z"],
        "email_verification": [None, None, None, None],
        "email_verification_timestamp": [None, None, None, None],
        "linkedin_url": [
            "https://linkedin.com/in/a-blake-cooper-7624897",
            "https://linkedin.com/in/aaditn",
            "https://linkedin.com/in/aakarsh-ramchandani",
            "https://www.linkedin.com/in/aakashban/",
        ],
        "job_title": [
            "Infrastructure Engineer",
            "Researcher",
            "Chief Strategy Officer",
            "Capital Markets Intern @Figure | UIUC | Ex-JPMC",
        ],
        "company_name": ["Braintree", "Stealth", "RavenPack", None],
        "company_domain": [None, None, None, None],
        "city": [
            "Chesterton, Indiana, United States",
            "San Francisco Bay Area",
            "New York, New York, United States",
            None,
        ],
        "country": [None, None, None, None],
        "enrichment_timestamp": [None, None, None, None],
        "type": [None, None, None, None],
        "biography": [
            "UNIX / Networking at Somewhere new; ;",
            "Researcher; ; Quantitative finance, biotech, fintech, and networking.",
            '"Chief Strategy Officer at RavenPack | Bigdata.com; Builder of products, services, orgs.\n\nMy portfolio (built with exceptional teams): \n\n FactSet RBICS - GICS was too expensive. We built our own.\n\n FactSet Screening - Institutional Grade screening engine with over 1600 data sources. \n\n FactSet Formula IDE - Innovative IDE for searching through millions of formulas. \n\nThirdPoint Data & Analytics Engine - Integrated, analyzed, and delivered trade recommendations on over 100 datasets. Ended up building basket construction infrastructure to execute on our ideas via swaps.\n\nBigdata.com - Search and Discovery engine for Finance. Journey just started...;"',
            None,
        ],
        "industry": [None, None, None, None],
    }
    df = pd.DataFrame(data)
    _LOG.info(
        "Created dataframe with %d rows and %d columns", len(df), len(df.columns)
    )
    return df


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        action="store",
        default=None,
        help="LLM model to use (e.g., gpt-4, claude-3-opus)",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Create sample dataframe.
    _LOG.info("Creating sample dataframe")
    df = _create_sample_dataframe()
    _LOG.info("Sample dataframe:\n%s", df)
    # Get the prompt and extractor.
    prompt = cmwoclut.get_person_industry_prompt()
    _LOG.debug("Using prompt:\n%s", prompt)
    # Apply LLM to classify industries.
    _LOG.info("Classifying industries using LLM")
    result_df = hllmcli.apply_llm_prompt_to_df(
        prompt=prompt,
        df=df,
        extractor=cmwoclut.extract_person_industry_from_df,
        target_col="industry",
        batch_size=20,
        model=args.model,
    )
    # Print results.
    _LOG.info("Classification completed")
    _LOG.info(
        "Results:\n%s",
        result_df[["first_name", "last_name", "company_name", "industry"]],
    )
    # Print detailed results for each row.
    _LOG.info("\nDetailed classifications:")
    for idx, row in result_df.iterrows():
        _LOG.info(
            "Row %d: %s %s (%s) -> %s",
            idx,
            row["first_name"],
            row["last_name"],
            row["company_name"],
            row["industry"],
        )


if __name__ == "__main__":
    _main(_parse())
