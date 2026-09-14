#!/usr/bin/env python3

r"""
Enrich contacts with email addresses and verify deliverability.

This tool runs in two stages:
  1. Find: Look up email addresses for contacts using Hunter.io
     (requires company_domain).
  2. Verify: Check deliverability of found emails via Hunter.io
     (valid, invalid, accept_all, unknown).

Both stages are incremental: contacts that already have an email or
verification status are skipped.

Usage
-----
    > outreach_enrich --in_path contacts.csv --out_path enriched.csv

Examples
--------
    # Enrich and verify.
    > outreach_enrich \
        --in_path contacts.csv \
        --out_path enriched.csv

    # Enrich without verification.
    > outreach_enrich \
        --in_path contacts.csv \
        --no_verify \
        --out_path enriched.csv

    # Only verify emails (skip finding new ones).
    > outreach_enrich \
        --in_path contacts.csv \
        --verify_only \
        --out_path verified.csv

Environment
-----------
    HUNTER_API_KEY      Required for Hunter.io

Import as:

import ai_outreach.scripts.outreach_enrich as aoscouen
"""

import argparse
import logging

import pandas as pd

import ai_outreach.plugins.hunterio.hunterio_api as aophhuap
import ai_outreach.scripts.utils as aouscuti
import helpers.hdbg as hdbg
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)


# #############################################################################
# ContactEnricher
# #############################################################################


class ContactEnricher:
    """
    Enrich contacts with emails and verify deliverability via Hunter.io.
    """

    @staticmethod
    def verify_emails(df: pd.DataFrame) -> pd.DataFrame:
        """
        Verify deliverability of email addresses via Hunter.io.

        :param df: contacts DataFrame with email column
        :return: DataFrame with email_verification column updated
        """
        has_email = (df["email"] != "") & (df["email"] != "nan")
        needs_verification = has_email
        if "email_verification" in df.columns:
            already_done = df["email_verification"].isin(
                ["valid", "invalid", "accept_all"]
            )
            needs_verification = has_email & ~already_done
        _LOG.info(
            "Verifying emails: %d need verification (%d have email)",
            needs_verification.sum(),
            has_email.sum(),
        )
        if needs_verification.sum() == 0:
            _LOG.info("No emails need verification.")
            return df
        df = aophhuap.hunterio_verify_emails_from_df(df)
        # Report verification results.
        if "email_verification" in df.columns:
            counts = df.loc[has_email, "email_verification"].value_counts()
            _LOG.info("Verification results:")
            for status, count in counts.items():
                if status and status != "" and status != "nan":
                    _LOG.info("  %s: %d", status, count)
        return df

    @staticmethod
    def find_emails(df: pd.DataFrame) -> pd.DataFrame:
        """
        Find email addresses for contacts missing them.

        :param df: contacts DataFrame with first_name, last_name,
            company_domain columns
        :return: DataFrame with email columns added/updated
        """
        missing_email = (df["email"] == "") | (df["email"] == "nan")
        has_domain = (df["company_domain"] != "") & (
            df["company_domain"] != "nan"
        )
        enrichable = missing_email & has_domain
        _LOG.info(
            "Finding emails: %d contacts enrichable "
            "(%d missing email, %d have domain)",
            enrichable.sum(),
            missing_email.sum(),
            has_domain.sum(),
        )
        if enrichable.sum() == 0:
            _LOG.info("No contacts need email enrichment.")
            return df
        # Enrich using Hunter.io.
        df = aophhuap.hunterio_find_emails_from_df(df)
        # Report results.
        still_missing = (df["email"] == "") | (df["email"] == "nan")
        found = missing_email.sum() - still_missing.sum()
        _LOG.info(
            "Found %d new emails (%d still missing)",
            found,
            still_missing.sum(),
        )
        return df

    def run(
        self,
        df: pd.DataFrame,
        *,
        verify: bool = True,
        verify_only: bool = False,
    ) -> pd.DataFrame:
        """
        Run the enrichment pipeline.

        :param df: input contacts DataFrame
        :param verify: whether to verify emails after finding them
        :param verify_only: skip finding, only verify existing emails
        :return: enriched DataFrame
        """
        if not verify_only:
            df = self.find_emails(df)
        if verify or verify_only:
            df = self.verify_emails(df)
        return df


# #############################################################################
# CLI.
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--in_path",
        required=True,
        help="Path to input contacts CSV",
    )
    parser.add_argument(
        "--out_path",
        required=True,
        help="Path to output enriched contacts CSV",
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip email verification after finding",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify existing emails (skip finding new ones)",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    """
    Run the enrichment pipeline from CLI arguments.
    """
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    df = aouscuti.load_contacts_csv(args.in_path)
    enricher = ContactEnricher()
    df = enricher.run(
        df,
        verify=not args.no_verify,
        verify_only=args.verify_only,
    )
    aouscuti.save_contacts_csv(df, args.out_path)
    _LOG.info("Saved %d enriched contacts to %s", len(df), args.out_path)


if __name__ == "__main__":
    _main(_parse())
