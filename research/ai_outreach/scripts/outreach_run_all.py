#!/usr/bin/env python3

r"""
Run the full outreach pipeline end-to-end.

This is the single entry point that chains all stages:
  1. Search:  persona -> SN query -> PhantomBuster -> raw contacts CSV
  2. Enrich:  raw contacts -> find emails + verify -> enriched CSV
  3. Draft:   enriched CSV -> LLM-generated email drafts -> drafts CSV
  4. Upload:  upload final results to Google Sheets

Each stage writes its output to a file, so the pipeline can be resumed
from any point by providing the intermediate file via --skip_to.

Usage
-----
    > outreach_run_all \
        --persona "Series A fintech founders in NYC" \
        --sn_query_url "https://www.linkedin.com/sales/search/people?..." \
        --linkedin_cookie "$LI_AT" \
        --campaign_goal "Introduce our AI platform" \
        --out_dir ./outreach_run/

Examples
--------
    # Full pipeline with all stages.
    > outreach_run_all \
        --persona "Partners at crypto VC funds" \
        --sn_query_url "https://www.linkedin.com/sales/search/people?..." \
        --linkedin_cookie "$LI_AT" \
        --campaign_goal "Introduce our causal AI platform to VCs" \
        --sender_name "GP" \
        --sender_title "CEO, Causify" \
        --out_dir ./campaign_2024q1/

    # Resume from enrichment (already have contacts CSV).
    > outreach_run_all \
        --skip_to enrich \
        --in_path ./campaign_2024q1/1_contacts.csv \
        --campaign_goal "Introduce our AI platform" \
        --out_dir ./campaign_2024q1/

    # Resume from drafting (already have enriched CSV).
    > outreach_run_all \
        --skip_to draft \
        --in_path ./campaign_2024q1/2_enriched.csv \
        --campaign_goal "Follow up after demo" \
        --out_dir ./campaign_2024q1/

    # Search + enrich only (skip drafting).
    > outreach_run_all \
        --persona "VP Engineering at B2B SaaS" \
        --sn_query_url "https://..." \
        --linkedin_cookie "$LI_AT" \
        --no_draft \
        --out_dir ./leads_q1/

Environment
-----------
    HUNTER_API_KEY       Required for email enrichment (Hunter.io)
    Phantom_API_KEY      Required for PhantomBuster scraping

Import as:

import ai_outreach.scripts.outreach_run_all as aoscoura
"""

import argparse
import json
import logging
import os

import ai_outreach.scripts.outreach_draft as aoscoudr
import ai_outreach.scripts.outreach_enrich as aoscouen
import ai_outreach.scripts.outreach_search as aoscouse
import ai_outreach.scripts.utils as aouscuti
import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)

_STAGES = ["search", "enrich", "draft"]


# #############################################################################
# Pipeline stages.
# #############################################################################


def _run_search(args: argparse.Namespace, out_dir: str) -> str:
    """
    Stage 1: Search for contacts.

    :param args: parsed CLI arguments
    :param out_dir: output directory
    :return: path to contacts CSV
    """
    out_path = os.path.join(out_dir, "1_contacts.csv")
    _LOG.info("=" * 60)
    _LOG.info("STAGE 1: SEARCH")
    _LOG.info("=" * 60)
    searcher = aoscouse.ContactSearcher(model=args.model)
    output = searcher.run(
        persona=args.persona,
        sn_query_url=args.sn_query_url,
        linkedin_cookie=args.linkedin_cookie,
    )
    if "contacts_df" in output:
        # Save scraped contacts.
        df = output["contacts_df"]
        aouscuti.save_contacts_csv(df, out_path)
        _LOG.info("Saved %d contacts to %s", len(df), out_path)
    else:
        # Query-only output (no scraping happened).
        query_path = os.path.join(out_dir, "1_sn_query.json")
        with open(query_path, "w", encoding="utf-8") as f:
            json.dump(output["query_result"], f, indent=2)
        _LOG.info("Query saved to %s", query_path)
        raise SystemExit(
            "Pipeline stopped: provide --sn_query_url to continue."
        )
    return out_path


def _run_enrich(
    args: argparse.Namespace, in_path: str, out_dir: str
) -> str:
    """
    Stage 2: Enrich contacts with emails.

    :param args: parsed CLI arguments
    :param in_path: path to input contacts CSV
    :param out_dir: output directory
    :return: path to enriched CSV
    """
    out_path = os.path.join(out_dir, "2_enriched.csv")
    _LOG.info("=" * 60)
    _LOG.info("STAGE 2: ENRICH")
    _LOG.info("=" * 60)
    df = aouscuti.load_contacts_csv(in_path)
    enricher = aoscouen.ContactEnricher()
    df = enricher.run(df, verify=not args.no_verify)
    aouscuti.save_contacts_csv(df, out_path)
    _LOG.info("Saved %d enriched contacts to %s", len(df), out_path)
    return out_path


def _run_draft(
    args: argparse.Namespace, in_path: str, out_dir: str
) -> str:
    """
    Stage 3: Generate email drafts via LLM.

    :param args: parsed CLI arguments
    :param in_path: path to input enriched contacts CSV
    :param out_dir: output directory
    :return: path to drafts CSV
    """
    out_path = os.path.join(out_dir, "3_drafts.csv")
    _LOG.info("=" * 60)
    _LOG.info("STAGE 3: DRAFT")
    _LOG.info("=" * 60)
    df = aouscuti.load_contacts_csv(in_path)
    drafter = aoscoudr.EmailDrafter(model=args.draft_model)
    df = drafter.run(
        df,
        campaign_goal=args.campaign_goal,
        sender_name=args.sender_name,
        sender_title=args.sender_title,
        batch_size=args.batch_size,
    )
    aouscuti.save_contacts_csv(df, out_path)
    _LOG.info("Saved %d drafts to %s", len(df), out_path)
    return out_path


def _run_upload(final_path: str, sheet_name: str) -> None:
    """
    Upload final results to Google Sheets.

    :param final_path: path to final contacts CSV
    :param sheet_name: name of the Google Sheet
    """
    _LOG.info("=" * 60)
    _LOG.info("UPLOAD TO GOOGLE SHEETS")
    _LOG.info("=" * 60)
    df = aouscuti.load_contacts_csv(final_path)
    aouscuti.update_google_sheet(df, sheet_name=sheet_name)


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
    # Search options.
    search = parser.add_argument_group("search options")
    search.add_argument(
        "--persona",
        default=None,
        help="Persona description for SN query generation",
    )
    search.add_argument(
        "--sn_query_url",
        default=None,
        help="Pre-built Sales Navigator search URL",
    )
    search.add_argument(
        "--linkedin_cookie",
        default=None,
        help="LinkedIn li_at session cookie",
    )
    search.add_argument(
        "--model",
        default="gpt-5.2",
        help="LLM model for query generation (default: gpt-5.2)",
    )
    # Enrich options.
    enrich = parser.add_argument_group("enrich options")
    enrich.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip email verification",
    )
    # Draft options.
    draft = parser.add_argument_group("draft options")
    draft.add_argument(
        "--campaign_goal",
        default=None,
        help="Campaign objective for email draft generation",
    )
    draft.add_argument(
        "--sender_name",
        default="",
        help="Name of the email sender",
    )
    draft.add_argument(
        "--sender_title",
        default="",
        help="Title/role of the email sender",
    )
    draft.add_argument(
        "--draft_model",
        default="gpt-5-nano",
        help="LLM model for draft generation (default: gpt-5-nano)",
    )
    draft.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Contacts per LLM batch (default: 20)",
    )
    draft.add_argument(
        "--no_draft",
        action="store_true",
        help="Skip the draft stage (search + enrich only)",
    )
    # Resume options.
    resume = parser.add_argument_group("resume options")
    resume.add_argument(
        "--skip_to",
        default=None,
        choices=_STAGES,
        help="Skip to a specific stage (provide --in_path for input)",
    )
    resume.add_argument(
        "--in_path",
        default=None,
        help="Input CSV when resuming with --skip_to",
    )
    # Upload options.
    upload = parser.add_argument_group("upload options")
    upload.add_argument(
        "--no_upload",
        action="store_true",
        help="Skip Google Sheets upload",
    )
    upload.add_argument(
        "--sheet_name",
        default="outreach_results",
        help="Google Sheet name (default: outreach_results)",
    )
    # Output.
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for all intermediate and final files",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    """
    Run the full outreach pipeline from CLI arguments.
    """
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Create output directory.
    hio.create_dir(args.out_dir, incremental=True)
    # Determine starting stage.
    start_stage = args.skip_to or "search"
    start_idx = _STAGES.index(start_stage)
    current_path = args.in_path
    _LOG.info("Output directory: %s", args.out_dir)
    _LOG.info("Starting from stage: %s", start_stage)
    # Stage 1: Search.
    if start_idx <= 0:
        current_path = _run_search(args, args.out_dir)
    else:
        hdbg.dassert_is_not(
            current_path,
            None,
            "Must provide --in_path when using --skip_to",
        )
    final_path = current_path
    # Stage 2: Enrich.
    if start_idx <= 1:
        final_path = _run_enrich(args, current_path, args.out_dir)
        current_path = final_path
    # Stage 3: Draft.
    if not args.no_draft and start_idx <= 2:
        if args.campaign_goal is None:
            _LOG.info(
                "Skipping draft stage: --campaign_goal required."
            )
        else:
            final_path = _run_draft(args, current_path, args.out_dir)
    # Upload to Google Sheets.
    if not args.no_upload:
        try:
            _run_upload(final_path, args.sheet_name)
        except Exception as e:  # pylint: disable=broad-exception-caught
            _LOG.warning("Google Sheets upload failed: %s", e)
    # Done.
    _LOG.info("=" * 60)
    _LOG.info("PIPELINE COMPLETE")
    _LOG.info("=" * 60)
    _LOG.info("Output directory: %s", args.out_dir)


if __name__ == "__main__":
    _main(_parse())
