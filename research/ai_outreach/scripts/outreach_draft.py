#!/usr/bin/env python3

r"""
Generate personalized email drafts for contacts using an LLM.

This tool takes an enriched contacts CSV and a campaign description,
then uses an LLM to generate a personalized email subject and body
for each contact. The drafts are added as columns to the CSV and
can be synced to Google Sheets for review before sending.

Usage
-----
    > outreach_draft \
        --in_path enriched.csv \
        --campaign_goal "Introduce our AI analytics platform to VCs" \
        --out_path drafts.csv

Examples
--------
    # Generate email drafts for all contacts.
    > outreach_draft \
        --in_path enriched.csv \
        --campaign_goal "Introduce our causal AI platform to VC partners" \
        --sender_name "GP" \
        --sender_title "CEO, Causify" \
        --out_path drafts.csv

    # Use a custom template prompt.
    > outreach_draft \
        --in_path enriched.csv \
        --campaign_goal "Follow up after demo" \
        --template_prompt "Write a brief follow-up email" \
        --out_path drafts.csv

    # Upload drafts to Google Sheets for review.
    > outreach_draft \
        --in_path enriched.csv \
        --campaign_goal "Cold outreach to fintech founders" \
        --upload_gsheet \
        --sheet_name "Q1 outreach drafts" \
        --out_path drafts.csv

Import as:

import ai_outreach.scripts.outreach_draft as aoscoudr
"""

import argparse
import logging
from typing import Union

import pandas as pd

import ai_outreach.scripts.utils as aouscuti
import helpers.hdbg as hdbg
import helpers.hllm_cli as hllmcli
import helpers.hparser as hparser
import helpers.hprint as hprint

_LOG = logging.getLogger(__name__)


# #############################################################################
# Prompt.
# #############################################################################


_DEFAULT_DRAFT_PROMPT = """\
You are an expert at writing personalized cold outreach emails.

Campaign goal: {campaign_goal}
Sender: {sender_name}, {sender_title}

Write a short, personalized cold email for the contact below.
The email should:
- Be concise (3-5 sentences in the body)
- Reference the contact's company or role to show research
- Have a clear call to action
- Sound human, not templated

Return ONLY a JSON object with two keys:
- "subject": the email subject line
- "body": the email body (plain text, use \\n for newlines)
"""


def _get_draft_prompt(
    *,
    campaign_goal: str,
    sender_name: str = "",
    sender_title: str = "",
    template_prompt: str = "",
) -> str:
    """
    Build the LLM prompt for email draft generation.

    :param campaign_goal: description of the campaign objective
    :param sender_name: name of the sender
    :param sender_title: title/role of the sender
    :param template_prompt: custom prompt override
    :return: formatted prompt string
    """
    if template_prompt:
        prompt = template_prompt
    else:
        prompt = _DEFAULT_DRAFT_PROMPT.format(
            campaign_goal=campaign_goal,
            sender_name=sender_name or "the sender",
            sender_title=sender_title or "",
        )
    return prompt


def _extract_contact_context(obj: Union[str, pd.Series]) -> str:
    """
    Extract contact context fields for the LLM prompt.

    :param obj: a contact row (Series) or string
    :return: text describing the contact for the LLM
    """
    if isinstance(obj, pd.Series):
        parts = []
        for col in [
            "first_name",
            "last_name",
            "job_title",
            "company_name",
            "company_domain",
            "biography",
        ]:
            if col in obj.index:
                val = str(obj[col])
                if val and val != "" and val != "nan":
                    parts.append("%s=%s" % (col, val))
        txt = " ".join(parts)
    else:
        hdbg.dassert_isinstance(obj, str)
        txt = obj
    return txt


# #############################################################################
# EmailDrafter
# #############################################################################


class EmailDrafter:
    """
    Generate personalized email drafts for contacts using an LLM.
    """

    def __init__(self, *, model: str = "gpt-5-nano") -> None:
        """
        Initialize the EmailDrafter.

        :param model: LLM model to use for draft generation
        """
        self.model = model

    def run(
        self,
        df: pd.DataFrame,
        *,
        campaign_goal: str,
        sender_name: str = "",
        sender_title: str = "",
        template_prompt: str = "",
        batch_size: int = 20,
    ) -> pd.DataFrame:
        """
        Generate email drafts for each contact.

        :param df: enriched contacts DataFrame
        :param campaign_goal: description of the campaign objective
        :param sender_name: name of the sender
        :param sender_title: title/role of the sender
        :param template_prompt: custom prompt override
        :param batch_size: number of contacts per LLM batch
        :return: DataFrame with `email_draft_subject` and
            `email_draft_body` columns added
        """
        df = df.copy()
        _LOG.info("Generating email drafts for %d contacts", len(df))
        _LOG.info(hprint.to_str("campaign_goal batch_size"))
        # Build the prompt.
        prompt = _get_draft_prompt(
            campaign_goal=campaign_goal,
            sender_name=sender_name,
            sender_title=sender_title,
            template_prompt=template_prompt,
        )
        # Generate drafts using the LLM.
        tag = "Generating email drafts"
        batch_mode = "individual"
        target_col = "email_draft"
        df, stats = hllmcli.apply_llm_prompt_to_df(
            prompt,
            df,
            _extract_contact_context,
            target_col,
            batch_mode,
            batch_size=batch_size,
            model=self.model,
            tag=tag,
        )
        _LOG.info("Draft generation stats: %s", stats)
        # Parse the JSON drafts into separate subject/body columns.
        _parse_draft_json(df)
        # Report results.
        has_draft = (df["email_draft_subject"] != "") & (
            df["email_draft_body"] != ""
        )
        _LOG.info(
            "Generated %d / %d drafts successfully",
            has_draft.sum(),
            len(df),
        )
        return df


def _parse_draft_json(df: pd.DataFrame) -> None:
    """
    Parse the raw LLM output in `email_draft` into separate columns.

    Modifies `df` in place, adding `email_draft_subject` and
    `email_draft_body` columns.

    :param df: DataFrame with `email_draft` column containing JSON
    """
    import json

    df["email_draft_subject"] = ""
    df["email_draft_body"] = ""
    for idx in df.index:
        raw = str(df.loc[idx, "email_draft"])
        if not raw or raw == "nan":
            continue
        # Strip markdown fences if present.
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.startswith("```")]
            text = "\n".join(lines)
        # Parse JSON.
        try:
            draft = json.loads(text)
            df.loc[idx, "email_draft_subject"] = draft.get("subject", "")
            df.loc[idx, "email_draft_body"] = draft.get("body", "")
        except json.JSONDecodeError:
            _LOG.warning("Failed to parse draft JSON for row %s", idx)
            # Fall back: use the raw text as body.
            df.loc[idx, "email_draft_body"] = text


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
        help="Path to input enriched contacts CSV",
    )
    parser.add_argument(
        "--out_path",
        required=True,
        help="Path to output CSV with email drafts",
    )
    parser.add_argument(
        "--campaign_goal",
        required=True,
        help="Description of the campaign objective "
        '(e.g., "Introduce our AI platform to VC partners")',
    )
    parser.add_argument(
        "--sender_name",
        default="",
        help="Name of the email sender",
    )
    parser.add_argument(
        "--sender_title",
        default="",
        help="Title/role of the email sender",
    )
    parser.add_argument(
        "--template_prompt",
        default="",
        help="Custom LLM prompt (overrides default template)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-nano",
        help="LLM model for draft generation (default: gpt-5-nano)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Number of contacts per LLM batch (default: 20)",
    )
    # Google Sheets upload.
    parser.add_argument(
        "--upload_gsheet",
        action="store_true",
        help="Upload drafts to Google Sheets after generation",
    )
    parser.add_argument(
        "--sheet_name",
        default="outreach_drafts",
        help="Google Sheet name (default: outreach_drafts)",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    """
    Run the email draft pipeline from CLI arguments.
    """
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Load contacts.
    df = aouscuti.load_contacts_csv(args.in_path)
    # Generate drafts.
    drafter = EmailDrafter(model=args.model)
    df = drafter.run(
        df,
        campaign_goal=args.campaign_goal,
        sender_name=args.sender_name,
        sender_title=args.sender_title,
        template_prompt=args.template_prompt,
        batch_size=args.batch_size,
    )
    # Save results.
    aouscuti.save_contacts_csv(df, args.out_path)
    _LOG.info("Saved %d contacts with drafts to %s", len(df), args.out_path)
    # Upload to Google Sheets if requested.
    if args.upload_gsheet:
        aouscuti.update_google_sheet(df, sheet_name=args.sheet_name)


if __name__ == "__main__":
    _main(_parse())
