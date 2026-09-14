#!/usr/bin/env python3

"""
Search for contacts matching a persona description.

This tool runs in two stages:
  1. Persona-to-query: Use an LLM to generate a LinkedIn Sales Navigator
     boolean search query from a natural-language persona description.
  2. Scrape: Launch a PhantomBuster Sales Navigator Search Export agent
     to scrape matching profiles into a contacts CSV.

Stage 1 can be run alone (to review the query before scraping) or both
stages can run end-to-end.

Usage
-----
    > outreach_search \
        --persona "Series A fintech founders in NYC" \
        --out_path contacts.csv

Examples
--------
    # Generate the SN query only (review before scraping).
    > outreach_search \
        --persona "VP of Engineering at Series B SaaS companies" \
        --query_only \
        --out_path sn_query.json

    # Run end-to-end: persona -> query -> PhantomBuster -> CSV.
    > outreach_search \
        --persona "Partners at crypto-focused VC funds in SF" \
        --linkedin_cookie "$LI_AT" \
        --out_path contacts.csv

    # Use a pre-built SN query URL (skip LLM, go straight to scrape).
    > outreach_search \
        --sn_query_url "https://www.linkedin.com/sales/search/people?..." \
        --linkedin_cookie "$LI_AT" \
        --out_path contacts.csv

Import as:

import ai_outreach.scripts.outreach_search as aoscouse
"""

import argparse
import json
import logging
from typing import Any, Dict, Optional, cast

import pandas as pd

import ai_outreach.plugins.linkedin.phantombuster_api as aoplphap
import ai_outreach.scripts.utils as aouscuti
import helpers.hdbg as hdbg
import helpers.hllm_cli as hllmcli
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)


# #############################################################################
# Prompt.
# #############################################################################

_PERSONA_TO_QUERY_PROMPT = """\
You are an expert at LinkedIn Sales Navigator search.

Given the following persona description, generate an optimized LinkedIn
Sales Navigator boolean search query. Return a JSON object with:
- "query": the boolean search string for the keyword/title field
- "filters": a dict of Sales Navigator filters to apply (e.g.,
  geography, company_size, seniority_level, industry, years_in_position)
- "explanation": a brief explanation of your query strategy

Persona: {persona}

Return ONLY valid JSON, no markdown fences.
"""


# #############################################################################
# ContactSearcher
# #############################################################################


class ContactSearcher:
    """
    Search for contacts matching a persona description.

    Two-stage pipeline:
      1. persona -> SN query (via LLM)
      2. SN query -> scraped profiles (via PhantomBuster)
    """

    def __init__(self, *, model: str = "gpt-5.2") -> None:
        """
        Initialize the ContactSearcher.

        :param model: LLM model for query generation
        """
        self.model = model

    def persona_to_query(self, persona: str) -> Dict[str, Any]:
        """
        Generate a Sales Navigator search query from a persona description.

        :param persona: natural-language persona description
        :return: dict with "query", "filters", "explanation"
        """
        prompt = _PERSONA_TO_QUERY_PROMPT.format(persona=persona)
        _LOG.info("Generating SN query for persona: %s", persona)
        response, _ = hllmcli.apply_llm(prompt, model=self.model)
        # Parse JSON from LLM response.
        response_text = response.strip()
        # Strip markdown fences if present.
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            lines = [line for line in lines if not line.startswith("```")]
            response_text = "\n".join(lines)
        result = cast(Dict[str, Any], json.loads(response_text))
        hdbg.dassert_in("query", result)
        _LOG.info("Generated SN query: %s", result.get("query", ""))
        return result

    def scrape_profiles(
        self,
        sn_query_url: str,
        linkedin_cookie: str,
        *,
        agent_name: str = "outreach_search",
    ) -> pd.DataFrame:
        """
        Scrape LinkedIn profiles via PhantomBuster.

        :param sn_query_url: Sales Navigator search URL
        :param linkedin_cookie: LinkedIn li_at session cookie
        :param agent_name: name for the PhantomBuster agent
        :return: DataFrame of scraped profiles
        """
        phantom = aoplphap.Phantom()
        _LOG.info("Creating PhantomBuster agent: %s", agent_name)
        resp = phantom.create_sales_nav_phantom(
            agent_name, sn_query_url, linkedin_cookie
        )
        agent_id = resp.get("id")
        hdbg.dassert_is_not(agent_id, None, "Failed to create phantom agent")
        _LOG.info("Launching agent %s and waiting for results...", agent_id)
        df = phantom.launch_and_get_df(agent_id)
        _LOG.info("Scraped %d profiles", len(df))
        return df

    def run(
        self,
        *,
        persona: Optional[str] = None,
        sn_query_url: Optional[str] = None,
        linkedin_cookie: Optional[str] = None,
        query_only: bool = False,
        agent_name: str = "outreach_search",
    ) -> Dict[str, Any]:
        """
        Run the search pipeline.

        :param persona: natural-language persona description
        :param sn_query_url: pre-built SN query URL (skips LLM)
        :param linkedin_cookie: LinkedIn li_at session cookie
        :param query_only: if True, stop after generating the query
        :param agent_name: name for the PhantomBuster agent
        :return: dict with "query_result" and optionally "contacts_df"
        """
        output: Dict[str, Any] = {}
        # Stage 1: Generate SN query.
        if sn_query_url is None:
            hdbg.dassert_is_not(
                persona, None, "Must provide --persona or --sn_query_url"
            )
            query_result = self.persona_to_query(persona)
            output["query_result"] = query_result
            _LOG.info("Generated SN query: %s", query_result.get("query", ""))
            _LOG.info("Filters: %s", query_result.get("filters", {}))
            _LOG.info("Strategy: %s", query_result.get("explanation", ""))
        else:
            output["query_result"] = {"sn_query_url": sn_query_url}
        if query_only:
            return output
        # Stage 2: Scrape via PhantomBuster.
        if sn_query_url is None:
            _LOG.info(
                "To scrape, provide the SN query URL via --sn_query_url "
                "(copy from Sales Navigator after reviewing the generated "
                "query)."
            )
            return output
        hdbg.dassert_is_not(
            linkedin_cookie,
            None,
            "Must provide --linkedin_cookie for scraping",
        )
        df = self.scrape_profiles(
            sn_query_url, linkedin_cookie, agent_name=agent_name
        )
        output["contacts_df"] = df
        return output


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
    # Input: persona or pre-built query.
    parser.add_argument(
        "--persona",
        default=None,
        help="Natural-language persona description "
        '(e.g., "Series A fintech founders in NYC")',
    )
    parser.add_argument(
        "--sn_query_url",
        default=None,
        help="Pre-built Sales Navigator search URL (skips LLM query "
        "generation)",
    )
    # PhantomBuster options.
    parser.add_argument(
        "--linkedin_cookie",
        default=None,
        help="LinkedIn li_at session cookie for PhantomBuster",
    )
    parser.add_argument(
        "--agent_name",
        default="outreach_search",
        help="Name for the PhantomBuster agent (default: outreach_search)",
    )
    # Control.
    parser.add_argument(
        "--query_only",
        action="store_true",
        help="Only generate the SN query (do not scrape)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="LLM model for query generation (default: gpt-5.2)",
    )
    # Output.
    parser.add_argument(
        "--out_path",
        required=True,
        help="Output path: CSV for contacts, JSON if --query_only",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    """
    Run the search pipeline from CLI arguments.
    """
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    searcher = ContactSearcher(model=args.model)
    output = searcher.run(
        persona=args.persona,
        sn_query_url=args.sn_query_url,
        linkedin_cookie=args.linkedin_cookie,
        query_only=args.query_only,
        agent_name=args.agent_name,
    )
    # Write output.
    if args.query_only or "contacts_df" not in output:
        # Write query result as JSON.
        out_path = args.out_path
        if not out_path.endswith(".json"):
            out_path = out_path.rsplit(".", 1)[0] + ".json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output["query_result"], f, indent=2)
        _LOG.info("Query saved to %s", out_path)
    else:
        # Write contacts as CSV.
        df = output["contacts_df"]
        aouscuti.save_contacts_csv(df, args.out_path)
        _LOG.info("Saved %d contacts to %s", len(df), args.out_path)


if __name__ == "__main__":
    _main(_parse())
