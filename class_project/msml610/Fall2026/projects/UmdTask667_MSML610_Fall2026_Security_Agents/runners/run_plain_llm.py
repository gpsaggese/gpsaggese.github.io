#!/usr/bin/env python

"""
Ask an LLM, with one fixed prompt and no agent loop, to review each snippet.

This is the "plain LLM" baseline: it shows what the agents add on top of the
same model. It works with any OpenAI-compatible API; the default is the free
Groq tier. The API key is read from `.env` (never printed).

Raw responses are cached in `results/raw/<agent>/<run_id>/responses/`, so a run
interrupted by a rate limit can be restarted with the same command.

Rate limits: per-minute 429s are retried after `retry-after`; at the daily limit
(a daily 429, or the shared usage ledger reaching `--daily_token_budget`) the
run stops with exit code 2 and writes no partial results.

# Smoke test on 2 snippets
> runners/run_plain_llm.py --run_id smoke --max_snippets 2

# Full run
> runners/run_plain_llm.py --run_id r1
"""

import argparse
import logging
import os
import sys
import time

import dotenv
import openai
from tqdm import tqdm

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser

import llm_lib as llmlib
import rate_limit_lib as ratlib
import runner_lib as runlib

_LOG = logging.getLogger(__name__)

# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    runlib.add_runner_args(parser)
    parser.add_argument(
        "--agent", action="store", default="plain_llm", help="Name of the run"
    )
    parser.add_argument(
        "--model",
        action="store",
        default="openai/gpt-oss-120b",
        help="Model id at the provider",
    )
    parser.add_argument(
        "--base_url",
        action="store",
        default="https://api.groq.com/openai/v1",
        help="OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--api_key_env",
        action="store",
        default="GROQ_API_KEY",
        help="Name of the env var (in `.env`) holding the API key",
    )
    parser.add_argument(
        "--max_snippets",
        action="store",
        type=int,
        default=0,
        help="Only review the first N snippets (0 = all)",
    )
    parser.add_argument(
        "--pause_seconds",
        action="store",
        type=float,
        default=2.0,
        help="Pause between uncached requests, to stay under free-tier limits",
    )
    parser.add_argument(
        "--usage_ledger",
        action="store",
        default=ratlib.DEFAULT_LEDGER_FILE,
        help="JSONL ledger of API calls, shared by all runners using the key",
    )
    parser.add_argument(
        "--daily_token_budget",
        action="store",
        type=int,
        default=ratlib.DEFAULT_DAILY_TOKEN_BUDGET,
        help="Stop before a call once the ledger reaches this many tokens/24h",
    )
    parser.add_argument(
        "--stop_margin_tokens",
        action="store",
        type=int,
        default=3000,
        help="Tokens kept free below the budget (about one call)",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    dotenv.load_dotenv()
    api_key = os.environ.get(args.api_key_env, "")
    hdbg.dassert(
        api_key, "Set %s in `.env` (see `.env.example`)", args.api_key_env
    )
    # Keep the cached responses of previous attempts of this run.
    paths = runlib.prepare_cases(args, args.agent, keep_cache=True)
    cache_dir = os.path.join(paths["work_dir"], "responses")
    case_map = paths["case_map"]
    if args.max_snippets:
        case_map = case_map.head(args.max_snippets)
    # Only `call_with_cache()` retries, so it can tell daily from per-minute
    # limits.
    client = openai.OpenAI(api_key=api_key, base_url=args.base_url, max_retries=0)
    started_at = runlib.get_timestamp()
    findings = []
    totals = {"tokens_in": 0, "tokens_out": 0, "n_requests": 0, "n_parse_errors": 0}
    start = time.time()
    stop_reason = ""
    n_done = 0
    for row in tqdm(case_map.itertuples(index=False), total=len(case_map)):
        code = hio.from_file(os.path.join(paths["cases_dir"], row.case_file))
        cache_file = os.path.join(cache_dir, row.case_file + ".json")
        messages = llmlib.build_messages(code)
        # Check the daily budget only before calls that cost tokens.
        if not llmlib.is_cached(cache_file, model=args.model, messages=messages):
            tokens_24h = ratlib.get_tokens_in_window(args.usage_ledger)
            if tokens_24h + args.stop_margin_tokens >= args.daily_token_budget:
                stop_reason = (
                    f"ledger at {tokens_24h:,} of {args.daily_token_budget:,} "
                    "tokens in the last 24h"
                )
                break
        try:
            response = llmlib.call_with_cache(
                client,
                model=args.model,
                messages=messages,
                cache_file=cache_file,
            )
        except ratlib.DailyLimitError as e:
            stop_reason = f"provider daily limit: {e}"
            break
        n_done += 1
        totals["tokens_in"] += response["tokens_in"]
        totals["tokens_out"] += response["tokens_out"]
        if not response["cached"]:
            totals["n_requests"] += 1
            ratlib.append_usage(
                args.usage_ledger,
                {
                    "agent": args.agent,
                    "run_id": args.run_id,
                    "case_file": row.case_file,
                    "tokens_in": response["tokens_in"],
                    "tokens_out": response["tokens_out"],
                    **response["rate_limits"],
                },
            )
            _LOG.info(
                "%s",
                ratlib.format_status(
                    response["rate_limits"],
                    tokens_24h=ratlib.get_tokens_in_window(args.usage_ledger),
                    daily_budget=args.daily_token_budget,
                ),
            )
            time.sleep(args.pause_seconds)
        parsed = llmlib.parse_findings(
            response["text"], case_file=row.case_file, agent=args.agent
        )
        if parsed is None:
            _LOG.warning("Could not parse the answer for '%s'", row.case_file)
            totals["n_parse_errors"] += 1
            continue
        findings.extend(parsed)
    if stop_reason:
        # Do not write a partial run: it would be scored as if complete.
        _LOG.warning(
            "Stopped after %s of %s snippets (%s). Answers so far are cached; "
            "rerun the same command after the limit resets.",
            n_done,
            len(case_map),
            stop_reason,
        )
        sys.exit(2)
    meta = {
        "tool_version": f"prompt-{llmlib.get_prompt_version()}",
        "model": args.model,
        "base_url": args.base_url,
        "configs": ["single prompt, temperature 0, json_object"],
        "started_at": started_at,
        "seconds": round(time.time() - start, 2),
        "cost_usd": 0.0,
        **totals,
    }
    runlib.finish_run(findings, meta, args=args, agent=args.agent, case_map=case_map)


if __name__ == "__main__":
    _main(_parse())
