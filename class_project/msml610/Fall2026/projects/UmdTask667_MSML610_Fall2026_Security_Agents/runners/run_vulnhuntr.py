#!/usr/bin/env python

"""
Run Vulnhuntr (Protect AI) on each snippet and write normalized findings.

Vulnhuntr needs Python 3.10, so it runs in its own Docker image (see
`docker/vulnhuntr.Dockerfile`), never in the project venv. It talks to any
OpenAI-compatible API; the default is `openai/gpt-oss-120b` on the free Groq
tier. The API key is read from `.env` and passed to Docker by name only (never
printed or written to disk).

How it works:
- One container per snippet, with only that snippet mounted, so a bad model
  answer loses one snippet and context lookups cannot pull in other snippets
- Vulnhuntr's JSON log is kept in `results/raw/vulnhuntr/<run_id>/out/<case>/`;
  rerunning the same command skips snippets that already finished
- Each vulnerability type in Vulnhuntr's final report becomes one finding
  - Vulnhuntr reports no line, so `line` is `findings_lib.NO_LINE` and the run
    is scored at snippet level only
  - Findings below `--min_confidence` (0-10) are not written; the raw log keeps
    them, so another threshold needs no new API calls
- Rate limits: a hook (`docker/vulnhuntr_hooks/`) logs every API call with its
  rate limit headers and tokens, and retries per-minute 429s. At the daily
  limit (a daily 429, or the shared usage ledger near `--daily_token_budget`)
  the run stops with exit code 2 and writes no partial results

# Smoke test on one snippet
> runners/run_vulnhuntr.py --run_id smoke --snippet_ids SEED-001

# Full run
> runners/run_vulnhuntr.py --run_id r1
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

import dotenv
import pandas as pd
from tqdm import tqdm

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser

import rate_limit_lib as ratlib
import runner_lib as runlib
import vulnhuntr_lib as vullib

_LOG = logging.getLogger(__name__)

# Dir with `sitecustomize.py`, which logs API calls inside the container.
_HOOKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docker",
    "vulnhuntr_hooks",
)
# File written by the hook in each snippet's output dir.
_CALL_LOG = "groq_calls.jsonl"

# #############################################################################
# Docker
# #############################################################################


def _run_one(
    args: argparse.Namespace,
    *,
    case_path: str,
    out_dir: str,
    env: Dict[str, str],
) -> Dict[str, Any]:
    """
    Run Vulnhuntr on one snippet in a fresh container.

    :return: status dict stored in `status.json`
    """
    case_file = os.path.basename(case_path)
    case_sha = _get_file_sha(case_path)
    hio.create_dir(out_dir, incremental=False)
    cmd = [
        "docker",
        "run",
        "--rm",
        # Pass the variables by name: the key never appears in the command.
        "-e",
        "OPENAI_API_KEY",
        "-e",
        "OPENAI_BASE_URL",
        "-e",
        "OPENAI_MODEL",
        # Load the call-logging hook (Vulnhuntr's code is unchanged).
        "-e",
        "PYTHONPATH=/hooks",
        "-e",
        f"VULNHUNTR_CALL_LOG=/out/{_CALL_LOG}",
        "-e",
        f"OPENAI_MAX_RETRIES={args.max_retries}",
        "-v",
        f"{_HOOKS_DIR}:/hooks:ro",
        "-v",
        f"{os.path.abspath(case_path)}:/src/{case_file}:ro",
        "-v",
        f"{os.path.abspath(out_dir)}:/out",
        args.image,
        "-r",
        "/src",
        "-a",
        case_file,
        "-l",
        "gpt",
        "-v",
    ]
    _LOG.debug("Running %s", " ".join(cmd))
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
        )
        returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        status = vullib.classify_failure(returncode, stderr)
    except subprocess.TimeoutExpired:
        returncode, stdout, stderr = -1, "", ""
        status = vullib.STATUS_TIMEOUT
    hio.to_file(os.path.join(out_dir, "stdout.txt"), stdout)
    hio.to_file(os.path.join(out_dir, "stderr.txt"), stderr)
    calls = _read_calls(out_dir)
    # A failure caused by a limit is reported as such, not as a crash.
    if status != vullib.STATUS_OK and calls["daily_limit"]:
        status = vullib.STATUS_DAILY_LIMIT
    elif status != vullib.STATUS_OK and calls["too_large"]:
        status = vullib.STATUS_TOO_LARGE
    result = {
        "case_file": case_file,
        "case_sha": case_sha,
        "status": status,
        "returncode": returncode,
        "seconds": round(time.time() - start, 2),
        "n_calls": len(calls["calls"]),
        "tokens_in": calls["tokens_in"],
        "tokens_out": calls["tokens_out"],
    }
    hio.to_file(os.path.join(out_dir, "status.json"), json.dumps(result))
    return result


def _get_file_sha(path: str) -> str:
    """
    Return a hash of a file, to detect snippets edited since their last run.
    """
    sha = hashlib.sha1(hio.from_file(path).encode("utf-8")).hexdigest()
    return sha


def _read_calls(out_dir: str) -> Dict[str, Any]:
    """
    Summarize the API calls logged by the hook in `out_dir`.
    """
    log_file = os.path.join(out_dir, _CALL_LOG)
    text = hio.from_file(log_file) if os.path.exists(log_file) else ""
    summary = vullib.summarize_calls(text)
    return summary


def _get_docker_env(args: argparse.Namespace) -> Dict[str, str]:
    """
    Build the environment of the `docker` process, with the API key.
    """
    dotenv.load_dotenv()
    api_key = os.environ.get(args.api_key_env, "")
    hdbg.dassert(
        api_key, "Set %s in `.env` (see `.env.example`)", args.api_key_env
    )
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = args.base_url
    env["OPENAI_MODEL"] = args.model
    return env


def _select_cases(
    case_map: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    """
    Keep the snippets requested with `--snippet_ids` or `--max_snippets`.
    """
    if args.snippet_ids:
        wanted = [sid.strip() for sid in args.snippet_ids.split(",")]
        missing = sorted(set(wanted) - set(case_map["snippet_id"]))
        hdbg.dassert_eq(len(missing), 0, "Unknown snippet ids: %s", missing)
        case_map = case_map[case_map["snippet_id"].isin(wanted)]
    if args.max_snippets:
        case_map = case_map.head(args.max_snippets)
    case_map = case_map.reset_index(drop=True)
    return case_map


# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    runlib.add_runner_args(parser)
    parser.add_argument(
        "--agent", action="store", default="vulnhuntr", help="Name of the run"
    )
    parser.add_argument(
        "--image",
        action="store",
        default="vulnhuntr:ead88c5",
        help="Docker image built from `docker/vulnhuntr.Dockerfile`",
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
        "--snippet_ids",
        action="store",
        default="",
        help="Comma-separated snippet ids to review, e.g., `SEED-001`",
    )
    parser.add_argument(
        "--max_snippets",
        action="store",
        type=int,
        default=0,
        help="Only review the first N snippets (0 = all)",
    )
    parser.add_argument(
        "--min_confidence",
        action="store",
        type=int,
        default=7,
        help="Lowest Vulnhuntr confidence (0-10) that becomes a finding",
    )
    parser.add_argument(
        "--timeout_seconds",
        action="store",
        type=int,
        default=600,
        help="Max seconds for one snippet",
    )
    parser.add_argument(
        "--pause_seconds",
        action="store",
        type=float,
        default=5.0,
        help="Pause between snippets, to stay under free-tier limits",
    )
    parser.add_argument(
        "--max_retries",
        action="store",
        type=int,
        default=8,
        help="Retries of a per-minute rate limit inside the container",
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
        help="Tokens per 24h the key may use, checked against the ledger",
    )
    parser.add_argument(
        "--snippet_token_reserve",
        action="store",
        type=int,
        default=20000,
        help="Do not start a snippet unless this many tokens are left today",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    hdbg.dassert_lte(0, args.min_confidence)
    hdbg.dassert_lte(args.min_confidence, 10)
    env = _get_docker_env(args)
    # Keep the logs of previous attempts of this run.
    paths = runlib.prepare_cases(args, args.agent, keep_cache=True)
    case_map = _select_cases(paths["case_map"], args)
    started_at = runlib.get_timestamp()
    start = time.time()
    findings: List[Dict[str, Any]] = []
    totals = {
        "tokens_in": 0,
        "tokens_out": 0,
        "n_requests": 0,
        "n_parse_errors": 0,
        "n_api_errors": 0,
        "n_too_large": 0,
        "n_other_errors": 0,
        "n_below_threshold": 0,
    }
    statuses = []
    stop_reason = ""
    for row in tqdm(case_map.itertuples(index=False), total=len(case_map)):
        case_path = os.path.join(paths["cases_dir"], row.case_file)
        out_dir = os.path.join(
            paths["work_dir"], "out", os.path.splitext(row.case_file)[0]
        )
        status_file = os.path.join(out_dir, "status.json")
        # Reuse a finished snippet whose code is unchanged; rerun the rest.
        result = None
        if os.path.exists(status_file):
            result = json.loads(hio.from_file(status_file))
            is_done = result["status"] == vullib.STATUS_OK
            is_same = result.get("case_sha", "") in ("", _get_file_sha(case_path))
            if not (is_done and is_same):
                result = None
        if result is None:
            # Vulnhuntr cannot resume inside a snippet, so do not start one that
            # the daily budget may cut off.
            tokens_24h = ratlib.get_tokens_in_window(args.usage_ledger)
            if tokens_24h + args.snippet_token_reserve > args.daily_token_budget:
                stop_reason = (
                    f"ledger at {tokens_24h:,} of {args.daily_token_budget:,} "
                    "tokens in the last 24h"
                )
                break
            result = _run_one(args, case_path=case_path, out_dir=out_dir, env=env)
            calls = _read_calls(out_dir)
            for call in calls["calls"]:
                ratlib.append_usage(
                    args.usage_ledger,
                    {
                        "agent": args.agent,
                        "run_id": args.run_id,
                        "case_file": row.case_file,
                        "status": call.get("status"),
                        "tokens_in": call.get("tokens_in", 0),
                        "tokens_out": call.get("tokens_out", 0),
                        **ratlib.extract_rate_limits(call),
                    },
                )
            _LOG.info(
                "%s: %s, %s calls, %s tokens. %s",
                row.case_file,
                result["status"],
                result["n_calls"],
                f"{result['tokens_in'] + result['tokens_out']:,}",
                ratlib.format_status(
                    ratlib.extract_rate_limits(calls["rate_limits"]),
                    tokens_24h=ratlib.get_tokens_in_window(args.usage_ledger),
                    daily_budget=args.daily_token_budget,
                ),
            )
            if result["status"] == vullib.STATUS_DAILY_LIMIT:
                stop_reason = "provider daily limit"
                break
            time.sleep(args.pause_seconds)
        statuses.append(result)
        totals["tokens_in"] += int(result.get("tokens_in") or 0)
        totals["tokens_out"] += int(result.get("tokens_out") or 0)
        log_file = os.path.join(out_dir, "vulnhuntr.log")
        log_text = hio.from_file(log_file) if os.path.exists(log_file) else ""
        parsed = vullib.parse_vulnhuntr_log(log_text)
        totals["n_requests"] += parsed["n_requests"]
        status = result["status"]
        if status != vullib.STATUS_OK:
            # Same rule as `run_plain_llm.py`: no findings from a failed answer.
            key = {
                vullib.STATUS_PARSE_ERROR: "n_parse_errors",
                vullib.STATUS_API_ERROR: "n_api_errors",
                vullib.STATUS_TOO_LARGE: "n_too_large",
            }.get(status, "n_other_errors")
            totals[key] += 1
            _LOG.warning("Vulnhuntr %s on '%s'", status, row.case_file)
            continue
        snippet_findings = vullib.reports_to_findings(
            parsed,
            case_file=row.case_file,
            agent=args.agent,
            min_confidence=args.min_confidence,
        )
        totals["n_below_threshold"] += len(parsed["final"]) - len(
            snippet_findings
        )
        findings.extend(snippet_findings)
    if stop_reason:
        # Do not write a partial run: it would be scored as if complete.
        _LOG.warning(
            "Stopped after %s of %s snippets (%s). Finished snippets are kept; "
            "rerun the same command after the limit resets.",
            len(statuses),
            len(case_map),
            stop_reason,
        )
        sys.exit(2)
    meta = {
        "tool_version": f"vulnhuntr {args.image.split(':')[-1]}",
        "model": args.model,
        "base_url": args.base_url,
        "configs": [
            "one container per snippet",
            f"min_confidence={args.min_confidence}",
            "no line numbers: scored at snippet level",
        ],
        "started_at": started_at,
        "seconds": round(time.time() - start, 2),
        "cost_usd": 0.0,
        **totals,
        "statuses": statuses,
    }
    runlib.finish_run(findings, meta, args=args, agent=args.agent, case_map=case_map)


if __name__ == "__main__":
    _main(_parse())
