#!/usr/bin/env python

"""
Run Cisco AI Deep SAST on the anonymized benchmark.

> python runners/run_ai_deep_sast.py --run_id r1 --dry_run
> python runners/run_ai_deep_sast.py --run_id r1

Install Cisco in third_party/ai-deep-sast and its requirements in the venv.
Provide the API key through the environment or the local git-ignored .env.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from dotenv import load_dotenv

import helpers.hdbg as hdbg
import helpers.hparser as hparser

import ai_deep_sast_lib as cislib
import benchmark_lib as benlib
import findings_lib as finlib
import runner_lib as runlib

_AGENT = "ai_deep_sast"
_LOG = logging.getLogger(__name__)


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=hparser.CustomHelpFormatter
    )
    runlib.add_runner_args(parser)
    parser.add_argument("--tool_dir", default="third_party/ai-deep-sast")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--base_url", default="https://api.groq.com/openai/v1")
    parser.add_argument("--api_key_env", default="GROQ_API_KEY")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=False)
    load_dotenv()
    tool_dir = Path(args.tool_dir).resolve()
    hdbg.dassert_file_exists(str(tool_dir / "deepscan.py"))
    errors = benlib.validate_benchmark(
        args.benchmark_dir,
        labels_file=os.path.join(args.benchmark_dir, "labels.csv"),
    )
    hdbg.dassert_eq(errors, [])
    output_paths = finlib.get_run_paths(
        args.results_dir, _AGENT, args.run_id
    )
    for path in output_paths.values():
        if os.path.exists(path):
            parser.error("Completed run already exists; use a new --run_id")
    if not args.dry_run and not os.environ.get(args.api_key_env):
        parser.error(
            f"Missing {args.api_key_env}; export the API key before running"
        )
    paths = runlib.prepare_cases(args, _AGENT, keep_cache=args.resume)
    work_dir = Path(paths["work_dir"]).resolve()
    if args.resume:
        archive = work_dir / "attempts" / uuid.uuid4().hex[:12]
        archive.mkdir(parents=True, exist_ok=False)
        for name in ("summary.json", "scan.log"):
            if (work_dir / name).exists():
                shutil.copyfile(work_dir / name, archive / name)
    report_dir = work_dir / "reports"
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("cisco_deepscan_entry.py")),
        "--target", str(Path(paths["cases_dir"]).resolve()),
        "--output-dir", str(report_dir),
        "--llm-url", args.base_url, "--llm-model", args.model,
        "--json-summary",
    ]
    if args.dry_run:
        command.append("--dry-run")
    env = dict(os.environ)
    env["CISCO_TOOL_DIR"] = str(tool_dir)
    env["CISCO_CACHE_DIR"] = str(work_dir / "responses")
    if not args.dry_run:
        env["LLM_API_KEY"] = env[args.api_key_env]
    env["SEMGREP_SETTINGS_FILE"] = str(work_dir / "semgrep-settings.yml")
    started_at = runlib.get_timestamp()
    with open(work_dir / "summary.json", "w") as stdout, open(
        work_dir / "scan.log", "w"
    ) as stderr:
        completed = subprocess.run(
            command, cwd=tool_dir, env=env, stdout=stdout, stderr=stderr,
            check=False,
        )
    try:
        with open(work_dir / "summary.json", encoding="utf-8") as stream:
            summary = json.load(stream)
    except (ValueError, OSError):
        log_text = (work_dir / "scan.log").read_text()
        error = (
            "Provider daily free-tier quota exhausted"
            if "Provider daily free-tier quota exhausted" in log_text
            else "No complete scan summary"
        )
        summary = {"success": False, "error": error}
    version = subprocess.check_output(
        ["git", "-C", str(tool_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    status = {
        "agent": _AGENT, "run_id": args.run_id,
        "status": "indexed_only" if args.dry_run else "pending_validation",
        "tool_version": version, "model": args.model,
        "started_at": started_at, "returncode": completed.returncode,
        "summary": summary,
    }
    status_dir = Path(args.results_dir).parent / _AGENT
    status_dir.mkdir(parents=True, exist_ok=True)
    status_file = status_dir / f"{args.run_id}.status.json"
    previous_attempts = []
    if args.resume and status_file.exists():
        previous = json.loads(status_file.read_text())
        previous_attempts = previous.get("previous_attempts", []) + [{
            "started_at": previous.get("attempt_started_at",
                                       previous["started_at"]),
            "status": previous["status"],
            "seconds": previous.get("summary", {}).get("elapsed_seconds", 0),
        }]
        status["started_at"] = previous["started_at"]
    status["attempt_started_at"] = started_at
    status["previous_attempts"] = previous_attempts

    def write_status() -> None:
        status_file.write_text(json.dumps(status, indent=2) + "\n")

    write_status()
    try:
        if completed.returncode:
            raise RuntimeError("Cisco returned a nonzero exit status")
        cislib.check_scan(summary, (work_dir / "scan.log").read_text())
    except RuntimeError as exc:
        status.update(status="failed", error=str(exc))
        write_status()
        raise
    if args.dry_run:
        _LOG.info("Index-only check completed; no detection scores written")
        return
    findings = cislib.read_report(
        str(report_dir / "deepscan_report.json"),
        str(report_dir / "deepscan_index.json"),
    )
    offsets = paths["case_map"].set_index("case_file")["line_offset"]
    for finding in findings:
        if finding["case_file"] not in offsets.index:
            raise RuntimeError("Cisco returned a finding on an unknown file")
        if finding["line"] == 0:
            finding["line"] = -int(offsets[finding["case_file"]])
    usage = summary.get("token_usage", {})
    # Include completed responses from earlier attempts of a resumed run.
    # These files contain response text and usage, never credentials.
    responses = [
        json.loads(path.read_text())
        for path in (work_dir / "responses").glob("*.json")
    ]
    if responses:
        usage = {
            "total_input_tokens": sum(r["input_tokens"] for r in responses),
            "total_output_tokens": sum(r["output_tokens"] for r in responses),
            "total_calls": len(responses),
        }
    meta = {
        "tool_version": version, "model": args.model,
        "base_url": args.base_url, "configs": ["default-brute-force"],
        "started_at": status["started_at"],
        "seconds": round(summary["elapsed_seconds"] + sum(
            attempt["seconds"] for attempt in previous_attempts
        ), 2),
        "tokens_in": usage.get("total_input_tokens", 0),
        "tokens_out": usage.get("total_output_tokens", 0),
        "n_requests": usage.get("total_calls", 0),
        "cost_usd": None,
        "line_resolution": "function_start_or_unknown_zero",
        "finding_policy": "published true-positive verdicts only",
        "transport": "cached_validated_json",
        "reasoning_effort": (
            "low" if args.model.startswith("openai/gpt-oss") else None
        ),
        "max_completion_tokens": 4096,
        "temperature": 0,
        "usage_scope": "all saved API responses across run attempts",
        "previous_attempts": previous_attempts,
        "upstream_summary": summary,
    }
    runlib.finish_run(
        findings, meta, args=args, agent=_AGENT, case_map=paths["case_map"]
    )
    status.update(status="completed")
    write_status()


if __name__ == "__main__":
    _main(_parse())
