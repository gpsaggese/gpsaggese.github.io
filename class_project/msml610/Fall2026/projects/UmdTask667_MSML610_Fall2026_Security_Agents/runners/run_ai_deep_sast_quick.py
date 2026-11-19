#!/usr/bin/env python
"""Run Cisco's quick scan with Semgrep and a local Foundation-Sec-8B model."""

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import helpers.hdbg as hdbg
import helpers.hparser as hparser

import benchmark_lib as benlib
import cisco_quick_lib as quicklib
import findings_lib as finlib
import runner_lib as runlib

_AGENT = "ai_deep_sast_quick"
_MODEL_SHA = "d0072df70235e92bd996d4efd2347b38c9db530ed83df21f6be3f8958a9832d0"
_MODEL_REPO = "fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF"
_MODEL_REVISION = "16deb1934d4b1d28769f7020c15ae018aa537d1f"
_MODEL_FILE = "foundation-sec-8b-instruct-q8_0.gguf"


def _parse():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=hparser.CustomHelpFormatter
    )
    runlib.add_runner_args(parser)
    parser.add_argument("--tool_dir", default="third_party/ai-deep-sast")
    parser.add_argument(
        "--model_file", default=f"third_party/models/{_MODEL_FILE}"
    )
    parser.add_argument(
        "--llama_binary",
        default="third_party/llama.cpp/build/bin/llama-completion",
    )
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--snippet_ids", nargs="+")
    parser.add_argument("--resume", action="store_true")
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser):
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=False)
    root = Path(__file__).resolve().parent.parent
    tool = Path(args.tool_dir).resolve()
    model = Path(args.model_file).resolve()
    binary = Path(args.llama_binary).resolve()
    for path in (tool / "aideepsast.py", model, binary):
        hdbg.dassert_file_exists(str(path))
    with model.open("rb") as stream:
        actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    hdbg.dassert_eq(actual_sha, _MODEL_SHA, "Unexpected model weights")
    errors = benlib.validate_benchmark(
        args.benchmark_dir,
        labels_file=os.path.join(args.benchmark_dir, "labels.csv"),
    )
    hdbg.dassert_eq(errors, [])
    outputs = finlib.get_run_paths(args.results_dir, _AGENT, args.run_id)
    control_paths = finlib.get_run_paths(
        args.results_dir, "semgrep_cisco_rules", args.run_id
    )
    if any(Path(path).exists() for path in [
        *outputs.values(), *control_paths.values()
    ]):
        parser.error("Run already exists; use a new --run_id")
    paths = runlib.prepare_cases(args, _AGENT, keep_cache=args.resume)
    case_map = paths["case_map"]
    if args.snippet_ids:
        known = set(case_map.snippet_id)
        if not set(args.snippet_ids) <= known:
            parser.error("Unknown snippet ID")
        case_map = case_map[case_map.snippet_id.isin(args.snippet_ids)]
    work = Path(paths["work_dir"]).resolve()
    reports = work / "reports"
    reports.mkdir(exist_ok=True)
    # Keep the target outside git so vendor Semgrep respects every case.
    # Cisco's CLI has no --no-git-ignore option; source copies stay archived.
    with tempfile.TemporaryDirectory(prefix="cisco-quick-cases-") as target:
        for name in case_map.case_file:
            shutil.copyfile(Path(paths["cases_dir"]) / name, Path(target) / name)
        config = {
            "target": target, "output_dir": str(reports),
            "hf_repo": _MODEL_REPO, "hf_file": _MODEL_FILE,
            "ctx_size": 8192, "n_gpu_layers": -1, "threads": 6,
            "max_tokens": args.max_tokens, "temperature": 0.1,
            "llm_timeout": 600, "semgrep_timeout": 300,
            "severity_threshold": "WARNING", "skip_llm": False,
            "skip_llm_rules": "config.",
            "semgrep_config": "p/default,p/secrets," + ",".join(
                str(tool / "config" / name)
                for name in ("custom-secrets.yaml", "custom-zipslip.yaml")
            ),
            "log_level": "INFO", "log_file": str(work / "scanner.log"),
        }
        config_path = work / "scanner_config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        wrappers = work / "bin"
        wrappers.mkdir(exist_ok=True)
        wrapper = wrappers / "llama-completion"
        transport = root / "runners/cisco_local_completion.py"
        wrapper.write_text(
            f"#!{sys.executable}\nimport runpy\n"
            f"runpy.run_path({str(transport)!r}, "
            "run_name='__main__')\n"
        )
        wrapper.chmod(0o755)
        env = dict(os.environ)
        env.update({
            "PATH": str(wrappers) + os.pathsep + str(Path(sys.executable).parent)
            + os.pathsep + env.get("PATH", ""),
            "CISCO_LLAMA_BINARY": str(binary), "CISCO_MODEL_FILE": str(model),
            "CISCO_MODEL_SHA256": _MODEL_SHA,
            "CISCO_LOCAL_CACHE": str(work / "responses"),
            "SEMGREP_SETTINGS_FILE": str(work / "semgrep-settings.yml"),
            "SEMGREP_SEND_METRICS": "off", "SEMGREP_ENABLE_VERSION_CHECK": "0",
        })
        started_at = runlib.get_timestamp()
        start = time.monotonic()
        with (work / "console.log").open("w") as log:
            completed = subprocess.run(
                [sys.executable, str(tool / "aideepsast.py"),
                 "--config", str(config_path)],
                cwd=tool, env=env, stdout=log, stderr=subprocess.STDOUT,
                check=False,
            )
        seconds = round(time.monotonic() - start, 2)
    if completed.returncode not in (0, 1):
        raise RuntimeError("Quick scan failed; inspect console.log")
    semgrep = json.loads((reports / "semgrep_report.json").read_text())
    if semgrep.get("errors"):
        raise RuntimeError("Semgrep reported scan errors")
    scanned = {Path(p).name for p in semgrep["paths"]["scanned"]}
    if not set(case_map.case_file) <= scanned:
        raise RuntimeError("Semgrep did not scan every selected snippet")
    report_path = reports / "owasp_ai_report.json"
    findings = quicklib.read_report(str(report_path))
    report = json.loads(report_path.read_text())
    records = [
        json.loads(p.read_text()) for p in (work / "responses").glob("*.json")
    ]
    statuses = collections.Counter(
        item["ai_analysis"]["status"] for item in report["findings"]
    )
    version = subprocess.check_output(
        ["git", "-C", str(tool), "rev-parse", "HEAD"], text=True
    ).strip()
    meta = {
        "tool_version": version, "model": _MODEL_REPO,
        "model_revision": _MODEL_REVISION, "model_sha256": _MODEL_SHA,
        "llama_cpp_revision": subprocess.check_output(
            ["git", "-C", str(binary.parents[2]), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "started_at": started_at, "seconds": seconds, "cost_usd": 0.0,
        "configs": config, "quality_gate_exit_code": completed.returncode,
        "tokens_in": sum(r["tokens_in"] or 0 for r in records),
        "tokens_out": sum(r["tokens_out"] or 0 for r in records),
        "n_requests": len(records), "analysis_statuses": dict(statuses),
        "token_usage_source": "llama.cpp native timing counters",
        "token_usage_complete": all(
            r["tokens_in"] is not None and r["tokens_out"] is not None
            for r in records
        ),
        "finding_policy": "all upstream Semgrep findings; AI adds analysis",
        "detector": "Semgrep", "llm_role": "enrichment_only",
        "outputs_at_token_limit": sum(
            (r["tokens_out"] or 0) >= args.max_tokens - 1 for r in records
        ),
        "fix_status": "suggested only; not applied or exploit-tested",
        "scan_complete": True,
    }
    runlib.finish_run(
        findings, meta, args=args, agent=_AGENT, case_map=case_map
    )
    # A rule-matched control distinguishes detector coverage from AI prose.
    control = [dict(
        finding, explanation=item["semgrep_message"], fix=""
    ) for finding, item in zip(findings, report["findings"])]
    control_meta = {
        "tool_version": semgrep["version"], "model": "",
        "configs": config["semgrep_config"].split(","),
        "started_at": started_at, "seconds": None,
        "tokens_in": 0, "tokens_out": 0, "n_requests": 0, "cost_usd": 0.0,
        "source": "Semgrep stage of the Cisco quick scan, before AI enrichment",
    }
    runlib.finish_run(
        control, control_meta, args=args, agent="semgrep_cisco_rules",
        case_map=case_map,
    )
    public = Path(args.results_dir).parent / _AGENT
    public.mkdir(parents=True, exist_ok=True)
    for source, suffix in (
        ("owasp_ai_report.md", "report.md"),
        ("owasp_ai_report.json", "report.json"),
        ("owasp_junit_report.xml", "junit.xml"),
    ):
        shutil.copyfile(reports / source, public / f"{args.run_id}.{suffix}")


if __name__ == "__main__":
    _main(_parse())
