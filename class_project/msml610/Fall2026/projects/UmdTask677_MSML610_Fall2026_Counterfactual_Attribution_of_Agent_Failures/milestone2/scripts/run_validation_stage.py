"""Run only the explicitly approved $3 validation stage, never later phases."""
from __future__ import annotations

import contextvars
import csv
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vendor/pr917")]
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = str(ROOT / ".offline_settings")
os.environ["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "1"

import litellm  # noqa: E402
from minisweagent.exceptions import FormatError  # noqa: E402
import caaf_agent as agent  # noqa: E402
import caaf_chess as board  # noqa: E402
import caaf_store as store  # noqa: E402
from m2_runtime.budget import Budget, BudgetBusy, BudgetExhausted, MODEL, NANO  # noqa: E402

STAGE = ROOT / "experiments/m2_v1/validation_stage_20261118"
WORK = ROOT / "paid_runs/validation_stage_20261118"
CURRENT_JOB = contextvars.ContextVar("m2_job_id", default="unassigned")
board._TB_DIR = ROOT / "inputs/tablebases"
litellm.num_retries = 0
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    path = STAGE / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def make_guarded_query(budget, stopped, completion=None):
    completion = completion or litellm.completion

    def query(self, messages, **kwargs):
        if self.config.model_name != MODEL or self.config.model_kwargs != {
            "drop_params": True, "max_tokens": 16000, "thinking": {"type": "adaptive"}
        }:
            raise RuntimeError("Refusing model or policy substitution")
        if kwargs:
            raise RuntimeError("Unexpected per-request sampling overrides")
        prepared = self._prepare_messages_for_api(messages)
        request_hash = hashlib.sha256(store.canonical(prepared)).hexdigest()
        while True:
            if stopped.is_set():
                raise RuntimeError("Stage stopped before another paid request")
            try:
                token = budget.reserve(CURRENT_JOB.get(), request_hash)
                break
            except BudgetBusy:
                stopped.wait(.2)
        try:
            response = completion(
                model=MODEL, messages=prepared, **self.config.model_kwargs,
                num_retries=0, max_retries=0, timeout=180,
                service_tier="standard", inference_geo="global",
                api_base="https://api.anthropic.com",
            )
        except BaseException as exc:
            code = getattr(exc, "status_code", None)
            # Definite pre-inference rejections are free; unknown delivery or
            # server/transport failures keep the maximum request liability.
            budget.settle(token, rejected=code in {400, 401, 403, 404, 413, 422, 429},
                          error_type=type(exc).__name__)
            stopped.set()
            raise RuntimeError(f"Provider request failed ({type(exc).__name__}, status={code}); stage halted") from None
        raw = response.model_dump(mode="json")
        try:
            cost = budget.settle(token, usage=raw.get("usage")) / NANO
        except BaseException:
            stopped.set()
            raise
        cost_output = {"cost": cost}
        try:
            actions = self._parse_actions(response)
        except FormatError as exc:
            # Preserve and charge a billed reply even when its command format
            # is invalid. The original harness determines whether to continue.
            exc.messages[0]["extra"].update(cost_output)
            exc.messages[0]["extra"]["response"] = raw
            raise
        message = response.choices[0].message.model_dump()
        message["extra"] = {"actions": actions, "response": raw,
                            "cost": cost, "timestamp": time.time()}
        return message

    return query


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    approval = json.loads((STAGE / "authorization.json").read_text())
    assert approval["approved_usd"] == 3 and approval["phase"] == "estimator_validation"
    assert approval["remaining_phases_approved"] is False
    if (STAGE / "execution_manifest.json").exists():
        raise RuntimeError("Stage already initialized; inspect recorded outcomes before any restart")
    preflight = json.loads((ROOT / "experiments/m2_v1/force_once_preflight.json").read_text())
    assert len(preflight["cases"]) == 3 and all(all(c["checks"].values()) for c in preflight["cases"])
    provenance = json.loads((ROOT / "source_provenance.json").read_text())
    for record in provenance["verified_files"]:
        assert sha(ROOT / "vendor/pr917" / record["path"]) == record["sha256"]
    assert sha(ROOT / "inputs/archive/m1_runs.tar.gz") == provenance["archive_sha256"]
    jobs = [r for r in csv.DictReader((ROOT / "experiments/m2_v1/planned_jobs.csv").open())
            if r["phase"] == "estimator_validation"]
    assert len(jobs) == 300
    original_manifest = json.loads((ROOT / "experiments/m2_v1/experiment_manifest.json").read_text())
    cases = {c["decision_id"]: c for c in original_manifest["validation"]["selected_decisions"]}
    key_path = Path("/Users/sriramm/.config/caaf_chess_m2/session_api_key")
    os.environ["ANTHROPIC_API_KEY"] = key_path.read_text().strip()
    budget = Budget(WORK / "budget.sqlite", 3*NANO)
    stopped = threading.Event()
    agent.CaafModel.query = make_guarded_query(budget, stopped)
    original = ROOT / "inputs/m1_original/runs/m1_think"
    source_db_sha, source_ledger_sha = sha(original / "manifest.sqlite"), sha(original.parent / "api_usage.jsonl")
    work_store = WORK / "store"
    shutil.copytree(original, work_store)
    (work_store / "manifest.sqlite").chmod(0o600)
    executable_paths = [ROOT / "requirements-preparation.lock", ROOT / "source_provenance.json"]
    for folder in ["m2_runtime", "scripts", "tests", "vendor/pr917"]:
        executable_paths += [p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    release = {str(p.relative_to(ROOT)): sha(p) for p in sorted(executable_paths)}
    save("execution_manifest.json", {
        "phase": "estimator_validation", "approved_usd": 3, "jobs": jobs,
        "source_hashes": release, "source_database_sha256": source_db_sha,
        "source_ledger_sha256": source_ledger_sha, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "maximum_per_request_usd": .665, "workers": 4,
        "pricing_checked_on": "2026-11-18",
        "pricing_url": "https://platform.claude.com/docs/en/about-claude/pricing",
        "bound_basis": "1M input tokens, all expensive 5m cache writes, 16K output tokens, standard global inference; full reservation survives unknown errors",
        "scope": "No profile/stability phase authorization; no live-response warmup; no automatic retries or model substitution.",
    })
    results = []
    start = time.monotonic()

    def run(job):
        CURRENT_JOB.set(job["job_id"])
        base = {"job_id": job["job_id"], "decision_id": job["decision_id"],
                "arm": job["arm"], "replicate": int(job["replicate"])}
        if stopped.is_set():
            return {**base, "status": "not_started_stage_halted"}
        branch = None
        st = store.Store(work_store)
        try:
            config = agent.load_config("chess_haiku_think.yaml")
            config["model"].pop("budget", None)
            intervention = {"type": "free"}
            case = cases[job["decision_id"]]
            if job["arm"] == "keep_original_reply":
                call = next(e for e in st.read_tape(case["branch_id"])
                            if e["kind"] == "model_call" and e["step"] == case["step"])
                reply = call["response"]["content"]
                assert hashlib.sha256(reply.encode()).hexdigest() == job["original_reply_sha256"]
                intervention = {"type": "force", "step": case["step"], "content": reply}
            branch = agent.fork(st, job["checkpoint_id"], config=config, intervention=intervention)
            outcome = agent.run_to_end(branch)
            return {**base, "status": "completed", "branch_id": branch.branch_id,
                    "parent_checkpoint": job["checkpoint_id"], "outcome": outcome}
        except BaseException as exc:
            stopped.set()
            if branch is not None:
                branch.agent.tape.close()
                st.finish_branch(branch.branch_id, "failed", {"error_type": type(exc).__name__})
            # Exception strings can contain provider internals; only the type
            # and a fixed summary are persisted, never credentials or headers.
            return {**base, "status": "infrastructure_error", "error_type": type(exc).__name__,
                    "branch_id": branch.branch_id if branch else None}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            save("results.json", sorted(results, key=lambda r: r["job_id"]))
            summary = {"finished_job_records": len(results),
                       "completed": sum(r["status"] == "completed" for r in results),
                       "infrastructure_errors": sum(r["status"] == "infrastructure_error" for r in results),
                       "elapsed_seconds": round(time.monotonic()-start), "budget": budget.summary()}
            save("progress.json", summary)
            if len(results) % 12 == 0 or stopped.is_set():
                print(json.dumps(summary), flush=True)
    assert sha(original / "manifest.sqlite") == source_db_sha
    assert sha(original.parent / "api_usage.jsonl") == source_ledger_sha
    for relative, expected in release.items():
        assert sha(ROOT / relative) == expected
    save("completion.json", {
        "all_300_completed": len(results) == 300 and all(r["status"] == "completed" for r in results),
        "source_database_unchanged": True, "source_ledger_unchanged": True,
        "executable_source_unchanged": True, "budget": budget.summary(),
        "elapsed_seconds": round(time.monotonic()-start), "later_phases_started": False,
    })
    print("VALIDATION_STAGE_FINISHED", flush=True)


if __name__ == "__main__":
    main()
