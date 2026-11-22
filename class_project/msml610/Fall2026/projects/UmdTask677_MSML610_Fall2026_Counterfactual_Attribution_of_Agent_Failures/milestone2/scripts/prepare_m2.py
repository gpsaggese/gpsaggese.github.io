"""Prepare inventories and a proposed experiment; never import a model client.

Run with Python 3.12 and chess==1.11.2. The original databases are read-only.
The only imported vendor modules are the board and store implementations.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/m2_v1"
STORE = ROOT / "inputs/m1_original/runs/m1_think"
sys.path.insert(0, str(ROOT / "vendor/pr917"))
import caaf_chess as board  # noqa: E402
import caaf_store as store  # noqa: E402

board._TB_DIR = ROOT / "inputs/tablebases"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def blob(kind, key):
    p = STORE / "blobs" / kind / f"{key}.json"
    assert digest(p) == key
    return json.loads(p.read_text())


def usage_cost(u):
    """Reprice historical raw API usage at verified standard 5m-cache rates."""
    prompt = u.get("prompt_tokens") or 0
    read = u.get("cache_read_input_tokens") or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    write = u.get("cache_creation_input_tokens") or 0
    uncached = prompt - read - write if prompt >= read + write else prompt
    prompt = max(prompt, uncached + read + write)
    pin, pout = (0.50, 2.50) if prompt > 100000 else (0.10, 0.50)
    return (uncached * pin + write * pin * 1.25 + read * pin * 0.1 + (u.get("completion_tokens") or 0) * pout) / 1e6


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original_db = STORE / "manifest.sqlite"
    before_db = digest(original_db)
    before_ledger = digest(ROOT / "inputs/m1_original/runs/api_usage.jsonl")
    db = sqlite3.connect(f"file:{original_db}?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    assert db.execute("pragma integrity_check").fetchone()[0] == "ok"
    integrity = {"blobs": 0, "tapes": 0, "archive_sha256": digest(ROOT / "inputs/archive/m1_runs.tar.gz")}
    assert integrity["archive_sha256"] == "d63ef2d94a695280cffa1896bbe201e0f2d3f79def9b90751801c6f09e3a7172"
    for p in sorted((STORE / "blobs").glob("*/*.json")):
        assert digest(p) == p.stem, p
        integrity["blobs"] += 1
    tapes = {}
    for p in sorted((STORE / "tapes").glob("*.jsonl")):
        records = [json.loads(line) for line in p.read_text().splitlines() if line]
        assert store.verify_tape(records), p
        tapes[p.stem] = records
        integrity["tapes"] += 1
    provenance = json.loads((ROOT / "source_provenance.json").read_text())
    for row in provenance["verified_files"]:
        assert digest(ROOT / "vendor/pr917" / row["path"]) == row["sha256"]
    for name, expected in {
        "KRvK.rtbw": "386fbde73308e49a4207836922c68b30b664e83c5a37f7fa37305a15cd16f2f1",
        "KRvK.rtbz": "cab59f42e75c2a25da3939231850a79fa838593fff5c28d7772da92062ed965a",
    }.items():
        assert digest(ROOT / "inputs/tablebases" / name) == expected

    roots = db.execute("select b.*,r.config_json from branches b join runs r using(run_id) where b.parent_snapshot_id is null order by r.config_json").fetchall()
    roots = sorted(roots, key=lambda b: json.loads(b["config_json"])["label"])
    inventory, decisions, runs, contexts = [], [], [], {}
    baseline_calls = 0
    for b in roots:
        meta, outcome = json.loads(b["config_json"]), json.loads(b["outcome_json"])
        _, task, trial = meta["label"].split(":")
        assert b["status"] == "finished"
        snapshots = [dict(s) for s in db.execute("select * from snapshots where branch_id=? order by seq", (b["branch_id"],))]
        calls = {e["step"]: e for e in tapes[b["branch_id"]] if e["kind"] == "model_call"}
        assert len(snapshots) == len(calls) + 1
        assert [s["step"] for s in snapshots] == list(range(len(calls) + 1))
        baseline_calls += len(calls)
        rows = []
        for i, snap in enumerate(snapshots):
            context, game = blob("context", snap["context_hash"]), blob("env", snap["env_hash"])
            contexts[snap["id"]] = context
            chessboard = board.board_of(game)
            value = board.white_value(chessboard)
            has_next = snap["step"] + 1 in calls
            row = {
                "task_id": task, "trial": int(trial), "run_id": b["run_id"], "branch_id": b["branch_id"],
                "source_failed": not outcome["won"], "source_outcome": outcome["status"],
                "checkpoint_id": snap["id"], "step": snap["step"], "seq": snap["seq"],
                "parent_checkpoint_id": snap["parent_id"] or "", "context_sha256": snap["context_hash"],
                "env_sha256": snap["env_hash"],
                "context_path": str((STORE / "blobs/context" / f"{snap['context_hash']}.json").relative_to(ROOT)),
                "env_path": str((STORE / "blobs/env" / f"{snap['env_hash']}.json").relative_to(ROOT)),
                "fen": chessboard.fen(), "halfmove_clock": chessboard.halfmove_clock,
                "game_status": game["status"], "has_next_model_call": has_next,
                "remaining_call_allowance": context["counters"]["step_limit"] - context["counters"]["n_calls"],
                "original_remaining_calls": len(calls) - snap["step"],
                "oracle_win": value["win"], "oracle_dtz": value["dtz"],
                "oracle_terminal": value["terminal"] or "", "model": context["model"]["name"],
                "harness_commit": context["harness"]["harness_commit"],
                "harness_dirty": context["harness"]["harness_dirty"],
                "profile_eligible": not outcome["won"] and has_next,
            }
            inventory.append(row); rows.append(row)
        for before, after in zip(rows, rows[1:]):
            k = after["step"]
            reply = calls[k]["response"]["content"]
            commands = [e["request"]["command"] for e in tapes[b["branch_id"]] if e["step"] == k and e["kind"] == "command"]
            decisions.append({
                "decision_id": f"{task}_trial{trial}_step{k:03d}", "task_id": task, "trial": int(trial),
                "run_id": b["run_id"], "branch_id": b["branch_id"], "source_failed": not outcome["won"],
                "source_outcome": outcome["status"], "step": k,
                "before_checkpoint": before["checkpoint_id"], "after_checkpoint": after["checkpoint_id"],
                "original_reply_sha256": hashlib.sha256(reply.encode()).hexdigest(),
                "commands_json": json.dumps(commands), "oracle_before_win": before["oracle_win"],
                "oracle_after_win": after["oracle_win"], "after_status": after["game_status"],
                "profile_eligible": not outcome["won"],
            })
        runs.append({"task_id": task, "trial": int(trial), "branch_id": b["branch_id"], "run_id": b["run_id"],
                     "outcome": outcome["status"], "won": outcome["won"], "calls": len(calls), "checkpoints": len(snapshots)})

    eligible = [r for r in inventory if r["profile_eligible"]]
    failed_decisions = [d for d in decisions if d["profile_eligible"]]
    # Select using original trajectories and oracle labels only, never M1 screen
    # success rates or any new M2 outcomes. These are diagnostic validation cases.
    selected, used_tasks = [], set()
    predicates = [
        ("loss_of_forced_win_nonterminal", lambda d: d["oracle_before_win"] and not d["oracle_after_win"] and d["after_status"] == "ongoing"),
        ("still_winning_nonterminal", lambda d: d["oracle_before_win"] and d["oracle_after_win"] and d["after_status"] == "ongoing"),
        ("terminal_rook_loss", lambda d: d["oracle_before_win"] and d["after_status"] == "insufficient_material"),
    ]
    for role, predicate in predicates:
        options = [d for d in failed_decisions if predicate(d) and d["task_id"] not in used_tasks]
        assert options, role
        chosen = dict(options[0]);chosen["selection_role"] = role
        selected.append(chosen);used_tasks.add(chosen["task_id"])

    # Repeat two entire profiles from distinct positions, selected by task ID,
    # independent of their future attribution results.
    stability_runs = [r for r in runs if not r["won"] and r["trial"] == 0 and r["task_id"] in {"krk-m3-1", "krk-m6-1"}]
    stability_branches = {r["branch_id"] for r in stability_runs}
    jobs = []
    def add(phase, cp, arm, rep, decision="", forced_step="", reply_hash=""):
        jobs.append({"job_id": f"m2v1_{len(jobs):05d}", "phase": phase, "checkpoint_id": cp,
                     "arm": arm, "replicate": rep, "decision_id": decision,
                     "forced_step": forced_step, "original_reply_sha256": reply_hash,
                     "status": "BLOCKED_AWAITING_BUDGET_APPROVAL"})
    for rep in range(50):
        for d in selected:
            for arm in ["keep_original_reply", "redecide_reply"]:
                add("estimator_validation", d["before_checkpoint"], arm, rep, d["decision_id"],
                    d["step"] if arm == "keep_original_reply" else "",
                    d["original_reply_sha256"] if arm == "keep_original_reply" else "")
    for rep in range(30):
        for row in eligible:
            add("profiles_primary", row["checkpoint_id"], "free", rep)
    for rep in range(30):
        for row in eligible:
            if row["branch_id"] in stability_branches:
                add("profiles_stability", row["checkpoint_id"], "free", rep)

    # Historical fresh forks, excluding root runs and zero-call replay forks.
    fresh_costs, fresh_calls = [], []
    for b in db.execute("select * from branches where parent_snapshot_id is not null"):
        if json.loads(b["intervention_json"] or "{}").get("type") != "free" or b["status"] != "finished":
            continue
        calls = [e for e in tapes[b["branch_id"]] if e["kind"] == "model_call" and e["response"].get("source") == "model"]
        fresh_calls.append(len(calls))
        fresh_costs.append(sum(usage_cost(e["response"].get("usage") or {}) for e in calls))
    mean = statistics.mean(fresh_costs)
    counts = dict(Counter(j["phase"] for j in jobs))
    stage_caps = {"estimator_validation": 3, "profiles_primary": 16, "profiles_stability": 6}
    budget = {"currency": "USD", "status": "PROPOSED_NOT_AUTHORIZED", "paid_spend_this_preparation": 0,
        "pricing_checked_on": "2026-11-13", "pricing_source": "https://platform.claude.com/docs/en/about-claude/pricing",
        "rates_per_million_tokens": {"input_up_to_100k": 0.10, "output_up_to_100k": 0.50,
            "cache_write_5m_up_to_100k": 0.125, "cache_hit_up_to_100k": 0.01,
            "input_over_100k": 0.50, "output_over_100k": 2.50},
        "historical_fresh_forks": len(fresh_costs), "historical_model_calls": sum(fresh_calls),
        "historical_repriced_usd": round(sum(fresh_costs), 6),
        "historical_mean_usd_per_fork": round(mean, 6),
        "historical_p95_usd_per_fork": round(sorted(fresh_costs)[math.ceil(.95*len(fresh_costs))-1],6),
        "phases": [{"phase": phase, "continuations": n, "expected_usd_at_historical_mean": round(n*mean,2),
                    "proposed_phase_ceiling_usd": stage_caps[phase]} for phase,n in counts.items()],
        "planned_continuations": len(jobs), "expected_total_usd": round(len(jobs)*mean,2),
        "planning_range_usd": [round(len(jobs)*mean*.75,2),round(len(jobs)*mean*1.5,2)],
        "proposed_total_ceiling_usd": 25,
        "caveats": ["Historical mean is from selected still-winnable checkpoints, not an unbiased cost forecast for all M2 checkpoints.",
                    "Cache hit rates, completion lengths, provider prices, and retries can change cost; taxes and account-specific charges are excluded.",
                    "Ceilings are proposed authorization limits, not claims that the current vendor budget guard enforces a hard cap.",
                    "Vendor guard checks spending before calls without reserving in-flight cost and can overshoot with concurrency; replace it before paid execution.",
                    "Implement atomic per-call maximum-cost reservations and reconcile all reported usage, retries, and failed calls. Pause if no safe reservation fits."]}
    config = contexts[eligible[0]["checkpoint_id"]]
    manifest = {"schema": "caaf.m2.experiment/1", "experiment_id": "chess_m2_v1", "created_on": "2026-11-13",
        "status": "PREPARED_OFFLINE_PAID_RUNS_BLOCKED", "budget_approval": {"approved": False,"approved_usd": 0},
        "source": {"pr": provenance["pr"], "commit": provenance["head_commit"], "archive_sha256": integrity["archive_sha256"],
                   "historical_source_dirty": True, "note": "Historical run source was dirty; PR code is pinned for new work, original replay validated separately."},
        "population": {"pilot_runs": len(runs), "natural_failed_runs": sum(not r["won"] for r in runs),
            "distinct_failed_tasks": len({r["task_id"] for r in runs if not r["won"]}),
            "profile_decisions": len(eligible), "include_flat_and_oracle_lost_states": True,
            "terminal_values": "Exact from sealed board outcome; no paid continuation from terminal checkpoints."},
        "policy": {"model": config["model"]["name"], "sampling": config["model"]["sampling"],
                   "model_seed": None, "reply_history": "text only as in original protocol", "defender": "original deterministic tablebase policy",
                   "limits": "Restore original counters and original absolute step limit in every arm; consume one step for a forced reply."},
        "estimand": {"unit": "whole assistant reply including explanation and command; not move-only",
            "repair_gain": "P(win | redecide from before) - P(win | force original reply from before)",
            "proposal_delta": "negative repair_gain", "shortcut": "V(before) - V(after)",
            "outcome": "White checkmates within original board and agent limits; do not substitute oracle winnability for actual success."},
        "validation": {"selected_decisions": selected, "per_arm": 50, "schedule": "A/B alternating, cases interleaved",
            "identity_gate": "Offline force-once environment and API-visible context must match original after-checkpoint; counters must match. Compare cost metadata separately.",
            "statistical_gate": "Evaluate after primary profiles supply independent V(before)/V(after) estimates. Form a conservative >=90% interval for direct_gain - shortcut_gain by subtracting their independent 95% interval endpoints (Bonferroni coverage). Shortcut accepted for a checked case only if the whole interval is within +/-0.20; otherwise inconclusive. Three cases do not establish universal equivalence. With 30/50 samples equivalence may remain unresolved; do not change tolerance after seeing results.",
            "fallback": "Do not label unvalidated shortcut as causal effect. Report checkpoint-value changes as descriptive; plan direct-arm profiles as a new version if equivalence is unresolved."},
        "primary": {"per_nonterminal_checkpoint": 30, "independent_of_M1_screening": True,
                    "reuse_M1_outcomes": False, "reuse_direct_validation_outcomes": False},
        "stability": {"runs": stability_runs, "per_nonterminal_checkpoint": 30, "independent_new_samples": True},
        "analysis": {"win_rate_interval": "Wilson 95%", "gain_interval": "Newcombe 95% for independent proportions; retain shared-boundary covariance when comparing whole profiles",
            "multiplicity": "Holm correction across primary decision-level tests; unadjusted intervals explicitly exploratory",
            "no_optional_stopping": True, "extra_sampling": "Not authorized by this manifest; new prespecified confirmation manifest required",
            "failure_classification": "Report top-step concentration only with uncertainty; sparse/diffuse classification exploratory, no forced binary labels.",
            "zero_successes": "Report uncertainty bound and no recovery observed under this policy; never impossible unless independently proven by sealed chess rules."},
        "execution": {"executor_implemented": False, "requires_before_paid": ["Explicit user budget approval", "Implement and test atomic budget reservations including concurrent in-flight calls and retries", "Offline force-once identity checks for all selected cases", "Pin actual executable source and dependencies", "Check account/model access without substituting a model", "Run validation phase first; review outcomes before releasing profile phases"],
            "retry_rule": "Up to two infrastructure retries per planned job, same experimental assignment; preserve all attempts, include costs, never retry based on chess outcome. Halt phase if failures remain.",
            "validation_phase_release": "Review force-once identity, balanced assignments, restored limits, infrastructure errors, and spend after phase 1. Statistical comparison with the shortcut happens only after the primary profiles; causal interpretation remains gated until then.",
            "scope": "No retail jobs, no new starting positions, no model/scaffold changes, no Milestone 4 training in this preparation."},
        "milestone4": {"split_unit": "whole starting-position task; two trials and all descendants stay together", "features": "Export only pre-decision features in a separate table; oracle diagnostics, future moves, outcomes, and gain labels excluded from predictor inputs.",
                       "coverage_note": "Seven failures cover only four positions. Expand with the remaining planned tasks before evaluating generalization."},
        "budget_file": "budget.json", "jobs_file": "planned_jobs.csv", "paid_continuations": len(jobs)}
    assert len(roots)==10 and len(inventory)==72 and baseline_calls==62 and len(eligible)==48
    assert len(selected)==3 and len(stability_runs)==2 and len(jobs)==2250
    assert counts=={"estimator_validation":300,"profiles_primary":1440,"profiles_stability":510}
    assert before_db == digest(original_db) and before_ledger == digest(ROOT / "inputs/m1_original/runs/api_usage.jsonl")
    integrity.update({"sqlite_integrity": "ok", "original_database_unchanged": True, "original_ledger_unchanged": True,
                      "source_files_verified": len(provenance["verified_files"]), "no_model_modules_imported": not any(n.startswith(("litellm","minisweagent","anthropic")) for n in sys.modules)})
    assert integrity["no_model_modules_imported"]
    write_csv("checkpoint_inventory.csv", inventory);write_csv("decision_inventory.csv", decisions)
    write_csv("run_inventory.csv", runs);write_csv("planned_jobs.csv", jobs)
    write_json("integrity.json", integrity);write_json("budget.json", budget);write_json("experiment_manifest.json", manifest)
    summary={"pilot_runs":len(roots),"checkpoints":len(inventory),"decision_boundaries":len(decisions),"failed_runs":7,"failure_profile_boundaries":len(eligible),"planned_jobs":len(jobs),"historical_forks":len(fresh_costs),"expected_usd":budget["expected_total_usd"],"selected_cases":[(d['decision_id'],d['selection_role']) for d in selected]}
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
