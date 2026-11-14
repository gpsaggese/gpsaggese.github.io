#!/usr/bin/env python
"""
Milestone 1 on the chess task: checkpoint and resume, replay checks, and the
A/A noise floor.

Subcommands, in order:
  tasks    the 20 tasks and 4 reserves (seeded positions) -> data/tasks.json
  pilot    Stage A: 5 tasks x 2 fresh runs, a checkpoint after every step
  replay   restore every checkpoint and replay the recorded steps (no API)
  screen   V at candidate checkpoints, to find ones with mixed outcomes
  aa       A/A at the 2 most mixed checkpoints, 50 vs 50, interleaved
  report   collect everything into results/m1/summary.json

Usage:

> python caaf_m1.py tasks
> set -a; source ~/.cf-m1.env; set +a   # ANTHROPIC_API_KEY
> python caaf_m1.py pilot
> ...

Import as:

import caaf_m1 as cm1
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import caaf_agent as cagent
import caaf_chess as cchess
import caaf_store as cstore
import caaf_utils as cutils

_ROOT = Path(__file__).resolve().parent
# `CAAF_RUN` names the experiment (default `m1`); a dry run uses another name.
_RUN = os.environ.get("CAAF_RUN", "m1")
_STORE = _ROOT / "runs" / _RUN
_RESULTS = _ROOT / "results" / _RUN
_TASKS = _ROOT / "data" / "tasks.json"

# Distance to mate in plies (mate in 2, 3, 4, 6 and 8 moves), 4 tasks each.
_DTZ_BINS = [3, 5, 7, 11, 15]
_SLACK_PLIES = 8


def _ledger(config: Dict) -> cstore.Ledger:
    b = config["model"]["budget"]
    return cstore.Ledger(_ROOT / b["ledger"], b["cap_usd"], b["prices"])


def _tasks() -> Dict:
    return json.loads(_TASKS.read_text())


def _cost_report(config: Dict, before: float) -> Dict:
    spent = _ledger(config).spent()
    return {"phase_usd": round(spent - before, 4), "total_usd": round(spent, 4)}


# #############################################################################
# Tasks
# #############################################################################


def cmd_tasks(args: argparse.Namespace) -> None:
    tasks, reserves = [], []
    for i, dtz in enumerate(_DTZ_BINS):
        pos = cchess.start_positions(
            5, seed=677 + i, dtz=dtz, slack_plies=_SLACK_PLIES
        )
        for j, p in enumerate(pos):
            row = {"id": f"krk-m{(dtz + 1) // 2}-{j + 1}", **p}
            (tasks if j < 4 else reserves).append(row)
    # Stage A pilot: the first task of every bin.
    pilot = [t["id"] for t in tasks if t["id"].endswith("-1")]
    obj = {
        "material": "KRvK",
        "slack_plies": _SLACK_PLIES,
        "rule": "seeded random positions, start_positions(seed=677+bin)",
        "tasks": tasks,
        "reserves": reserves[:4],
        "pilot": pilot,
    }
    cutils.write_json(_TASKS, obj)
    print(f"{len(tasks)} tasks, {len(obj['reserves'])} reserves, pilot {pilot}")


# #############################################################################
# Stage A: pilot
# #############################################################################


def cmd_pilot(args: argparse.Namespace) -> None:
    config = cagent.load_config()
    before = _ledger(config).spent()
    tasks = {t["id"]: t for t in _tasks()["tasks"]}
    ids = _tasks()["pilot"][: args.tasks]
    jobs = [(tid, r) for r in range(args.runs) for tid in ids]

    def one(job):
        tid, r = job
        try:
            out = cutils.play_new_game(
                _STORE, tasks[tid]["fen"], config, f"pilot:{tid}:{r}"
            )
            return {"task": tid, "run": r, "infra_error": None, **out}
        except Exception as e:
            return {
                "task": tid,
                "run": r,
                "infra_error": f"{type(e).__name__}: {e}",
            }

    res = cutils.parallel(one, jobs, args.workers)
    ok = [r for r in res if not r["infra_error"]]
    summary = {
        "runs": len(res),
        "won": sum(r["won"] for r in ok),
        "not_won": sum(not r["won"] for r in ok),
        "infra_errors": len(res) - len(ok),
        "by_task": {
            tid: [
                r.get("status", "infra_error") for r in res if r["task"] == tid
            ]
            for tid in ids
        },
        "cost": _cost_report(config, before),
    }
    cutils.write_json(_RESULTS / "pilot.json", {"summary": summary, "runs": res})
    print(json.dumps(summary, indent=1))


# #############################################################################
# Replay checks (no API calls)
# #############################################################################


def cmd_replay(args: argparse.Namespace) -> None:
    """
    For every checkpoint of every recorded run:
    - one-step: restore checkpoint k-1, feed step k's recorded reply, compare
      the model input, the board output, the environment and the conversation
      after the step with the original
    - to-end: restore it and replay every later reply; the game must end
      exactly as the original did
    """
    config = cagent.load_config()
    st = cstore.Store(_STORE)
    roots = cutils.root_branches(st)
    jobs = []
    for b in roots:
        calls = cutils.model_calls(st, b["branch_id"])
        last = max(calls)
        # The checkpoint after a step limit has no next step.
        for s in st.snapshots(b["branch_id"]):
            if s["step"] + 1 in calls and s["seq"] == s["step"]:
                jobs.append((b["branch_id"], s, last))
    print(
        f"{len(roots)} recorded runs, {len(jobs)} checkpoints with a next step"
    )

    def one(job):
        bid, s, last = job
        st2 = cstore.Store(_STORE)
        k = s["step"] + 1
        iv = {"type": "replay", "source_branch": bid, "steps": [k]}
        r = cutils.play_fork(_STORE, s["id"], iv, config, stop_after_calls=k)
        orig = [e for e in st2.read_tape(bid) if e["step"] == k]
        new = [e for e in st2.read_tape(r["branch_id"]) if e["step"] == k]
        o_call = next(e for e in orig if e["kind"] == "model_call")
        n_call = next(e for e in new if e["kind"] == "model_call")
        o_cmd = [e["response"] for e in orig if e["kind"] == "command"]
        n_cmd = [e["response"] for e in new if e["kind"] == "command"]
        after_o = st2.snapshots(bid)[k]
        after_n = st2.snapshots(r["branch_id"])[0]
        iv_all = {
            "type": "replay",
            "source_branch": bid,
            "steps": list(range(k, last + 1)),
        }
        full = cutils.play_fork(_STORE, s["id"], iv_all, config)
        orig_out = json.loads(st2.branch(bid)["outcome_json"])
        end_o = st2.snapshots(bid)[-1]
        end_n = st2.snapshot(full["last_snapshot"])
        return {
            "source_branch": bid,
            "checkpoint": s["id"],
            "step": k,
            "input_hash_match": o_call["request"]["input_hash"]
            == n_call["request"]["input_hash"],
            "command_output_match": o_cmd == n_cmd,
            "env_after_match": after_o["env_hash"] == after_n["env_hash"],
            "conversation_after_match": cutils.api_view_hash(st2, after_o)
            == cutils.api_view_hash(st2, after_n),
            "to_end_same_moves": full["moves_san"] == orig_out["moves_san"],
            "to_end_same_outcome": full["won"] == orig_out["won"],
            "to_end_same_final_env": end_o["env_hash"] == end_n["env_hash"],
            "tape_chain_ok": cstore.verify_tape(st2.read_tape(r["branch_id"])),
        }

    res = cutils.parallel(one, jobs, args.workers)
    keys = [k for k in res[0] if k.endswith(("_match", "_ok")) or "to_end" in k]
    summary = {k: f"{sum(r[k] for r in res)}/{len(res)}" for k in keys}
    summary["all_checks_pass"] = (
        f"{sum(all(r[k] for k in keys) for r in res)}/{len(res)}"
    )
    summary["runs"] = len(roots)
    summary["source_tapes_chain_ok"] = sum(
        cstore.verify_tape(st.read_tape(b["branch_id"])) for b in roots
    )
    mism = [r for r in res if not all(r[k] for k in keys)]
    cutils.write_json(
        _RESULTS / "replay.json", {"summary": summary, "mismatches": mism}
    )
    print(json.dumps(summary, indent=1))


# #############################################################################
# Screening for mixed checkpoints
# #############################################################################


def _candidates(st: cstore.Store, per_run: int) -> List[Dict]:
    """
    Checkpoints of the recorded runs where the game is on and still won for
    White (by the tablebase), evenly spaced, up to `per_run` per run.
    """
    out = []
    for b in cutils.root_branches(st):
        task = json.loads(b["config_json"])["label"].split(":")[1]
        won_cps = []
        for s in st.snapshots(b["branch_id"])[:-1]:
            game = json.loads(st.get_blob("env", s["env_hash"]))
            if game["status"] != "ongoing":
                continue
            v = cchess.white_value(cchess.board_of(game))
            if v["win"]:
                won_cps.append((s, v))
        if not won_cps:
            continue
        idx = sorted(
            {
                round(i * (len(won_cps) - 1) / max(1, per_run - 1))
                for i in range(per_run)
            }
        )
        for i in idx:
            s, v = won_cps[i]
            out.append(
                {
                    "checkpoint": s["id"],
                    "branch_id": b["branch_id"],
                    "task": task,
                    "step": s["step"],
                    "dtz": v["dtz"],
                    "spare_plies": 100 - v["clock"] - v["dtz"],
                }
            )
    return out


def cmd_screen(args: argparse.Namespace) -> None:
    config = cagent.load_config()
    before = _ledger(config).spent()
    st = cstore.Store(_STORE)
    cands = _candidates(st, args.per_run)
    jobs = [(c, r) for r in range(args.n) for c in cands]
    print(f"{len(cands)} candidate checkpoints x {args.n} = {len(jobs)} forks")

    def one(job):
        c, r = job
        try:
            out = cutils.play_fork(
                _STORE, c["checkpoint"], {"type": "free"}, config
            )
            return {
                "checkpoint": c["checkpoint"],
                "won": out["won"],
                "error": None,
            }
        except Exception as e:
            return {
                "checkpoint": c["checkpoint"],
                "won": None,
                "error": str(e)[:300],
            }

    res = cutils.parallel(one, jobs, args.workers)
    rows = []
    for c in cands:
        rs = [
            r
            for r in res
            if r["checkpoint"] == c["checkpoint"] and r["error"] is None
        ]
        k, n = sum(r["won"] for r in rs), len(rs)
        lo, hi = cutils.wilson(k, n)
        rows.append(
            {
                **c,
                "wins": k,
                "n": n,
                "v": k / n if n else None,
                "ci95": [round(lo, 3), round(hi, 3)],
            }
        )
    summary = {
        "candidates": len(rows),
        "n_per_checkpoint": args.n,
        "errors": sum(r["error"] is not None for r in res),
        "mixed_0.3_0.7": sum(
            r["v"] is not None and 0.3 <= r["v"] <= 0.7 for r in rows
        ),
        "always_won": sum(r["v"] == 1.0 for r in rows),
        "never_won": sum(r["v"] == 0.0 for r in rows),
        "cost": _cost_report(config, before),
    }
    cutils.write_json(
        _RESULTS / "screen.json", {"summary": summary, "rows": rows}
    )
    print(json.dumps(summary, indent=1))
    for r in sorted(rows, key=lambda r: abs((r["v"] or 0) - 0.5)):
        print(
            r["task"],
            r["step"],
            r["dtz"],
            r["spare_plies"],
            f"{r['wins']}/{r['n']}",
        )


# #############################################################################
# A/A noise floor
# #############################################################################


def _pick_mixed(rows: List[Dict], k: int) -> List[Dict]:
    """
    The `k` checkpoints with V closest to 0.5, from different tasks if possible.
    """
    mixed = sorted(
        (r for r in rows if r["v"] is not None and 0.3 <= r["v"] <= 0.7),
        key=lambda r: abs(r["v"] - 0.5),
    )
    out, tasks = [], set()
    for r in mixed:
        if r["task"] not in tasks:
            out.append(r)
            tasks.add(r["task"])
    for r in mixed:
        if len(out) >= k:
            break
        if r not in out:
            out.append(r)
    return out[:k]


def cmd_aa(args: argparse.Namespace) -> None:
    config = cagent.load_config()
    before = _ledger(config).spent()
    rows = json.loads((_RESULTS / "screen.json").read_text())["rows"]
    picked = _pick_mixed(rows, args.checkpoints)
    if len(picked) < args.checkpoints:
        raise SystemExit(
            f"only {len(picked)} mixed checkpoints; screen more first"
        )
    # Interleaved: A and B alternate in the job queue, so both groups see the
    # same API conditions over time.
    jobs = [(c, g, r) for r in range(args.n) for c in picked for g in ("A", "B")]
    print(f"{len(picked)} checkpoints x 2 groups x {args.n} = {len(jobs)} forks")

    def one(job):
        c, g, r = job
        try:
            out = cutils.play_fork(
                _STORE, c["checkpoint"], {"type": "free"}, config
            )
            return {
                "checkpoint": c["checkpoint"],
                "group": g,
                "rep": r,
                "won": out["won"],
                "status": out["status"],
                "n_calls": out["n_calls"],
                "error": None,
            }
        except Exception as e:
            return {
                "checkpoint": c["checkpoint"],
                "group": g,
                "rep": r,
                "won": None,
                "error": str(e)[:300],
            }

    res = cutils.parallel(one, jobs, args.workers)
    out_rows = []
    for c in picked:
        grp = {
            g: [
                r
                for r in res
                if r["checkpoint"] == c["checkpoint"]
                and r["group"] == g
                and r["error"] is None
            ]
            for g in ("A", "B")
        }
        a, na = sum(r["won"] for r in grp["A"]), len(grp["A"])
        b, nb = sum(r["won"] for r in grp["B"]), len(grp["B"])
        lo, hi = cutils.newcombe_diff(a, na, b, nb)
        out_rows.append(
            {
                "checkpoint": c["checkpoint"],
                "task": c["task"],
                "step": c["step"],
                "screen_v": c["v"],
                "a_wins": a,
                "n_a": na,
                "b_wins": b,
                "n_b": nb,
                "v_a": a / na,
                "v_b": b / nb,
                "gap": a / na - b / nb,
                "ci95": [round(lo, 3), round(hi, 3)],
                "half_width": round((hi - lo) / 2, 3),
                "fisher_p": round(cutils.fisher_two_sided(a, na, b, nb), 4),
                "pooled_v": (a + b) / (na + nb),
            }
        )
    summary = {
        "checkpoints": len(out_rows),
        "n_per_group": args.n,
        "errors": sum(r["error"] is not None for r in res),
        "max_abs_gap": max(abs(r["gap"]) for r in out_rows),
        "ci_contains_zero": sum(
            r["ci95"][0] <= 0 <= r["ci95"][1] for r in out_rows
        ),
        "false_alarms_p_lt_0.05": sum(r["fisher_p"] < 0.05 for r in out_rows),
        "cost": _cost_report(config, before),
    }
    cutils.write_json(
        _RESULTS / "aa.json",
        {"summary": summary, "rows": out_rows, "results": res},
    )
    print(json.dumps({"summary": summary, "rows": out_rows}, indent=1))


# #############################################################################
# Report
# #############################################################################


def cmd_report(args: argparse.Namespace) -> None:
    config = cagent.load_config()
    st = cstore.Store(_STORE)
    led = _ledger(config)
    rows = [
        json.loads(line) for line in led.path.read_text().splitlines() if line
    ]
    n_branches = st.query("select count(*) n from branches")[0]["n"]
    n_snaps = st.query("select count(*) n from snapshots")[0]["n"]
    n_failed = st.query("select count(*) n from branches where status='failed'")[
        0
    ]["n"]
    out = {
        "store": {
            "branches": n_branches,
            "checkpoints": n_snaps,
            "failed_branches": n_failed,
        },
        "api": {
            "calls": len(rows),
            "usd": round(sum(r["cost_usd"] for r in rows), 4),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "cache_read": sum(r["cache_read"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "cap_usd": led.cap,
        },
    }
    for name in ("pilot", "replay", "screen", "aa"):
        p = _RESULTS / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())["summary"]
    cutils.write_json(_RESULTS / "summary.json", out)
    print(json.dumps(out, indent=1))


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tasks")
    p = sub.add_parser("pilot")
    p.add_argument("--tasks", type=int, default=5)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("replay")
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("screen")
    p.add_argument("--per-run", type=int, default=4)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("aa")
    p.add_argument("--checkpoints", type=int, default=2)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--workers", type=int, default=8)
    sub.add_parser("report")
    args = ap.parse_args()
    globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    _main()
