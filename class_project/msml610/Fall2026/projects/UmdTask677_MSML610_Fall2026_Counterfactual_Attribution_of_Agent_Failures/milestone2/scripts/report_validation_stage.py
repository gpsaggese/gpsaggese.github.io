"""Audit recorded validation branches and report direct gains without API calls."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "vendor/pr917")]
import caaf_store as store  # noqa: E402
import caaf_chess as board  # noqa: E402
from m2_runtime.budget import Budget, NANO, price_usage  # noqa: E402

board._TB_DIR = ROOT / "inputs/tablebases"

STAGE = ROOT / "experiments/m2_v1/validation_stage_20261118"
WORK = ROOT / "paid_runs/validation_stage_20261118"


def wilson(k, n):
    p, z = k/n, 1.96
    denom = 1+z*z/n
    center = (p+z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return max(0,center-half), min(1,center+half)


def difference(a, na, b, nb):
    pa, pb = a/na,b/nb
    la,ua = wilson(a,na);lb,ub = wilson(b,nb)
    return pa-pb-math.sqrt((pa-la)**2+(ub-pb)**2), pa-pb+math.sqrt((ua-pa)**2+(pb-lb)**2)


def fisher(a, na, b, nb):
    total = a+b
    def p(k):
        return math.comb(na,k)*math.comb(nb,total-k)/math.comb(na+nb,total)
    observed = p(a)
    return min(1,sum(p(k) for k in range(max(0,total-nb),min(total,na)+1) if p(k)<=observed*(1+1e-9)))


def main():
    completion = json.loads((STAGE / "completion.json").read_text())
    results = json.loads((STAGE / "results.json").read_text())
    manifest = json.loads((ROOT / "experiments/m2_v1/experiment_manifest.json").read_text())
    execution = json.loads((STAGE / "execution_manifest.json").read_text())
    cases = {c["decision_id"]:c for c in manifest["validation"]["selected_decisions"]}
    dbpath = WORK / "store/manifest.sqlite"
    db = sqlite3.connect(f"file:{dbpath}?mode=ro",uri=True);db.row_factory=sqlite3.Row
    billed_calls = 0
    raw_cost = 0
    for result in results:
        if result["status"] != "completed":
            continue
        branch = db.execute("select * from branches where branch_id=?",(result["branch_id"],)).fetchone()
        case = cases[result["decision_id"]]
        assert branch["parent_snapshot_id"] == case["before_checkpoint"]
        assert json.loads(branch["outcome_json"]) == result["outcome"]
        tape = [json.loads(line) for line in (WORK / "store/tapes" / f"{result['branch_id']}.jsonl").read_text().splitlines()]
        assert store.verify_tape(tape)
        calls = [e for e in tape if e["kind"] == "model_call"]
        first = calls[0]
        assert first["step"] == case["step"] and first["request"]["model"] == manifest["policy"]["model"]
        assert first["request"]["sampling"] == manifest["policy"]["sampling"]
        initial = db.execute("select * from snapshots where id=?",(case["before_checkpoint"],)).fetchone()
        initial_ctx = json.loads((WORK/"store/blobs/context"/f"{initial['context_hash']}.json").read_text())
        initial_api_view = [{k:v for k,v in m.items() if k!='extra'} for m in initial_ctx['messages']]
        assert first['request']['input_hash'] == store.sha256(store.canonical(initial_api_view))
        if result["arm"] == "keep_original_reply":
            assert first["response"]["source"] == "forced"
            assert hashlib.sha256(first["response"]["content"].encode()).hexdigest() == case["original_reply_sha256"]
        else:
            assert first["response"]["source"] == "model"
        for e in calls:
            assert e["request"]["model"] == manifest["policy"]["model"]
            assert e["request"]["sampling"] == manifest["policy"]["sampling"]
            if e["response"].get("source") == "model":
                billed_calls += 1
                raw_cost += price_usage(e["response"]["usage"])
        snapshots = db.execute("select * from snapshots where branch_id=?",(result["branch_id"],)).fetchall()
        for snap in snapshots:
            for kind,key in [("context",snap["context_hash"]),("env",snap["env_hash"])]:
                p=WORK / "store/blobs" / kind / f"{key}.json"
                assert hashlib.sha256(p.read_bytes()).hexdigest()==key
        final = db.execute("select * from snapshots where id=?",(result["outcome"]["last_snapshot"],)).fetchone()
        final_ctx = json.loads((WORK/"store/blobs/context"/f"{final['context_hash']}.json").read_text())
        assert final_ctx['counters']['step_limit'] == initial_ctx['counters']['step_limit']
        assert final_ctx['counters']['n_calls'] <= initial_ctx['counters']['step_limit']
        game = json.loads((WORK / "store/blobs/env" / f"{final['env_hash']}.json").read_text())
        # Reconstruct legal moves and independently re-run the referee from
        # the saved game, rather than trusting the reported branch outcome.
        rebuilt = board.board_of(game)
        rescored = board.referee(game["start_fen"],game["moves_uci"])
        assert rescored["won"] == result["outcome"]["won"]
        assert board._status_of(rebuilt)[0] == game["status"] == result["outcome"]["status"]
    budget = Budget(WORK / "budget.sqlite",3*NANO).summary()
    actual = sum(r["amount"] for r in budget["by_status"] if r["status"] == "settled")
    assert budget["within_cap"]
    if completion["all_300_completed"]:
        assert len(results)==300 and budget["requests"]==billed_calls
        assert raw_cost==actual
        assert all(r["status"] == "settled" for r in budget["by_status"])
    rows = []
    for case in cases.values():
        subset=[r for r in results if r["decision_id"]==case["decision_id"] and r["status"]=="completed"]
        groups={g:[r for r in subset if r["arm"]==g] for g in ["keep_original_reply","redecide_reply"]}
        keep,redo=groups["keep_original_reply"],groups["redecide_reply"]
        a,b=sum(r["outcome"]["won"] for r in redo),sum(r["outcome"]["won"] for r in keep)
        row={"decision_id":case["decision_id"],"selection_role":case["selection_role"],
             "keep_wins":b,"keep_n":len(keep),"redecide_wins":a,"redecide_n":len(redo)}
        if keep and redo:
            row.update({"repair_gain":a/len(redo)-b/len(keep),"repair_gain_ci95":difference(a,len(redo),b,len(keep)),
                        "keep_ci95":wilson(b,len(keep)),"redecide_ci95":wilson(a,len(redo)),
                        "fisher_p":fisher(a,len(redo),b,len(keep))})
        row["outcomes"]={g:dict(Counter(r["outcome"]["status"] for r in rs)) for g,rs in groups.items()}
        rows.append(row)
    # Family-wise correction across the three validation cases (two-sided).
    ordered=sorted((row for row in rows if "fisher_p" in row),key=lambda x:x["fisher_p"])
    running=0
    for rank,row in enumerate(ordered):
        running=max(running,min(1,row["fisher_p"]*(len(ordered)-rank)))
        row["holm_adjusted_p"]=running
    report={"all_300_completed":completion["all_300_completed"],"results":rows,
            "paid_model_calls":billed_calls,"observed_usage_cost_usd":actual/NANO,
            "accounted_liability_usd":budget["accounted_plus_reserved_usd"],"approved_usd":3,
            "audited_completed_branches":sum(r["status"]=="completed" for r in results),
            "checks":"branch parents, initial API-visible input hashes, retained absolute step limits, forced reply hashes, every model call's policy, tape chains, snapshot hashes, legal move reconstruction, independent referee outcomes and raw usage/ledger reconciliation",
            "source_preserved":completion["source_database_unchanged"] and completion["source_ledger_unchanged"],
            "elapsed_seconds":completion["elapsed_seconds"],"later_phases_started":False,
            "milestone2_complete":False,"shortcut_estimator_validated":False,
            "limitations":["Selected cases from only three positions; not population-level attribution.",
                           "Effects refer to re-deciding a whole reply, including reasoning text, under the same model policy.",
                           "Independent V(before)/V(after) profiles and stability repetitions have not been run.",
                           "Reported 95% effect intervals are unadjusted; Holm p-values correct across these three cases."],
            "analysis_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "execution_manifest_sha256":hashlib.sha256((STAGE/'execution_manifest.json').read_bytes()).hexdigest()}
    report["interpretation"] = [
        "For krk-m4-1 trial 0 decision 1, keeping the original complete reply wins 66% versus 34% when re-deciding: repair gain -32 percentage points, unadjusted 95% interval [-48.4,-12.4], Holm-adjusted p approximately 0.00757. This is evidence that the original reply was helpful under this tested continuation policy.",
        "For the selected krk-m3-1 and krk-m8-1 decisions, both arms won 0/50. No recovery was observed. Each sampled arm's Wilson 95% upper win-rate bound is approximately 7.1%; these results do not prove recovery is impossible or that the task is underspecified.",
        "A theoretically losing chess move can have near-zero measured repair gain when the tested agent already has little chance of recovery immediately before it. Earlier checkpoints still need profiling; do not assign blame from the tablebase pivot alone.",
        "The direct-arm infrastructure and first selected contrasts are demonstrated. The checkpoint-value shortcut and full Milestone 2 are not yet validated/completed.",
    ]
    (STAGE/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    lines=['# Chess Milestone 2 — first validation stage','',
           f"Completed branches: {report['audited_completed_branches']}/300. Paid calls: {billed_calls}. "
           f"Usage-rated spending: ${actual/NANO:.6f}; total accounted liability: ${budget['accounted_plus_reserved_usd']:.6f} / $3 approved.",'',
           '| Original decision | Keep wins | Re-decide wins | Repair gain | 95% interval | Holm p |',
           '| --- | ---: | ---: | ---: | --- | ---: |']
    for row in rows:
        if 'repair_gain' in row:
            lo,hi=row['repair_gain_ci95']
            lines.append(f"| {row['decision_id']} | {row['keep_wins']}/{row['keep_n']} | {row['redecide_wins']}/{row['redecide_n']} | {row['repair_gain']:+.2f} | [{lo:+.3f}, {hi:+.3f}] | {row['holm_adjusted_p']:.4g} |")
    lines += ['', 'The original data is unchanged. Raw branch/checkpoint/tape evidence and per-request accounting are retained in the local paid-run store.', '',
              '## Interpretation', '', *[f'- {s}' for s in report['interpretation']], '',
              'This completes only the first validation stage if all 300 branches finished. Milestone 2 and statistical validation of the shortcut remain incomplete. Primary profiles and stability repeats require subsequent release; they were not started.', '',
              'Effect intervals are exploratory unadjusted 95% intervals. Holm p-values cover the three selected validation cases. Selection and the whole-reply intervention limit generalization.']
    (STAGE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
