"""Verify proposed keep-arm restoration offline, in a disposable store copy."""
import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/pr917"))
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = str(ROOT / ".offline_settings")


def deny_network(*args, **kwargs):
    raise RuntimeError("Network prohibited during offline verification")


socket.socket.connect = deny_network
socket.create_connection = deny_network

import caaf_agent as agent  # noqa: E402
import caaf_chess as board  # noqa: E402
import caaf_store as store  # noqa: E402
import caaf_utils as utils  # noqa: E402

board._TB_DIR = ROOT / "inputs/tablebases"
agent.CaafModel.query = deny_network

original = ROOT / "inputs/m1_original/runs/m1_think"
manifest = json.loads((ROOT / "experiments/m2_v1/experiment_manifest.json").read_text())
reports = []
with tempfile.TemporaryDirectory(prefix="chess-m2-offline-") as scratch:
    copied = Path(scratch) / "store"
    shutil.copytree(original, copied)
    (copied / "manifest.sqlite").chmod(0o600)
    st = store.Store(copied)
    config = agent.load_config("chess_haiku_think.yaml")
    # No ledger object is created; forced replies cannot call a hosted model.
    config["model"].pop("budget", None)
    for case in manifest["validation"]["selected_decisions"]:
        snap = st.snapshot(case["before_checkpoint"])
        calls = utils.model_calls(st, case["branch_id"])
        reply = calls[case["step"]]["response"]["content"]
        assert hashlib.sha256(reply.encode()).hexdigest() == case["original_reply_sha256"]
        branch = agent.fork(st, snap["id"], config=config, intervention={
            "type": "force", "step": case["step"], "content": reply,
        })
        agent.run_to_end(branch, stop_after_calls=case["step"])
        after = st.snapshot(branch.agent.last_snapshot_id)
        expected = st.snapshot(case["after_checkpoint"])
        ctx_actual = json.loads(st.get_blob("context", after["context_hash"]))
        ctx_expected = json.loads(st.get_blob("context", expected["context_hash"]))
        counters = ["n_calls", "step_limit", "n_consecutive_format_errors"]
        checks = {
            "environment": after["env_hash"] == expected["env_hash"],
            "api_visible_conversation": utils.api_view_hash(st, after) == utils.api_view_hash(st, expected),
            "termination_and_step_counters": all(ctx_actual["counters"][k] == ctx_expected["counters"][k] for k in counters),
        }
        assert all(checks.values()), (case["decision_id"], checks)
        reports.append({"decision_id": case["decision_id"], "checks": checks,
                        "cost_counter_note": "Forced reply has zero new API cost; cost metadata intentionally differs."})
out = ROOT / "experiments/m2_v1/force_once_preflight.json"
out.write_text(json.dumps({"network_blocked": True, "live_model_query_blocked": True,
                           "source_store_modified": False, "cases": reports}, indent=2) + "\n")
print(json.dumps({"passed_cases": len(reports), "paid_calls": 0}))
