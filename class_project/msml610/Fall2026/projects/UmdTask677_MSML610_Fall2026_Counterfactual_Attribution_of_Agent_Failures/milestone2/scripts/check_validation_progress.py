"""Read-only status of the approved stage; no model imports or calls."""
import json
import sqlite3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
stage = root / "experiments/m2_v1/validation_stage_20261118"
progress = json.loads((stage / "progress.json").read_text())
path = root / "paid_runs/validation_stage_20261118/budget.sqlite"
with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
    settled = db.execute("select coalesce(sum(liability),0) from calls where status='settled'").fetchone()[0]
    liability = db.execute("select coalesce(sum(liability),0) from calls").fetchone()[0]
assert liability <= 3_000_000_000
print(json.dumps({"completed": progress["completed"], "total": 300,
                  "errors": progress["infrastructure_errors"], "settled_usd": settled/1e9,
                  "elapsed_minutes": round(progress["elapsed_seconds"]/60,1),
                  "finished": (stage/"completion.json").exists()}))
