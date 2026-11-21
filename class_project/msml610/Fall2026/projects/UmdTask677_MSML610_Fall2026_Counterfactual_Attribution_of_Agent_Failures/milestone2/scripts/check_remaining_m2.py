"""Read-only live status; never imports a model or starts work."""
import json
import shutil
import sqlite3
from pathlib import Path

root=Path(__file__).resolve().parents[1];out=root/'experiments/m2_direct_v2';work=root/'paid_runs/m2_direct_v2'
with sqlite3.connect(f"file:{work/'jobs.sqlite'}?mode=ro",uri=True) as db:
    counts=dict(db.execute('select status,count(*) from jobs group by status'))
    phases={phase:dict(db.execute('select status,count(*) from jobs where phase=? group by status',(phase,))) for phase in ['profiles_primary','profiles_stability']}
with sqlite3.connect(f"file:{work/'budget.sqlite'}?mode=ro",uri=True) as db:
    settled=db.execute("select coalesce(sum(liability),0) from calls where status='settled'").fetchone()[0]
    liability=db.execute('select coalesce(sum(liability),0) from calls').fetchone()[0]
assert liability<=22_000_000_000
events=out/'rate_limit_events.json'
completion=json.loads((out/'completion.json').read_text()) if (out/'completion.json').exists() else {}
print(json.dumps({'counts':counts,'phases':phases,'settled_additional_usd':settled/1e9,
                  'cap_usd':22,'within_cap':True,'free_disk_mib':round(shutil.disk_usage(root).free/1024**2),
                  'capacity_retry_events':len(json.loads(events.read_text())) if events.exists() else 0,
                  'status':completion.get('status','RUNNING'),
                  'finished':completion.get('all_3900_completed',False)}))
