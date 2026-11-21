"""Verify all interrupted contexts offline without changing sampled prefixes."""
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/pr917')]
os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'
os.environ['MSWEA_GLOBAL_CONFIG_DIR']=str(ROOT/'.offline_settings')
def deny(*a,**kw):raise RuntimeError('Offline resume preflight prohibits network/model calls')
socket.socket.connect=deny;socket.create_connection=deny
import caaf_agent as agent
import caaf_chess as board
import caaf_store as base
from m2_runtime.compact_store import CompactStore

agent.CaafModel.query=deny;board._TB_DIR=ROOT/'inputs/tablebases'
OUT=ROOT/'experiments/m2_direct_v2';WORK=ROOT/'paid_runs/m2_direct_v2'
with sqlite3.connect(WORK/'jobs.sqlite') as db:
    interrupted=[json.loads(r[0]) for r in db.execute("select result_json from jobs where status!='completed' and result_json is not null") if json.loads(r[0]).get('branch_id')]
checks=[]
with tempfile.TemporaryDirectory(prefix='m2-credit-resume-') as td:
    copied=Path(td)/'store';shutil.copytree(WORK/'store',copied);(copied/'manifest.sqlite').chmod(0o600)
    st=CompactStore(copied)
    config=agent.load_config('chess_haiku_think.yaml');config['model'].pop('budget',None)
    for row in interrupted:
        old=row['branch_id'];snaps=st.snapshots(old)
        checkpoint=snaps[-1] if snaps else st.snapshot(st.branch(old)['parent_snapshot_id'])
        context=json.loads(st.get_blob('context',checkpoint['context_hash']))
        assert context['messages'][-1]['role']!='exit'
        assert context['counters']['n_calls']<context['counters']['step_limit']
        branch=agent.fork(st,checkpoint['id'],config=config,intervention={'type':'free'})
        assert base.sha256(base.canonical(branch.agent.context()))==checkpoint['context_hash']
        assert base.sha256(branch.env.state())==checkpoint['env_hash']
        paid=[e for e in st.read_tape(old) if e['kind']=='model_call' and e['response']['source']=='model']
        checks.append({'job_id':row['job_id'],'existing_branch':old,'resume_checkpoint':checkpoint['id'],
                       'context_sha256':checkpoint['context_hash'],'env_sha256':checkpoint['env_hash'],
                       'saved_paid_model_calls':len(paid),'full_restore_verified':True})
    st.db.close()
(OUT/'credit_resume_preflight.json').write_text(json.dumps({'passed':len(checks),'paid_calls':0,'network_blocked':True,
    'rule':'Continue existing prefixes; never restart a sampled trajectory solely because it was long-running when credits ran out.',
    'jobs':checks},indent=2)+'\n')
print(json.dumps({'resume_contexts_verified':len(checks),'saved_paid_calls_preserved':sum(r['saved_paid_model_calls'] for r in checks),'new_paid_calls':0}))
