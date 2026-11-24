"""Execute the approved direct-arm fallback with one shared additional $22 cap."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/pr917')]
os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'
os.environ['MSWEA_GLOBAL_CONFIG_DIR']=str(ROOT/'.offline_settings')
import run_validation_stage as transport
from m2_runtime.budget import Budget,NANO
from m2_runtime.compact_store import CompactStore,seed_pilot_store

OUT=ROOT/'experiments/m2_direct_v2'
WORK=ROOT/'paid_runs/m2_direct_v2'
STORE=WORK/'store'
STOPPED=threading.Event()


def save(name,data):
    p=OUT/name;temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n');temp.replace(p)


class RateGate:
    """Share capacity backoff across workers without changing agent policy."""
    def __init__(self):
        self.lock=threading.Lock();self.interval=.25;self.next_start=0;self.cooldown=0
        self.last_adjust=0;self.retries=Counter();self.rate_events=[]

    def completion(self,**kwargs):
        job=transport.CURRENT_JOB.get()
        while True:
            while True:
                if STOPPED.is_set() or (OUT/'STOP').exists():
                    STOPPED.set();raise RuntimeError('Stage stopped before wire request')
                with self.lock:
                    now=time.monotonic();wait=max(self.next_start,self.cooldown)-now
                    if wait<=0:self.next_start=now+self.interval;break
                STOPPED.wait(min(wait,.5))
            try:
                return transport.litellm.completion(**kwargs)
            except BaseException as exc:
                if getattr(exc,'status_code',None)!=429:raise
                with self.lock:
                    self.retries[job]+=1
                    if self.retries[job]>2:raise
                    now=time.monotonic()
                    if now-self.last_adjust>30:
                        self.interval=max(2,min(6,self.interval*2));self.last_adjust=now
                    self.cooldown=max(self.cooldown,now+65)
                    event={'job_id':job,'http_status':429,'job_retry':self.retries[job],
                           'start_interval_seconds':self.interval,'cooldown_seconds':65}
                    self.rate_events.append(event)
                    save('rate_limit_events.json',self.rate_events)


def main():
    OUT.mkdir(exist_ok=True);WORK.mkdir(exist_ok=True)
    manifest=json.loads((OUT/'experiment_manifest.json').read_text())
    assert manifest['authorization']['additional_usd']==22
    assert json.loads((OUT/'preflight.json').read_text())['all_48_passed']
    if (OUT/'execution_manifest.json').exists():raise RuntimeError('Existing execution must be audited before restart')
    assert shutil.disk_usage(ROOT).free>512*1024**2
    provenance=json.loads((ROOT/'source_provenance.json').read_text())
    for record in provenance['verified_files']:
        assert transport.sha(ROOT/'vendor/pr917'/record['path'])==record['sha256']
    original=ROOT/'inputs/m1_original/runs/m1_think'
    original_db=transport.sha(original/'manifest.sqlite');original_ledger=transport.sha(original.parent/'api_usage.jsonl')
    decisions={d['decision_id']:d for d in manifest['decisions']}
    seeded=seed_pilot_store(original,STORE,sorted({d['branch_id'] for d in decisions.values()}))
    budget=Budget(WORK/'budget.sqlite',22*NANO)
    os.environ['ANTHROPIC_API_KEY']=Path('/Users/sriramm/.config/caaf_chess_m2/session_api_key').read_text().strip()
    gate=RateGate()
    transport.agent.CaafModel.query=transport.make_guarded_query(budget,STOPPED,completion=gate.completion)
    for s in [signal.SIGINT,signal.SIGTERM]:signal.signal(s,lambda *args:STOPPED.set())
    jobs=list(csv.DictReader((OUT/'planned_jobs.csv').open()));assert len(jobs)==3900
    source_paths=[ROOT/'requirements-preparation.lock',ROOT/'source_provenance.json',OUT/'experiment_manifest.json',OUT/'planned_jobs.csv']
    for folder in ['m2_runtime','scripts','tests','vendor/pr917']:
        source_paths += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    release={str(p.relative_to(ROOT)):transport.sha(p) for p in source_paths}
    save('execution_manifest.json',{'source_hashes':release,'additional_approved_usd':22,
        'original_database_sha256':original_db,'original_ledger_sha256':original_ledger,
        'seeded_original_rows':seeded,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'jobs':3900,'phase_order':['profiles_primary','profiles_stability'],
        'store_format':'gzip compressed blobs with original uncompressed identity hashes',
        'workers':16,'shared_budget_usd':22,'maximum_per_request_usd':.665,
        'transport':'Original first-stage tested guard; bounded zero-billed 429 retries and shared capacity backoff.',
        'completion_scope':'Direct causal profiles; shortcut not used for labels; no Milestone 3 or 4 paid experiments.'})
    with sqlite3.connect(WORK/'jobs.sqlite') as db:
        db.execute('create table jobs (id text primary key,phase text,decision_id text,arm text,rep integer,status text,result_json text)')
        db.executemany("insert into jobs values (?,?,?,?,?,'pending',null)",[(j['job_id'],j['phase'],j['decision_id'],j['arm'],int(j['replicate'])) for j in jobs])
    start=time.monotonic();finished=0

    def one(j):
        transport.CURRENT_JOB.set(j['job_id'])
        base={'job_id':j['job_id'],'phase':j['phase'],'decision_id':j['decision_id'],'arm':j['arm'],'replicate':int(j['replicate'])}
        if STOPPED.is_set():return {**base,'status':'not_started_stage_halted'}
        if shutil.disk_usage(ROOT).free<512*1024**2:
            STOPPED.set();return {**base,'status':'disk_space_halt'}
        branch=None;st=CompactStore(STORE)
        try:
            config=transport.agent.load_config('chess_haiku_think.yaml');config['model'].pop('budget',None)
            d=decisions[j['decision_id']];intervention={'type':'free'}
            if j['arm']=='keep_original_reply':
                k=int(d['step']);call=next(e for e in st.read_tape(d['branch_id']) if e['kind']=='model_call' and e['step']==k)
                text=call['response']['content'];assert hashlib.sha256(text.encode()).hexdigest()==d['original_reply_sha256']
                intervention={'type':'force','step':k,'content':text}
            branch=transport.agent.fork(st,j['checkpoint_id'],config=config,intervention=intervention)
            with sqlite3.connect(WORK/'jobs.sqlite',timeout=120) as jobs_db:
                jobs_db.execute("update jobs set status='running',result_json=? where id=?",
                                (json.dumps({**base,'status':'running','branch_id':branch.branch_id}),j['job_id']))
            outcome=transport.agent.run_to_end(branch)
            return {**base,'status':'completed','branch_id':branch.branch_id,'outcome':outcome}
        except BaseException as exc:
            STOPPED.set()
            if branch:
                branch.agent.tape.close();st.finish_branch(branch.branch_id,'failed',{'error_type':type(exc).__name__})
            return {**base,'status':'infrastructure_error','error_type':type(exc).__name__,
                    'branch_id':branch.branch_id if branch else None}
        finally:
            st.db.close()

    def publish(phase):
        with sqlite3.connect(WORK/'jobs.sqlite') as db:
            counts=dict(db.execute('select status,count(*) from jobs group by status'))
            per_phase={}
            for name in manifest['phases']:
                per_phase[name]=dict(db.execute('select status,count(*) from jobs where phase=? group by status',(name,)))
        b=budget.summary()
        progress={'current_phase':phase,'jobs':3900,'counts':counts,'per_phase':per_phase,
                  'elapsed_seconds':round(time.monotonic()-start),'budget':b,
                  'free_disk_mib':round(shutil.disk_usage(ROOT).free/1024**2)}
        save('progress.json',progress)
        print(json.dumps({'phase':phase,'completed':counts.get('completed',0),'errors':counts.get('infrastructure_error',0),
                          'liability_usd':b['accounted_plus_reserved_usd']}),flush=True)

    for phase in ['profiles_primary','profiles_stability']:
        if STOPPED.is_set():break
        phase_jobs=[j for j in jobs if j['phase']==phase]
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures=[pool.submit(one,j) for j in phase_jobs]
            with sqlite3.connect(WORK/'jobs.sqlite',timeout=120) as db:
                for future in as_completed(futures):
                    result=future.result();finished+=1
                    db.execute('update jobs set status=?,result_json=? where id=?',
                               (result['status'],json.dumps(result),result['job_id']));db.commit()
                    if finished%16==0 or STOPPED.is_set():publish(phase)
        publish(phase)
    with sqlite3.connect(WORK/'jobs.sqlite') as db:
        results=[json.loads(r[0]) for r in db.execute('select result_json from jobs where result_json is not null order by id')]
    save('results.json',results)
    assert transport.sha(original/'manifest.sqlite')==original_db
    assert transport.sha(original.parent/'api_usage.jsonl')==original_ledger
    for rel,expected in release.items():assert transport.sha(ROOT/rel)==expected
    save('completion.json',{'all_3900_completed':len(results)==3900 and all(r['status']=='completed' for r in results),
        'source_preserved':True,'executable_source_unchanged':True,'budget':budget.summary(),
        'elapsed_seconds':round(time.monotonic()-start),'shortcut_used':False,'milestone3_or4_started':False})
    print('REMAINING_PHASES_FINISHED',flush=True)


if __name__=='__main__':main()
