"""Continue administratively interrupted samples under the existing $22 cap."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/pr917')]
import run_remaining_m2 as rt

OUT=rt.OUT;WORK=rt.WORK;STORE=rt.STORE;STOPPED=rt.STOPPED


def credit_check(key):
    request=urllib.request.Request('https://api.anthropic.com/v1/messages/count_tokens',
        data=json.dumps({'model':'claude-haiku-5-5','messages':[{'role':'user','content':'test'}]}).encode(),
        headers={'x-api-key':key,'anthropic-version':'2023-06-01','content-type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=30) as response:
            result={'http_status':response.status,'input_tokens':json.load(response).get('input_tokens'),'paid_inference':False}
    except urllib.error.HTTPError as exc:
        body=json.loads(exc.read());message=str(body.get('error',{}).get('message','')).replace(key,'[REDACTED]')
        result={'http_status':exc.code,'error_type':body.get('error',{}).get('type'),'message':message,'paid_inference':False}
    except Exception as exc:result={'error_type':type(exc).__name__,'paid_inference':False}
    rt.save('credit_probe_latest.json',result)
    return result


def continuation_plan(st,job,previous):
    segments=list(previous.get('segments') or ([previous['branch_id']] if previous.get('branch_id') else []))
    if segments:
        snaps=st.snapshots(segments[-1])
        checkpoint=snaps[-1]['id'] if snaps else st.branch(segments[-1])['parent_snapshot_id']
    else:checkpoint=job['checkpoint_id']
    return checkpoint,segments


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-only',action='store_true');group.add_argument('--resume',action='store_true');args=parser.parse_args()
    key=Path('/Users/sriramm/.config/caaf_chess_m2/session_api_key').read_text().strip()
    check=credit_check(key);print(json.dumps(check),flush=True)
    if args.check_only or check.get('http_status')!=200:return
    lock=(WORK/'runner.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another resume runner is active')
    manifest=json.loads((OUT/'experiment_manifest.json').read_text())
    assert manifest['authorization']['additional_usd']==22
    budget=rt.Budget(WORK/'budget.sqlite',22*rt.NANO)
    with sqlite3.connect(WORK/'budget.sqlite') as db:
        assert db.execute("select count(*) from calls where status in ('reserved','uncertain_maximum_retained')").fetchone()[0]==0
    with sqlite3.connect(WORK/'jobs.sqlite') as db:
        assert db.execute("select count(*) from jobs where status='running'").fetchone()[0]==0
        before=dict(db.execute('select status,count(*) from jobs group by status'))
    prior=json.loads((OUT/'execution_manifest.json').read_text())
    for rel,expected in prior['source_hashes'].items():
        if '.pytest_cache' not in Path(rel).parts:assert rt.transport.sha(ROOT/rel)==expected,rel
    original=ROOT/'inputs/m1_original/runs/m1_think'
    assert rt.transport.sha(original/'manifest.sqlite')==prior['original_database_sha256']
    assert rt.transport.sha(original.parent/'api_usage.jsonl')==prior['original_ledger_sha256']
    proof=json.loads((OUT/'credit_resume_preflight.json').read_text())
    with sqlite3.connect(WORK/'jobs.sqlite') as db:
        partial_ids={jid for jid,raw in db.execute("select id,result_json from jobs where status!='completed' and result_json is not null") if json.loads(raw).get('branch_id')}
    assert partial_ids=={r['job_id'] for r in proof['jobs']}
    check_store=rt.CompactStore(STORE)
    for record in proof['jobs']:
        snap=check_store.snapshot(record['resume_checkpoint'])
        assert snap['context_hash']==record['context_sha256'] and snap['env_hash']==record['env_sha256']
        check_store.get_blob('context',snap['context_hash']);check_store.get_blob('env',snap['env_hash'])
    check_store.db.close()
    epochs=sorted(OUT.glob('resume_epoch_*'))
    epoch=OUT/f'resume_epoch_{len(epochs)+1:02d}';epoch.mkdir()
    for name in ['jobs.sqlite','budget.sqlite']:
        with sqlite3.connect(WORK/name) as src,sqlite3.connect(epoch/name) as dst:src.backup(dst)
    previous_completion=json.loads((OUT/'completion.json').read_text())
    (epoch/'previous_completion.json').write_text(json.dumps(previous_completion,indent=2)+'\n')
    source_paths=[ROOT/'scripts/resume_remaining_m2.py',ROOT/'scripts/run_remaining_m2.py',ROOT/'scripts/run_validation_stage.py',
                  ROOT/'m2_runtime/budget.py',ROOT/'m2_runtime/compact_store.py',ROOT/'requirements-preparation.lock',
                  OUT/'experiment_manifest.json',OUT/'planned_jobs.csv']
    source_paths += list((ROOT/'vendor/pr917').glob('*.py'))+list((ROOT/'vendor/pr917/config').glob('*.yaml'))
    release={str(p.relative_to(ROOT)):rt.transport.sha(p) for p in source_paths}
    (epoch/'execution_manifest.json').write_text(json.dumps({'approved_additional_usd':22,'counts_before':before,
        'source_hashes':release,'runtime_cache_excluded':True,
        'resume_rule':'Continue partial trajectories from their last saved checkpoint. Preserve all completed outcomes and all paid prefixes. No redraws.',
        'original_execution_sha256':rt.transport.sha(OUT/'execution_manifest.json'),
        'utc_start':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2)+'\n')
    os.environ['ANTHROPIC_API_KEY']=key
    if shutil.which('caffeinate'):
        subprocess.Popen([shutil.which('caffeinate'),'-i','-w',str(os.getpid())],env={},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    gate=rt.RateGate()
    events=OUT/'rate_limit_events.json'
    if events.exists():
        gate.rate_events=json.loads(events.read_text())
        for e in gate.rate_events:gate.retries[e['job_id']]=max(gate.retries[e['job_id']],e['job_retry'])
    class NotDispatched(RuntimeError):status_code=400
    def completion(**kwargs):
        try:return gate.completion(**kwargs)
        except RuntimeError as exc:
            if str(exc)=='Stage stopped before wire request':raise NotDispatched('Internal stop before HTTP dispatch') from None
            raise
        except BaseException as exc:
            message=str(getattr(exc,'message','')).replace(key,'[REDACTED]')
            message=re.sub(r'sk-ant-[A-Za-z0-9_-]+','[REDACTED]',message)[:1500]
            (epoch/'provider_error.json').write_text(json.dumps({'error_type':type(exc).__name__,
                'http_status':getattr(exc,'status_code',None),'message':message},indent=2)+'\n')
            raise
    rt.transport.agent.CaafModel.query=rt.transport.make_guarded_query(budget,STOPPED,completion=completion)
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *args:STOPPED.set())
    jobs=list(csv.DictReader((OUT/'planned_jobs.csv').open()));decisions={d['decision_id']:d for d in manifest['decisions']}
    start=time.monotonic();count=0
    def one(job,previous):
        rt.transport.CURRENT_JOB.set(job['job_id'])
        base={'job_id':job['job_id'],'phase':job['phase'],'decision_id':job['decision_id'],'arm':job['arm'],'replicate':int(job['replicate'])}
        if STOPPED.is_set():return {**previous,**base,'status':'not_started_stage_halted'}
        if shutil.disk_usage(ROOT).free<512*1024**2:
            STOPPED.set();return {**previous,**base,'status':'disk_space_halt'}
        st=rt.CompactStore(STORE);branch=None;segments=[]
        try:
            checkpoint,segments=continuation_plan(st,job,previous)
            config=rt.transport.agent.load_config('chess_haiku_think.yaml');config['model'].pop('budget',None)
            d=decisions[job['decision_id']];snap=st.snapshot(checkpoint)
            intervention={'type':'free'}
            if job['arm']=='keep_original_reply' and snap['step']<int(d['step']):
                call=next(e for e in st.read_tape(d['branch_id']) if e['kind']=='model_call' and e['step']==int(d['step']))
                text=call['response']['content'];assert hashlib.sha256(text.encode()).hexdigest()==d['original_reply_sha256']
                intervention={'type':'force','step':int(d['step']),'content':text}
            branch=rt.transport.agent.fork(st,checkpoint,config=config,intervention=intervention)
            segments.append(branch.branch_id)
            running={**base,'status':'running','branch_id':branch.branch_id,'segments':segments,'resumed_from':checkpoint,'resume_epoch':epoch.name}
            with sqlite3.connect(WORK/'jobs.sqlite',timeout=120) as db:
                db.execute("update jobs set status='running',result_json=? where id=?",(json.dumps(running),job['job_id']))
            outcome=rt.transport.agent.run_to_end(branch)
            return {**running,'status':'completed','outcome':outcome}
        except BaseException as exc:
            STOPPED.set()
            if branch:
                branch.agent.tape.close();st.finish_branch(branch.branch_id,'failed',{'error_type':type(exc).__name__})
            return {**base,'status':'infrastructure_error','error_type':type(exc).__name__,
                    'branch_id':branch.branch_id if branch else previous.get('branch_id'),'segments':segments}
        finally:st.db.close()
    def publish(phase):
        with sqlite3.connect(WORK/'jobs.sqlite') as db:
            counts=dict(db.execute('select status,count(*) from jobs group by status'))
            per={name:dict(db.execute('select status,count(*) from jobs where phase=? group by status',(name,))) for name in manifest['phases']}
        progress={'current_phase':phase,'jobs':3900,'counts':counts,'per_phase':per,
            'elapsed_seconds':previous_completion['elapsed_seconds']+round(time.monotonic()-start),
            'resume_epoch':epoch.name,'budget':budget.summary(),'free_disk_mib':round(shutil.disk_usage(ROOT).free/1024**2)}
        rt.save('progress.json',progress);print(json.dumps({'counts':counts,'liability_usd':progress['budget']['accounted_plus_reserved_usd']}),flush=True)
    rt.save('completion.json',{'all_3900_completed':False,'status':'RUNNING_RESUMED','source_preserved':True,
        'executable_source_unchanged':True,'elapsed_seconds':previous_completion['elapsed_seconds']})
    handoff=json.loads((OUT/'HANDOFF.json').read_text());handoff.update({'status':'RUNNING_RESUMED','process_pid':os.getpid(),'resume_epoch':epoch.name})
    rt.save('HANDOFF.json',handoff)
    for phase in ['profiles_primary','profiles_stability']:
        if STOPPED.is_set():break
        with sqlite3.connect(WORK/'jobs.sqlite') as db:
            previous={jid:json.loads(raw) if raw else {} for jid,status,raw in db.execute('select id,status,result_json from jobs where phase=?',(phase,)) if status!='completed'}
        todo=[j for j in jobs if j['phase']==phase and j['job_id'] in previous]
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures=[pool.submit(one,j,previous[j['job_id']]) for j in todo]
            with sqlite3.connect(WORK/'jobs.sqlite',timeout=120) as db:
                for future in as_completed(futures):
                    result=future.result();count+=1
                    db.execute('update jobs set status=?,result_json=? where id=?',(result['status'],json.dumps(result),result['job_id']));db.commit()
                    if count%16==0 or STOPPED.is_set():publish(phase)
        publish(phase)
    with sqlite3.connect(WORK/'jobs.sqlite') as db:
        results=[json.loads(r[0]) for r in db.execute('select result_json from jobs where result_json is not null order by id')]
    rt.save('results.json',results)
    assert rt.transport.sha(original/'manifest.sqlite')==prior['original_database_sha256']
    assert rt.transport.sha(original.parent/'api_usage.jsonl')==prior['original_ledger_sha256']
    for rel,expected in release.items():assert rt.transport.sha(ROOT/rel)==expected,rel
    rt.save('completion.json',{'all_3900_completed':len(results)==3900 and all(r['status']=='completed' for r in results),
        'status':'COMPLETE' if all(r['status']=='completed' for r in results) and len(results)==3900 else 'HALTED',
        'source_preserved':True,'executable_source_unchanged':True,'budget':budget.summary(),
        'elapsed_seconds':previous_completion['elapsed_seconds']+round(time.monotonic()-start),
        'source_cache_note':'Original pytest-cache change was non-executable; corrected source capture excludes caches.',
        'resume_epochs':len(epochs)+1,'shortcut_used':False,'milestone3_or4_started':False})
    print('RESUME_FINISHED',flush=True)


if __name__=='__main__':main()
