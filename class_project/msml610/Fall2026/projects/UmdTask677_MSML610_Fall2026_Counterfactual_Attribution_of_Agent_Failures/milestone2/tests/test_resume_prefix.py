import importlib.util
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'vendor/pr917')]
os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'
spec=importlib.util.spec_from_file_location('credit_resume',ROOT/'scripts/resume_remaining_m2.py')
resume=importlib.util.module_from_spec(spec);spec.loader.exec_module(resume)
from m2_runtime.compact_store import CompactStore,seed_pilot_store


def test_partial_trajectory_continues_without_redrawing_paid_prefix(tmp_path,monkeypatch):
    agent=resume.rt.transport.agent
    monkeypatch.setattr(agent.CaafModel,'query',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Live model forbidden')))
    original=ROOT/'inputs/m1_original/runs/m1_think'
    import json
    manifest=json.loads((ROOT/'experiments/m2_direct_v2/experiment_manifest.json').read_text())
    d=next(d for d in manifest['decisions'] if d['decision_id']=='krk-m4-1_trial0_step001')
    target=tmp_path/'store';seed_pilot_store(original,target,[d['branch_id']]);st=CompactStore(target)
    config=agent.load_config('chess_haiku_think.yaml');config['model'].pop('budget')
    calls={e['step']:e for e in st.read_tape(d['branch_id']) if e['kind']=='model_call'}
    first=agent.fork(st,d['before_checkpoint'],config=config,intervention={'type':'force','step':1,'content':calls[1]['response']['content']})
    agent.run_to_end(first,stop_after_calls=1)
    checkpoint,segments=resume.continuation_plan(st,{'checkpoint_id':d['before_checkpoint']},{'branch_id':first.branch_id})
    assert checkpoint==first.agent.last_snapshot_id and segments==[first.branch_id]
    child=agent.fork(st,checkpoint,config=config,intervention={'type':'replay','source_branch':d['branch_id'],'steps':list(range(2,max(calls)+1))})
    outcome=agent.run_to_end(child)
    source_final=st.snapshots(d['branch_id'])[-1];child_final=st.snapshot(outcome['last_snapshot'])
    assert child_final['env_hash']==source_final['env_hash']
    recorded=[e['step'] for bid in [first.branch_id,child.branch_id] for e in st.read_tape(bid) if e['kind']=='model_call']
    assert recorded==list(range(1,max(calls)+1))
    assert child_final['step']==source_final['step']
    st.db.close()


def test_empty_interruption_uses_original_parent(tmp_path):
    agent=resume.rt.transport.agent
    import json
    manifest=json.loads((ROOT/'experiments/m2_direct_v2/experiment_manifest.json').read_text());d=manifest['decisions'][0]
    target=tmp_path/'store';seed_pilot_store(ROOT/'inputs/m1_original/runs/m1_think',target,[d['branch_id']]);st=CompactStore(target)
    config=agent.load_config('chess_haiku_think.yaml');config['model'].pop('budget')
    first=agent.fork(st,d['before_checkpoint'],config=config,intervention={'type':'free'});first.agent.tape.close()
    checkpoint,segments=resume.continuation_plan(st,{'checkpoint_id':d['before_checkpoint']},{'branch_id':first.branch_id})
    assert checkpoint==d['before_checkpoint'] and segments==[first.branch_id]
    st.db.close()
