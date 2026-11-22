"""A mocked full job keeps its sampled prefix and reconciles across segments."""
import importlib.util
import json
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'vendor/pr917')]
import run_validation_stage as transport
from m2_runtime.compact_store import CompactStore,seed_pilot_store
from m2_runtime.budget import Budget,NANO


def test_segmented_job_replay_and_ledger_reconcile(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('segmented_audit',ROOT/'scripts/report_remaining_m2.py')
    report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
    monkeypatch.setattr(report,'WORK',tmp_path)
    manifest=json.loads((ROOT/'experiments/m2_direct_v2/experiment_manifest.json').read_text())
    d=next(d for d in manifest['decisions'] if d['decision_id']=='krk-m4-1_trial0_step001')
    seed_pilot_store(ROOT/'inputs/m1_original/runs/m1_think',tmp_path/'store',[d['branch_id']])
    st=CompactStore(tmp_path/'store');budget=Budget(tmp_path/'budget.sqlite',22*NANO)
    calls={e['step']:e for e in st.read_tape(d['branch_id']) if e['kind']=='model_call'}
    remaining=list(range(2,max(calls)+1))
    def fake_completion(**kwargs):
        step=remaining.pop(0)
        return transport.litellm.ModelResponse(model='claude-haiku-5-5',
            choices=[{'index':0,'message':{'role':'assistant','content':calls[step]['response']['content']},'finish_reason':'stop'}],
            usage={'prompt_tokens':1000,'completion_tokens':100,'total_tokens':1100})
    monkeypatch.setattr(transport.agent.CaafModel,'query',transport.make_guarded_query(budget,threading.Event(),completion=fake_completion))
    token=transport.CURRENT_JOB.set('fake-segmented-job')
    config=transport.agent.load_config('chess_haiku_think.yaml');config['model'].pop('budget')
    try:
        first=transport.agent.fork(st,d['before_checkpoint'],config=config,
            intervention={'type':'force','step':1,'content':calls[1]['response']['content']})
        transport.agent.run_to_end(first,stop_after_calls=3)
        checkpoint=first.agent.last_snapshot_id
        st.finish_branch(first.branch_id,'failed',{'error_type':'AdministrativePause'})
        child=transport.agent.fork(st,checkpoint,config=config,intervention={'type':'free'})
        outcome=transport.agent.run_to_end(child)
        result={'job_id':'fake-segmented-job','status':'completed','decision_id':d['decision_id'],
                'arm':'keep_original_reply','branch_id':child.branch_id,
                'segments':[first.branch_id,child.branch_id],'outcome':outcome}
        audit=report.audit_all([result],manifest,require_complete=False)
        assert audit['stored_branch_segments']==2
        assert audit['paid_model_calls']==max(calls)-1
        assert audit['replayed_decision_steps']==max(calls)
        assert audit['request_ledger_reconciles_exactly'] and not remaining
    finally:
        transport.CURRENT_JOB.reset(token);st.db.close()
