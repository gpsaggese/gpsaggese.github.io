"""Freeze the prospectively chosen direct-arm fallback within the new $22."""
import csv
import hashlib
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/pr917')]
os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'
os.environ['MSWEA_GLOBAL_CONFIG_DIR']=str(ROOT/'.offline_settings')

def deny(*args,**kwargs):
    raise RuntimeError('No network in the direct-arm preflight')

socket.socket.connect=deny
socket.create_connection=deny
import caaf_agent as agent
import caaf_chess as board
import caaf_store as store
import caaf_utils as utils
from m2_runtime.compact_store import CompactStore,seed_pilot_store

board._TB_DIR=ROOT/'inputs/tablebases'
agent.CaafModel.query=deny
OUT=ROOT/'experiments/m2_direct_v2'
OUT.mkdir(exist_ok=True)
if (OUT/'execution_manifest.json').exists():raise RuntimeError('Execution already frozen')
decisions=[d for d in csv.DictReader((ROOT/'experiments/m2_v1/decision_inventory.csv').open()) if d['profile_eligible']=='True']
assert len(decisions)==48
stable=[d for d in decisions if d['trial']=='0' and d['task_id'] in ['krk-m3-1','krk-m6-1']]
assert len(stable)==17
original=ROOT/'inputs/m1_original/runs/m1_think'
source_hash=hashlib.sha256((original/'manifest.sqlite').read_bytes()).hexdigest()
checks=[]
with tempfile.TemporaryDirectory(prefix='m2-direct-offline-') as td:
    root=Path(td)/'store'
    seeded=seed_pilot_store(original,root,sorted({d['branch_id'] for d in decisions}))
    assert seeded=={'runs':7,'branches':7,'snapshots':55}
    st=CompactStore(root)
    config=agent.load_config('chess_haiku_think.yaml');config['model'].pop('budget',None)
    for d in decisions:
        k=int(d['step']);reply=utils.model_calls(st,d['branch_id'])[k]['response']['content']
        assert hashlib.sha256(reply.encode()).hexdigest()==d['original_reply_sha256']
        branch=agent.fork(st,d['before_checkpoint'],config=config,intervention={'type':'force','step':k,'content':reply})
        agent.run_to_end(branch,stop_after_calls=k)
        actual=st.snapshot(branch.agent.last_snapshot_id);expected=st.snapshot(d['after_checkpoint'])
        a=json.loads(st.get_blob('context',actual['context_hash']));e=json.loads(st.get_blob('context',expected['context_hash']))
        passed={'env':actual['env_hash']==expected['env_hash'],
                'api_context':utils.api_view_hash(st,actual)==utils.api_view_hash(st,expected),
                'counters':all(a['counters'][key]==e['counters'][key] for key in ['n_calls','step_limit','n_consecutive_format_errors'])}
        assert all(passed.values()),d['decision_id']
        checks.append({'decision_id':d['decision_id'],'checks':passed})
    st.db.close()
assert hashlib.sha256((original/'manifest.sqlite').read_bytes()).hexdigest()==source_hash
jobs=[]
for phase,ds in [('profiles_primary',decisions),('profiles_stability',stable)]:
    for rep in range(30):
        for d in ds:
            for arm in ['keep_original_reply','redecide_reply']:
                jobs.append({'job_id':f'm2direct_{len(jobs):05d}','phase':phase,'decision_id':d['decision_id'],
                             'checkpoint_id':d['before_checkpoint'],'arm':arm,'replicate':rep})
assert len(jobs)==3900
with (OUT/'planned_jobs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(jobs[0]));w.writeheader();w.writerows(jobs)
manifest={'experiment_id':'chess_m2_direct_v2','date':'2026-11-19','status':'APPROVED_FOR_EXECUTION',
 'authorization':{'additional_usd':22,'source':'User explicitly approved the additional $22 to finish remaining Milestone 2 phases using the same key.',
                  'previous_stage_spend_usd':1.5271121,'maximum_cumulative_usage_rated_usd':23.5271121},
 'design_amendment':{'prior_manifest':'../m2_v1/experiment_manifest.json',
  'reason':'Conservative +/-0.20 shortcut equivalence is underpowered at the originally proposed 30/50 sample sizes. Use the documented direct-arm fallback before any new outcomes are collected.',
  'shortcut_used_for_causal_labels':False,'old_results_unchanged':True,
  'frozen_before_new_outcomes':True,'additional_sampling_based_on_new_outcomes':False},
 'population':{'natural_failed_runs':7,'distinct_tasks':4,'decisions':48},'decisions':decisions,
 'phases':{'profiles_primary':{'decisions':48,'per_arm':30,'jobs':2880,'planning_allowance_usd':16,'workers':16},
           'profiles_stability':{'decisions':17,'per_arm':30,'jobs':1020,'planning_allowance_usd':6,'workers':16}},
 'total_jobs':3900,'budget':{'additional_ceiling_usd':22,'planning_allowances_sum_usd':22,
  'guard':'One shared hard $22 reservation ledger across both phases. Phase planning allowances are not independent spending authorizations.',
  'source_suffix_cost_proxy_usd':14.45019345,'historical_fresh_mean_proxy_usd':18.8955,
  'costs_are_estimates':True,'rates_checked':'2026-11-19','pricing_url':'https://platform.claude.com/docs/en/about-claude/pricing'},
 'estimand':'P(win | re-decide complete reply) - P(win | keep original complete reply); proposal Delta has opposite sign',
 'policy':'Identical Haiku 5.5, adaptive thinking, max_tokens 16000, text-only reply history, original absolute step and format-error limits, deterministic original defender.',
 'sampling':{'independent_arms':True,'independent_decisions':True,'independent_repeats':True,
             'reuse_M1_or_first_stage_outcomes':False,'no_optional_stopping':True,
             'order':'replicate, original decision order, keep then re-decide; phases sequential'},
 'analysis':{'win_ci':'Wilson 95%','gain_ci':'Newcombe 95%','primary_tests':'Two-sided Fisher exact, Holm across 48 decisions',
   'stability':'Independent repeated profiles for trial-0 krk-m3-1 and krk-m6-1; report effect differences and rank agreement, not nonsignificance as proof of equality.',
   'exploratory_pattern_rules':'Report number of resolved harmful steps (G>0, Holm p<.05), top-two share of positive point gains, and unresolved small effects. Do not force sparse/diffuse labels.',
   'no_recovery':'Use confidence bounds. Zero wins never implies impossible or underspecified.'},
 'milestone4_export':{'features':'Separate table of pre-decision board and conversation features only; no oracle values, future events, outcomes or labels.',
                     'labels':'Primary direct gains with uncertainty, per task/run/decision; stability labels separate.',
                     'split_unit':'Whole starting-position task; keep both trials and all descendants in one split.'},
 'runtime':{'compression':'Lossless gzip; snapshot IDs and context/environment hashes remain those of uncompressed canonical bytes.',
   'retained_state':'Every step, full context and board; no checkpoint truncation.',
   'retry':'Up to two definite pre-inference HTTP 429 retries per job; count them separately, wait shared cooldown, reduce request rate. No hidden retries.',
   'hard_failure':'Stop on ambiguous transport/provider errors or exhausted budget; preserve request maximum and partial evidence.',
   'space_guard':'Stop before starting new work if free disk is below 512 MiB.'},
 'completion_criteria':['All 2880 primary and 1020 stability branches complete or explicitly document a blocking limit; never silently drop runs.',
    'Audit parents, before-input hashes, policy, limits, compressed blobs, tape chains, legal games, independent referee scores and spending.',
    'Report all seven direct profiles, first sparse/diffuse evidence with uncertainty, two independent stability repeats and Milestone 4-ready features/labels.',
    'No claim that the abandoned shortcut was statistically validated.']}
(OUT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'preflight.json').write_text(json.dumps({'all_48_passed':True,'source_preserved':True,'paid_calls':0,'network_blocked':True,'cases':checks},indent=2)+'\n')
print(json.dumps({'offline_identity_checks':len(checks),'jobs':len(jobs),'approved_additional_usd':22}))
