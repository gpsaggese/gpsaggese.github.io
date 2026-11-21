"""Offline audits, direct causal profiles, stability, and M4-ready exports."""
from __future__ import annotations
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import sqlite3
import sys
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/pr917')]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib_cache'))
import caaf_store as base
import caaf_chess as board
from m2_runtime.compact_store import CompactStore
from m2_runtime.budget import NANO,price_usage
from m2_runtime.features import extract_features

board._TB_DIR=ROOT/'inputs/tablebases'
OUT=ROOT/'experiments/m2_direct_v2'
WORK=ROOT/'paid_runs/m2_direct_v2'


def write_json(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def wilson(k,n):
    z=1.96;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,c-h),min(1,c+h)


def diff(a,na,b,nb):
    pa,pb=a/na,b/nb;la,ua=wilson(a,na);lb,ub=wilson(b,nb)
    return pa-pb-math.sqrt((pa-la)**2+(ub-pb)**2),pa-pb+math.sqrt((ua-pa)**2+(pb-lb)**2)


def fisher(a,na,b,nb):
    k=a+b
    def p(x):return math.comb(na,x)*math.comb(nb,k-x)/math.comb(na+nb,k)
    observed=p(a)
    return min(1,sum(p(x) for x in range(max(0,k-nb),min(k,na)+1) if p(x)<=observed*(1+1e-9)))


def holm(rows):
    previous=0
    for i,row in enumerate(sorted(rows,key=lambda r:r['fisher_p'])):
        previous=max(previous,min(1,row['fisher_p']*(len(rows)-i)));row['holm_p']=previous


def ranks(values):
    out=[]
    for value in values:
        less=sum(v<value for v in values);equal=sum(v==value for v in values)
        out.append(less+(equal+1)/2)
    return out


def correlation(a,b):
    a,b=ranks(a),ranks(b);ma=sum(a)/len(a);mb=sum(b)/len(b)
    den=math.sqrt(sum((x-ma)**2 for x in a)*sum((x-mb)**2 for x in b))
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/den if den else None


def audit_all(results,manifest,require_complete=True):
    st=CompactStore(WORK/'store');decisions={d['decision_id']:d for d in manifest['decisions']}
    actual_cost=0;paid_calls=0;steps=0;blob_keys=set()
    with sqlite3.connect(f"file:{WORK/'budget.sqlite'}?mode=ro",uri=True) as db:
        ledger={jid:(n,cost) for jid,n,cost in db.execute("select job_id,count(*),sum(liability) from calls where status='settled' group by job_id")}
        rejected=db.execute("select count(*) from calls where status='rejected_before_inference' and liability=0").fetchone()[0]
        if require_complete:assert db.execute("select count(*) from calls where status!='settled' and not(status='rejected_before_inference' and liability=0)").fetchone()[0]==0
    segment_count=0
    try:
        for result in results:
            assert result['status']=='completed'
            d=decisions[result['decision_id']];k=int(d['step']);bid=result['branch_id']
            branch=st.branch(bid)
            assert json.loads(branch['outcome_json'])==result['outcome']
            initial=st.snapshot(d['before_checkpoint'])
            ctx=json.loads(st.get_blob('context',initial['context_hash']))
            env=board.ChessEnv(json.loads(st.get_blob('env',initial['env_hash']))['start_fen'])
            env.restore(st.get_blob('env',initial['env_hash']),initial['env_hash'])
            prior=initial
            tape=[];snaps=[];parent=d['before_checkpoint']
            for segment in result.get('segments') or [bid]:
                segment_count+=1;record=st.branch(segment)
                assert record['parent_snapshot_id']==parent
                part=st.read_tape(segment);assert base.verify_tape(part);tape.extend(part)
                segment_snaps=st.snapshots(segment);expected_parent=parent
                for seq,snap in enumerate(segment_snaps):
                    assert snap['seq']==seq and snap['parent_id']==expected_parent
                    assert snap['id']==base.sha256(f"{snap['context_hash']}:{snap['env_hash']}:{snap['parent_id'] or ''}:{segment}".encode())
                    expected_parent=snap['id']
                if segment_snaps:parent=segment_snaps[-1]['id']
                snaps.extend(segment_snaps)
            calls=[e for e in tape if e['kind']=='model_call'];assert calls[0]['step']==k
            assert [c['step'] for c in calls]==list(range(k,k+len(calls)))
            if result['arm']=='keep_original_reply':
                assert calls[0]['response']['source']=='forced'
                assert hashlib.sha256(calls[0]['response']['content'].encode()).hexdigest()==d['original_reply_sha256']
            else:assert calls[0]['response']['source']=='model'
            assert all(call['response']['source']=='model' for call in calls[1:])
            by_step=defaultdict(list)
            for snap in snaps:
                by_step[snap['step']].append(snap)
                for kind,key in [('context',snap['context_hash']),('env',snap['env_hash'])]:
                    st.get_blob(kind,key);blob_keys.add((kind,key))
                saved=json.loads(st.get_blob('context',snap['context_hash']))
                assert saved['counters']['n_calls']==snap['step']
                assert saved['counters']['step_limit']==ctx['counters']['step_limit']
                assert saved['model']['name']==ctx['model']['name']
                assert saved['model']['sampling']==ctx['model']['sampling']
            per_job_cost=0;per_job_calls=0
            for call in calls:
                before=json.loads(st.get_blob('context',prior['context_hash']))
                visible=[{key:value for key,value in m.items() if key!='extra'} for m in before['messages']]
                assert base.sha256(base.canonical(visible))==call['request']['input_hash']
                assert call['request']['model']=='anthropic/claude-haiku-5-5'
                assert call['request']['sampling']=={'drop_params':True,'max_tokens':16000,'thinking':{'type':'adaptive'}}
                for command in [e for e in tape if e['kind']=='command' and e['step']==call['step']]:
                    output=env.execute(command['request'])
                    for key in ['output','returncode','exception_info']:
                        assert output.get(key,'')==command['response'].get(key,''), (result['job_id'],key)
                candidates=by_step[call['step']]
                for snap in candidates:assert base.sha256(env.state())==snap['env_hash']
                # First snapshot after the call is the pre-next-call state;
                # a later exit-at-limit snapshot can share its call index.
                prior=candidates[0]
                if call['response']['source']=='model':
                    per_job_calls+=1;per_job_cost+=price_usage(call['response']['usage'])
                steps+=1
            final=st.snapshot(result['outcome']['last_snapshot'])
            final_ctx=json.loads(st.get_blob('context',final['context_hash']))
            assert final_ctx['counters']['step_limit']==ctx['counters']['step_limit']
            assert final_ctx['counters']['n_calls']<=ctx['counters']['step_limit']
            game=json.loads(st.get_blob('env',final['env_hash']))
            assert game==env.game
            independently_scored=board.referee(game['start_fen'],game['moves_uci'])
            assert independently_scored['won']==result['outcome']['won']
            assert board._status_of(board.board_of(game))[0]==result['outcome']['status']
            assert ledger.get(result['job_id'],(0,0))==(per_job_calls,per_job_cost)
            paid_calls+=per_job_calls;actual_cost+=per_job_cost
        if require_complete:
            assert len(results)==3900
            assert sum(n for n,c in ledger.values())==paid_calls
            assert sum(c for n,c in ledger.values())==actual_cost
        assert actual_cost<=22*NANO
        return {'branches':len(results),'stored_branch_segments':segment_count,'replayed_decision_steps':steps,'verified_blobs':len(blob_keys),
                'paid_model_calls':paid_calls,'additional_usage_rated_usd':actual_cost/NANO,
                'all_environment_replays_match':True,'all_model_input_hashes_match':True,
                'all_snapshot_identities_match':True,'all_referee_scores_match':True,
                'request_ledger_reconciles_exactly':True,'complete_population_audit':require_complete,
                'outstanding_requests':0 if require_complete else None,'zero_billed_preinference_rejections':rejected}
    finally:st.db.close()


def make_plots(primary,stability):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=defaultdict(list)
    for row in primary:groups[(row['task_id'],row['trial'])].append(row)
    fig,axes=plt.subplots(4,2,figsize=(12,13),sharey=True,layout='constrained')
    for ax,((task,trial),rows) in zip(axes.flat,sorted(groups.items())):
        rows=sorted(rows,key=lambda r:r['step']);x=[r['step'] for r in rows];y=[r['repair_gain'] for r in rows]
        lo=[r['gain_ci_low'] for r in rows];hi=[r['gain_ci_high'] for r in rows]
        ax.errorbar(x,y,yerr=[[a-b for a,b in zip(y,lo)],[b-a for a,b in zip(y,hi)]],fmt='o-',capsize=3,color='#285f9e')
        ax.axhline(0,color='gray',lw=1);ax.set_ylim(-1.05,1.05)
        ax.set_xticks(x);ax.set_xlim(min(x)-.4,max(x)+.4)
        ax.set_title(f'{task}, trial {trial}');ax.set_xlabel('Original decision');ax.set_ylabel('Repair gain (re-decide − keep)')
        ax.grid(axis='y',alpha=.2)
        strong=[r for r in rows if r['holm_p']<.05]
        if strong:ax.scatter([r['step'] for r in strong],[r['repair_gain'] for r in strong],color='#c04b30',zorder=5,label='Holm p < .05');ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(groups):]:ax.set_visible(False)
    fig.suptitle('Direct counterfactual profiles — 30 continuations per arm\nUnadjusted 95% Newcombe intervals; seven natural failures',fontsize=13)
    fig.savefig(OUT/'primary_profiles.png',dpi=180);fig.savefig(OUT/'primary_profiles.pdf');plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(12,13),sharey=True,layout='constrained')
    for ax,((task,trial),rows) in zip(axes.flat,sorted(groups.items())):
        rows=sorted(rows,key=lambda r:r['step'])
        for arm,offset,color,label in [('redecide',0,'#285f9e','Before (re-decide)'),('keep',1,'#c04b30','After (keep reply)')]:
            x=[r['step']-1+offset for r in rows];y=[r[f'{arm}_probability'] for r in rows]
            ax.errorbar(x,y,yerr=[[r[f'{arm}_probability']-r[f'{arm}_ci_low'] for r in rows],
                                [r[f'{arm}_ci_high']-r[f'{arm}_probability'] for r in rows]],
                        fmt='o-',capsize=3,color=color,label=label)
        ax.set_ylim(-.03,1.03);ax.set_title(f'{task}, trial {trial}')
        ax.set_xticks(range(rows[-1]['step']+1));ax.set_xlim(-.3,rows[-1]['step']+.3)
        ax.set_xlabel('Checkpoint after original step');ax.set_ylabel('Continuation win probability')
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(groups):]:ax.set_visible(False)
    fig.suptitle('Checkpoint value curves — independent direct-arm samples\nWilson 95% intervals; duplicate checkpoint estimates are not pooled',fontsize=13)
    fig.savefig(OUT/'checkpoint_values.png',dpi=180);fig.savefig(OUT/'checkpoint_values.pdf');plt.close(fig)
    repeats=defaultdict(list)
    for row in stability:repeats[(row['task_id'],row['trial'])].append(row)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True,layout='constrained')
    for ax,((task,trial),rows) in zip(axes,sorted(repeats.items())):
        original=sorted(groups[(task,trial)],key=lambda r:r['step']);rows=sorted(rows,key=lambda r:r['step'])
        for data,color,label,offset in [(original,'#285f9e','Primary',-.08),(rows,'#c04b30','Independent repeat',.08)]:
            x=[r['step']+offset for r in data];y=[r['repair_gain'] for r in data]
            ax.errorbar(x,y,yerr=[[r['repair_gain']-r['gain_ci_low'] for r in data],[r['gain_ci_high']-r['repair_gain'] for r in data]],fmt='o-',capsize=3,color=color,label=label)
        ax.axhline(0,color='gray',lw=1);ax.set_ylim(-1.05,1.05);ax.set_title(f'{task}, trial {trial}');ax.set_xlabel('Original decision');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
        ax.set_xticks([r['step'] for r in rows])
    axes[0].set_ylabel('Repair gain');fig.suptitle('Independent stability repeats — 30 new continuations per arm')
    fig.savefig(OUT/'stability_profiles.png',dpi=180);fig.savefig(OUT/'stability_profiles.pdf');plt.close(fig)


def main():
    completion=json.loads((OUT/'completion.json').read_text());assert completion['all_3900_completed']
    assert completion['source_preserved'] and completion['executable_source_unchanged']
    manifest=json.loads((OUT/'experiment_manifest.json').read_text());results=json.loads((OUT/'results.json').read_text())
    planned=list(csv.DictReader((OUT/'planned_jobs.csv').open()))
    assert len({r['job_id'] for r in results})==len(planned)==3900
    audit=audit_all(results,manifest);write_json('audit.json',audit)
    grouped=defaultdict(list)
    for r in results:grouped[(r['phase'],r['decision_id'],r['arm'])].append(r)
    primary=[];stability=[]
    for phase,destination in [('profiles_primary',primary),('profiles_stability',stability)]:
        for d in manifest['decisions']:
            arms={g:grouped[(phase,d['decision_id'],g)] for g in ['keep_original_reply','redecide_reply']}
            if not arms['keep_original_reply']:continue
            assert all(len(v)==30 for v in arms.values())
            keep=sum(r['outcome']['won'] for r in arms['keep_original_reply']);redo=sum(r['outcome']['won'] for r in arms['redecide_reply'])
            lo,hi=diff(redo,30,keep,30);kl,kh=wilson(keep,30);rl,rh=wilson(redo,30)
            destination.append({'decision_id':d['decision_id'],'task_id':d['task_id'],'trial':int(d['trial']),'step':int(d['step']),
                'source_run_id':d['run_id'],'source_branch_id':d['branch_id'],'before_checkpoint':d['before_checkpoint'],'after_checkpoint':d['after_checkpoint'],
                'keep_wins':keep,'keep_n':30,'redecide_wins':redo,'redecide_n':30,
                'keep_probability':keep/30,'redecide_probability':redo/30,
                'keep_ci_low':kl,'keep_ci_high':kh,'redecide_ci_low':rl,'redecide_ci_high':rh,
                'repair_gain':(redo-keep)/30,'proposal_delta':(keep-redo)/30,'gain_ci_low':lo,'gain_ci_high':hi,
                'fisher_p':fisher(redo,30,keep,30),'holm_p':None})
        holm(destination)
    assert len(primary)==48 and len(stability)==17
    write_csv('primary_profiles.csv',primary);write_csv('stability_profiles.csv',stability)
    write_csv('primary_state_values.csv',[{'decision_id':r['decision_id'],'task_id':r['task_id'],'trial':r['trial'],'step':r['step'],
        'before_checkpoint':r['before_checkpoint'],'before_wins':r['redecide_wins'],'before_n':r['redecide_n'],
        'before_probability':r['redecide_probability'],'before_ci_low':r['redecide_ci_low'],'before_ci_high':r['redecide_ci_high'],
        'after_checkpoint':r['after_checkpoint'],'after_wins':r['keep_wins'],'after_n':r['keep_n'],
        'after_probability':r['keep_probability'],'after_ci_low':r['keep_ci_low'],'after_ci_high':r['keep_ci_high']} for r in primary])
    features=[];diagnostics=[];prefix_groups=defaultdict(list);st=CompactStore(WORK/'store')
    try:
        for d in manifest['decisions']:
            snap=st.snapshot(d['before_checkpoint']);ctx=json.loads(st.get_blob('context',snap['context_hash']));game=json.loads(st.get_blob('env',snap['env_hash']))
            meta={'decision_id':d['decision_id'],'task_id':d['task_id'],'difficulty_bin':int(d['task_id'].split('-')[1][1:]),
                  'trial':int(d['trial']),'step':int(d['step']),'source_run_id':d['run_id']}
            features.append({**meta,**extract_features(ctx,game)})
            prefix_groups[(snap['context_hash'],snap['env_hash'])].append(d)
            diagnostics.append({**meta,'original_command':d['commands_json'],'oracle_before_win':d['oracle_before_win'],
                                'oracle_after_win':d['oracle_after_win'],'source_outcome':d['source_outcome']})
    finally:st.db.close()
    write_csv('m4_predecision_features.csv',features);write_csv('decision_diagnostics_NOT_FEATURES.csv',diagnostics)
    write_csv('m4_primary_labels.csv',[{k:v for k,v in r.items() if k not in ['fisher_p']} for r in primary])
    labels_by_id={r['decision_id']:r for r in primary};shared_prefixes=[]
    for (context_hash,env_hash),items in prefix_groups.items():
        if len(items)<2:continue
        gains=[labels_by_id[d['decision_id']]['repair_gain'] for d in items]
        shared_prefixes.append({'before_context_sha256':context_hash,'before_env_sha256':env_hash,
            'opposite_point_gain_signs':min(gains)<0<max(gains),
            'decisions':[{'decision_id':d['decision_id'],'original_reply_sha256':d['original_reply_sha256'],
                'repair_gain':labels_by_id[d['decision_id']]['repair_gain'],
                'gain_ci95':[labels_by_id[d['decision_id']]['gain_ci_low'],labels_by_id[d['decision_id']]['gain_ci_high']]} for d in items]})
    timing_audit={'shared_saved_prefix_groups':shared_prefixes,
        'groups_with_opposite_point_gain_signs':sum(x['opposite_point_gain_signs'] for x in shared_prefixes),
        'interpretation':'Identical saved pre-generation context and board can precede different original replies and reply-specific gains. State-only inputs cannot distinguish those replies; opposite point estimates are diagnostic, not an additional significance test. This file contains labels and is excluded from predictors.'}
    write_json('m4_input_timing_audit_NOT_FEATURES.json',timing_audit)
    write_json('m4_dataset_schema.json',{'rows':48,'task_groups':4,'source_failed_runs':7,
        'join_key':'decision_id','metadata_columns':['decision_id','task_id','difficulty_bin','trial','step','source_run_id'],
        'feature_allowlist':list(extract_features(ctx,game)),
        'label_file':'m4_primary_labels.csv','independent_repeat_file':'stability_profiles.csv',
        'excluded_file':'decision_diagnostics_NOT_FEATURES.csv',
        'additional_excluded_file':'m4_input_timing_audit_NOT_FEATURES.json',
        'split_rule':'All trials and descendants of a starting-position task stay together. Task ID and difficulty bin are grouping metadata, excluded from the feature allowlist. This selected dataset has one starting position per represented difficulty bin.',
        'prediction_timing':'These features precede generation of the original reply. The label is action-specific; identical prefixes can have different labels for different recorded replies. A state-only model learns an average label in this selected training distribution, not the effect of a specific unseen reply. Action-specific prediction may require a separately declared after-generation/before-execution feature set, never future outcomes.',
        'limitation':'Only four starting positions and seven selected failures; this is not enough for robust Milestone 4 generalization claims.'})
    by_run=defaultdict(list)
    for r in primary:by_run[(r['task_id'],r['trial'])].append(r)
    patterns=[]
    for (task,trial),rows in sorted(by_run.items()):
        pos=sorted((max(0,r['repair_gain']) for r in rows),reverse=True);total=sum(pos)
        resolved=[r['step'] for r in rows if r['repair_gain']>0 and r['holm_p']<.05]
        patterns.append({'task_id':task,'trial':trial,'decisions':len(rows),
            'resolved_harmful_steps':resolved,'resolved_helpful_steps':[r['step'] for r in rows if r['repair_gain']<0 and r['holm_p']<.05],
            'top_two_share_of_positive_point_gains':sum(pos[:2])/total if total else None,
            'baseline_redecide_wins':next(r['redecide_wins'] for r in rows if r['step']==1),
            'baseline_redecide_ci95':list(wilson(next(r['redecide_wins'] for r in rows if r['step']==1),30)),
            'baseline_n':30,'pattern_conclusion':'Localized resolved harmful steps' if 1<=len(resolved)<=2 else ('Multiple resolved harmful steps' if len(resolved)>2 else 'No harmful step resolved after multiplicity correction'),
            'caveat':'Exploratory observed concentration; unresolved small effects prevent a definitive sparse/diffuse classification.'})
    write_json('pattern_summary.json',patterns)
    repeats=[]
    for task,trial in sorted({(r['task_id'],r['trial']) for r in stability}):
        a=sorted(by_run[(task,trial)],key=lambda r:r['step']);b=sorted([r for r in stability if (r['task_id'],r['trial'])==(task,trial)],key=lambda r:r['step'])
        assert [r['step'] for r in a]==[r['step'] for r in b]
        comparison=[{'step':x['step'],'primary_gain':x['repair_gain'],'repeat_gain':y['repair_gain'],
                     'difference':y['repair_gain']-x['repair_gain'],
                     'difference_ci90_conservative':[y['gain_ci_low']-x['gain_ci_high'],y['gain_ci_high']-x['gain_ci_low']]} for x,y in zip(a,b)]
        repeats.append({'task_id':task,'trial':trial,'spearman_rank_agreement':correlation([r['repair_gain'] for r in a],[r['repair_gain'] for r in b]),
                        'mean_absolute_gain_difference':sum(abs(r['difference']) for r in comparison)/len(comparison),
                        'steps':comparison,'interpretation':'Rank agreement and effect differences describe stability; interval overlap or nonsignificance is not proof of equivalence.'})
    write_json('stability_summary.json',repeats)
    make_plots(primary,stability)
    harmful=[r for r in primary if r['repair_gain']>0 and r['holm_p']<.05]
    helpful=[r for r in primary if r['repair_gain']<0 and r['holm_p']<.05]
    report={'milestone2_experimental_deliverables_complete':True,'method':'Direct randomized-policy resumption contrasts; whole assistant reply intervention',
        'shortcut_used':False,'shortcut_statistically_validated':False,'natural_failed_runs':7,'distinct_positions':4,
        'primary_decisions':48,'primary_continuations':2880,'stability_decisions':17,'stability_continuations':1020,
        'per_arm':30,'audit':audit,'additional_approved_usd':22,'first_stage_usd':1.5271121,
        'total_m2_usage_rated_usd':audit['additional_usage_rated_usd']+1.5271121,
        'resolved_harmful_decisions':harmful,'resolved_helpful_decisions':helpful,
        'administrative_resume_samples':sum(len(r.get('segments') or [r['branch_id']])>1 for r in results),
        'infrastructure_note':'Provider API credits ran out after 1414 completed samples. Completed outcomes were retained, and interrupted samples continued from their last saved states; no sampled prefixes were redrawn.',
        'patterns':patterns,'stability':repeats,'milestone4_dataset_rows':48,
        'm4_input_timing_audit':timing_audit,
        'milestone3_or4_experiments_started':False,'official_benchmark_score_claimed':False,
        'limitations':['Selected feasibility study: seven failed trajectories from only four positions.',
            '30 samples per arm leave small effects unresolved; confidence intervals are unadjusted, p-values Holm-corrected across 48 primary decisions.',
            'Sparse/diffuse summaries are exploratory and policy-conditional, not population classifications.',
            'Independent repeats measure stability, but do not prove equality of all effects.',
            'The predeclared repeats cover krk-m3-1 and krk-m6-1; krk-m4-1 primary discoveries are not independently repeated by this stability phase.',
            'Fresh sampling uses no model seed; deterministic environment replay is distinct from model determinism.',
            'The requested model ID and sampling settings are pinned; no immutable provider model-weight digest is recorded.',
            'Pre-decision features cannot identify an unobserved candidate reply; action-specific surrogate inputs need an explicit timing definition in Milestone 4.',
            'Milestone 4 needs expanded task groups and held-out evaluation before a surrogate or feedback benefit can be claimed.'],
        'analysis_source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'m2_runtime/features.py']}}
    write_json('REPORT.json',report)
    example_gains=[labels_by_id[f'krk-m4-1_trial{trial}_step001']['repair_gain'] for trial in [0,1]]
    lines=['# Milestone 2 — direct chess blame profiles','',
        'Experimental deliverables are complete for the seven-run chess feasibility scope. This is not an official benchmark score or a claim of broad generalization.','',
        f"All 3,900 continuations completed: 2,880 primary and 1,020 independent stability continuations. Additional usage-rated API cost: ${audit['additional_usage_rated_usd']:.6f} / $22 approved. Including the first validation stage: ${report['total_m2_usage_rated_usd']:.6f}.",'',
        f"Audited {audit['paid_model_calls']} paid model calls across {audit['stored_branch_segments']} stored branch segments and replayed {audit['replayed_decision_steps']} recorded decision steps. Parents, initial and subsequent API input hashes, compressed snapshot identities, tool outputs, legal games, referee outcomes and billing all reconcile.",'',
        report['infrastructure_note'],'',
        '## Method','',
        'Every original decision in all seven failed runs has 30 independent keep-original-reply continuations and 30 re-decide continuations. Positive repair gain means retrying helps; negative gain means the original reply helped. The entire reply, including explanation and command, is the intervention. Original step limits, model policy and defender are preserved.', '',
        'The initial shortcut equivalence design was underpowered. The prospectively versioned v2 protocol uses its documented direct-arm fallback, so labels do not depend on the unvalidated V-curve shortcut. Samples are fresh and independent of M1 and the first validation stage.','',
        'The state-value curves use re-decide wins for V(before) and keep-reply wins for V(after). Offline force-once checks verified that keeping the original reply restores its after-state conversation, board and remaining-call counters. Adjacent decisions provide independent estimates of shared checkpoint values; these are retained separately, not pooled or forced to telescope. See primary_state_values.csv and checkpoint_values.png/PDF.','',
        '## Profiles','',
        '| Task / trial | Decisions | Baseline wins | Resolved harmful steps | Resolved helpful steps | Top-two positive point share |',
        '| --- | ---: | ---: | --- | --- | ---: |']
    for row in patterns:
        share=row['top_two_share_of_positive_point_gains']
        lines.append(f"| {row['task_id']} / {row['trial']} | {row['decisions']} | {row['baseline_redecide_wins']}/30 | {row['resolved_harmful_steps']} | {row['resolved_helpful_steps']} | {share:.2f} |" if share is not None else f"| {row['task_id']} / {row['trial']} | {row['decisions']} | {row['baseline_redecide_wins']}/30 | {row['resolved_harmful_steps']} | {row['resolved_helpful_steps']} | n/a |")
    lines += ['', 'Resolved means two-sided Fisher p < .05 after Holm correction across all 48 primary decisions. Point concentration is descriptive and can be distorted by sampling noise; it is not proof that all remaining effects are zero. Full gains and unadjusted 95% intervals are in primary_profiles.csv and primary_profiles.png/PDF.', '',
              'A zero-win arm with 30 continuations has a Wilson 95% upper win-probability bound of approximately 11.4%. No observed recovery does not establish impossibility. Baseline recovery intervals are retained in pattern_summary.json.', '',
              '## Independent stability repeats','']
    for row in repeats:
        lines.append(f"- {row['task_id']} trial {row['trial']}: rank agreement {row['spearman_rank_agreement']}; mean absolute gain difference {row['mean_absolute_gain_difference']:.3f}. See stability_summary.json and stability_profiles.png/PDF for step-level intervals.")
    lines += ['', '## Milestone 4 handoff','',
        'm4_predecision_features.csv and m4_primary_labels.csv join on decision_id. The feature allowlist and group-split rules are in m4_dataset_schema.json. Oracle diagnostics and future outcomes are isolated in decision_diagnostics_NOT_FEATURES.csv. Independent repeat labels are separate. No surrogate has been trained.', '',
        f'The two krk-m4-1 first decisions have identical saved pre-generation context and board, but gains of {example_gains[0]:+.3f} and {example_gains[1]:+.3f} for different original replies. The state-only feature vectors are consequently identical. The timing audit is in m4_input_timing_audit_NOT_FEATURES.json; it contains labels and must not be used as predictor input. Define candidate-reply feature timing explicitly before attempting reply-specific Milestone 4 prediction.','',
        'Only four starting positions are represented. Expand the task population, hold out entire positions, and evaluate scaffold changes on unseen tasks before claiming Milestone 4 success.', '',
        '## Limits','',*[f'- {s}' for s in report['limitations']]]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'deliverables_complete':True,'additional_usd':audit['additional_usage_rated_usd'],
                      'resolved_harmful':len(harmful),'resolved_helpful':len(helpful),'primary_profiles':7,'independent_repeats':2},indent=2))


if __name__=='__main__':main()
