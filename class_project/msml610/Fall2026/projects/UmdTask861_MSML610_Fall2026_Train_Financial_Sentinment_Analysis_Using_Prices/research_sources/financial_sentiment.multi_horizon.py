# %% [markdown]
# # Registered multi-horizon development comparison
# 
# Fixed A / AMZN / MSFT 2023 dataset; 5/10/20 sessions are project choices, not a course-prescribed list.
# Two objectives: raw and SPY-excess **price** returns, excluding dividends. Strict total-return track remains separate with zero eligible labels.
# This is further development analysis, not blind confirmation or alpha. Run offline with existing private archives; no new data, credentials, downloads, or tensor inference.

# %%
import hashlib, json, math, sys, tempfile
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display
repo=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'research/financial_sentiment_using_prices/run_historical.py').is_file())
research=repo/'research/financial_sentiment_using_prices';sys.path.insert(0,str(research))
from multi_horizon import read,sha,load_archive,run_registered,named_target
from price_labels import _connected_groups
root=repo/'reports/financial_sentiment_runs/multi_horizon_2023_20261021'
base=root/'evaluation';saved=read(base/'summary.json');regpath=research/'multi_horizon_2023_protocol.json';reg=read(regpath)
assert saved['status']=='complete' and len(saved['evaluations'])==36
assert sha(regpath)==saved['registration_sha256']
assert reg['horizons_sessions']==[5,10,20] and reg['companies']==['A','AMZN','MSFT']
assert reg['new_5_20_model_outcomes_inspected_before_registration'] is False
for name,digest in saved['output_files_sha256'].items():assert sha(base/name)==digest,name
for name,digest in saved['generation_sources_sha256'].items():assert sha(research/name)==digest,name
print('Registration:',reg['registered_at'],saved['registration_sha256'])
print('36 fixed combinations; strict eligible labels:',saved['strict_eligible'])

# %% [markdown]
# ## Independently recompute labels and split evidence
# 
# Daily `t` is the New York day anchor. Scheduled 09:30 is a logical clock, not certification of the eligible trade behind `o`, an auction price or a fill. All missingness and cutoff exclusions are retained. No additional fixed embargo is introduced; test labels overlap.

# %%
coverage=[];cross_sources=[]
for symbol in reg['companies']:
    news,prices,calendar,protocol,audit=load_archive(repo/reg['company_bindings'][symbol]['capture'],reg['company_bindings'][symbol])
    days=[s['date'] for s in calendar['sessions']]
    assert not any(audit['missing_calendar_dates'].values())
    assert audit['official_auction_or_fill_certified'] is False
    for h in reg['horizons_sessions']:
        rows=read(base/symbol/f'h{h}'/'company_days.json');split=read(base/symbol/f'h{h}'/'split.json')
        for r in rows:
            i,j=r['entry_session_index'],r['exit_session_index']
            assert j-i==h
            assert r['entry_at']==calendar['sessions'][i]['open_at'] and r['exit_at']==calendar['sessions'][j]['open_at']
            assert max(r['source_dates'])<days[i]
            raw=prices[symbol,days[j]]/prices[symbol,days[i]]-1
            spy=prices['SPY',days[j]]/prices['SPY',days[i]]-1
            assert math.isclose(raw,r['stock_return'],abs_tol=1e-14)
            assert math.isclose(spy,r['spy_return'],abs_tol=1e-14)
            assert math.isclose(raw-spy,r['excess_return'],abs_tol=1e-14)
            expected=[prices[symbol,days[i-1]]/prices[symbol,days[i-1-lag]]-1 for lag in [5,20]]
            assert r['price_features']==expected
        assert all(r['exit_at']<reg['test_start']+'T00:00:00Z' for r in split['train'])
        assert all(r['entry_at'][:10]>=reg['test_start'] for r in split['test'])
        groups=_connected_groups(rows)
        assert not ({groups[r['sample_id']] for r in split['train']}&{groups[r['sample_id']] for r in split['test']})
        assert len(split['train'])+len(split['test'])+len(split['excluded'])==len(rows)
        coverage.append({'company':symbol,'sessions':h,'observed':len(rows),'train':len(split['train']),'test':len(split['test']),'excluded':len(split['excluded'])})
for h in reg['horizons_sessions']:
    partition={part:{s for symbol in reg['companies'] for r in read(base/symbol/f'h{h}'/'split.json')[part] for s in r['source_ids']} for part in ['train','test']}
    shared=partition['train']&partition['test'];assert not shared
    cross_sources.append({'sessions':h,'cross_company_direct_source_train_test_overlap':len(shared)})
display(pd.DataFrame(coverage));display(pd.DataFrame(cross_sources))
print('Cross-company economic-event identity is not independently certified; companies fit separately.')

# %%
for symbol in reg['companies']:
    for part in ['train','test']:
        sets=[{r['sample_id'] for r in read(base/symbol/f'h{h}'/'split.json')[part]} for h in [5,10,20]]
        common=set.intersection(*sets)
        for e in saved['evaluations']:
            if e['company']==symbol and e['pool']=='common_eligible':
                report=read(base/e['path']/'report.json');assert set(report[part+'_ids'])==common
    binding=reg['company_bindings'][symbol]
    for name,digest in binding['existing10_core_sha256'].items():assert sha(repo/binding['existing10_base']/name)==digest
    assert sha(repo/binding['existing10_six']/'predictions.json')==binding['existing10_six_predictions_sha256']
print('Common IDs are eligibility intersections only; old ten-session artifacts unchanged.')

# %% [markdown]
# ## Fixed offline reproduction
# 
# All 36 recorded comparisons are reproduced into a new private directory. No new headline inference is allowed; full five-session FinBERT failures remain explicit rather than changing rows. Numeric predictions and metrics must match the saved run.

# %%
replay_parent=Path(tempfile.mkdtemp(prefix='notebook_replay_',dir=root));(replay_parent/'.gitignore').write_text('*\n')
replay=replay_parent/'evaluation';again=run_registered(repo,regpath,replay)
index=lambda s:{(e['company'],e['horizon'],e['pool'],e['target']):e for e in s['evaluations']}
left,right=index(saved),index(again);assert set(left)==set(right)
for key,e in left.items():
    a,b=read(base/e['path']/'report.json'),read(replay/right[key]['path']/'report.json')
    for name in ['train_ids','test_ids','tone_coverage','metrics','cost_evaluation']:assert a[name]==b[name],(key,name)
    pa,pb=read(base/e['path']/'predictions.json'),read(replay/right[key]['path']/'predictions.json')
    assert pa==pb,key
    for p in pb:
        assert math.isclose(p['target_return'],p['stock_return'] if e['target']=='raw_price_return' else p['excess_return'],abs_tol=1e-14)
print('PASS: 36 exact metric/prediction/cost reproductions. New tensor inference: 0.')

# %%
metrics=pd.read_csv(root/'delivery/metrics.csv');costs=pd.read_csv(root/'delivery/cost_scenarios.csv')
assert len(metrics)==216 and sum(metrics.status=='Available')==204
assert len(costs)==936
for pool in ['horizon_specific','common_eligible']:
    for target in reg['targets']:
        view=metrics[(metrics.pool==pool)&(metrics.target==target)].copy();view['MAE_pp']=view.mae*100
        print(pool,target)
        display(view.pivot(index=['company','horizon'],columns='model',values='MAE_pp'))
print('Unavailable slots:',metrics[metrics.status!='Available'][['company','horizon','pool','target','model']].to_dict('records'))

# %%
for e in saved['evaluations']:
    report=read(base/e['path']/'report.json');pred=read(base/e['path']/'predictions.json');c=report['cost_evaluation']
    lookup={p['sample_id']:p for p in pred};selected=[lookup[s] for s in c['selected_sample_ids']]
    assert all(b['entry_session_index']>=a['exit_session_index'] for a,b in zip(selected,selected[1:]))
    for model,scenarios in c['models'].items():
        for scenario in scenarios:
            capital=1.;trades=0
            for p in selected:
                active=model in ['always_company','always_spy'] or (model!='cash' and p['predictions'][model]>0)
                if active:
                    value=p['spy_return'] if model=='always_spy' else p['stock_return']
                    capital*=(1-scenario['assumed_roundtrip_cost_bps']/20000)**2*(1+value);trades+=1
            assert math.isclose(capital-1,scenario['cumulative_simulated_price_return'],abs_tol=1e-12)
            assert trades==scenario['roundtrip_trades']
    assert c['actual_broker_charges'] is None and c['cash_dividends_included'] is False
display(costs[(costs.pool=='horizon_specific')&(costs.target=='excess_price_return')&(costs.assumed_roundtrip_cost_bps==10)][['company','horizon','model','selected_intervals','roundtrip_trades','active_held_sessions','cumulative_simulated_price_return']])
print('All cost factors recomputed; assumed costs, unknown actual charges; no cross-horizon cumulative-return ranking.')

# %%
fig,axes=plt.subplots(1,3,figsize=(13,3.8),sharey=True)
for ax,symbol in zip(axes,reg['companies']):
    d=metrics[(metrics.pool=='common_eligible')&(metrics.target=='excess_price_return')&(metrics.company==symbol)]
    for model in reg['baseline_models']+reg['cached_tone_models']:
        v=d[d.model==model].sort_values('horizon');ax.plot(v.horizon,v.mae_relative_to_train_mean,marker='o',label=model)
    ax.axhline(1,color='gray',linestyle=':');ax.set_xticks([5,10,20]);ax.set_title(symbol);ax.set_xlabel('Trading sessions')
axes[0].set_ylabel('MAE / same-horizon training-mean MAE')
axes[-1].legend(fontsize=6,loc='best');fig.suptitle('Common IDs, SPY-excess price returns: development sensitivity')
fig.tight_layout();fig.savefig(replay_parent/'common_excess_mae_ratio.png',dpi=180);display(fig);plt.close(fig)
receipt={'status':'verified','comparisons':36,'exact_reproductions':36,'potential_model_slots':216,'measured_model_slots':204,'unavailable_slots':12,'strict_eligible':0,'alpha_established':False,'cost_scenarios':len(costs),'replay_directory':str(replay_parent)}
(replay_parent/'notebook_validation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt)

# %% [markdown]
# ## Conclusions and remaining requirements
# 
# Results vary across company, target and horizon; no preferred horizon or alpha is selected. Full-pool five-session added tone controls are unavailable, while common-row six-model comparisons are complete. Strict point-in-time total return remains unverified. See the separate report for exact maturity exclusions, execution-price limitations and cost assumptions.
# 
# Remaining: LM baseline; price-label text-model fine-tuning; independent large-move news-attribution gold evaluation (including unexplained moves and false attribution); strict publication/corporate-action evidence; Docker execution and required course narrative/video/submission with owner authorization. No prior PDF is changed.
