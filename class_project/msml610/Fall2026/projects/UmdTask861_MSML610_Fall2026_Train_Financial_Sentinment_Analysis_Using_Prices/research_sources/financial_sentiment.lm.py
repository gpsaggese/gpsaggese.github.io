# %% [markdown]
# # Loughran–McDonald dictionary baseline: fixed 2023 development study
# 
# The official project's Milestone 2 explicitly requests the Loughran–McDonald dictionary. LM here is this financial lexicon, not a generic language model. Official Notre Dame source permits academic research; commercial use requires separate licensing. The current 1993–2025 dictionary was downloaded in 2026 and is not certified as a 2023 as-of snapshot.
# 
# Two added Ridge controls use the same fixed A/AMZN/MSFT rows, raw/SPY-excess **price** targets, 5/10/20-session horizons and September 1 cutoff. No transformer fine-tuning, installs, credentials, downloads or tensor inference occur in this notebook. Previous controls and reports remain intact. This is development analysis, not alpha or a blind confirmation. Dividends are excluded; the original strict total-return track remains separate.

# %%
import hashlib,json,math,re,sys,tempfile,csv
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display
repo=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'research/financial_sentiment_using_prices/lm_baseline.py').is_file())
research=repo/'research/financial_sentiment_using_prices';sys.path.insert(0,str(research))
from lm_baseline import run_registered,verify_inputs,case_split
from multi_horizon import read,sha,named_target
root=repo/'reports/financial_sentiment_runs/lm_2023_20261021';base=root/'evaluation'
regpath=research/'lm_2023_protocol.json';reg=read(regpath);saved=read(base/'summary.json')
assert saved['status']=='complete' and len(saved['cases'])==36
assert sha(regpath)==saved['registration_sha256']
assert reg['new_LM_outcomes_inspected_before_registration'] is False
prior,old,source=verify_inputs(repo,reg)
for name,digest in saved['output_files_sha256'].items():assert sha(base/name)==digest,name
for name,digest in saved['generation_sources_sha256'].items():assert sha(research/name)==digest,name
print('Registered before LM outcomes:',reg['registered_at'],saved['registration_sha256'])
print('Official definition: Loughran–McDonald dictionary; academic research only.')
print('Dictionary hash:',saved['dictionary_audit']['dictionary_sha256'])

# %% [markdown]
# ## Independent score and coverage checks
# 
# The preregistered score counts occurrences, uses all retained ASCII tokens as denominator, and averages distinct original headlines equally per company-day. No negation, stemming, stopword removal, or return-dependent row selection occurs. Zero polarity hits give numeric zero with an explicit no-match diagnostic; they are not independently labeled neutral. Missing/zero-token titles would make the entire case's added controls unavailable.

# %%
with (source/'Loughran-McDonald_MasterDictionary_1993-2025.csv').open(encoding='utf-8-sig',newline='') as f:
    words=list(csv.DictReader(f))
positive={r['Word'] for r in words if int(r['Positive'])>0}
negative={r['Word'] for r in words if int(r['Negative'])>0}
assert len(words)==86553 and len(positive)==347 and len(negative)==2345
assert not positive&negative
priorreg=read(repo/reg['prior_registration']);cache=read(base/'headline_scores.json');feature_by_company={}
for symbol in reg['companies']:
    news=read(repo/priorreg['company_bindings'][symbol]['capture']/'news.json');lookup={n['source_row_index']:n for n in news}
    rows=read(prior/symbol/'h10/company_days.json');audit=read(base/symbol/'company_day_scores.json');feature_by_company[symbol]={}
    for row in rows:
        titles=list(dict.fromkeys(lookup[i]['headline'] for i in row['source_row_indices']))
        assert '\n'.join(titles)==row['text']
        scores=[]
        for title in titles:
            tokens=[t.upper() for t in re.findall('[A-Za-z]+',title) if len(t)>=2];assert tokens
            p=sum(t in positive for t in tokens);n=sum(t in negative for t in tokens)
            score=(p-n)/len(tokens);h=hashlib.sha256(title.encode()).hexdigest();receipt=cache[h]
            assert receipt['status']=='Available' and receipt['input_sha256']==h
            assert receipt['tokens']==len(tokens) and receipt['positive']==p and receipt['negative']==n
            assert receipt['score']==score
            assert (receipt['polarity_evidence']=='NoPolarityMatch')==(p+n==0)
            scores.append(score)
        expected=sum(scores)/len(scores);a=audit[row['sample_id']]
        assert a['status']=='Available' and a['score']==expected and a['headline_count']==len(titles)
        feature_by_company[symbol][row['sample_id']]=expected
display(pd.DataFrame(saved['companies']).T)
print('Distinct titles:',saved['unique_headlines_scored'],'without polarity:',saved['unique_headlines_without_polarity'])
assert saved['unique_headlines_scored']==7293 and saved['unique_headlines_without_polarity']==5235
assert sum(c['zero_polarity_company_days'] for c in saved['companies'].values())==57
print('No score unavailable; zero-hit rows retained. Current dictionary does not certify 2023 availability.')

# %%
for case in saved['cases']:
    split=case_split(prior,case);report=read(base/case['path']/'report.json');original=read(prior/case['path']/'report.json')
    pred=read(base/case['path']/'predictions.json');op=read(prior/case['path']/'predictions.json');fit=read(base/case['path']/'fitted_lm_models.json')
    assert report['train_ids']==original['train_ids'] and report['test_ids']==original['test_ids']
    assert [r['sample_id'] for r in split['test']]==[p['sample_id'] for p in pred]
    assert all(r['exit_at']<'2023-09-01T00:00:00Z' for r in split['train'])
    assert all(r['entry_at'][:10]>='2023-09-01' for r in split['test'])
    for p,oldp,row in zip(pred,op,split['test']):
        assert p['target_return']==named_target(row,case['target'])
        for name,value in oldp['predictions'].items():assert p['predictions'][name]==value
        for key in oldp:
            if key!='predictions':assert p[key]==oldp[key]
    for name,model in fit.items():
        assert model['training_sample_ids']==report['train_ids'] and model['test_sample_ids']==report['test_ids']
        feature=feature_by_company[case['company']]
        x=[(r['price_features'] if name=='price_lm_ridge' else [])+[feature[r['sample_id']]] for r in split['train']]
        means=[sum(row[j] for row in x)/len(x) for j in range(len(x[0]))]
        assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(means,model['means']))
        mean_y=sum(named_target(r,case['target']) for r in split['train'])/len(split['train'])
        assert math.isclose(mean_y,model['intercept'],abs_tol=1e-12)
        for name2,m in report['metrics'].items():
            computed=sum(abs(p['predictions'][name2]-p['target_return']) for p in pred)/len(pred)
            assert math.isclose(computed,m['mae'],abs_tol=1e-12)
    assert report['cost_evaluation']['selected_sample_ids']==original['cost_evaluation']['selected_sample_ids']
    for name,values in original['cost_evaluation']['models'].items():assert report['cost_evaluation']['models'][name]==values
print('Exact prior rows/labels/predictions/costs retained; training-only calibration and named-target MAE verified.')

# %% [markdown]
# ## Offline reproduction of the fixed comparison
# 
# Existing baselines remain unchanged; only the two added LM calibrations are fitted. This repeat checks reproducibility, not a new test set. No dictionary or model retrieval occurs.

# %%
replay_parent=Path(tempfile.mkdtemp(prefix='notebook_replay_',dir=root));(replay_parent/'.gitignore').write_text('*\n')
replay=replay_parent/'evaluation';again=run_registered(repo,regpath,replay)
assert len(again['cases'])==36
for a,b in zip(saved['cases'],again['cases']):
    assert a==b
    for file in ['report.json','predictions.json','fitted_lm_models.json','paired_losses.json']:
        assert read(base/a['path']/file)==read(replay/b['path']/file),(a['path'],file)
assert read(base/'headline_scores.json')==read(replay/'headline_scores.json')
for symbol in reg['companies']:assert read(base/symbol/'company_day_scores.json')==read(replay/symbol/'company_day_scores.json')
print('36 exact predictions, fitted calibrations, metrics, paired losses, costs and score reproductions passed.')

# %%
metrics=pd.read_csv(root/'delivery/metrics.csv');costs=pd.read_csv(root/'delivery/cost_scenarios.csv')
assert len(metrics)==288 and sum(metrics.status=='Available')==276
assert sum(metrics.status!='Available')==12 and len(costs)==1152
for pool in reg['pools']:
    for target in reg['targets']:
        view=metrics[(metrics.pool==pool)&(metrics.target==target)].copy();view['MAE_pp']=100*view.mae
        print(pool,target);display(view.pivot(index=['company','horizon'],columns='model',values='MAE_pp'))
display(pd.read_csv(root/'delivery/coverage.csv'))
print('12 unavailable slots are inherited full-five-session FinBERT; every LM case has full coverage.')

# %%
for case in saved['cases']:
    report=read(base/case['path']/'report.json');pred=read(base/case['path']/'predictions.json');cost=report['cost_evaluation']
    lookup={p['sample_id']:p for p in pred};selected=[lookup[s] for s in cost['selected_sample_ids']]
    assert all(b['entry_session_index']>=a['exit_session_index'] for a,b in zip(selected,selected[1:]))
    for model,scenarios in cost['models'].items():
        for s in scenarios:
            capital=1.;trades=0;held=0
            for p in selected:
                active=model in ['always_company','always_spy'] or (model!='cash' and p['predictions'][model]>0)
                if active:
                    value=p['spy_return'] if model=='always_spy' else p['stock_return']
                    capital*=(1-s['assumed_roundtrip_cost_bps']/20000)**2*(1+value);trades+=1
                    held+=p['exit_session_index']-p['entry_session_index']
            assert math.isclose(capital-1,s['cumulative_simulated_price_return'],abs_tol=1e-12)
            assert trades==s['roundtrip_trades'] and held==cost['interval_audit']['active_held_sessions'][model]
    assert cost['actual_broker_charges'] is None and cost['cash_dividends_included'] is False
view=costs[(costs.pool=='horizon_specific')&(costs.target=='excess_price_return')&(costs.assumed_roundtrip_cost_bps==10)&costs.model.isin(['lm_tone_ridge','price_lm_ridge','price_ridge','always_company'])]
display(view[['company','horizon','model','selected_intervals','roundtrip_trades','active_held_sessions','cumulative_simulated_price_return']])
print('1152 assumed-cost scenarios independently checked; no actual fee or executable-fill certification.')

# %%
fig,axes=plt.subplots(2,3,figsize=(13,7))
for i,target in enumerate(reg['targets']):
    for j,symbol in enumerate(reg['companies']):
        ax=axes[i,j];d=metrics[(metrics.pool=='common_eligible')&(metrics.target==target)&(metrics.company==symbol)]
        for model in ['train_mean','price_ridge','lm_tone_ridge','price_lm_ridge']:
            v=d[d.model==model].sort_values('horizon');ax.plot(v.horizon,v.mae_relative_to_train_mean,marker='o',label=model)
        ax.axhline(1,color='gray',linestyle=':');ax.set_xticks([5,10,20]);ax.set_title(symbol+' / '+target.replace('_price_return',''));ax.set_xlabel('Trading sessions')
        if j==0:ax.set_ylabel('MAE / same-case training-mean MAE')
axes[0,-1].legend(fontsize=7);fig.suptitle('Fixed common IDs: additional LM development sensitivity')
fig.tight_layout();fig.savefig(replay_parent/'common_LM_mae_ratios.png',dpi=180);display(fig);plt.close(fig)
receipt={'status':'verified','cases':36,'exact_reproductions':36,'LM_available_company_days':451,'LM_unavailable_company_days':0,'unique_headlines':7293,'no_polarity_unique_headlines':5235,'model_slots':288,'measured_slots':276,'unavailable_inherited_slots':12,'cost_scenarios_checked':1152,'prior_controls_changed':False,'transformer_fine_tuned':False,'strict_eligible':0,'alpha_established':False,'replay_directory':str(replay_parent)}
(replay_parent/'notebook_validation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt)

# %% [markdown]
# ## Interpretation and remaining requirements
# 
# LM has complete computational coverage but most distinct titles have no positive/negative dictionary match. Numeric zero is retained with explicit diagnostics. Error and sparse cost-return changes are inconsistent; every company, target and horizon is reported. No horizon, model or positive interval example is promoted as confirmed alpha.
# 
# The lexicon version postdates 2023, headline historical availability remains unknown, labels overlap, daily opens are not certified fills, and dividends are excluded. The original strict total-return target is unchanged. Remaining: price-label transformer fine-tuning, independent large-move attribution gold evaluation with unexplained/false-attribution outcomes, strict source/corporate-action evidence, Docker execution and owner-authorized course video/submission. Previous reports and PDFs are unchanged.
