"""Registered offline 5/10/20-session development sensitivity, no new data or NLP.

Raw price return and SPY-excess price return remain separate named objectives.
Both exclude cash dividends; neither certifies the strict total-return contract.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from run_historical import (NY, build_rows, retrospective_split, evaluate_models,
                            cost_evaluation, price_class, save)
from price_labels import _connected_groups

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()


def session_date(timestamp,days):
    t=datetime.fromisoformat(timestamp.replace('Z','+00:00'))
    if t.tzinfo is None or t.utcoffset() is None:raise ValueError('Naive price timestamp')
    local=t.astimezone(NY)
    if (local.hour,local.minute,local.second,local.microsecond)!=(0,0,0,0):raise ValueError('Daily price timestamp is not New York midnight')
    day=local.date().isoformat()
    if day not in days:raise ValueError('Price timestamp outside retained session calendar')
    return day


def load_archive(path,binding):
    path=Path(path);manifest=read(path/'manifest.json')
    if manifest['status']!='complete_bounded_capture' or sha(path/'manifest.json')!=binding['capture_manifest_sha256']:
        raise ValueError('Capture status/hash changed')
    if manifest['files_sha256']!=binding['input_files_sha256']:raise ValueError('Registered input list changed')
    for n,h in manifest['files_sha256'].items():
        if Path(n).name!=n or sha(path/n)!=h:raise ValueError('Capture input hash mismatch')
    protocol=read(path/'protocol.json');calendar=read(path/'calendar.json');news=read(path/'news.json')
    dates=[s['date'] for s in calendar['sessions']]
    if dates!=sorted(set(dates)):raise ValueError('Calendar dates must be distinct and ordered')
    for s in calendar['sessions']:
        t=datetime.fromisoformat(s['open_at'].replace('Z','+00:00'))
        if t.tzinfo is None or t.astimezone(NY).date().isoformat()!=s['date'] or (t.astimezone(NY).hour,t.astimezone(NY).minute)!=(9,30):
            raise ValueError('Scheduled open/session-date alignment failed')
    pages=[r for r in manifest['requests'] if r['file'].startswith('prices_')]
    if not pages or manifest['bars_summary']['pagination_complete'] is not True:raise ValueError('Incomplete observed price pagination')
    prices={};counts=Counter();token=None;offsets=Counter()
    for i,r in enumerate(pages):
        if r['status']!=200 or r['params'].get('page_token')!=token or r['params'].get('timeframe')!='1Day' or r['params'].get('adjustment')!='split' or r['params'].get('feed')!='sip':
            raise ValueError('Price request chain/semantics changed')
        payload=read(path/r['file']);token=payload.get('next_page_token')
        if token!=r.get('next_page_token') or bool(token)!=(i<len(pages)-1):raise ValueError('Incomplete or changed page token')
        for symbol,bars in payload['bars'].items():
            if symbol not in {protocol['symbol'],'SPY'}:raise ValueError('Unexpected price symbol')
            for b in bars:
                day=session_date(b['t'],set(dates));v=b['o']
                if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:raise ValueError('Invalid observed daily o')
                if (symbol,day) in prices:raise ValueError('Duplicate observed daily price')
                prices[symbol,day]=v;counts[symbol]+=1
                offsets[str(datetime.fromisoformat(b['t'].replace('Z','+00:00')).astimezone(NY).utcoffset())]+=1
    if dict(counts)!=manifest['bars_summary']['records']:raise ValueError('Bar count mismatch')
    audit={'price_bars':dict(counts),'pagination_complete':True,'session_dates':len(dates),
           'calendar_first':dates[0],'calendar_last':dates[-1],'daily_timestamp_anchor':'New York midnight',
           'ny_utc_offsets':dict(offsets),'missing_calendar_dates':{s:[d for d in dates if (s,d) not in prices] for s in [protocol['symbol'],'SPY']},
           'scheduled_open_clock':'09:30 America/New_York; logical label/maturity clock',
           'provider_o_execution_timestamp_certified':False,'official_auction_or_fill_certified':False,
           'limit':'Daily t identifies the aggregation date, not the exact eligible trade behind o; no execution-price certification'}
    return news,prices,calendar,protocol,audit


def common_splits(splits,anchor=10):
    ids={part:set.intersection(*[{r['sample_id'] for r in s[part]} for s in splits.values()]) for part in ['train','test']}
    if ids['train']&ids['test']:raise ValueError('Sample changes train/test partition across horizons')
    result={}
    for h,s in splits.items():
        common={**s,'train':[r for r in s['train'] if r['sample_id'] in ids['train']],
                'test':[r for r in s['test'] if r['sample_id'] in ids['test']], 'pool':'common_eligible'}
        for part in ['train','test']:
            expected=[r['sample_id'] for r in splits[anchor][part] if r['sample_id'] in ids[part]]
            if [r['sample_id'] for r in common[part]]!=expected:raise ValueError('Common ID order differs')
        result[h]=common
    return result


def tone_features(rows,news,cache):
    lookup={n['source_row_index']:n for n in news};features={};missing={}
    for r in rows:
        titles=list(dict.fromkeys(lookup[i]['headline'] for i in r['source_row_indices']))
        if '\n'.join(titles)!=r['text']:raise ValueError('Headline input budget changed')
        values=[];bad=[]
        for text in titles:
            h=hashlib.sha256(text.encode()).hexdigest();v=cache.get(h)
            if not v or v.get('input_sha256')!=h or v.get('status')!='Available' or v.get('truncated') is not False:
                bad.append(h);continue
            scores=v['scores']
            if set(scores)!={'positive','neutral','negative'} or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and 0<=x<=1 for x in scores.values()) or abs(sum(scores.values())-1)>1e-5:
                bad.append(h);continue
            values.append(scores['positive']-scores['negative'])
        if bad:missing[r['sample_id']]=bad
        else:features[r['sample_id']]=sum(values)/len(values)
    return features,missing


def named_target(row,target):
    if target=='excess_price_return':return row['excess_return']
    if target=='raw_price_return':return row['stock_return']
    raise ValueError('Unknown named price-return target')


def fit_target(split,protocol,target,tone,missing):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error,mean_squared_error,balanced_accuracy_score,f1_score
    adapted=deepcopy(split)
    # Internal adapter for the established fitter; restore genuine component
    # returns below, never publish raw returns disguised as excess returns.
    for part in ['train','test']:
        for r in adapted[part]:
            r['excess_return']=named_target(r,target);r['price_class']=price_class(r['excess_return'],protocol['class_neutral_band'])
    metrics,predictions,artifacts=evaluate_models(adapted,protocol)
    originals={r['sample_id']:r for r in split['test']}
    for p in predictions:
        r=originals[p['sample_id']];p.update(excess_return=r['excess_return'],price_class=r['price_class'],
            target_name=target,target_return=named_target(r,target),target_class=price_class(named_target(r,target),protocol['class_neutral_band']))
    required=split['train']+split['test'];uncovered=[r['sample_id'] for r in required if r['sample_id'] in missing or r['sample_id'] not in tone]
    status={'status':'Unavailable' if uncovered else 'Available','required_rows':len(required),'missing_rows':uncovered,
            'missing_unique_headline_hashes':sorted({h for s in uncovered for h in missing.get(s,[])}),'selective_row_removal':False,'new_tensor_inference':False}
    if not uncovered:
        y=np.asarray([named_target(r,target) for r in split['train']]);truth=np.asarray([named_target(r,target) for r in split['test']])
        models={}
        for name,price in [('finbert_tone_ridge',False),('price_finbert_ridge',True)]:
            def design(rows):return [(r['price_features'] if price else [])+[tone[r['sample_id']]] for r in rows]
            scaler=StandardScaler().fit(design(split['train']));model=Ridge(alpha=protocol['ridge_alpha'],solver='lsqr',tol=1e-8).fit(scaler.transform(design(split['train'])),y)
            values=model.predict(scaler.transform(design(split['test'])))
            if not np.isfinite(values).all():raise ValueError('Nonfinite calibrated prediction')
            classes=[price_class(float(v),protocol['class_neutral_band']) for v in values];labels=[p['target_class'] for p in predictions]
            metrics[name]={'samples':len(truth),'mae':float(mean_absolute_error(truth,values)),'rmse':float(math.sqrt(mean_squared_error(truth,values))),
                'relative_direction_accuracy':float(np.mean(np.sign(values)==np.sign(truth))),
                'price_class_balanced_accuracy':float(balanced_accuracy_score(labels,classes)),
                'price_class_macro_f1':float(f1_score(labels,classes,labels=[-1,0,1],average='macro',zero_division=0))}
            for p,v in zip(predictions,values):p['predictions'][name]=float(v)
            models[name]={'training_sample_ids':[r['sample_id'] for r in split['train']], 'means':scaler.mean_.tolist(),'scales':scaler.scale_.tolist(),
                          'coefficients':model.coef_.tolist(),'intercept':float(model.intercept_),'alpha':protocol['ridge_alpha']}
        artifacts['tone_models']=models
    denominator=metrics['train_mean']['mae']
    for m in metrics.values():
        m['direction_accuracy']=m.pop('relative_direction_accuracy')
        m['mae_relative_to_train_mean']=m['mae']/denominator if denominator else None
    return metrics,predictions,artifacts,status


def retained10(repo,binding,split,target,tone,missing):
    base=Path(repo)/binding['existing10_base'];six=Path(repo)/binding['existing10_six']
    for n,h in binding['existing10_core_sha256'].items():
        if sha(base/n)!=h:raise ValueError('Retained ten-session result changed')
    if sha(six/'predictions.json')!=binding['existing10_six_predictions_sha256']:raise ValueError('Retained six-control predictions changed')
    original=read(base/'split.json')
    if split!=original:raise ValueError('Ten-session source rows/split no longer reproduce retained result')
    report=read(six/'report.json');metrics=deepcopy(report['metrics']);predictions=deepcopy(read(six/'predictions.json'))
    required=split['train']+split['test']
    if any(r['sample_id'] in missing or r['sample_id'] not in tone for r in required):raise ValueError('Retained ten-session tone coverage changed')
    if [p['sample_id'] for p in predictions]!=[r['sample_id'] for r in split['test']]:raise ValueError('Retained prediction order changed')
    for p,r in zip(predictions,split['test']):p.update(target_name=target,target_return=r['excess_return'],target_class=r['price_class'])
    denominator=metrics['train_mean']['mae']
    for m in metrics.values():m['direction_accuracy']=m.pop('relative_direction_accuracy');m['mae_relative_to_train_mean']=m['mae']/denominator if denominator else None
    status={'status':'Available','required_rows':len(split['train'])+len(split['test']),'missing_rows':[], 'selective_row_removal':False,'new_tensor_inference':False}
    return metrics,predictions,{'retained_existing10_base':str(base),'retained_six':str(six)},status


def interval_audit(predictions,costs):
    selected={p['sample_id']:p for p in predictions if p['sample_id'] in costs['selected_sample_ids']}
    intervals=[{k:selected[s][k] for k in ['sample_id','entry_at','exit_at','entry_session_index','exit_session_index']} for s in costs['selected_sample_ids']]
    return {'selected_intervals':intervals,'total_available_held_sessions':sum(p['exit_session_index']-p['entry_session_index'] for p in intervals),
        'active_held_sessions':{name:sum(p['exit_session_index']-p['entry_session_index'] for p in selected.values() if name in ['always_company','always_spy'] or (name!='cash' and p['predictions'][name]>0)) for name in costs['models']},
        'cost_formula':'(1 - assumed_roundtrip_bps/20000)^2 * (1 + observed_price_return), per active interval',
        'cross_horizon_exposure_equal':False,'cash_interest_included':False}


def run_registered(repo,registration,output):
    repo,registration,output=Path(repo),Path(registration),Path(output);reg=read(registration);started=time.monotonic()
    if output.exists() or output.is_symlink():raise FileExistsError('Choose a new multi-horizon output')
    output.mkdir(mode=0o700,parents=True);(output/'.gitignore').write_text('*\n')
    if reg['horizons_sessions']!=[5,10,20] or reg['companies']!=['A','AMZN','MSFT']:raise ValueError('Use the registered project choices')
    registration_hash=sha(registration);generation={p.name:sha(p) for p in [Path(__file__),Path(build_rows.__code__.co_filename),Path(_connected_groups.__code__.co_filename)]}
    if sha(repo/reg['tone_cache'])!=reg['tone_cache_sha256']:raise ValueError('Cached tone receipts changed')
    cache=read(repo/reg['tone_cache'])['headline_results'];summary={'status':'running','started_at':stamp(),'registration_sha256':registration_hash,'companies':{},'evaluations':[],'strict_eligible':0,'alpha_established':False,'generation_sources_sha256':generation}
    save(output/'summary.json',summary)
    try:
        for symbol in reg['companies']:
            binding=reg['company_bindings'][symbol];news,prices,cal,protocol,price_audit=load_archive(repo/binding['capture'],binding)
            for key in ['source_date_start','source_date_end_exclusive','test_start','entry_delay_sessions','class_neutral_band','ridge_alpha','tfidf','assumed_roundtrip_cost_bps']:
                if protocol[key]!=reg[key]:raise ValueError('Capture differs from registration: '+key)
            if protocol['symbol']!=symbol or any(not reg['source_date_start']<=n['source_date']<reg['source_date_end_exclusive'] for n in news):raise ValueError('Source company/date coverage changed')
            splits={};rows_by_h={};audits={}
            for h in reg['horizons_sessions']:
                cfg={**protocol,'horizon_sessions':h};rows,outside=build_rows(news,prices,cal,cfg);split=retrospective_split(rows,reg['test_start'])
                rows_by_h[h]=rows;splits[h]=split;groups=_connected_groups(rows)
                shared={groups[r['sample_id']] for r in split['train']}&{groups[r['sample_id']] for r in split['test']}
                if shared:raise ValueError('Retained source/event groups cross split')
                audits[h]={'company_days':len(rows),'label_status_counts':dict(Counter(r['label_status'] for r in rows)),'outside_calendar':outside,
                    'train_rows':len(split['train']),'test_rows':len(split['test']),'exclusion_counts':dict(Counter(reason for x in split['excluded'] for reason in x['reasons'])),
                    'excluded':split['excluded'],'shared_proxy_groups':len(shared),'last_training_outcome':max((r['exit_at'] for r in split['train']),default=None),
                    'first_test_entry':min((r['entry_at'] for r in split['test']),default=None),'last_test_outcome':max((r['exit_at'] for r in split['test']),default=None)}
                dest=output/symbol/f'h{h}';dest.mkdir(parents=True)
                save(dest/'company_days.json',rows);save(dest/'split.json',split);save(dest/'audit.json',audits[h])
            common=common_splits(splits)
            tone,missing=tone_features(rows_by_h[10],news,cache)
            summary['companies'][symbol]={'price_audit':price_audit,'horizons':audits,'common_train_ids':[r['sample_id'] for r in common[10]['train']],
                'common_test_ids':[r['sample_id'] for r in common[10]['test']],'tone_missing_source_rows':len(missing)}
            for pool,pool_splits in [('horizon_specific',splits),('common_eligible',common)]:
                for h in reg['horizons_sessions']:
                    for target in reg['targets']:
                        if time.monotonic()-started>reg['resource_budget']['wall_seconds']:raise TimeoutError('Registered analysis wall budget exhausted')
                        split=pool_splits[h];cfg={**protocol,'horizon_sessions':h};reused=pool=='horizon_specific' and h==10 and target=='excess_price_return'
                        if len(split['train'])<10 or len(split['test'])<5:
                            summary['evaluations'].append({'company':symbol,'horizon':h,'pool':pool,'target':target,'status':'Unavailable','reason':'Insufficient common/eligible training or test rows','train_rows':len(split['train']),'test_rows':len(split['test'])});continue
                        if reused:metrics,predictions,models,tone_status=retained10(repo,binding,split,target,tone,missing)
                        else:metrics,predictions,models,tone_status=fit_target(split,cfg,target,tone,missing)
                        costs=cost_evaluation(predictions,reg['assumed_roundtrip_cost_bps'])
                        costs['interval_audit']=interval_audit(predictions,costs)
                        dest=output/symbol/f'h{h}'/pool/target;dest.mkdir(parents=True)
                        report={'status':'Completed retrospective development comparison','company':symbol,'horizon':h,'pool':pool,'target':target,
                            'train_rows':len(split['train']),'test_rows':len(split['test']),'metrics':metrics,'tone_coverage':tone_status,'cost_evaluation':costs,
                            'retained_existing10_result':reused,'train_ids':[r['sample_id'] for r in split['train']],'test_ids':[r['sample_id'] for r in split['test']],
                            'strict_eligible':0,'alpha_established':False,'cash_dividends_included':False,'price_execution_time_certified':False}
                        save(dest/'report.json',report);save(dest/'predictions.json',predictions);save(dest/'fitted_models.json',models)
                        summary['evaluations'].append({**{k:report[k] for k in ['company','horizon','pool','target','train_rows','test_rows','metrics','tone_coverage','retained_existing10_result']},'status':'Completed','path':str(dest.relative_to(output)),'nonoverlap_intervals':costs['eligible_intervals']})
                        save(output/'summary.json',summary);print(json.dumps({'company':symbol,'horizon':h,'pool':pool,'target':target,'train':len(split['train']),'test':len(split['test']),'tone':tone_status['status'],'retained10':reused}),flush=True)
            save(output/'summary.json',summary)
        if sha(registration)!=registration_hash:raise ValueError('Registration changed during evaluation')
        if sha(repo/reg['tone_cache'])!=reg['tone_cache_sha256']:raise ValueError('Tone cache changed during evaluation')
        for binding in reg['company_bindings'].values():
            capture=repo/binding['capture']
            if sha(capture/'manifest.json')!=binding['capture_manifest_sha256'] or any(sha(capture/n)!=h for n,h in binding['input_files_sha256'].items()):raise ValueError('Capture changed during evaluation')
        for p in [Path(__file__),Path(build_rows.__code__.co_filename),Path(_connected_groups.__code__.co_filename)]:
            if sha(p)!=generation[p.name]:raise ValueError('Evaluation source changed')
        summary.update(status='complete',completed_at=stamp(),elapsed_wall_seconds=time.monotonic()-started,
            output_files_sha256={str(p.relative_to(output)):sha(p) for p in output.rglob('*.json') if p.name!='summary.json'})
    except Exception as exc:
        summary.update(status='failed',reason=str(exc),error_type=type(exc).__name__,completed_at=stamp());raise
    finally:save(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--registration',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();r=run_registered(a.repo,a.registration,a.output_dir);print(json.dumps({'status':r['status'],'evaluations':len(r['evaluations']),'seconds':r.get('elapsed_wall_seconds')}))
