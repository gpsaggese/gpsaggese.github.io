"""Registered Loughran–McDonald dictionary baseline, local academic research.

Current dictionary versions are not backdated. These additional development
controls do not fine-tune a language model or certify total-return/live alpha.
The dictionary stays in the ignored private run directory, outside product code.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time

from run_historical import save,price_class,cost_evaluation
from multi_horizon import read,sha,load_archive,named_target,interval_audit


def stamp():return datetime.now(timezone.utc).isoformat()


def load_dictionary(path):
    words=set();positive=set();negative=set();removed=Counter();newer=Counter()
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or not {'Word','Positive','Negative'}<=set(reader.fieldnames):raise ValueError('Missing LM dictionary columns')
        for row in reader:
            word=row['Word']
            if not word or not re.fullmatch('[A-Z]+',word) or word in words:raise ValueError('Invalid or duplicate dictionary word')
            words.add(word)
            for category,dest in [('Positive',positive),('Negative',negative)]:
                flag=int(row[category])
                if flag>0:
                    dest.add(word)
                    if flag>2023:newer[category]+=1
                elif flag<0:removed[category]+=1
    if not words or not positive or not negative:raise ValueError('Empty master/polarity dictionary')
    if positive&negative:raise ValueError('Positive/negative categories overlap')
    audit={'master_words':len(words),'active_positive':len(positive),'active_negative':len(negative),
        'negative_flag_removed_entries':dict(removed),'active_category_flags_after_2023':dict(newer),
        'dictionary_sha256':sha(path),'historical_asof_2023_verified':False,'commercial_deployment_authorized':False}
    return {'words':words,'positive':positive,'negative':negative},audit


def score_headline(text,lexicon):
    if not isinstance(text,str) or not text.strip():return {'status':'Unavailable','reason':'Missing or invalid headline'}
    tokens=[x.upper() for x in re.findall('[A-Za-z]+',text) if len(x)>=2]
    if not tokens:return {'status':'Unavailable','reason':'No retained ASCII word tokens'}
    pos=sum(x in lexicon['positive'] for x in tokens);neg=sum(x in lexicon['negative'] for x in tokens)
    return {'status':'Available','score':(pos-neg)/len(tokens),'tokens':len(tokens),'positive':pos,'negative':neg,
        'master_matches':sum(x in lexicon['words'] for x in tokens),
        'polarity_evidence':'MatchedPolarityWords' if pos+neg else 'NoPolarityMatch','input_sha256':hashlib.sha256(text.encode()).hexdigest()}


def score_company(rows,news,lexicon,cache):
    lookup={n['source_row_index']:n for n in news}
    if len(lookup)!=len(news):raise ValueError('Duplicate source row identity')
    features={};audits={};missing={}
    for row in rows:
        sample=row['sample_id'];titles=[]
        try:
            titles=list(dict.fromkeys(lookup[i]['headline'] for i in row['source_row_indices']))
            if not titles or any(not isinstance(t,str) for t in titles) or '\n'.join(titles)!=row['text']:raise ValueError('Original headline mapping/text mismatch')
        except (KeyError,TypeError,ValueError) as exc:
            missing[sample]=[str(exc)];audits[sample]={'status':'Unavailable','reasons':missing[sample]};continue
        values=[];hashes=[];bad=[]
        for text in titles:
            h=hashlib.sha256(text.encode()).hexdigest();hashes.append(h)
            if h not in cache:cache[h]=score_headline(text,lexicon)
            value=cache[h]
            if value['status']!='Available':bad.append({'input_sha256':h,'reason':value['reason']})
            else:values.append(value)
        if bad:
            missing[sample]=bad;audits[sample]={'status':'Unavailable','reasons':bad,'headline_hashes':hashes};continue
        score=sum(v['score'] for v in values)/len(values);features[sample]=score
        audits[sample]={'status':'Available','score':score,'headline_count':len(values),'headline_hashes':hashes,
            **{k:sum(v[k] for v in values) for k in ['tokens','positive','negative','master_matches']},
            'zero_polarity_headlines':sum(v['polarity_evidence']=='NoPolarityMatch' for v in values),
            'no_polarity_company_day':all(v['polarity_evidence']=='NoPolarityMatch' for v in values)}
    return features,audits,missing


def calibrate_lm(split,target,features,missing,configuration):
    required=split['train']+split['test']
    unavailable=[r['sample_id'] for r in required if r['sample_id'] not in features or r['sample_id'] in missing]
    status={'status':'Unavailable' if unavailable else 'Available','required_train':len(split['train']),'required_test':len(split['test']),
        'unavailable_sample_ids':unavailable,'selective_row_removal':False}
    if unavailable:return {},{}, {},status
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error,mean_squared_error,balanced_accuracy_score,f1_score
    if len(split['train'])<10 or len(split['test'])<5:raise ValueError('Insufficient registered train/test rows')
    y=np.asarray([named_target(r,target) for r in split['train']]);truth=np.asarray([named_target(r,target) for r in split['test']])
    band=configuration['class_neutral_band'];labels=[price_class(v,band) for v in truth]
    metrics={};predictions={r['sample_id']:{} for r in split['test']};fitted={}
    for name,price in [('lm_tone_ridge',False),('price_lm_ridge',True)]:
        def design(rows):return [(r['price_features'] if price else [])+[features[r['sample_id']]] for r in rows]
        scaler=StandardScaler().fit(design(split['train']))
        model=Ridge(alpha=configuration['ridge_alpha'],solver=configuration['solver'],tol=configuration['tol']).fit(scaler.transform(design(split['train'])),y)
        values=model.predict(scaler.transform(design(split['test'])))
        if not np.isfinite(values).all():raise ValueError('Nonfinite LM model prediction')
        classes=[price_class(v,band) for v in values]
        metrics[name]={'samples':len(truth),'mae':float(mean_absolute_error(truth,values)),
            'rmse':float(math.sqrt(mean_squared_error(truth,values))),'direction_accuracy':float(np.mean(np.sign(values)==np.sign(truth))),
            'price_class_balanced_accuracy':float(balanced_accuracy_score(labels,classes)),
            'price_class_macro_f1':float(f1_score(labels,classes,labels=[-1,0,1],average='macro',zero_division=0))}
        fitted[name]={'training_sample_ids':[r['sample_id'] for r in split['train']],
            'test_sample_ids':[r['sample_id'] for r in split['test']], 'means':scaler.mean_.tolist(),'scales':scaler.scale_.tolist(),
            'coefficients':model.coef_.tolist(),'intercept':float(model.intercept_),'alpha':configuration['ridge_alpha'],
            'target':target,'lexicon_fitted_on_test':False}
        for row,v in zip(split['test'],values):predictions[row['sample_id']][name]=float(v)
    return metrics,predictions,fitted,status


def case_split(prior,case):
    split=read(Path(prior)/case['company']/f"h{case['horizon']}"/'split.json')
    report=read(Path(prior)/case['path']/'report.json');result={}
    for part in ['train','test']:
        wanted=report[part+'_ids'];lookup={r['sample_id']:r for r in split[part]}
        if len(wanted)!=len(set(wanted)) or any(s not in lookup for s in wanted):raise ValueError('Case sample membership changed')
        result[part]=[lookup[s] for s in wanted]
    return result


def verify_inputs(repo,registration):
    prior=repo/registration['prior_evaluation'];summary=read(prior/'summary.json')
    if sha(prior/'summary.json')!=registration['prior_summary_sha256'] or summary['status']!='complete':raise ValueError('Prior comparison summary changed')
    for name,h in summary['output_files_sha256'].items():
        if sha(prior/name)!=h:raise ValueError('Prior comparison file changed: '+name)
    if sha(repo/registration['prior_registration'])!=registration['prior_registration_sha256']:raise ValueError('Prior registration changed')
    source=repo/registration['source']['local_root']
    for name,h in registration['source']['files_sha256'].items():
        if Path(name).name!=name or sha(source/name)!=h:raise ValueError('Dictionary/provenance source changed')
    if len(summary['evaluations'])!=36:raise ValueError('Prior case count changed')
    return prior,summary,source


def run_registered(repo,registration,output):
    repo,registration,output=Path(repo),Path(registration),Path(output)
    if output.exists() or output.is_symlink():raise FileExistsError('Choose a new private LM output directory')
    reg=read(registration);regsha=sha(registration);prior,old,source=verify_inputs(repo,reg)
    if reg['companies']!=['A','AMZN','MSFT'] or reg['horizons_sessions']!=[5,10,20] or reg['new_models']!=['lm_tone_ridge','price_lm_ridge']:raise ValueError('Registered comparison choices changed')
    output.mkdir(mode=0o700,parents=True);(output/'.gitignore').write_text('*\n');started=time.monotonic()
    summary={'status':'running','started_at':stamp(),'registration_sha256':regsha,'source_files_sha256':reg['source']['files_sha256'],
        'generation_sources_sha256':{p.name:sha(p) for p in [Path(__file__),Path(load_archive.__code__.co_filename),Path(cost_evaluation.__code__.co_filename)]},
        'companies':{},'cases':[],'strict_eligible':0,'alpha_established':False,'new_tensor_inferences':0,'network_calls_during_evaluation':0}
    save(output/'summary.json',summary)
    try:
        lexicon,dictionary_audit=load_dictionary(source/'Loughran-McDonald_MasterDictionary_1993-2025.csv')
        save(output/'dictionary_audit.json',dictionary_audit);summary['dictionary_audit']=dictionary_audit
        priorreg=read(repo/reg['prior_registration']);cache={};features={};missing={}
        for symbol in reg['companies']:
            b=priorreg['company_bindings'][symbol];news,_,_,_,_=load_archive(repo/b['capture'],b)
            rows=read(prior/symbol/'h10/company_days.json');features[symbol],audits,missing[symbol]=score_company(rows,news,lexicon,cache)
            dest=output/symbol;dest.mkdir();save(dest/'company_day_scores.json',audits)
            summary['companies'][symbol]={'company_days':len(rows),'available_scores':len(features[symbol]),'unavailable_scores':len(missing[symbol]),
                'source_headline_records':len(news),'headline_uses':sum(a.get('headline_count',0) for a in audits.values()),
                'zero_polarity_company_days':sum(a.get('no_polarity_company_day',False) for a in audits.values()),
                'zero_polarity_headline_uses':sum(a.get('zero_polarity_headlines',0) for a in audits.values()),
                **{k:sum(a.get(k,0) for a in audits.values()) for k in ['tokens','positive','negative','master_matches']}}
        save(output/'headline_scores.json',cache)
        summary['unique_headlines_scored']=len(cache);summary['unique_headlines_without_polarity']=sum(v.get('polarity_evidence')=='NoPolarityMatch' for v in cache.values())
        for case in old['evaluations']:
            if time.monotonic()-started>reg['resource_budget']['wall_seconds']:raise TimeoutError('Registered LM analysis budget exhausted')
            split=case_split(prior,case);symbol=case['company'];baseline=read(prior/case['path']/'report.json')
            original=read(prior/case['path']/'predictions.json');predictions=deepcopy(original)
            if [p['sample_id'] for p in original]!=[r['sample_id'] for r in split['test']]:raise ValueError('Prior prediction row order changed')
            metrics,added,fitted,status=calibrate_lm(split,case['target'],features[symbol],missing[symbol],reg['calibration'])
            for name,m in metrics.items():
                denominator=baseline['metrics']['train_mean']['mae'];m['mae_relative_to_train_mean']=m['mae']/denominator if denominator else None
            combined_metrics={**deepcopy(baseline['metrics']),**metrics}
            for p in predictions:p['predictions'].update(added.get(p['sample_id'],{}))
            costs=cost_evaluation(predictions,reg['assumed_roundtrip_cost_bps']);costs['interval_audit']=interval_audit(predictions,costs)
            if costs['selected_sample_ids']!=baseline['cost_evaluation']['selected_sample_ids']:raise ValueError('Cost selection changed')
            for name,values in baseline['cost_evaluation']['models'].items():
                if costs['models'][name]!=values:raise ValueError('Existing baseline cost result changed')
            for p,op in zip(predictions,original):
                if any(p['predictions'][name]!=value for name,value in op['predictions'].items()):raise ValueError('Existing prediction changed')
            pairs=[]
            for p in predictions:
                y=p['target_return'];losses={name:abs(value-y) for name,value in p['predictions'].items()}
                pairs.append({'sample_id':p['sample_id'],'absolute_losses':losses,
                    'lm_minus_existing_absolute_loss':{name:{control:losses[name]-losses[control] for control in baseline['metrics']} for name in metrics}})
            report={**deepcopy(baseline),'status':'Completed additional registered LM development comparison',
                'metrics':combined_metrics,'cost_evaluation':costs,'lm_coverage':status,'baseline_predictions_sha256':sha(prior/case['path']/'predictions.json'),
                'baseline_controls_refitted':False,'dictionary_asof_2023_verified':False,'transformer_fine_tuned':False}
            dest=output/case['path'];dest.mkdir(parents=True)
            save(dest/'report.json',report);save(dest/'predictions.json',predictions);save(dest/'fitted_lm_models.json',fitted);save(dest/'paired_losses.json',pairs)
            result={k:case[k] for k in ['company','horizon','pool','target','train_rows','test_rows','path']}
            result.update(lm_coverage=status,metrics=combined_metrics,nonoverlap_intervals=costs['eligible_intervals'])
            summary['cases'].append(result);save(output/'summary.json',summary)
            print(json.dumps({**{k:case[k] for k in ['company','horizon','pool','target']},'LM':status['status'],'train':len(split['train']),'test':len(split['test'])}),flush=True)
        verify_inputs(repo,reg)
        if sha(registration)!=regsha:raise ValueError('LM registration changed during comparison')
        for name,h in summary['generation_sources_sha256'].items():
            if sha(Path(__file__).parent/name)!=h:raise ValueError('Comparison source changed during evaluation')
        summary.update(status='complete',completed_at=stamp(),elapsed_wall_seconds=time.monotonic()-started,
            output_files_sha256={str(p.relative_to(output)):sha(p) for p in output.rglob('*.json') if p!=output/'summary.json'})
    except Exception as exc:
        summary.update(status='failed',reason=str(exc),error_type=type(exc).__name__,completed_at=stamp());raise
    finally:save(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--registration',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();result=run_registered(a.repo,a.registration,a.output_dir);print(json.dumps({'status':result['status'],'cases':len(result['cases']),'seconds':result.get('elapsed_wall_seconds')}))
