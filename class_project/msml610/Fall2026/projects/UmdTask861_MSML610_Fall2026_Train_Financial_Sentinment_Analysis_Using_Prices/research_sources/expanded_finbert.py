"""Registered same-row multi-company FinBERT inference with a finite CPU budget.

Inference reads text-only packets; evaluation remains in finbert_comparison.py.
The shared cache is a within-run optimization of the same frozen text model,
never a fitted cross-company price model. A failed required headline makes the
whole company's added comparison unavailable; existing baseline rows stay intact.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from finbert_comparison import MODEL_ID, REVISION, infer_packet, prepare_packet, save_new, sha


def read(p): return json.loads(Path(p).read_text())
def stamp(): return datetime.now(timezone.utc).isoformat()
def write(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def object_sha(x): return sha(json.dumps(x,separators=(',',':'),ensure_ascii=False).encode())


def prepare_registered(repo,registration,output):
    repo,registration=Path(repo),Path(registration);reg=read(registration)
    packets={}
    for symbol in reg['companies']:
        binding=reg['company_bindings'][symbol]
        base,capture=repo/binding['base'],repo/binding['capture']
        if sha((base/'split.json').read_bytes())!=binding['split_sha256'] or sha((capture/'news.json').read_bytes())!=binding['news_sha256'] or sha((base/'predictions.json').read_bytes())!=binding['base_predictions_sha256']:
            raise ValueError('Registered company data changed: '+symbol)
        packet=prepare_packet(capture,base,registration)
        ids=[r['sample_id'] for r in packet['rows']]
        if object_sha(ids)!=binding['ordered_sample_ids_sha256'] or len(ids)!=binding['train_rows']+binding['test_rows']:
            raise ValueError('Registered paired sample order changed: '+symbol)
        if len({h for r in packet['rows'] for h in r['headline_hashes']})!=binding['unique_headlines'] or sum(len(r['headlines']) for r in packet['rows'])!=binding['headline_uses']:
            raise ValueError('Registered headline budget changed: '+symbol)
        packets[symbol]=packet
    output=Path(output)
    preparation={'registered_sha256':sha(registration.read_bytes()),'companies':reg['companies'],'rows':sum(len(p['rows']) for p in packets.values())}
    save_new(output,{'preparation.json':preparation})
    for symbol,packet in packets.items():save_new(output/symbol,{'tone_request.json':packet})
    preparation['packet_files_sha256']={symbol:sha((output/symbol/'tone_request.json').read_bytes()) for symbol in reg['companies']}
    write(output/'preparation.json',preparation)
    return packets


class InferenceBudgetExceeded(TimeoutError): pass


class BoundedToneCache:
    def __init__(self,model,*,seconds,batch_size,checkpoint,clock=time.monotonic,started=None):
        self.model=model;self.metadata=model.metadata;self.seconds=seconds;self.batch_size=batch_size
        self.checkpoint=checkpoint;self.clock=clock;self.started=clock() if started is None else started
        self.cache={};self.batches=0;self.inferred=0

    def save(self):
        self.checkpoint({'model':self.metadata,'headline_results':self.cache,'actual_unique_inferences':self.inferred,
                         'elapsed_wall_seconds':self.clock()-self.started,'checkpointed_at':stamp()})

    def predict(self,texts):
        missing=list(dict.fromkeys(t for t in texts if sha(t.encode()) not in self.cache))
        for offset in range(0,len(missing),self.batch_size):
            if self.clock()-self.started>=self.seconds:
                self.save();raise InferenceBudgetExceeded('Registered shared inference wall budget exhausted')
            batch=missing[offset:offset+self.batch_size];results=self.model.predict(batch)
            if len(results)!=len(batch):raise ValueError('Missing batch results')
            for text,result in zip(batch,results):
                if result.get('input_sha256')!=sha(text.encode()):raise ValueError('Model receipt input mismatch')
                self.cache[sha(text.encode())]=result
            self.inferred+=len(batch);self.batches+=1
            if self.batches%4==0:
                self.save();print(json.dumps({'unique_headlines_inferred':self.inferred,'elapsed_seconds':round(self.clock()-self.started,1)}),flush=True)
        self.save()
        return [self.cache[sha(t.encode())] for t in texts]


def infer_registered(registration,packets,output,*,model=None,clock=time.monotonic):
    registration,packets,output=Path(registration),Path(packets),Path(output)
    reg=read(registration);registration_hash=sha(registration.read_bytes());budget=reg['resource_budget']
    started=clock();save_new(output,{})
    m={'status':'running','started_at':stamp(),'registration_sha256':registration_hash,
       'generation_source_sha256':sha(Path(__file__).read_bytes()),'companies':{},'resource_budget':budget,
       'comparison_scope':reg['study'],'rows_selectively_removed':0,'network_used':False}
    def checkpoint(obj):
        tmp=output/'partial_headline_receipts.tmp';write(tmp,obj);tmp.replace(output/'partial_headline_receipts.json')
    write(output/'inference_manifest.json',m)
    try:
        preparation=read(packets/'preparation.json')
        if preparation['registered_sha256']!=registration_hash:raise ValueError('Packet preparation registration changed')
        if model is None:
            from sentiment_model import FinBertTone
            policy=reg['finbert'];snapshot=Path(policy['cached_snapshot'])
            for name,key in [('onnx/model_quantized.onnx','model_sha256'),('tokenizer.json','tokenizer_sha256'),('config.json','config_sha256')]:
                if sha((snapshot/name).read_bytes())!=policy[key]:raise ValueError('Pinned local model file changed')
            model=FinBertTone(offline=True,cached_snapshot=snapshot)
        if model.metadata['model_id']!=MODEL_ID or model.metadata['revision']!=REVISION:raise ValueError('Unregistered model')
        cache=BoundedToneCache(model,seconds=budget['inference_wall_seconds'],batch_size=budget['batch_headlines'],checkpoint=checkpoint,clock=clock,started=started)
        for symbol in reg['companies']:
            binding=reg['company_bindings'][symbol];packet_path=packets/symbol/'tone_request.json'
            coverage={'required_train_rows':binding['train_rows'],'required_test_rows':binding['test_rows'],
                      'required_unique_headlines':binding['unique_headlines'],'required_headline_uses':binding['headline_uses']}
            try:
                packet=read(packet_path)
                if sha(packet_path.read_bytes())!=preparation['packet_files_sha256'][symbol]:
                    raise ValueError('Prepared packet bytes changed')
                if packet['registration_sha256']!=registration_hash or packet['source_split_sha256']!=binding['split_sha256'] or packet['source_news_sha256']!=binding['news_sha256']:
                    raise ValueError('Prepared packet binding changed')
                if object_sha([r['sample_id'] for r in packet['rows']])!=binding['ordered_sample_ids_sha256']:
                    raise ValueError('Prepared sample order changed')
                result=infer_packet(packet,classifier=cache)
                result['request_file_sha256']=sha(packet_path.read_bytes())
                result['within_run_shared_cache']=True
                save_new(output/symbol,{'tone_features.json':result})
                coverage.update(status='Available',any_truncated=False,provided_rows=len(result['samples']),
                                unique_headlines=result['unique_headlines'],headline_uses=result['headline_uses'],
                                features_sha256=sha((output/symbol/'tone_features.json').read_bytes()))
            except (ValueError,TimeoutError,RuntimeError,KeyError,FileNotFoundError) as exc:
                cache.save();coverage.update(status='Unavailable',reason=str(exc),error_type=type(exc).__name__,
                                            provided_rows=0,baseline_rows_retained=True)
            m['companies'][symbol]=coverage;write(output/'inference_manifest.json',m)
            print(json.dumps({'company':symbol,**coverage,'elapsed_seconds':round(clock()-started,1)}),flush=True)
        if sha(registration.read_bytes())!=registration_hash or sha(Path(__file__).read_bytes())!=m['generation_source_sha256']:
            raise ValueError('Registration or inference source changed during run')
        m.update(status='complete' if all(v['status']=='Available' for v in m['companies'].values()) else 'complete_with_unavailable_comparisons',
                 completed_at=stamp(),elapsed_wall_seconds=clock()-started,actual_unique_inferences=cache.inferred,
                 model=model.metadata)
    except Exception as exc:
        m.update(status='failed',error_type=type(exc).__name__,reason=str(exc),completed_at=stamp())
        raise
    finally:write(output/'inference_manifest.json',m)
    return m


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    prep=s.add_parser('prepare');prep.add_argument('--repo',type=Path,required=True)
    inf=s.add_parser('infer')
    for sub in [prep,inf]:sub.add_argument('--registration',type=Path,required=True);sub.add_argument('--output-dir',type=Path,required=True)
    inf.add_argument('--packets',type=Path,required=True)
    a=p.parse_args(argv)
    if a.action=='prepare':
        r=prepare_registered(a.repo,a.registration,a.output_dir);print(json.dumps({'prepared_companies':list(r),'rows':sum(len(x['rows']) for x in r.values())}))
    else:
        r=infer_registered(a.registration,a.packets,a.output_dir);print(json.dumps({'status':r['status'],'elapsed_seconds':r.get('elapsed_wall_seconds')}))
    return 0


if __name__=='__main__':raise SystemExit(main())
