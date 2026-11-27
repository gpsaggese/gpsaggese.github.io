"""Artificial fixtures for registered horizon analysis; no market downloads."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from multi_horizon import (session_date,common_splits,tone_features,fit_target,
    named_target,load_archive,sha,run_registered,interval_audit)
from run_historical import build_rows,retrospective_split,cost_evaluation

CFG={'class_neutral_band':.005,'ridge_alpha':10.,'tfidf':{'min_df':2,'max_features':2000,'ngram_range':[1,2]}}

def split_fixture():
    rows=[]
    for i in range(40):
        raw=(i%7-2)*.01;spy=.003*(i%3+1)
        rows.append({'sample_id':str(i),'text':'company reports shared earnings '+('rising growth' if i%2 else 'falling sales'),
            'entry_session_index':i,'exit_session_index':i+5,'entry_at':f'2023-08-{i%28+1:02}T13:30:00Z',
            'exit_at':f'2023-09-{i%28+1:02}T13:30:00Z','stock_return':raw,'spy_return':spy,
            'excess_return':raw-spy,'price_class':1 if raw-spy>.005 else -1 if raw-spy<-.005 else 0,
            'price_features':[i*.001,(i%3)*.01],'source_ids':['source'+str(i)]})
    return {'train':rows[:30],'test':rows[30:],'excluded':[]}

def archive_fixture(path):
    def put(name,v):(path/name).write_text(json.dumps(v))
    put('calendar.json',{'sessions':[{'date':'2023-03-13','open_at':'2023-03-13T13:30:00Z'}]})
    put('news.json',[]);put('protocol.json',{'symbol':'A'})
    put('prices_000.json',{'bars':{s:[{'t':'2023-03-13T04:00:00Z','o':100.}] for s in ['A','SPY']},'next_page_token':None})
    inputs={n:sha(path/n) for n in ['calendar.json','news.json','protocol.json','prices_000.json']}
    m={'status':'complete_bounded_capture','files_sha256':inputs,'bars_summary':{'pagination_complete':True,'records':{'A':1,'SPY':1}},
       'requests':[{'file':'prices_000.json','status':200,'params':{'timeframe':'1Day','adjustment':'split','feed':'sip'},'next_page_token':None}]}
    put('manifest.json',m)
    return {'capture_manifest_sha256':sha(path/'manifest.json'),'input_files_sha256':inputs}


class HorizonTests(unittest.TestCase):
    def test_daily_anchor_follows_dst_not_scheduled_open(self):
        self.assertEqual(session_date('2023-03-10T05:00:00Z',{'2023-03-10'}),'2023-03-10')
        self.assertEqual(session_date('2023-03-13T04:00:00Z',{'2023-03-13'}),'2023-03-13')

    def test_invalid_daily_timestamps_are_rejected(self):
        for t in ['2023-03-13T00:00:00','2023-03-13T13:30:00Z','2023-03-13T05:00:00Z']:
            with self.subTest(t=t),self.assertRaises(ValueError):session_date(t,{'2023-03-13'})
        with self.assertRaises(ValueError):session_date('2023-03-13T04:00:00Z',{'2023-03-14'})

    def test_archive_audits_anchor_without_claiming_execution(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);binding=archive_fixture(path);_,prices,_,_,audit=load_archive(path,binding)
            self.assertEqual(prices['A','2023-03-13'],100)
            self.assertEqual(audit['missing_calendar_dates'],{'A':[],'SPY':[]})
            self.assertFalse(audit['official_auction_or_fill_certified'])

    def test_archive_rejects_changed_price_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);binding=archive_fixture(path)
            (path/'prices_000.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash'):load_archive(path,binding)

    def test_common_pool_intersects_partition_ids_only(self):
        base=split_fixture();a=deepcopy(base);b=deepcopy(base);a['train']=a['train'][1:];b['train']=b['train'][:-2]
        for r in b['train']+b['test']:r['stock_return']=999
        common=common_splits({5:a,10:base,20:b})
        self.assertEqual([r['sample_id'] for r in common[5]['train']],[str(i) for i in range(1,28)])
        self.assertEqual([r['sample_id'] for r in common[20]['test']],[str(i) for i in range(30,40)])
        self.assertEqual(len(base['train']),30)

    def test_raw_target_never_publishes_disguised_excess(self):
        split=split_fixture();tone={r['sample_id']:.1 for r in split['train']+split['test']}
        metrics,pred,models,status=fit_target(split,CFG,'raw_price_return',tone,{})
        expected=sum(r['stock_return'] for r in split['train'])/30
        self.assertAlmostEqual(models['models']['train_mean']['intercept'],expected)
        for p,r in zip(pred,split['test']):
            self.assertEqual(p['target_return'],r['stock_return']);self.assertEqual(p['excess_return'],r['excess_return'])
            self.assertEqual(p['spy_return'],r['spy_return']);self.assertEqual(p['target_name'],'raw_price_return')
        self.assertEqual(metrics['train_mean']['mae_relative_to_train_mean'],1.)
        self.assertEqual(status['status'],'Available');self.assertEqual(len(metrics),6)

    def test_test_outcomes_never_fit_transforms_or_parameters(self):
        a=split_fixture();b=deepcopy(a);tone={r['sample_id']:.1 for r in a['train']+a['test']}
        for r in b['test']:r['stock_return']+=.9;r['excess_return']+=.9
        _,pa,ma,_=fit_target(a,CFG,'raw_price_return',tone,{})
        _,pb,mb,_=fit_target(b,CFG,'raw_price_return',tone,{})
        self.assertEqual(ma,mb);self.assertEqual([p['predictions'] for p in pa],[p['predictions'] for p in pb])

    def test_missing_tone_disables_whole_added_controls_without_row_drop(self):
        split=split_fixture();tone={r['sample_id']:.1 for r in split['train']+split['test'] if r['sample_id']!='0'}
        metrics,pred,_,status=fit_target(split,CFG,'excess_price_return',tone,{'0':['missing']})
        self.assertEqual(len(metrics),4);self.assertEqual(len(pred),10)
        self.assertEqual(status['required_rows'],40);self.assertEqual(status['missing_rows'],['0'])
        self.assertFalse(status['selective_row_removal']);self.assertEqual(status['status'],'Unavailable')

    def test_tone_cache_uses_exact_headline_inputs_and_marks_missing(self):
        titles=['first title','second title'];h=hashlib.sha256(titles[0].encode()).hexdigest()
        news=[{'source_row_index':i,'headline':t} for i,t in enumerate(titles)]
        rows=[{'sample_id':'x','source_row_indices':[0,1],'text':'\n'.join(titles)}]
        features,missing=tone_features(rows,news,{h:{'input_sha256':h,'status':'Available','truncated':False,'scores':{'positive':.7,'neutral':.2,'negative':.1}}})
        self.assertEqual(features,{});self.assertEqual(len(missing['x']),1)
        rows[0]['text']='changed'
        with self.assertRaisesRegex(ValueError,'Headline'):tone_features(rows,news,{})

    def test_longer_horizon_requires_earlier_maturity(self):
        # Deliberately artificial daily calendar, never presented as XNYS data.
        start=datetime(2023,7,1,tzinfo=timezone.utc)
        sessions=[{'date':(start+timedelta(days=i)).date().isoformat(),'open_at':(start+timedelta(days=i,hours=13,minutes=30)).isoformat()} for i in range(100)]
        news=[{'source_date':s['date'],'headline':f'unique item {i}','source_row_index':i,'url':f'https://example.test/{i}'} for i,s in enumerate(sessions[25:65])]
        prices={(symbol,s['date']):100+i for symbol in ['A','SPY'] for i,s in enumerate(sessions)}
        splits={}
        for h in [5,10,20]:
            rows,_=build_rows(news,prices,{'sessions':sessions},{**CFG,'symbol':'A','horizon_sessions':h,'entry_delay_sessions':2})
            splits[h]=retrospective_split(rows,'2023-09-01')
            self.assertTrue(all(r['exit_at']<'2023-09-01T00:00:00Z' for r in splits[h]['train']))
        self.assertGreater(len(splits[5]['train']),len(splits[10]['train']))
        self.assertGreater(len(splits[10]['train']),len(splits[20]['train']))

    def test_cost_intervals_never_compound_overlapping_rows(self):
        split=split_fixture();tone={r['sample_id']:.1 for r in split['train']+split['test']}
        _,pred,_,_=fit_target(split,CFG,'raw_price_return',tone,{})
        costs=cost_evaluation(pred,[0,10,20]);audit=interval_audit(pred,costs)
        self.assertEqual(costs['eligible_intervals'],2);self.assertEqual(audit['total_available_held_sessions'],10)
        self.assertEqual(audit['active_held_sessions']['cash'],0)
        intervals=audit['selected_intervals'];self.assertGreaterEqual(intervals[1]['entry_session_index'],intervals[0]['exit_session_index'])
        self.assertLess(costs['models']['always_company'][2]['cumulative_simulated_price_return'],costs['models']['always_company'][0]['cumulative_simulated_price_return'])
        self.assertIsNone(costs['actual_broker_charges'])

    def test_existing_outputs_and_invalid_target_are_rejected(self):
        with self.assertRaises(ValueError):named_target({},'total_return')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reg=root/'reg';reg.write_text('{}');out=root/'out';out.mkdir();(out/'keep').write_text('untouched')
            with self.assertRaises(FileExistsError):run_registered(root,reg,out)
            self.assertEqual((out/'keep').read_text(),'untouched')


if __name__=='__main__':unittest.main()
