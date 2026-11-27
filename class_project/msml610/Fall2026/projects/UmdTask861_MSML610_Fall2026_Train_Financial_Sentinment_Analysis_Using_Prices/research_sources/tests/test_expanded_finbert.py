"""Finite cache/coverage contracts with artificial text; no real inference."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from expanded_finbert import BoundedToneCache,InferenceBudgetExceeded,infer_registered,object_sha,sha,MODEL_ID,REVISION


class FakeTone:
    metadata={'model_id':MODEL_ID,'revision':REVISION,'synthetic_fixture':True}
    def __init__(self,truncated=False):self.calls=[];self.truncated=truncated
    def predict(self,texts):
        self.calls.append(list(texts))
        return [{'input_sha256':sha(t.encode()),'status':'Available','truncated':self.truncated and t=='bad',
                 'scores':{'positive':.7,'neutral':.2,'negative':.1}} for t in texts]


def fixture(root):
    packets=root/'packets';packets.mkdir();reg=root/'reg.json'
    bindings={};data={}
    for symbol in ['A','AMZN','MSFT']:
        rows=[{'sample_id':symbol+str(i),'headlines':[t],'headline_hashes':[sha(t.encode())]}
              for i,t in enumerate(['common', 'bad' if symbol=='MSFT' else symbol+' exclusive'])]
        bindings[symbol]={'train_rows':1,'test_rows':1,'unique_headlines':2,'headline_uses':2,
                          'split_sha256':'split-'+symbol,'news_sha256':'news-'+symbol,
                          'ordered_sample_ids_sha256':object_sha([r['sample_id'] for r in rows])}
        data[symbol]={'model_id':MODEL_ID,'revision':REVISION,'rows':rows,'source_split_sha256':'split-'+symbol,
                      'source_news_sha256':'news-'+symbol}
    reg.write_text(json.dumps({'companies':['A','AMZN','MSFT'],'company_bindings':bindings,'study':'Synthetic fixture',
                  'resource_budget':{'inference_wall_seconds':30,'batch_headlines':2}}))
    for symbol,p in data.items():
        p['registration_sha256']=sha(reg.read_bytes());(packets/symbol).mkdir()
        (packets/symbol/'tone_request.json').write_text(json.dumps(p))
    (packets/'preparation.json').write_text(json.dumps({'registered_sha256':sha(reg.read_bytes()),
        'packet_files_sha256':{s:sha((packets/s/'tone_request.json').read_bytes()) for s in data}}))
    return reg,packets


class ExpandedTests(unittest.TestCase):
    def test_cache_deduplicates_without_changing_input_order(self):
        model=FakeTone();saved=[];c=BoundedToneCache(model,seconds=10,batch_size=2,checkpoint=saved.append,clock=lambda:0)
        a=c.predict(['same','other']);b=c.predict(['other','same'])
        self.assertEqual(model.calls,[['same','other']]);self.assertEqual(a,list(reversed(b)))
        self.assertEqual(c.inferred,2);self.assertTrue(saved)

    def test_budget_is_checked_before_more_compute(self):
        model=FakeTone();c=BoundedToneCache(model,seconds=1,batch_size=2,checkpoint=lambda _:None,clock=lambda:2,started=0)
        with self.assertRaises(InferenceBudgetExceeded):c.predict(['new'])
        self.assertEqual(model.calls,[])

    def test_missing_result_is_not_filled_or_dropped(self):
        class Missing(FakeTone):
            def predict(self,texts):return []
        c=BoundedToneCache(Missing(),seconds=10,batch_size=2,checkpoint=lambda _:None,clock=lambda:0)
        with self.assertRaisesRegex(ValueError,'Missing'):c.predict(['one'])

    def test_three_company_success_keeps_all_required_rows(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reg,packets=fixture(root);model=FakeTone()
            m=infer_registered(reg,packets,root/'out',model=model,clock=lambda:0)
            self.assertEqual(m['status'],'complete');self.assertEqual(m['actual_unique_inferences'],4)
            self.assertTrue(all(v['provided_rows']==2 for v in m['companies'].values()))
            self.assertEqual(m['rows_selectively_removed'],0)

    def test_one_bad_headline_makes_whole_company_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reg,packets=fixture(root)
            m=infer_registered(reg,packets,root/'out',model=FakeTone(truncated=True),clock=lambda:0)
            failed=m['companies']['MSFT']
            self.assertEqual(failed['status'],'Unavailable');self.assertEqual(failed['required_test_rows'],1)
            self.assertEqual(failed['provided_rows'],0);self.assertTrue(failed['baseline_rows_retained'])
            self.assertFalse((root/'out/MSFT/tone_features.json').exists())
            self.assertEqual(m['companies']['A']['status'],'Available')
            self.assertTrue((root/'out/partial_headline_receipts.json').is_file())

    def test_changed_packet_bytes_fail_only_added_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reg,packets=fixture(root);p=packets/'A/tone_request.json';p.write_text(p.read_text()+' ')
            m=infer_registered(reg,packets,root/'out',model=FakeTone(),clock=lambda:0)
            self.assertEqual(m['companies']['A']['status'],'Unavailable')
            self.assertIn('bytes changed',m['companies']['A']['reason'])
            self.assertEqual(m['companies']['AMZN']['status'],'Available')

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reg,packets=fixture(root);out=root/'out';out.mkdir();(out/'keep').write_text('keep')
            with self.assertRaises(FileExistsError):infer_registered(reg,packets,out,model=FakeTone(),clock=lambda:0)
            self.assertEqual((out/'keep').read_text(),'keep')


if __name__=='__main__':unittest.main()
