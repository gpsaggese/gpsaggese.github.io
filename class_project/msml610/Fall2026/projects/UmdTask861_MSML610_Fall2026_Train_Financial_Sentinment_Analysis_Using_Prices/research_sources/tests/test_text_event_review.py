"""Artificial review/split contracts; never market labels or alpha evidence."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from text_event_review import apply_review,split_gate_inputs
from prepare_text_candidates import prepare_candidates
from price_labels import make_time_split,build_labels
from research.financial_sentiment_using_prices.tests.test_price_labels import dataset,observation,DAYS


def fixture():
    def article(identity,symbols,body):
        return {'id':identity,'symbols':symbols,'headline':'Artificial headline '+identity,'text':body,
                'created_at':'2026-10-01T10:00:00Z','updated_at':'2026-10-01T10:01:00Z',
                'retrieved_at':'2026-10-01T10:02:00Z','url':'https://example.test/'+identity}
    articles=[article('a',['PLTR'],'Palantir events G and H'),article('b',['MSFT'],'Microsoft event G'),
              article('c',['PLTR','MSFT'],'Palantir Microsoft context C')]
    candidates=prepare_candidates(articles,dataset()['calendar'],symbols=['PLTR','MSFT'],
                                  decision_session=DAYS[1],as_of='2026-10-01T23:00:00Z')
    digest=hashlib.sha256(json.dumps(candidates,sort_keys=True).encode()).hexdigest()
    groups=[{'group_id':g,'kind':k,'description':'Artificial '+g,'independent_event_verified':False}
            for g,k in [('event:G','reported_event'),('event:H','reported_event'),('context:C','conservative_context')]]
    decisions=[]
    for row in candidates['candidates']:
        for a in row['texts']:
            events=['event:G','event:H'] if a['article_id']=='a' else ['event:G'] if a['article_id']=='b' else []
            contexts=['context:C'] if a['article_id']=='c' else []
            span={'field':'body','start':0,'end':8,'span_sha256':hashlib.sha256(a['body'][:8].encode()).hexdigest()}
            decisions.append({'article_id':a['article_id'],'symbol':row['symbol'],'retained_text_sha256':a['retained_text_sha256'],
                'relevance':'contextual' if contexts else 'direct','review_basis':'body_and_headline','rationale':'Artificial source review',
                'evidence_spans':[span],'reported_event_ids':events,'association_group_ids':contexts,
                'group_evidence':[{'group_id':g,'evidence_spans':[span]} for g in events+contexts]})
    review={'schema_version':'financial-text-event-review-1.0','candidate_sha256':digest,
            'review_created_at':'2026-10-02T14:00:00Z','reviewer':{'kind':'assistant','independent_human_review':False},
            'groups':groups,'decisions':decisions}
    return candidates,review,digest


class TextEventReviewTests(unittest.TestCase):
    def apply(self,c=None,r=None,digest=None,**changes):
        base,review,h=fixture();args={'candidate_sha256':digest or h,'as_of':'2026-10-02T15:00:00Z'};args.update(changes)
        return apply_review(c or base,r or review,**args)

    def test_full_texts_sources_and_original_input_clocks_are_preserved(self):
        c,r,h=fixture();before=deepcopy(c);out=self.apply(c,r,h)
        self.assertEqual(c,before)
        for original,row in zip(c['candidates'],out['candidates']):
            for key in ['texts','source_ids','input_snapshot_created_at','input_frozen_before_cutoff','prediction_completed_at','prediction']:
                self.assertEqual(row[key],original[key])
            self.assertIn('review_selection_after_input_cutoff',row['blockers'])
            self.assertIn('review_after_prediction_deadline',row['blockers'])
            self.assertFalse(row['review_available_before_prediction_deadline'])
            self.assertFalse(row['eligible_for_training'])

    def test_same_event_different_articles_keeps_shared_event_not_shared_fake_source(self):
        out=self.apply();a,b=out['candidates']
        self.assertNotEqual(a['proposed_direct_source_ids'],b['proposed_direct_source_ids'])
        self.assertIn('event:G',a['event_cluster_ids']);self.assertIn('event:G',b['event_cluster_ids'])
        self.assertIn('event:H',a['event_cluster_ids'])
        self.assertIn('context:C',a['event_cluster_ids'])
        self.assertEqual(out['review_summary']['multi_article_reported_groups'],1)

    def test_context_sources_remain_in_gate_pool_even_when_not_selected_direct(self):
        out=self.apply();stubs=split_gate_inputs(out)
        self.assertIn('alpaca-news:c',stubs[0]['source_ids']);self.assertIn('alpaca-news:c',stubs[1]['source_ids'])
        self.assertNotIn('alpaca-news:c',out['candidates'][0]['proposed_direct_source_ids'])
        self.assertTrue(all(not row['is_label'] for row in stubs))
        split=make_time_split(stubs,'2026-10-01T08:00:00Z','2026-10-02T13:30:00Z','2026-10-05T13:30:00Z')
        self.assertEqual(split['train'],[]);self.assertEqual(split['test'],[]);self.assertEqual(len(split['excluded']),2)

    def test_review_bound_to_exact_candidate_version(self):
        with self.assertRaisesRegex(ValueError,'bound'):self.apply(digest='0'*64)
        c,r,h=fixture();r['decisions'][0]['retained_text_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'version'):self.apply(c,r,h)

    def test_bad_evidence_coordinates_or_hash_are_rejected(self):
        for change in [{'end':10000},{'span_sha256':'0'*64},{'field':'future_return'}]:
            c,r,h=fixture();r['decisions'][0]['evidence_spans'][0].update(change)
            with self.assertRaisesRegex(ValueError,'evidence'):self.apply(c,r,h)

    def test_every_pair_has_exactly_one_disposition(self):
        for mode in ['missing','duplicate','extra']:
            c,r,h=fixture()
            if mode=='missing':r['decisions'].pop()
            elif mode=='duplicate':r['decisions'].append(deepcopy(r['decisions'][0]))
            else:r['decisions'].append({**r['decisions'][0],'article_id':'not-in-corpus'})
            with self.assertRaises(ValueError):self.apply(c,r,h)

    def test_no_backdating_review_or_independent_human_claim(self):
        with self.assertRaisesRegex(ValueError,'backdate'):self.apply(as_of='2026-10-01T23:00:00Z')
        c,r,h=fixture();r['reviewer']['independent_human_review']=True
        with self.assertRaisesRegex(ValueError,'independent human'):self.apply(c,r,h)
        c,r,h=fixture();r['review_created_at']='2026-10-01T22:00:00Z'
        with self.assertRaisesRegex(ValueError,'precede'):self.apply(c,r,h)

    def test_undefined_or_unsupported_context_group_is_rejected(self):
        c,r,h=fixture();r['decisions'][0]['reported_event_ids']=['undefined'];r['decisions'][0]['group_evidence'][0]['group_id']='undefined'
        r['decisions'][0]['group_evidence']=r['decisions'][0]['group_evidence'][:1]
        with self.assertRaisesRegex(ValueError,'Undefined'):self.apply(c,r,h)
        c,r,h=fixture();r['groups'][0]['kind']='conservative_context'
        with self.assertRaisesRegex(ValueError,'Context association'):self.apply(c,r,h)

    def test_body_missing_cannot_be_marked_as_body_review(self):
        c,r,h=fixture();r['decisions'][0]['review_basis']='headline_only'
        with self.assertRaisesRegex(ValueError,'body'):self.apply(c,r,h)

    def test_each_group_link_requires_evidence(self):
        c,r,h=fixture();r['decisions'][0]['group_evidence'].pop()
        with self.assertRaisesRegex(ValueError,'Every association'):self.apply(c,r,h)

    def test_direct_and_uncertain_dispositions_cannot_invent_event_links(self):
        c,r,h=fixture();r['decisions'][0]['reported_event_ids']=[];r['decisions'][0]['group_evidence']=[]
        with self.assertRaisesRegex(ValueError,'Direct'):self.apply(c,r,h)
        c,r,h=fixture();r['decisions'][0]['relevance']='uncertain'
        with self.assertRaisesRegex(ValueError,'invent events'):self.apply(c,r,h)

    def test_existing_purger_removes_shared_events_with_distinct_article_ids(self):
        out=self.apply();labels=build_labels(dataset([observation(0),observation(13)]))
        labels[0]['source_ids']=['artificial:a'];labels[1]['source_ids']=['artificial:b']
        labels[0]['event_cluster_ids']=out['candidates'][0]['source_event_ids']
        labels[1]['event_cluster_ids']=out['candidates'][1]['source_event_ids']
        split=make_time_split(labels,'2026-10-16T12:00:00Z','2026-10-20T13:30:00Z','2026-10-21T13:30:00Z')
        self.assertEqual(split['train'],[]);self.assertEqual(len(split['test']),1)
        self.assertIn('event_cluster_shared_with_test',split['excluded'][0]['reasons'])

    def test_multievent_article_bridge_is_retained_transitively(self):
        out=self.apply();bridge_groups=out['candidates'][0]['source_event_ids']
        labels=build_labels(dataset([observation(0),observation(13)]));labels[0]['source_ids']=['artificial:a'];labels[1]['source_ids']=['artificial:b']
        labels[0]['event_cluster_ids']=['event:G'];labels[1]['event_cluster_ids']=['event:H']
        bridge=deepcopy(labels[0]);bridge.update(sample_id='artificial:omitted-bridge',status='Unavailable',label_status='Unavailable',
                    return_basis=None,source_ids=['artificial:bridge'],event_cluster_ids=bridge_groups)
        split=make_time_split([*labels,bridge],'2026-10-16T12:00:00Z','2026-10-20T13:30:00Z','2026-10-21T13:30:00Z')
        self.assertEqual(split['train'],[])
        self.assertTrue(any('transitive_group_shared_with_test' in row['reasons'] for row in split['excluded']))


if __name__=='__main__':unittest.main()
