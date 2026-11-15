"""Stdlib fixtures/mocks only; no optimizer, torch, market predictions or IO."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grouped_collation import collate_training, chunks
from daily_execution import build_manifest, validate_manifest, mock_replay


def fixture(batch_size=4):
    companies=['A','AMZN','MSFT'];training={};features={};partition={};batches={}
    for symbol,counts in zip(companies,[[1,3,9,2,10],[2,1,8,11,1,4],[1,2,9]]):
        rows=[];tokens=[]
        for day,n in enumerate(counts):
            sid=symbol+':'+str(day);titles=[sid+'/'+str(i) for i in range(n)]
            rows.append({'sample_id':sid,'headlines':titles,'target':(day-2)*.01})
            tokens.append({'sample_id':sid,'headlines':[{
                'input_sha256':hashlib.sha256(t.encode()).hexdigest(),
                'input_ids':[101,10+i,102], 'attention_mask':[1]*3,'token_type_ids':[0]*3,
                'length':3,'untruncated_length':3,'truncated':False,'unknown_tokens':0}
                for i,t in enumerate(titles)]})
        ids=[r['sample_id'] for r in rows];heldout=[symbol+':heldout']
        training[symbol]=rows;features[symbol]=tokens
        partition[symbol]={'train_ids':ids,'test_ids':heldout}
        batches[symbol]=[collate_training(group,{r['sample_id']:r['target'] for r in rows},ids,heldout)
            for group in chunks(tokens,batch_size)]
    reg={'companies':companies,'horizon_sessions':10,'target':'excess_price_return','pool':'horizon_specific',
        'inputs':{'headline_chunk_size':8,'max_tokens':64},
        'training':{'epochs':1,'seed':610,'gradient_clip_norm':1.,'max_total_steps':14}}
    return reg,training,batches,partition


class DailyExecutionTests(unittest.TestCase):
    def setUp(self):self.inputs=fixture();self.plan=build_manifest(*self.inputs)

    def test_identical_manifest_bytes_and_no_source_mutation(self):
        before=deepcopy(self.inputs);again=build_manifest(*self.inputs)
        self.assertEqual(json.dumps(self.plan),json.dumps(again));self.assertEqual(self.inputs,before)
        self.assertFalse(self.plan['target_values_exported']);self.assertFalse(self.plan['real_training_started'])
        for c in self.plan['companies']:
            for d in c['days']:self.assertNotIn('target',d);self.assertNotIn('labels',d)

    def test_registered_fixed_company_seed_and_golden_day_order(self):
        self.assertEqual([c['symbol'] for c in self.plan['companies']],['A','AMZN','MSFT'])
        self.assertEqual([c['seed'] for c in self.plan['companies']],[610,611,612])
        # Golden permutations captured independently from stdlib Random(610..612).
        self.assertEqual([d['source_row_index'] for d in self.plan['companies'][0]['days']],[0,2,1,4,3])
        self.assertEqual([d['source_row_index'] for d in self.plan['companies'][1]['days']],[0,3,5,1,2,4])
        self.assertEqual([d['source_row_index'] for d in self.plan['companies'][2]['days']],[1,2,0])
        self.assertEqual(sorted(d['sample_id'] for c in self.plan['companies'] for d in c['days']),
            sorted(s for p in self.inputs[3].values() for s in p['train_ids']))

    def test_collator_batch_size_does_not_change_optimization_days(self):
        for n in [1,2,3,7]:
            with self.subTest(batch_size=n):self.assertEqual(self.plan,build_manifest(*fixture(n)))

    def test_tail_groups_and_incomplete_title_chunks_preserved(self):
        for company in self.plan['companies']:
            rows=self.inputs[1][company['symbol']]
            self.assertEqual(len(company['days']),len(rows))
            for day in company['days']:
                cover=[i for start,stop in day['chunks'] for i in range(start,stop)]
                self.assertEqual(cover,list(range(day['headline_count'])))
                self.assertTrue(all(0<stop-start<=8 for start,stop in day['chunks']))
        self.assertEqual(len(self.inputs[2]['A'][-1]['sample_ids']),1)
        self.assertTrue(any(stop-start<8 for c in self.plan['companies'] for d in c['days'] for start,stop in d['chunks']))

    def test_company_and_day_boundaries_match_mocks_not_batches_or_chunks(self):
        replay=mock_replay(self.plan);events=replay['events']
        self.assertEqual(replay['status'],'mock_complete');self.assertEqual(len(replay['completed_mock_day_ids']),14)
        self.assertEqual([e[1] for e in events if e[0]=='company_reset_marker'],['A','AMZN','MSFT'])
        cursor=0
        for c in self.plan['companies']:
            self.assertEqual(events[cursor],['company_reset_marker',c['symbol']]);cursor+=1
            for d in c['days']:
                sid=d['sample_id'];expected=[['zero_grad_marker',sid]]
                expected += [['headline_chunk_marker',sid,a,b] for a,b in d['chunks']]
                expected += [['day_loss_marker',sid],['backward_marker',sid],['clip_marker',sid,1.],['day_step_marker',sid]]
                self.assertEqual(events[cursor:cursor+len(expected)],expected);cursor+=len(expected)
        self.assertEqual(cursor,len(events));self.assertEqual(replay['real_optimizer_steps'],0)
        self.assertFalse(replay['real_training_complete']);self.assertFalse(replay['heldout_evaluation_authorized'])

    def test_zero_budget_before_day_never_begins_or_completes_day(self):
        r=mock_replay(self.plan,0)
        self.assertEqual(r['status'],'mock_partial');self.assertEqual(r['completed_mock_day_ids'],[])
        self.assertIsNone(r['incomplete_mock_day']);self.assertEqual(len(r['not_started_mock_day_ids']),14)
        self.assertFalse(any(e[0]=='zero_grad_marker' for e in r['events']))

    def test_mid_day_budget_partial_never_emits_loss_backward_or_step(self):
        prefix=0
        for c in self.plan['companies']:
            for d in c['days']:
                if len(d['chunks'])>1:break
                prefix+=1+len(d['chunks'])
            if len(d['chunks'])>1:break
        r=mock_replay(self.plan,prefix+2);sid=d['sample_id']
        self.assertEqual(r['status'],'mock_partial');self.assertEqual(r['incomplete_mock_day'],sid)
        self.assertNotIn(sid,r['completed_mock_day_ids']);self.assertNotIn(sid,r['not_started_mock_day_ids'])
        self.assertEqual([e[0] for e in r['events'] if len(e)>1 and e[1]==sid],['zero_grad_marker','headline_chunk_marker'])
        self.assertEqual(len(r['completed_mock_day_ids'])+len(r['not_started_mock_day_ids'])+1,14)

    def test_exact_budget_finishes_and_replays_without_stale_marker_state(self):
        n=sum(1+len(d['chunks']) for c in self.plan['companies'] for d in c['days'])
        a=mock_replay(self.plan,n);b=mock_replay(self.plan,n)
        self.assertEqual(a,b);self.assertEqual(a['status'],'mock_complete')
        self.assertEqual(sum(e[0]=='zero_grad_marker' for e in a['events']),14)

    def test_duplicate_missing_and_reordered_canonical_groups_rejected(self):
        for mode in ['duplicate','missing','reverse']:
            reg,training,batches,partition=deepcopy(self.inputs)
            if mode=='duplicate':batches['A'].append(deepcopy(batches['A'][0]))
            elif mode=='missing':batches['A'].pop()
            else:batches['A'].reverse()
            with self.subTest(mode=mode),self.assertRaises(ValueError):build_manifest(reg,training,batches,partition)

    def test_wrong_label_id_value_headline_or_group_boundary_rejected(self):
        for field,value in [('label_sample_ids',['A:wrong']*4),('labels',[99.]*4),
            ('headline_hashes',['0'*64]*15),('group_offsets',[0,1,1,4,15])]:
            reg,training,batches,partition=deepcopy(self.inputs);batches['A'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):build_manifest(reg,training,batches,partition)

    def test_training_only_scale_and_targets_ignore_heldout_metadata(self):
        reg,training,batches,partition=deepcopy(self.inputs)
        for p in partition.values():p['test_ids'] += ['unknown:future']
        self.assertEqual(self.plan,build_manifest(reg,training,batches,partition))
        norm=self.plan['companies'][0]['normalization']
        self.assertEqual(norm['source'],'original_training_packet_only');self.assertEqual(norm['count'],5)
        self.assertAlmostEqual(norm['mean'],0.);self.assertAlmostEqual(norm['scale'],(.0002)**.5)

    def test_heldout_id_target_source_or_overlap_cannot_enter(self):
        reg,training,batches,partition=deepcopy(self.inputs);partition['A']['test_ids'].append('A:0')
        with self.assertRaises(ValueError):build_manifest(reg,training,batches,partition)
        reg,training,batches,partition=deepcopy(self.inputs);training['A'][0].pop('target')
        with self.assertRaises(ValueError):build_manifest(reg,training,batches,partition)

    def test_changed_registration_company_epoch_or_step_unit_rejected(self):
        for key,value in [('epochs',2),('max_total_steps',13),('gradient_clip_norm',2.)]:
            reg,training,batches,partition=deepcopy(self.inputs);reg['training'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):build_manifest(reg,training,batches,partition)
        reg,training,batches,partition=deepcopy(self.inputs);reg['companies'].reverse()
        with self.assertRaises(ValueError):build_manifest(reg,training,batches,partition)

    def test_tampered_manifest_missing_or_duplicate_chunk_rejected_before_markers(self):
        for mutation in ['day','chunk','unit']:
            p=deepcopy(self.plan)
            if mutation=='day':p['companies'][0]['days'].pop()
            elif mutation=='chunk':p['companies'][0]['days'][0]['chunks'].append([0,1])
            else:p['optimization_unit']='headline'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):mock_replay(p)

    def test_invalid_mock_budget_rejected_without_running_work(self):
        for value in [-1,True,1.5]:
            with self.subTest(value=value),self.assertRaises(ValueError):mock_replay(self.plan,value)


if __name__=='__main__':unittest.main()
