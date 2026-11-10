"""No market data or downloads: real tiny-BERT gradients and provenance guards."""
import copy
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_price_bert import check_updates, fit_company, metrics, public_summary, validate_features
from price_transformer import Budget, BudgetStop, parameter_hashes, partial_model


def fixture():
    rows = [{'sample_id':'synthetic:1','headlines':['first','second'],'target':.02},
            {'sample_id':'synthetic:2','headlines':['third'],'target':-.01}]
    features=[]
    for row in rows:
        headlines=[]
        for text in row['headlines']:
            ids=[101,1000+len(text),102]
            headlines.append({'input_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'input_ids':ids,'attention_mask':[1]*3,'token_type_ids':[0]*3,
                'length':3,'untruncated_length':3,'truncated':False,'unknown_tokens':0})
        features.append({'sample_id':row['sample_id'],'headlines':headlines})
    return rows,features


class PriceBertTests(unittest.TestCase):
    def test_complete_group_and_label_alignment(self):
        rows,features=fixture()
        tokens=validate_features(features,rows,[r['sample_id'] for r in rows],{'heldout'})
        self.assertEqual(len(tokens),3)
        broken=copy.deepcopy(features);broken[0]['headlines'].pop()
        with self.assertRaises(ValueError):validate_features(broken,rows,[r['sample_id'] for r in rows],{'heldout'})
        with self.assertRaises(ValueError):validate_features(features,rows[::-1],[r['sample_id'] for r in rows],{'heldout'})
        with self.assertRaises(ValueError):validate_features(features,rows,[r['sample_id'] for r in rows],{'synthetic:1'})

    def test_real_tiny_bert_steps_and_freezing(self):
        import torch
        from transformers import BertConfig,BertModel
        torch.set_num_threads(1);torch.manual_seed(610)
        model=partial_model(BertModel(BertConfig(vocab_size=30522,hidden_size=12,
            num_hidden_layers=2,num_attention_heads=3,intermediate_size=16)))
        rows,features=fixture();ids=[r['sample_id'] for r in rows]
        tokens=validate_features(features,rows,ids,{'heldout'})
        count=[]
        result=fit_company(model,rows,tokens,{'heldout'},
            {'learning_rate':2e-5,'weight_decay':0,'gradient_clip_norm':1,'headline_chunk_size':1},
            Budget(30,10**9,0,resources=lambda:(1,10**10)),610,count.append)
        self.assertEqual(result['optimizer_steps'],2)
        self.assertEqual(len(count),2)
        self.assertGreater(result['parameter_update_audit']['changed_trainable_parameter_tensors'],2)
        self.assertEqual(set(result['parameter_update_audit']['before_sha256']),
                         set(result['parameter_update_audit']['after_sha256']))
        self.assertAlmostEqual(result['training_mean'],.005)
        self.assertAlmostEqual(result['training_scale'],.015)
        before=parameter_hashes(model)
        with torch.no_grad():model.backbone.embeddings.word_embeddings.weight.add_(.001)
        with self.assertRaisesRegex(ValueError,'Frozen'):check_updates(model,before)

    def test_stop_before_step_keeps_count_zero(self):
        import torch
        from transformers import BertConfig,BertModel
        model=partial_model(BertModel(BertConfig(vocab_size=30522,hidden_size=12,
            num_hidden_layers=2,num_attention_heads=3,intermediate_size=16)))
        rows,features=fixture();tokens=validate_features(features,rows,[r['sample_id'] for r in rows],{'heldout'})
        steps=[]
        with self.assertRaises(BudgetStop):
            fit_company(model,rows,tokens,{'heldout'},
                {'learning_rate':2e-5,'weight_decay':0,'gradient_clip_norm':1,'headline_chunk_size':1},
                Budget(30,1,0,resources=lambda:(2,10**10)),610,steps.append)
        self.assertEqual(steps,[])

    def test_metrics_are_price_labels_not_sentiment_accuracy(self):
        m=metrics([-.02,0,.02],[-.01,.001,-.03])
        self.assertAlmostEqual(m['mae'],.061/3)
        self.assertEqual(m['direction_accuracy'],1/3)
        self.assertEqual(m['price_class_balanced_accuracy'],2/3)
        with self.assertRaises(ValueError):metrics([1],[float('nan')])


class PublicSummaryTests(unittest.TestCase):
    """Progress receipts must not turn a planned pool into completed research."""

    def fixture(self):
        registration = {'checkpoint': {'repository': 'fixture/bert', 'revision': 'fixture',
                                      'license': 'fixture'},
                        'training': {'max_total_steps': 272}}
        receipt = {'status': 'incomplete', 'registration_sha256': 'fixture',
                   'actual_optimizer_steps': 0, 'actual_finetuned_companies': [],
                   'heldout_outcomes_opened': False, 'failures': [{'type': 'BudgetStop'}],
                   'elapsed_seconds': 1, 'planned_training_rows': 272,
                   'planned_heldout_rows': 154}
        return receipt, registration

    def test_zero_step_run_reports_zero_actual_rows_and_preserves_plan(self):
        receipt, registration = self.fixture()
        summary = public_summary(receipt, registration)
        self.assertEqual((summary['training_rows'], summary['heldout_rows']), (0, 0))
        self.assertEqual((summary['actual_training_rows'], summary['actual_evaluated_rows']), (0, 0))
        self.assertEqual((summary['planned_training_rows'], summary['planned_heldout_rows']), (272, 154))
        self.assertFalse(summary['heldout_predictions_completed'])
        self.assertEqual(summary['cases'], [])

    def test_partial_company_training_counts_only_completed_steps(self):
        receipt, registration = self.fixture()
        receipt.update(actual_optimizer_steps=55, actual_finetuned_companies=['A'])
        summary = public_summary(receipt, registration)
        self.assertEqual(summary['training_rows'], 55)
        self.assertEqual(summary['actual_training_rows'], 55)
        self.assertEqual(summary['heldout_rows'], 0)
        self.assertEqual(summary['completed_companies'], ['A'])

    def test_frozen_predictions_are_not_completed_evaluation(self):
        receipt, registration = self.fixture()
        receipt.update(actual_optimizer_steps=272, actual_finetuned_companies=['A', 'AMZN', 'MSFT'],
                       heldout_prediction_sha256='fixture')
        summary = public_summary(receipt, registration)
        self.assertTrue(summary['heldout_predictions_completed'])
        self.assertEqual(summary['training_rows'], 272)
        self.assertEqual(summary['heldout_rows'], 0)
        self.assertEqual(summary['actual_evaluated_rows'], 0)

    def test_partial_evaluation_count_does_not_publish_unfinished_metrics(self):
        receipt, registration = self.fixture()
        receipt.update(actual_optimizer_steps=272, actual_evaluated_rows=28,
                       heldout_outcomes_opened=True)
        summary = public_summary(receipt, registration)
        self.assertEqual(summary['heldout_rows'], 28)
        self.assertEqual(summary['actual_evaluated_rows'], 28)
        self.assertEqual(summary['planned_heldout_rows'], 154)
        self.assertEqual(summary['cases'], [])
        self.assertEqual(summary['status'], 'incomplete')

    def test_legacy_complete_receipt_retains_272_154_counts(self):
        receipt, registration = self.fixture()
        del receipt['planned_training_rows']
        del receipt['planned_heldout_rows']
        receipt.update(status='complete', actual_optimizer_steps=272,
                       actual_finetuned_companies=['A', 'AMZN', 'MSFT'],
                       heldout_outcomes_opened=True, heldout_prediction_sha256='fixture', failures=[])
        receipt['evaluation_cases'] = [
            {'company': company, 'training_rows': train, 'test_rows': test, 'metrics': {},
             'cost_evaluation': {'mode': 'fixture', 'eligible_intervals': 0, 'models': {}}}
            for company, train, test in [('A', 52, 28), ('AMZN', 110, 61), ('MSFT', 110, 65)]]
        summary = public_summary(receipt, registration)
        self.assertEqual((summary['training_rows'], summary['heldout_rows']), (272, 154))
        self.assertEqual((summary['actual_training_rows'], summary['actual_evaluated_rows']), (272, 154))
        self.assertEqual((summary['planned_training_rows'], summary['planned_heldout_rows']), (272, 154))
        self.assertEqual(len(summary['cases']), 3)

    def test_failure_before_preparation_keeps_unknown_heldout_pool_unknown(self):
        receipt, registration = self.fixture()
        del receipt['planned_training_rows']
        del receipt['planned_heldout_rows']
        summary = public_summary(receipt, registration)
        self.assertIsNone(summary['planned_heldout_rows'])
        self.assertEqual(summary['planned_training_rows'], 272)
        self.assertEqual(summary['heldout_rows'], 0)


if __name__=='__main__':unittest.main()
