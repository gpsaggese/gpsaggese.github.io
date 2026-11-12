"""Synthetic word lists only; no proprietary dictionary is redistributed here."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lm_baseline import (load_dictionary,score_headline,score_company,calibrate_lm,
    case_split,verify_inputs,run_registered)
from multi_horizon import sha

LEX={'words':{'PROFIT','LOSS','NEWS','TODAY'},'positive':{'PROFIT'},'negative':{'LOSS'}}
CFG={'class_neutral_band':.005,'ridge_alpha':10.,'solver':'lsqr','tol':1e-8}

def rows_fixture():
    rows=[{'sample_id':str(i),'price_features':[i*.001,(i%3)*.01],
        'stock_return':(i%7-2)*.01,'excess_return':(i%7-2)*.01-.004} for i in range(40)]
    split={'train':rows[:30],'test':rows[30:]};features={str(i):(i%5-2)*.02 for i in range(40)}
    return split,features


class LMTests(unittest.TestCase):
    def dictionary(self,d,text):
        p=Path(d)/'lex.csv';p.write_text(text);return load_dictionary(p)

    def test_flags_use_current_version_and_negative_removed_is_inactive(self):
        with tempfile.TemporaryDirectory() as d:
            lex,audit=self.dictionary(d,'Word,Positive,Negative\nPROFIT,2025,0\nLOSS,0,2009\nOLD,-2024,0\n')
            self.assertEqual(lex['positive'],{'PROFIT'});self.assertEqual(lex['negative'],{'LOSS'})
            self.assertEqual(audit['active_category_flags_after_2023'],{'Positive':1})
            self.assertFalse(audit['historical_asof_2023_verified']);self.assertFalse(audit['commercial_deployment_authorized'])

    def test_duplicate_or_missing_column_dictionary_fails(self):
        with tempfile.TemporaryDirectory() as d:
            for text in ['Word,Positive\nPROFIT,1\n','Word,Positive,Negative\nPROFIT,1,0\nLOSS,0,1\nPROFIT,1,0\n']:
                with self.subTest(text=text),self.assertRaises(ValueError):self.dictionary(d,text)

    def test_overlapping_polarity_categories_fail(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'overlap'):self.dictionary(d,'Word,Positive,Negative\nPROFIT,1,1\n')

    def test_case_repetition_denominator_and_no_stemming(self):
        v=score_headline('Profit profit LOSS x profits 123!',LEX)
        self.assertEqual((v['tokens'],v['positive'],v['negative']),(4,2,1))
        self.assertEqual(v['score'],.25);self.assertEqual(v['master_matches'],3)

    def test_unknown_words_are_zero_with_explicit_no_match_diagnostic(self):
        v=score_headline('neutral-looking unknown words',LEX)
        self.assertEqual(v['score'],0);self.assertEqual(v['polarity_evidence'],'NoPolarityMatch')
        self.assertEqual(v['status'],'Available')

    def test_empty_or_numeric_only_headline_is_unavailable(self):
        for text in ['',None,'123 $ % I A']:
            with self.subTest(text=text):self.assertEqual(score_headline(text,LEX)['status'],'Unavailable')

    def test_company_day_averages_distinct_original_headlines_equally(self):
        news=[{'source_row_index':1,'headline':'profit profit'},{'source_row_index':2,'headline':'loss news today'}]
        rows=[{'sample_id':'A','source_row_indices':[1,2,1],'text':'profit profit\nloss news today'}]
        cache={};features,audits,missing=score_company(rows,news,LEX,cache)
        self.assertAlmostEqual(features['A'],1/3);self.assertEqual(audits['A']['headline_count'],2)
        self.assertEqual(len(cache),2);self.assertEqual(missing,{})

    def test_changed_original_text_is_unavailable_without_replacement(self):
        news=[{'source_row_index':1,'headline':'profit'}]
        rows=[{'sample_id':'A','source_row_indices':[1],'text':'different'}]
        features,audits,missing=score_company(rows,news,LEX,{})
        self.assertEqual(features,{});self.assertIn('A',missing);self.assertEqual(audits['A']['status'],'Unavailable')

    def test_zero_token_headline_cannot_be_removed_from_a_company_day(self):
        news=[{'source_row_index':1,'headline':'profit'},{'source_row_index':2,'headline':'123'}]
        rows=[{'sample_id':'A','source_row_indices':[1,2],'text':'profit\n123'}]
        features,_,missing=score_company(rows,news,LEX,{})
        self.assertEqual(features,{});self.assertIn('A',missing)

    def test_missing_score_marks_both_added_models_unavailable_keeps_denominators(self):
        split,features=rows_fixture();features.pop('0')
        metrics,pred,fitted,status=calibrate_lm(split,'raw_price_return',features,{'0':['missing']},CFG)
        self.assertEqual((metrics,pred,fitted),({},{},{}));self.assertEqual(status['required_train'],30)
        self.assertEqual(status['required_test'],10);self.assertFalse(status['selective_row_removal'])
        self.assertEqual(len(split['train']),30)

    def test_raw_and_excess_calibrate_the_named_targets(self):
        split,features=rows_fixture()
        _,_,raw,_=calibrate_lm(split,'raw_price_return',features,{},CFG)
        _,_,excess,_=calibrate_lm(split,'excess_price_return',features,{},CFG)
        for model in raw:
            self.assertAlmostEqual(raw[model]['intercept']-excess[model]['intercept'],.004)
            self.assertEqual(raw[model]['target'],'raw_price_return')

    def test_test_features_or_outcomes_never_fit_scaler_or_coefficients(self):
        a,features=rows_fixture();b=deepcopy(a);features_b=deepcopy(features)
        for row in b['test']:
            row['price_features']=[999.,999.];row['stock_return']+=.4;row['excess_return']+=.4;features_b[row['sample_id']]=999.
        _,_,fit_a,_=calibrate_lm(a,'excess_price_return',features,{},CFG)
        _,_,fit_b,_=calibrate_lm(b,'excess_price_return',features_b,{},CFG)
        self.assertEqual(fit_a,fit_b)
        expected=sum(features[r['sample_id']] for r in a['train'])/30
        self.assertAlmostEqual(fit_a['lm_tone_ridge']['means'][0],expected)

    def test_case_selection_preserves_exact_registered_common_order(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'A/h5').mkdir(parents=True);(root/'case').mkdir()
            split,features=rows_fixture();(root/'A/h5/split.json').write_text(json.dumps(split))
            (root/'case/report.json').write_text(json.dumps({'train_ids':['2','0','1'],'test_ids':['31','30']}))
            selected=case_split(root,{'company':'A','horizon':5,'path':'case'})
            self.assertEqual([r['sample_id'] for r in selected['train']],['2','0','1'])
            self.assertEqual([r['sample_id'] for r in selected['test']],['31','30'])

    def test_existing_output_is_preserved_before_any_source_access(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out';out.mkdir();(out/'keep').write_text('keep')
            with self.assertRaises(FileExistsError):run_registered(root,root/'missing_registration',out)
            self.assertEqual((out/'keep').read_text(),'keep')


if __name__=='__main__':unittest.main()
