"""Artificial schema fixtures; no API calls or observed performance evidence."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from corporate_actions import audit_cash_dividends, audit_return_series
from price_labels import build_labels, make_time_split
from price_baselines import synthetic_dataset
from run_course import run_pipeline
from research.financial_sentiment_using_prices.tests.test_price_labels import dataset,DAYS


def dividend(**changes):
    row={'event_id':'fixture-spy','symbol':'SPY','ex_date':'2026-09-18',
         'record_date':'2026-09-18','payment_date':'2026-10-30','process_date':'2026-10-30',
         'cash_per_share':'1.888834','currency':'USD','treatment':'standard_cash_dividend'}
    row.update(changes)
    return row


def proof(symbol='PLTR'):
    return {'method':'vendor_total_return_index','series_id':'fixture-tri-'+symbol,
            'provenance_id':'fixture-contract-'+symbol,'cash_dividends_embedded':True,
            'splits_embedded':True,'additional_cash_dividends_added':False,
            'available_at':'2026-10-15T13:31:00Z',
            'ex_date_coverage':{'date_basis':'ex_date','start':DAYS[0],'end':DAYS[-1],
                                'reviewed_complete':True,'evidence_id':'fixture-ex-date-review'}}


def observed_branch_fixture():
    # Observed-input validation branch only: all values/evidence are artificial.
    data=dataset();data['data_kind']='observed'
    data['calendar']['calendar_id']='artificial-contract-fixture'
    data['return_series_evidence']={s:proof(s) for s in ['PLTR','SPY','IGV']}
    for point in data['prices']:point['return_series_id']='fixture-tri-'+point['symbol']
    return data


class CashDividendDateTests(unittest.TestCase):
    def audit(self,records,entry='2026-09-01',exit_='2026-09-30',**changes):
        query={'date_basis':'process_date','start':'2026-09-01','end':'2026-10-30'};query.update(changes)
        return audit_cash_dividends(records,symbol='SPY',entry_session=entry,exit_session=exit_,query=query)

    def test_empty_narrow_process_query_never_proves_complete_ex_date_coverage(self):
        narrow=self.audit([],end='2026-10-16');broad=self.audit([dividend()])
        self.assertEqual(narrow['selected_event_ids'],[])
        self.assertEqual(broad['selected_event_ids'],['fixture-spy'])
        self.assertFalse(narrow['ex_date_coverage_verified']);self.assertFalse(broad['ex_date_coverage_verified'])

    def test_spy_ex_date_is_separate_from_later_pay_and_process_dates(self):
        report=self.audit([dividend()]);row=report['retained_records'][0]
        self.assertEqual((row['ex_date'],row['payment_date'],row['process_date']),
                         ('2026-09-18','2026-10-30','2026-10-30'))
        self.assertEqual(report['selected_event_ids'],['fixture-spy']);self.assertEqual(report['labels_created'],0)

    def test_msft_september_payment_does_not_move_august_ex_date(self):
        event=dividend(event_id='fixture-msft',symbol='MSFT',ex_date='2026-08-20',record_date='2026-08-20',
                       payment_date='2026-09-10',process_date='2026-09-10',cash_per_share='.91')
        report=audit_cash_dividends([event],symbol='MSFT',entry_session='2026-09-01',exit_session='2026-09-30',
                                  query={'date_basis':'process_date','start':'2026-09-01','end':'2026-09-30'})
        self.assertEqual(report['selected_event_ids'],[])

    def test_entry_ex_date_excluded_and_exit_ex_date_included(self):
        self.assertEqual(self.audit([dividend()],entry='2026-09-18')['selected_event_ids'],[])
        self.assertEqual(self.audit([dividend()],exit_='2026-09-18')['selected_event_ids'],['fixture-spy'])

    def test_payment_date_is_not_a_second_event(self):
        self.assertEqual(self.audit([dividend()],entry='2026-10-01',exit_='2026-10-30')['selected_event_ids'],[])

    def test_overlapping_captures_deduplicate_stable_id_once(self):
        report=self.audit([dividend(),dividend(cash_per_share='1.8888340',capture='second')])
        self.assertEqual(len(report['retained_records']),1)
        self.assertEqual(report['selected_event_ids'],['fixture-spy'])
        self.assertEqual(report['duplicate_capture_ids'],['fixture-spy'])

    def test_conflicting_versions_and_ambiguous_different_ids_are_rejected(self):
        for second in [dividend(cash_per_share='2'),dividend(event_id='second-id')]:
            with self.assertRaises(ValueError):self.audit([dividend(),second])

    def test_missing_ex_date_never_falls_back_to_pay_or_process(self):
        for value in [None,'','2026-09-99']:
            with self.assertRaises((ValueError,TypeError)):self.audit([dividend(ex_date=value)])

    def test_query_date_basis_must_explicitly_be_process_date(self):
        with self.assertRaisesRegex(ValueError,'process_date'):self.audit([],date_basis='ex_date')

    def test_non_cash_due_bill_and_invalid_amounts_are_not_guessed(self):
        events=[dividend(treatment='stock_dividend'),dividend(treatment='due_bill')]
        events += [dividend(cash_per_share=x) for x in [None,True,0,-1,'NaN','Infinity']]
        for event in events:
            with self.assertRaises(ValueError):self.audit([event])

    def test_audit_is_pure_and_unknown_process_date_remains_unknown(self):
        records=[dividend(process_date=None)];before=deepcopy(records);report=self.audit(records)
        self.assertEqual(records,before);self.assertIsNone(report['retained_records'][0]['process_date'])


class TotalReturnEvidenceTests(unittest.TestCase):
    def row(self,data=None):return build_labels(data or observed_branch_fixture())[0]

    def test_missing_observed_proof_keeps_labels_out_of_split(self):
        data=observed_branch_fixture();data.pop('return_series_evidence');row=self.row(data)
        self.assertEqual(row['status'],'Retrospective');self.assertIn('total_return_evidence_missing:SPY',row['reasons'])
        split=make_time_split([row],'2026-10-16T12:00:00Z','2026-10-17T12:00:00Z','2026-10-20T12:00:00Z')
        self.assertEqual(split['train'],[])

    def test_supplied_contract_preserves_index_ratio_without_adding_cash(self):
        row=self.row();self.assertEqual(row['status'],'Eligible')
        self.assertAlmostEqual(row['stock_return'],.1);self.assertAlmostEqual(row['spy_return'],.05)
        self.assertAlmostEqual(row['excess_return'],.05);self.assertEqual(len(row['return_evidence_audit']),3)

    def test_process_range_cannot_substitute_for_ex_date_coverage(self):
        data=observed_branch_fixture();data['return_series_evidence']['SPY']['ex_date_coverage']['date_basis']='process_date'
        row=self.row(data);self.assertEqual(row['status'],'Retrospective')
        self.assertIn('ex_date_coverage_unverified:SPY',row['reasons'])

    def test_double_cash_addition_or_unknown_policy_cannot_be_eligible(self):
        for extra in [True,None,'false']:
            data=observed_branch_fixture();data['return_series_evidence']['SPY']['additional_cash_dividends_added']=extra
            row=self.row(data);self.assertNotEqual(row['status'],'Eligible')
            self.assertIn('cash_dividend_double_count_or_unknown:SPY',row['reasons'])

    def test_adjustment_names_are_not_a_total_return_method(self):
        for method in ['raw_plus_cash','split','dividend','all',None]:
            data=observed_branch_fixture();data['return_series_evidence']['SPY']['method']=method
            self.assertIn('unsupported_total_return_method:SPY',self.row(data)['reasons'])

    def test_both_endpoints_bind_same_index_scale(self):
        data=observed_branch_fixture()
        next(p for p in data['prices'] if p['symbol']=='SPY' and p['session_date']==DAYS[10])['return_series_id']='different-scale'
        self.assertIn('total_return_series_binding_missing_or_inconsistent:SPY',self.row(data)['reasons'])

    def test_cash_and_split_treatment_must_both_be_explicit(self):
        for field in ['splits_embedded','cash_dividends_embedded']:
            data=observed_branch_fixture();data['return_series_evidence']['SPY'].pop(field)
            self.assertNotEqual(self.row(data)['status'],'Eligible')

    def test_future_evidence_keeps_price_mature_window_pending(self):
        data=observed_branch_fixture();data['as_of']='2026-10-20T20:00:00Z'
        data['return_series_evidence']['SPY']['available_at']='2026-10-21T20:00:00Z'
        row=self.row(data);self.assertEqual(row['label_available_at'],'2026-10-21T20:00:00Z')
        self.assertEqual(row['status'],'Pending');self.assertIsNone(row['excess_return'])
        self.assertIn('total_return_evidence_not_available_as_of',row['reasons'])

    def test_late_evidence_cannot_enter_earlier_training(self):
        data=observed_branch_fixture();data['return_series_evidence']['SPY']['available_at']='2026-10-20T20:00:00Z'
        row=self.row(data);self.assertEqual(row['status'],'Eligible')
        split=make_time_split([row],'2026-10-16T12:00:00Z','2026-10-17T12:00:00Z','2026-10-20T12:00:00Z')
        self.assertEqual(split['train'],[])
        self.assertIn('training_label_not_mature_before_cutoff',split['excluded'][0]['reasons'])

    def test_incomplete_review_does_not_certify_zero_actions(self):
        for change in [{'start':'2026-10-02'},{'end':'2026-10-14'},{'reviewed_complete':False},{'evidence_id':None}]:
            data=observed_branch_fixture();data['return_series_evidence']['SPY']['ex_date_coverage'].update(change)
            self.assertNotEqual(self.row(data)['status'],'Eligible')

    def test_omitted_data_kind_is_conservative(self):
        data=observed_branch_fixture();data.pop('data_kind');data.pop('return_series_evidence')
        self.assertNotEqual(self.row(data)['status'],'Eligible')

    def test_missing_naive_or_invalid_evidence_clock_stays_unverified(self):
        for value in [None,'2026-10-15T13:31:00','invalid']:
            data=observed_branch_fixture();data['return_series_evidence']['SPY']['available_at']=value
            self.assertNotEqual(self.row(data)['status'],'Eligible')

    def test_unverified_optional_sector_return_is_omitted(self):
        data=observed_branch_fixture();data['return_series_evidence'].pop('IGV');row=self.row(data)
        self.assertEqual(row['status'],'Eligible');self.assertIsNone(row['sector_return'])
        self.assertIn('sector_total_return_evidence_unavailable',row['reasons'])

    def test_prepared_input_missing_proof_does_not_fit_models_and_hashes_new_module(self):
        data=synthetic_dataset();data['data_kind']='observed';data.pop('synthetic_fixture')
        data['calendar']['calendar_id']='artificial-observed-branch-fixture'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.json';path.write_text(json.dumps(data));out=Path(directory)/'out'
            report=run_pipeline(dataset_path=path,output_dir=out)
            self.assertEqual(report['eligible_labels'],0);self.assertEqual(report['evaluation_status'],'Not evaluated')
            self.assertIn('return_series_evidence.SPY',report['missing_inputs'])
            self.assertEqual(json.loads((out/'comparison.json').read_text())['models'],{})
            names=[Path(p).name for p in json.loads((out/'run_manifest.json').read_text())['generation_sources_sha256']]
            self.assertIn('corporate_actions.py',names)

    def test_evidence_check_is_pure_and_not_independent_authentication(self):
        evidence=proof();before=deepcopy(evidence)
        result=audit_return_series(evidence,symbol='PLTR',entry_session=DAYS[0],exit_session=DAYS[10],endpoint_series_ids=['fixture-tri-PLTR']*2)
        self.assertTrue(result['contract_eligible']);self.assertEqual(evidence,before)
        self.assertIn('not independent',result['scope'])


if __name__=='__main__':unittest.main()
