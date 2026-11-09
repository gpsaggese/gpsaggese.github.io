"""Small offline archive-integrity regressions; no performance assertions."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import hashlib,json,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from corporate_action_archive import capture,audit_archive,URL


def event(identity='a',**changes):
    row={'id':identity,'symbol':'SPY','ex_date':'2026-09-18','payable_date':'2026-10-30',
         'record_date':'2026-09-18','process_date':'2026-10-30','rate':1.888834,'special':False,'foreign':False}
    row.update(changes);return row


class Response:
    def __init__(self,rows,token=None,status=200):
        self.status_code=status;self.headers={'Content-Type':'application/json','Authorization':'never-retain'}
        self.content=json.dumps({'corporate_actions':{'cash_dividends':rows},'next_page_token':token}).encode()


class Session:
    def __init__(self,responses):self.responses=iter(responses);self.calls=[]
    def get(self,url,**kwargs):self.calls.append((url,kwargs));return next(self.responses)


class CorporateActionArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.tick=iter(datetime(2026,10,19,tzinfo=timezone.utc)+timedelta(seconds=i) for i in range(200))
    def collect(self,responses,**kwargs):
        output=self.root/('capture-'+str(len(list(self.root.iterdir()))));session=Session(responses)
        manifest=capture(output,symbols=['SPY'],start='2026-08-01',end='2026-11-30',
                         session=session,credential_loader=lambda:('private-key','private-secret'),clock=lambda:next(self.tick),**kwargs)
        return output,manifest,session
    def audit(self,output):return audit_archive(output,entry_session='2026-09-01',exit_session='2026-09-30')

    def test_realistic_response_preserves_dates_currency_unknown_and_zero_labels(self):
        output,m,_=self.collect([Response([event()])]);a=self.audit(output);r=a['cash_dividends'][0]
        self.assertEqual((r['ex_date'],r['payment_date'],r['process_date']),('2026-09-18','2026-10-30','2026-10-30'))
        self.assertIsNone(r['currency']);self.assertFalse(r['historical_availability_verified'])
        self.assertTrue(a['pagination_complete_under_query']);self.assertFalse(a['ex_date_coverage_verified'])
        self.assertFalse(a['total_return_series_verified']);self.assertEqual(a['eligible_labels'],0)

    def test_two_pages_bind_query_tokens_and_keep_credentials_out_of_archive(self):
        output,m,session=self.collect([Response([event()],token='next'),Response([event('b',rate=2)])],limit=1)
        self.assertEqual(m['status'],'complete_under_query');self.assertEqual(self.audit(output)['unique_events'],2)
        self.assertEqual(session.calls[1][1]['params']['page_token'],'next')
        self.assertTrue(all(url==URL and call['allow_redirects'] is False for url,call in session.calls))
        self.assertNotIn('private-key',(output/'manifest.json').read_text())
        self.assertNotIn('private-secret',(output/'manifest.json').read_text())
        self.assertNotIn('never-retain',(output/'manifest.json').read_text())

    def test_page_budget_is_explicitly_incomplete(self):
        output,m,_=self.collect([Response([event()],token='more')],max_pages=1)
        self.assertEqual(m['status'],'truncated_by_page_budget');self.assertFalse(self.audit(output)['pagination_complete_under_query'])

    def test_corrupt_raw_bytes_are_rejected(self):
        output,_,_=self.collect([Response([event()])]);(output/'page_000.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'hash'):self.audit(output)

    def test_changed_pagination_query_is_rejected(self):
        output,m,_=self.collect([Response([event()],token='next'),Response([event('b')])])
        m['requests'][1]['params']['start']='2026-09-01';(output/'manifest.json').write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'Query'):self.audit(output)

    def test_repeated_token_is_failed_and_not_complete(self):
        output,m,_=self.collect([Response([event()],token='same'),Response([event('b')],token='same')])
        self.assertEqual(m['status'],'failed')
        with self.assertRaisesRegex(ValueError,'Repeated'):self.audit(output)

    def test_failed_http_is_not_saved_as_successful_json(self):
        output,m,_=self.collect([Response([],status=403)])
        self.assertEqual(m['status'],'failed');self.assertFalse((output/'page_000.json').exists())
        with self.assertRaises(ValueError):self.audit(output)

    def test_input_and_existing_output_are_not_overwritten(self):
        output,_,_=self.collect([Response([event()])]);before=(output/'manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):capture(output,symbols=['SPY'],start='2026-08-01',end='2026-11-30')
        self.assertEqual(before,(output/'manifest.json').read_bytes())

    def test_receipt_preceding_request_is_rejected(self):
        output,m,_=self.collect([Response([event()])]);m['requests'][0]['response_received_at']='2026-01-01T00:00:00Z'
        (output/'manifest.json').write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'receipt'):self.audit(output)

    def test_missing_early_announcement_fields_stay_unknown(self):
        output,_,_=self.collect([Response([event(ex_date=None,rate=None,process_date=None)])],data_quality='all')
        row=self.audit(output)['cash_dividends'][0]
        self.assertIsNone(row['ex_date']);self.assertIsNone(row['cash_per_share'])
        self.assertFalse(row['in_ex_date_holding_interval']);self.assertIn('ex_date_missing',row['cash_contract_blockers'])

    def test_process_date_outside_query_is_not_reassigned(self):
        output,_,_=self.collect([Response([event(process_date='2026-12-01')])])
        with self.assertRaisesRegex(ValueError,'Process date outside'):self.audit(output)

    def test_instrumented_probe_can_be_audited_without_redownload(self):
        output=self.root/'probe';output.mkdir();raw=Response([event()]).content
        (output/'response.json').write_bytes(raw)
        m={'url':URL,'date_basis':'process_date','params':{'symbols':'SPY','start':'2026-08-01','end':'2026-11-30','sort':'asc','limit':1000},
           'http_status':200,'request_started_at':'2026-10-19T10:00:00Z','response_received_at':'2026-10-19T10:00:01Z',
           'response_sha256':hashlib.sha256(raw).hexdigest(),'response_bytes':len(raw)}
        (output/'manifest.json').write_text(json.dumps(m));self.assertEqual(self.audit(output)['unique_events'],1)

    def test_duplicate_versions_count_once_and_conflicts_require_review(self):
        output,_,_=self.collect([Response([event(),event()])]);a=self.audit(output)
        self.assertEqual(a['unique_events'],1);self.assertEqual(a['duplicate_capture_ids'],['a'])
        output,_,_=self.collect([Response([event(),event(rate=2)])])
        with self.assertRaisesRegex(ValueError,'Conflicting'):self.audit(output)


if __name__=='__main__':unittest.main()
