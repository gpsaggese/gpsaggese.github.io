"""Artificial boundaries for the separately declared retrospective experiment."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from historical_backfill import parse_prefix, capture, REVISION
from run_historical import (build_rows, retrospective_split, evaluate_models, cost_evaluation,
                            canonical_url, price_class, run)


def protocol():
    return {"symbol": "A", "revision": REVISION, "source_date_start": "2016-01-01",
            "source_date_end_exclusive": "2020-06-06", "horizon_sessions": 10,
            "entry_delay_sessions": 2, "test_start": "2019-01-01", "class_neutral_band": .005,
            "ridge_alpha": 10., "tfidf": {"max_features": 2000, "min_df": 2, "ngram_range": [1, 2]},
            "assumed_roundtrip_cost_bps": [0, 10, 20], "limitations": ["Synthetic test fixture only"]}


def fixture():
    days = [(date(2018, 1, 1) + timedelta(days=i)).isoformat() for i in range(100)]
    calendar = {"sessions": [{"date": d, "open_at": d + "T14:30:00Z"} for d in days]}
    prices = {(s, d): 100 + i * (.1 if s == "A" else .05) for s in ["A", "SPY"] for i, d in enumerate(days)}
    def news(i, title="Agilent announces a new instrument", url=None):
        return {"headline": title, "source_date": days[i], "symbol": "A",
                "url": url or f"https://example.test/article/{i}", "source_row_index": i}
    return days, calendar, prices, news


def model_split():
    def row(i):
        return {"sample_id": str(i), "text": ("revenue improved " if i % 2 else "margin declined ") + f"company topic{i}",
                "price_features": [.001 * i, .01 * (i % 3)], "excess_return": .005 * (i % 3 - 1),
                "entry_at":"2019-01-02T14:30:00Z", "exit_at":"2019-01-16T14:30:00Z",
                "entry_session_index": i, "exit_session_index":i+10,
                "stock_return": .01, "spy_return": .005, "price_class":0,"source_ids":[str(i)]}
    return {"train": [row(i) for i in range(30)], "test": [row(i) for i in range(30, 40)]}


class RetrospectiveTests(unittest.TestCase):
    def test_incomplete_csv_suffix_is_never_a_sample(self):
        raw = b'Date,Article_title,Stock_symbol,Url\n2019-01-01,Good title,A,https://e/1\n2019-01-02,"unterminated'
        rows, incomplete = parse_prefix(raw)
        self.assertEqual(len(rows), 1)
        self.assertTrue(incomplete)

    def test_complete_csv_last_row_is_kept(self):
        rows, incomplete = parse_prefix(b'Date,Article_title,Stock_symbol,Url\n2019-01-01,Good title,A,https://e/1\n')
        self.assertEqual(len(rows), 1)
        self.assertFalse(incomplete)

    def test_unknown_schema_fails(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            parse_prefix(b'date,text\n2019-01-01,text\n')

    def test_tracking_query_removed_but_document_identity_kept(self):
        self.assertEqual(canonical_url('http://EXAMPLE.test/a?id=2&utm_source=x'), 'https://example.test/a?id=2')
        self.assertNotEqual(canonical_url('https://e.test/?id=1'), canonical_url('https://e.test/?id=2'))

    def test_price_classes_are_continuous_return_thresholds_not_tone(self):
        self.assertEqual([price_class(x) for x in [-.01, -.005, 0, .005, .01]], [-1, 0, 0, 0, 1])

    def test_two_session_delay_and_exact_ten_session_arithmetic(self):
        days, cal, prices, news = fixture()
        rows, _ = build_rows([news(25)], prices, cal, protocol())
        r = rows[0]
        self.assertEqual((r['entry_session_index'], r['exit_session_index']), (27, 37))
        expected = prices['A', days[37]] / prices['A', days[27]] - prices['SPY', days[37]] / prices['SPY', days[27]]
        self.assertAlmostEqual(r['excess_return'], expected)
        self.assertEqual(r['status'], 'Retrospective')
        self.assertFalse(r['historical_availability_verified'])

    def test_entry_and_future_prices_cannot_change_features(self):
        days, cal, prices, news = fixture()
        a, _ = build_rows([news(25)], prices, cal, protocol())
        prices['A', days[27]] *= 2
        prices['A', days[37]] *= 3
        b, _ = build_rows([news(25)], prices, cal, protocol())
        self.assertEqual(a[0]['price_features'], b[0]['price_features'])
        self.assertNotEqual(a[0]['excess_return'], b[0]['excess_return'])

    def test_duplicate_articles_aggregate_once_per_company_day(self):
        _, cal, prices, news = fixture()
        rows, _ = build_rows([news(25), news(25)], prices, cal, protocol())
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]['source_ids']), 1)
        self.assertEqual(rows[0]['text'].count('Agilent'), 1)

    def test_missing_price_is_excluded_without_imputation(self):
        days, cal, prices, news = fixture()
        del prices['SPY', days[37]]
        rows, _ = build_rows([news(25)], prices, cal, protocol())
        self.assertIn('missing_observed_price_endpoint', rows[0]['exclusion_reasons'])
        self.assertNotIn('excess_return', rows[0])

    def test_mature_split_purges_transitive_bridge_even_if_bridge_excluded(self):
        def row(s, entry, exit_, sources, excluded=None):
            return {'sample_id': s, 'entry_at': entry + 'T14:30:00Z', 'exit_at': exit_ + 'T14:30:00Z',
                    'source_ids': sources, 'event_cluster_ids': [], 'exclusion_reasons': excluded or []}
        rows = [row('train','2018-10-01','2018-10-15',['left']),
                row('bridge','2018-11-01','2018-11-15',['left','right'],['missing_price']),
                row('test','2019-01-02','2019-01-16',['right'])]
        split = retrospective_split(rows, '2019-01-01')
        self.assertEqual(split['train'], [])
        self.assertEqual(len(split['test']), 1)
        self.assertIn('source_or_proxy_event_linked_to_test', str(split['excluded']))

    def test_training_label_crossing_boundary_is_purged(self):
        row = {'sample_id':'late','entry_at':'2018-12-28T14:30:00Z','exit_at':'2019-01-10T14:30:00Z',
               'source_ids':['a'],'event_cluster_ids':[],'exclusion_reasons':[]}
        split = retrospective_split([row], '2019-01-01')
        self.assertEqual(split['train'], [])
        self.assertIn('training_outcome_crosses_time_boundary',str(split['excluded']))

    def test_test_text_and_labels_never_fit_vocabulary_scaler_or_model(self):
        split = model_split()
        _, pred, fitted = evaluate_models(split, protocol())
        changed = deepcopy(split)
        for row in changed['test']:
            row['text'] += ' secretfutureword'
            row['excess_return'] += 100
        _, pred2, fitted2 = evaluate_models(changed, protocol())
        self.assertNotIn('secretfutureword', fitted2['vocabulary'])
        self.assertEqual(fitted, fitted2)
        self.assertEqual([p['predictions'] for p in pred], [p['predictions'] for p in pred2])

    def test_costs_do_not_compound_overlapping_labels(self):
        rows = [{'sample_id':str(i),'entry_session_index':i,'exit_session_index':i+10,
                 'stock_return':.1,'spy_return':.02,'predictions':{'model':1.}} for i in range(21)]
        result = cost_evaluation(rows,[0,10,20])
        self.assertEqual(result['selected_sample_ids'], ['0','10','20'])
        self.assertEqual(result['overlapping_test_rows_not_compounded'], 18)
        values = result['models']['model']
        self.assertAlmostEqual(values[0]['cumulative_simulated_price_return'], 1.1**3-1)
        self.assertLess(values[2]['cumulative_simulated_price_return'],values[1]['cumulative_simulated_price_return'])
        self.assertIsNone(result['actual_broker_charges'])
        self.assertFalse(result['cash_dividends_included'])

    def test_cash_predictions_have_no_trades_or_assumed_costs(self):
        row = {'sample_id':'x','entry_session_index':1,'exit_session_index':11,
               'stock_return':.1,'spy_return':.02,'predictions':{'model':-1.}}
        self.assertEqual(cost_evaluation([row],[20])['models']['model'][0]['roundtrip_trades'],0)

    def test_invalid_assumed_cost_is_rejected(self):
        with self.assertRaises(ValueError):cost_evaluation([],[-1])
        # Empty prediction pools have no models; a real pool must reject costs
        # large enough to create a negative per-side capital factor.
        row={'sample_id':'x','entry_session_index':1,'exit_session_index':11,
             'stock_return':.1,'spy_return':.02,'predictions':{'model':1.}}
        with self.assertRaises(ValueError):cost_evaluation([row],[20000])

    def test_dangling_output_symlink_is_not_followed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'link').symlink_to(p/'new-target',target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'symlink'):
                run(p/'input',p/'link')
            self.assertFalse((p/'new-target').exists())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            sentinel = Path(d)/'keep';sentinel.write_text('keep')
            with self.assertRaises(FileExistsError):
                capture(d,protocol(),session=object())
            self.assertEqual(sentinel.read_text(),'keep')

    def test_failed_capture_cannot_be_evaluated(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'manifest.json').write_text(json.dumps({'status':'failed'}))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                run(p,p.parent/(p.name+'out'))

    def test_capture_audits_pagination_without_sending_credentials_to_public_host(self):
        raw=b'Date,Article_title,Stock_symbol,Url\n2019-01-03,Agilent instrument news,A,https://example.test/a\n'
        class Response:
            status_code=200
            headers={}
            def __init__(self,b):self.content=b
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def iter_content(self,chunk_size):yield self.content
        class Session:
            def __init__(self):self.calls=[]
            def get(self,url,**kw):
                self.calls.append((url,kw))
                if 'huggingface.co' in url:return Response(raw)
                page=len(self.calls)
                return Response(json.dumps({'bars':{'A':[{'t':f'2019-01-0{page+1}T05:00:00Z','o':100}],
                                                    'SPY':[{'t':f'2019-01-0{page+1}T05:00:00Z','o':100}]},
                                            'next_page_token':'next' if page==2 else None}).encode())
        with tempfile.TemporaryDirectory() as d, patch('historical_backfill.export_calendar',return_value={'sessions':[]}):
            session=Session()
            m=capture(Path(d)/'new',protocol(),session=session,credential_loader=lambda:('private-key','private-secret'))
            self.assertTrue(m['bars_summary']['pagination_complete'])
            self.assertEqual(m['bars_summary']['records'],{'A':2,'SPY':2})
            self.assertEqual(session.calls[2][1]['params']['page_token'],'next')
            self.assertNotIn('APCA-API-KEY-ID',session.calls[0][1]['headers'])
            self.assertEqual(session.calls[1][1]['headers']['APCA-API-KEY-ID'],'private-key')
            self.assertNotIn('private-secret',json.dumps(m))
            self.assertNotIn('private-key',json.dumps(m))

    def test_price_page_budget_exhaustion_is_failed_not_complete(self):
        raw=b'Date,Article_title,Stock_symbol,Url\n2019-01-03,Agilent instrument news,A,https://example.test/a\n'
        class Response:
            status_code=200;headers={}
            def __init__(self,b):self.content=b
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def iter_content(self,chunk_size):yield self.content
        class Session:
            def get(self,url,**kw):
                if 'huggingface.co' in url:return Response(raw)
                return Response(json.dumps({'bars':{'A':[{}],'SPY':[{}]},'next_page_token':'more'}).encode())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'new'
            with self.assertRaisesRegex(ValueError,'budget'):
                capture(p,protocol(),session=Session(),credential_loader=lambda:('k','s'),max_price_pages=1)
            self.assertEqual(json.loads((p/'manifest.json').read_text())['status'],'failed')
            self.assertTrue((p/'prices_000.json').exists())


if __name__ == '__main__':
    unittest.main()
