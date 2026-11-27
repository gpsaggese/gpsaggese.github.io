"""Daily overlap/cursor and exact after-hours windows without network access."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from daily_collection import run_daily
from collect_research import collect_research, CollectionError
from research.financial_sentiment_using_prices.tests.test_collect_research import Session, Response, Clock, bars, news

NOW = datetime(2026,10,21,17,tzinfo=timezone.utc)


class FakeCollector:
    def __init__(self, complete=True, error=False):
        self.calls=[];self.complete=complete;self.error=error

    def __call__(self, output, symbols, start_date, end_date, **kwargs):
        self.calls.append((output, symbols, start_date, end_date, kwargs))
        output.mkdir();(output/'captured.json').write_text(json.dumps({'version':len(self.calls)}))
        if self.error:raise ValueError('Fixture transport failure')
        return {'status':'complete' if self.complete else 'truncated',
                'news_summary':{'pagination_complete':self.complete},
                'bars_summary':{'pagination_complete':True}}


class DailyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'daily'

    def daily(self, collector, **kwargs):
        return run_daily(self.root,collector=collector,clock=lambda:NOW,**kwargs)

    def test_default_initial_window_uses_actual_clock_including_post_close(self):
        f=FakeCollector();r=self.daily(f)
        self.assertEqual(r['window_end_at'],'2026-10-21T16:40:00+00:00')
        self.assertEqual(r['window_start_at'],'2026-10-18T16:40:00+00:00')
        self.assertEqual(f.calls[0][4]['end_at'],r['window_end_at'])
        self.assertFalse(r['historical_availability_verified'])

    def test_rerun_overlaps_cursor_keeps_each_archive_and_never_rolls_back(self):
        f=FakeCollector()
        a=self.daily(f,end_at='2026-10-20T23:00:00Z')
        content=(Path(a['capture_dir'])/'captured.json').read_bytes()
        b=self.daily(f)
        self.assertEqual(b['window_start_at'],'2026-10-20T22:00:00+00:00')
        self.assertNotEqual(a['capture_dir'],b['capture_dir'])
        self.assertEqual((Path(a['capture_dir'])/'captured.json').read_bytes(),content)
        self.daily(f,start_at='2026-10-19T00:00:00Z',end_at='2026-10-20T00:00:00Z')
        cursor=json.loads((self.root/'cursor.json').read_text())
        self.assertEqual(cursor['last_complete_end_at'],'2026-10-21T16:40:00+00:00')
        self.assertEqual(len(cursor['captures']),3)

    def test_truncation_does_not_advance_cursor(self):
        self.daily(FakeCollector())
        before=(self.root/'cursor.json').read_bytes()
        r=self.daily(FakeCollector(complete=False))
        self.assertFalse(r['cursor_advanced_or_retained'])
        self.assertEqual((self.root/'cursor.json').read_bytes(),before)

    def test_failed_request_keeps_old_cursor_and_raw_attempt(self):
        self.daily(FakeCollector());before=(self.root/'cursor.json').read_bytes()
        with self.assertRaises(ValueError):self.daily(FakeCollector(error=True))
        self.assertEqual((self.root/'cursor.json').read_bytes(),before)
        self.assertEqual(len(list(self.root.glob('capture_*'))),2)
        self.assertFalse((self.root/'.daily.lock').exists())

    def test_start_after_cursor_is_rejected_as_a_gap(self):
        self.daily(FakeCollector(),end_at='2026-10-20T00:00:00Z')
        with self.assertRaisesRegex(ValueError,'skip'):
            self.daily(FakeCollector(),start_at='2026-10-21T00:00:00Z')

    def test_future_endpoint_and_company_pool_change_rejected(self):
        with self.assertRaisesRegex(ValueError,'future'):
            self.daily(FakeCollector(),end_at='2027-01-01T00:00:00Z')
        self.daily(FakeCollector())
        with self.assertRaisesRegex(ValueError,'company pool'):
            run_daily(self.root,['A'],collector=FakeCollector(),clock=lambda:NOW)

    def test_existing_lock_preserved(self):
        self.root.mkdir();(self.root/'.daily.lock').write_text('another-capture')
        with self.assertRaises(FileExistsError):self.daily(FakeCollector())
        self.assertEqual((self.root/'.daily.lock').read_text(),'another-capture')

    def test_user_changed_cursor_is_not_overwritten(self):
        self.daily(FakeCollector())
        class Concurrent(FakeCollector):
            def __call__(self,*a,**kw):
                result=super().__call__(*a,**kw)
                (a[0].parent/'cursor.json').write_text('user replacement')
                return result
        with self.assertRaisesRegex(ValueError,'changed'):
            self.daily(Concurrent())
        self.assertEqual((self.root/'cursor.json').read_text(),'user replacement')

    def test_exact_after_hours_and_next_preopen_news_is_collected(self):
        item=news();item.update(created_at='2026-10-20T23:00:00Z',updated_at='2026-10-20T23:00:00Z')
        session=Session([Response(bars()),Response({'news':[item],'next_page_token':None})])
        m=collect_research(self.root,['PLTR'],'2026-10-20','2026-10-21',
                           start_at='2026-10-20T16:00:00-04:00',end_at='2026-10-21T08:45:00-04:00',
                           session=session,credential_loader=lambda:('fixture-key','fixture-secret'),clock=Clock())
        self.assertEqual(m['news_summary']['unique_article_ids'],1)
        self.assertEqual(session.calls[1][1]['params']['end'],'2026-10-21T12:45:00+00:00')
        self.assertIn('after-hours',m['window']['close_policy'])
        self.assertEqual(session.calls[0][1]['params']['start'],'2026-10-20T04:00:00+00:00')

    def test_intraday_rerun_refreshes_daily_bar_from_date_boundary(self):
        session=Session([Response(bars()),Response({'news':[],'next_page_token':None})])
        m=collect_research(self.root,['PLTR'],'2026-10-21','2026-10-21',
                           start_at='2026-10-21T15:00:00Z',end_at='2026-10-21T16:00:00Z',
                           session=session,credential_loader=lambda:('fixture-key','fixture-secret'),clock=Clock())
        self.assertEqual(session.calls[0][1]['params']['start'],'2026-10-21T04:00:00+00:00')
        self.assertEqual(session.calls[1][1]['params']['start'],'2026-10-21T15:00:00+00:00')
        self.assertEqual(m['bar_window']['start'],'2026-10-21T04:00:00+00:00')

    def test_realtime_access_can_explicitly_set_zero_lag(self):
        r=self.daily(FakeCollector(),lag_minutes=0)
        self.assertEqual(r['window_end_at'],NOW.isoformat())

    def test_exact_window_requires_dates_timezone_and_both_endpoints(self):
        for options in [dict(start_at='2026-10-20T16:00:00-04:00'),
                        dict(start_at='2026-10-19T16:00:00-04:00',end_at='2026-10-21T08:45:00-04:00'),
                        dict(start_at='2026-10-20T16:00:00',end_at='2026-10-21T08:45:00-04:00')]:
            with self.assertRaises((ValueError,CollectionError)):
                collect_research(self.root,['PLTR'],'2026-10-20','2026-10-21',**options)


if __name__=='__main__':unittest.main()
