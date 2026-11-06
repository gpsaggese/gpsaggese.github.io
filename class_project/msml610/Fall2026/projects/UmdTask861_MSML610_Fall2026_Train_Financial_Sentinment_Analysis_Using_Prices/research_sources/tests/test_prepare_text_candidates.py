"""Artificial timing/group fixtures; no synthetic return/model evaluation."""
from copy import deepcopy
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
# Use explicit artificial calendar fixtures already exercised by label tests.
from research.financial_sentiment_using_prices.tests.test_price_labels import dataset,DAYS
from prepare_text_candidates import prepare_candidates


def article(identity='a',**changes):
    row={'id':identity,'symbols':['PLTR','MSFT'],'created_at':'2026-10-01T10:00:00Z',
         'updated_at':'2026-10-01T10:01:00Z','retrieved_at':'2026-10-01T10:02:00Z',
         'headline':'Company test headline','text':'Artificial body text','url':'https://example.test/a'}
    row.update(changes);return row


class TextCandidateTests(unittest.TestCase):
    def prepare(self,articles=None,**changes):
        args={'symbols':['PLTR','AMZN','MSFT'],'decision_session':DAYS[1],
              'as_of':'2026-10-01T23:00:00Z'};args.update(changes)
        return prepare_candidates(articles if articles is not None else [article()],dataset()['calendar'],**args)

    def test_retained_shared_text_is_not_invented_as_two_independent_events(self):
        report=self.prepare();a,b=report['candidates'][0],report['candidates'][2]
        self.assertEqual(a['source_ids'],b['source_ids']);self.assertIsNone(a['source_event_ids'])
        self.assertEqual(report['summary']['article_company_pairs'],2)
        self.assertEqual(report['summary']['eligible_training_samples'],0)
        self.assertIsNone(a['prediction_completed_at'])

    def test_later_revision_or_receipt_is_not_backfilled(self):
        for changes in [{'updated_at':'2026-10-02T13:00:00Z','retrieved_at':'2026-10-02T13:01:00Z'},
                        {'retrieved_at':'2026-10-02T13:01:00Z'}]:
            report=self.prepare([article(**changes)])
            self.assertEqual(report['summary']['time_qualified_unique_articles'],0)
            self.assertTrue(report['excluded_articles'][0]['reasons'])

    def test_created_today_does_not_claim_a_past_prediction(self):
        report=self.prepare(as_of='2026-10-02T20:00:00Z');row=report['candidates'][0]
        self.assertFalse(row['input_frozen_before_cutoff'])
        self.assertIn('input_freeze_after_decision_cutoff',row['blockers'])
        self.assertFalse(row['eligible_for_training'])

    def test_empty_company_or_body_remains_explicit(self):
        report=self.prepare([article(text='')]);empty=report['candidates'][1]
        self.assertIn('no_retained_company_text',empty['blockers'])
        self.assertEqual(report['candidates'][0]['body_article_company_pairs'],0)
        self.assertEqual(report['candidates'][0]['texts'][0]['body'],'')

    def test_exact_horizon_exit_still_has_no_return_label(self):
        row=self.prepare()['candidates'][0]
        self.assertEqual(row['horizons']['10']['exit_session'],DAYS[11])
        self.assertIsNone(row['horizons']['10']['return_value'])
        self.assertEqual(row['horizons']['10']['label_status'],'Not constructed')

    def test_missing_calendar_tail_is_not_guessed(self):
        report=self.prepare(decision_session=DAYS[-2]);row=report['candidates'][0]
        self.assertIn('calendar_tail_missing',row['blockers'])
        self.assertIsNone(row['horizons']['10']['exit_at'])

    def test_duplicate_article_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.prepare([article(),article()])

    def test_inputs_remain_unchanged_and_no_text_training_is_triggered(self):
        articles=[article()];before=deepcopy(articles);report=self.prepare(articles)
        self.assertEqual(articles,before);self.assertIn('Not run',report['summary']['baseline_status'])
        self.assertEqual(report['summary']['labels_created'],0)


if __name__=='__main__':unittest.main()
