"""Event provenance, deduplication and no-metadata-direction research checks."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from semantic_signal_gate import (GatePolicy,Horizon,SemanticCalibration,StructuredEvent,
                                 apply_semantic_overlay,evaluate_events)

H30=Horizon('minutes',30)
DECISION='2026-10-23T14:00:00Z'


def verified_event(**updates):
    event=StructuredEvent(source_id='doc1',source='news',target_symbol='NVDA',
        published_at='2026-10-23T13:30:00Z',received_at='2026-10-23T13:31:00Z',
        receipt_verified=True,capture_id='capture1',event_id='event1',event_identity_verified=True,
        subject_symbol='NVDA',target_relation='direct_subject',relation_verified=True,
        direction=1,direction_verified=True,horizon=H30,horizon_verified=True,is_question=False,
        is_reported_opinion=False,is_conditional_or_negated=False,
        annotation_available_at='2026-10-23T13:32:00Z',annotation_clock_verified=True,
        annotation_receipt_id='review1',text_for_question_guard='Verified company-specific event')
    return replace(event,**updates)


def gate(events,**options):
    return evaluate_events(events,symbol='NVDA',decision_at=DECISION,horizon=H30,**options)


def calibration(**updates):
    value=SemanticCalibration(artifact_id='saved-prior-calibration',target_symbol='NVDA',horizon=H30,
        training_cutoff='2026-10-22T04:00:00Z',latest_label_end='2026-10-21T19:55:00Z',
        training_event_ids=tuple('prior_event_'+str(i) for i in range(20)),
        training_receipt_dates=tuple(f'2026-10-{i:02d}' for i in range(1,11)),
        event_groups_reviewed=True,return_per_signed_feature=.0002)
    return replace(value,**updates)


class SemanticGateTests(unittest.TestCase):
    def test_verified_target_event_and_prior_calibration_can_pass(self):
        result=gate([verified_event()])
        self.assertEqual(result['qualified_event_count'],1)
        self.assertAlmostEqual(result['semantic_feature'],2**(-.5))
        overlay=apply_semantic_overlay(.001,result,calibration())
        self.assertFalse(overlay['exact_C_fallback'])
        self.assertGreater(overlay['prediction'],.001)
        self.assertAlmostEqual(overlay['adjustment'],.0002*2**(-.5))

    def test_tags_or_metadata_cannot_create_a_direction(self):
        event=StructuredEvent(source_id='1',source='news',target_symbol='NVDA',
                               published_at='2026-10-23T13:30:00Z',received_at='2026-10-23T13:31:00Z',
                               receipt_verified=True,capture_id='capture')
        result=gate([event]);self.assertIsNone(result['semantic_feature'])
        self.assertIn('target_subject_relation_unverified',result['rejection_counts'])
        self.assertEqual(apply_semantic_overlay(.001,result,calibration())['prediction'],.001)

    def test_no_text_is_unknown_and_zero_is_not_certified_neutral(self):
        self.assertIsNone(gate([])['semantic_feature'])
        result=gate([verified_event(direction=0)])
        self.assertIsNone(result['semantic_feature'])
        self.assertIn('direction_unknown_or_unverified',result['rejection_counts'])

    def test_age_only_attenuates_verified_direction(self):
        early=gate([verified_event()])
        older=gate([verified_event(published_at='2026-10-23T13:00:00Z')])
        self.assertGreater(early['semantic_feature'],older['semantic_feature'])
        self.assertGreater(older['semantic_feature'],0)
        unknown=gate([verified_event(direction=None,direction_verified=False)])
        self.assertIsNone(unknown['semantic_feature'])

    def test_duplicate_event_does_not_add_independent_votes(self):
        result=gate([verified_event(),verified_event(source_id='syndicated-doc')])
        self.assertEqual(result['qualified_event_count'],1)
        self.assertEqual(result['rejection_counts']['duplicate_independent_event'],1)
        self.assertAlmostEqual(result['semantic_feature'],gate([verified_event()])['semantic_feature'])

    def test_conflicting_same_event_is_not_averaged_as_neutral(self):
        result=gate([verified_event(),verified_event(source_id='other-doc',direction=-1)])
        self.assertEqual(result['qualified_event_count'],0)
        self.assertIsNone(result['semantic_feature'])

    def test_chinese_question_particles_cannot_be_bullish_assertions(self):
        for text in ('應該買入溝淡嗎','应该买入吗','Buy now?'):
            result=gate([verified_event(text_for_question_guard=text)])
            self.assertEqual(result['qualified_event_count'],0)
            self.assertIn('question_or_unverified_assertion',result['rejection_counts'])

    def test_reported_opinion_and_wrong_target_relation_are_rejected(self):
        for event in (verified_event(is_reported_opinion=True),
                      verified_event(subject_symbol='META'),verified_event(relation_verified=False)):
            self.assertEqual(gate([event])['qualified_event_count'],0)

    def test_late_annotation_is_not_backfilled_into_old_decision(self):
        for event in (verified_event(annotation_available_at='2026-10-23T14:01:00Z'),
                      verified_event(annotation_available_at=None,annotation_clock_verified=False),
                      verified_event(received_at='2026-10-23T14:01:00Z')):
            self.assertEqual(gate([event])['qualified_event_count'],0)

    def test_minutes_and_sessions_cannot_be_mixed(self):
        result=gate([verified_event(horizon=Horizon('sessions',10))])
        self.assertIn('event_horizon_unknown_or_mismatched',result['rejection_counts'])
        result=apply_semantic_overlay(.001,gate([verified_event()]),calibration(horizon=Horizon('sessions',10)))
        self.assertEqual(result['prediction'],.001)

    def test_bar_count_or_duplicate_event_ids_do_not_pass_coverage(self):
        result=apply_semantic_overlay(.001,gate([verified_event()]),
            calibration(training_event_ids=('same_event',)*337,training_receipt_dates=('2026-10-21',)*5))
        self.assertEqual(result['prediction'],.001)
        self.assertIn('insufficient_independent_training_events',result['fallback_reasons'])
        self.assertIn('insufficient_training_receipt_dates',result['fallback_reasons'])

    def test_future_calibration_or_unmature_outcomes_fall_back(self):
        for artifact in (calibration(training_cutoff='2026-10-23T14:01:00Z'),
                         calibration(latest_label_end='2026-10-22T05:00:00Z')):
            self.assertEqual(apply_semantic_overlay(.001,gate([verified_event()]),artifact)['prediction'],.001)

    def test_explicit_reviewed_spillover_can_pass_without_guessing_from_tags(self):
        event=verified_event(subject_symbol='META',target_relation='explicit_spillover')
        self.assertEqual(gate([event])['qualified_event_count'],1)
        self.assertEqual(gate([replace(event,relation_verified=False)])['qualified_event_count'],0)

    def test_truthy_strings_are_not_verification_booleans(self):
        for field in ('event_identity_verified','relation_verified','direction_verified',
                      'horizon_verified','receipt_verified','annotation_clock_verified'):
            self.assertEqual(gate([verified_event(**{field:'false'})])['qualified_event_count'],0)
        overlay=apply_semantic_overlay(.001,gate([verified_event()]),calibration(event_groups_reviewed='false'))
        self.assertEqual(overlay['prediction'],.001)


if __name__=='__main__':unittest.main()
