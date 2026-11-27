"""Conservative research gate for already structured and verified text events.

This is NOT an NLP extractor, model trainer or trading integration. Callers must
retain evidence supporting the supplied entity, event, direction and clock
fields. Source ticker tags alone do not fill those fields. Metadata cannot create
a directional return; age only attenuates a verified semantic feature. Thresholds
below are configurable engineering gates, not statistical power or alpha claims.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo('America/New_York')


def _time(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Timezone-aware clocks are required')
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class Horizon:
    unit: str
    value: int

    def __post_init__(self):
        if self.unit not in {'minutes', 'sessions'} or type(self.value) is not int or self.value <= 0:
            raise ValueError('Use an explicit positive minutes/sessions horizon; do not convert implicitly')


@dataclass(frozen=True)
class StructuredEvent:
    source_id: str
    source: str
    target_symbol: str
    published_at: str
    received_at: str | None
    updated_at: str | None = None
    receipt_verified: bool = False
    capture_id: str | None = None
    # These fields require a retained extraction/review result, not ticker tags.
    event_id: str | None = None
    event_identity_verified: bool = False
    subject_symbol: str | None = None
    target_relation: str | None = None  # direct_subject or explicit_spillover
    relation_verified: bool = False
    direction: int | None = None  # unknown is None; 0 is not certified neutral
    direction_verified: bool = False
    horizon: Horizon | None = None
    horizon_verified: bool = False
    is_question: bool | None = None
    is_reported_opinion: bool | None = None
    is_conditional_or_negated: bool | None = None
    annotation_available_at: str | None = None
    annotation_clock_verified: bool = False
    annotation_receipt_id: str | None = None
    text_for_question_guard: str = ''


@dataclass(frozen=True)
class GatePolicy:
    max_age_minutes: float = 240.0
    half_life_minutes: float = 60.0
    minimum_training_events: int = 20
    minimum_training_receipt_dates: int = 10
    max_abs_return_adjustment: float = 0.0005

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.max_age_minutes, self.half_life_minutes, self.max_abs_return_adjustment)):
            raise ValueError('Positive finite age, decay and adjustment limits required')
        if any(type(v) is not int or v < 1 for v in
               (self.minimum_training_events, self.minimum_training_receipt_dates)):
            raise ValueError('Positive explicit engineering coverage gates required')


@dataclass(frozen=True)
class SemanticCalibration:
    artifact_id: str
    target_symbol: str
    horizon: Horizon
    training_cutoff: str
    latest_label_end: str
    training_event_ids: tuple[str, ...]
    training_receipt_dates: tuple[str, ...]
    event_groups_reviewed: bool
    # Must be estimated externally from prior, same-horizon residual outcomes.
    # There is intentionally no intercept or metadata term in this overlay.
    return_per_signed_feature: float


def evaluate_events(events, *, symbol, decision_at, horizon, policy=GatePolicy()):
    """Return qualified event features and explicit unknown/rejection reasons.

    An eligible event can pass even when calibration coverage is not yet enough.
    Output semantic_feature is bounded [-1,1], never a probability or return.
    No eligible events => None, not an invented neutral sentiment score.
    """
    if not isinstance(horizon, Horizon):
        raise ValueError('An explicit Horizon is required')
    decision = _time(decision_at)
    eligible, rejected = [], []
    for event in events:
        reasons = []
        if not event.source or not event.source_id:
            reasons.append('source_identity_missing')
        if event.target_symbol != symbol:
            reasons.append('target_symbol_mismatch')
        if event.event_identity_verified is not True or not event.event_id:
            reasons.append('independent_event_identity_unverified')
        if (event.relation_verified is not True or not event.subject_symbol
                or event.target_relation not in {'direct_subject', 'explicit_spillover'}
                or (event.target_relation == 'direct_subject' and event.subject_symbol != symbol)):
            reasons.append('target_subject_relation_unverified')
        if event.direction_verified is not True or type(event.direction) is not int or event.direction not in {-1, 1}:
            reasons.append('direction_unknown_or_unverified')
        if event.horizon_verified is not True or event.horizon != horizon:
            reasons.append('event_horizon_unknown_or_mismatched')
        if event.is_question is not False or any(c in event.text_for_question_guard for c in ('?', '？', '吗', '嗎')):
            reasons.append('question_or_unverified_assertion')
        if event.is_reported_opinion is not False:
            reasons.append('reported_opinion_or_unverified_voice')
        if event.is_conditional_or_negated is not False:
            reasons.append('conditional_negated_or_unverified_statement')
        if event.receipt_verified is not True or not event.received_at or not event.capture_id:
            reasons.append('observed_receipt_unverified')
        if event.annotation_clock_verified is not True or not event.annotation_available_at or not event.annotation_receipt_id:
            reasons.append('semantic_annotation_availability_unverified')
        age = None
        try:
            published = _time(event.published_at)
            updated = _time(event.updated_at or event.published_at)
            if updated < published:
                reasons.append('invalid_publication_update_order')
            clocks = [published, updated]
            if event.received_at:
                clocks.append(_time(event.received_at))
            if event.annotation_available_at:
                annotation = _time(event.annotation_available_at)
                clocks.append(annotation)
                if event.received_at and annotation < _time(event.received_at):
                    reasons.append('annotation_precedes_observed_receipt')
            if max(clocks) > decision:
                reasons.append('not_available_before_decision')
            age = (decision-published).total_seconds()/60
            if age > policy.max_age_minutes:
                reasons.append('outside_declared_event_age_limit')
        except (ValueError, TypeError, AttributeError):
            reasons.append('invalid_event_clock')
        if reasons:
            rejected.append(dict(source_id=event.source_id,event_id=event.event_id,reasons=sorted(set(reasons))))
        else:
            eligible.append((event, age))
    groups = {}
    for event,age in eligible:
        groups.setdefault(event.event_id,[]).append((event,age))
    retained=[]
    for event_id, versions in sorted(groups.items()):
        # Multiple unresolved stances about one event are not independent votes.
        if len({e.direction for e,_ in versions}) > 1:
            rejected.extend(dict(source_id=e.source_id,event_id=event_id,reasons=['conflicting_direction_within_event']) for e,_ in versions)
            continue
        selected=max(versions,key=lambda item:(_time(item[0].updated_at or item[0].published_at),
                                               _time(item[0].annotation_available_at),item[0].source_id))
        for item in versions:
            if item is not selected:
                rejected.append(dict(source_id=item[0].source_id,event_id=event_id,reasons=['duplicate_independent_event']))
        event,age=selected
        decay=math.exp(-math.log(2)*age/policy.half_life_minutes)
        retained.append(dict(event_id=event_id,source_id=event.source_id,direction=event.direction,
                             decay=decay,signed_feature=event.direction*decay,capture_id=event.capture_id))
    feature=sum(e['signed_feature'] for e in retained)/len(retained) if retained else None
    return dict(symbol=symbol,decision_at=decision.isoformat(),horizon=dict(unit=horizon.unit,value=horizon.value),
                qualified_event_count=len(retained),semantic_feature=feature,events=retained,rejected=rejected,
                rejection_counts=dict(Counter(reason for r in rejected for reason in r['reasons'])),
                extraction_performed=False,return_prediction_performed=False,
                interpretation='Caller-supplied verified event fields are required; this gate does not extract or independently certify them. No alpha claim.')


def apply_semantic_overlay(base_prediction, gate, calibration=None, *, policy=GatePolicy()):
    """No-intercept same-horizon overlay; invalid coverage gives exact C fallback."""
    if type(base_prediction) not in (int,float) or not math.isfinite(base_prediction):
        raise ValueError('Finite C forecast required')
    reasons=[]
    if gate['qualified_event_count']==0 or gate['semantic_feature'] is None:
        reasons.append('no_qualified_semantic_event')
    elif (gate['qualified_event_count']!=len(gate['events'])
          or type(gate['semantic_feature']) not in (int,float)
          or not math.isfinite(gate['semantic_feature']) or abs(gate['semantic_feature'])>1):
        reasons.append('invalid_qualified_semantic_feature')
    if calibration is None:
        reasons.append('no_semantic_calibration')
    else:
        if calibration.target_symbol!=gate['symbol'] or dict(unit=calibration.horizon.unit,value=calibration.horizon.value)!=gate['horizon']:
            reasons.append('calibration_symbol_or_horizon_mismatch')
        if not calibration.artifact_id or calibration.event_groups_reviewed is not True:
            reasons.append('calibration_event_groups_unverified')
        if any(not isinstance(item,str) or not item for item in calibration.training_event_ids):
            reasons.append('invalid_training_event_ids')
        if len(set(calibration.training_event_ids))<policy.minimum_training_events:
            reasons.append('insufficient_independent_training_events')
        if len(set(calibration.training_receipt_dates))<policy.minimum_training_receipt_dates:
            reasons.append('insufficient_training_receipt_dates')
        try:
            cutoff,decision=_time(calibration.training_cutoff),_time(gate['decision_at'])
            if not _time(calibration.latest_label_end)<cutoff<decision:
                reasons.append('calibration_not_prior_and_mature')
            if any(date.fromisoformat(day)>cutoff.astimezone(NY).date() for day in calibration.training_receipt_dates):
                reasons.append('training_receipt_date_after_cutoff')
        except (ValueError,TypeError,AttributeError):
            reasons.append('invalid_calibration_clock')
        if type(calibration.return_per_signed_feature) not in (int,float) or not math.isfinite(calibration.return_per_signed_feature):
            reasons.append('invalid_calibrated_coefficient')
    delta=0.0
    if not reasons:
        raw=calibration.return_per_signed_feature*gate['semantic_feature']
        delta=max(-policy.max_abs_return_adjustment,min(policy.max_abs_return_adjustment,raw))
    return dict(base_prediction=base_prediction,prediction=base_prediction+delta,adjustment=delta,
                fallback_reasons=reasons,exact_C_fallback=bool(reasons),
                alpha_established=False,models_fit=0)


def audit_frozen_decisions(comparison):
    """Missing semantic fields stay unknown; do not manufacture retrospective NLP."""
    path=Path(comparison)/'decisions.jsonl.gz'
    with gzip.open(path,'rt') as stream:
        rows=[json.loads(line) for line in stream]
    totals=Counter(); reasons=Counter(); detail=[]
    for row in rows:
        if not row['has_text']:
            continue
        events=[StructuredEvent(source_id=e['source_id'],source=e['source'],target_symbol=row['symbol'],
                    published_at=e['published_at'],updated_at=e['updated_at'],received_at=e['received_at'],
                    receipt_verified=e['receipt_verified'],
                    capture_id=hashlib.sha256(e['provenance']['response_file'].encode()).hexdigest()) for e in row['evidence']]
        gate=evaluate_events(events,symbol=row['symbol'],decision_at=row['decision_at'],horizon=Horizon('minutes',30))
        overlay=apply_semantic_overlay(row['C_mu'],gate)
        old_applied=row['text_fallback'] is None
        totals.update(text_decisions=1,old_applied_decisions=int(old_applied),
                      new_qualified_decisions=int(gate['qualified_event_count']>0),
                      old_applied_still_qualified=int(old_applied and gate['qualified_event_count']>0),
                      semantic_prediction_equals_C=int(overlay['prediction']==row['C_mu']))
        reasons.update(gate['rejection_counts'])
        detail.append(dict(symbol=row['symbol'],decision_at=row['decision_at'],old_applied=old_applied,
                           qualified_events=gate['qualified_event_count'],semantic_feature=gate['semantic_feature'],
                           semantic_adjustment=overlay['adjustment'],same_C_prediction=overlay['prediction']==row['C_mu'],
                           rejection_counts=gate['rejection_counts']))
    return dict(schema_version=1,input_sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest()},
                total_decisions=len(rows),**totals,rejected_evidence_occurrences=dict(reasons),rows=detail,
                local_training_performed=False,portfolio_replayed=False,original_comparison_modified=False,
                interpretation=[
                    'Zero eligibility means the archive lacks verified structured event identity, target relation, direction, horizon and annotation availability; it does not prove the articles irrelevant.',
                    'NLP extraction plus independent annotation/review and retained processing clocks are still required. Provider tags are not replacements.',
                    'Semantic-only adjustments are zero at the saved decision level and predictions equal C. No new portfolio-return result is claimed.',
                    'Default 20-event/10-receipt-date calibration gates and four-hour event-age limit are explicit provisional engineering settings, not statistical power or alpha evidence.',
                    'Original 48-hour metadata adapter and its negative comparison remain unchanged.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--audit-output',type=Path,required=True)
    args=parser.parse_args()
    if args.audit_output.exists() or args.audit_output.resolve().is_relative_to(args.comparison.resolve()):
        raise ValueError('Write a new independent audit outside the frozen comparison')
    report=audit_frozen_decisions(args.comparison)
    report['gate_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.audit_output.parent.mkdir(parents=True,exist_ok=True)
    args.audit_output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    args.audit_output.chmod(0o600)
    print(json.dumps({k:v for k,v in report.items() if k in ('total_decisions','text_decisions','old_applied_decisions','new_qualified_decisions','old_applied_still_qualified','semantic_prediction_equals_C')}))


if __name__=='__main__':main()
