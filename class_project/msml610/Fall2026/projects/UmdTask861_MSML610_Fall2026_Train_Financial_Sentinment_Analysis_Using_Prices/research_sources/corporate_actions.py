"""Offline cash-dividend date and supplied total-return evidence contracts.

These functions neither query Alpaca nor reconstruct a total-return index.
Process-date query coverage cannot certify ex-date event completeness. Evidence
checks validate caller declarations, not external source authenticity.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation


def _day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('Expected an explicit ISO date')
    return value


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _time(value):
    if not isinstance(value, str):
        raise ValueError('Expected an aware evidence timestamp')
    clock = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if clock.tzinfo is None or clock.utcoffset() is None:
        raise ValueError('Expected an aware evidence timestamp')
    return clock.astimezone(timezone.utc)


def audit_cash_dividends(records, *, symbol, entry_session, exit_session, query):
    """Select normalized cash events by entry < ex_date <= exit for open returns.

    On ex-date, a buyer at the entry open has no earlier entitlement; a holder
    exiting at that open has held across the ex-date boundary. Payment/record/
    process dates remain distinct metadata. Special due-bill distributions and
    non-cash actions require a separate reviewed treatment and are unsupported.
    Records require provider-stable event_id, ex_date, decimal cash_per_share,
    currency and the explicit standard_cash_dividend treatment.
    """
    entry, exit_ = _day(entry_session), _day(exit_session)
    if not _text(symbol) or entry >= exit_ or not isinstance(records, list):
        raise ValueError('Require symbol, increasing entry/exit sessions and a record list')
    if (not isinstance(query, dict) or query.get('date_basis') != 'process_date'
            or _day(query.get('start')) > _day(query.get('end'))):
        raise ValueError('Alpaca query range must explicitly describe process_date')
    unique, economics, duplicates = {}, {}, []
    for original in records:
        if not isinstance(original, dict) or not all(_text(original.get(k)) for k in ('event_id','symbol','currency')):
            raise ValueError('Require stable event_id, symbol and currency')
        if original.get('treatment') != 'standard_cash_dividend':
            raise ValueError('Unsupported corporate-action/dividend treatment')
        ex = _day(original.get('ex_date'))  # Never fall back to process or pay.
        dates = {k: _day(original[k]) if original.get(k) is not None else None
                 for k in ('record_date','payment_date','process_date')}
        try:
            amount = Decimal(str(original['cash_per_share']))
        except (KeyError, InvalidOperation, ValueError) as exc:
            raise ValueError('Require explicit positive finite cash_per_share') from exc
        if not amount.is_finite() or amount <= 0 or isinstance(original['cash_per_share'], bool):
            raise ValueError('Require explicit positive finite cash_per_share')
        record = {'event_id':original['event_id'], 'symbol':original['symbol'],
                  'ex_date':ex, **dates, 'cash_per_share':str(amount.normalize()),
                  'currency':original['currency'], 'treatment':original['treatment']}
        # Source/retrieval metadata may differ between overlapping captures;
        # stable IDs with identical economics count once. Revisions conflict.
        identity = record['event_id']
        economic = tuple(record[k] for k in ('symbol','ex_date','cash_per_share','currency'))
        if identity in unique:
            if record != unique[identity]:
                raise ValueError('Conflicting versions of a cash-dividend event')
            duplicates.append(identity)
            continue
        if economic in economics:
            raise ValueError('Ambiguous duplicate economics under different event IDs')
        economics[economic] = identity
        unique[identity] = record
    retained = sorted(unique.values(), key=lambda r:(r['ex_date'],r['event_id']))
    return {'symbol':symbol, 'entry_session':entry, 'exit_session':exit_,
            'entitlement_interval':'entry_session < ex_date <= exit_session',
            'query':dict(query), 'query_coverage_basis':'process_date',
            'ex_date_coverage_verified':False,
            'retained_records':retained, 'duplicate_capture_ids':duplicates,
            'selected_event_ids':[r['event_id'] for r in retained if r['symbol']==symbol and entry < r['ex_date'] <= exit_],
            'labels_created':0,
            'limitations':['A complete process-date response cannot establish complete ex-date coverage.',
                           'No dividend is added to prices or indices; payment is not a second return event.',
                           'Normalized fields and standard entitlement require independent source review.']}


def audit_return_series(evidence, *, symbol, entry_session, exit_session, endpoint_series_ids):
    """Check a vendor total-return-index contract over one label interval.

    Raw/split-only/dividend-adjusted price series are not certified by their
    adjustment names. This minimum contract supports only externally supplied
    total-return indices with cash/splits embedded exactly once and independently
    reviewed ex-date coverage. No index conversion or extra cash addition occurs.
    """
    entry, exit_ = _day(entry_session), _day(exit_session)
    reasons = []
    if not isinstance(evidence, dict):
        return {'symbol':symbol, 'contract_eligible':False, 'available_at':None,
                'reasons':['total_return_evidence_missing']}
    if evidence.get('method') != 'vendor_total_return_index':
        reasons.append('unsupported_total_return_method')
    series = evidence.get('series_id')
    if not _text(series) or endpoint_series_ids != [series, series]:
        reasons.append('total_return_series_binding_missing_or_inconsistent')
    if not _text(evidence.get('provenance_id')):
        reasons.append('total_return_provenance_missing')
    if evidence.get('cash_dividends_embedded') is not True or evidence.get('splits_embedded') is not True:
        reasons.append('total_return_action_treatment_unverified')
    if evidence.get('additional_cash_dividends_added') is not False:
        reasons.append('cash_dividend_double_count_or_unknown')
    coverage = evidence.get('ex_date_coverage')
    if not isinstance(coverage, dict):
        reasons.append('ex_date_coverage_evidence_missing')
    else:
        if (coverage.get('date_basis') != 'ex_date' or coverage.get('reviewed_complete') is not True
                or not _text(coverage.get('evidence_id'))):
            reasons.append('ex_date_coverage_unverified')
        try:
            if not _day(coverage.get('start')) <= entry < exit_ <= _day(coverage.get('end')):
                reasons.append('ex_date_coverage_incomplete')
        except (ValueError, TypeError):
            reasons.append('ex_date_coverage_dates_missing_or_invalid')
    available = None
    try:
        available = _time(evidence.get('available_at')).isoformat().replace('+00:00','Z')
    except (ValueError, TypeError, OverflowError):
        reasons.append('total_return_evidence_availability_missing_or_invalid')
    return {'symbol':symbol, 'series_id':series, 'contract_eligible':not reasons,
            'available_at':available, 'reasons':reasons,
            'scope':'Caller-supplied evidence contract; not independent source certification'}
