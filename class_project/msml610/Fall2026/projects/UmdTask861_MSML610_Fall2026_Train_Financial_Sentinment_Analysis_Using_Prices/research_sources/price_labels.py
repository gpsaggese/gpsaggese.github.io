"""Offline price-label contracts for the financial-text course project.

The caller supplies a versioned, complete exchange calendar, observed input
times and opening total-return indices. No calendar is inferred, data fetched,
model fitted or archive changed. Raw opening prices yield retrospective price
diagnostics only. Availability evidence is checked, not independently verified.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time, timezone
import math
from typing import Any
from zoneinfo import ZoneInfo


if __package__:
    from .corporate_actions import audit_return_series
else:
    from corporate_actions import audit_return_series


NEW_YORK = ZoneInfo("America/New_York")
HORIZONS = {5, 10, 20}
BASES = {"raw_price", "total_return_index"}
OUTCOME_FIELDS = {"stock_return", "spy_return", "sector_return", "excess_return",
                  "excess_vs_sector", "label_status", "label_available_at"}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without outer whitespace")
    return value


def _day(value: Any, name: str) -> date:
    try:
        result = date.fromisoformat(value)
        if result.isoformat() != value:
            raise ValueError()
        return result
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD)") from exc


def _time(value: Any, name: str) -> datetime:
    try:
        if not isinstance(value, str):
            raise ValueError()
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError()
        return result.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a timezone-aware timestamp") from exc


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if value else None


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _ids(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of strings")
    result = [_text(item, name) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate {name}")
    return result


def _calendar(raw: Any) -> tuple[list[dict], dict[str, int]]:
    if not isinstance(raw, dict):
        raise ValueError("calendar must be a dictionary")
    _text(raw.get("calendar_id"), "calendar_id")
    _text(raw.get("calendar_version"), "calendar_version")
    if type(raw.get("complete")) is not bool:
        raise ValueError("calendar.complete must explicitly be a boolean")
    if not isinstance(raw.get("sessions"), list) or not raw["sessions"]:
        raise ValueError("calendar.sessions must be a nonempty list")
    sessions = []
    for row in raw["sessions"]:
        if not isinstance(row, dict):
            raise ValueError("Each calendar session must be a dictionary")
        day = _day(row.get("date"), "session date")
        opening = _time(row.get("open_at"), "session open_at")
        closing = _time(row.get("close_at"), "session close_at")
        deadline = datetime.combine(day, time(9, 15), NEW_YORK)
        if (opening.astimezone(NEW_YORK).date() != day
                or closing.astimezone(NEW_YORK).date() != day
                or not deadline < opening < closing):
            raise ValueError("Session times must match its New York date and follow 09:15")
        if sessions and day.isoformat() <= sessions[-1]["date"]:
            raise ValueError("Calendar session dates must be strictly increasing and unique")
        sessions.append({"date": day.isoformat(), "open_at": opening, "close_at": closing})
    return sessions, {row["date"]: index for index, row in enumerate(sessions)}


def _prices(rows: Any) -> dict[tuple[str, str], dict]:
    if not isinstance(rows, list):
        raise ValueError("prices must be a list")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each price must be a dictionary")
        symbol = _text(row.get("symbol"), "price symbol")
        day = _day(row.get("session_date"), "price session_date").isoformat()
        key = symbol, day
        if key in result:
            raise ValueError(f"Duplicate price for {symbol} on {day}")
        basis = row.get("return_basis")
        if basis not in BASES:
            raise ValueError("Price return_basis must be raw_price or total_return_index")
        value = row.get("open_value")
        result[key] = {"value": None if value is None else _number(value, "open_value", positive=True),
                       "basis": basis,
                       "available_at": _time(row["available_at"], "price available_at") if row.get("available_at") else None,
                       "provenance_id": row.get("provenance_id"),
                       "return_series_id": row.get("return_series_id")}
        if row.get("provenance_id") is not None:
            _text(row["provenance_id"], "price provenance_id")
    return result


def _observation(row: dict, cutoff: datetime, deadline: datetime) -> tuple[dict, list, list, bool]:
    """Separate unknown provenance from known violations of the fixed clock."""
    unknown, invalid = [], []
    features = row.get("feature_values")
    if not isinstance(features, dict):
        raise ValueError("feature_values must be a dictionary of finite numeric inputs")
    for name, value in features.items():
        _text(name, "feature name")
        if name in OUTCOME_FIELDS or name.startswith(("future_", "label_", "outcome_")):
            raise ValueError("Outcome fields must not enter feature_values")
        _number(value, f"feature_values.{name}")
    provenance = row.get("feature_provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("feature_provenance must be a dictionary")
    feature_times = {}
    for name in features:
        evidence = provenance.get(name)
        if not isinstance(evidence, dict):
            unknown.append(f"feature_provenance_missing:{name}")
            continue
        if not evidence.get("available_at") or not evidence.get("provenance_id"):
            unknown.append(f"feature_provenance_missing:{name}")
        if evidence.get("provenance_id"):
            _text(evidence["provenance_id"], f"feature_provenance.{name}.provenance_id")
        if evidence.get("available_at"):
            feature_times[name] = _time(evidence["available_at"], f"feature_provenance.{name}.available_at")
            if feature_times[name] > cutoff:
                invalid.append(f"feature_available_after_cutoff:{name}")
    fields = ("published_at", "updated_at", "response_received_at",
              "extraction_completed_at", "prediction_completed_at")
    times = {}
    for field in fields:
        times[field] = _time(row[field], field) if row.get(field) else None
        if times[field] is None:
            unknown.append(f"time_missing:{field}")
        elif times[field] > (deadline if field.endswith("completed_at") else cutoff):
            invalid.append(f"after_{'deadline' if field.endswith('completed_at') else 'cutoff'}:{field}")
    for earlier, later in zip(fields, fields[1:]):
        if times[earlier] and times[later] and times[earlier] > times[later]:
            invalid.append(f"time_order:{earlier}>{later}")
    for name, available in feature_times.items():
        for field in ("extraction_completed_at", "prediction_completed_at"):
            if times[field] and available > times[field]:
                invalid.append(f"feature_available_after_{field}:{name}")
    if row.get("historical_availability_verified") is not True:
        unknown.append("historical_availability_unverified")
    verified = not unknown and not invalid
    return {"feature_values": deepcopy(features), "feature_provenance": deepcopy(provenance),
            **{field: _iso(value) for field, value in times.items()},
            "signal_available_at": _iso(max([value for value in times.values() if value]
                                             + list(feature_times.values()), default=None))}, unknown, invalid, verified


def build_labels(dataset: dict, horizon_sessions: int = 10) -> list[dict]:
    """Build one label per company/decision-day observation without I/O.

    Required dataset keys are ``as_of``, ``calendar``, ``prices`` and
    ``observations``. Calendar sessions explicitly specify ``date/open_at/
    close_at``; metadata includes ``calendar_id/calendar_version/complete``.
    Prices specify ``symbol/session_date/open_value/return_basis/available_at/
    provenance_id``. An observation specifies ``sample_id/symbol/
    decision_session/source_ids/event_cluster_ids/feature_values`` and the
    publication, revision, actual receipt and two computation times. Every
    feature needs ``feature_provenance[name]={available_at, provenance_id}``;
    that availability describes its latest raw input, not inference completion.

    ``status`` is Eligible, Pending, Unavailable or Retrospective;
    ``label_status`` is Available, Pending or Unavailable. Unknown evidence or
    raw prices or empty source/event groups cannot yield Eligible; this schema
    has no exception for a price-only observation. Returns use decimal units. The calendar
    is trusted only as the caller's versioned completeness assertion; missing
    tail sessions are never guessed. The caller must establish that supplied
    total-return indices consistently account for all corporate actions. Observed
    (or unspecified data_kind) requires return_series_evidence per symbol and
    matching prices[].return_series_id. Missing evidence stays Retrospective;
    evidence receipt delays label availability. Explicit synthetic fixtures
    exercise arithmetic only and are never observed-market evidence.
    """
    if type(horizon_sessions) is not int or horizon_sessions not in HORIZONS:
        raise ValueError("horizon_sessions must be 5, 10 or 20")
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a dictionary")
    data_kind = dataset.get("data_kind", "observed")
    if data_kind not in {"observed", "synthetic"}:
        raise ValueError("data_kind must be observed or synthetic")
    as_of = _time(dataset.get("as_of"), "as_of")
    calendar = dataset.get("calendar")
    sessions, positions = _calendar(calendar)
    prices = _prices(dataset.get("prices"))
    if not isinstance(dataset.get("observations"), list):
        raise ValueError("observations must be a list")
    result, seen, company_days = [], set(), set()
    for observation in dataset["observations"]:
        if not isinstance(observation, dict):
            raise ValueError("Each observation must be a dictionary")
        sample_id = _text(observation.get("sample_id"), "sample_id")
        symbol = _text(observation.get("symbol"), "observation symbol")
        day = _day(observation.get("decision_session"), "decision_session")
        if sample_id in seen or (symbol, day) in company_days:
            raise ValueError("Duplicate sample_id or company-decision-session observation")
        seen.add(sample_id); company_days.add((symbol, day))
        cutoff = datetime.combine(day, time(8, 45), NEW_YORK)
        deadline = datetime.combine(day, time(9, 15), NEW_YORK)
        evidence, unknown, invalid, verified = _observation(observation, cutoff, deadline)
        source_ids = _ids(observation.get("source_ids"), "source_ids")
        event_ids = _ids(observation.get("event_cluster_ids"), "event_cluster_ids")
        if not source_ids:
            unknown.append("source_ids_missing")
        if not event_ids:
            unknown.append("event_cluster_ids_missing")
        row = {"sample_id": sample_id, "symbol": symbol, "decision_session": day.isoformat(),
               "source_ids": source_ids, "event_cluster_ids": event_ids,
               **evidence, "historical_availability_verified": verified,
               "horizon_sessions": horizon_sessions, "calendar_id": calendar["calendar_id"],
               "calendar_version": calendar["calendar_version"], "as_of": _iso(as_of),
               "feature_cutoff": _iso(cutoff), "prediction_deadline": _iso(deadline),
               "entry_at": None, "exit_at": None, "label_available_at": None,
               "return_basis": None, "stock_return": None, "spy_return": None,
               "sector_symbol": observation.get("sector_symbol"), "sector_return": None,
               "excess_return": None, "excess_vs_sector": None, "return_evidence_audit": [],
               "label_status": "Unavailable", "status": "Unavailable", "reasons": []}
        if not calendar["complete"]:
            invalid.append("calendar_not_complete")
        position = positions.get(day.isoformat())
        if position is None:
            invalid.append("decision_session_not_in_calendar")
        elif position + horizon_sessions >= len(sessions):
            row["entry_at"] = _iso(sessions[position]["open_at"])
            invalid.append("calendar_horizon_missing")
        else:
            entry, exit_ = sessions[position], sessions[position + horizon_sessions]
            row.update(entry_at=_iso(entry["open_at"]), exit_at=_iso(exit_["open_at"]))
            endpoints = [prices.get((ticker, session["date"]))
                         for ticker in (symbol, "SPY") for session in (entry, exit_)]
            if as_of < exit_["open_at"]:
                row["label_status"] = "Pending"
                row["reasons"].append("horizon_not_mature")
            elif any(point is None or point["value"] is None for point in endpoints):
                invalid.append("required_open_price_missing")
            elif any(point["available_at"] is None for point in endpoints):
                invalid.append("price_availability_missing")
            elif any(point["available_at"] < session["open_at"]
                     for point, session in zip(endpoints, (entry, exit_, entry, exit_))):
                invalid.append("price_available_before_open")
            elif len({point["basis"] for point in endpoints}) != 1:
                invalid.append("inconsistent_return_basis")
            else:
                row["return_basis"] = endpoints[0]["basis"]
                available = max(exit_["open_at"], *(point["available_at"] for point in endpoints))
                price_available = available
                if row["return_basis"] == "total_return_index" and data_kind != "synthetic":
                    supplied = dataset.get("return_series_evidence", {})
                    for ticker, points in ((symbol, endpoints[:2]), ("SPY", endpoints[2:])):
                        audit = audit_return_series(
                            supplied.get(ticker) if isinstance(supplied, dict) else None,
                            symbol=ticker, entry_session=entry["date"], exit_session=exit_["date"],
                            endpoint_series_ids=[point["return_series_id"] for point in points])
                        row["return_evidence_audit"].append(audit)
                        unknown.extend(f"{reason}:{ticker}" for reason in audit["reasons"])
                        if audit["available_at"]:
                            available = max(available, _time(audit["available_at"], "return evidence availability"))
                row["label_available_at"] = _iso(available)
                if as_of < available:
                    row["label_status"] = "Pending"
                    if as_of < price_available:
                        row["reasons"].append("price_not_available_as_of")
                    if available > price_available:
                        row["reasons"].append("total_return_evidence_not_available_as_of")
                else:
                    row["label_status"] = "Available"
                    if row["return_basis"] == "raw_price":
                        unknown.append("raw_price_not_total_return")
                    if any(not point["provenance_id"] for point in endpoints):
                        unknown.append("price_provenance_unverified")
                    row["stock_return"] = endpoints[1]["value"] / endpoints[0]["value"] - 1
                    row["spy_return"] = endpoints[3]["value"] / endpoints[2]["value"] - 1
                    row["excess_return"] = row["stock_return"] - row["spy_return"]
                    for name in ("stock_return", "spy_return", "excess_return"):
                        _number(row[name], name)
                    sector = row["sector_symbol"]
                    sector_points = [prices.get((sector, session["date"])) for session in (entry, exit_)] if sector else []
                    sector_early = any(point and point["available_at"] is not None
                                       and point["available_at"] < session["open_at"]
                                       for point, session in zip(sector_points, (entry, exit_)))
                    if sector_early:
                        invalid.append("sector_price_available_before_open")
                    sector_valid = (len(sector_points) == 2 and all(point and point["value"] is not None
                            and point["available_at"] is not None and point["available_at"] <= as_of
                            and point["basis"] == row["return_basis"] for point in sector_points)
                            and not sector_early)
                    if sector_valid and row["return_basis"] == "total_return_index" and data_kind != "synthetic":
                        supplied = dataset.get("return_series_evidence", {})
                        audit = audit_return_series(
                            supplied.get(sector) if isinstance(supplied, dict) else None,
                            symbol=sector, entry_session=entry["date"], exit_session=exit_["date"],
                            endpoint_series_ids=[point["return_series_id"] for point in sector_points])
                        row["return_evidence_audit"].append(audit)
                        sector_valid = (audit["contract_eligible"] and audit["available_at"] is not None
                                        and _time(audit["available_at"], "sector return evidence availability") <= as_of)
                        if not sector_valid:
                            row["reasons"].append("sector_total_return_evidence_unavailable")
                    if sector_valid:
                        row["sector_return"] = sector_points[1]["value"] / sector_points[0]["value"] - 1
                        row["excess_vs_sector"] = row["stock_return"] - row["sector_return"]
                        _number(row["sector_return"], "sector_return")
                        _number(row["excess_vs_sector"], "excess_vs_sector")
                    elif sector:
                        row["reasons"].append("sector_return_unavailable")
        if invalid:
            row["status"] = "Unavailable"
        elif row["label_status"] == "Pending":
            row["status"] = "Pending"
        elif unknown:
            row["status"] = "Retrospective"
        elif row["label_status"] == "Available":
            row["status"] = "Eligible"
        row["historical_availability_verified"] = not unknown and not invalid
        row["reasons"] = list(dict.fromkeys([*row["reasons"], *invalid, *unknown]))
        result.append(row)
    return result


def _connected_groups(labels: list[dict]) -> dict[str, tuple[str, str] | None]:
    """Connect source/event identities using IDs alone, including omitted rows."""
    parents: dict[tuple[str, str], tuple[str, str]] = {}
    tokens_by_sample = {}

    def root(token: tuple[str, str]) -> tuple[str, str]:
        parents.setdefault(token, token)
        while parents[token] != token:
            parents[token] = parents[parents[token]]
            token = parents[token]
        return token

    for row in labels:
        tokens = [("source", item) for item in _ids(row.get("source_ids"), "source_ids")]
        tokens += [("event", item) for item in _ids(row.get("event_cluster_ids"), "event_cluster_ids")]
        tokens_by_sample[row["sample_id"]] = tokens
        if tokens:
            first = root(tokens[0])
            for token in tokens[1:]:
                other = root(token)
                if first != other:
                    parents[other] = first
    return {sample_id: root(tokens[0]) if tokens else None
            for sample_id, tokens in tokens_by_sample.items()}


def make_time_split(labels: list[dict], training_cutoff: str, test_start: str, test_end: str) -> dict:
    """Select strict mature train/test rows and purge shared source/event IDs.

    Training entry and label availability must both precede training_cutoff.
    Test entries fall in [test_start, test_end); all rows must be Eligible with
    Available labels. The training cutoff cannot follow any selected test
    prediction's actual completion or deadline. Mixing horizons, return bases
    or calendar identities/versions raises ValueError. Unknown return bases on
    Pending/Unavailable rows are allowed, but never included. Test rows take
    priority over training rows connected by any source/event ID chain. All
    supplied rows, including omitted rows, connect these groups using IDs only.
    No random split or sample-count gap is used.
    Returned ``train`` and ``test`` contain deep-copied label rows; ``excluded``
    contains each omitted sample_id with its explicit reasons.
    """
    cutoff = _time(training_cutoff, "training_cutoff")
    start, end = _time(test_start, "test_start"), _time(test_end, "test_end")
    if not cutoff <= start < end:
        raise ValueError("Require training_cutoff <= test_start < test_end")
    if not isinstance(labels, list):
        raise ValueError("labels must be a list")
    metadata = {name: set() for name in ("horizon_sessions", "return_basis", "calendar_id", "calendar_version")}
    for row in labels:
        if not isinstance(row, dict):
            raise ValueError("Each label must be a dictionary")
        horizon = row.get("horizon_sessions")
        if type(horizon) is not int or horizon not in HORIZONS:
            raise ValueError("Label horizon_sessions must be 5, 10 or 20")
        metadata["horizon_sessions"].add(horizon)
        for name in ("calendar_id", "calendar_version"):
            metadata[name].add(_text(row.get(name), name))
        basis = row.get("return_basis")
        if basis is None and row.get("label_status") in {"Pending", "Unavailable"}:
            continue
        if basis not in BASES:
            raise ValueError("Available label return_basis must be raw_price or total_return_index")
        metadata["return_basis"].add(basis)
    for name, values in metadata.items():
        if len(values) > 1:
            raise ValueError(f"Mixed {name} values in label pool")
    train, test, excluded, seen = [], [], [], set()
    for row in labels:
        sample_id = _text(row.get("sample_id"), "sample_id")
        if sample_id in seen:
            raise ValueError("Duplicate sample_id in labels")
        seen.add(sample_id)
        reasons = []
        if row.get("status") != "Eligible":
            reasons.append("sample_not_eligible:" + str(row.get("status")))
        if row.get("label_status") != "Available":
            reasons.append("label_not_available:" + str(row.get("label_status")))
        if reasons:
            excluded.append({"sample_id": sample_id, "reasons": reasons})
            continue
        entry = _time(row.get("entry_at"), "entry_at")
        maturity = _time(row.get("label_available_at"), "label_available_at")
        if entry < cutoff:
            if maturity >= cutoff:
                excluded.append({"sample_id": sample_id, "reasons": ["training_label_not_mature_before_cutoff"]})
            else:
                train.append(deepcopy(row))
        elif start <= entry < end:
            completed = _time(row.get("prediction_completed_at"), "prediction_completed_at")
            deadline = _time(row.get("prediction_deadline"), "prediction_deadline")
            if cutoff > completed or cutoff > deadline:
                raise ValueError(f"training_cutoff follows test prediction completion or deadline: {sample_id}")
            test.append(deepcopy(row))
        else:
            excluded.append({"sample_id": sample_id, "reasons": ["outside_requested_windows"]})
    test_sources = {item for row in test for item in _ids(row.get("source_ids"), "source_ids")}
    test_events = {item for row in test for item in _ids(row.get("event_cluster_ids"), "event_cluster_ids")}
    groups = _connected_groups(labels)
    test_groups = {groups[row["sample_id"]] for row in test if groups[row["sample_id"]] is not None}
    purged = []
    for row in train:
        reasons = []
        if test_sources.intersection(_ids(row.get("source_ids"), "source_ids")):
            reasons.append("source_shared_with_test")
        if test_events.intersection(_ids(row.get("event_cluster_ids"), "event_cluster_ids")):
            reasons.append("event_cluster_shared_with_test")
        if not reasons and groups[row["sample_id"]] in test_groups:
            reasons.append("transitive_group_shared_with_test")
        if reasons:
            excluded.append({"sample_id": row["sample_id"], "reasons": reasons})
        else:
            purged.append(row)
    return {"train": purged, "test": test, "excluded": excluded,
            "training_cutoff": _iso(cutoff), "test_start": _iso(start), "test_end": _iso(end),
            "target": "excess_return", "units": "decimal", "is_trading_evaluation": False}
