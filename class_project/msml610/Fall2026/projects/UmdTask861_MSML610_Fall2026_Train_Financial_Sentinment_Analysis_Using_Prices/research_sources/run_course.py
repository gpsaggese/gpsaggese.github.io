"""One offline entry point for archive readiness and existing price/tone controls.

Archive mode audits retained news/prices, without inventing a company-day adapter.
Prepared mode delegates labels, chronological event isolation and paired numeric
controls to evaluate_dataset. Text-price models, LM and attribution remain pending.
Importing or running this module never collects data or starts the trading app.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from data_preflight import audit_archives
from price_baselines import evaluate_dataset, synthetic_dataset


MODULES = ("run_course.py", "data_preflight.py", "run_research.py", "event_extraction.py",
           "price_context.py", "price_labels.py", "price_baselines.py", "corporate_actions.py")
PREPARED_CONTRACT = {
    "data_kind": "Explicit synthetic or observed declaration; not independent verification",
    "as_of": "Timezone-aware evaluation clock; outcomes must actually be available by it",
    "calendar": {
        "calendar_id": "Independent exchange-calendar identity",
        "calendar_version": "Retained rule/data version", "complete": "Explicit completeness declaration",
        "sessions": "Ordered date/open_at/close_at records; no inference from weekdays or observed bars",
    },
    "prices": {
        "required_fields": ["symbol", "session_date", "open_value", "return_basis", "available_at", "provenance_id"],
        "basis_evidence": "Independently audited opening total-return indices; raw_price stays retrospective",
    },
    "return_series_evidence": "Observed total-return inputs require per-symbol series binding, embedded cash/split treatment, no added dividends, independently reviewed ex-date coverage and actual evidence availability; process-date queries alone do not certify coverage",
    "observations": {
        "required_fields": ["sample_id", "symbol", "decision_session", "source_ids", "event_cluster_ids",
                            "published_at", "updated_at", "response_received_at", "extraction_completed_at",
                            "prediction_completed_at", "historical_availability_verified", "feature_values"],
        "feature_provenance": "Every feature key needs actual available_at and provenance_id; raw inputs by 08:45 New York, calculation by 09:15",
        "availability_declaration": "Unknown history must not be converted into verified=True; source audit required",
    },
    "evaluation": {"training_cutoff": "Mature training endpoint", "test_start": "Later test entry start",
                   "test_end": "Exclusive test entry end"},
    "baseline_features": ["momentum_5d", "momentum_20d", "tone_score"],
    "external_evidence": ["Corporate actions and consistent opening total-return indices",
                          "Source-version and actual availability audit",
                          "Cross-article/company event identity and independent review"],
    "source_family_evidence": {
        "families": ["financial_report", "earnings_call", "news", "investor_comment"],
        "audit": "data_preflight.audit_source_records; separate source ID, event IDs, original publication, exact text version and bound actual receipt",
        "financial_dates": "Explicit event calendar date; fiscal-year labels and report-period ends are not event timestamps",
    },
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def missing_fields(dataset: dict) -> tuple[list[str], list[str]]:
    """List absent contract fields, leaving all validity decisions to core code.

    Missing per-row evidence is retained for label-status diagnostics; only
    missing schema scaffolding prevents invoking the existing evaluator.
    """
    scaffold, evidence = [], []
    if not isinstance(dataset, dict):
        return ["dataset (JSON object)"], []
    for name in ("data_kind", "as_of", "calendar", "prices", "observations", "evaluation"):
        if name not in dataset or dataset[name] is None:
            scaffold.append(name)
    for parent, names in (("calendar", ("calendar_id", "calendar_version", "complete", "sessions")),
                          ("evaluation", ("training_cutoff", "test_start", "test_end"))):
        value = dataset.get(parent)
        if value is not None and not isinstance(value, dict):
            scaffold.append(parent + " (object)")
        elif isinstance(value, dict):
            scaffold.extend(f"{parent}.{name}" for name in names if name not in value or value[name] is None)
    calendar = dataset.get("calendar")
    sessions = calendar.get("sessions") if isinstance(calendar, dict) else None
    if sessions is not None:
        if not isinstance(sessions, list) or not sessions:
            scaffold.append("calendar.sessions (nonempty list)")
        else:
            for index, row in enumerate(sessions):
                prefix = f"calendar.sessions[{index}]"
                if not isinstance(row, dict):
                    scaffold.append(prefix + " (object)")
                else:
                    scaffold.extend(f"{prefix}.{name}" for name in ("date", "open_at", "close_at")
                                    if name not in row or row[name] is None)
    for collection in ("prices", "observations"):
        rows = dataset.get(collection)
        if rows is not None and not isinstance(rows, list):
            scaffold.append(collection + " (list)")
            continue
        for index, row in enumerate(rows or []):
            prefix = f"{collection}[{index}]"
            if not isinstance(row, dict):
                evidence.append(prefix + " (object)")
                continue
            required = (("symbol", "session_date", "open_value", "return_basis", "available_at", "provenance_id")
                        if collection == "prices" else (
                            "sample_id", "symbol", "decision_session", "source_ids", "event_cluster_ids",
                            "published_at", "updated_at", "response_received_at", "extraction_completed_at",
                            "prediction_completed_at", "historical_availability_verified", "feature_values"))
            evidence.extend(f"{prefix}.{name}" for name in required if name not in row or row[name] is None)
            if collection == "observations":
                features = row.get("feature_values")
                if isinstance(features, dict):
                    for name in PREPARED_CONTRACT["baseline_features"]:
                        if features.get(name) is None:
                            evidence.append(f"{prefix}.feature_values.{name}")
                    for name in features:
                        entry = row.get("feature_provenance", {})
                        entry = entry.get(name) if isinstance(entry, dict) else None
                        for field in ("available_at", "provenance_id"):
                            if not isinstance(entry, dict) or not entry.get(field):
                                evidence.append(f"{prefix}.feature_provenance.{name}.{field}")
    if dataset.get("data_kind") == "observed":
        prices = dataset.get("prices", [])
        for index, row in enumerate(prices if isinstance(prices, list) else []):
            if isinstance(row, dict) and row.get("return_basis") == "total_return_index":
                if not row.get("return_series_id"):
                    evidence.append(f"prices[{index}].return_series_id")
                supplied = dataset.get("return_series_evidence", {})
                symbol = row.get("symbol")
                name = f"return_series_evidence.{symbol}"
                if (not isinstance(supplied, dict) or symbol not in supplied) and name not in evidence:
                    evidence.append(name)
    return scaffold, evidence


def _stages() -> dict:
    return {name: {"status": "Pending", "detail": detail} for name, detail in (
        ("input_quality_audit", "No input checked yet"),
        ("source_family_eligibility", "Publication/version/receipt/event evidence for financial reports, calls, news and comments requires source audit"),
        ("archive_to_company_day_adapter", "Not implemented; raw text/price archives are not a prepared dataset"),
        ("prepared_contract_evaluation", "Delegates the prepared contract to existing evaluate_dataset without independent source certification"),
        ("ten_session_labels", "Uses price_labels.build_labels via evaluate_dataset when a prepared contract exists"),
        ("time_and_event_split", "Uses the existing mature chronological split and transitive source/event purging"),
        ("paired_numeric_controls", "Five existing controls require identical complete-case samples; tone_score origin is caller supplied"),
        ("price_supervised_text_model", "Not implemented; numeric controls do not train a text encoder"),
        ("lm_and_full_text_baseline_comparison", "Not implemented"),
        ("large_move_news_attribution", "Not implemented"),
    )}


def run_pipeline(*, output_dir: Path, dry_run: bool = False, dataset_path: Path | None = None,
                 demo: bool = False, input_dir: Path | None = None,
                 legacy_alpaca_dir: Path | None = None, month: str | None = None,
                 as_of: str | None = None, symbols: list[str] | None = None,
                 source_snapshot_dir: Path | None = None) -> dict:
    """Publish a new private report; never replace an input or historical run.

    A successful dry-run has exit_code 0 while research remains blocked. A
    missing/invalid contract or insufficient paired data yields exit_code 2 and
    a report, without fabricated predictions. Existing core exceptions remain
    explicit in that report. Prepared provenance is a caller assertion.
    """
    if sum((dry_run, dataset_path is not None, demo)) != 1:
        raise ValueError("Choose exactly one of dry_run, dataset_path or demo")
    if dry_run and any(value is None for value in (input_dir, legacy_alpaca_dir, month, as_of)):
        raise ValueError("Dry-run requires input_dir, legacy_alpaca_dir, month and as_of")
    if not dry_run and any(value is not None for value in (input_dir, legacy_alpaca_dir, month, as_of, symbols, source_snapshot_dir)):
        raise ValueError("Prepared/demo mode uses its own dataset scope and clock; omit archive arguments")
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Choose a new output directory")
    output = output_dir.resolve()
    protected = [input_dir, legacy_alpaca_dir]
    if source_snapshot_dir is not None:
        protected.append(source_snapshot_dir)
    if dry_run and any(output.is_relative_to(Path(p).resolve()) for p in protected):
        raise ValueError("Output must be outside both input archives and the optional source snapshot")
    modules = {str(Path(__file__).with_name(name).resolve()):
               _digest(Path(__file__).with_name(name)) for name in MODULES}
    documents, inputs = {}, {}
    report = {
        "schema_version": "financial-sentiment-course-run-1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "Archive dry-run" if dry_run else "Synthetic contract check" if demo else "Caller-supplied prepared contract",
        "execution_status": "Blocked", "research_status": "Incomplete; pending data/model stages",
        "exit_code": 2, "horizon_sessions": 10, "stages": _stages(),
        "evaluation_status": "Not evaluated", "labels_created": 0, "eligible_labels": 0,
        "missing_inputs": [], "blocking_reasons": [], "pending_research_stages": [
            "archive_to_company_day_adapter", "price_supervised_text_model",
            "lm_and_full_text_baseline_comparison", "large_move_news_attribution"],
        "market_predictive_skill_established": False,
        "limitations": [
            "Software success does not establish predictive skill or trading alpha.",
            "Prepared times, calendars, corporate actions and feature origins require independent source audit.",
            "tone_score is not certified as FinBERT; no LM or supervised text-model comparison is claimed.",
            "Synthetic fixtures cannot establish real-market performance.",
            "No data acquisition, deployment, portfolio test or trade occurs here.",
        ],
    }
    stage = "input_quality_audit"
    try:
        if dry_run:
            preflight = audit_archives(Path(input_dir), Path(legacy_alpaca_dir),
                                        symbols=symbols or ["PLTR", "AMZN", "MSFT"], month=month, as_of=as_of,
                                        source_snapshot_dir=source_snapshot_dir)
            documents["preflight.json"] = preflight
            inputs.update(preflight["input_files_sha256"])
            report["archive_summary"] = preflight["summary"]
            report["missing_inputs"] = preflight["missing_evidence"] + [
                "prepared_dataset.json following required_contract.json",
                "evaluation.training_cutoff/test_start/test_end fixed before evaluation"]
            audits = {"month_news": preflight["source_family_audit"]}
            if "source_snapshot_audit" in preflight:
                audits["snapshot"] = preflight["source_snapshot_audit"]
            documents["source_family_audit.json"] = audits
            report["source_family_audits"] = {}
            for scope, audit in audits.items():
                report["source_family_audits"][scope] = {key: audit[key] for key in (
                    "cutoff", "eligible_source_records", "families", "missing_source_families", "reason_counts")}
                report["missing_inputs"].extend(
                    f"{scope}.{family}.retained_text_with_publication_version_and_actual_receipt"
                    for family in audit["missing_source_families"])
                if audit["reason_counts"].get("event_identity_unreviewed_or_missing"):
                    report["missing_inputs"].append(scope + ".reviewed_source_to_event_identity")
            report["stages"]["source_family_eligibility"] = {
                "status": "Completed with evidence gaps", "detail": "See source_family_audit.json; source eligibility is separate from label eligibility"}
            report["stages"][stage] = {"status": "Completed with evidence gaps", "detail": "See preflight.json"}
            report["blocking_reasons"] = ["No audited company-day dataset; labels and controls were not invoked"]
            report.update(execution_status="Dry-run completed", exit_code=0)
        else:
            if demo:
                dataset = synthetic_dataset()
                documents["synthetic_input.json"] = dataset
            else:
                path = Path(dataset_path).resolve()
                encoded = path.read_bytes()
                inputs[str(path)] = hashlib.sha256(encoded).hexdigest()
                dataset = json.loads(encoded)
            scaffold, evidence = missing_fields(dataset)
            report["missing_inputs"] = scaffold + evidence
            if scaffold:
                report["blocking_reasons"] = ["Missing required dataset scaffolding; see missing_inputs"]
                report["stages"][stage] = {"status": "Blocked", "detail": "Missing contract fields"}
            else:
                report["stages"][stage] = {"status": "Schema present; source audit required",
                                            "detail": "Caller-supplied contract; no raw archive adapter or independent certification"}
                stage = "prepared_contract_evaluation"
                labels, split, comparison = evaluate_dataset(dataset, horizon_sessions=10)
                report["stages"][stage] = {"status": "Completed", "detail": "Existing label/split/control contract executed; see separate stage statuses"}
                documents.update({"labels.json": labels, "split.json": split, "comparison.json": comparison})
                report.update(labels_created=len(labels),
                              eligible_labels=sum(row["status"] == "Eligible" for row in labels),
                              evaluation_status=comparison["evaluation_status"],
                              data_kind=dataset["data_kind"], label_coverage=comparison["label_coverage"],
                              paired_coverage=comparison["coverage"],
                              label_reason_counts=dict(Counter(reason for row in labels for reason in row["reasons"])),
                              split_exclusion_reason_counts=dict(Counter(reason for row in split["excluded"] for reason in row["reasons"])))
                report["stages"]["ten_session_labels"] = {"status": "Completed", "detail": "See statuses and reasons in labels.json"}
                report["stages"]["time_and_event_split"] = {"status": "Completed", "detail": "See split.json; only eligible mature records retained"}
                ran = bool(comparison["models"])
                report["stages"]["paired_numeric_controls"] = {
                    "status": "Completed" if ran else "Blocked", "detail": comparison.get("reason", comparison["mode"])}
                if ran:
                    report.update(execution_status="Controls completed; research stages pending", exit_code=0)
                else:
                    report["blocking_reasons"] = [comparison["reason"]]
                report["paired_missing_field_exclusions"] = comparison["excluded"]
                if dataset["data_kind"] == "synthetic":
                    report["mode"] = "Synthetic contract check"
    except (OSError, ValueError, TypeError, KeyError, ArithmeticError) as exc:
        report["blocking_reasons"].append(f"{stage}: {type(exc).__name__}: {exc}")
        report["error_stage"] = stage
        if stage in report["stages"]:
            report["stages"][stage] = {"status": "Blocked", "detail": "See blocking_reasons"}
    documents["required_contract.json"] = PREPARED_CONTRACT
    documents["report.json"] = report
    serialized = {name: _json(value) for name, value in documents.items()}
    # Input or generator changes during execution invalidate this run instead of
    # silently certifying a different set of bytes at publication time.
    if any(_digest(Path(path)) != expected for path, expected in {**inputs, **modules}.items()):
        raise ValueError("Input or generation source changed during execution")
    manifest = {
        "schema_version": "financial-sentiment-course-manifest-1.0", "generated_at": report["generated_at"],
        "input_files_sha256": inputs, "generation_sources_sha256": modules,
        "python_version": sys.version.split()[0], "outputs": {
            name: {"bytes": len(data.encode()), "sha256": hashlib.sha256(data.encode()).hexdigest()}
            for name, data in serialized.items()},
    }
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    (output / ".gitignore").write_text("*\n", encoding="utf-8")
    for name, data in {**serialized, "run_manifest.json": _json(manifest)}.items():
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(data)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dry-run", action="store_true", help="Audit existing archives; do not construct labels or fit controls")
    source.add_argument("--dataset", type=Path, help="Existing prepared JSON contract; not raw text or a preflight report")
    source.add_argument("--demo", action="store_true", help="Existing synthetic fixture; software evidence only")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--legacy-alpaca-dir", type=Path)
    parser.add_argument("--month")
    parser.add_argument("--as-of")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--source-snapshot-dir", type=Path, help="Optional existing multi-source snapshot for dry-run only")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run_pipeline(output_dir=args.output_dir, dry_run=args.dry_run, dataset_path=args.dataset,
                          demo=args.demo, input_dir=args.input_dir, legacy_alpaca_dir=args.legacy_alpaca_dir,
                          month=args.month, as_of=args.as_of, symbols=args.symbols,
                          source_snapshot_dir=args.source_snapshot_dir)
    print(json.dumps({name: report[name] for name in (
        "mode", "execution_status", "evaluation_status", "labels_created", "eligible_labels", "exit_code")}))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
