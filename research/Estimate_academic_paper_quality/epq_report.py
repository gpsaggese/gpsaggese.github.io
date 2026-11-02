"""
Write and read every file that the pipeline keeps in the output dir.

The files are:
- `<PAPERS>.csv`, `<PAPERS>.issues.csv`, and the input CSV: files that the
  executable `resolve_papers.py` reads and writes
- `state/issues.<stage>.json`: issues of the stages `download` and `extract`
- `papers/<BASE>.features.json`: features of one paper, also the rerun cache of
  the `extract` stage
- `references.md`, `report.csv`, and `issues.csv`: final outputs

The design is in `docs/02.7-design.report.md`.

Import as:

import research.Estimate_academic_paper_quality.epq_report as epqrepor
"""

import csv
import dataclasses
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import helpers.hdbg as hdbg
import helpers.hio as hio

import research.Estimate_academic_paper_quality.epq_types as epqtypes

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

# Version of the layout of the JSON files. A file with another version is not
# read: the user runs the stage again.
_SCHEMA_VERSION = 1

# Columns of `report.csv`, in order.
REPORT_COLUMNS = [
    "reference",
    "paper_quality",
    "performance",
    "recency",
    "asset_class",
    "theoretical_domain",
    "mechanism",
    "page_length_score",
    "affiliation_score",
    "citations_score",
    "base_name",
]

# Columns of `issues.csv`, in order.
ISSUES_COLUMNS = [
    "line_no",
    "input",
    "base_name",
    "stage",
    "status",
    "reason",
    "candidates",
]

# Name of the column of the input CSV of `resolve_papers.py`.
INPUT_COLUMN = "input"

# Columns of the papers CSV of `resolve_papers.py`, in order.
PAPERS_COLUMNS = [
    "reference",
    "year",
    "authors",
    "title",
    "doi",
    "arxiv_id",
    "s2_paper_id",
    "source",
    "base_name",
    "download_inputs",
    "line_nos",
    "key",
    "aliases",
]

# Separator of the items in a CSV cell that holds a list.
_LIST_SEP = "|"

# Order of the stages in `issues.csv`.
_STAGE_ORDER = ["resolve", "download", "extract"]


# #############################################################################
# JSON helpers
# #############################################################################


def _write_json(path: str, data: Dict[str, Any]) -> None:
    """
    Write a dict to a JSON file, with the schema version.

    :param path: path of the file, created with its dir
    :param data: content of the file
    """
    hio.create_dir(os.path.dirname(path) or ".", incremental=True)
    data = {"schema_version": _SCHEMA_VERSION, **data}
    hio.to_file(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _read_json(path: str) -> Dict[str, Any]:
    """
    Read a JSON file and check its schema version.

    :param path: path of the file
    :return: content of the file
    """
    hdbg.dassert_file_exists(path)
    data: Dict[str, Any] = json.loads(hio.from_file(path))
    hdbg.dassert_eq(
        data.get("schema_version"),
        _SCHEMA_VERSION,
        "File '%s' has an old schema: run the stage again with --force_extract "
        "or delete the file",
        path,
    )
    return data


def _evidence_from_dict(data: Dict[str, Any]) -> epqtypes.Evidence:
    """
    Build an `Evidence` from its JSON dict.

    :param data: dict with `quote`, `location`, and `verified`
    :return: evidence
    """
    evidence = epqtypes.Evidence(
        quote=data["quote"],
        location=data["location"],
        verified=bool(data["verified"]),
    )
    return evidence


def _optional_evidence(
    data: Optional[Dict[str, Any]],
) -> Optional[epqtypes.Evidence]:
    """
    Build an `Evidence` from its JSON dict, or return `None`.

    :param data: dict or `None`
    :return: evidence or `None`
    """
    evidence = None if data is None else _evidence_from_dict(data)
    return evidence


def _feature_from_dict(data: Dict[str, Any]) -> epqtypes.FeatureValue:
    """
    Build a `FeatureValue` from its JSON dict.

    :param data: dict with `value` and `evidence`
    :return: feature value
    """
    feature = epqtypes.FeatureValue(
        value=data["value"], evidence=_optional_evidence(data["evidence"])
    )
    return feature


def _facts_list_from_dict(
    data: List[Dict[str, Any]],
) -> Tuple[epqtypes.LabeledFact, ...]:
    """
    Build a tuple of `LabeledFact` from a list of JSON dicts.

    :param data: list of dicts with `label`, `value`, and `evidence`
    :return: labeled facts
    """
    facts = tuple(
        epqtypes.LabeledFact(
            label=item["label"],
            value=item["value"],
            evidence=_evidence_from_dict(item["evidence"]),
        )
        for item in data
    )
    return facts


def _facts_from_dict(data: Dict[str, Any]) -> epqtypes.ExtractedFacts:
    """
    Build an `ExtractedFacts` from its JSON dict.

    :param data: dict of the facts
    :return: extracted facts
    """
    facts = epqtypes.ExtractedFacts(
        sharpe_ratios=_facts_list_from_dict(data["sharpe_ratios"]),
        annual_returns=_facts_list_from_dict(data["annual_returns"]),
        volatilities=_facts_list_from_dict(data["volatilities"]),
        max_drawdowns=_facts_list_from_dict(data["max_drawdowns"]),
        sample_periods=_facts_list_from_dict(data["sample_periods"]),
        universe=_feature_from_dict(data["universe"]),
        data_assets=_feature_from_dict(data["data_assets"]),
        data_frequency=_feature_from_dict(data["data_frequency"]),
        data_fields=_feature_from_dict(data["data_fields"]),
    )
    return facts


def _signal_from_dict(data: Dict[str, Any]) -> epqtypes.SignalScore:
    """
    Build a `SignalScore` from its JSON dict.

    :param data: dict of the signal
    :return: signal score
    """
    signal = epqtypes.SignalScore(
        name=data["name"],
        value=data["value"],
        score=data["score"],
        evidence=_optional_evidence(data["evidence"]),
        note=data["note"],
    )
    return signal


def _quality_from_dict(data: Dict[str, Any]) -> epqtypes.QualityRating:
    """
    Build a `QualityRating` from its JSON dict.

    :param data: dict of the rating
    :return: quality rating
    """
    quality = epqtypes.QualityRating(
        rating=data["rating"],
        page_length=_signal_from_dict(data["page_length"]),
        affiliation=_signal_from_dict(data["affiliation"]),
        citations=_signal_from_dict(data["citations"]),
    )
    return quality


def _issue_from_dict(data: Dict[str, Any]) -> epqtypes.Issue:
    """
    Build an `Issue` from its JSON dict.

    :param data: dict of the issue
    :return: issue
    """
    issue = epqtypes.Issue(
        line_no=data["line_no"],
        input=data["input"],
        base_name=data["base_name"],
        stage=data["stage"],
        status=data["status"],
        reason=data["reason"],
        candidates=tuple(data["candidates"]),
    )
    return issue


# #############################################################################
# Issues of the stages
# #############################################################################


def write_issues(path: str, issues: List[epqtypes.Issue]) -> None:
    """
    Write the issues of one stage.

    :param path: path of `state/issues.<stage>.json`
    :param issues: issues of the stage `download` or `extract`
    """
    _write_json(path, {"issues": [dataclasses.asdict(i) for i in issues]})


def read_issues(path: str) -> List[epqtypes.Issue]:
    """
    Read the issues of one stage.

    :param path: path of `state/issues.<stage>.json`
    :return: issues of the stage
    """
    data = _read_json(path)
    issues = [_issue_from_dict(item) for item in data["issues"]]
    return issues


# #############################################################################
# Features file
# #############################################################################


def write_features_file(path: str, features: epqtypes.PaperFeatures) -> None:
    """
    Write the features file of one paper.

    :param path: path of `papers/<BASE>.features.json`
    :param features: features, facts, rating, and provenance of the paper
    """
    _write_json(path, dataclasses.asdict(features))


def read_features_file(path: str) -> epqtypes.PaperFeatures:
    """
    Read the features file of one paper.

    :param path: path of `papers/<BASE>.features.json`
    :return: features, facts, rating, and provenance of the paper
    """
    data = _read_json(path)
    provenance = epqtypes.Provenance(**data["provenance"])
    features = epqtypes.PaperFeatures(
        base_name=data["base_name"],
        features={
            name: _feature_from_dict(item)
            for name, item in data["features"].items()
        },
        facts=_facts_from_dict(data["facts"]),
        quality=_quality_from_dict(data["quality"]),
        provenance=provenance,
    )
    return features


# #############################################################################
# Final outputs
# #############################################################################


def _score_to_str(score: Optional[int]) -> str:
    """
    Convert a score or a rating to the text of a CSV cell.

    :param score: whole number, or `None` if unknown
    :return: the number, or `unknown`
    """
    return "unknown" if score is None else str(score)


def build_report_rows(
    papers: List[epqtypes.Paper],
    features: Dict[str, epqtypes.PaperFeatures],
) -> List[Dict[str, str]]:
    """
    Build the rows of `report.csv`, one per paper with a features file.

    :param papers: papers in reverse chronological order
    :param features: features of the papers, by base name
    :return: rows in the order of `papers`
    """
    rows = []
    for paper in papers:
        if paper.base_name not in features:
            # A paper with no features file is only in `issues.csv`.
            continue
        paper_features = features[paper.base_name]
        quality = paper_features.quality
        row = {
            "reference": paper.reference,
            "paper_quality": _score_to_str(quality.rating),
            "performance": paper_features.features["performance"].value,
            "recency": paper_features.features["recency"].value,
            "asset_class": paper_features.features["asset_class"].value,
            "theoretical_domain": paper_features.features[
                "theoretical_domain"
            ].value,
            "mechanism": paper_features.features["mechanism"].value,
            "page_length_score": _score_to_str(quality.page_length.score),
            "affiliation_score": _score_to_str(quality.affiliation.score),
            "citations_score": _score_to_str(quality.citations.score),
            "base_name": paper.base_name,
        }
        rows.append(row)
    return rows


def _write_csv(
    path: str, columns: List[str], rows: List[Dict[str, str]]
) -> None:
    """
    Write rows to a CSV file in UTF-8.

    :param path: path of the file, created with its dir
    :param columns: names of the columns, in order
    :param rows: rows with one value per column
    """
    hio.create_dir(os.path.dirname(path) or ".", incremental=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: str, rows: List[Dict[str, str]]) -> None:
    """
    Write `report.csv`.

    :param path: path of the file
    :param rows: rows from `build_report_rows()`
    """
    _write_csv(path, REPORT_COLUMNS, rows)


def write_issues_csv(path: str, issues: List[epqtypes.Issue]) -> None:
    """
    Write `issues.csv`, by stage and then by line number.

    :param path: path of the file
    :param issues: issues of every stage
    """
    ordered = sorted(
        issues,
        key=lambda i: (
            _STAGE_ORDER.index(i.stage)
            if i.stage in _STAGE_ORDER
            else len(_STAGE_ORDER),
            i.line_no,
            i.base_name,
        ),
    )
    rows = [
        {
            "line_no": str(issue.line_no),
            "input": issue.input,
            "base_name": issue.base_name,
            "stage": issue.stage,
            "status": issue.status,
            "reason": issue.reason,
            "candidates": _join_items(issue.candidates),
        }
        for issue in ordered
    ]
    _write_csv(path, ISSUES_COLUMNS, rows)


def write_references(path: str, papers: List[epqtypes.Paper]) -> None:
    """
    Write `references.md`, one bullet per resolved paper.

    :param path: path of the file
    :param papers: papers in reverse chronological order
    """
    hio.create_dir(os.path.dirname(path) or ".", incremental=True)
    lines = [f"- {paper.reference}" for paper in papers]
    hio.to_file(path, "\n".join(lines) + "\n")


# #############################################################################
# CSV files of `resolve_papers.py`
# #############################################################################


def _read_csv(path: str, columns: List[str]) -> List[Dict[str, str]]:
    """
    Read the rows of a CSV file in UTF-8, and check that it has the columns.

    A BOM written by a spreadsheet is skipped. The cell of a short row is `""`.

    :param path: path of the file
    :param columns: names of the columns that the file must have
    :return: one dict per row, with the extra columns too
    """
    hdbg.dassert_file_exists(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        hdbg.dassert_eq(
            len(missing),
            0,
            "File '%s' has no column %s: it has %s",
            path,
            missing,
            reader.fieldnames,
        )
        rows = [
            {key: value or "" for key, value in row.items() if key is not None}
            for row in reader
        ]
    return rows


def _join_items(items: Iterable[Any]) -> str:
    """
    Join the items of a list into the text of one CSV cell.

    :param items: items, which must not hold the separator `|`
    :return: text like `a|b`
    """
    return _LIST_SEP.join(str(item) for item in items)


def _split_items(text: str) -> List[str]:
    """
    Split the text of a CSV cell into the items of a list.

    :param text: text like `a|b`, or ""
    :return: items, an empty list for ""
    """
    return text.split(_LIST_SEP) if text else []


def read_input_csv(path: str) -> List[Tuple[int, str]]:
    """
    Read the input CSV of `resolve_papers.py`.

    The column `input` holds a title (with the authors in parentheses), a link,
    a DOI, or the path of a local PDF. The other columns are ignored.

    :param path: path of the input CSV
    :return: pairs `(row number, text)`, with the first row after the header
        numbered 1
    """
    rows = _read_csv(path, [INPUT_COLUMN])
    entries = [(i, row[INPUT_COLUMN]) for i, row in enumerate(rows, start=1)]
    return entries


def write_papers_csv(path: str, papers: List[epqtypes.Paper]) -> None:
    """
    Write the papers CSV, the output of `resolve_papers.py`.

    A list (e.g., the authors) is one cell with the items joined by `|`.

    :param path: path of the file, created with its dir
    :param papers: papers, in reverse chronological order
    """
    rows = [
        {
            "reference": paper.reference,
            "year": str(paper.year),
            "authors": _join_items(paper.authors),
            "title": paper.title,
            "doi": paper.doi,
            "arxiv_id": paper.arxiv_id,
            "s2_paper_id": paper.s2_paper_id,
            "source": paper.source,
            "base_name": paper.base_name,
            "download_inputs": _join_items(paper.download_inputs),
            "line_nos": _join_items(paper.line_nos),
            "key": paper.key,
            "aliases": _join_items(paper.aliases),
        }
        for paper in papers
    ]
    _write_csv(path, PAPERS_COLUMNS, rows)


def read_papers_csv(path: str) -> List[epqtypes.Paper]:
    """
    Read the papers CSV, the input of the stages `download` and `extract`.

    :param path: path of the papers CSV
    :return: papers, in the order of the rows
    """
    rows = _read_csv(path, PAPERS_COLUMNS)
    papers = [
        epqtypes.Paper(
            key=row["key"],
            aliases=tuple(_split_items(row["aliases"])),
            line_nos=tuple(int(n) for n in _split_items(row["line_nos"])),
            source=row["source"],
            year=int(row["year"]),
            authors=tuple(_split_items(row["authors"])),
            title=row["title"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            s2_paper_id=row["s2_paper_id"],
            reference=row["reference"],
            base_name=row["base_name"],
            download_inputs=tuple(_split_items(row["download_inputs"])),
        )
        for row in rows
    ]
    return papers


def read_issues_csv(path: str) -> List[epqtypes.Issue]:
    """
    Read an issues CSV, e.g., the one that `resolve_papers.py` wrote.

    :param path: path of the issues CSV
    :return: issues, in the order of the rows
    """
    rows = _read_csv(path, ISSUES_COLUMNS)
    issues = [
        epqtypes.Issue(
            line_no=int(row["line_no"]),
            input=row["input"],
            base_name=row["base_name"],
            stage=row["stage"],
            status=row["status"],
            reason=row["reason"],
            candidates=tuple(_split_items(row["candidates"])),
        )
        for row in rows
    ]
    return issues
