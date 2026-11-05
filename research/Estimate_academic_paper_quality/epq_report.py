"""
Write and read every file that the pipeline keeps in the output dir.

The files are:
- The input CSV: the list that the user writes, which `resolve_papers.py` reads: it
  is the only CSV, and the pipeline never writes a CSV
- `01-resolve.paper_list.json`: the papers that `resolve_papers.py` wrote
- `NN-<stage>.issues.json`: issues of the stages `resolve`, `download`, `extract`,
  and `rate`
- `papers/<BASE>.features.json`: features of one paper, also the rerun cache of
  the `extract` stage
- `04-report.report.json`: one row per paper or unresolved input line, with the
  issue of the row if there is one (decision D35)
- `04-report.references.md`: the references of all the resolved papers

The names of the files are in `epq_paths.py`. Every JSON file holds a
`schema_version` (decisions D31, D34, and D35).

The design is in `docs/02.7-design.report.md`.

Import as:

import research.Estimate_academic_paper_quality.epq_report as epqrepor
"""

import csv
import dataclasses
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import helpers.hdbg as hdbg
import helpers.hio as hio

import research.Estimate_academic_paper_quality.epq_types as epqtypes

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

# Version of the layout of the JSON files. A file with another version is not
# read: the user runs the stage again.
# - Version 2: the features file holds the affiliations and the page count, and
#   the rating is in its own file.
_SCHEMA_VERSION = 2

# Name of the optional header column of the input CSV of `resolve_papers.py`.
INPUT_COLUMN = "input"

# Version of the layout of `01-resolve.paper_list.json`. Decision D37 changed it
# with no change to the other files, so that the features files are not
# extracted again.
# - Version 3: the key `links` holds the links of the reference, and `reference`
#   no longer has them.
_PAPERS_SCHEMA_VERSION = 3

# Version of the layout of `04-report.report.json`, which no stage reads. Decisions
# D35 and D37 changed it with no change to the other files, so that the features
# files are not extracted again.
# - Version 3: one row per paper or unresolved line, under the key `rows`, with
#   the issue of the row.
# - Version 4: the key `links` follows the key `reference`.
_REPORT_SCHEMA_VERSION = 4

# Order of the stages, to choose the issue of a row that has many.
_STAGE_ORDER = ["resolve", "download", "extract", "rate"]

# Keys of a row with the values of the features and of the rating, in this order.
_VALUE_KEYS = (
    "paper_quality",
    "performance",
    "recency",
    "asset_class",
    "theoretical_domain",
    "mechanism",
    "page_length_score",
    "affiliation_score",
    "citations_score",
)


# #############################################################################
# JSON helpers
# #############################################################################


def _write_json(
    path: str, data: Dict[str, Any], *, schema_version: int = _SCHEMA_VERSION
) -> None:
    """
    Write a dict to a JSON file, with the schema version.

    :param path: path of the file, created with its dir
    :param data: content of the file
    :param schema_version: version of the layout of the file
    """
    hio.create_dir(os.path.dirname(path) or ".", incremental=True)
    data = {"schema_version": schema_version, **data}
    hio.to_file(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _read_json(
    path: str, *, schema_version: int = _SCHEMA_VERSION
) -> Dict[str, Any]:
    """
    Read a JSON file and check its schema version.

    :param path: path of the file
    :param schema_version: version of the layout that the file must have
    :return: content of the file
    """
    hdbg.dassert_file_exists(path)
    data: Dict[str, Any] = json.loads(hio.from_file(path))
    hdbg.dassert_eq(
        data.get("schema_version"),
        schema_version,
        "File '%s' has an old schema: run the stage again (`resolve_papers.py` "
        "for the papers or the resolve issues, `extract_features.py "
        "--force_extract` for a features file, `report_papers.py --action rate` "
        "for a rating file) or delete the file",
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


def _affiliation_from_dict(data: Dict[str, Any]) -> epqtypes.Affiliation:
    """
    Build an `Affiliation` from its JSON dict.

    :param data: dict of the affiliation
    :return: affiliation
    """
    affiliation = epqtypes.Affiliation(
        author=data["author"],
        institution=data["institution"],
        kind=data["kind"],
        evidence=_evidence_from_dict(data["evidence"]),
    )
    return affiliation


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

    :param path: path of `NN-<stage>.issues.json`
    :param issues: issues of the stage `resolve`, `download`, `extract`, or
        `rate`
    """
    _write_json(path, {"issues": [dataclasses.asdict(i) for i in issues]})


def read_issues(path: str) -> List[epqtypes.Issue]:
    """
    Read the issues of one stage.

    :param path: path of `NN-<stage>.issues.json`
    :return: issues of the stage
    """
    data = _read_json(path)
    issues = [_issue_from_dict(item) for item in data["issues"]]
    return issues


# #############################################################################
# Papers file
# #############################################################################


def _paper_from_dict(data: Dict[str, Any]) -> epqtypes.Paper:
    """
    Build a `Paper` from its JSON dict.

    :param data: dict of the paper
    :return: paper, with a tuple for each list
    """
    paper = epqtypes.Paper(
        key=data["key"],
        aliases=tuple(data["aliases"]),
        line_nos=tuple(data["line_nos"]),
        source=data["source"],
        year=data["year"],
        authors=tuple(data["authors"]),
        title=data["title"],
        doi=data["doi"],
        arxiv_id=data["arxiv_id"],
        s2_paper_id=data["s2_paper_id"],
        reference=data["reference"],
        links=data["links"],
        base_name=data["base_name"],
        download_inputs=tuple(data["download_inputs"]),
    )
    return paper


def write_papers(path: str, papers: List[epqtypes.Paper]) -> None:
    """
    Write the papers file, the output of `resolve_papers.py`.

    :param path: path of `01-resolve.paper_list.json`, created with its dir
    :param papers: papers, in reverse chronological order
    """
    _write_json(
        path,
        {"papers": [dataclasses.asdict(p) for p in papers]},
        schema_version=_PAPERS_SCHEMA_VERSION,
    )


def read_papers(path: str) -> List[epqtypes.Paper]:
    """
    Read the papers file, the input of the stages after `resolve`.

    :param path: path of `01-resolve.paper_list.json`
    :return: papers, in the order of the file
    """
    data = _read_json(path, schema_version=_PAPERS_SCHEMA_VERSION)
    papers = [_paper_from_dict(item) for item in data["papers"]]
    return papers


# #############################################################################
# Features file
# #############################################################################


def write_features_file(path: str, features: epqtypes.PaperFeatures) -> None:
    """
    Write the features file of one paper.

    :param path: path of `papers/<BASE>.features.json`
    :param features: features, facts, affiliations, page count, and provenance
        of the paper
    """
    _write_json(path, dataclasses.asdict(features))


def read_features_file(path: str) -> epqtypes.PaperFeatures:
    """
    Read the features file of one paper.

    :param path: path of `papers/<BASE>.features.json`
    :return: features, facts, affiliations, page count, and provenance of the
        paper
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
        affiliations=tuple(
            _affiliation_from_dict(item) for item in data["affiliations"]
        ),
        n_pages=data["n_pages"],
        provenance=provenance,
    )
    return features


# #############################################################################
# Rating file
# #############################################################################


def write_rating_file(path: str, quality: epqtypes.QualityRating) -> None:
    """
    Write the rating file of one paper.

    :param path: path of `papers/<BASE>.rating.json`
    :param quality: rating and the three signals behind it
    """
    _write_json(path, dataclasses.asdict(quality))


def read_rating_file(path: str) -> epqtypes.QualityRating:
    """
    Read the rating file of one paper.

    :param path: path of `papers/<BASE>.rating.json`
    :return: rating and the three signals behind it
    """
    data = _read_json(path)
    quality = _quality_from_dict(data)
    return quality


# #############################################################################
# Final outputs
# #############################################################################


def _get_stage_rank(stage: str) -> int:
    """
    Get the position of a stage in the pipeline.

    :param stage: one of `resolve`, `download`, `extract`, `rate`
    :return: position, after the last stage if the stage is unknown
    """
    rank = (
        _STAGE_ORDER.index(stage) if stage in _STAGE_ORDER else len(_STAGE_ORDER)
    )
    return rank


def _pick_issue(
    base_name: str, paper_issues: List[epqtypes.Issue]
) -> Optional[epqtypes.Issue]:
    """
    Pick the issue to show in the row of a paper.

    The row holds the issue of the earliest stage: the later stages often fail
    because of it. The files `NN-<stage>.issues.json` keep all the issues.

    :param base_name: base name of the paper, for the warning
    :param paper_issues: issues of the paper, from all the stages
    :return: issue of the earliest stage, `None` if the paper has no issue
    """
    issue = None
    if paper_issues:
        ordered = sorted(paper_issues, key=lambda i: _get_stage_rank(i.stage))
        issue = ordered[0]
        if len(ordered) > 1:
            _LOG.warning(
                "Paper '%s' has %s issues: the report shows only the one of the "
                "stage `%s`",
                base_name,
                len(ordered),
                issue.stage,
            )
    return issue


def _get_values(
    paper_features: Optional[epqtypes.PaperFeatures],
    quality: Optional[epqtypes.QualityRating],
) -> Optional[Dict[str, Any]]:
    """
    Get the values of the features and of the rating of a paper.

    A score can be `None`, e.g., a paper with no citation count.

    :param paper_features: features of the paper, `None` if it has none
    :param quality: rating of the paper, `None` if it has none
    :return: values by key of `_VALUE_KEYS`, `None` if the paper lacks the
        features or the rating
    """
    values = None
    if paper_features is not None and quality is not None:
        values = {
            "paper_quality": quality.rating,
            "performance": paper_features.features["performance"].value,
            "recency": paper_features.features["recency"].value,
            "asset_class": paper_features.features["asset_class"].value,
            "theoretical_domain": paper_features.features[
                "theoretical_domain"
            ].value,
            "mechanism": paper_features.features["mechanism"].value,
            "page_length_score": quality.page_length.score,
            "affiliation_score": quality.affiliation.score,
            "citations_score": quality.citations.score,
        }
    return values


def _make_row(
    *,
    line_nos: List[int],
    reference: str,
    links: Optional[str],
    base_name: Optional[str],
    values: Optional[Dict[str, Any]],
    issue: Optional[epqtypes.Issue],
) -> Dict[str, Any]:
    """
    Build one row of `report.json`.

    The row is `ok` if the paper has the values and no issue.

    :param line_nos: lines of the input list that gave the row
    :param reference: reference of the paper, without the links, or text of the
        unresolved line
    :param links: links of the reference, `None` if the line has no paper
    :param base_name: base name of the paper, `None` if the line has no paper
    :param values: values from `_get_values()`, `None` if the row has none
    :param issue: issue of the row, `None` if there is none
    :return: row, with the keys in the order of `build_report_rows()`
    """
    status = "ok" if values is not None and issue is None else "failed"
    if values is None:
        values = dict.fromkeys(_VALUE_KEYS)
    issue_dict = None
    if issue is not None:
        issue_dict = {
            "stage": issue.stage,
            "status": issue.status,
            "reason": issue.reason,
            "candidates": list(issue.candidates),
        }
    row = {
        "line_nos": line_nos,
        "status": status,
        "reference": reference,
        "links": links,
        **values,
        "base_name": base_name,
        "issue": issue_dict,
    }
    return row


def build_report_rows(
    papers: List[epqtypes.Paper],
    features: Dict[str, epqtypes.PaperFeatures],
    ratings: Dict[str, epqtypes.QualityRating],
    issues: List[epqtypes.Issue],
) -> List[Dict[str, Any]]:
    r"""
    Build the rows of `report.json`: one for each paper and one for each issue
    that no paper owns, e.g., an unresolved line.

    A row is `ok` if the paper has a features file, a rating file, and no issue.
    Any other row is `failed`: it has the issue, and the values if the paper has
    the two files. A rating or a score that is unknown is `None`, which
    `write_report()` writes as `null`. The other values are the text of the
    features.

    :param papers: papers in reverse chronological order
    :param features: features of the papers, by base name
    :param ratings: ratings of the papers, by base name
    :param issues: issues of all the stages
    :return: rows of the papers in the order of `papers`, then the rows of the
        other issues by line number, with the keys in this order
        ```
        {
          "line_nos": [16],
          "status": "ok",
          "reference": "2020, Cohen et al., \"Lazy Prices\"",
          "links": "(https://...)",
          "paper_quality": 3,
          "performance": "Sharpe 1.4",
          "recency": "2020",
          "asset_class": "US equities",
          "theoretical_domain": "unknown",
          "mechanism": "associational",
          "page_length_score": 1,
          "affiliation_score": 3,
          "citations_score": null,
          "base_name": "2020.Cohen_et_al.Lazy_Prices",
          "issue": null
        }
        ```
        and the row of an unresolved line
        ```
        {
          "line_nos": [10],
          "status": "failed",
          "reference": "Monetary Policy Transmission",
          "links": null,
          "paper_quality": null,
          "...": "...",
          "citations_score": null,
          "base_name": null,
          "issue": {
            "stage": "resolve",
            "status": "unresolved",
            "reason": "no candidate has a title like ...",
            "candidates": []
          }
        }
        ```
    """
    # Give each issue to the paper that it names, if the papers file has it.
    paper_names = {paper.base_name for paper in papers}
    issues_by_paper: Dict[str, List[epqtypes.Issue]] = {}
    other_issues: List[epqtypes.Issue] = []
    for issue in issues:
        if issue.base_name in paper_names:
            issues_by_paper.setdefault(issue.base_name, []).append(issue)
        else:
            other_issues.append(issue)
    rows = []
    for paper in papers:
        row = _make_row(
            line_nos=list(paper.line_nos),
            reference=paper.reference,
            links=paper.links,
            base_name=paper.base_name,
            values=_get_values(
                features.get(paper.base_name), ratings.get(paper.base_name)
            ),
            issue=_pick_issue(
                paper.base_name, issues_by_paper.get(paper.base_name, [])
            ),
        )
        rows.append(row)
    # Add a row for each issue that has no paper, e.g., an unresolved line.
    other_issues = sorted(
        other_issues, key=lambda i: (i.line_no, _get_stage_rank(i.stage))
    )
    for issue in other_issues:
        row = _make_row(
            line_nos=[issue.line_no],
            reference=issue.input,
            links=None,
            base_name=issue.base_name or None,
            values=_get_values(None, None),
            issue=issue,
        )
        rows.append(row)
    return rows


def write_report(path: str, rows: List[Dict[str, Any]]) -> None:
    """
    Write `report.json`.

    :param path: path of `04-report.report.json`
    :param rows: rows from `build_report_rows()`
    """
    _write_json(path, {"rows": rows}, schema_version=_REPORT_SCHEMA_VERSION)


def join_reference(reference: str, links: str) -> str:
    """
    Join a reference and its links in the format of
    `.claude/skills/references.rules.md`.

    The JSON files keep the two apart (decision D37), and the text for a person
    has them together: `references.md` and the candidates of an issue.

    - `('2020, Cohen et al., "Lazy Prices"', "(https://x.org/p.pdf)")` ->
      `2020, Cohen et al., "Lazy Prices" (https://x.org/p.pdf)`

    :param reference: reference without the links
    :param links: links in parentheses, "" if there are none
    :return: reference, a space, and the links
    """
    return " ".join(part for part in (reference, links) if part)


def write_references(path: str, papers: List[epqtypes.Paper]) -> None:
    """
    Write `references.md`, one bullet per resolved paper.

    Each bullet has the reference followed by its links.

    :param path: path of the file
    :param papers: papers in reverse chronological order
    """
    hio.create_dir(os.path.dirname(path) or ".", incremental=True)
    lines = [
        f"- {join_reference(paper.reference, paper.links)}" for paper in papers
    ]
    hio.to_file(path, "\n".join(lines) + "\n")


# #############################################################################
# Input CSV written by the user
# #############################################################################


def read_input_csv(path: str) -> List[Tuple[int, str]]:
    """
    Read the input CSV of `resolve_papers.py`.

    Each row is a title (with the authors in parentheses), a link, a DOI, or the
    path of a local PDF. The header is optional:
    - With no header, the text of the row is the entry. The cells of the row are
      joined back with `,`, so a title with a comma stays whole.
    - With a header that has a column `input`, the entry is that column. The
      other columns are ignored.

    A BOM written by a spreadsheet is skipped. An empty row is skipped.

    :param path: path of the input CSV
    :return: pairs `(row number, text)`, with the first row after the header (or
        the first row, with no header) numbered 1
    """
    hdbg.dassert_file_exists(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [row for row in csv.reader(f) if row]
    # Find the column `input` in the header, if there is one.
    header = [cell.strip().lower() for cell in rows[0]] if rows else []
    if INPUT_COLUMN in header:
        col = header.index(INPUT_COLUMN)
        texts = [row[col] if col < len(row) else "" for row in rows[1:]]
    else:
        texts = [",".join(row) for row in rows]
    entries = list(enumerate(texts, start=1))
    return entries
