#!/usr/bin/env -S uv run

# /// script
# dependencies = ["feedparser", "requests", "pyyaml", "beautifulsoup4", "tqdm"]
# ///

"""
Rate the extracted papers and write the final report.

The input is the papers file that `resolve_papers.py` writes, by default
`01-resolve.paper_list.json` in `--output_dir`, and the features files that
`extract_features.py` writes there.

The script has two stages, which talk through files:
- `rate`: rate the quality of each paper from its page count, its affiliations,
  and its citations, and write `papers/<BASE>.rating.json`,
  `04-rate.issues.json`, and `04-rate.summary.txt`
- `report`: write `04-report.references.md`, `04-report.report.json`, and
  `04-report.summary.txt`

Each stage ends with a summary of its counts, its output files, and its issues
file. The summary goes to the log and to the `summary.txt` file of the stage.

The stage `rate` is free: it makes no LLM call, only a citation lookup. A change
of the tier list `affiliation_tiers.yaml` or of the bands needs only a new run of
this stage. The citation counts grow with time: use `--refresh_citations` to ask
for fresh counts.

The report has one row for each paper and for each unresolved input line, with the
issue of the row if there is one. The stage `report` reads the issues of the other
stages from `01-resolve.issues.json`, `02-download.issues.json`,
`03-extract.issues.json`, and `04-rate.issues.json` in `--output_dir`.

The design is in `research/Estimate_academic_paper_quality/docs/02-design.architecture.md`.

# Usage Example

- Rate the papers and write the report:
> report_papers.py --output_dir out

- Rate the papers again with fresh citation counts:
> report_papers.py --output_dir out --action rate --refresh_citations

- Write the report from the existing rating files:
> report_papers.py --output_dir out --clear_actions --action report

For a description of this file, see the file
`research/Estimate_academic_paper_quality/report_papers.py.README.md`
"""

import argparse
import dataclasses
import datetime
import logging
import math
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

import helpers.hcache_simple as hcacsimp
import helpers.hdbg as hdbg
import helpers.hgit as hgit
import helpers.hio as hio
import helpers.hparser as hparser
import helpers.hselect_action as hselacti
import helpers.htimer as htimer
import tqdm
import yaml

import research.Estimate_academic_paper_quality.epq_bib_sources as epqbibso
import research.Estimate_academic_paper_quality.epq_issues as epqissue
import research.Estimate_academic_paper_quality.epq_paths as epqpaths
import research.Estimate_academic_paper_quality.epq_report as epqrepor
import research.Estimate_academic_paper_quality.epq_summary as epqsumm
import research.Estimate_academic_paper_quality.epq_types as epqtypes

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

# Lower bounds of the bands of page length, from the best score to the worst.
# E.g., 35 pages or more score 5.
_PAGE_BANDS = [(35, 5), (20, 4), (10, 3), (5, 2), (0, 1)]

# Lower bounds of the bands of citation count, from the best score to the worst.
_CITATION_BANDS = [(1000, 5), (300, 4), (50, 3), (10, 2), (0, 1)]

# Score of an affiliation that is not in the tier list, by the kind that the
# LLM read in the paper.
_KIND_SCORES = {
    "central_bank": 5,
    "university": 3,
    "research_institute": 3,
    "company": 2,
    "none": 1,
}

# A paper younger than this many years has its citations annualized.
_YOUNG_PAPER_YEARS = 2

# The age of a paper is at least this many years.
_MIN_AGE_YEARS = 0.5

# The stages, in the order they run.
_VALID_ACTIONS = ["rate", "report"]
_DEFAULT_ACTIONS = _VALID_ACTIONS


# #############################################################################
# Inputs
# #############################################################################


@dataclasses.dataclass(frozen=True)
class _RatingInputs:
    """
    Inputs to rate one paper.
    """

    # `None` when the PDF cannot be read.
    n_pages: Optional[int]
    affiliations: Tuple[epqtypes.Affiliation, ...]
    # `None` when the lookup source does not have the paper.
    citations: Optional[epqtypes.CitationCount]
    year: int
    today: datetime.date


# #############################################################################
# Tier list
# #############################################################################


def _parse_entries(data: Any) -> Tuple[epqtypes.TierEntry, ...]:
    """
    Convert a list of YAML entries into tier entries.

    :param data: list of dicts with `name` and optional `aliases`, or `None`
        when the YAML key is empty
    :return: tier entries
    """
    entries = []
    for item in data or []:
        entry = epqtypes.TierEntry(
            name=str(item["name"]),
            aliases=tuple(str(alias) for alias in item.get("aliases", [])),
        )
        entries.append(entry)
    return tuple(entries)


def _parse_tiers(data: Dict[str, Any]) -> epqtypes.AffiliationTiers:
    """
    Convert the YAML dict of `affiliation_tiers.yaml` into tiers.

    :param data: dict with the keys `as_of`, `source`, `tier_5`, and `tier_4`
    :return: tier list
    """
    tiers = epqtypes.AffiliationTiers(
        as_of=str(data.get("as_of", "")),
        source=str(data.get("source", "")),
        tier_5=_parse_entries(data.get("tier_5")),
        tier_4=_parse_entries(data.get("tier_4")),
    )
    return tiers


# #############################################################################
# Scores
# #############################################################################


def _score_from_bands(value: float, bands: List[Tuple[int, int]]) -> int:
    """
    Find the score of a value in a list of bands.

    :param value: value to score
    :param bands: pairs `(lower bound, score)`, from the best to the worst
    :return: score of the first band whose lower bound is at most the value
    """
    score = bands[-1][1]
    for lower_bound, band_score in bands:
        if value >= lower_bound:
            score = band_score
            break
    return score


def _score_page_length(n_pages: Optional[int]) -> epqtypes.SignalScore:
    """
    Score the page count of a paper.

    :param n_pages: number of pages of the PDF, all pages included
    :return: score from 1 to 5, or `unknown` if the PDF cannot be read
    """
    if n_pages is None:
        signal = epqtypes.SignalScore(
            name="page_length",
            value="unknown",
            score=None,
            evidence=None,
            note="the PDF cannot be read",
        )
    else:
        signal = epqtypes.SignalScore(
            name="page_length",
            value=f"{n_pages} pages",
            score=_score_from_bands(n_pages, _PAGE_BANDS),
            evidence=None,
            note="",
        )
    return signal


def _normalize_institution(text: str) -> str:
    """
    Normalize an institution name to compare it with the tier list.

    The name is ASCII, lowercase, with no punctuation, single spaces, and no
    leading `the`.
    - `The Massachusetts Institute of Technology (MIT)` ->
      `massachusetts institute of technology mit`

    :param text: institution name as written in the paper
    :return: normalized name
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    text = re.sub(r"^the ", "", text)
    return text


def _match_tier(
    institution: str, tiers: epqtypes.AffiliationTiers
) -> Optional[Tuple[int, str]]:
    """
    Find the entry of the tier list that an institution matches.

    An entry matches if its name or one of its aliases is a whole-word part of
    the institution. The longest matching entry wins, so that
    `University of California, Berkeley` does not match a shorter entry.

    :param institution: institution name as written in the paper
    :param tiers: tier list
    :return: pair `(score, entry name)`, `None` if no entry matches
    """
    padded = f" {_normalize_institution(institution)} "
    best: Optional[Tuple[int, int, str]] = None
    for score, entries in [(5, tiers.tier_5), (4, tiers.tier_4)]:
        for entry in entries:
            for name in (entry.name,) + entry.aliases:
                normalized = _normalize_institution(name)
                if not normalized or f" {normalized} " not in padded:
                    continue
                # Prefer the longest entry, then the higher score.
                candidate = (len(normalized), score, entry.name)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
    result = None if best is None else (best[1], best[2])
    return result


def _score_affiliation(
    affiliations: Tuple[epqtypes.Affiliation, ...],
    tiers: epqtypes.AffiliationTiers,
) -> epqtypes.SignalScore:
    """
    Score the best affiliation of the authors.

    Each author scores 5 or 4 if the institution is in the tier list, else by
    the kind that the LLM read in the paper. The signal is the best score.

    :param affiliations: affiliations read from the paper
    :param tiers: dated tier list
    :return: score from 1 to 5, with the quote of the best affiliation as
        evidence
    """
    if not affiliations:
        # The rules score a paper with no affiliation as 1.
        signal = epqtypes.SignalScore(
            name="affiliation",
            value="no affiliation listed",
            score=1,
            evidence=None,
            note="the paper lists no affiliation",
        )
        return signal
    best_score = 0
    best_value = ""
    best_affiliation = affiliations[0]
    for affiliation in affiliations:
        tier_match = _match_tier(affiliation.institution, tiers)
        if tier_match is not None:
            score, entry_name = tier_match
            value = f"{entry_name} (tier list)"
        else:
            score = _KIND_SCORES[affiliation.kind]
            value = f"{affiliation.institution} ({affiliation.kind})"
        # Keep the first author with the best score.
        if score > best_score:
            best_score = score
            best_value = value
            best_affiliation = affiliation
    signal = epqtypes.SignalScore(
        name="affiliation",
        value=best_value,
        score=best_score,
        evidence=best_affiliation.evidence,
        note="",
    )
    return signal


def _score_citations(
    citations: Optional[epqtypes.CitationCount],
    year: int,
    today: datetime.date,
) -> epqtypes.SignalScore:
    """
    Score the citation count of a paper.

    A paper under 2 years old has its count annualized as `count * 2 / age`
    before scoring, with the age at least 0.5 year.

    :param citations: count with its source and date, `None` if the lookup
        source does not have the paper
    :param year: publication year
    :param today: date used to compute the age
    :return: score from 1 to 5, or `unknown` if there is no count
    """
    if citations is None:
        signal = epqtypes.SignalScore(
            name="citations",
            value="unknown",
            score=None,
            evidence=None,
            note="Semantic Scholar does not have the paper",
        )
        return signal
    # Annualize the count of a young paper.
    age = max(today.year - year, _MIN_AGE_YEARS)
    count_to_score: float = citations.count
    note = ""
    if age < _YOUNG_PAPER_YEARS:
        count_to_score = citations.count * 2 / age
        note = f"annualized: {citations.count} * 2 / {age:g} years"
    value = f"{citations.count} ({citations.source}, {citations.fetched_at})"
    evidence = epqtypes.Evidence(
        quote=str(citations.count), location=citations.source, verified=True
    )
    signal = epqtypes.SignalScore(
        name="citations",
        value=value,
        score=_score_from_bands(count_to_score, _CITATION_BANDS),
        evidence=evidence,
        note=note,
    )
    return signal


# #############################################################################
# Rating
# #############################################################################


def _compute_rating(
    inputs: _RatingInputs, tiers: epqtypes.AffiliationTiers
) -> epqtypes.QualityRating:
    """
    Rate a paper from 1 to 5 with the mean of the known signal scores.

    The mean is rounded half up with `math.floor(mean + 0.5)`, since Python
    `round()` rounds half to even.
    - 12 pages (3), a top 20 university (5), 40 citations after 3 years (2) ->
      mean 3.33, rating 3

    :param inputs: page count, affiliations, citations, year, and today
    :param tiers: dated tier list
    :return: rating, `None` if no signal is known, with the three signals
    """
    page_length = _score_page_length(inputs.n_pages)
    affiliation = _score_affiliation(inputs.affiliations, tiers)
    citations = _score_citations(inputs.citations, inputs.year, inputs.today)
    known = [
        signal.score
        for signal in [page_length, affiliation, citations]
        if signal.score is not None
    ]
    rating: Optional[int] = None
    if known:
        rating = math.floor(sum(known) / len(known) + 0.5)
    _LOG.debug("known=%s rating=%s", known, rating)
    quality = epqtypes.QualityRating(
        rating=rating,
        page_length=page_length,
        affiliation=affiliation,
        citations=citations,
    )
    return quality


# #############################################################################
# Load the tier list
# #############################################################################


def _load_tiers(path: str) -> epqtypes.AffiliationTiers:
    """
    Load the dated affiliation tier list.

    :param path: path of `affiliation_tiers.yaml`
    :return: tier list
    """
    hdbg.dassert_file_exists(path)
    data = yaml.safe_load(hio.from_file(path)) or {}
    tiers = _parse_tiers(data)
    if not tiers.tier_5 and not tiers.tier_4:
        _LOG.warning(
            "The affiliation tier list '%s' is empty: each affiliation scores "
            "by its kind",
            path,
        )
    return tiers


# #############################################################################
# Stage `rate`
# #############################################################################


def _rate_paper(
    paper: epqtypes.Paper,
    output_dir: str,
    *,
    sources: epqbibso.BibSources,
    tiers: epqtypes.AffiliationTiers,
    today: datetime.date,
    refresh_citations: bool,
) -> None:
    """
    Rate one paper and write its rating file.

    :param paper: paper with a features file
    :param output_dir: dir of all the outputs of the run
    :param sources: clients of the lookup APIs, for the citation count
    :param tiers: dated affiliation tier list
    :param today: date used to annualize the citations
    :param refresh_citations: if True, ask for a fresh citation count
    """
    features = epqrepor.read_features_file(
        epqpaths.get_features_path(output_dir, paper.base_name)
    )
    citations = sources.get_citation_count(
        s2_paper_id=paper.s2_paper_id,
        doi=paper.doi,
        arxiv_id=paper.arxiv_id,
        refresh=refresh_citations,
    )
    rating_inputs = _RatingInputs(
        n_pages=features.n_pages,
        affiliations=features.affiliations,
        citations=citations,
        year=paper.year,
        today=today,
    )
    quality = _compute_rating(rating_inputs, tiers)
    epqrepor.write_rating_file(
        epqpaths.get_rating_path(output_dir, paper.base_name), quality
    )
    _LOG.debug("Rated '%s': rating=%s", paper.base_name, quality.rating)


def _rate(
    papers: List[epqtypes.Paper],
    output_dir: str,
    *,
    sources: epqbibso.BibSources,
    tiers: epqtypes.AffiliationTiers,
    today: datetime.date,
    refresh_citations: bool,
    dry_run: bool,
) -> List[epqtypes.Issue]:
    """
    Rate the papers that have a features file.

    The stage rates every paper at each run, since a new tier list or a new
    citation count changes the rating.

    :param papers: papers from the papers file
    :param output_dir: dir of all the outputs of the run
    :param sources: clients of the lookup APIs, for the citation count
    :param tiers: dated affiliation tier list
    :param today: date used to annualize the citations
    :param refresh_citations: if True, ask for fresh citation counts
    :param dry_run: if True, only log the papers to rate
    :return: issues of the papers that failed
    """
    timer = htimer.Timer()
    issues: List[epqtypes.Issue] = []
    n_no_input = 0
    n_ok = 0
    for paper in tqdm.tqdm(papers):
        features_path = epqpaths.get_features_path(output_dir, paper.base_name)
        if not os.path.isfile(features_path):
            # The extract issue already says why.
            n_no_input += 1
            continue
        if dry_run:
            _LOG.warning("[DRY_RUN] Would rate '%s'", paper.base_name)
            continue
        try:
            _rate_paper(
                paper,
                output_dir,
                sources=sources,
                tiers=tiers,
                today=today,
                refresh_citations=refresh_citations,
            )
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            # Decision D21: an error of one paper must not stop the run.
            _LOG.warning("Paper '%s' failed: %s", paper.base_name, e)
            issues.append(
                epqissue.make_paper_issue(
                    paper,
                    "rate",
                    "rating_failed",
                    epqissue.get_last_line(str(e)),
                )
            )
    issues_path = epqpaths.get_issues_path(output_dir, "rate")
    if dry_run:
        _LOG.warning("[DRY_RUN] Would write the issues of the stage `rate`")
    else:
        epqrepor.write_issues(issues_path, issues)
    # Summarize the stage.
    summary = epqtypes.StageSummary(
        stage="rate",
        unit="paper",
        n_total=len(papers),
        n_done_before=0,
        n_no_input=n_no_input,
        n_ok=n_ok,
        n_failed=len(issues),
        outputs=((epqpaths.get_rating_path(output_dir, "<BASE>"), n_ok),),
        issues_path=issues_path,
        issue_counts=epqsumm.count_issues(issues, add_stage=False),
        extra=(),
        elapsed_sec=timer.get_elapsed(),
        dry_run=dry_run,
    )
    epqsumm.emit_stage_summary(summary, output_dir)
    return issues


# #############################################################################
# Stage `report`
# #############################################################################


def _read_state_issues(output_dir: str, stage: str) -> List[epqtypes.Issue]:
    """
    Read the issues that a stage wrote, if the stage ran.

    :param output_dir: dir of all the outputs of the run
    :param stage: one of `resolve`, `download`, `extract`, `rate`
    :return: issues of the stage, empty if the stage did not run
    """
    path = epqpaths.get_issues_path(output_dir, stage)
    issues = epqrepor.read_issues(path) if os.path.isfile(path) else []
    return issues


@dataclasses.dataclass(frozen=True)
class _PaperFiles:
    """
    Content of the features files and of the rating files of the papers.
    """

    # Features by base name.
    features: Dict[str, epqtypes.PaperFeatures]
    # Ratings by base name.
    ratings: Dict[str, epqtypes.QualityRating]
    # Issues of the files that cannot be read or are missing.
    issues: List[epqtypes.Issue]


def _read_paper_files(
    papers: List[epqtypes.Paper],
    output_dir: str,
    rate_issues: List[epqtypes.Issue],
) -> _PaperFiles:
    """
    Read the features file and the rating file of each paper that has them.

    :param papers: papers from the papers file
    :param output_dir: dir of all the outputs of the run
    :param rate_issues: issues of the stage `rate`
    :return: features, ratings, and the issues of the files that cannot be read
        or are missing
    """
    features: Dict[str, epqtypes.PaperFeatures] = {}
    ratings: Dict[str, epqtypes.QualityRating] = {}
    issues: List[epqtypes.Issue] = []
    # The papers that the stage `rate` already reported.
    failed_rating: Set[str] = {i.base_name for i in rate_issues}
    for paper in papers:
        features_path = epqpaths.get_features_path(output_dir, paper.base_name)
        if not os.path.isfile(features_path):
            continue
        try:
            features[paper.base_name] = epqrepor.read_features_file(
                features_path
            )
        except Exception as e:  # noqa: BLE001
            # An old schema is an issue that asks for `--force_extract`. The
            # stage `rate` has already reported the same file, if it ran.
            if paper.base_name not in failed_rating:
                issues.append(
                    epqissue.make_paper_issue(
                        paper,
                        "extract",
                        "extraction_failed",
                        epqissue.get_last_line(str(e)),
                    )
                )
            continue
        rating_path = epqpaths.get_rating_path(output_dir, paper.base_name)
        if os.path.isfile(rating_path):
            try:
                ratings[paper.base_name] = epqrepor.read_rating_file(rating_path)
            except Exception as e:  # noqa: BLE001
                # An old schema is an issue that asks for the stage `rate`.
                issues.append(
                    epqissue.make_paper_issue(
                        paper,
                        "rate",
                        "rating_failed",
                        epqissue.get_last_line(str(e)),
                    )
                )
        elif paper.base_name not in failed_rating:
            # The stage `rate` did not run, or did not run after the extraction.
            issues.append(
                epqissue.make_paper_issue(
                    paper,
                    "rate",
                    "rating_missing",
                    "no rating file: run the action `rate`",
                )
            )
    return _PaperFiles(features=features, ratings=ratings, issues=issues)


def _report(
    papers: List[epqtypes.Paper],
    output_dir: str,
    *,
    dry_run: bool,
) -> None:
    """
    Write `references.md` and `report.json`.

    The issues of every stage are read from the files in the output dir, so that
    this stage can run alone. `report.json` has one row for each paper and for
    each unresolved line, with the issue of the row (decision D35).

    :param papers: papers from the papers file
    :param output_dir: dir of all the outputs of the run
    :param dry_run: if True, only log the files to write
    """
    timer = htimer.Timer()
    # Collect the issues of the previous stages.
    issues: List[epqtypes.Issue] = []
    resolve_issues_path = epqpaths.get_issues_path(output_dir, "resolve")
    if not os.path.isfile(resolve_issues_path):
        _LOG.warning(
            "No issues file of `resolve_papers.py` at '%s': the report lists "
            "no unresolved line",
            resolve_issues_path,
        )
    for stage in ["resolve", "download", "extract"]:
        issues.extend(_read_state_issues(output_dir, stage))
    rate_issues = _read_state_issues(output_dir, "rate")
    issues.extend(rate_issues)
    # Read the files of each paper that has them.
    files = _read_paper_files(papers, output_dir, rate_issues)
    issues.extend(files.issues)
    rows = epqrepor.build_report_rows(
        papers, files.features, files.ratings, issues
    )
    if dry_run:
        _LOG.warning(
            "[DRY_RUN] Would write `references.md` and `report.json` in '%s'",
            output_dir,
        )
    else:
        epqrepor.write_references(
            epqpaths.get_references_path(output_dir), papers
        )
        epqrepor.write_report(epqpaths.get_report_path(output_dir), rows)
    # Summarize the stage: the issues are in the rows of `report.json`.
    n_ok = sum(1 for row in rows if row["status"] == "ok")
    summary = epqtypes.StageSummary(
        stage="report",
        unit="row",
        n_total=len(rows),
        n_done_before=0,
        n_no_input=0,
        n_ok=n_ok,
        n_failed=len(rows) - n_ok,
        outputs=(
            (epqpaths.get_references_path(output_dir), 1),
            (epqpaths.get_report_path(output_dir), 1),
        ),
        issues_path=epqpaths.get_report_path(output_dir),
        issue_counts=epqsumm.count_issues(issues, add_stage=True),
        extra=(),
        elapsed_sec=timer.get_elapsed(),
        dry_run=dry_run,
    )
    epqsumm.emit_stage_summary(summary, output_dir)


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    # The flag `--output_dir` follows decision D10: the output dir is required
    # and the user decides where it lives.
    parser.add_argument(
        "-i",
        "--input",
        action="store",
        default="",
        help="Papers file that `resolve_papers.py` wrote; empty is "
        "`01-resolve.paper_list.json` in the output dir",
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        required=True,
        help="Dir of all the outputs and of the state of the run",
    )
    hparser.add_bool_arg(
        parser,
        "refresh_citations",
        default_value=False,
        help_="Ask for fresh citation counts in the stage `rate`",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show the papers to rate and the files to write, with no lookup "
        "and no output file",
    )
    hcacsimp.add_cache_control_arg(parser)
    hselacti.add_action_arg(parser, _VALID_ACTIONS, _DEFAULT_ACTIONS)
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    hcacsimp.parse_cache_control_args(args)
    # Check the boundary of the system.
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    papers_file = args.input or epqpaths.get_papers_path(output_dir)
    papers_file = os.path.abspath(os.path.expanduser(papers_file))
    hdbg.dassert_file_exists(
        papers_file, "Run resolve_papers.py to make the papers file"
    )
    # Select the stages to run.
    actions = hselacti.select_actions(args, _VALID_ACTIONS, _DEFAULT_ACTIONS)
    _LOG.info(
        "\n%s",
        hselacti.actions_to_string(actions, _VALID_ACTIONS, add_frame=True),
    )
    # Run the stages on the papers of the papers file.
    papers = epqrepor.read_papers(papers_file)
    if not args.dry_run:
        hio.create_dir(epqpaths.get_cache_dir(output_dir), incremental=True)
    while actions:
        action = actions[0]
        to_execute, actions = hselacti.mark_action(action, actions)
        if not to_execute:
            continue
        if action == "rate":
            # The cache of the helpers lives in the output dir with the other
            # state.
            cache_dir = epqpaths.get_cache_dir(output_dir)
            hcacsimp.set_cache_dir(cache_dir)
            # The stage only asks for citation counts, which need no contact
            # email.
            sources = epqbibso.BibSources(
                email="",
                s2_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
                cache_dir=cache_dir,
                cache_mode=args.cache_mode,
            )
            tiers_path = hgit.find_file_in_git_tree("affiliation_tiers.yaml")
            tiers = _load_tiers(tiers_path)
            _rate(
                papers,
                output_dir,
                sources=sources,
                tiers=tiers,
                today=datetime.date.today(),
                refresh_citations=args.refresh_citations,
                dry_run=args.dry_run,
            )
        elif action == "report":
            _report(papers, output_dir, dry_run=args.dry_run)
        else:
            raise ValueError(f"Invalid action='{action}'")
    hdbg.dassert_eq(len(actions), 0, "Unprocessed actions: %s", str(actions))
    _LOG.info("Outputs are in '%s'", output_dir)


if __name__ == "__main__":
    _main(_parse())
