"""
Compute the Paper quality rating of one paper from three signals.

The signals are page length, best affiliation of the authors, and citation
count. The score tables are copied from `.claude/skills/paper.rules.md`,
`# Paper Rating`. The module is pure code: no network, no LLM, no file I/O.

The design is in `docs/02.6-design.rater.md`.

Import as:

import research.Estimate_academic_paper_quality.epq_rater as epqrater
"""

import dataclasses
import datetime
import logging
import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

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


# #############################################################################
# Inputs
# #############################################################################


@dataclasses.dataclass(frozen=True)
class RatingInputs:
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


def parse_tiers(data: Dict[str, Any]) -> epqtypes.AffiliationTiers:
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


def score_page_length(n_pages: Optional[int]) -> epqtypes.SignalScore:
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


def score_affiliation(
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


def score_citations(
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


def rate_paper(
    inputs: RatingInputs, tiers: epqtypes.AffiliationTiers
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
    page_length = score_page_length(inputs.n_pages)
    affiliation = score_affiliation(inputs.affiliations, tiers)
    citations = score_citations(inputs.citations, inputs.year, inputs.today)
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
