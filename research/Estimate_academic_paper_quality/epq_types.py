"""
Dataclasses shared by all the modules of the Estimate Paper Quality pipeline.

This module holds data only and imports nothing of the project. All the
dataclasses are frozen, since they cross a module boundary.

The design is in `docs/02-design.architecture.md`, section "Data Model".

Import as:

import research.Estimate_academic_paper_quality.epq_types as epqtypes
"""

import dataclasses
from typing import Dict, Optional, Tuple

# #############################################################################
# Resolve stage
# #############################################################################


@dataclasses.dataclass(frozen=True)
class InputLine:
    """
    One line of the input file, classified.
    """

    # Number of the first line of the entry in the input file, counted from 1.
    line_no: int
    # Text of the entry without the bullet marker.
    text: str
    # One of `local_pdf`, `ssrn`, `doi`, `arxiv`, `url`, `title`.
    kind: str
    # Identifier of the paper that the line gives, e.g., the arXiv id or the
    # title text.
    key: str
    # Author names in parentheses after a title, e.g., `("Lauren Cohen",)`.
    author_hints: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class BibRecord:
    """
    One record of one bibliographic source.
    """

    # One of `semantic_scholar`, `crossref`, `arxiv`.
    source: str
    title: str
    # Authors in the form `Family, Given`.
    authors: Tuple[str, ...]
    # Publication year, 0 if the source does not give it.
    year: int
    # E.g., `journal-article`, `posted-content`, or "" if unknown.
    record_type: str
    doi: str
    arxiv_id: str
    ssrn_id: str
    s2_paper_id: str
    # Public links to the PDF that the source gives.
    pdf_urls: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class UrlMeta:
    """
    Facts read from the page or the PDF that a `url` line points to.
    """

    # Path of the PDF saved in the cache, "" if the URL is not a PDF.
    pdf_path: str
    doi: str
    title: str
    authors: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class CitationCount:
    """
    Citation count of a paper, with its source and date.
    """

    count: int
    source: str
    # Date of the lookup, in UTC, e.g., `2026-10-28`.
    fetched_at: str


@dataclasses.dataclass(frozen=True)
class Paper:
    """
    A resolved, deduped paper.
    """

    # Normalized title plus the family name of the first author.
    key: str
    # Other identifiers of the same paper, e.g., `doi:10.1/x`.
    aliases: Tuple[str, ...]
    # Lines of the input file that give this paper.
    line_nos: Tuple[int, ...]
    # Source of the canonical record.
    source: str
    year: int
    authors: Tuple[str, ...]
    title: str
    doi: str
    arxiv_id: str
    s2_paper_id: str
    # Reference in the format of `.claude/skills/references.rules.md`.
    reference: str
    # Base name of the files of the paper, without extension.
    base_name: str
    # Ordered inputs to try with the download tool.
    download_inputs: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Issue:
    """
    An unresolved line or a failed paper.
    """

    # 0 if the issue is not tied to a line.
    line_no: int
    input: str
    base_name: str
    # One of `resolve`, `download`, `extract`.
    stage: str
    # One of `unresolved`, `ambiguous`, `no_download_link`, `download_failed`,
    # `conversion_failed`, `extraction_failed`.
    status: str
    reason: str
    # References of the candidate papers, if any.
    candidates: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ResolveResult:
    """
    Output of the resolve stage.
    """

    papers: Tuple[Paper, ...]
    issues: Tuple[Issue, ...]


# #############################################################################
# Download stage
# #############################################################################


@dataclasses.dataclass(frozen=True)
class FetchResult:
    """
    Outcome of the download and the conversion of one paper.
    """

    # One of `ok`, `no_download_link`, `download_failed`, `conversion_failed`.
    status: str
    reason: str


# #############################################################################
# Extract stage
# #############################################################################


@dataclasses.dataclass(frozen=True)
class Evidence:
    """
    Where a value comes from.
    """

    # Text of the paper, or the value given by an external source.
    quote: str
    # E.g., `Sec. 4.2, p. 17`, `Table 3, p. 21`, `Semantic Scholar`.
    location: str
    # True if the quote was found in the text of the PDF.
    verified: bool


@dataclasses.dataclass(frozen=True)
class FeatureValue:
    """
    Value of one feature with its evidence.
    """

    # Free text as the paper states it, or `unknown`.
    value: str
    evidence: Optional[Evidence]


@dataclasses.dataclass(frozen=True)
class LabeledFact:
    """
    A fact that a paper states with a label, e.g., a Sharpe ratio.
    """

    # E.g., `net of costs, out of sample, 2005 to 2020`.
    label: str
    value: str
    evidence: Evidence


@dataclasses.dataclass(frozen=True)
class ExtractedFacts:
    """
    Facts of `R15`, as the paper states them.
    """

    sharpe_ratios: Tuple[LabeledFact, ...]
    annual_returns: Tuple[LabeledFact, ...]
    volatilities: Tuple[LabeledFact, ...]
    max_drawdowns: Tuple[LabeledFact, ...]
    sample_periods: Tuple[LabeledFact, ...]
    universe: FeatureValue
    data_assets: FeatureValue
    data_frequency: FeatureValue
    data_fields: FeatureValue


@dataclasses.dataclass(frozen=True)
class Affiliation:
    """
    Affiliation of one author, as the paper writes it.
    """

    author: str
    # Top-level institution as written, e.g., `Harvard University`.
    institution: str
    # One of `university`, `research_institute`, `central_bank`, `company`,
    # `none`.
    kind: str
    evidence: Evidence


@dataclasses.dataclass(frozen=True)
class Provenance:
    """
    How the features of a paper were produced.
    """

    model: str
    prompt_sha1: str
    chars_sent: int
    text_was_cut: bool
    cost_usd: float


@dataclasses.dataclass(frozen=True)
class ExtractionResult:
    """
    Output of the extractor for one paper.
    """

    performance: FeatureValue
    asset_class: FeatureValue
    theoretical_domain: FeatureValue
    mechanism: FeatureValue
    facts: ExtractedFacts
    affiliations: Tuple[Affiliation, ...]
    provenance: Provenance


# #############################################################################
# Rating
# #############################################################################


@dataclasses.dataclass(frozen=True)
class TierEntry:
    """
    An institution of the affiliation tier list.
    """

    name: str
    aliases: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class AffiliationTiers:
    """
    Dated list of the institutions that score 5 and 4.
    """

    as_of: str
    source: str
    tier_5: Tuple[TierEntry, ...]
    tier_4: Tuple[TierEntry, ...]


@dataclasses.dataclass(frozen=True)
class SignalScore:
    """
    Score of one signal of the Paper quality rating.
    """

    # One of `page_length`, `affiliation`, `citations`.
    name: str
    # What was measured, e.g., `12 pages`, or `unknown`.
    value: str
    # Score from 1 to 5, `None` if the signal is `unknown`.
    score: Optional[int]
    evidence: Optional[Evidence]
    # E.g., the annualization of a young paper, or why the signal is unknown.
    note: str


@dataclasses.dataclass(frozen=True)
class QualityRating:
    """
    Paper quality rating and the three signals behind it.
    """

    # Whole number from 1 to 5, `None` if no signal is known.
    rating: Optional[int]
    page_length: SignalScore
    affiliation: SignalScore
    citations: SignalScore


# #############################################################################
# Output
# #############################################################################


@dataclasses.dataclass(frozen=True)
class PaperFeatures:
    """
    Everything that goes into the features file of one paper.
    """

    base_name: str
    # Keys: `performance`, `recency`, `asset_class`, `theoretical_domain`,
    # `mechanism`.
    features: Dict[str, FeatureValue]
    facts: ExtractedFacts
    quality: QualityRating
    provenance: Provenance
