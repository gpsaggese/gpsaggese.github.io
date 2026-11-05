"""
Dataclasses shared by all the modules of the Estimate Paper Quality pipeline.

This module holds data only and imports nothing of the project. All the
dataclasses are frozen, since they cross a module boundary.

The design is in `docs/02-design.architecture.md`, section "Data Model".

The examples in the docstrings use the paper `Lazy Prices` as the running case.
The identity of the paper is real, all the numbers and quotes are made up.

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

    Example, for the line `- Lazy Prices (Lauren Cohen; Christopher Malloy)`:
    ```
    InputLine(
        line_no=12,
        text="Lazy Prices (Lauren Cohen; Christopher Malloy)",
        kind="title",
        key="Lazy Prices",
        author_hints=("Lauren Cohen", "Christopher Malloy"),
    )
    ```
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

    Example, a Crossref record:
    ```
    BibRecord(
        source="crossref",
        title="Lazy Prices",
        authors=("Cohen, Lauren", "Malloy, Christopher", "Nguyen, Quoc"),
        year=2020,
        record_type="journal-article",
        doi="10.1111/jofi.12885",
        arxiv_id="",
        ssrn_id="",
        s2_paper_id="",
        pdf_urls=(),
    )
    ```
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

    Example, for a publisher page that is not a PDF:
    ```
    UrlMeta(
        pdf_path="",
        doi="10.1111/jofi.12885",
        title="Lazy Prices",
        authors=("Lauren Cohen", "Christopher Malloy", "Quoc Nguyen"),
    )
    ```
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

    Example:
    ```
    CitationCount(count=512, source="Semantic Scholar", fetched_at="2026-10-28")
    ```
    """

    count: int
    source: str
    # Date of the lookup, in UTC, e.g., `2026-10-28`.
    fetched_at: str


@dataclasses.dataclass(frozen=True)
class Paper:
    """
    A resolved, deduped paper.

    Example, the paper that two input lines (3 and 17) give:
    ```
    Paper(
        key="lazy prices|cohen",
        aliases=("doi:10.1111/jofi.12885",),
        line_nos=(3, 17),
        source="semantic_scholar",
        year=2020,
        authors=("Cohen, Lauren", "Malloy, Christopher", "Nguyen, Quoc"),
        title="Lazy Prices",
        doi="10.1111/jofi.12885",
        arxiv_id="",
        s2_paper_id="0123456789abcdef0123456789abcdef01234567",
        reference='2020, Cohen et al., "Lazy Prices"',
        links="(https://doi.org/10.1111/jofi.12885)",
        base_name="2020.Cohen_et_al.Lazy_Prices",
        download_inputs=("https://example.org/lazy_prices.pdf",),
    )
    ```
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
    # Reference in the format of `.claude/skills/references.rules.md`, without
    # the links: `<Year>, <Last name> [et al.], "<Title>"`.
    reference: str
    # Links of the reference, in parentheses: `(<Link>)`, where `<Link>` is the
    # first public download input, the DOI URL followed by ` ???` if there is
    # none, or `???` if there is no DOI either (decision D37).
    links: str
    # Base name of the files of the paper, without extension.
    base_name: str
    # Ordered inputs to try with the download tool.
    download_inputs: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Issue:
    """
    An unresolved line or a failed paper.

    Example, a title line that matches two papers:
    ```
    Issue(
        line_no=8,
        input="Deep Learning in Asset Pricing",
        base_name="",
        stage="resolve",
        status="ambiguous",
        reason="2 papers match the title with the same score",
        candidates=(
            '2020, Chen et al., "Deep Learning in Asset Pricing" (...)',
            '2019, Chen et al., "Deep Learning in Asset Pricing" (...)',
        ),
    )
    ```
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

    Example, from a list of 22 lines:
    ```
    ResolveResult(
        papers=(Paper(...), ...),  # 20 papers
        issues=(Issue(...), ...),  # 2 issues
    )
    ```
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

    Example:
    ```
    FetchResult(status="ok", reason="")
    FetchResult(status="download_failed", reason="HTTP 403 for the last input")
    ```
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

    Example:
    ```
    Evidence(
        quote="The out of sample Sharpe ratio is 1.4.",
        location="Table 3, p. 21",
        verified=True,
    )
    ```
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

    Example:
    ```
    FeatureValue(
        value="Equity long short portfolio",
        evidence=Evidence(
            quote="We form a long short portfolio of stocks.",
            location="Sec. 4.2, p. 17",
            verified=True,
        ),
    )
    FeatureValue(value="unknown", evidence=None)
    ```
    """

    # Free text as the paper states it, or `unknown`.
    value: str
    evidence: Optional[Evidence]


@dataclasses.dataclass(frozen=True)
class LabeledFact:
    """
    A fact that a paper states with a label, e.g., a Sharpe ratio.

    Example:
    ```
    LabeledFact(
        label="net of costs, out of sample, 2005 to 2020",
        value="1.4",
        evidence=Evidence(
            quote="The out of sample Sharpe ratio is 1.4.",
            location="Table 3, p. 21",
            verified=True,
        ),
    )
    ```
    """

    # E.g., `net of costs, out of sample, 2005 to 2020`.
    label: str
    value: str
    evidence: Evidence


@dataclasses.dataclass(frozen=True)
class ExtractedFacts:
    """
    Facts of `R15`, as the paper states them.

    Example, a paper with one Sharpe ratio and no other number:
    ```
    ExtractedFacts(
        sharpe_ratios=(LabeledFact(...),),  # See `LabeledFact`
        annual_returns=(),
        volatilities=(),
        max_drawdowns=(),
        sample_periods=(
            LabeledFact(label="full sample", value="2005 to 2020", ...),
        ),
        universe=FeatureValue(value="US common stocks", evidence=...),
        data_assets=FeatureValue(value="10-K and 10-Q filings", evidence=...),
        data_frequency=FeatureValue(value="quarterly", evidence=...),
        data_fields=FeatureValue(value="unknown", evidence=None),
    )
    ```
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

    Example:
    ```
    Affiliation(
        author="Lauren Cohen",
        institution="Harvard Business School",
        kind="university",
        evidence=Evidence(
            quote="Lauren Cohen, Harvard Business School",
            location="Title page, p. 1",
            verified=True,
        ),
    )
    ```
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

    Example:
    ```
    Provenance(
        model="gpt-5-mini",
        prompt_sha1="da39a3ee5e6b4b0d3255bfef95601890afd80709",
        chars_sent=212345,
        text_was_cut=False,
        cost_usd=0.0412,
    )
    ```
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

    Example:
    ```
    ExtractionResult(
        performance=FeatureValue(value="Sharpe ratio of 1.4", evidence=...),
        asset_class=FeatureValue(value="Equities", evidence=...),
        theoretical_domain=FeatureValue(value="Textual analysis", evidence=...),
        mechanism=FeatureValue(value="Investors miss changes in filings", ...),
        facts=ExtractedFacts(...),  # See `ExtractedFacts`
        affiliations=(Affiliation(...), ...),  # One per author
        provenance=Provenance(...),  # See `Provenance`
    )
    ```
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

    Example:
    ```
    TierEntry(name="Harvard University", aliases=("Harvard", "HBS"))
    ```
    """

    name: str
    aliases: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class AffiliationTiers:
    """
    Dated list of the institutions that score 5 and 4.

    Example:
    ```
    AffiliationTiers(
        as_of="2026-10-01",
        source="QS World University Rankings",
        tier_5=(TierEntry(name="Harvard University", aliases=("Harvard",)),),
        tier_4=(TierEntry(name="Boston College", aliases=("BC",)),),
    )
    ```
    """

    as_of: str
    source: str
    tier_5: Tuple[TierEntry, ...]
    tier_4: Tuple[TierEntry, ...]


@dataclasses.dataclass(frozen=True)
class SignalScore:
    """
    Score of one signal of the Paper quality rating.

    Example, a paper of 41 pages (score 5) and an unknown signal:
    ```
    SignalScore(
        name="page_length", value="41 pages", score=5, evidence=None, note=""
    )
    SignalScore(
        name="citations",
        value="unknown",
        score=None,
        evidence=None,
        note="Semantic Scholar does not have the paper",
    )
    ```
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

    Example, with 41 pages (5), a university (3), and 512 citations (4):
    ```
    QualityRating(
        rating=4,  # Mean 4.0 of the three scores
        page_length=SignalScore(name="page_length", score=5, ...),
        affiliation=SignalScore(name="affiliation", score=3, ...),
        citations=SignalScore(name="citations", score=4, ...),
    )
    ```
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

    Example:
    ```
    PaperFeatures(
        base_name="2020.Cohen_et_al.Lazy_Prices",
        features={
            "performance": FeatureValue(value="Sharpe ratio of 1.4", ...),
            "recency": FeatureValue(value="2020", ...),
            "asset_class": FeatureValue(value="Equities", ...),
            "theoretical_domain": FeatureValue(value="Textual analysis", ...),
            "mechanism": FeatureValue(value="unknown", evidence=None),
        },
        facts=ExtractedFacts(...),
        affiliations=(Affiliation(...), ...),
        n_pages=41,
        provenance=Provenance(...),
    )
    ```

    The rating is not here: `report_papers.py` rates a paper from this file, so
    that a change of the tier list or of the bands needs no new LLM call.
    """

    base_name: str
    # Keys: `performance`, `recency`, `asset_class`, `theoretical_domain`,
    # `mechanism`.
    features: Dict[str, FeatureValue]
    facts: ExtractedFacts
    # One per author, as the input of the rating.
    affiliations: Tuple[Affiliation, ...]
    # Pages of the PDF, `None` when the PDF is missing or cannot be read.
    n_pages: Optional[int]
    provenance: Provenance


# #############################################################################
# Stage summary
# #############################################################################


@dataclasses.dataclass(frozen=True)
class StageSummary:
    """
    The counts and the output files of one stage in one run.

    Each item of the input of the stage is in exactly one of four groups:
    `n_done_before`, `n_no_input`, `n_ok`, or `n_failed`. In a dry run the items
    that the stage would process are in none of them.

    Example, for the stage `download` with 20 papers:
    ```
    StageSummary(
        stage="download",
        unit="paper",
        n_total=20,
        n_done_before=12,
        n_no_input=0,
        n_ok=6,
        n_failed=2,
        outputs=(("run/paper_pdf/<BASE>.pdf", 6), ("run/paper_md/<BASE>.md", 6)),
        issues_path="run/02-download.issues.json",
        issue_counts=(("download_failed", 1), ("no_download_link", 1)),
        extra=(),
        elapsed_sec=75.3,
        dry_run=False,
    )
    ```
    """

    # One of `resolve`, `download`, `extract`, `rate`, `report`.
    stage: str
    # What the stage counts, `line`, `paper`, or `row` (of the report).
    unit: str
    # Items in the input of the stage.
    n_total: int
    # Skipped since the output of an earlier run exists.
    n_done_before: int
    # Skipped since an earlier stage failed for the item.
    n_no_input: int
    # Processed in this run with no issue.
    n_ok: int
    # Processed in this run with an issue.
    n_failed: int
    # Pairs of a path (or a path with `<BASE>`) and the number of files that the
    # run wrote there.
    outputs: Tuple[Tuple[str, int], ...]
    # Path of the file with the issues of the stage, empty in a dry run.
    issues_path: str
    # Pairs of a label and the number of issues with that label.
    issue_counts: Tuple[Tuple[str, int], ...]
    # Other lines to add to the summary, e.g., the cost of the LLM calls.
    extra: Tuple[str, ...]
    # Wall-clock time of the stage in this run, in seconds (decision D36).
    elapsed_sec: float
    dry_run: bool
