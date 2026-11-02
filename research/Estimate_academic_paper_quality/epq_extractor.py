"""
Read the markdown of one paper and return its features with one LLM call.

The call returns:
- The features Performance, Asset class, Theoretical domain, and Mechanism
- The facts of the PRD requirement `R15`: every labeled Sharpe ratio, annual
  return, volatility, max drawdown, sample period, universe, and data needs
- The affiliations of the authors, for the rating of `epq_rater`
- A quote and a location for each value

Each quote is checked against the text of the PDF, which also gives its page.
The LLM function is injected, so this module does not import `helpers.hllm`.

The design is in `docs/02.5-design.extractor.md`.

Import as:

import research.Estimate_academic_paper_quality.epq_extractor as epqextra
"""

import functools
import hashlib
import logging
import re
import unicodedata
from typing import Any, Callable, List, Literal, Optional, Tuple

import helpers.hgit as hgit
import helpers.hio as hio
import helpers.hprint as hprint
import pydantic

import research.Estimate_academic_paper_quality.epq_types as epqtypes

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

# Marker put where the middle of a long paper was cut.
_CUT_MARKER = "[... text cut ...]"

# Share of the character budget kept from the start of a long paper. The rest
# is kept from the end, since finance papers put their tables there.
_HEAD_SHARE = 0.6

# Titles of the headings that start the reference list.
_REFERENCE_TITLES = ["references", "bibliography"]

# Same signature as `hllm.get_structured_completion()` once `model` and
# `cost_tracker` are bound.
LlmFunc = Callable[..., Any]


# #############################################################################
# Shape of the answer of the LLM
# #############################################################################


class _LlmEvidence(pydantic.BaseModel):
    quote: str
    # E.g., "Sec. 4.2", "Table 3".
    section: str


class _LlmFeature(pydantic.BaseModel):
    # Free text as the paper states it, or "unknown".
    value: str
    evidence: Optional[_LlmEvidence]


class _LlmLabeledFact(pydantic.BaseModel):
    # E.g., "net of costs, out of sample, 2005 to 2020".
    label: str
    value: str
    evidence: _LlmEvidence


class _LlmAffiliation(pydantic.BaseModel):
    author: str
    # Top-level institution as written, e.g., "Harvard University".
    institution: str
    kind: Literal[
        "university", "research_institute", "central_bank", "company", "none"
    ]
    evidence: _LlmEvidence


class _LlmExtraction(pydantic.BaseModel):
    performance: _LlmFeature
    asset_class: _LlmFeature
    theoretical_domain: _LlmFeature
    mechanism: _LlmFeature
    sharpe_ratios: List[_LlmLabeledFact]
    annual_returns: List[_LlmLabeledFact]
    volatilities: List[_LlmLabeledFact]
    max_drawdowns: List[_LlmLabeledFact]
    sample_periods: List[_LlmLabeledFact]
    universe: _LlmFeature
    data_assets: _LlmFeature
    data_frequency: _LlmFeature
    data_fields: _LlmFeature
    affiliations: List[_LlmAffiliation]


# #############################################################################
# Prepare the text
# #############################################################################


@functools.cache
def _get_system_prompt() -> str:
    """
    Read the system prompt, which lives in its own file for review.

    :return: text of `epq_extractor.prompt.md`
    """
    path = hgit.find_file_in_git_tree("epq_extractor.prompt.md")
    return hio.from_file(path)


def _drop_references(md_text: str) -> str:
    """
    Drop the reference list from the markdown of a paper.

    The list runs from a heading `References` or `Bibliography` to the next
    heading of the same level or a higher one, or to the end.

    :param md_text: markdown of the paper
    :return: markdown without the reference list
    """
    kept = []
    drop_level = 0
    for line in md_text.split("\n"):
        match = re.match(r"^(#{1,6})\s+(.*?)\s*$", line)
        if match:
            level = len(match.group(1))
            # Remove the emphasis and the number, e.g., `**7. References**`.
            title = re.sub(r"[*_`]", "", match.group(2)).strip().lower()
            title = re.sub(r"^(?:\d+(?:\.\d+)*\.?|[ivx]+\.)\s*", "", title)
            if drop_level and level <= drop_level:
                drop_level = 0
            if not drop_level and title in _REFERENCE_TITLES:
                drop_level = level
        if not drop_level:
            kept.append(line)
    return "\n".join(kept)


def _prepare_text(md_text: str, max_chars: int) -> Tuple[str, bool]:
    """
    Drop the references of a paper and cut it to the character budget.

    A paper over the budget keeps the first 60% and the last 40% of the budget,
    joined by a marker.

    :param md_text: markdown of the paper
    :param max_chars: budget of characters to send to the LLM
    :return: text to send, and True if the middle was cut
    """
    text = _drop_references(md_text)
    was_cut = len(text) > max_chars
    if was_cut:
        head_chars = int(max_chars * _HEAD_SHARE)
        tail_chars = max_chars - head_chars
        tail = text[len(text) - tail_chars :] if tail_chars > 0 else ""
        text = f"{text[:head_chars]}\n{_CUT_MARKER}\n{tail}"
    return text, was_cut


# #############################################################################
# Check the quotes
# #############################################################################


def _normalize_for_search(text: str) -> str:
    """
    Normalize a text so that a quote of the markdown matches the PDF text.

    The text is NFKC, so that a ligature glyph matches its letters. Words cut by
    a hyphen at the end of a line are joined. The markdown marks of emphasis and
    of tables are removed, and the spaces are collapsed.

    :param text: text from the markdown or from a page of the PDF
    :return: lowercase text with single spaces
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"[*`|#]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _locate_quote(quote: str, page_texts: List[str]) -> int:
    """
    Find the first page of the PDF whose text contains a quote.

    :param quote: text that the LLM copied from the paper
    :param page_texts: text of each page of the PDF
    :return: page number counted from 1, 0 if no page contains the quote
    """
    page_no = 0
    needle = _normalize_for_search(quote)
    if needle:
        for i, page_text in enumerate(page_texts, start=1):
            if needle in _normalize_for_search(page_text):
                page_no = i
                break
    return page_no


# #############################################################################
# Convert the answer
# #############################################################################


def _to_evidence(
    llm_evidence: _LlmEvidence, page_texts: List[str]
) -> epqtypes.Evidence:
    """
    Convert the evidence of the LLM, adding the page when the quote is found.

    :param llm_evidence: quote and section from the LLM
    :param page_texts: text of each page of the PDF
    :return: evidence, verified if the quote is in the text of the PDF
    """
    section = llm_evidence.section.strip()
    page_no = _locate_quote(llm_evidence.quote, page_texts)
    if page_no:
        location = f"{section}, p. {page_no}" if section else f"p. {page_no}"
    else:
        location = section
    return epqtypes.Evidence(
        quote=llm_evidence.quote, location=location, verified=page_no > 0
    )


def _to_feature(
    llm_feature: _LlmFeature, page_texts: List[str]
) -> epqtypes.FeatureValue:
    """
    Convert a feature of the LLM into a feature value.

    :param llm_feature: value and evidence from the LLM
    :param page_texts: text of each page of the PDF
    :return: feature value, `unknown` with no evidence if the paper has none
    """
    value = llm_feature.value.strip()
    if value.lower() in ["", "unknown"]:
        feature = epqtypes.FeatureValue(value="unknown", evidence=None)
    elif llm_feature.evidence is None:
        # Keep the value: the report shows that it has no quote to check.
        feature = epqtypes.FeatureValue(value=value, evidence=None)
    else:
        feature = epqtypes.FeatureValue(
            value=value, evidence=_to_evidence(llm_feature.evidence, page_texts)
        )
    return feature


def _to_facts(
    llm_facts: List[_LlmLabeledFact], page_texts: List[str]
) -> Tuple[epqtypes.LabeledFact, ...]:
    """
    Convert the labeled facts of the LLM.

    :param llm_facts: facts from the LLM
    :param page_texts: text of each page of the PDF
    :return: facts with checked evidence
    """
    return tuple(
        epqtypes.LabeledFact(
            label=fact.label,
            value=fact.value,
            evidence=_to_evidence(fact.evidence, page_texts),
        )
        for fact in llm_facts
    )


def extract_features(
    md_text: str,
    page_texts: List[str],
    *,
    llm_func: LlmFunc,
    max_chars: int,
    model: str,
) -> epqtypes.ExtractionResult:
    """
    Extract the features of a paper with one LLM call.

    :param md_text: markdown of the paper
    :param page_texts: text of each page of the PDF, to check the quotes
        - An empty list leaves every quote unverified
    :param llm_func: function with the signature of
        `hllm.get_structured_completion()`, with the model already bound
    :param max_chars: budget of characters to send to the LLM
    :param model: name of the model, recorded in the provenance
    :return: features, facts, affiliations, and provenance (model, SHA-1 of the
        prompt, characters sent, whether the text was cut)
    """
    _LOG.debug(hprint.to_str("max_chars model"))
    text, was_cut = _prepare_text(md_text, max_chars)
    system_prompt = _get_system_prompt()
    llm_answer = llm_func(
        user_prompt=text,
        response_format=_LlmExtraction,
        system_prompt=system_prompt,
    )
    # Convert the answer and check each quote against the PDF.
    facts = epqtypes.ExtractedFacts(
        sharpe_ratios=_to_facts(llm_answer.sharpe_ratios, page_texts),
        annual_returns=_to_facts(llm_answer.annual_returns, page_texts),
        volatilities=_to_facts(llm_answer.volatilities, page_texts),
        max_drawdowns=_to_facts(llm_answer.max_drawdowns, page_texts),
        sample_periods=_to_facts(llm_answer.sample_periods, page_texts),
        universe=_to_feature(llm_answer.universe, page_texts),
        data_assets=_to_feature(llm_answer.data_assets, page_texts),
        data_frequency=_to_feature(llm_answer.data_frequency, page_texts),
        data_fields=_to_feature(llm_answer.data_fields, page_texts),
    )
    affiliations = tuple(
        epqtypes.Affiliation(
            author=item.author,
            institution=item.institution,
            kind=item.kind,
            evidence=_to_evidence(item.evidence, page_texts),
        )
        for item in llm_answer.affiliations
    )
    provenance = epqtypes.Provenance(
        model=model,
        prompt_sha1=hashlib.sha1(system_prompt.encode("utf-8")).hexdigest(),
        chars_sent=len(text),
        text_was_cut=was_cut,
        cost_usd=0.0,
    )
    result = epqtypes.ExtractionResult(
        performance=_to_feature(llm_answer.performance, page_texts),
        asset_class=_to_feature(llm_answer.asset_class, page_texts),
        theoretical_domain=_to_feature(
            llm_answer.theoretical_domain, page_texts
        ),
        mechanism=_to_feature(llm_answer.mechanism, page_texts),
        facts=facts,
        affiliations=affiliations,
        provenance=provenance,
    )
    return result
