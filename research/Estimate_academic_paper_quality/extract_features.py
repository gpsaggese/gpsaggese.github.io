#!/usr/bin/env -S uv run

# /// script
# dependencies = ["pymupdf", "pyyaml", "pydantic", "tqdm", "llm", "llm-openrouter", "tokencost", "requests"]
# ///

"""
Extract the features of each downloaded paper with an LLM.

The input is the papers file that `resolve_papers.py` writes, by default
`01-resolve.paper_list.json` in `--output_dir`, and the markdown and PDF files
that `download_papers.py` writes there. The script calls the LLM once per paper,
and checks every quote of the answer against the text of the PDF.

The LLM is called with `helpers.hllm_cli.apply_llm()`, which uses the `llm`
library. The system prompt asks for a JSON answer in the shape of the pydantic
class `_LlmExtraction`, and the script validates the answer with that class.

The outputs live in `--output_dir`:
- `papers/<BASE>.features.json` for each paper: the features, the facts, the
  affiliations, the page count, and the cost of the call
- `03-extract.issues.json`: the papers that failed, for `report_papers.py`
- `03-extract.summary.txt`: the counts, the output files, the issues file, and the
  cost of the run, the same text as the end of the log

The script does not rate the paper: a change of the tier list or of the bands
needs no new LLM call, see `report_papers.py`.

A rerun does not pay again for the papers that have a features file.

The next stage is `report_papers.py`.

The design is in `research/Estimate_academic_paper_quality/docs/02-design.architecture.md`.

# Usage Example

- Extract the features of the downloaded papers:
> extract_features.py --output_dir out

- Extract again the papers that have a features file, with another model:
> extract_features.py --output_dir out --force_extract --model gpt-4o-mini

- List the papers to extract, with no LLM call:
> extract_features.py --output_dir out --dry_run

For a description of this file, see the file
`research/Estimate_academic_paper_quality/extract_features.py.README.md`
"""

import argparse
import dataclasses
import functools
import hashlib
import json
import logging
import os
import re
import unicodedata
from typing import Callable, List, Literal, Optional, Tuple

import helpers.hdbg as hdbg
import helpers.hgit as hgit
import helpers.hio as hio
import helpers.hllm_cli as hllmcli
import helpers.hparser as hparser
import helpers.hprint as hprint
import helpers.htimer as htimer
import pydantic
import tqdm

import research.Estimate_academic_paper_quality.epq_fetcher as epqfetch
import research.Estimate_academic_paper_quality.epq_issues as epqissue
import research.Estimate_academic_paper_quality.epq_paths as epqpaths
import research.Estimate_academic_paper_quality.epq_report as epqrepor
import research.Estimate_academic_paper_quality.epq_summary as epqsumm
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

# Function that calls the LLM with the model already bound: it takes
# `user_prompt` and `system_prompt` and returns the text of the answer with the
# `hllmcli.TokenStats` of the call.
_LlmFunc = Callable[..., Tuple[str, hllmcli.TokenStats]]

# Names shown as the evidence of the year of a paper, by the source of the
# record.
_SOURCE_NAMES = {
    "semantic_scholar": "Semantic Scholar",
    "crossref": "Crossref",
    "arxiv": "arXiv",
}


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
    Build the system prompt: the rules of the file and the JSON schema.

    The rules live in their own file for review. The schema is added here, so
    that it always matches the class `_LlmExtraction` that checks the answer.

    :return: text of `extract_features.prompt.md` followed by the output format
    """
    path = hgit.find_file_in_git_tree("extract_features.prompt.md")
    rules = hio.from_file(path)
    schema = json.dumps(_LlmExtraction.model_json_schema(), indent=2)
    system_prompt = (
        f"{rules.rstrip()}\n\n"
        "# Output Format\n\n"
        "- Answer with one JSON object and nothing else\n"
        "  - Do not add text before or after it\n"
        "  - Do not wrap it in a markdown code block\n"
        "- The JSON object must follow this JSON schema:\n\n"
        f"{schema}\n"
    )
    return system_prompt


def _parse_llm_answer(answer: str) -> _LlmExtraction:
    """
    Parse the text of the answer of the LLM into the expected shape.

    A model can wrap the JSON in a markdown code block or add a sentence around
    it, so the text from the first `{` to the last `}` is parsed.

    :param answer: text that the LLM returned
    :return: validated answer
    """
    start = answer.find("{")
    end = answer.rfind("}")
    hdbg.dassert_lt(
        -1, start, "The answer of the LLM has no JSON object:", answer[:200]
    )
    hdbg.dassert_lt(start, end, "The JSON object is not closed:", answer[:200])
    llm_answer = _LlmExtraction.model_validate_json(answer[start : end + 1])
    return llm_answer


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


def _extract_features(
    md_text: str,
    page_texts: List[str],
    *,
    llm_func: _LlmFunc,
    max_chars: int,
    model: str,
) -> epqtypes.ExtractionResult:
    """
    Extract the features of a paper with one LLM call.

    :param md_text: markdown of the paper
    :param page_texts: text of each page of the PDF, to check the quotes
        - An empty list leaves every quote unverified
    :param llm_func: function that calls the LLM, see `_LlmFunc`
    :param max_chars: budget of characters to send to the LLM
    :param model: name of the model, recorded in the provenance
    :return: features, facts, affiliations, and provenance (model, SHA-1 of the
        prompt, characters sent, whether the text was cut)
    """
    _LOG.debug(hprint.to_str("max_chars model"))
    text, was_cut = _prepare_text(md_text, max_chars)
    system_prompt = _get_system_prompt()
    answer, token_stats = llm_func(user_prompt=text, system_prompt=system_prompt)
    llm_answer = _parse_llm_answer(answer)
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
        cost_usd=token_stats.to_float(),
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


# #############################################################################
# Extract the papers
# #############################################################################


@dataclasses.dataclass(frozen=True)
class _ExtractConfig:
    """
    Settings of one run of the extraction.
    """

    # Dir of all the outputs of the run.
    output_dir: str
    # LLM for the `llm` library, e.g., `hllmcli.DEFAULT_MODEL`.
    model: str
    force_extract: bool
    # Max chars of markdown sent to the LLM (decision D13).
    max_chars: int
    dry_run: bool


def _call_llm(
    user_prompt: str,
    system_prompt: str,
    *,
    model: str,
    force_refresh: bool,
) -> Tuple[str, hllmcli.TokenStats]:
    """
    Call the LLM with `hllmcli.apply_llm()`.

    `apply_llm()` caches the answer on disk by prompts and model, so a rerun with
    the same inputs does not call the LLM again.

    :param user_prompt: text of the paper
    :param system_prompt: rules of the extraction and the output format
    :param model: LLM for the `llm` library
    :param force_refresh: if True, call the LLM even if the answer is cached
    :return: text of the answer and the stats of the call
    """
    answer, token_stats = hllmcli.apply_llm(
        user_prompt,
        model,
        system_prompt=system_prompt,
        backend="library",
        force_refresh=force_refresh,
    )
    return answer, token_stats


def _make_llm_func(model: str, *, force_refresh: bool) -> _LlmFunc:
    """
    Build the LLM function, with the model bound.

    :param model: LLM for the `llm` library
    :param force_refresh: if True, ignore the cached answers
    :return: function that takes `user_prompt` and `system_prompt`
    """
    llm_func = functools.partial(
        _call_llm, model=model, force_refresh=force_refresh
    )
    return llm_func


def _extract_paper(
    paper: epqtypes.Paper,
    config: _ExtractConfig,
    *,
    llm_func: _LlmFunc,
) -> float:
    """
    Extract the features of one paper and write its features file.

    :param paper: paper with a markdown file
    :param config: settings of the run
    :param llm_func: function that calls the LLM
    :return: cost of the LLM call, in USD
    """
    md_path = epqpaths.get_markdown_path(config.output_dir, paper.base_name)
    pdf_path = epqpaths.get_pdf_path(config.output_dir, paper.base_name)
    md_text = epqfetch.read_markdown(md_path)
    # Without the PDF the quotes stay unverified and the page count is unknown.
    has_pdf = os.path.isfile(pdf_path)
    page_texts = epqfetch.read_pdf_pages(pdf_path) if has_pdf else []
    n_pages = epqfetch.count_pages(pdf_path) if has_pdf else None
    # Extract the features with one LLM call.
    extraction = _extract_features(
        md_text,
        page_texts,
        llm_func=llm_func,
        max_chars=config.max_chars,
        model=config.model,
    )
    cost_usd = extraction.provenance.cost_usd
    # The year comes from the record, with the name of its source as evidence.
    year_text = str(paper.year) if paper.year else "unknown"
    recency = epqtypes.FeatureValue(
        value=year_text,
        evidence=epqtypes.Evidence(
            quote=year_text,
            location=_SOURCE_NAMES.get(paper.source, paper.source),
            verified=True,
        ),
    )
    features = {
        "performance": extraction.performance,
        "recency": recency,
        "asset_class": extraction.asset_class,
        "theoretical_domain": extraction.theoretical_domain,
        "mechanism": extraction.mechanism,
    }
    paper_features = epqtypes.PaperFeatures(
        base_name=paper.base_name,
        features=features,
        facts=extraction.facts,
        affiliations=extraction.affiliations,
        n_pages=n_pages,
        provenance=extraction.provenance,
    )
    epqrepor.write_features_file(
        epqpaths.get_features_path(config.output_dir, paper.base_name),
        paper_features,
    )
    _LOG.info("Extracted '%s': cost %.4f USD", paper.base_name, cost_usd)
    return cost_usd


def _extract(
    papers: List[epqtypes.Paper],
    config: _ExtractConfig,
    *,
    llm_func: _LlmFunc,
) -> List[epqtypes.Issue]:
    """
    Extract the features of the converted papers that have none yet.

    :param papers: papers from the papers file
    :param config: settings of the run
    :param llm_func: function that calls the LLM
    :return: issues of the papers that failed
    """
    timer = htimer.Timer()
    issues: List[epqtypes.Issue] = []
    total_cost = 0.0
    n_done_before = 0
    n_no_input = 0
    n_ok = 0
    for paper in tqdm.tqdm(papers):
        md_path = epqpaths.get_markdown_path(config.output_dir, paper.base_name)
        features_path = epqpaths.get_features_path(
            config.output_dir, paper.base_name
        )
        if not os.path.isfile(md_path):
            # The download issue already says why.
            n_no_input += 1
            continue
        if os.path.isfile(features_path) and not config.force_extract:
            _LOG.debug("Skip '%s': has a features file", paper.base_name)
            n_done_before += 1
            continue
        if config.dry_run:
            _LOG.warning("[DRY_RUN] Would extract '%s'", paper.base_name)
            continue
        try:
            total_cost += _extract_paper(paper, config, llm_func=llm_func)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            # Decision D21: an error of one paper must not stop the run.
            _LOG.warning("Paper '%s' failed: %s", paper.base_name, e)
            issues.append(
                epqissue.make_paper_issue(
                    paper,
                    "extract",
                    "extraction_failed",
                    epqissue.get_last_line(str(e)),
                )
            )
    issues_path = epqpaths.get_issues_path(config.output_dir, "extract")
    if config.dry_run:
        _LOG.warning("[DRY_RUN] Would write the issues of the stage `extract`")
    else:
        epqrepor.write_issues(issues_path, issues)
    # Summarize the stage.
    features_pattern = epqpaths.get_features_path(config.output_dir, "<BASE>")
    summary = epqtypes.StageSummary(
        stage="extract",
        unit="paper",
        n_total=len(papers),
        n_done_before=n_done_before,
        n_no_input=n_no_input,
        n_ok=n_ok,
        n_failed=len(issues),
        outputs=((features_pattern, n_ok),),
        issues_path=issues_path,
        issue_counts=epqsumm.count_issues(issues, add_stage=False),
        extra=(f"LLM cost in this run: {total_cost:.4f} USD",),
        elapsed_sec=timer.get_elapsed(),
        dry_run=config.dry_run,
    )
    epqsumm.emit_stage_summary(summary, config.output_dir)
    return issues


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
    parser.add_argument(
        "--model",
        action="store",
        default=hllmcli.DEFAULT_MODEL,
        help="LLM to call with the `llm` library, e.g., `gpt-4o-mini`",
    )
    hparser.add_bool_arg(
        parser,
        "force_extract",
        default_value=False,
        help_="Extract again the papers that have a features file",
    )
    parser.add_argument(
        "--max_chars",
        action="store",
        type=int,
        default=300000,
        help="Cap on the characters of markdown sent to the LLM",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show the papers to extract, with no LLM call and no output file",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Check the boundary of the system.
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    papers_file = args.input or epqpaths.get_papers_path(output_dir)
    papers_file = os.path.abspath(os.path.expanduser(papers_file))
    hdbg.dassert_file_exists(
        papers_file, "Run resolve_papers.py to make the papers file"
    )
    hdbg.dassert_dir_exists(
        epqpaths.get_papers_dir(output_dir),
        "Run download_papers.py to make the papers",
    )
    hdbg.dassert_lt(0, args.max_chars)
    config = _ExtractConfig(
        output_dir=output_dir,
        model=args.model,
        force_extract=args.force_extract,
        max_chars=args.max_chars,
        dry_run=args.dry_run,
    )
    # Extract the features of the papers.
    papers = epqrepor.read_papers(papers_file)
    llm_func = _make_llm_func(config.model, force_refresh=config.force_extract)
    _extract(papers, config, llm_func=llm_func)
    _LOG.info("Outputs are in '%s'", output_dir)


if __name__ == "__main__":
    _main(_parse())
