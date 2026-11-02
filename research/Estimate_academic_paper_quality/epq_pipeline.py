"""
Run the stages of the pipeline on one papers CSV and one output dir.

The stage `resolve` is a separate executable, `resolve_papers.py`, which turns
an input CSV into a papers CSV and an issues CSV. This module runs it
(`run_resolve_csv()`), and runs the stages that read the papers CSV:
- `download`: PDF and markdown of each paper
- `extract`: features file of each converted paper
- `report`: `references.md`, `report.csv`, and `issues.csv`

Each stage loops on the lines or the papers and turns the failure of one of
them into an `Issue`, so the run goes on (decision D21). A rerun skips the work
that a previous run did.

The design is in `docs/02.1-design.pipeline.md`.

Import as:

import research.Estimate_academic_paper_quality.epq_pipeline as epqpipel
"""

import dataclasses
import datetime
import functools
import logging
import os
from typing import Dict, List, Tuple

import helpers.hdbg as hdbg
import helpers.hgit as hgit
import helpers.hio as hio
import helpers.hllm_cost as hllmcost
import tqdm
import yaml

import research.Estimate_academic_paper_quality.epq_bib_sources as epqbibso
import research.Estimate_academic_paper_quality.epq_extractor as epqextra
import research.Estimate_academic_paper_quality.epq_fetcher as epqfetch
import research.Estimate_academic_paper_quality.epq_rater as epqrater
import research.Estimate_academic_paper_quality.epq_report as epqrepor
import research.Estimate_academic_paper_quality.epq_resolver as epqresol
import research.Estimate_academic_paper_quality.epq_types as epqtypes

_LOG = logging.getLogger(__name__)

# #############################################################################
# Configuration and paths
# #############################################################################


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    """
    Settings of one run of the pipeline.
    """

    # Papers CSV that `resolve_papers.py` wrote.
    papers_file: str
    # Issues CSV that `resolve_papers.py` wrote, "" if there is none.
    resolve_issues_file: str
    output_dir: str
    # LLM of `extract`, "" for the default model of `helpers.hllm`.
    model: str
    force_extract: bool
    # Max chars of markdown sent to the LLM (decision D13).
    max_chars: int
    # Date used to annualize the citations, injected for tests.
    today: datetime.date
    # One of "", `REFRESH_CACHE`, `DISABLE_CACHE`, `HIT_CACHE_OR_ABORT`.
    cache_mode: str
    # Key of Semantic Scholar, "" for the shared pool.
    s2_api_key: str


def get_papers_dir(config: PipelineConfig) -> str:
    """
    Get the dir of the PDF, markdown, and features files of the papers.

    :param config: settings of the run
    :return: `<output_dir>/papers`
    """
    return os.path.join(config.output_dir, "papers")


def get_state_dir(config: PipelineConfig) -> str:
    """
    Get the dir of the outputs of the stages and of the HTTP cache.

    :param config: settings of the run
    :return: `<output_dir>/state`
    """
    return os.path.join(config.output_dir, "state")


def get_cache_dir(config: PipelineConfig) -> str:
    """
    Get the dir of the cache of the HTTP answers.

    :param config: settings of the run
    :return: `<output_dir>/state/cache`
    """
    return os.path.join(get_state_dir(config), "cache")


def get_features_path(config: PipelineConfig, base_name: str) -> str:
    """
    Get the path of the features file of a paper.

    :param config: settings of the run
    :param base_name: base name of the paper
    :return: `<output_dir>/papers/<BASE>.features.json`
    """
    return os.path.join(get_papers_dir(config), f"{base_name}.features.json")


def get_resolve_issues_path(papers_file: str) -> str:
    """
    Get the default path of the issues CSV that goes with a papers CSV.

    - `run/papers.csv` -> `run/papers.issues.csv`

    :param papers_file: path of the papers CSV
    :return: path of the issues CSV of `resolve_papers.py`
    """
    return os.path.splitext(papers_file)[0] + ".issues.csv"


def make_dirs(config: PipelineConfig) -> None:
    """
    Check the input and create the dirs of the output.

    :param config: settings of the run
    """
    hdbg.dassert_file_exists(
        config.papers_file, "Run resolve_papers.py to make the papers CSV"
    )
    if config.resolve_issues_file:
        hdbg.dassert_file_exists(config.resolve_issues_file)
    hdbg.dassert_ne(config.output_dir, "", "The output dir is required")
    hio.create_dir(get_papers_dir(config), incremental=True)
    hio.create_dir(get_cache_dir(config), incremental=True)


def make_sources(
    *, email: str, s2_api_key: str, cache_dir: str, cache_mode: str
) -> epqbibso.BibSources:
    """
    Build the clients of the lookup APIs.

    :param email: contact for Unpaywall, "" to skip Unpaywall
    :param s2_api_key: key of Semantic Scholar, "" for the shared pool
    :param cache_dir: dir of the cache of the HTTP answers
    :param cache_mode: one of "", `REFRESH_CACHE`, `DISABLE_CACHE`,
        `HIT_CACHE_OR_ABORT`
    :return: clients with the cache in `cache_dir`
    """
    sources = epqbibso.BibSources(
        email=email,
        s2_api_key=s2_api_key,
        cache_dir=cache_dir,
        cache_mode=cache_mode,
    )
    return sources


def make_llm_func(
    config: PipelineConfig,
) -> Tuple[epqextra.LlmFunc, hllmcost.LLMCostTracker]:
    """
    Build the LLM function of the `extract` stage, with its cost tracker.

    `helpers.hllm` is imported here because it pulls the LLM client libraries,
    which the other stages do not need.

    :param config: settings of the run
    :return: function with the model bound, and the tracker of its cost
    """
    import helpers.hllm as hllm

    tracker = hllmcost.LLMCostTracker(model=config.model)
    llm_func = functools.partial(
        hllm.get_structured_completion,
        model=config.model,
        cost_tracker=tracker,
    )
    return llm_func, tracker


def load_tiers(path: str) -> epqtypes.AffiliationTiers:
    """
    Load the dated affiliation tier list.

    :param path: path of `affiliation_tiers.yaml`
    :return: tier list
    """
    hdbg.dassert_file_exists(path)
    data = yaml.safe_load(hio.from_file(path)) or {}
    tiers = epqrater.parse_tiers(data)
    if not tiers.tier_5 and not tiers.tier_4:
        _LOG.warning(
            "The affiliation tier list '%s' is empty: each affiliation scores "
            "by its kind",
            path,
        )
    return tiers


def find_tiers_path() -> str:
    """
    Find the affiliation tier list in the Git tree.

    :return: absolute path of `affiliation_tiers.yaml`
    """
    return hgit.find_file_in_git_tree("affiliation_tiers.yaml")


# #############################################################################
# State
# #############################################################################


def read_papers(config: PipelineConfig) -> List[epqtypes.Paper]:
    """
    Read the papers that `resolve_papers.py` wrote.

    :param config: settings of the run
    :return: papers in the order of the papers CSV
    """
    return epqrepor.read_papers_csv(config.papers_file)


def _read_state_issues(
    config: PipelineConfig, stage: str
) -> List[epqtypes.Issue]:
    """
    Read the issues that a stage wrote, if the stage ran.

    :param config: settings of the run
    :param stage: one of `download`, `extract`
    :return: issues of the stage, empty if the stage did not run
    """
    path = os.path.join(get_state_dir(config), f"issues.{stage}.json")
    issues = epqrepor.read_issues(path) if os.path.isfile(path) else []
    return issues


def _last_line(message: str) -> str:
    """
    Keep the last non-empty line of an error message as the reason.

    The frame lines of a failed `hdbg.dassert` are only `#`, and they are
    skipped, since the line before them has the message.

    :param message: text of an exception
    :return: last line, or `unknown error` if the message is empty
    """
    lines = [line.strip() for line in message.split("\n") if line.strip()]
    lines = [line for line in lines if set(line) != {"#"}]
    return lines[-1] if lines else "unknown error"


def _make_paper_issue(
    paper: epqtypes.Paper, stage: str, status: str, reason: str
) -> epqtypes.Issue:
    """
    Build the issue of a paper that failed in a stage.

    :param paper: paper that failed
    :param stage: one of `download`, `extract`
    :param status: status of the issue
    :param reason: why the paper failed
    :return: issue tied to the first line that gave the paper
    """
    return epqtypes.Issue(
        line_no=min(paper.line_nos),
        input=paper.reference,
        base_name=paper.base_name,
        stage=stage,
        status=status,
        reason=reason,
        candidates=(),
    )


# #############################################################################
# Stages
# #############################################################################


def read_input_lines(input_csv: str) -> List[epqtypes.InputLine]:
    """
    Read and classify the rows of the input CSV of `resolve_papers.py`.

    A blank row and a row that starts with `#` or `//` are skipped.

    :param input_csv: path of the input CSV, with the column `input`
    :return: classified lines, numbered by their row in the CSV
    """
    input_lines = []
    for line_no, text in epqrepor.read_input_csv(input_csv):
        input_line = epqresol.parse_line(line_no, text)
        if input_line is not None:
            input_lines.append(input_line)
    _LOG.debug("return=%s lines", len(input_lines))
    return input_lines


def run_resolve_csv(
    input_csv: str,
    papers_csv: str,
    issues_csv: str,
    sources: epqbibso.BibSources,
) -> epqtypes.ResolveResult:
    """
    Resolve the rows of an input CSV into a papers CSV and an issues CSV.

    :param input_csv: path of the input CSV, with the column `input`
    :param papers_csv: path of the papers CSV to write
    :param issues_csv: path of the issues CSV to write
    :param sources: clients of the lookup APIs
    :return: deduped papers in reverse chronological order, and the issues
    """
    input_lines = read_input_lines(input_csv)
    _LOG.info("Resolving %s lines", len(input_lines))
    papers: List[epqtypes.Paper] = []
    issues: List[epqtypes.Issue] = []
    for input_line in tqdm.tqdm(input_lines):
        try:
            result = epqresol.resolve_line(input_line, sources)
        except Exception as e:  # noqa: BLE001
            # Decision D21: an error of one line must not stop the run.
            _LOG.warning("Line %s failed: %s", input_line.line_no, e)
            issue = epqtypes.Issue(
                line_no=input_line.line_no,
                input=input_line.text,
                base_name="",
                stage="resolve",
                status="unresolved",
                reason=_last_line(str(e)),
                candidates=(),
            )
            result = epqtypes.ResolveResult(papers=(), issues=(issue,))
        papers.extend(result.papers)
        issues.extend(result.issues)
    papers = epqresol.finalize_papers(papers)
    epqrepor.write_papers_csv(papers_csv, papers)
    epqrepor.write_issues_csv(issues_csv, issues)
    _LOG.info("Resolved %s papers, %s issues", len(papers), len(issues))
    return epqtypes.ResolveResult(papers=tuple(papers), issues=tuple(issues))


def run_download(
    config: PipelineConfig,
    papers: List[epqtypes.Paper],
    *,
    tool_path: str = "",
    log_level: str = "INFO",
) -> List[epqtypes.Issue]:
    """
    Download and convert the papers that are not converted yet.

    :param config: settings of the run
    :param papers: papers from the papers CSV
    :param tool_path: path of the download tool
        - Default: find it in the Git tree
    :param log_level: logging level to forward to the tool
    :return: issues of the papers that failed
    """
    papers_dir = get_papers_dir(config)
    log_dir = os.path.join(get_state_dir(config), "logs")
    issues: List[epqtypes.Issue] = []
    for paper in tqdm.tqdm(papers):
        if epqfetch.is_fetched(papers_dir, paper.base_name):
            # A rerun does not download or convert a converted paper.
            _LOG.debug("Skip '%s': already fetched", paper.base_name)
            continue
        try:
            result = epqfetch.fetch_paper(
                paper,
                papers_dir,
                log_dir=log_dir,
                tool_path=tool_path,
                log_level=log_level,
            )
        except Exception as e:  # noqa: BLE001
            # Decision D21: an error of one paper must not stop the run.
            result = epqtypes.FetchResult(
                status="download_failed", reason=_last_line(str(e))
            )
        if result.status != "ok":
            _LOG.warning(
                "Paper '%s': %s: %s",
                paper.base_name,
                result.status,
                result.reason,
            )
            issues.append(
                _make_paper_issue(
                    paper, "download", result.status, result.reason
                )
            )
    path = os.path.join(get_state_dir(config), "issues.download.json")
    epqrepor.write_issues(path, issues)
    _LOG.info("Download: %s of %s papers failed", len(issues), len(papers))
    return issues


def _extract_paper(
    config: PipelineConfig,
    paper: epqtypes.Paper,
    *,
    llm_func: epqextra.LlmFunc,
    cost_tracker: hllmcost.LLMCostTracker,
    sources: epqbibso.BibSources,
    tiers: epqtypes.AffiliationTiers,
) -> None:
    """
    Extract the features of one paper and write its features file.

    :param config: settings of the run
    :param paper: paper with a markdown file
    :param llm_func: function that calls the LLM
    :param cost_tracker: tracker of the cost of `llm_func`
    :param sources: clients of the lookup APIs, for the citation count
    :param tiers: dated affiliation tier list
    """
    papers_dir = get_papers_dir(config)
    md_path = os.path.join(papers_dir, f"{paper.base_name}.md")
    pdf_path = os.path.join(papers_dir, f"{paper.base_name}.pdf")
    md_text = epqfetch.read_markdown(md_path)
    # Without the PDF the quotes stay unverified and the page count is unknown.
    has_pdf = os.path.isfile(pdf_path)
    page_texts = epqfetch.read_pdf_pages(pdf_path) if has_pdf else []
    n_pages = epqfetch.count_pages(pdf_path) if has_pdf else None
    # Extract the features with one LLM call.
    cost_before = cost_tracker.get_current_cost()
    extraction = epqextra.extract_features(
        md_text,
        page_texts,
        llm_func=llm_func,
        max_chars=config.max_chars,
        model=config.model,
    )
    cost_usd = cost_tracker.get_current_cost() - cost_before
    # Rate the paper. A forced run asks for a fresh citation count.
    citations = sources.get_citation_count(
        s2_paper_id=paper.s2_paper_id,
        doi=paper.doi,
        arxiv_id=paper.arxiv_id,
        refresh=config.force_extract,
    )
    rating_inputs = epqrater.RatingInputs(
        n_pages=n_pages,
        affiliations=extraction.affiliations,
        citations=citations,
        year=paper.year,
        today=config.today,
    )
    quality = epqrater.rate_paper(rating_inputs, tiers)
    # The year comes from the record, with the name of its source as evidence.
    source_names = {
        "semantic_scholar": "Semantic Scholar",
        "crossref": "Crossref",
        "arxiv": "arXiv",
    }
    year_text = str(paper.year) if paper.year else "unknown"
    recency = epqtypes.FeatureValue(
        value=year_text,
        evidence=epqtypes.Evidence(
            quote=year_text,
            location=source_names.get(paper.source, paper.source),
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
        quality=quality,
        provenance=dataclasses.replace(extraction.provenance, cost_usd=cost_usd),
    )
    epqrepor.write_features_file(
        get_features_path(config, paper.base_name), paper_features
    )
    _LOG.info("Extracted '%s': cost %.4f USD", paper.base_name, cost_usd)


def run_extract(
    config: PipelineConfig,
    papers: List[epqtypes.Paper],
    *,
    llm_func: epqextra.LlmFunc,
    cost_tracker: hllmcost.LLMCostTracker,
    sources: epqbibso.BibSources,
    tiers: epqtypes.AffiliationTiers,
) -> List[epqtypes.Issue]:
    """
    Extract the features of the converted papers that have none yet.

    :param config: settings of the run
    :param papers: papers from the papers CSV
    :param llm_func: function that calls the LLM
    :param cost_tracker: tracker of the cost of `llm_func`
    :param sources: clients of the lookup APIs, for the citation count
    :param tiers: dated affiliation tier list
    :return: issues of the papers that failed
    """
    papers_dir = get_papers_dir(config)
    issues: List[epqtypes.Issue] = []
    cost_start = cost_tracker.get_current_cost()
    for paper in tqdm.tqdm(papers):
        md_path = os.path.join(papers_dir, f"{paper.base_name}.md")
        features_path = get_features_path(config, paper.base_name)
        if not os.path.isfile(md_path):
            # The download issue already says why.
            continue
        if os.path.isfile(features_path) and not config.force_extract:
            _LOG.debug("Skip '%s': has a features file", paper.base_name)
            continue
        try:
            _extract_paper(
                config,
                paper,
                llm_func=llm_func,
                cost_tracker=cost_tracker,
                sources=sources,
                tiers=tiers,
            )
        except Exception as e:  # noqa: BLE001
            # Decision D21: an error of one paper must not stop the run.
            _LOG.warning("Paper '%s' failed: %s", paper.base_name, e)
            issues.append(
                _make_paper_issue(
                    paper, "extract", "extraction_failed", _last_line(str(e))
                )
            )
    path = os.path.join(get_state_dir(config), "issues.extract.json")
    epqrepor.write_issues(path, issues)
    total_cost = cost_tracker.get_current_cost() - cost_start
    _LOG.info(
        "Extract: %s of %s papers failed, total LLM cost %.4f USD",
        len(issues),
        len(papers),
        total_cost,
    )
    return issues


def run_report(config: PipelineConfig, papers: List[epqtypes.Paper]) -> None:
    """
    Write `references.md`, `report.csv`, and `issues.csv`.

    The issues of `resolve_papers.py` come from its issues CSV, and the issues of
    the other stages are read from `state/`, so that this stage can run alone.

    :param config: settings of the run
    :param papers: papers from the papers CSV
    """
    # Read the features file of each paper that has one.
    features: Dict[str, epqtypes.PaperFeatures] = {}
    issues: List[epqtypes.Issue] = []
    if config.resolve_issues_file:
        issues.extend(epqrepor.read_issues_csv(config.resolve_issues_file))
    for stage in ["download", "extract"]:
        issues.extend(_read_state_issues(config, stage))
    for paper in papers:
        path = get_features_path(config, paper.base_name)
        if not os.path.isfile(path):
            continue
        try:
            features[paper.base_name] = epqrepor.read_features_file(path)
        except Exception as e:  # noqa: BLE001
            # An old schema is an issue that asks for `--force_extract`.
            issues.append(
                _make_paper_issue(
                    paper, "extract", "extraction_failed", _last_line(str(e))
                )
            )
    # Write the final outputs.
    out = config.output_dir
    epqrepor.write_references(os.path.join(out, "references.md"), papers)
    rows = epqrepor.build_report_rows(papers, features)
    epqrepor.write_report(os.path.join(out, "report.csv"), rows)
    epqrepor.write_issues_csv(os.path.join(out, "issues.csv"), issues)
    _LOG.info(
        "Report: %s papers, %s rows, %s issues",
        len(papers),
        len(rows),
        len(issues),
    )
