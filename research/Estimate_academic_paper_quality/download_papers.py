#!/usr/bin/env -S uv run

# /// script
# dependencies = ["pymupdf", "pyyaml", "tqdm"]
# ///

"""
Download the PDF of each paper of a papers CSV and convert it to markdown.

The input is the papers CSV that `resolve_papers.py` writes. For each paper the
script tries the download inputs in order, and calls
`download_academic_paper_to_md.py` on the first one that works. SSRN blocks
scripted clients: the tool uses Chrome for it.

The outputs live in `--output_dir`:
- `papers/<BASE>.pdf` and `papers/<BASE>.md` for each paper
- `state/issues.download.json`: the papers that failed, for `report_papers.py`

A rerun does not download or convert a paper that is already converted.

The next stage is `extract_features.py`.

The design is in `research/Estimate_academic_paper_quality/docs/02-design.architecture.md`.

# Usage Example

- Download the papers of a papers CSV:
> download_papers.py --input papers.csv --output_dir out

- List the papers to download, with no download:
> download_papers.py --input papers.csv --output_dir out --dry_run

For a description of this file, see the file
`research/Estimate_academic_paper_quality/download_papers.py.README.md`
"""

import argparse
import logging
import os
from typing import List

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser
import tqdm

import research.Estimate_academic_paper_quality.epq_fetcher as epqfetch
import research.Estimate_academic_paper_quality.epq_issues as epqissue
import research.Estimate_academic_paper_quality.epq_paths as epqpaths
import research.Estimate_academic_paper_quality.epq_report as epqrepor
import research.Estimate_academic_paper_quality.epq_types as epqtypes

_LOG = logging.getLogger(__name__)


def _download(
    papers: List[epqtypes.Paper],
    output_dir: str,
    *,
    tool_path: str,
    log_level: str,
    dry_run: bool,
) -> List[epqtypes.Issue]:
    """
    Download and convert the papers that are not converted yet.

    :param papers: papers from the papers CSV
    :param output_dir: dir of all the outputs of the run
    :param tool_path: path of the download tool
        - Empty: find it in the Git tree
    :param log_level: logging level to forward to the tool
    :param dry_run: if True, only log the papers to download
    :return: issues of the papers that failed
    """
    papers_dir = epqpaths.get_papers_dir(output_dir)
    log_dir = epqpaths.get_log_dir(output_dir)
    issues: List[epqtypes.Issue] = []
    for paper in tqdm.tqdm(papers):
        if epqfetch.is_fetched(papers_dir, paper.base_name):
            # A rerun does not download or convert a converted paper.
            _LOG.debug("Skip '%s': already fetched", paper.base_name)
            continue
        if dry_run:
            _LOG.warning("[DRY_RUN] Would download '%s'", paper.base_name)
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
                status="download_failed", reason=epqissue.get_last_line(str(e))
            )
        if result.status != "ok":
            _LOG.warning(
                "Paper '%s': %s: %s",
                paper.base_name,
                result.status,
                result.reason,
            )
            issues.append(
                epqissue.make_paper_issue(
                    paper, "download", result.status, result.reason
                )
            )
    if dry_run:
        _LOG.warning("[DRY_RUN] Would write the issues of the stage `download`")
    else:
        path = epqpaths.get_issues_path(output_dir, "download")
        epqrepor.write_issues(path, issues)
    _LOG.info("Download: %s of %s papers failed", len(issues), len(papers))
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
        required=True,
        help="Papers CSV that `resolve_papers.py` wrote",
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        required=True,
        help="Dir of all the outputs and of the state of the run",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show the papers to download, with no download and no output file",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Check the boundary of the system.
    papers_file = os.path.abspath(os.path.expanduser(args.input))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    hdbg.dassert_file_exists(
        papers_file, "Run resolve_papers.py to make the papers CSV"
    )
    # Download the papers.
    papers = epqrepor.read_papers_csv(papers_file)
    if not args.dry_run:
        hio.create_dir(epqpaths.get_papers_dir(output_dir), incremental=True)
    _download(
        papers,
        output_dir,
        tool_path="",
        log_level=args.log_level,
        dry_run=args.dry_run,
    )
    _LOG.info("Outputs are in '%s'", output_dir)


if __name__ == "__main__":
    _main(_parse())
