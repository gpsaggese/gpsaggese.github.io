#!/usr/bin/env -S uv run

# /// script
# dependencies = ["pymupdf", "pyyaml", "tqdm"]
# ///

"""
Download the PDF of each paper of a papers file and convert it to markdown.

The input is the papers file that `resolve_papers.py` writes, by default
`01-resolve.paper_list.json` in `--output_dir`. For each paper the script tries
the download inputs in order, and calls `download_academic_paper_to_md.py` on the
first one that works. SSRN blocks scripted clients: the tool uses Chrome for it.

The outputs live in `--output_dir`:
- `papers/<BASE>.pdf` and `papers/<BASE>.md` for each paper
- `02-download.issues.json`: the papers that failed, for `report_papers.py`

The script waits a random time between two downloads, so that the hosts do not
take it for a bot. A rerun does not download or convert a paper that is already
converted.

The next stage is `extract_features.py`.

The design is in `research/Estimate_academic_paper_quality/docs/02-design.architecture.md`.

# Usage Example

- Download the papers that `resolve_papers.py` wrote in `out`:
> download_papers.py --output_dir out

- Wait 5 to 20 seconds between two downloads:
> download_papers.py --output_dir out --min_delay_sec 5 --max_delay_sec 20

- List the papers to download, with no download:
> download_papers.py --output_dir out --dry_run

For a description of this file, see the file
`research/Estimate_academic_paper_quality/download_papers.py.README.md`
"""

import argparse
import logging
import os
import random
import time
from typing import Callable, List

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
    min_delay_sec: float,
    max_delay_sec: float,
    dry_run: bool,
    sleep_func: Callable[[float], None] = time.sleep,
) -> List[epqtypes.Issue]:
    """
    Download and convert the papers that are not converted yet.

    The script waits a random time between two downloads, so that a host does not
    take it for a bot. It does not wait before the first download, and it does not
    wait for a paper that it skips.

    :param papers: papers from the papers file
    :param output_dir: dir of all the outputs of the run
    :param tool_path: path of the download tool
        - Empty: find it in the Git tree
    :param log_level: logging level to forward to the tool
    :param min_delay_sec: shortest wait between two downloads
    :param max_delay_sec: longest wait between two downloads
    :param dry_run: if True, only log the papers to download
    :param sleep_func: function that waits, replaced in the tests
    :return: issues of the papers that failed
    """
    papers_dir = epqpaths.get_papers_dir(output_dir)
    log_dir = epqpaths.get_log_dir(output_dir)
    issues: List[epqtypes.Issue] = []
    n_downloads = 0
    for paper in tqdm.tqdm(papers):
        if epqfetch.is_fetched(papers_dir, paper.base_name):
            # A rerun does not download or convert a converted paper.
            _LOG.debug("Skip '%s': already fetched", paper.base_name)
            continue
        if dry_run:
            _LOG.warning("[DRY_RUN] Would download '%s'", paper.base_name)
            continue
        if n_downloads > 0:
            delay = random.uniform(min_delay_sec, max_delay_sec)
            _LOG.debug("Wait %.1f sec before '%s'", delay, paper.base_name)
            sleep_func(delay)
        n_downloads += 1
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
        "--min_delay_sec",
        action="store",
        type=float,
        default=3.0,
        help="Shortest random wait between two downloads, in seconds",
    )
    parser.add_argument(
        "--max_delay_sec",
        action="store",
        type=float,
        default=10.0,
        help="Longest random wait between two downloads, in seconds",
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
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    papers_file = args.input or epqpaths.get_papers_path(output_dir)
    papers_file = os.path.abspath(os.path.expanduser(papers_file))
    hdbg.dassert_file_exists(
        papers_file, "Run resolve_papers.py to make the papers file"
    )
    hdbg.dassert_lte(0, args.min_delay_sec)
    hdbg.dassert_lte(args.min_delay_sec, args.max_delay_sec)
    # Download the papers.
    papers = epqrepor.read_papers(papers_file)
    if not args.dry_run:
        hio.create_dir(epqpaths.get_papers_dir(output_dir), incremental=True)
    _download(
        papers,
        output_dir,
        tool_path="",
        log_level=args.log_level,
        min_delay_sec=args.min_delay_sec,
        max_delay_sec=args.max_delay_sec,
        dry_run=args.dry_run,
    )
    _LOG.info("Outputs are in '%s'", output_dir)


if __name__ == "__main__":
    _main(_parse())
