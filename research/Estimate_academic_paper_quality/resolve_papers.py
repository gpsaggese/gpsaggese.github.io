#!/usr/bin/env -S uv run

# /// script
# dependencies = ["feedparser", "pymupdf", "requests", "pyyaml", "beautifulsoup4", "pydantic", "tqdm"]
# ///

"""
Turn a CSV of loose paper names into a CSV of resolved, deduped papers.

The input CSV has a column `input`. Each row is one paper:
- a title (with the authors in parentheses)
- an arXiv or SSRN link, a DOI, a publisher link, or the path of a local PDF

The output is two CSV files:
- The papers CSV (`--output`): one row per paper, from the newest to the oldest,
  with the reference, the base name, and the ordered download inputs
- The issues CSV (`--issues_output`): one row per input that did not give a paper

The papers CSV is the input of `estimate_paper_quality.py`.

# Usage Example

- Resolve a list of papers:
> resolve_papers.py --input papers_in.csv --output papers.csv

- Show how each row is classified, with no lookup and no output file:
> resolve_papers.py --input papers_in.csv --output papers.csv --dry_run

For a description of this file, see the file
`research/Estimate_academic_paper_quality/resolve_papers.py.README.md`
"""

import argparse
import logging
import os

import helpers.hcache_simple as hcacsimp
import helpers.hdbg as hdbg
import helpers.hparser as hparser

import research.Estimate_academic_paper_quality.epq_pipeline as epqpipel

_LOG = logging.getLogger(__name__)


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        action="store",
        required=True,
        help="Input CSV with a column `input`, one paper per row",
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        required=True,
        help="Papers CSV to write",
    )
    parser.add_argument(
        "--issues_output",
        action="store",
        default="",
        help="Issues CSV to write; empty is the papers CSV with `.issues.csv`",
    )
    parser.add_argument(
        "--cache_dir",
        action="store",
        default="",
        help="Dir of the cache of the API answers; empty is `cache/` next to "
        "the papers CSV",
    )
    parser.add_argument(
        "--email",
        action="store",
        default="",
        help="Contact email for Unpaywall; empty skips Unpaywall",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show how each row is classified, with no lookup and no output",
    )
    hcacsimp.add_cache_control_arg(parser)
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    hcacsimp.parse_cache_control_args(args)
    input_csv = os.path.abspath(os.path.expanduser(args.input))
    papers_csv = os.path.abspath(os.path.expanduser(args.output))
    issues_csv = args.issues_output or epqpipel.get_resolve_issues_path(
        papers_csv
    )
    issues_csv = os.path.abspath(os.path.expanduser(issues_csv))
    cache_dir = args.cache_dir or os.path.join(
        os.path.dirname(papers_csv), "cache"
    )
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    hdbg.dassert_file_exists(input_csv)
    if args.dry_run:
        # Classify the rows only: no lookup, no cache dir, no output file.
        for input_line in epqpipel.read_input_lines(input_csv):
            _LOG.warning(
                "[DRY_RUN] Would resolve row %s as %s: '%s'",
                input_line.line_no,
                input_line.kind,
                input_line.key,
            )
        return
    # The cache of the helpers lives with the cache of the API answers.
    hcacsimp.set_cache_dir(cache_dir)
    sources = epqpipel.make_sources(
        email=args.email,
        s2_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
        cache_dir=cache_dir,
        cache_mode=args.cache_mode,
    )
    epqpipel.run_resolve_csv(input_csv, papers_csv, issues_csv, sources)
    _LOG.info("Papers are in '%s', issues in '%s'", papers_csv, issues_csv)


if __name__ == "__main__":
    _main(_parse())
