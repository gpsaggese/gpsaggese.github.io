#!/usr/bin/env -S uv run

# /// script
# dependencies = ["feedparser", "pymupdf", "requests", "pyyaml", "beautifulsoup4", "pydantic", "tqdm", "openai"]
# ///

"""
Turn a CSV of resolved papers into a CSV table of candidate trading strategies.

The input is the papers CSV that `resolve_papers.py` writes from a loose list of
papers. Run `resolve_papers.py` first.

The pipeline has three stages:
- `download`: save the PDF and the markdown of each paper
- `extract`: extract the features of each paper with an LLM, and rate its quality
- `report`: write `references.md`, `report.csv`, and `issues.csv`

All the outputs and the state of the run live in `--output_dir`. A rerun on the
same dir does not pay again for the papers already processed.

The design is in `research/Estimate_academic_paper_quality/docs/02-design.architecture.md`.

For a description of this file, see the file
`research/Estimate_academic_paper_quality/estimate_paper_quality.py.README.md`
"""

import argparse
import datetime
import logging
import os

import helpers.hcache_simple as hcacsimp
import helpers.hdbg as hdbg
import helpers.hparser as hparser
import helpers.hselect_action as hselacti

import research.Estimate_academic_paper_quality.epq_pipeline as epqpipel

_LOG = logging.getLogger(__name__)

_VALID_ACTIONS = ["download", "extract", "report"]
_DEFAULT_ACTIONS = _VALID_ACTIONS


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
        "--resolve_issues",
        action="store",
        default="",
        help="Issues CSV that `resolve_papers.py` wrote; empty is the papers "
        "CSV with `.issues.csv`, if that file exists",
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
        default="",
        help="LLM of the stage `extract`; empty is the default of `helpers.hllm`",
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
    hcacsimp.add_cache_control_arg(parser)
    hselacti.add_action_arg(parser, _VALID_ACTIONS, _DEFAULT_ACTIONS)
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    hcacsimp.parse_cache_control_args(args)
    # Build the config and check the boundary of the system.
    papers_file = os.path.abspath(os.path.expanduser(args.input))
    resolve_issues_file = args.resolve_issues
    if not resolve_issues_file:
        default_issues_file = epqpipel.get_resolve_issues_path(papers_file)
        if os.path.isfile(default_issues_file):
            resolve_issues_file = default_issues_file
        else:
            _LOG.warning(
                "No issues CSV of `resolve_papers.py` at '%s': `issues.csv` "
                "lists no unresolved line",
                default_issues_file,
            )
    config = epqpipel.PipelineConfig(
        papers_file=papers_file,
        resolve_issues_file=resolve_issues_file,
        output_dir=os.path.abspath(os.path.expanduser(args.output_dir)),
        model=args.model,
        force_extract=args.force_extract,
        max_chars=args.max_chars,
        today=datetime.date.today(),
        cache_mode=args.cache_mode,
        s2_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
    )
    epqpipel.make_dirs(config)
    # The cache of the helpers lives in the output dir with the other state.
    hcacsimp.set_cache_dir(epqpipel.get_cache_dir(config))
    # The stages only ask for citation counts, which need no contact email.
    sources = epqpipel.make_sources(
        email="",
        s2_api_key=config.s2_api_key,
        cache_dir=epqpipel.get_cache_dir(config),
        cache_mode=config.cache_mode,
    )
    # Select the stages to run.
    actions = hselacti.select_actions(args, _VALID_ACTIONS, _DEFAULT_ACTIONS)
    _LOG.info(
        "\n%s",
        hselacti.actions_to_string(actions, _VALID_ACTIONS, add_frame=True),
    )
    # Run the stages on the papers of the papers CSV.
    papers = epqpipel.read_papers(config)
    while actions:
        action = actions[0]
        to_execute, actions = hselacti.mark_action(action, actions)
        if not to_execute:
            continue
        if action == "download":
            epqpipel.run_download(config, papers, log_level=args.log_level)
        elif action == "extract":
            llm_func, cost_tracker = epqpipel.make_llm_func(config)
            tiers = epqpipel.load_tiers(epqpipel.find_tiers_path())
            epqpipel.run_extract(
                config,
                papers,
                llm_func=llm_func,
                cost_tracker=cost_tracker,
                sources=sources,
                tiers=tiers,
            )
        elif action == "report":
            epqpipel.run_report(config, papers)
        else:
            raise ValueError(f"Invalid action='{action}'")
    hdbg.dassert_eq(len(actions), 0, "Unprocessed actions: %s", str(actions))
    _LOG.info("Outputs are in '%s'", config.output_dir)


if __name__ == "__main__":
    _main(_parse())
